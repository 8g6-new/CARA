/**
 * @file beat_track.c
 * @brief Beat tracking and rhythm analysis implementation
 * 
 * This module implements dynamic programming-based beat tracking algorithms
 * compatible with librosa's beat_track() function, following Ellis (2007).
 */

#include "audio_tools/beat_track.h"
#include "audio_tools/audio_visualizer.h"
#include "utils/bench.h"
#include <float.h>

// Helper function prototypes
static float compute_std(const float *data, size_t length);
static float gaussian_weight(float x, float sigma);
static size_t find_last_beat(const float *cumscore, size_t length);
static bool is_local_max(const float *data, size_t index, size_t length);

/**
 * Get default beat tracking parameters.
 */
beat_params_t get_default_beat_params(void) {
    beat_params_t params = {
        .tightness = 100.0f,
        .trim = true,
        .sparse = true,
        .tempo_params = NULL  // Use default tempo params
    };
    return params;
}

/**
 * Track beats in an audio signal using dynamic programming.
 */
beat_result_t beat_track(
    const onset_envelope_t *onset_env,
    float tempo_bpm,
    const beat_params_t *params,
    int hop_length,
    float sample_rate,
    beat_units_t units
) {
    beat_result_t result = {0};
    
    if (!onset_env || !onset_env->envelope || onset_env->length == 0) {
        ERROR("Invalid onset envelope");
        return result;
    }
    
    // Use default parameters if none provided
    beat_params_t default_params = get_default_beat_params();
    if (!params) {
        params = &default_params;
    }
    
    // Estimate tempo if not provided
    float estimated_tempo = tempo_bpm;
    if (tempo_bpm <= 0.0f) {
        tempo_params_t tempo_params = params->tempo_params ? 
            *params->tempo_params : get_default_tempo_params();
        
        tempo_result_t tempo_result = estimate_tempo(onset_env, &tempo_params, hop_length);
        if (tempo_result.bpm_estimates && tempo_result.length > 0) {
            estimated_tempo = tempo_result.bpm_estimates[0];
            result.confidence = tempo_result.confidence;
        } else {
            ERROR("Failed to estimate tempo");
            free_tempo_result(&tempo_result);
            return result;
        }
        free_tempo_result(&tempo_result);
    }
    
    if (estimated_tempo <= 0.0f) {
        ERROR("Invalid tempo estimate: %.2f BPM", estimated_tempo);
        return result;
    }
    
    // Check for empty onset envelope
    bool has_onsets = false;
    for (size_t i = 0; i < onset_env->length; i++) {
        if (onset_env->envelope[i] > 0.0f) {
            has_onsets = true;
            break;
        }
    }
    
    if (!has_onsets) {
        LOG("No onsets detected, returning empty beat sequence");
        result.tempo_bpm = estimated_tempo;
        result.frame_rate = onset_env->frame_rate;
        result.total_frames = onset_env->length;
        return result;
    }
    
    // Run the core DP beat tracker
    bool *beat_mask = dp_beat_tracker(
        onset_env->envelope,
        onset_env->length,
        estimated_tempo,
        onset_env->frame_rate,
        params->tightness,
        params->trim
    );
    
    if (!beat_mask) {
        ERROR("Beat tracking failed");
        return result;
    }
    
    // Count beats
    size_t num_beats = 0;
    for (size_t i = 0; i < onset_env->length; i++) {
        if (beat_mask[i]) num_beats++;
    }
    
    // Fill result structure
    result.tempo_bpm = estimated_tempo;
    result.frame_rate = onset_env->frame_rate;
    result.total_frames = onset_env->length;
    result.num_beats = num_beats;
    
    if (num_beats > 0) {
        // Allocate arrays
        result.beat_frames = (size_t *)malloc(num_beats * sizeof(size_t));
        result.beat_times = (float *)malloc(num_beats * sizeof(float));
        
        if (!result.beat_frames || !result.beat_times) {
            ERROR("Failed to allocate beat arrays");
            free(beat_mask);
            free_beat_result(&result);
            return result;
        }
        
        // Extract beat indices
        beats_to_indices(beat_mask, onset_env->length, result.beat_frames);
        
        // Convert to requested units
        convert_beat_units(result.beat_frames, result.beat_times, num_beats,
                          hop_length, sample_rate, units);
    }
    
    // Store beat mask if not sparse
    if (!params->sparse) {
        result.beat_mask = beat_mask;
    } else {
        free(beat_mask);
    }
    
    return result;
}

/**
 * Track beats from audio signal directly.
 */
beat_result_t beat_track_audio(
    audio_data *audio,
    size_t window_size,
    size_t hop_length,
    size_t n_mels,
    const beat_params_t *params,
    beat_units_t units
) {
    beat_result_t result = {0};
    
    if (!audio || !audio->samples || audio->num_samples == 0) {
        ERROR("Invalid audio data");
        return result;
    }
    
    // Compute STFT
    float *window_values = (float *)malloc(window_size * sizeof(float));
    if (!window_values) {
        ERROR("Failed to allocate window");
        return result;
    }
    
    window_function(window_values, window_size, "hann");
    
    fft_t fft_plan = init_fftw_plan(window_size, "cache/FFT");
    stft_t stft_result = stft(audio, window_size, hop_length, window_values, &fft_plan);
    
    free(window_values);
    
    if (!stft_result.magnitudes) {
        ERROR("STFT computation failed");
        free_fft_plan(&fft_plan);
        return result;
    }
    
    // Compute mel spectrogram
    const size_t filterbank_size = stft_result.num_frequencies * (n_mels + 2);
    float *filterbank = (float *)calloc(filterbank_size, sizeof(float));
    
    filterbank_config_t config = get_default_filterbank_config(0.0f, audio->sample_rate / 2.0f, n_mels, audio->sample_rate, window_size);
    filter_bank_t bank = gen_filterbank(&config, filterbank);
    
    bounds2d_t bounds = {0};
    bounds.time.start_d = 0;
    bounds.time.end_d = stft_result.output_size;
    bounds.freq.start_d = 0;
    bounds.freq.end_d = stft_result.num_frequencies;
    
    const size_t t_len = stft_result.output_size;
    const size_t f_len = stft_result.num_frequencies;
    float *stft_power = (float *)malloc(t_len * f_len * sizeof(float));
    
    // Convert magnitude to power
    for (size_t i = 0; i < t_len * f_len; i++) {
        stft_power[i] = stft_result.magnitudes[i] * stft_result.magnitudes[i];
    }
    
    float *mel_spec_time_major = apply_filter_bank(stft_power, n_mels, f_len, filterbank, &bounds);
    
    if (!mel_spec_time_major) {
        ERROR("Mel spectrogram computation failed");
        free(stft_power);
        free(filterbank);
        free(bank.freq_indexs);
        free(bank.weights);
        free_stft(&stft_result);
        free_fft_plan(&fft_plan);
        return result;
    }
    
    // Transpose mel_spec from (time × mels) to (mels × time)
    float *mel_spec = (float *)malloc(n_mels * t_len * sizeof(float));
    for (size_t t = 0; t < t_len; t++) {
        for (size_t m = 0; m < n_mels; m++) {
            float val = mel_spec_time_major[t * n_mels + m];
            mel_spec[m * t_len + t] = fabsf(val);
        }
    }
    
    // Convert to dB
    float *mel_db = (float *)malloc(n_mels * t_len * sizeof(float));
    float max_val = mel_spec[0];
    for (size_t i = 1; i < n_mels * t_len; i++) {
        if (mel_spec[i] > max_val) max_val = mel_spec[i];
    }
    power_to_db(mel_spec, mel_db, n_mels * t_len, max_val);
    
    // Compute onset strength
    float frame_rate = (float)audio->sample_rate / hop_length;
    onset_envelope_t onset_env = onset_strength(
        mel_db, n_mels, t_len,
        1,           // lag
        1,           // max_size
        false,       // detrend
        AGG_MEDIAN,  // aggregation
        NULL,        // ref_spec
        frame_rate
    );
    
    // Track beats
    result = beat_track(&onset_env, 0.0f, params, hop_length, audio->sample_rate, units);
    
    // Cleanup
    free_onset_envelope(&onset_env);
    free(mel_db);
    free(mel_spec);
    free(mel_spec_time_major);
    free(stft_power);
    free(filterbank);
    free(bank.freq_indexs);
    free(bank.weights);
    free_stft(&stft_result);
    free_fft_plan(&fft_plan);
    
    return result;
}

/**
 * Core dynamic programming beat tracker.
 */
bool *dp_beat_tracker(
    const float *onset_env,
    size_t num_frames,
    float tempo_bpm,
    float frame_rate,
    float tightness,
    bool trim
) {
    if (!onset_env || num_frames == 0 || tempo_bpm <= 0.0f || frame_rate <= 0.0f) {
        ERROR("Invalid parameters for DP beat tracker");
        return NULL;
    }
    
    // Allocate working arrays
    float *normalized = (float *)malloc(num_frames * sizeof(float));
    float *local_score = (float *)malloc(num_frames * sizeof(float));
    int *backlink = (int *)malloc(num_frames * sizeof(int));
    float *cumscore = (float *)malloc(num_frames * sizeof(float));
    bool *beats = (bool *)calloc(num_frames, sizeof(bool));
    
    if (!normalized || !local_score || !backlink || !cumscore || !beats) {
        ERROR("Failed to allocate DP arrays");
        free(normalized);
        free(local_score);
        free(backlink);
        free(cumscore);
        free(beats);
        return NULL;
    }
    
    // Step 1: Normalize onsets
    normalize_onsets(onset_env, normalized, num_frames);
    
    // Step 2: Compute local score
    float frames_per_beat = frame_rate * 60.0f / tempo_bpm;
    compute_local_score(normalized, local_score, num_frames, frames_per_beat);
    
    // Step 3: Run DP algorithm
    beat_track_dp(local_score, backlink, cumscore, num_frames, frames_per_beat, tightness);
    
    // Step 4: Backtrack to find beats
    dp_backtrack(backlink, cumscore, beats, num_frames);
    
    // Step 5: Trim weak beats
    trim_beats(local_score, beats, num_frames, trim);
    
    // Cleanup working arrays
    free(normalized);
    free(local_score);
    free(backlink);
    free(cumscore);
    
    return beats;
}

/**
 * Normalize onset strength envelope by standard deviation.
 */
void normalize_onsets(
    const float *onset_env,
    float *normalized,
    size_t length
) {
    if (!onset_env || !normalized || length == 0) return;
    
    float std_dev = compute_std(onset_env, length);
    if (std_dev <= 0.0f) {
        // If std is zero, just copy the input
        memcpy(normalized, onset_env, length * sizeof(float));
        return;
    }
    
    for (size_t i = 0; i < length; i++) {
        normalized[i] = onset_env[i] / std_dev;
    }
}

/**
 * Compute local score using Gaussian-weighted smoothing.
 */
void compute_local_score(
    const float *normalized_onsets,
    float *local_score,
    size_t length,
    float frames_per_beat
) {
    if (!normalized_onsets || !local_score || length == 0 || frames_per_beat <= 0.0f) return;
    
    // Gaussian kernel parameters
    float sigma = frames_per_beat / 32.0f;  // Match librosa's scaling
    int half_window = (int)roundf(frames_per_beat);
    
    // Compute local score for each frame
    for (size_t i = 0; i < length; i++) {
        float score = 0.0f;
        float weight_sum = 0.0f;
        
        // Apply Gaussian kernel
        for (int k = -half_window; k <= half_window; k++) {
            int idx = (int)i + k;
            if (idx >= 0 && idx < (int)length) {
                float weight = gaussian_weight((float)k, sigma);
                score += weight * normalized_onsets[idx];
                weight_sum += weight;
            }
        }
        
        // Normalize by weight sum
        local_score[i] = weight_sum > 0.0f ? score / weight_sum : 0.0f;
    }
}

/**
 * Core dynamic programming algorithm for beat tracking.
 */
void beat_track_dp(
    const float *local_score,
    int *backlink,
    float *cumscore,
    size_t length,
    float frames_per_beat,
    float tightness
) {
    if (!local_score || !backlink || !cumscore || length == 0) return;
    
    // Threshold for first beat
    float max_score = local_score[0];
    for (size_t i = 1; i < length; i++) {
        if (local_score[i] > max_score) max_score = local_score[i];
    }
    float score_thresh = 0.01f * max_score;
    
    // Initialize
    bool first_beat = true;
    backlink[0] = -1;
    cumscore[0] = local_score[0];
    
    // DP loop
    for (size_t i = 1; i < length; i++) {
        float best_score = -FLT_MAX;
        int beat_location = -1;
        
        // Search over possible predecessors
        int min_gap = (int)roundf(frames_per_beat / 2.0f);
        int max_gap = (int)roundf(2.0f * frames_per_beat);
        
        for (int gap = min_gap; gap <= max_gap && gap <= (int)i; gap++) {
            int loc = (int)i - gap;
            if (loc < 0) break;
            
            // Compute score with tightness penalty
            float log_gap = logf((float)gap);
            float log_expected = logf(frames_per_beat);
            float penalty = tightness * (log_gap - log_expected) * (log_gap - log_expected);
            float score = cumscore[loc] - penalty;
            
            if (score > best_score) {
                best_score = score;
                beat_location = loc;
            }
        }
        
        // Update cumulative score
        if (beat_location >= 0) {
            cumscore[i] = local_score[i] + best_score;
        } else {
            cumscore[i] = local_score[i];
        }
        
        // Handle first beat threshold
        if (first_beat && local_score[i] < score_thresh) {
            backlink[i] = -1;
        } else {
            backlink[i] = beat_location;
            first_beat = false;
        }
    }
}

/**
 * Reconstruct beat path from backlinks.
 */
void dp_backtrack(
    const int *backlink,
    const float *cumscore,
    bool *beats,
    size_t length
) {
    if (!backlink || !cumscore || !beats || length == 0) return;
    
    // Find the last beat position
    size_t tail = find_last_beat(cumscore, length);
    
    // Backtrack from tail
    int n = (int)tail;
    while (n >= 0) {
        beats[n] = true;
        n = backlink[n];
    }
}

/**
 * Remove weak leading and trailing beats.
 */
void trim_beats(
    const float *local_score,
    bool *beats,
    size_t length,
    bool trim
) {
    if (!local_score || !beats || length == 0 || !trim) return;
    
    // Compute threshold: 0.5 * RMS of smoothed beat envelope
    float sum_squares = 0.0f;
    size_t beat_count = 0;
    
    for (size_t i = 0; i < length; i++) {
        if (beats[i]) {
            sum_squares += local_score[i] * local_score[i];
            beat_count++;
        }
    }
    
    float threshold = 0.0f;
    if (beat_count > 0) {
        float rms = sqrtf(sum_squares / beat_count);
        threshold = 0.5f * rms;
    }
    
    // Trim leading beats
    for (size_t i = 0; i < length; i++) {
        if (beats[i] && local_score[i] <= threshold) {
            beats[i] = false;
        } else if (beats[i]) {
            break;  // Found first strong beat
        }
    }
    
    // Trim trailing beats
    for (int i = (int)length - 1; i >= 0; i--) {
        if (beats[i] && local_score[i] <= threshold) {
            beats[i] = false;
        } else if (beats[i]) {
            break;  // Found last strong beat
        }
    }
}

/**
 * Convert beat mask to sparse beat indices.
 */
size_t beats_to_indices(
    const bool *beat_mask,
    size_t length,
    size_t *beat_indices
) {
    if (!beat_mask || !beat_indices || length == 0) return 0;
    
    size_t count = 0;
    for (size_t i = 0; i < length; i++) {
        if (beat_mask[i]) {
            beat_indices[count++] = i;
        }
    }
    
    return count;
}

/**
 * Convert frame indices to time or sample positions.
 */
void convert_beat_units(
    const size_t *frame_indices,
    float *output,
    size_t num_beats,
    int hop_length,
    float sample_rate,
    beat_units_t units
) {
    if (!frame_indices || !output || num_beats == 0) return;
    
    switch (units) {
        case BEAT_UNITS_FRAMES:
            for (size_t i = 0; i < num_beats; i++) {
                output[i] = (float)frame_indices[i];
            }
            break;
            
        case BEAT_UNITS_SAMPLES:
            for (size_t i = 0; i < num_beats; i++) {
                output[i] = (float)(frame_indices[i] * hop_length);
            }
            break;
            
        case BEAT_UNITS_TIME:
            for (size_t i = 0; i < num_beats; i++) {
                output[i] = (float)(frame_indices[i] * hop_length) / sample_rate;
            }
            break;
    }
}

/**
 * Free memory allocated for a beat result structure.
 */
void free_beat_result(beat_result_t *result) {
    if (!result) return;
    
    free(result->beat_times);
    free(result->beat_frames);
    free(result->beat_mask);
    
    memset(result, 0, sizeof(beat_result_t));
}

// Helper function implementations

/**
 * Compute standard deviation of data array.
 */
static float compute_std(const float *data, size_t length) {
    if (!data || length == 0) return 0.0f;
    
    // Compute mean
    float sum = 0.0f;
    for (size_t i = 0; i < length; i++) {
        sum += data[i];
    }
    float mean = sum / length;
    
    // Compute variance
    float var_sum = 0.0f;
    for (size_t i = 0; i < length; i++) {
        float diff = data[i] - mean;
        var_sum += diff * diff;
    }
    
    // Use sample standard deviation (N-1)
    float variance = length > 1 ? var_sum / (length - 1) : 0.0f;
    return sqrtf(variance);
}

/**
 * Compute Gaussian weight for given distance and sigma.
 */
static float gaussian_weight(float x, float sigma) {
    if (sigma <= 0.0f) return 0.0f;
    return expf(-0.5f * (x / sigma) * (x / sigma));
}

/**
 * Find the position of the last detected beat.
 */
static size_t find_last_beat(const float *cumscore, size_t length) {
    if (!cumscore || length == 0) return 0;
    
    // Find local maxima
    float median_peak = 0.0f;
    size_t peak_count = 0;
    
    for (size_t i = 1; i < length - 1; i++) {
        if (is_local_max(cumscore, i, length)) {
            median_peak += cumscore[i];
            peak_count++;
        }
    }
    
    if (peak_count > 0) {
        median_peak /= peak_count;
    }
    
    float threshold = 0.5f * median_peak;
    
    // Find last beat above threshold
    for (int i = (int)length - 1; i >= 0; i--) {
        if (is_local_max(cumscore, i, length) && cumscore[i] >= threshold) {
            return (size_t)i;
        }
    }
    
    return length - 1;
}

/**
 * Check if a point is a local maximum.
 */
static bool is_local_max(const float *data, size_t index, size_t length) {
    if (!data || index >= length) return false;
    
    float val = data[index];
    
    // Check left neighbor
    if (index > 0 && data[index - 1] > val) return false;
    
    // Check right neighbor
    if (index < length - 1 && data[index + 1] > val) return false;
    
    return true;
}
