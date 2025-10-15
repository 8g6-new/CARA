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
 * Return a beat_params_t initialized with library defaults.
 *
 * @returns A beat_params_t configured as follows:
 *  - tightness = 100.0f
 *  - trim = true
 *  - sparse = true
 *  - tempo_params = NULL (use default tempo estimation parameters)
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
 * Track beats in an onset envelope using a dynamic-programming beat tracker.
 *
 * Processes the provided onset envelope to produce beat locations and related
 * metadata. If `tempo_bpm` is less than or equal to 0, tempo is estimated
 * from the onset envelope using `params->tempo_params` or default tempo
 * parameters. If `params` is NULL, default beat parameters are used. When the
 * onset envelope contains no detected onsets, an empty result is returned with
 * tempo, frame_rate, and total_frames populated.
 *
 * @param onset_env Pointer to the onset envelope and its framing information.
 * @param tempo_bpm Specified tempo in beats per minute; when <= 0, tempo will
 *                  be estimated from `onset_env`.
 * @param params    Optional beat tracking parameters; when NULL, defaults are used.
 * @param hop_length Number of samples between successive frames (used for unit conversions).
 * @param sample_rate Audio sample rate in Hz (used for unit conversions).
 * @param units     Desired units for beat times (frames, samples, or seconds).
 *
 * @returns A populated `beat_result_t` containing:
 *          - `tempo_bpm`: the used tempo (specified or estimated),
 *          - `confidence`: estimation confidence when tempo was estimated,
 *          - `frame_rate` and `total_frames` copied from `onset_env`,
 *          - `num_beats`, `beat_frames`, and `beat_times` when beats were found,
 *          - `beat_mask` populated only if `params->sparse` is false;
 *          an empty/zeroed `beat_result_t` is returned on invalid input or failure.
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
 * Detect beats directly from raw audio and return a populated beat_result_t.
 *
 * Processes the provided audio buffer (STFT → mel spectrogram → onset strength)
 * and runs the beat tracking pipeline with the given parameters.
 *
 * @param audio Pointer to audio_data containing samples and sample_rate.
 * @param window_size STFT analysis window size in samples.
 * @param hop_length Hop length between successive STFT frames in samples.
 * @param n_mels Number of mel bands to use for the mel spectrogram.
 * @param params Optional beat tracking parameters; pass NULL to use defaults.
 * @param units Units for returned beat positions (frames, samples, or time).
 * @returns A beat_result_t with detected beats, tempo estimate, frame_rate,
 *          and allocated arrays for beat_times / beat_frames / beat_mask as
 *          appropriate; returns an empty-initialized result on error or if no
 *          beats are detected. */
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
    if (!filterbank) {
        ERROR("Failed to allocate filterbank");
        free_stft(&stft_result);
        free_fft_plan(&fft_plan);
        return result;
    }
    
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
    if (!stft_power) {
        ERROR("Failed to allocate stft_power");
        free(filterbank);
        free(bank.freq_indexs);
        free(bank.weights);
        free_stft(&stft_result);
        free_fft_plan(&fft_plan);
        return result;
    }
    
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
    if (!mel_spec) {
        ERROR("Failed to allocate mel_spec");
        free(mel_spec_time_major);
        free(stft_power);
        free(filterbank);
        free(bank.freq_indexs);
        free(bank.weights);
        free_stft(&stft_result);
        free_fft_plan(&fft_plan);
        return result;
    }
    for (size_t t = 0; t < t_len; t++) {
        for (size_t m = 0; m < n_mels; m++) {
            float val = mel_spec_time_major[t * n_mels + m];
            mel_spec[m * t_len + t] = fabsf(val);
        }
    }
    
    // Convert to dB
    float *mel_db = (float *)malloc(n_mels * t_len * sizeof(float));
    if (!mel_db) {
        ERROR("Failed to allocate mel_db");
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
 * Identify beat positions in an onset strength envelope using a dynamic-programming algorithm.
 *
 * Given an onset strength sequence and an estimated tempo, computes a boolean mask of beat
 * positions (one entry per frame). The function normalizes the onset envelope, scores
 * frames for beat-likelihood, performs dynamic programming to find an optimal beat sequence,
 * backtracks to produce beat locations, and optionally trims weak leading/trailing beats.
 *
 * @param onset_env Pointer to an array of onset strength values (length = num_frames).
 * @param num_frames Number of frames in `onset_env`.
 * @param tempo_bpm Estimated tempo in beats per minute; must be greater than 0.
 * @param frame_rate Frame rate (frames per second) of `onset_env`; must be greater than 0.
 * @param tightness Penalty scale controlling tempo deviation tolerance (higher = stricter).
 * @param trim If true, remove weak leading and trailing beats from the resulting beat mask.
 *
 * @returns Pointer to a newly allocated boolean array of length `num_frames` where `true`
 *          indicates a detected beat frame; returns NULL on invalid input or allocation failure.
 *          Caller is responsible for freeing the returned array.
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
 * Normalize an onset strength envelope by its sample standard deviation.
 *
 * If the computed standard deviation is less than or equal to zero, the input
 * array is copied to the output unchanged.
 *
 * @param onset_env Input onset strength array of length `length`.
 * @param normalized Output buffer that receives the normalized onset strengths;
 *        must be able to hold `length` floats.
 * @param length Number of elements in `onset_env` and `normalized`.
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
 * Compute a Gaussian-weighted local onset score for each frame.
 *
 * Applies a Gaussian window centered at each frame to produce a smoothed
 * local score from the input normalized onset envelope. The kernel width
 * is derived from `frames_per_beat`.
 *
 * @param normalized_onsets Input onset values, length `length`. Must not be NULL.
 * @param local_score Output array of length `length` receiving the per-frame scores. Must not be NULL.
 * @param length Number of frames in the input and output arrays.
 * @param frames_per_beat Expected number of frames per beat; controls the Gaussian kernel width and must be > 0.
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
 * Compute cumulative beat scores and predecessor links using dynamic programming.
 *
 * Given an array of per-frame local scores, fills `cumscore` with the best achievable
 * cumulative score at each frame and fills `backlink` with the index of the chosen
 * predecessor frame for that best path (or -1 when no predecessor is chosen).
 *
 * @param local_score Array of length `length` containing per-frame local scores.
 * @param backlink Output array of length `length` that will be populated with
 *                 predecessor indices for each frame (use -1 to indicate no predecessor).
 * @param cumscore Output array of length `length` that will be populated with the
 *                 cumulative best score ending at each frame.
 * @param length Number of frames (length of the arrays).
 * @param frames_per_beat Expected number of frames between successive beats (tempo-derived).
 * @param tightness Penalty scaling factor that controls tolerance for deviations from `frames_per_beat`.
 *
 * If any pointer is NULL or `length` is zero, the function returns without modifying outputs.
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
 * Backtracks the DP predecessor chain to mark detected beat positions.
 *
 * Scans backward from the last beat index (determined from `cumscore`) and
 * sets the corresponding entries in `beats` to `true` following the `backlink`
 * chain until a sentinel predecessor (negative index) is reached.
 *
 * @param backlink Array of predecessor indices for each frame; a negative value
 *                 indicates no predecessor. Must have length `length`.
 * @param cumscore Array of cumulative scores per frame used to locate the
 *                 final beat. Must have length `length`.
 * @param beats    Output boolean array of length `length`; entries corresponding
 *                 to backtracked beat frames are set to `true`.
 * @param length   Number of frames / length of the input and output arrays.
 *
 * If `backlink`, `cumscore`, or `beats` is NULL, or `length` is zero, the
 * function performs no action.
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
 * Remove weak leading and trailing detected beats from a beat mask.
 *
 * Computes a threshold equal to 0.5 times the root-mean-square (RMS) of
 * `local_score` values at positions currently marked as beats, then clears
 * beat flags at the start and end of `beats` whose `local_score` is less
 * than or equal to that threshold. The array `beats` is modified in-place.
 *
 * @param local_score Array of per-frame local beat scores; must be length `length`.
 * @param beats Boolean beat mask of length `length`; entries are updated in-place.
 * @param length Number of frames in `local_score` and `beats`.
 * @param trim If false, the function returns immediately without modifying `beats`.
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
 * Convert a boolean per-frame beat mask into a compact array of frame indices where beats occur.
 *
 * @param beat_mask Boolean array of length `length` indicating beat presence per frame.
 * @param length Number of elements in `beat_mask`.
 * @param beat_indices Output buffer that will be filled with the frame indices of detected beats; must have space for at least `length` entries.
 * @returns The number of beat indices written into `beat_indices`.
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
 * Convert beat frame indices into frames, samples, or time values.
 *
 * If any pointer is NULL or `num_beats` is zero, the function returns without modifying `output`.
 * @param frame_indices Array of beat indices expressed in frames.
 * @param output Destination array (must have space for `num_beats` floats) to receive converted values.
 * @param num_beats Number of beat indices to convert.
 * @param hop_length Number of audio samples per frame (used to compute sample and time values).
 * @param sample_rate Audio sampling rate in Hz (used to convert samples to seconds when `units` is `BEAT_UNITS_TIME`).
 * @param units Target units for conversion: `BEAT_UNITS_FRAMES`, `BEAT_UNITS_SAMPLES`, or `BEAT_UNITS_TIME`.
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
 * Release memory held by a beat_result_t and reset it to zero.
 *
 * Frees any allocated arrays inside the result (beat_times, beat_frames, beat_mask)
 * and clears all fields of the structure. Passing NULL has no effect.
 *
 * @param result Pointer to the beat_result_t to free and reset.
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
 * Calculate the sample standard deviation of an array of floats.
 *
 * @param data Pointer to the input array; may be NULL.
 * @param length Number of elements in the array.
 * @returns Sample standard deviation (uses division by N-1). Returns 0.0 if `data` is NULL, `length` is 0, or `length` is 1.
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
 * Compute the Gaussian kernel weight for a given distance and standard deviation.
 *
 * @param x Distance from the Gaussian center (signed or unsigned).
 * @param sigma Standard deviation of the Gaussian; if sigma <= 0 the function returns 0.
 * @returns Weight in the range [0, 1], equal to 1 at x == 0 and decreasing as |x| increases; 0 if sigma <= 0.
 */
static float gaussian_weight(float x, float sigma) {
    if (sigma <= 0.0f) return 0.0f;
    return expf(-0.5f * (x / sigma) * (x / sigma));
}

/**
 * Locate the index of the last significant local maximum in a cumulative-score array.
 *
 * Scans `cumscore` for local maxima, computes the mean value of those peaks, and uses
 * half of that mean as a threshold to find the last peak considered significant.
 *
 * @param cumscore Array of cumulative scores per frame.
 * @param length Number of elements in `cumscore`.
 * @returns Index of the last local maximum whose value is greater than or equal to
 *          0.5 * (mean of all local-maximum values). If no peaks are found, returns
 *          length - 1. If `cumscore` is NULL or `length` is 0, returns 0.
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
 * Determine whether the value at the given index is a local maximum compared to its immediate neighbors.
 * 
 * @param data Array of float values to inspect.
 * @param index Index of the point to test.
 * @param length Number of elements in `data`.
 * @returns `true` if `data[index]` is greater than or equal to its immediate neighbor(s) (handles edges by comparing only existing neighbors), `false` otherwise.
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