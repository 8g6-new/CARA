/**
 * @file test_beat_track.c
 * @brief Test program for beat tracking
 * 
 * This program tests the beat tracking module by:
 * 1. Loading a test audio file
 * 2. Computing STFT and mel spectrogram
 * 3. Computing onset strength envelope
 * 4. Estimating tempo
 * 5. Tracking beats using various parameters
 * 6. Saving results for comparison with librosa
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "audio_tools/audio_visualizer.h"
#include "audio_tools/beat_track.h"
#include "utils/bench.h"

/**
 * Save beat tracking results to a text file for comparison with librosa.
 */
static void save_beat_results(const char *filename, const beat_result_t *beat_res, 
                             const beat_params_t *params) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        ERROR("Failed to open file for writing: %s", filename);
        return;
    }
    
    fprintf(fp, "# Beat tracking results\n");
    fprintf(fp, "# Parameters:\n");
    fprintf(fp, "#   tightness: %.2f\n", params->tightness);
    fprintf(fp, "#   trim: %s\n", params->trim ? "true" : "false");
    fprintf(fp, "#   sparse: %s\n", params->sparse ? "true" : "false");
    fprintf(fp, "# Results:\n");
    fprintf(fp, "#   num_beats: %zu\n", beat_res->num_beats);
    fprintf(fp, "#   tempo_bpm: %.6f\n", beat_res->tempo_bpm);
    fprintf(fp, "#   confidence: %.6f\n", beat_res->confidence);
    fprintf(fp, "#   frame_rate: %.2f Hz\n", beat_res->frame_rate);
    fprintf(fp, "#   total_frames: %zu\n", beat_res->total_frames);
    fprintf(fp, "# Format: beat_index frame_position time_seconds\n");
    
    for (size_t i = 0; i < beat_res->num_beats; i++) {
        fprintf(fp, "%zu %zu %.6f\n", i, beat_res->beat_frames[i], beat_res->beat_times[i]);
    }
    
    fclose(fp);
    LOG("Saved beat results to: %s", filename);
}

/**
 * Save beat mask for dense output analysis.
 */
static void save_beat_mask(const char *filename, const beat_result_t *beat_res) {
    if (!beat_res->beat_mask) {
        LOG("No beat mask available for saving");
        return;
    }
    
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        ERROR("Failed to open file for writing: %s", filename);
        return;
    }
    
    fprintf(fp, "# Beat mask (dense format)\n");
    fprintf(fp, "# Total frames: %zu\n", beat_res->total_frames);
    fprintf(fp, "# Format: frame_index is_beat\n");
    
    for (size_t i = 0; i < beat_res->total_frames; i++) {
        fprintf(fp, "%zu %d\n", i, beat_res->beat_mask[i] ? 1 : 0);
    }
    
    fclose(fp);
    LOG("Saved beat mask to: %s", filename);
}

/**
 * Print statistics about the beat tracking results.
 */
static void print_beat_stats(const beat_result_t *beat_res, const beat_params_t *params) {
    LOG("%s%s%s", BAR_COLOR, line, RESET);
    LOG("🥁  %sBeat Tracking Results%s", BRIGHT_CYAN, RESET);
    LOG("%s%s%s", BAR_COLOR, line, RESET);
    
    LOG("📊  Parameters:");
    LOG("    Tightness:     %.2f", params->tightness);
    LOG("    Trim:          %s", params->trim ? "Yes" : "No");
    LOG("    Sparse:        %s", params->sparse ? "Yes" : "No");
    
    LOG("🎯  Results:");
    LOG("    Beats Found:   %zu", beat_res->num_beats);
    LOG("    Tempo BPM:     %.2f", beat_res->tempo_bpm);
    LOG("    Confidence:    %.6f", beat_res->confidence);
    LOG("    Frame Rate:    %.2f Hz", beat_res->frame_rate);
    LOG("    Total Frames:  %zu", beat_res->total_frames);
    
    if (beat_res->num_beats > 0) {
        LOG("    First Beat:    %.3f seconds (frame %zu)", 
            beat_res->beat_times[0], beat_res->beat_frames[0]);
        LOG("    Last Beat:     %.3f seconds (frame %zu)", 
            beat_res->beat_times[beat_res->num_beats - 1], 
            beat_res->beat_frames[beat_res->num_beats - 1]);
        
        // Compute beat intervals
        if (beat_res->num_beats > 1) {
            float total_duration = beat_res->beat_times[beat_res->num_beats - 1] - beat_res->beat_times[0];
            float avg_interval = total_duration / (beat_res->num_beats - 1);
            float estimated_bpm = 60.0f / avg_interval;
            
            LOG("    Duration:      %.3f seconds", total_duration);
            LOG("    Avg Interval:  %.3f seconds", avg_interval);
            LOG("    Interval BPM:  %.2f", estimated_bpm);
        }
    }
    
    LOG("%s%s%s", BAR_COLOR, line, RESET);
}

/**
 * Test beat tracking with different parameter sets.
 */
static void test_beat_variations(const onset_envelope_t *onset_env, int hop_length,
                                float sample_rate, const char *base_output_path) {
    // Test 1: Default parameters
    LOG("\n%s=== Test 1: Default Parameters ===%s", BRIGHT_YELLOW, RESET);
    beat_params_t default_params = get_default_beat_params();
    
    START_TIMING();
    beat_result_t beat_default = beat_track(onset_env, 0.0f, &default_params, 
                                           hop_length, sample_rate, BEAT_UNITS_TIME);
    END_TIMING("beat_default");
    
    if (beat_default.num_beats > 0 || beat_default.tempo_bpm > 0) {
        print_beat_stats(&beat_default, &default_params);
        
        char output_file[512];
        snprintf(output_file, sizeof(output_file), "%s_default.txt", base_output_path);
        save_beat_results(output_file, &beat_default, &default_params);
    }
    
    // Test 2: Higher tightness
    LOG("\n%s=== Test 2: Higher Tightness (200) ===%s", BRIGHT_YELLOW, RESET);
    beat_params_t tight_params = default_params;
    tight_params.tightness = 200.0f;
    
    START_TIMING();
    beat_result_t beat_tight = beat_track(onset_env, 0.0f, &tight_params, 
                                         hop_length, sample_rate, BEAT_UNITS_TIME);
    END_TIMING("beat_tight");
    
    if (beat_tight.num_beats > 0 || beat_tight.tempo_bpm > 0) {
        print_beat_stats(&beat_tight, &tight_params);
        
        char output_file[512];
        snprintf(output_file, sizeof(output_file), "%s_tight.txt", base_output_path);
        save_beat_results(output_file, &beat_tight, &tight_params);
    }
    
    // Test 3: Lower tightness
    LOG("\n%s=== Test 3: Lower Tightness (50) ===%s", BRIGHT_YELLOW, RESET);
    beat_params_t loose_params = default_params;
    loose_params.tightness = 50.0f;
    
    START_TIMING();
    beat_result_t beat_loose = beat_track(onset_env, 0.0f, &loose_params, 
                                         hop_length, sample_rate, BEAT_UNITS_TIME);
    END_TIMING("beat_loose");
    
    if (beat_loose.num_beats > 0 || beat_loose.tempo_bpm > 0) {
        print_beat_stats(&beat_loose, &loose_params);
        
        char output_file[512];
        snprintf(output_file, sizeof(output_file), "%s_loose.txt", base_output_path);
        save_beat_results(output_file, &beat_loose, &loose_params);
    }
    
    // Test 4: No trimming
    LOG("\n%s=== Test 4: No Trimming ===%s", BRIGHT_YELLOW, RESET);
    beat_params_t no_trim_params = default_params;
    no_trim_params.trim = false;
    
    START_TIMING();
    beat_result_t beat_no_trim = beat_track(onset_env, 0.0f, &no_trim_params, 
                                           hop_length, sample_rate, BEAT_UNITS_TIME);
    END_TIMING("beat_no_trim");
    
    if (beat_no_trim.num_beats > 0 || beat_no_trim.tempo_bpm > 0) {
        print_beat_stats(&beat_no_trim, &no_trim_params);
        
        char output_file[512];
        snprintf(output_file, sizeof(output_file), "%s_no_trim.txt", base_output_path);
        save_beat_results(output_file, &beat_no_trim, &no_trim_params);
    }
    
    // Test 5: Dense output (beat mask)
    LOG("\n%s=== Test 5: Dense Output (Beat Mask) ===%s", BRIGHT_YELLOW, RESET);
    beat_params_t dense_params = default_params;
    dense_params.sparse = false;
    
    START_TIMING();
    beat_result_t beat_dense = beat_track(onset_env, 0.0f, &dense_params, 
                                         hop_length, sample_rate, BEAT_UNITS_TIME);
    END_TIMING("beat_dense");
    
    if (beat_dense.num_beats > 0 || beat_dense.tempo_bpm > 0) {
        print_beat_stats(&beat_dense, &dense_params);
        
        char output_file[512];
        snprintf(output_file, sizeof(output_file), "%s_dense.txt", base_output_path);
        save_beat_results(output_file, &beat_dense, &dense_params);
        
        char mask_file[512];
        snprintf(mask_file, sizeof(mask_file), "%s_mask.txt", base_output_path);
        save_beat_mask(mask_file, &beat_dense);
    }
    
    // Test 6: Frame units
    LOG("\n%s=== Test 6: Frame Units ===%s", BRIGHT_YELLOW, RESET);
    START_TIMING();
    beat_result_t beat_frames = beat_track(onset_env, 0.0f, &default_params, 
                                          hop_length, sample_rate, BEAT_UNITS_FRAMES);
    END_TIMING("beat_frames");
    
    if (beat_frames.num_beats > 0 || beat_frames.tempo_bpm > 0) {
        print_beat_stats(&beat_frames, &default_params);
        
        char output_file[512];
        snprintf(output_file, sizeof(output_file), "%s_frames.txt", base_output_path);
        save_beat_results(output_file, &beat_frames, &default_params);
    }
    
    // Cleanup
    free_beat_result(&beat_default);
    free_beat_result(&beat_tight);
    free_beat_result(&beat_loose);
    free_beat_result(&beat_no_trim);
    free_beat_result(&beat_dense);
    free_beat_result(&beat_frames);
}

/**
 * Test direct audio beat tracking.
 */
static void test_audio_beat_tracking(audio_data *audio, const char *base_output_path) {
    LOG("\n%s=== Testing Direct Audio Beat Tracking ===%s", BRIGHT_GREEN, RESET);
    
    const size_t window_size = 2048;
    const size_t hop_length = 512;
    const size_t n_mels = 128;
    
    beat_params_t params = get_default_beat_params();
    
    START_TIMING();
    beat_result_t beat_audio = beat_track_audio(audio, window_size, hop_length, n_mels,
                                               &params, BEAT_UNITS_TIME);
    END_TIMING("beat_audio");
    
    if (beat_audio.num_beats > 0 || beat_audio.tempo_bpm > 0) {
        print_beat_stats(&beat_audio, &params);
        
        char output_file[512];
        snprintf(output_file, sizeof(output_file), "%s_audio_direct.txt", base_output_path);
        save_beat_results(output_file, &beat_audio, &params);
    }
    
    free_beat_result(&beat_audio);
}

int main(int argc, char *argv[]) {
    const char *input_file = "tests/files/riad.wav";
    const char *output_base = "outputs/beat_tracking";
    
    // Allow custom input file
    if (argc > 1) {
        input_file = argv[1];
    }
    if (argc > 2) {
        output_base = argv[2];
    }
    
    LOG("%s=== Beat Tracking Test ===%s", BRIGHT_CYAN, RESET);
    LOG("Input file: %s", input_file);
    LOG("Output base: %s", output_base);
    
    // Step 1: Load audio
    START_TIMING();
    audio_data audio = auto_detect(input_file);
    END_TIMING("audio_load");
    
    if (!audio.samples || audio.num_samples == 0) {
        ERROR("Failed to load audio file: %s", input_file);
        return 1;
    }
    
    LOG("Loaded audio: %zu samples, %f Hz, %zu channels",
        audio.num_samples, (float)audio.sample_rate, audio.channels);
    
    // Step 2: Compute STFT
    const size_t window_size = 2048;
    const size_t hop_size = 512;
    const size_t n_mels = 128;
    
    float *window_values = (float *)malloc(window_size * sizeof(float));
    window_function(window_values, window_size, "hann");
    
    START_TIMING();
    fft_t fft_plan = init_fftw_plan(window_size, "cache/FFT");
    END_TIMING("fft_init");
    
    START_TIMING();
    stft_t stft_result = stft(&audio, window_size, hop_size, window_values, &fft_plan);
    END_TIMING("stft");
    
    if (!stft_result.magnitudes) {
        ERROR("STFT computation failed");
        free(window_values);
        free_audio(&audio);
        return 1;
    }
    
    LOG("STFT computed: %zu frequencies × %zu frames",
        stft_result.num_frequencies, stft_result.output_size);
    
    // Step 3: Compute mel spectrogram
    const size_t filterbank_size = (stft_result.num_frequencies) * (n_mels + 2);
    float *filterbank = (float *)calloc(filterbank_size, sizeof(float));
    
    START_TIMING();
    filterbank_config_t config = get_default_filterbank_config(0.0f, audio.sample_rate / 2.0f, n_mels, audio.sample_rate, window_size);
    filter_bank_t bank = gen_filterbank(&config, filterbank);
    END_TIMING("mel_filterbank");
    
    bounds2d_t bounds = {0};
    bounds.time.start_d = 0;
    bounds.time.end_d = stft_result.output_size;
    bounds.freq.start_d = 0;
    bounds.freq.end_d = stft_result.num_frequencies;

    const size_t t_len = stft_result.output_size;
    const size_t f_len = stft_result.num_frequencies;
    float *stft_power = (float *)malloc(t_len * f_len * sizeof(float));

    // Convert magnitude to power
    START_TIMING();
    for (size_t i = 0; i < t_len * f_len; i++) {
        stft_power[i] = stft_result.magnitudes[i] * stft_result.magnitudes[i];
    }
    END_TIMING("mag_to_power");
    
    START_TIMING();
    float *mel_spec_time_major = apply_filter_bank(stft_power, n_mels, f_len, filterbank, &bounds);
    END_TIMING("mel_apply");
    
    if (!mel_spec_time_major) {
        ERROR("Mel spectrogram computation failed");
        free(stft_power);
        free(filterbank);
        free(window_values);
        free_stft(&stft_result);
        free_audio(&audio);
        return 1;
    }
    
    // Transpose mel_spec from (time × mels) to (mels × time)
    float *mel_spec = (float *)malloc(n_mels * t_len * sizeof(float));
    if (!mel_spec) {
        ERROR("Failed to allocate transposed mel spectrogram");
        free(mel_spec_time_major);
        free(stft_power);
        free(filterbank);
        free(window_values);
        free_stft(&stft_result);
        free_audio(&audio);
        return 1;
    }
    
    for (size_t t = 0; t < t_len; t++) {
        for (size_t m = 0; m < n_mels; m++) {
            float val = mel_spec_time_major[t * n_mels + m];
            mel_spec[m * t_len + t] = fabsf(val);
        }
    }
    free(mel_spec_time_major);
    
    LOG("Mel spectrogram computed: %zu mels × %zu frames", n_mels, t_len);
    
    // Step 4: Convert to dB
    START_TIMING();
    float *mel_db = (float *)malloc(n_mels * t_len * sizeof(float));
    
    float max_val = mel_spec[0];
    for (size_t i = 1; i < n_mels * t_len; i++) {
        if (mel_spec[i] > max_val) max_val = mel_spec[i];
    }
    
    power_to_db(mel_spec, mel_db, n_mels * t_len, max_val);
    END_TIMING("power_to_db");
    
    // Step 5: Compute onset strength envelope
    float frame_rate = (float)audio.sample_rate / hop_size;
    
    LOG("\n%s=== Computing Onset Strength ===%s", BRIGHT_YELLOW, RESET);
    START_TIMING();
    onset_envelope_t onset_env = onset_strength(
        mel_db, n_mels, t_len,
        1,           // lag
        1,           // max_size
        false,       // detrend
        AGG_MEDIAN,  // aggregation
        NULL,        // ref_spec
        frame_rate
    );
    END_TIMING("onset_strength");
    
    if (!onset_env.envelope) {
        ERROR("Failed to compute onset strength envelope");
        free(mel_db);
        free(mel_spec);
        free(stft_power);
        free(filterbank);
        free(bank.freq_indexs);
        free(bank.weights);
        free(window_values);
        free_stft(&stft_result);
        free_fft_plan(&fft_plan);
        free_audio(&audio);
        return 1;
    }
    
    LOG("Onset envelope computed: %zu frames, %.2f Hz", 
        onset_env.length, onset_env.frame_rate);
    
    // Step 6: Test beat tracking with various parameters
    LOG("\n%s=== Testing Beat Tracking ===%s", BRIGHT_GREEN, RESET);
    test_beat_variations(&onset_env, hop_size, audio.sample_rate, output_base);
    
    // Step 7: Test direct audio beat tracking
    test_audio_beat_tracking(&audio, output_base);
    
    // Print benchmark summary
    LOG("\n%s=== Benchmark Summary ===%s", BRIGHT_GREEN, RESET);
    print_bench_ranked();
    
    // Cleanup
    free_onset_envelope(&onset_env);
    free(mel_db);
    free(mel_spec);
    free(stft_power);
    free(filterbank);
    free(bank.freq_indexs);
    free(bank.weights);
    free(window_values);
    free_stft(&stft_result);
    free_fft_plan(&fft_plan);
    free_audio(&audio);
    
    LOG("\n%s✅ Beat tracking test completed successfully!%s", BRIGHT_GREEN, RESET);
    return 0;
}
