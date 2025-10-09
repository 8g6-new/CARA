/**
 * @file test_tempo.c
 * @brief Test program for tempo estimation
 * 
 * This program tests the tempo estimation module by:
 * 1. Loading a test audio file
 * 2. Computing STFT and mel spectrogram
 * 3. Computing onset strength envelope
 * 4. Estimating tempo using various parameters
 * 5. Saving results for comparison with librosa
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "audio_tools/audio_visualizer.h"
#include "audio_tools/onset.h"
#include "audio_tools/tempo.h"
#include "utils/bench.h"

/**
 * Save tempo estimation results to a text file for comparison with librosa.
 */
static void save_tempo_results(const char *filename, const tempo_result_t *tempo_res, 
                              const tempo_params_t *params) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        ERROR("Failed to open file for writing: %s", filename);
        return;
    }
    
    fprintf(fp, "# Tempo estimation results\n");
    fprintf(fp, "# Parameters:\n");
    fprintf(fp, "#   start_bpm: %.2f\n", params->start_bpm);
    fprintf(fp, "#   std_bpm: %.2f\n", params->std_bpm);
    fprintf(fp, "#   max_tempo: %.2f\n", params->max_tempo);
    fprintf(fp, "#   ac_size: %.2f seconds\n", params->ac_size);
    fprintf(fp, "#   use_prior: %s\n", params->use_prior ? "true" : "false");
    fprintf(fp, "#   aggregate: %s\n", params->aggregate ? "true" : "false");
    fprintf(fp, "# Results:\n");
    fprintf(fp, "#   length: %zu estimates\n", tempo_res->length);
    fprintf(fp, "#   frame_rate: %.2f Hz\n", tempo_res->frame_rate);
    fprintf(fp, "#   is_global: %s\n", tempo_res->is_global ? "true" : "false");
    fprintf(fp, "#   confidence: %.6f\n", tempo_res->confidence);
    fprintf(fp, "# Format: estimate_index tempo_bpm\n");
    
    for (size_t i = 0; i < tempo_res->length; i++) {
        fprintf(fp, "%zu %.6f\n", i, tempo_res->bpm_estimates[i]);
    }
    
    fclose(fp);
    LOG("Saved tempo results to: %s", filename);
}

/**
 * Save autocorrelation data for analysis.
 */
static void save_autocorr_data(const char *filename, const autocorr_result_t *autocorr_res,
                              const float *bpm_freqs) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        ERROR("Failed to open file for writing: %s", filename);
        return;
    }
    
    fprintf(fp, "# Autocorrelation data\n");
    fprintf(fp, "# Length: %zu lags\n", autocorr_res->length);
    fprintf(fp, "# Max lag: %.2f seconds\n", autocorr_res->max_lag_seconds);
    fprintf(fp, "# Format: lag_index autocorr_value bpm_frequency\n");
    
    for (size_t i = 0; i < autocorr_res->length; i++) {
        fprintf(fp, "%zu %.8f %.6f\n", i, autocorr_res->autocorr[i], 
                bpm_freqs ? bpm_freqs[i] : 0.0f);
    }
    
    fclose(fp);
    LOG("Saved autocorrelation data to: %s", filename);
}

/**
 * Print statistics about the tempo estimation.
 */
static void print_tempo_stats(const tempo_result_t *tempo_res, const tempo_params_t *params) {
    LOG("%s%s%s", BAR_COLOR, line, RESET);
    LOG("🎵  %sTempo Estimation Results%s", BRIGHT_CYAN, RESET);
    LOG("%s%s%s", BAR_COLOR, line, RESET);
    
    LOG("📊  Parameters:");
    LOG("    Start BPM:     %.2f", params->start_bpm);
    LOG("    Std BPM:       %.2f", params->std_bpm);
    LOG("    Max Tempo:     %.2f BPM", params->max_tempo);
    LOG("    AC Size:       %.2f seconds", params->ac_size);
    LOG("    Use Prior:     %s", params->use_prior ? "Yes" : "No");
    LOG("    Aggregate:     %s", params->aggregate ? "Global" : "Frame-wise");
    
    LOG("🎯  Results:");
    LOG("    Estimates:     %zu", tempo_res->length);
    LOG("    Frame Rate:    %.2f Hz", tempo_res->frame_rate);
    LOG("    Is Global:     %s", tempo_res->is_global ? "Yes" : "No");
    LOG("    Confidence:    %.6f", tempo_res->confidence);
    
    if (tempo_res->length > 0) {
        LOG("    Estimated BPM: %.2f", tempo_res->bpm_estimates[0]);
        
        if (tempo_res->length > 1) {
            float min_bpm = tempo_res->bpm_estimates[0];
            float max_bpm = tempo_res->bpm_estimates[0];
            float sum_bpm = 0.0f;
            
            for (size_t i = 0; i < tempo_res->length; i++) {
                float bpm = tempo_res->bpm_estimates[i];
                if (bpm < min_bpm) min_bpm = bpm;
                if (bpm > max_bpm) max_bpm = bpm;
                sum_bpm += bpm;
            }
            
            float mean_bpm = sum_bpm / tempo_res->length;
            LOG("    BPM Range:     %.2f - %.2f", min_bpm, max_bpm);
            LOG("    Mean BPM:      %.2f", mean_bpm);
        }
    }
    
    LOG("%s%s%s", BAR_COLOR, line, RESET);
}

/**
 * Test tempo estimation with different parameter sets.
 */
static void test_tempo_variations(const onset_envelope_t *onset_env, int hop_length,
                                 const char *base_output_path) {
    // Test 1: Default parameters
    LOG("\n%s=== Test 1: Default Parameters ===%s", BRIGHT_YELLOW, RESET);
    tempo_params_t default_params = get_default_tempo_params();
    
    START_TIMING();
    tempo_result_t tempo_default = estimate_tempo(onset_env, &default_params, hop_length);
    END_TIMING("tempo_default");
    
    if (tempo_default.bpm_estimates) {
        print_tempo_stats(&tempo_default, &default_params);
        
        char output_file[512];
        snprintf(output_file, sizeof(output_file), "%s_default.txt", base_output_path);
        save_tempo_results(output_file, &tempo_default, &default_params);
    }
    
    // Test 2: No prior
    LOG("\n%s=== Test 2: No Bayesian Prior ===%s", BRIGHT_YELLOW, RESET);
    tempo_params_t no_prior_params = default_params;
    no_prior_params.use_prior = false;
    
    START_TIMING();
    tempo_result_t tempo_no_prior = estimate_tempo(onset_env, &no_prior_params, hop_length);
    END_TIMING("tempo_no_prior");
    
    if (tempo_no_prior.bpm_estimates) {
        print_tempo_stats(&tempo_no_prior, &no_prior_params);
        
        char output_file[512];
        snprintf(output_file, sizeof(output_file), "%s_no_prior.txt", base_output_path);
        save_tempo_results(output_file, &tempo_no_prior, &no_prior_params);
    }
    
    // Test 3: Different start BPM
    LOG("\n%s=== Test 3: Different Start BPM (140) ===%s", BRIGHT_YELLOW, RESET);
    tempo_params_t bpm140_params = default_params;
    bpm140_params.start_bpm = 140.0f;
    
    START_TIMING();
    tempo_result_t tempo_bpm140 = estimate_tempo(onset_env, &bpm140_params, hop_length);
    END_TIMING("tempo_bpm140");
    
    if (tempo_bpm140.bpm_estimates) {
        print_tempo_stats(&tempo_bpm140, &bpm140_params);
        
        char output_file[512];
        snprintf(output_file, sizeof(output_file), "%s_bpm140.txt", base_output_path);
        save_tempo_results(output_file, &tempo_bpm140, &bpm140_params);
    }
    
    // Test 4: Shorter autocorrelation window
    LOG("\n%s=== Test 4: Shorter AC Window (4 seconds) ===%s", BRIGHT_YELLOW, RESET);
    tempo_params_t short_ac_params = default_params;
    short_ac_params.ac_size = 4.0f;
    
    START_TIMING();
    tempo_result_t tempo_short_ac = estimate_tempo(onset_env, &short_ac_params, hop_length);
    END_TIMING("tempo_short_ac");
    
    if (tempo_short_ac.bpm_estimates) {
        print_tempo_stats(&tempo_short_ac, &short_ac_params);
        
        char output_file[512];
        snprintf(output_file, sizeof(output_file), "%s_short_ac.txt", base_output_path);
        save_tempo_results(output_file, &tempo_short_ac, &short_ac_params);
    }
    
    // Test 5: Lower max tempo
    LOG("\n%s=== Test 5: Lower Max Tempo (200 BPM) ===%s", BRIGHT_YELLOW, RESET);
    tempo_params_t low_max_params = default_params;
    low_max_params.max_tempo = 200.0f;
    
    START_TIMING();
    tempo_result_t tempo_low_max = estimate_tempo(onset_env, &low_max_params, hop_length);
    END_TIMING("tempo_low_max");
    
    if (tempo_low_max.bpm_estimates) {
        print_tempo_stats(&tempo_low_max, &low_max_params);
        
        char output_file[512];
        snprintf(output_file, sizeof(output_file), "%s_low_max.txt", base_output_path);
        save_tempo_results(output_file, &tempo_low_max, &low_max_params);
    }
    
    // Cleanup
    free_tempo_result(&tempo_default);
    free_tempo_result(&tempo_no_prior);
    free_tempo_result(&tempo_bpm140);
    free_tempo_result(&tempo_short_ac);
    free_tempo_result(&tempo_low_max);
}

int main(int argc, char *argv[]) {
    const char *input_file = "tests/files/riad.wav";
    const char *output_base = "outputs/tempo_estimation";
    
    // Allow custom input file
    if (argc > 1) {
        input_file = argv[1];
    }
    if (argc > 2) {
        output_base = argv[2];
    }
    
    LOG("%s=== Tempo Estimation Test ===%s", BRIGHT_CYAN, RESET);
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

    // Convert magnitude to power (magnitude² = power) to match librosa
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
    
    // Transpose mel_spec from (time × mels) to (mels × time) for onset_strength
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
    
    // Step 6: Test tempo estimation with various parameters
    LOG("\n%s=== Testing Tempo Estimation ===%s", BRIGHT_GREEN, RESET);
    test_tempo_variations(&onset_env, hop_size, output_base);
    
    // Step 7: Test autocorrelation data export
    LOG("\n%s=== Exporting Autocorrelation Data ===%s", BRIGHT_YELLOW, RESET);
    tempo_params_t default_params = get_default_tempo_params();
    size_t max_lag = (size_t)(default_params.ac_size * onset_env.frame_rate);
    
    START_TIMING();
    autocorr_result_t autocorr_res = compute_onset_autocorr(
        onset_env.envelope,
        onset_env.length,
        max_lag,
        onset_env.frame_rate
    );
    END_TIMING("autocorr_export");
    
    if (autocorr_res.autocorr) {
        float sample_rate = onset_env.frame_rate * hop_size;
        float *bpm_freqs = tempo_frequencies(autocorr_res.length, sample_rate, hop_size);
        
        char autocorr_file[512];
        snprintf(autocorr_file, sizeof(autocorr_file), "%s_autocorr.txt", output_base);
        save_autocorr_data(autocorr_file, &autocorr_res, bpm_freqs);
        
        free(bpm_freqs);
        free_autocorr_result(&autocorr_res);
    }
    
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
    
    LOG("\n%s✅ Tempo estimation test completed successfully!%s", BRIGHT_GREEN, RESET);
    return 0;
}
