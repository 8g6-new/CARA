#include "audio_tools/tempo.h"
#include "utils/bench.h"

/*
 * The MIT License (MIT)
 * 
 * Copyright © 2025 Devadut S Balan
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * Compute the smallest power of two greater than or equal to n.
 * @param n Input value to round up.
 * @return The smallest power-of-two value >= n. When n is 0, returns 1.
 */
static size_t next_power_of_2(size_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    if (sizeof(size_t) > 4) {
        n |= n >> 32;
    }
    return n + 1;
}

/**
 * Produce a tempo_params_t populated with sensible default values for tempo estimation.
 *
 * @returns A tempo_params_t initialized as:
 *          - start_bpm = 120.0f
 *          - std_bpm = 1.0f
 *          - max_tempo = 320.0f
 *          - ac_size = 8.0f
 *          - use_prior = true
 *          - aggregate = true
 */
tempo_params_t get_default_tempo_params(void) {
    tempo_params_t params = {
        .start_bpm = 120.0f,
        .std_bpm = 1.0f,
        .max_tempo = 320.0f,
        .ac_size = 8.0f,
        .use_prior = true,
        .aggregate = true
    };
    return params;
}

/**
 * Compute BPM values corresponding to autocorrelation lag indices.
 *
 * Produces an array where each element i gives the tempo in beats per minute
 * represented by autocorrelation lag i (index 0 is set to INFINITY).
 *
 * @param n_lags Number of lag values to compute (length of the returned array).
 * @param sample_rate Audio sample rate in Hz.
 * @param hop_length Number of audio samples between consecutive onset-envelope frames.
 * @returns Pointer to a newly allocated array of `n_lags` floats (caller must free)
 *          where `result[0] == INFINITY` and for `i >= 1`:
 *          `result[i] == 60.0 * sample_rate / (hop_length * i)`. Returns `NULL`
 *          on invalid input or allocation failure.
 */
float *tempo_frequencies(size_t n_lags, float sample_rate, int hop_length) {
    if (n_lags == 0 || hop_length <= 0 || sample_rate <= 0) {
        ERROR("Invalid parameters for tempo_frequencies");
        return NULL;
    }
    
    float *bpm_freqs = (float *)malloc(n_lags * sizeof(float));
    if (!bpm_freqs) {
        ERROR("Failed to allocate BPM frequencies array");
        return NULL;
    }
    
    // Convert lag indices to BPM values
    // bpm = 60.0 * sample_rate / (hop_length * lag_samples)
    // Skip lag 0 (infinite BPM) by starting from lag 1
    bpm_freqs[0] = INFINITY; // lag 0 corresponds to infinite BPM
    
    for (size_t i = 1; i < n_lags; i++) {
        bpm_freqs[i] = 60.0f * sample_rate / (hop_length * (float)i);
    }
    
    return bpm_freqs;
}

/**
 * Compute the onset autocorrelation for an onset envelope using FFT-based convolution.
 *
 * Produces a truncated, lag-normalized autocorrelation sequence for lags [0, max_lag-1].
 *
 * @param onset_env Pointer to the onset envelope array (length samples).
 * @param length Number of samples in `onset_env`.
 * @param max_lag Maximum lag (in frames) to compute; will be clamped to `length`.
 * @param frame_rate Frame rate of the onset envelope (frames per second) used to compute max_lag_seconds.
 *
 * @returns An autocorr_result_t with:
 *   - `autocorr`: heap-allocated array of length `result.length` containing autocorrelation values normalized by lag 0 (values in [0,1]) or NULL on failure.
 *   - `bpm_freqs`: always set to NULL (BPM frequencies are not computed here).
 *   - `length`: actual number of lags returned (<= requested `max_lag`).
 *   - `max_lag_seconds`: `length / frame_rate`.
 *
 * On invalid input or allocation/FFT failure, returns a zero-initialized result (all pointers NULL, length zero).
 */
autocorr_result_t compute_onset_autocorr(
    const float *onset_env,
    size_t length,
    size_t max_lag,
    float frame_rate
) {
    autocorr_result_t result = {0};
    
    if (!onset_env || length == 0 || max_lag == 0) {
        ERROR("Invalid parameters for autocorrelation computation");
        return result;
    }
    
    // Limit max_lag to signal length
    if (max_lag > length) {
        max_lag = length;
    }
    
    // Pad signal to next power of 2 for efficient FFT
    size_t padded_size = next_power_of_2(2 * length - 1);
    
    // Allocate padded input signal
    float *padded_signal = (float *)calloc(padded_size, sizeof(float));
    if (!padded_signal) {
        ERROR("Failed to allocate padded signal for autocorrelation");
        return result;
    }
    
    // Copy input signal to padded buffer
    memcpy(padded_signal, onset_env, length * sizeof(float));
    
    // Create FFTW plans
    fftwf_complex *fft_result = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * (padded_size / 2 + 1));
    float *autocorr_full = (float *)fftwf_malloc(sizeof(float) * padded_size);
    
    if (!fft_result || !autocorr_full) {
        ERROR("Failed to allocate FFT buffers for autocorrelation");
        free(padded_signal);
        if (fft_result) fftwf_free(fft_result);
        if (autocorr_full) fftwf_free(autocorr_full);
        return result;
    }
    
    fftwf_plan plan_forward = fftwf_plan_dft_r2c_1d(padded_size, padded_signal, fft_result, FFTW_ESTIMATE);
    fftwf_plan plan_backward = fftwf_plan_dft_c2r_1d(padded_size, fft_result, autocorr_full, FFTW_ESTIMATE);
    
    if (!plan_forward || !plan_backward) {
        ERROR("Failed to create FFTW plans for autocorrelation");
        free(padded_signal);
        fftwf_free(fft_result);
        fftwf_free(autocorr_full);
        return result;
    }
    
    // Execute forward FFT
    fftwf_execute(plan_forward);
    
    // Compute power spectrum (|FFT|²)
    for (size_t i = 0; i < padded_size / 2 + 1; i++) {
        float real = crealf(fft_result[i]);
        float imag = cimagf(fft_result[i]);
        fft_result[i] = real * real + imag * imag + 0.0f * I;
    }
    
    // Execute inverse FFT to get autocorrelation
    fftwf_execute(plan_backward);
    
    // Normalize by padded_size (FFTW doesn't normalize IFFT)
    for (size_t i = 0; i < padded_size; i++) {
        autocorr_full[i] /= (float)padded_size;
    }
    
    // Allocate result arrays
    result.autocorr = (float *)malloc(max_lag * sizeof(float));
    result.bpm_freqs = (float *)malloc(max_lag * sizeof(float));
    
    if (!result.autocorr || !result.bpm_freqs) {
        ERROR("Failed to allocate autocorrelation result arrays");
        free(result.autocorr);
        free(result.bpm_freqs);
        result.autocorr = NULL;
        result.bpm_freqs = NULL;
        goto cleanup;
    }
    
    // Copy truncated autocorrelation and normalize
    float max_val = autocorr_full[0]; // lag 0 should be maximum
    if (max_val > 0) {
        for (size_t i = 0; i < max_lag; i++) {
            result.autocorr[i] = autocorr_full[i] / max_val;
        }
    } else {
        // Handle edge case where signal is all zeros
        for (size_t i = 0; i < max_lag; i++) {
            result.autocorr[i] = 0.0f;
        }
    }
    
    // Set metadata
    result.length = max_lag;
    result.max_lag_seconds = (float)max_lag / frame_rate;
    
    // BPM frequencies will be computed by caller using tempo_frequencies()
    // For now, set to NULL to indicate they need to be computed
    free(result.bpm_freqs);
    result.bpm_freqs = NULL;
    
cleanup:
    // Cleanup
    fftwf_destroy_plan(plan_forward);
    fftwf_destroy_plan(plan_backward);
    fftwf_free(fft_result);
    fftwf_free(autocorr_full);
    free(padded_signal);
    
    return result;
}

/**
 * Apply a log-normal tempo prior to an autocorrelation array in-place.
 *
 * For each lag with a finite, positive BPM in `bpm_freqs`, the function adds
 * a log-normal prior (centered at `start_bpm` with scale `std_bpm`) to the
 * autocorrelation value in log-space. Entries with non-finite or non-positive
 * BPMs are set to `-INFINITY`. If parameters are invalid the function returns
 * without modifying `autocorr`.
 *
 * @param autocorr In/out array of length `length` containing autocorrelation
 *                 values expressed in linear (pre-log) form; values are updated
 *                 in-place in log-space with the prior applied.
 * @param bpm_freqs Array of length `length` containing BPM values corresponding
 *                  to each autocorrelation lag; must be finite and > 0 to be
 *                  considered valid.
 * @param length Number of elements in `autocorr` and `bpm_freqs`.
 * @param start_bpm Center BPM of the log-normal prior; must be > 0.
 * @param std_bpm Standard deviation of the prior in log2 space; must be > 0.
 */
void apply_log_normal_prior(
    float *autocorr,
    const float *bpm_freqs,
    size_t length,
    float start_bpm,
    float std_bpm
) {
    if (!autocorr || !bpm_freqs || length == 0) {
        ERROR("Invalid parameters for log-normal prior application");
        return;
    }
    
    if (start_bpm <= 0 || std_bpm <= 0) {
        WARN("Invalid prior parameters, skipping prior application");
        return;
    }
    
    // Apply log-normal prior: -0.5 * ((log2(bpm) - log2(start_bpm)) / std_bmp)²
    float log2_start = log2f(start_bpm);
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < length; i++) {
        if (bpm_freqs[i] > 0 && isfinite(bpm_freqs[i])) {
            float log2_bpm = log2f(bpm_freqs[i]);
            float log_diff = (log2_bpm - log2_start) / std_bpm;
            float log_prior = -0.5f * log_diff * log_diff;
            
            // Apply prior in log space, then convert back
            // Using log1p for numerical stability as in librosa
            autocorr[i] = log1pf(1e6f * autocorr[i]) + log_prior;
        } else {
            // Invalid BPM, set to very low probability
            autocorr[i] = -INFINITY;
        }
    }
}

/**
 * Locate the best tempo peak index within autocorrelation values subject to a tempo limit.
 *
 * Scans autocorr (skipping lag 0) and considers only entries whose corresponding bpm_freqs
 * are less than or equal to max_tempo and have finite autocorrelation; optionally writes
 * a confidence score in [0, 1] that reflects separation between the best and second-best peaks.
 *
 * @param autocorr Array of autocorrelation values indexed by lag.
 * @param bpm_freqs Array of BPM frequencies corresponding to each lag (same length as autocorr).
 * @param length Number of elements in `autocorr` and `bpm_freqs`.
 * @param max_tempo Maximum BPM to consider when selecting a peak.
 * @param confidence Output pointer where a confidence value in [0, 1] will be stored; may be NULL.
 * @returns Index of the selected lag with the highest valid autocorrelation within `max_tempo`;
 *          returns 0 if no valid peak is found or on error.
 */
size_t find_tempo_peak(
    const float *autocorr,
    const float *bpm_freqs,
    size_t length,
    float max_tempo,
    float *confidence
) {
    if (!autocorr || !bpm_freqs || length == 0) {
        ERROR("Invalid parameters for tempo peak finding");
        return 0;
    }
    
    size_t best_idx = 0;
    float best_val = -INFINITY;
    float second_best = -INFINITY;
    
    // Find the maximum value, respecting max_tempo constraint
    for (size_t i = 1; i < length; i++) { // Skip lag 0 (infinite BPM)
        if (bpm_freqs[i] <= max_tempo && isfinite(autocorr[i])) {
            if (autocorr[i] > best_val) {
                second_best = best_val;
                best_val = autocorr[i];
                best_idx = i;
            } else if (autocorr[i] > second_best) {
                second_best = autocorr[i];
            }
        }
    }
    
    // Compute confidence as ratio of best to second-best peak
    if (confidence) {
        if (second_best > -INFINITY && best_val > second_best) {
            *confidence = (best_val - second_best) / (fabsf(best_val) + fabsf(second_best) + 1e-8f);
            *confidence = fminf(1.0f, fmaxf(0.0f, *confidence)); // Clamp to [0, 1]
        } else {
            *confidence = 0.0f;
        }
    }
    
    return best_idx;
}

/**
 * Estimate the dominant tempo (beats per minute) from an onset envelope.
 *
 * Computes an autocorrelation of the provided onset envelope, derives BPM
 * candidates for each lag, optionally applies a log-normal prior, and selects
 * the best tempo peak with an associated confidence score.
 *
 * @param onset_env Pointer to an onset_envelope_t containing the onset strength
 *        array, its length (number of frames), and the frame_rate (frames per second).
 * @param params Optional pointer to tempo_params_t controlling estimation behavior;
 *        if NULL, default tempo parameters are used.
 * @param hop_length Number of audio samples between successive onset frames (must be > 0).
 *
 * @returns tempo_result_t populated with:
 *          - `bpm_estimates`: pointer to an array of detected BPM values (length 1 on success, NULL on failure),
 *          - `length`: number of BPM estimates (0 on failure),
 *          - `frame_rate`: copied from `onset_env->frame_rate`,
 *          - `is_global`: set from `params->aggregate`,
 *          - `confidence`: confidence of the selected tempo in [0, 1].
 */
tempo_result_t estimate_tempo(
    const onset_envelope_t *onset_env,
    const tempo_params_t *params,
    int hop_length
) {
    tempo_result_t result = {0};
    
    if (!onset_env || !onset_env->envelope || onset_env->length == 0) {
        ERROR("Invalid onset envelope for tempo estimation");
        return result;
    }
    
    if (hop_length <= 0) {
        ERROR("Invalid hop_length for tempo estimation");
        return result;
    }
    
    // Use default parameters if none provided
    tempo_params_t default_params = get_default_tempo_params();
    if (!params) {
        params = &default_params;
    }
    
    // Calculate maximum lag from ac_size parameter
    size_t max_lag = (size_t)(params->ac_size * onset_env->frame_rate);
    if (max_lag > onset_env->length) {
        max_lag = onset_env->length;
    }
    if (max_lag < 2) {
        ERROR("Autocorrelation window too small for tempo estimation");
        return result;
    }
    
    LOG("Computing tempo with max_lag=%zu frames (%.2f seconds)", 
        max_lag, (float)max_lag / onset_env->frame_rate);
    
    // Compute autocorrelation
    START_TIMING();
    autocorr_result_t autocorr_res = compute_onset_autocorr(
        onset_env->envelope,
        onset_env->length,
        max_lag,
        onset_env->frame_rate
    );
    END_TIMING("autocorrelation");
    
    if (!autocorr_res.autocorr) {
        ERROR("Failed to compute autocorrelation for tempo estimation");
        return result;
    }
    
    // Compute BPM frequencies
    START_TIMING();
    float sample_rate = onset_env->frame_rate * hop_length;
    float *bpm_freqs = tempo_frequencies(autocorr_res.length, sample_rate, hop_length);
    END_TIMING("bpm_frequencies");
    
    if (!bpm_freqs) {
        ERROR("Failed to compute BPM frequencies");
        free_autocorr_result(&autocorr_res);
        return result;
    }
    
    // Apply prior if requested
    if (params->use_prior) {
        START_TIMING();
        apply_log_normal_prior(
            autocorr_res.autocorr,
            bpm_freqs,
            autocorr_res.length,
            params->start_bpm,
            params->std_bpm
        );
        END_TIMING("prior_application");
    }
    
    // Find tempo peak
    START_TIMING();
    float confidence = 0.0f;
    size_t best_idx = find_tempo_peak(
        autocorr_res.autocorr,
        bpm_freqs,
        autocorr_res.length,
        params->max_tempo,
        &confidence
    );
    END_TIMING("peak_finding");

    // If no peak is found, best_idx will be 0, which corresponds to infinite BPM.
    // Return an empty result to avoid downstream errors.
    if (best_idx == 0) {
        WARN("No reliable tempo peak found.");
        free(bpm_freqs);
        free_autocorr_result(&autocorr_res);
        return result;
    }
    
    // Allocate result
    result.bpm_estimates = (float *)malloc(sizeof(float));
    if (!result.bpm_estimates) {
        ERROR("Failed to allocate tempo result");
        free(bpm_freqs);
        free_autocorr_result(&autocorr_res);
        return result;
    }
    
    // Set result values
    result.bpm_estimates[0] = bpm_freqs[best_idx];
    result.length = 1;
    result.frame_rate = onset_env->frame_rate;
    result.is_global = params->aggregate;
    result.confidence = confidence;
    
    LOG("Estimated tempo: %.2f BPM (confidence: %.3f)", 
        result.bpm_estimates[0], confidence);
    
    // Cleanup
    free(bpm_freqs);
    free_autocorr_result(&autocorr_res);
    
    return result;
}

/**
 * Release resources held by a tempo_result_t and reset its fields to defaults.
 *
 * Frees the internal `bpm_estimates` array if present and sets `bpm_estimates` to NULL,
 * `length` to 0, `frame_rate` to 0.0f, `is_global` to false, and `confidence` to 0.0f.
 *
 * @param result Pointer to the tempo_result_t to clear; no action is taken if NULL.
 */
void free_tempo_result(tempo_result_t *result) {
    if (result) {
        if (result->bpm_estimates) {
            free(result->bpm_estimates);
            result->bpm_estimates = NULL;
        }
        result->length = 0;
        result->frame_rate = 0.0f;
        result->is_global = false;
        result->confidence = 0.0f;
    }
}

/**
 * Deallocate and reset the contents of an autocorr_result_t in place.
 *
 * Frees the internal `autocorr` and `bpm_freqs` arrays if they are non-NULL,
 * sets those pointers to NULL, and resets `length` to 0 and
 * `max_lag_seconds` to 0.0f. Does nothing if `result` is NULL.
 *
 * @param result Pointer to the autocorr_result_t to clear.
 */
void free_autocorr_result(autocorr_result_t *result) {
    if (result) {
        if (result->autocorr) {
            free(result->autocorr);
            result->autocorr = NULL;
        }
        if (result->bpm_freqs) {
            free(result->bpm_freqs);
            result->bpm_freqs = NULL;
        }
        result->length = 0;
        result->max_lag_seconds = 0.0f;
    }
}