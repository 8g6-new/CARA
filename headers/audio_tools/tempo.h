/**
 * @file tempo.h
 * @brief Tempo estimation and beat tracking utilities
 * 
 * This module provides functions for estimating tempo (BPM) from onset strength
 * envelopes using autocorrelation-based methods. It implements algorithms
 * compatible with librosa's tempo estimation approach.
 * 
 * @ingroup audio_features
 */

#ifndef TEMPO_H
#define TEMPO_H

#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include "onset.h"

/** @addtogroup audio_features
 *  @{
 */

/**
 * @brief Tempo estimation result structure
 * 
 * Contains the computed tempo estimates and metadata about
 * the estimation parameters and confidence.
 */
typedef struct {
    float *bpm_estimates;     /**< Tempo estimates in BPM (one per frame or global) */
    size_t length;           /**< Number of tempo estimates */
    float frame_rate;        /**< Frames per second (sr / hop_length) */
    bool is_global;          /**< True if single global estimate, false for frame-wise */
    float confidence;        /**< Confidence score of the estimation (0-1) */
} tempo_result_t;

/**
 * @brief Tempo estimation parameters
 * 
 * Configuration structure for controlling tempo estimation behavior.
 */
typedef struct {
    float start_bpm;         /**< Initial BPM guess for prior (default: 120.0) */
    float std_bpm;           /**< Standard deviation for log-normal prior (default: 1.0) */
    float max_tempo;         /**< Maximum allowed tempo in BPM (default: 320.0) */
    float ac_size;           /**< Autocorrelation window size in seconds (default: 8.0) */
    bool use_prior;          /**< Enable Bayesian log-normal prior (default: true) */
    bool aggregate;          /**< Global estimation vs frame-wise (default: true) */
} tempo_params_t;

/**
 * @brief Autocorrelation result structure
 * 
 * Contains autocorrelation data and associated frequency mappings.
 */
typedef struct {
    float *autocorr;         /**< Autocorrelation values */
    float *bpm_freqs;        /**< Corresponding BPM frequencies */
    size_t length;           /**< Number of autocorrelation lags */
    float max_lag_seconds;   /**< Maximum lag in seconds */
} autocorr_result_t;

/**
 * Estimate tempo from an onset strength envelope.
 *
 * This function implements autocorrelation-based tempo estimation similar to
 * librosa's tempo() function. It computes the autocorrelation of the onset
 * envelope, applies optional Bayesian priors, and finds the most likely tempo.
 *
 * @param onset_env Input onset strength envelope
 * @param params Tempo estimation parameters (can be NULL for defaults)
 * @param hop_length Hop length used in STFT analysis (for BPM conversion)
 * 
 * @return Tempo estimation result structure
 * 
 * @note Caller is responsible for freeing the returned structure with free_tempo_result()
 */
tempo_result_t estimate_tempo(
    const onset_envelope_t *onset_env,
    const tempo_params_t *params,
    int hop_length
);

/**
 * Compute autocorrelation of onset strength envelope.
 *
 * Uses FFT-based autocorrelation for efficiency. The autocorrelation is
 * computed up to a maximum lag determined by the ac_size parameter.
 *
 * @param onset_env Input onset strength values
 * @param length Number of onset frames
 * @param max_lag Maximum autocorrelation lag (in frames)
 * @param frame_rate Frames per second (for metadata)
 * 
 * @return Autocorrelation result structure
 * 
 * @note Caller is responsible for freeing the returned structure with free_autocorr_result()
 */
autocorr_result_t compute_onset_autocorr(
    const float *onset_env,
    size_t length,
    size_t max_lag,
    float frame_rate
);

/**
 * Convert autocorrelation lag indices to BPM frequencies.
 *
 * Computes the BPM values corresponding to each autocorrelation lag using:
 * bpm = 60.0 * sample_rate / (hop_length * lag_samples)
 *
 * @param n_lags Number of autocorrelation lags
 * @param sample_rate Audio sample rate
 * @param hop_length STFT hop length in samples
 * 
 * @return Array of BPM frequencies (caller must free)
 */
float *tempo_frequencies(
    size_t n_lags,
    float sample_rate,
    int hop_length
);

/**
 * Apply log-normal prior to autocorrelation values.
 *
 * Weights the autocorrelation by a log-normal distribution centered at
 * start_bpm with the specified standard deviation. This implements
 * Bayesian tempo estimation.
 *
 * @param autocorr Autocorrelation values (modified in-place)
 * @param bpm_freqs Corresponding BPM frequencies
 * @param length Number of values
 * @param start_bpm Center of log-normal prior
 * @param std_bpm Standard deviation of prior
 */
void apply_log_normal_prior(
    float *autocorr,
    const float *bpm_freqs,
    size_t length,
    float start_bpm,
    float std_bpm
);

/**
 * Find the peak in weighted autocorrelation.
 *
 * Locates the maximum value in the autocorrelation, respecting the
 * maximum tempo constraint.
 *
 * @param autocorr Autocorrelation values
 * @param bpm_freqs Corresponding BPM frequencies
 * @param length Number of values
 * @param max_tempo Maximum allowed tempo (BPM)
 * @param confidence Output confidence score (can be NULL)
 * 
 * @return Index of the peak (best tempo estimate)
 */
size_t find_tempo_peak(
    const float *autocorr,
    const float *bpm_freqs,
    size_t length,
    float max_tempo,
    float *confidence
);

/**
 * Get default tempo estimation parameters.
 *
 * Returns a parameter structure with sensible defaults matching
 * librosa's tempo estimation behavior.
 *
 * @return Default tempo parameters
 */
tempo_params_t get_default_tempo_params(void);

/**
 * Free memory allocated for a tempo result structure.
 *
 * @param result Pointer to tempo result structure to free
 */
void free_tempo_result(tempo_result_t *result);

/**
 * Free memory allocated for an autocorrelation result structure.
 *
 * @param result Pointer to autocorrelation result structure to free
 */
void free_autocorr_result(autocorr_result_t *result);

/** @} */  // end of audio_features

#endif // TEMPO_H
