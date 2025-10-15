/**
 * @file beat_track.h
 * @brief Beat tracking and rhythm analysis
 * 
 * This module provides functions for beat tracking using dynamic programming
 * methods. It implements algorithms compatible with librosa's beat_track()
 * function, following the Ellis (2007) dynamic programming approach.
 * 
 * The beat tracking process follows three stages:
 * 1. Onset strength computation (handled by onset.h)
 * 2. Tempo estimation (handled by tempo.h)
 * 3. Dynamic programming beat selection (this module)
 * 
 * @ingroup audio_features
 */

#ifndef BEAT_TRACK_H
#define BEAT_TRACK_H

#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "onset.h"
#include "tempo.h"

/** @addtogroup audio_features
 *  @{
 */

/**
 * @brief Beat tracking result structure
 * 
 * Contains the detected beat positions and metadata about
 * the tracking parameters and confidence.
 */
typedef struct {
    float *beat_times;       /**< Beat positions in seconds */
    size_t *beat_frames;     /**< Beat positions in frame indices */
    size_t num_beats;        /**< Number of detected beats */
    float tempo_bpm;         /**< Estimated tempo in BPM */
    float confidence;        /**< Beat tracking confidence (0-1) */
    float frame_rate;        /**< Frames per second (sr / hop_length) */
    bool *beat_mask;         /**< Boolean mask indicating beat frames (optional) */
    size_t total_frames;     /**< Total number of frames in the signal */
} beat_result_t;

/**
 * @brief Beat tracking parameters
 * 
 * Configuration structure for controlling beat tracking behavior.
 */
typedef struct {
    float tightness;         /**< Tempo adherence constraint (default: 100.0) */
    bool trim;               /**< Remove weak leading/trailing beats (default: true) */
    bool sparse;             /**< Return sparse beat indices vs dense mask (default: true) */
    tempo_params_t *tempo_params; /**< Tempo estimation parameters (can be NULL) */
} beat_params_t;

/**
 * @brief Beat tracking units for output
 */
typedef enum {
    BEAT_UNITS_FRAMES,       /**< Beat positions as frame indices */
    BEAT_UNITS_SAMPLES,      /**< Beat positions as sample indices */
    BEAT_UNITS_TIME          /**< Beat positions as time in seconds */
} beat_units_t;

/**
 * Track beats in an audio signal using dynamic programming.
 *
 * This function implements the Ellis (2007) dynamic programming beat tracker,
 * compatible with librosa's beat_track() function. It takes an onset strength
 * envelope and tempo estimate to find the most likely sequence of beat positions.
 *
 * @param onset_env Input onset strength envelope
 * @param tempo_bpm Tempo estimate in BPM (if 0, will be estimated automatically)
 * @param params Beat tracking parameters (can be NULL for defaults)
 * @param hop_length Hop length used in STFT analysis (for unit conversion)
 * @param sample_rate Audio sample rate (for unit conversion)
 * @param units Output units for beat positions
 * 
 * @return Beat tracking result structure
 * 
 * @note Caller is responsible for freeing the returned structure with free_beat_result()
 */
beat_result_t beat_track(
    const onset_envelope_t *onset_env,
    float tempo_bpm,
    const beat_params_t *params,
    int hop_length,
    float sample_rate,
    beat_units_t units
);

/**
 * Track beats from audio signal directly.
 *
 * Convenience function that computes onset strength and tempo estimation
 * internally, then performs beat tracking. This is the highest-level interface.
 *
 * @param audio Input audio data
 * @param window_size STFT window size
 * @param hop_length STFT hop length
 * @param n_mels Number of mel frequency bands
 * @param params Beat tracking parameters (can be NULL for defaults)
 * @param units Output units for beat positions
 * 
 * @return Beat tracking result structure
 * 
 * @note Caller is responsible for freeing the returned structure with free_beat_result()
 */
beat_result_t beat_track_audio(
    audio_data *audio,
    size_t window_size,
    size_t hop_length,
    size_t n_mels,
    const beat_params_t *params,
    beat_units_t units
);

/**
 * Core dynamic programming beat tracker.
 *
 * Implements the Ellis (2007) dynamic programming algorithm for beat tracking.
 * This is the core algorithm that finds the optimal sequence of beat positions
 * given an onset strength envelope and tempo estimate.
 *
 * @param onset_env Normalized onset strength envelope
 * @param tempo_bpm Tempo estimate in BPM
 * @param frame_rate Frames per second
 * @param tightness Tempo adherence constraint
 * @param trim Remove weak leading/trailing beats
 * 
 * @return Boolean mask indicating beat positions (caller must free)
 */
bool *dp_beat_tracker(
    const float *onset_env,
    size_t num_frames,
    float tempo_bpm,
    float frame_rate,
    float tightness,
    bool trim
);

/**
 * Normalize onset strength envelope by standard deviation.
 *
 * Applies normalization: onset_norm = onset / std(onset)
 * This preprocessing step is essential for the DP algorithm.
 *
 * @param onset_env Input onset strength values
 * @param normalized Output normalized values (pre-allocated)
 * @param length Number of frames
 */
void normalize_onsets(
    const float *onset_env,
    float *normalized,
    size_t length
);

/**
 * Compute local score using Gaussian-weighted smoothing.
 *
 * Applies a Gaussian kernel centered on the expected beat period to smooth
 * the onset envelope. This creates a tempo-aware local score that emphasizes
 * onsets occurring at regular intervals.
 *
 * @param normalized_onsets Normalized onset strength values
 * @param local_score Output local score values (pre-allocated)
 * @param length Number of frames
 * @param frames_per_beat Expected frames between beats
 */
void compute_local_score(
    const float *normalized_onsets,
    float *local_score,
    size_t length,
    float frames_per_beat
);

/**
 * Core dynamic programming algorithm for beat tracking.
 *
 * Finds the optimal sequence of beat positions using dynamic programming
 * with a tempo tightness constraint. This implements the core DP loop
 * from Ellis (2007).
 *
 * @param local_score Local score values from Gaussian smoothing
 * @param backlink Output backlink array for path reconstruction (pre-allocated)
 * @param cumscore Output cumulative score array (pre-allocated)
 * @param length Number of frames
 * @param frames_per_beat Expected frames between beats
 * @param tightness Tempo adherence constraint
 */
void beat_track_dp(
    const float *local_score,
    int *backlink,
    float *cumscore,
    size_t length,
    float frames_per_beat,
    float tightness
);

/**
 * Reconstruct beat path from backlinks.
 *
 * Uses the backlink array from the DP algorithm to reconstruct the
 * optimal sequence of beat positions by backtracking from the end.
 *
 * @param backlink Backlink array from DP algorithm
 * @param cumscore Cumulative score array from DP algorithm
 * @param beats Output boolean mask for beat positions (pre-allocated)
 * @param length Number of frames
 */
void dp_backtrack(
    const int *backlink,
    const float *cumscore,
    bool *beats,
    size_t length
);

/**
 * Remove weak leading and trailing beats.
 *
 * Trims beats at the beginning and end of the signal that have weak
 * onset strength, reducing false positive detections.
 *
 * @param local_score Local score values
 * @param beats Beat mask (modified in-place)
 * @param length Number of frames
 * @param trim Enable trimming (if false, no trimming is performed)
 */
void trim_beats(
    const float *local_score,
    bool *beats,
    size_t length,
    bool trim
);

/**
 * Convert beat mask to sparse beat indices.
 *
 * Extracts the frame indices where beats occur from a boolean mask.
 *
 * @param beat_mask Boolean mask indicating beat positions
 * @param length Number of frames
 * @param beat_indices Output array for beat indices (caller must allocate)
 * 
 * @return Number of beats found
 */
size_t beats_to_indices(
    const bool *beat_mask,
    size_t length,
    size_t *beat_indices
);

/**
 * Convert frame indices to time or sample positions.
 *
 * Converts beat positions from frame indices to the requested units.
 *
 * @param frame_indices Input frame indices
 * @param output Output positions in requested units (pre-allocated)
 * @param num_beats Number of beats
 * @param hop_length STFT hop length
 * @param sample_rate Audio sample rate
 * @param units Target units for conversion
 */
void convert_beat_units(
    const size_t *frame_indices,
    float *output,
    size_t num_beats,
    int hop_length,
    float sample_rate,
    beat_units_t units
);

/**
 * Get default beat tracking parameters.
 *
 * Returns a parameter structure with sensible defaults matching
 * librosa's beat tracking behavior.
 *
 * @return Default beat tracking parameters
 */
beat_params_t get_default_beat_params(void);

/**
 * Free memory allocated for a beat result structure.
 *
 * @param result Pointer to beat result structure to free
 */
void free_beat_result(beat_result_t *result);

/** @} */  // end of audio_features

#endif // BEAT_TRACK_H
