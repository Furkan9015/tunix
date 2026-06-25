#ifndef REVENT_H
#define REVENT_H

#include <stdint.h>
#include "roptions.h"

#ifdef __cplusplus
extern "C" {
#endif

float* normalize_signal(void *km,
						const float* sig,
						const uint32_t s_len,
						double* mean_sum,
						double* std_dev_sum,
						uint32_t* n_events_sum,
						uint32_t* n_sig);
/**
 * Detects events from signals
 *
 * @param km	thread-local memory pool; using NULL falls back to malloc()
 * @param s_len	length of $sig
 * @param sig	signal values
 * @param opt	mapping options @TODO: Should be decoupled from the mapping options
 * @param n		number of events
 * 
 * @return		list of event values of length $n
 */
float* detect_events(void *km,
					 const uint32_t s_len,
					 const float* sig,
					 const uint32_t window_length1,
					 const uint32_t window_length2,
					 const float threshold1,
					 const float threshold2,
					 const float peak_height,
					 const uint32_t min_seg_len,
					 const uint32_t max_seg_len,
					 double* mean_sum,
					 double* std_dev_sum,
					 uint32_t* n_events_sum,
					 uint32_t *n_events);

/**
 * Generate events using externally-provided peak positions.
 * Normalizes the signal, remaps raw-signal peak indices to normalized-signal
 * positions, then calls gen_events() on the normalized signal.
 *
 * @param km            thread-local memory pool
 * @param s_len         length of raw signal
 * @param sig           raw signal values (pA, before normalization)
 * @param ext_peaks     array of peak positions in raw signal space
 * @param n_ext_peaks   number of external peaks
 * @param min_seg_len   minimum segment length (skip shorter segments)
 * @param max_seg_len   maximum segment length (skip longer segments)
 * @param mean_sum      running mean accumulator (updated)
 * @param std_dev_sum   running variance accumulator (updated)
 * @param n_events_sum  running sample count (updated)
 * @param n_events      output: number of events generated
 *
 * @return  array of event values, or NULL if no valid events
 */
float* detect_events_with_ext_peaks(void *km,
					 const uint32_t s_len,
					 const float* sig,
					 const uint32_t *ext_peaks,
					 const uint32_t n_ext_peaks,
					 const uint32_t min_seg_len,
					 const uint32_t max_seg_len,
					 double* mean_sum,
					 double* std_dev_sum,
					 uint32_t* n_events_sum,
					 uint32_t *n_events);

#ifdef __cplusplus
}
#endif
#endif //REVENT_H