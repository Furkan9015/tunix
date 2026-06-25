#ifndef DTW_H
#define DTW_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct position_pair {
    size_t i; /* position in the reference */
    size_t j; /* position in the read */
} position_pair;

typedef struct alignment_element {
    position_pair position;
    float difference;
} alignment_element;

typedef struct dtw_result {
    float cost;
    alignment_element *alignment; /* caller must free() this */
    size_t alignment_length;
} dtw_result;

/* Cost-only DTW variants */
float DTW_global(const float* a_values, uint32_t a_length, const float* b_values, uint32_t b_length, bool exclude_last_element);
float DTW_global_slow(const float* a_values, uint32_t a_length, const float* b_values, uint32_t b_length, bool exclude_last_element);
float DTW_global_diagonalbanded(const float* a_values, uint32_t a_length, const float* b_values, uint32_t b_length, int band_radius, bool exclude_last_element);
float DTW_global_slantedbanded(const float* a_values, uint32_t a_length, const float* b_values, uint32_t b_length, int band_radius, bool exclude_last_element);
float DTW_global_slantedbanded_antidiagonalwise(const float* a_values, uint32_t a_length, const float* b_values, uint32_t b_length, int band_radius, bool exclude_last_element);
float DTW_semiglobal(const float* a_values, uint32_t a_length, const float* b_values, uint32_t b_length, bool exclude_last_element);
float DTW_semiglobal_slow(const float* a_values, uint32_t a_length, const float* b_values, uint32_t b_length, bool exclude_last_element);

/* Traceback DTW variants - caller must free(result.alignment) */
dtw_result DTW_global_tb(const float* a_values, uint32_t a_length, const float* b_values, uint32_t b_length, bool exclude_last_element);
dtw_result DTW_semiglobal_tb(const float* a_values, uint32_t a_length, const float* b_values, uint32_t b_length, bool exclude_last_element);

#ifdef __cplusplus
}
#endif

#endif /* DTW_H */
