#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#ifdef HAVE_IMMINTRIN_H
#include <immintrin.h>
#endif
#include "dtw.h"

/* C99 'restrict' is not a keyword in C++; use compiler extension instead */
#ifdef __cplusplus
#define RESTRICT __restrict__
#else
#define RESTRICT restrict
#endif

#define DISTANCE(A, B) fabsf((A)-(B))
#if defined(__AVX512F__) && defined(HAVE_IMMINTRIN_H)
#define DISTANCE16(A, B) _mm512_abs_ps(_mm512_sub_ps((A), (B)))
#endif

/* check the maximum available float SIMD width */
#if defined(__AVX512F__)
	#define F32SIMD_WIDTH 16
#elif defined(__AVX__)
	#define F32SIMD_WIDTH 8
#elif defined(__SSE__)
	#define F32SIMD_WIDTH 4
#else
	#define F32SIMD_WIDTH 1
#endif

#define DTW_MIN(a, b) ((a) < (b) ? (a) : (b))
#define DTW_MAX(a, b) ((a) > (b) ? (a) : (b))

/* #define DEBUG */

#define PRINTDP {								\
	uint32_t _k;								\
	for(_k = 0; _k < a_length; _k++){			\
		fprintf(stdout, "%.2f ", dp[_k]);		\
	}											\
	fprintf(stdout, "\n");						\
}

float DTW_global(const float* a_values, uint32_t a_length, const float* b_values, uint32_t b_length, bool exclude_last_element){
	uint32_t i, j;
	float *dp = (float *)malloc((size_t)a_length * sizeof(float));
	float result;

	dp[0] = DISTANCE(a_values[0], b_values[0]);
	for(j = 1; j < a_length; j++){
		dp[j] = dp[j-1] + DISTANCE(a_values[j], b_values[0]);
	}

	for(i = 1; i < b_length; i++){
		float old_left = dp[0];
		dp[0] = dp[0]+DISTANCE(a_values[0], b_values[i]);
		for(j = 1; j < a_length; j++){
			float top = dp[j-1];
			float left = dp[j];
			float topleft = old_left;
			float center = DTW_MIN(
							DTW_MIN(top, left),
							topleft
						) + DISTANCE(a_values[j], b_values[i]);
			dp[j] = center;
			old_left = left;
		}
	}

	if(exclude_last_element){
		result = dp[a_length-1] - DISTANCE(a_values[a_length-1], b_values[b_length-1]);
	}
	else{
		result = dp[a_length-1];
	}

	free(dp);
	return result;
}

/* 512 bit SIMD vector print macro */
#define PRINT512(VEC) {							\
	int _k;									\
	fprintf(stdout, #VEC ": ");				\
	for(_k = 0; _k < 16; _k++){				\
		fprintf(stdout, "%f ", (VEC)[_k]);	\
	}											\
	fprintf(stdout, "\n");					\
}


float DTW_global_slow(const float* a_values, uint32_t a_length, const float* b_values, uint32_t b_length, bool exclude_last_element){
	uint32_t i, j;
	float result;
	float *dp;

	assert(a_length > 0 && b_length > 0);

	dp = (float *)malloc((size_t)a_length * b_length * sizeof(float));
	#define DP(i, j) dp[(size_t)(i) * b_length + (j)]

	DP(0, 0) = DISTANCE(a_values[0], b_values[0]);

	for(i = 1; i < a_length; i++){
		DP(i, 0) = DP(i-1, 0) + DISTANCE(a_values[i], b_values[0]);
	}
	for(j = 1; j < b_length; j++){
		DP(0, j) = DP(0, j-1) + DISTANCE(a_values[0], b_values[j]);
	}

	for(i = 1; i < a_length; i++){
		for(j = 1; j < b_length; j++){
			float best_in = DTW_MIN(DTW_MIN(DP(i-1, j), DP(i, j-1)), DP(i-1, j-1));
			DP(i, j) = best_in + DISTANCE(a_values[i], b_values[j]);
		}
	}

	if(exclude_last_element){
		result = DP(a_length-1, b_length-1) - DISTANCE(a_values[a_length-1], b_values[b_length-1]);
	}
	else{
		result = DP(a_length-1, b_length-1);
	}

	#undef DP
	free(dp);
	return result;
}

float DTW_global_diagonalbanded(const float* a_values, uint32_t a_length, const float* b_values, uint32_t b_length, int band_radius, bool exclude_last_element){
	int i;
	int row_offset;
	float *dp;
	int band_width;
	int dp_center;
	float prev;
	float result;

	assert(a_length > 0 && b_length > 0);
	assert(a_length < (uint32_t)INT_MAX);
	assert(b_length < (uint32_t)INT_MAX);
	assert(band_radius >= 0);

	band_width = band_radius*2+1;
	dp = (float *)malloc((size_t)band_width * sizeof(float));
	for(i = 0; i < band_width; i++) dp[i] = 1e10f;
	dp_center = band_radius;

	prev = 0.0f;
	for(row_offset = 0; row_offset <= DTW_MIN(band_radius, (int)b_length-1); row_offset++){
		int j = row_offset;
		float cur = prev + DISTANCE(a_values[0], b_values[j]);
		dp[dp_center+row_offset] = cur;
		prev = cur;
	}

	for(i = 1; i < (int)a_length; i++){
		const int center_row = i;
		const int row_offset_start = DTW_MAX(-(int)band_radius, -center_row);
		const int row_offset_end = DTW_MIN(band_radius, (int)b_length - center_row - 1);

		float top = 1e10f;
		for(row_offset = row_offset_start; row_offset <= row_offset_end; row_offset++){
			int j = center_row + row_offset;
			float topleft = dp[dp_center+row_offset]; /*for the first few iterations this might go oob, make sure to init to 1e10*/
			float left = row_offset==band_radius?1e10f:dp[dp_center+row_offset+1];
			float center = DTW_MIN(
							DTW_MIN(top, left),
							topleft
						) + DISTANCE(a_values[i], b_values[j]);
			dp[dp_center+row_offset] = center;
			top = center;
		}
	}

	{
		const int center_row = a_length-1;
		const int row_offset_start = DTW_MAX(-(int)band_radius, -center_row);
		const int row_offset_end = DTW_MIN(band_radius, (int)b_length - center_row - 1);
		const int desired_j = b_length-1;
		const int desired_row_offset = desired_j - center_row;

		if(row_offset_start > desired_row_offset ||
		   row_offset_end   < desired_row_offset     ){
			/*diagonal band does not cover the desired element*/
			/*i.e., the bottom right corner of the matrix*/
			result = 1e10f;
		}
		else{
			result = dp[dp_center+desired_row_offset];
			if(exclude_last_element){
				result = result - DISTANCE(a_values[a_length-1], b_values[b_length-1]);
			}
		}
	}

	free(dp);
	return result;
}

float DTW_global_slantedbanded(const float* a_values, uint32_t a_length, const float* b_values, uint32_t b_length, int band_radius, bool exclude_last_element){
	int i;
	int row_offset;
	float *dp;
	int band_width;
	int dp_center;
	float prev;
	float result;
	int center_row;

	assert(a_length > 0 && b_length > 0);
	assert(a_length < (uint32_t)INT_MAX);
	assert(b_length < (uint32_t)INT_MAX);
	assert(band_radius >= 0);

#ifdef DEBUG
	float *debug_matrix = (float *)calloc((size_t)a_length * b_length, sizeof(float));
	#define DBGM(i, j) debug_matrix[(size_t)(i) * b_length + (j)]
#endif

	/*make sure a is the longer sequence*/
	if(a_length < b_length){
		/*swap*/
		const float* tmp_values = a_values;
		const uint32_t tmp_length = a_length;
		a_values = b_values;
		a_length = b_length;
		b_values = tmp_values;
		b_length = tmp_length;
	}

	band_width = band_radius*2+1;
	dp = (float *)malloc((size_t)band_width * sizeof(float));
	for(i = 0; i < band_width; i++) dp[i] = 1e10f;
	dp_center = band_radius;

	prev = 0.0f;
	for(row_offset = 0; row_offset <= DTW_MIN(band_radius, (int)b_length-1); row_offset++){
		int j = row_offset;
		float cur = prev + DISTANCE(a_values[0], b_values[j]);
		dp[dp_center+row_offset] = cur;
		prev = cur;

#ifdef DEBUG
		DBGM(0, j) = cur;
#endif
	}

	center_row = 0;
	for(i = 1; i < (int)a_length; i++){
		/*increment center_row to follow the slope from top left to bottom right*/
		int next_row = center_row+1;
		/*floating point logic reformulated and implemented as integers for performance*/
		int64_t next_slope = next_row*(int64_t)a_length;
		int64_t target_slope = b_length*(int64_t)i;
		bool increment_center_row = false;
		int row_offset_start, row_offset_end;
		float top, topleft;

		if(next_slope <= target_slope){
			center_row++;
			increment_center_row = true;
		}

		row_offset_start = DTW_MAX(-(int)band_radius, -center_row);
		row_offset_end = DTW_MIN(band_radius, (int)b_length - center_row - 1);

		top = 1e10f;
		topleft = increment_center_row && (center_row + row_offset_start > 0)?dp[dp_center+row_offset_start]:1e10f;
		for(row_offset = row_offset_start; row_offset <= row_offset_end; row_offset++){
			int j = center_row + row_offset;
			float left;
			float center;
			if(increment_center_row){
				left = row_offset==band_radius?1e10f:dp[dp_center+row_offset+1];
			}
			else{
				left = dp[dp_center+row_offset];
			}
			center = DTW_MIN(
							DTW_MIN(top, left),
							topleft
						) + DISTANCE(a_values[i], b_values[j]);
			dp[dp_center+row_offset] = center;
			top = center;
			topleft = left;

#ifdef DEBUG
			if(i < (int)a_length && j < (int)b_length)
				DBGM(i, j) = center;
#endif
		}
	}

#ifdef DEBUG
	fprintf(stdout, "debug_matrix for DTW_global_slantedbanded:\n");
	{
		int dj, di;
		for(dj = 0; dj < (int)b_length; dj++){
			for(di = 0; di < (int)a_length; di++){
				fprintf(stdout, "%.2f\t", DBGM(di, dj));
			}
			fprintf(stdout, "\n");
		}
	}
	#undef DBGM
	free(debug_matrix);
#endif

	{
		const int desired_j = b_length-1;
		const int desired_row_offset = desired_j - center_row;
		result = dp[dp_center+desired_row_offset];
	}

	if(exclude_last_element){
		result = result - DISTANCE(a_values[a_length-1], b_values[b_length-1]);
	}

	free(dp);
	return result;
}

float DTW_global_slantedbanded_antidiagonalwise(const float* RESTRICT a_values, uint32_t a_length, const float* RESTRICT b_values, uint32_t b_length, int band_radius, bool exclude_last_element){
	int i;
	int extra_band_radius_due_to_slanting;
	int primary_antidiagonal_length;
	int secondary_antidiagonal_length;
	bool primary_larger;
	int dpsize;
	float *dp_storage;
	float *RESTRICT dp0, *RESTRICT dp1, *RESTRICT dp2;
	int center_row;
	bool previous_increment_center_row;
	float result;

	assert(a_length > 0 && b_length > 0);
	assert(a_length < (uint32_t)INT_MAX);
	assert(b_length < (uint32_t)INT_MAX);
	assert(band_radius >= 0);

#ifdef DEBUG
	float *debug_matrix = (float *)calloc((size_t)a_length * b_length, sizeof(float));
	#define DBGM(i, j) debug_matrix[(size_t)(i) * b_length + (j)]
#endif

	/*make sure a is the longer sequence*/
	if(a_length < b_length){
		/*swap*/
		const float* tmp_values = a_values;
		const uint32_t tmp_length = a_length;
		a_values = b_values;
		a_length = b_length;
		b_values = tmp_values;
		b_length = tmp_length;
		(void)tmp_length;
	}

	/*math.ceil((1-m/n)*band_radius)*/
	/*math.ceil((n-m)/n*band_radius)*/
	/*math.ceil((n-m)*band_radius/n)*/
	/*((n-m)*band_radius + n - 1) / n*/
	extra_band_radius_due_to_slanting = ((a_length-b_length)*band_radius + a_length - 1) / a_length;

	band_radius += extra_band_radius_due_to_slanting;
	primary_antidiagonal_length = band_radius + (band_radius % 2 == 0 ? 1 : 0);
	secondary_antidiagonal_length = band_radius + (band_radius % 2 == 1 ? 1 : 0);
	primary_larger = primary_antidiagonal_length > secondary_antidiagonal_length;

	dpsize = DTW_MAX(primary_antidiagonal_length, secondary_antidiagonal_length);
	dp_storage = (float *)malloc((size_t)dpsize * 3 * sizeof(float));
	dp0 = dp_storage;
	dp1 = dp_storage + dpsize;
	dp2 = dp_storage + dpsize*2;
	for(i = 0; i < dpsize; i++){ /*initialize to 1e10 to simplify (literal) corner cases*/
		dp0[i] = 1e10f;
		dp1[i] = 1e10f;
		dp2[i] = 1e10f;
	}

	center_row = 0;
	{ /*iteration 0*/
		int iteration = 0;
		int center_column = iteration;
		int antidiagonal_start_i = center_column + primary_antidiagonal_length/2;
		int antidiagonal_start_j = center_row - primary_antidiagonal_length/2;
		float *tmp;

		{
			int antidiagonal_offset = primary_antidiagonal_length/2;
			int ai = antidiagonal_start_i - antidiagonal_offset;
			int aj = antidiagonal_start_j + antidiagonal_offset;
			if(aj >= 0 && aj < (int)b_length && ai >= 0 && ai < (int)a_length){
				if(primary_larger)
					dp2[antidiagonal_offset] = DISTANCE(a_values[ai], b_values[aj]);
				else
					dp2[antidiagonal_offset+1] = DISTANCE(a_values[ai], b_values[aj]);

#ifdef DEBUG
				if(primary_larger)
					DBGM(ai, aj) = dp2[antidiagonal_offset];
				else
					DBGM(ai, aj) = dp2[antidiagonal_offset+1];
#endif
			}
		}
		tmp = dp0;
		dp0 = dp1;
		dp1 = dp2;
		dp2 = tmp;
	}

	previous_increment_center_row = false;
	for(i = 1; (uint32_t)i < a_length; i++){
		int center_column = i;
		int next_row = center_row+1;
		int64_t next_slope = next_row*(int64_t)a_length;
		int64_t target_slope = b_length*(int64_t)center_column;
		bool increment_center_row = false;
		float *tmp;

		if(next_slope <= target_slope){
			center_row++;
			increment_center_row = true;
		}

		if(increment_center_row){
			int antidiagonal_start_i = center_column + secondary_antidiagonal_length/2 - 1;
			int antidiagonal_start_j = center_row - secondary_antidiagonal_length/2;

			int antidiagonal_offset_start = DTW_MAX(DTW_MAX(0, antidiagonal_start_i-(int)a_length+1), -antidiagonal_start_j);
			int antidiagonal_offset_end = DTW_MIN(DTW_MIN(secondary_antidiagonal_length, (int)antidiagonal_start_i+1), (int)b_length-antidiagonal_start_j);
			int antidiagonal_offset;

			if(primary_larger){
				for(antidiagonal_offset = antidiagonal_offset_start; antidiagonal_offset < antidiagonal_offset_end; antidiagonal_offset++){
					int ai = antidiagonal_start_i - antidiagonal_offset;
					int aj = antidiagonal_start_j + antidiagonal_offset;

					/*calculate dp entry*/
					float top = dp1[antidiagonal_offset];
					float topleft = dp0[antidiagonal_offset];
					float left = dp1[antidiagonal_offset+1];
					float center = DTW_MIN(
									DTW_MIN(top, left),
									topleft
								) + DISTANCE(a_values[ai], b_values[aj]);
					dp2[antidiagonal_offset] = center;

#ifdef DEBUG
					DBGM(ai, aj) = center;
#endif
				}
			}
			else{
				for(antidiagonal_offset = antidiagonal_offset_start; antidiagonal_offset < antidiagonal_offset_end; antidiagonal_offset++){
					int ai = antidiagonal_start_i - antidiagonal_offset;
					int aj = antidiagonal_start_j + antidiagonal_offset;

					bool is_first = antidiagonal_offset==0;
					bool is_last = antidiagonal_offset==secondary_antidiagonal_length-1;
					float top = is_first?1e10f:dp1[antidiagonal_offset];
					float topleft = is_first && !previous_increment_center_row ?
						1e10f : dp0[antidiagonal_offset]; /*when the secondary is larger, topleft is only available if dp0 was a secondary one*/
					float left = is_last?1e10f:dp1[antidiagonal_offset+1];
					float center = DTW_MIN(
									DTW_MIN(top, left),
									topleft
								) + DISTANCE(a_values[ai], b_values[aj]);
					dp2[antidiagonal_offset] = center;

#ifdef DEBUG
					DBGM(ai, aj) = center;
#endif
				}
			}

			tmp = dp0;
			dp0 = dp1;
			dp1 = dp2;
			dp2 = tmp;
		}

		{
			int antidiagonal_start_i = center_column + primary_antidiagonal_length/2;
			int antidiagonal_start_j = center_row - primary_antidiagonal_length/2;

			int antidiagonal_offset_start = DTW_MAX(DTW_MAX(0, antidiagonal_start_i-(int)a_length+1), -antidiagonal_start_j);
			int antidiagonal_offset_end = DTW_MIN(DTW_MIN(primary_antidiagonal_length, (int)antidiagonal_start_i+1), (int)b_length-antidiagonal_start_j);
			int antidiagonal_offset;

			if(primary_larger){
				for(antidiagonal_offset = antidiagonal_offset_start; antidiagonal_offset < antidiagonal_offset_end; antidiagonal_offset++){
					int ai = antidiagonal_start_i - antidiagonal_offset;
					int aj = antidiagonal_start_j + antidiagonal_offset;

					float top, topleft, left;
					if(increment_center_row){
						bool is_first = antidiagonal_offset==0;
						bool is_last = antidiagonal_offset==primary_antidiagonal_length-1;
						top = is_first?1e10f:dp1[antidiagonal_offset-1]; /*the first element of a primary antidiagonal never has a top*/
						topleft = dp0[antidiagonal_offset]; /*all elements have a topleft when going down*/
						left = is_last?1e10f:dp1[antidiagonal_offset]; /*the last element of a primary antidiagonal does not have a left when going down*/
					}
					else{
						bool is_first = antidiagonal_offset==0;
						top = is_first?1e10f:dp1[antidiagonal_offset-1]; /*the first element of a primary antidiagonal never has a top*/
						topleft = is_first?1e10f:dp0[antidiagonal_offset-1]; /*the first element of a primary antidiagonal does not have a topleft when not going down*/
						left = dp1[antidiagonal_offset]; /*all elements have a left when not going down*/
					}

					{
						float center = DTW_MIN(
										DTW_MIN(top, left),
										topleft
									) + DISTANCE(a_values[ai], b_values[aj]);
						dp2[antidiagonal_offset] = center;

#ifdef DEBUG
						DBGM(ai, aj) = center;
#endif
					}
				}
			}
			else{
				for(antidiagonal_offset = antidiagonal_offset_start; antidiagonal_offset < antidiagonal_offset_end; antidiagonal_offset++){
					int ai = antidiagonal_start_i - antidiagonal_offset;
					int aj = antidiagonal_start_j + antidiagonal_offset;

					/*to simplify the code, accesses to primary anti diagonal will be starting at dp0[1] instead of dp0[0]*/
					float top, topleft, left;
					if(increment_center_row){
						top = dp1[antidiagonal_offset]; /*the first element of a primary antidiagonal always has a top when going down*/
						topleft = dp0[antidiagonal_offset+1]; /*all elements have a topleft when going down. +1 due to the simplification (see comment above)*/
						left = dp1[antidiagonal_offset+1]; /*all elements have a left when going down*/
					}
					else{
						bool is_first = antidiagonal_offset==0;
						top = is_first?1e10f:dp1[antidiagonal_offset]; /*the first element of a primary antidiagonal never has a top. No -1 due to the simplification (see comment above)*/
						topleft = is_first && !previous_increment_center_row ?
							1e10f:dp0[antidiagonal_offset]; /*the first element of a primary antidiagonal does not have a topleft when not previously going down*/
						left = dp1[antidiagonal_offset+1]; /*all elements have a left when not going down*/
					}

					{
						float center = DTW_MIN(
										DTW_MIN(top, left),
										topleft
									) + DISTANCE(a_values[ai], b_values[aj]);
						dp2[antidiagonal_offset+1] = center; /*+1 due to the simplification (see comment above)*/

#ifdef DEBUG
						DBGM(ai, aj) = center;
#endif
					}
				}
			}
		}

		tmp = dp0;
		dp0 = dp1;
		dp1 = dp2;
		dp2 = tmp;
		previous_increment_center_row = increment_center_row;
	}

#ifdef DEBUG
	fprintf(stdout, "debug_matrix for DTW_global_slantedbanded_antidiagonalwise:\n");
	{
		int dj, di;
		for(dj = 0; dj < (int)b_length; dj++){
			for(di = 0; di < (int)a_length; di++){
				fprintf(stdout, "%.2f\t", DBGM(di, dj));
			}
			fprintf(stdout, "\n");
		}
	}
	#undef DBGM
	free(debug_matrix);
#endif

	if(primary_larger){
		result = dp1[primary_antidiagonal_length/2];
	}
	else{
		result = dp1[primary_antidiagonal_length/2+1]; /*+1 due to the simplification (see comment above)*/
	}

	if(exclude_last_element){
		result = result - DISTANCE(a_values[a_length-1], b_values[b_length-1]);
	}

	free(dp_storage);
	return result;
}

/*
 * a is aligned fully (globally) to the best matching substring of b (i.e., b is not aligned globally)
 * a is typically the shorter sequence
 */
float DTW_semiglobal(const float* a_values, uint32_t a_length, const float* b_values, uint32_t b_length, bool exclude_last_element){
	uint32_t i, j;
	float *dp = (float *)malloc((size_t)a_length * sizeof(float));
	float best = 1e10f;

	assert(a_length >= 1);
	assert(b_length >= 1);

	for(j = 0; j < a_length; j++) dp[j] = 1e10f;

	for(i = 0; i < b_length; i++){
		float old_left = dp[0];
		dp[0] = DISTANCE(a_values[0], b_values[i]);

		for(j = 1; j < a_length; j++){
			float top = dp[j-1];
			float left = dp[j];
			float topleft = old_left;
			float center = DTW_MIN(
							DTW_MIN(top, left),
							topleft
						) + DISTANCE(a_values[j], b_values[i]);
			dp[j] = center;
			old_left = left;
		}
		best = DTW_MIN(best, dp[a_length-1]);
	}

	free(dp);
	return best;
}

/*
 * a is aligned fully (globally) to the best matching substring of b (i.e., b is not aligned globally)
 * a is typically the shorter sequence
 */
float DTW_semiglobal_slow(const float* a_values, uint32_t a_length, const float* b_values, uint32_t b_length, bool exclude_last_element){
	uint32_t i, j;
	float *dp;
	float best;
	uint32_t best_j;

	assert(a_length > 0 && b_length > 0);

	dp = (float *)malloc((size_t)a_length * b_length * sizeof(float));
	#define DP(i, j) dp[(size_t)(i) * b_length + (j)]

	DP(0, 0) = DISTANCE(a_values[0], b_values[0]);

	for(i = 1; i < a_length; i++){
		DP(i, 0) = DP(i-1, 0) + DISTANCE(a_values[i], b_values[0]);
	}
	for(j = 1; j < b_length; j++){
		DP(0, j) = DISTANCE(a_values[0], b_values[j]);
	}

	for(i = 1; i < a_length; i++){
		for(j = 1; j < b_length; j++){
			float best_in = DTW_MIN(DTW_MIN(DP(i-1, j), DP(i, j-1)), DP(i-1, j-1));
			DP(i, j) = best_in + DISTANCE(a_values[i], b_values[j]);
		}
	}

	best = DP(a_length-1, 0);
	best_j = 0;
	for(j = 1; j < b_length; j++){
		if(DP(a_length-1, j) < best){
			best = DP(a_length-1, j);
			best_j = j;
		}
	}

	if(exclude_last_element){
		best = best - DISTANCE(a_values[a_length-1], b_values[best_j]);
	}

	#undef DP
	free(dp);
	return best;
}

dtw_result DTW_global_tb(const float* a_values, uint32_t a_length, const float* b_values, uint32_t b_length, bool exclude_last_element){
	uint32_t ii, jj;
	float *dp;
	dtw_result result;
	alignment_element *rev;
	size_t n_rev, k;

	assert(a_length > 0 && b_length > 0);

	dp = (float *)malloc((size_t)a_length * b_length * sizeof(float));
	#define DP(i, j) dp[(size_t)(i) * b_length + (j)]

	DP(0, 0) = DISTANCE(a_values[0], b_values[0]);

	for(ii = 1; ii < a_length; ii++){
		DP(ii, 0) = DP(ii-1, 0) + DISTANCE(a_values[ii], b_values[0]);
	}
	for(jj = 1; jj < b_length; jj++){
		DP(0, jj) = DP(0, jj-1) + DISTANCE(a_values[0], b_values[jj]);
	}

	for(ii = 1; ii < a_length; ii++){
		for(jj = 1; jj < b_length; jj++){
			float best_in = DTW_MIN(DTW_MIN(DP(ii-1, jj), DP(ii, jj-1)), DP(ii-1, jj-1));
			DP(ii, jj) = best_in + DISTANCE(a_values[ii], b_values[jj]);
		}
	}

	/* traceback */
	rev = (alignment_element *)malloc((size_t)(a_length + b_length) * sizeof(alignment_element));
	n_rev = 0;

	ii = a_length-1;
	jj = b_length-1;
	{
		alignment_element ae;
		ae.position.i = ii;
		ae.position.j = jj;
		ae.difference = DISTANCE(a_values[ii], b_values[jj]);
		rev[n_rev++] = ae;
	}

	while(ii > 0 || jj > 0){
		if(ii==0){
			jj--;
		}
		else if(jj==0){
			ii--;
		}
		else{
			float left = DP(ii-1, jj);
			float top = DP(ii, jj-1);
			float topleft = DP(ii-1, jj-1);

			if(left < DTW_MIN(top, topleft)){
				ii--;
			}
			else if(top < DTW_MIN(left, topleft)){
				jj--;
			}
			else{
				ii--;
				jj--;
			}
		}

		{
			alignment_element ae;
			ae.position.i = ii;
			ae.position.j = jj;
			ae.difference = DISTANCE(a_values[ii], b_values[jj]);
			rev[n_rev++] = ae;
		}
	}

	/* copy in reverse order */
	if(exclude_last_element){
		result.alignment_length = n_rev - 1;
		result.cost = DP(a_length-1, b_length-1) - DISTANCE(a_values[a_length-1], b_values[b_length-1]);
	}
	else{
		result.alignment_length = n_rev;
		result.cost = DP(a_length-1, b_length-1);
	}

	result.alignment = (alignment_element *)malloc(result.alignment_length * sizeof(alignment_element));
	for(k = 0; k < result.alignment_length; k++){
		result.alignment[k] = rev[n_rev - 1 - k];
	}

	#undef DP
	free(dp);
	free(rev);
	return result;
}

dtw_result DTW_semiglobal_tb(const float* a_values, uint32_t a_length, const float* b_values, uint32_t b_length, bool exclude_last_element){
	uint32_t ii, jj;
	float *dp;
	dtw_result result;
	alignment_element *rev;
	size_t n_rev, k;
	float best;
	uint32_t best_j;

	assert(a_length > 0 && b_length > 0);

	dp = (float *)malloc((size_t)a_length * b_length * sizeof(float));
	#define DP(i, j) dp[(size_t)(i) * b_length + (j)]

	DP(0, 0) = DISTANCE(a_values[0], b_values[0]);

	for(ii = 1; ii < a_length; ii++){
		DP(ii, 0) = DP(ii-1, 0) + DISTANCE(a_values[ii], b_values[0]);
	}
	for(jj = 1; jj < b_length; jj++){
		DP(0, jj) = DISTANCE(a_values[0], b_values[jj]);
	}

	for(ii = 1; ii < a_length; ii++){
		for(jj = 1; jj < b_length; jj++){
			float best_in = DTW_MIN(DTW_MIN(DP(ii-1, jj), DP(ii, jj-1)), DP(ii-1, jj-1));
			DP(ii, jj) = best_in + DISTANCE(a_values[ii], b_values[jj]);
		}
	}

	best = DP(a_length-1, 0);
	best_j = 0;
	for(jj = 1; jj < b_length; jj++){
		if(DP(a_length-1, jj) < best){
			best = DP(a_length-1, jj);
			best_j = jj;
		}
	}

	/* traceback */
	rev = (alignment_element *)malloc((size_t)(a_length + b_length) * sizeof(alignment_element));
	n_rev = 0;

	ii = a_length-1;
	jj = best_j;
	{
		alignment_element ae;
		ae.position.i = ii;
		ae.position.j = jj;
		ae.difference = DISTANCE(a_values[ii], b_values[jj]);
		rev[n_rev++] = ae;
	}

	while(ii > 0){
		if(ii==0){
			jj--;
		}
		else if(jj==0){
			ii--;
		}
		else{
			float left = DP(ii-1, jj);
			float top = DP(ii, jj-1);
			float topleft = DP(ii-1, jj-1);

			if(left < DTW_MIN(top, topleft)){
				ii--;
			}
			else if(top < DTW_MIN(left, topleft)){
				jj--;
			}
			else{
				ii--;
				jj--;
			}
		}

		{
			alignment_element ae;
			ae.position.i = ii;
			ae.position.j = jj;
			ae.difference = DISTANCE(a_values[ii], b_values[jj]);
			rev[n_rev++] = ae;
		}
	}

	/* copy in reverse order */
	if(exclude_last_element){
		/* the last element pushed to rev is the first in the alignment;
		   the first element pushed (at position n_rev-1) is the last in alignment
		   and is the one to exclude */
		result.alignment_length = n_rev - 1;
		result.cost = DP(a_length-1, best_j) - rev[0].difference;
	}
	else{
		result.alignment_length = n_rev;
		result.cost = DP(a_length-1, best_j);
	}

	result.alignment = (alignment_element *)malloc(result.alignment_length * sizeof(alignment_element));
	for(k = 0; k < result.alignment_length; k++){
		result.alignment[k] = rev[n_rev - 1 - k];
	}

	#undef DP
	free(dp);
	free(rev);
	return result;
}
