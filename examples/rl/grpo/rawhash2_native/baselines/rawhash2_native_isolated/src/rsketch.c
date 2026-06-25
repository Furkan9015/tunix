#include "rsketch.h"
#include <assert.h>
#include "rh_kvec.h"
#include <math.h>
#include <float.h>

static inline uint64_t hash64(uint64_t key, uint64_t mask){
    key = (~key + (key << 21)) & mask; // key = (key << 21) - key - 1;
    key = key ^ key >> 24;
    key = ((key + (key << 3)) + (key << 8)) & mask; // key * 265
    key = key ^ key >> 14;
    key = ((key + (key << 2)) + (key << 4)) & mask; // key * 21
    key = key ^ key >> 28;
    key = (key + (key << 31)) & mask;
    return key;
}

uint32_t dynamic_quantize(float signal,
                          float fine_min,
                          float fine_max,
                          float fine_range,
                          uint32_t n_buckets)
{
    if (n_buckets == 0) return 0;

    float nb_f = (float)n_buckets * fine_range + 0.5f;
    int32_t n_fine_i = (int32_t)nb_f;
    if (n_fine_i < 1) n_fine_i = 1;
    if ((uint32_t)n_fine_i > n_buckets) n_fine_i = (int32_t)n_buckets;
    uint32_t n_fine = (uint32_t)n_fine_i;
    uint32_t n_low  = (n_buckets - n_fine) / 2;
    uint32_t n_high = n_buckets - n_fine - n_low;

    float lo_in = -3.0f, hi_in = 3.0f;
    if (signal < fine_min) {
        if (n_low == 0) return 0;
        float denom = fine_min - lo_in;
        float t = (denom > 0.0f) ? (signal - lo_in) / denom : 0.0f;
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        uint32_t b = (uint32_t)(t * n_low);
        if (b >= n_low) b = n_low - 1;
        return b;
    } else if (signal <= fine_max) {
        float denom = fine_max - fine_min;
        float t = (denom > 0.0f) ? (signal - fine_min) / denom : 0.0f;
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        uint32_t b = (uint32_t)(t * n_fine);
        if (b >= n_fine) b = n_fine - 1;
        return n_low + b;
    } else {
        if (n_high == 0) return n_buckets - 1;
        float denom = hi_in - fine_max;
        float t = (denom > 0.0f) ? (signal - fine_max) / denom : 1.0f;
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        uint32_t b = (uint32_t)(t * n_high);
        if (b >= n_high) b = n_high - 1;
        return n_low + n_fine + b;
    }
}

void ri_sketch_min(void *km,
				   const float* s_values,
				   uint32_t id,
				   int strand,
				   uint32_t len,
				   int diff,
				   int w,
				   int e,
				   uint32_t quant_bit,
				   uint32_t n_buckets,
				   int k,
				   float fine_min,
				   float fine_max,
				   float fine_range,
				   mm128_v *p,
				   short out)
{
	assert(len > 0 && (w > 0 && w < 256) && e*quant_bit <= (64-RI_HASH_SHIFT));
	assert(n_buckets >= 1 && n_buckets <= (uint32_t)(1U<<quant_bit));

	int j, buf_pos, min_pos;
	mm128_t buf[256], min = { UINT64_MAX, UINT64_MAX };

	uint32_t span = k+e-1;

	const uint64_t id_shift = (uint64_t)id<<RI_ID_SHIFT, mask = (1ULL<<32)-1, mask_events = (1ULL<<(quant_bit*e))-1, mask_quant_bit = (1ULL<<quant_bit)-1;

	memset(buf, 0xff, w * 16);
	rh_kv_resize(mm128_t, km, *p, p->n + len/w);

	int sigBufFull = 0;
	uint32_t f_pos, l, sigBufPos = 0;
	uint32_t l_sigpos = 0; //last signal position
	uint32_t f_tmpQuantSignal = 0;
	uint64_t quantVal = 0;

	mm128_t sigBuf[e];
	memset(sigBuf, 0, e*sizeof(mm128_t));

	uint32_t prev_quant_for_diff = UINT32_MAX;

    for (f_pos = l = buf_pos = min_pos = 0; f_pos < len; ++f_pos) {
		f_tmpQuantSignal = dynamic_quantize(s_values[f_pos], fine_min, fine_max, fine_range, n_buckets)&mask_quant_bit;

		/* Quantized sig-diff: skip if |quant_diff| <= diff. diff=-1 disables filtering. */
		if(f_pos > 0 && diff >= 0) {
			int qdiff = (int)f_tmpQuantSignal - (int)prev_quant_for_diff;
			if(qdiff < 0) qdiff = -qdiff;
			if(qdiff <= diff) continue;
		}

		l++;
		mm128_t info = { UINT64_MAX, UINT64_MAX };
		l_sigpos = f_pos;
		prev_quant_for_diff = f_tmpQuantSignal;

		quantVal = (quantVal<<quant_bit|f_tmpQuantSignal)&mask_events;

		sigBuf[sigBufPos].y = id_shift | (uint32_t)f_pos<<RI_POS_SHIFT | strand;
		if(++sigBufPos == e) {sigBufFull = 1; sigBufPos = 0;}
		sigBuf[sigBufPos].x = hash64(quantVal, mask)<<RI_HASH_SHIFT | span;

		if(!sigBufFull) continue;

		info.x = sigBuf[sigBufPos].x;
		info.y = sigBuf[sigBufPos].y;

		buf[buf_pos] = info; // need to do this here as appropriate buf_pos and buf[buf_pos] are needed below
		if (l == w + e - 1 && min.x != UINT64_MAX) { // special case for the first window - because identical k-mers are not stored yet
			for (j = buf_pos + 1; j < w; ++j)
				if (min.x == buf[j].x && buf[j].y != min.y) rh_kv_push(mm128_t, km, *p, buf[j]);
			for (j = 0; j < buf_pos; ++j)
				if (min.x == buf[j].x && buf[j].y != min.y) rh_kv_push(mm128_t, km, *p, buf[j]);
		}
		if (info.x <= min.x) { // a new minimum; then write the old min
			if (l >= w + e && min.x != UINT64_MAX) rh_kv_push(mm128_t, km, *p, min);
			min = info, min_pos = buf_pos;
		} else if (buf_pos == min_pos) { // old min has moved outside the window
			if (l >= w + e - 1 && min.x != UINT64_MAX) rh_kv_push(mm128_t, km, *p, min);
			for (j = buf_pos + 1, min.x = UINT64_MAX; j < w; ++j) // the two loops are necessary when there are identical k-mers
				if (min.x >= buf[j].x) min = buf[j], min_pos = j; // >= is important s.t. min is always the closest e-mer
			for (j = 0; j <= buf_pos; ++j)
				if (min.x >= buf[j].x) min = buf[j], min_pos = j;
			if (l >= w + e - 1 && min.x != UINT64_MAX) { // write identical k-mers
				for (j = buf_pos + 1; j < w; ++j) // these two loops make sure the output is sorted
					if (min.x == buf[j].x && min.y != buf[j].y) rh_kv_push(mm128_t, km, *p, buf[j]);
				for (j = 0; j <= buf_pos; ++j)
					if (min.x == buf[j].x && min.y != buf[j].y) rh_kv_push(mm128_t, km, *p, buf[j]);
			}
		}
		if (++buf_pos == w) buf_pos = 0;
    }
	if (min.x != UINT64_MAX)
		rh_kv_push(mm128_t, km, *p, min);
}

void ri_sketch_reg(void *km,
				   const float* s_values,
				   uint32_t id,
				   int strand,
				   uint32_t len,
				   int diff,
				   int e,
				   uint32_t quant_bit,
				   uint32_t n_buckets,
				   int k,
				   float fine_min,
				   float fine_max,
				   float fine_range,
				   mm128_v *p,
				   short out){

	assert(len > 0 && (uint32_t)e*quant_bit <= 64);
	assert(n_buckets >= 1 && n_buckets <= (uint32_t)(1U<<quant_bit));

	uint32_t span = k+e-1;

	const uint64_t id_shift = (uint64_t)id<<RI_ID_SHIFT, mask = (1ULL<<32)-1, mask_events = (1ULL<<(quant_bit*e))-1, mask_quant_bit = (1ULL<<quant_bit)-1;

	int sigBufFull = 0;
	uint32_t f_pos = 0, sigBufPos = 0, l_sigpos = 0; //last signal position
	uint32_t f_tmpQuantSignal = 0;
	uint64_t quantVal = 0;

	if(!out)rh_kv_resize(mm128_t, km, *p, p->n + len-span-1);

	mm128_t sigBuf[e];
	memset(sigBuf, 0, e*sizeof(mm128_t));

	//First quantization is done here
	l_sigpos = f_pos;
	f_tmpQuantSignal = dynamic_quantize(s_values[f_pos], fine_min, fine_max, fine_range, n_buckets)&mask_quant_bit;
	uint32_t prev_quant_for_diff = f_tmpQuantSignal;
	if(out) fprintf(stdout, "%u", f_tmpQuantSignal);
	sigBuf[sigBufPos].y = id_shift | (uint32_t)f_pos<<RI_POS_SHIFT | strand;
	if(++sigBufPos == e) {sigBufFull = 1; sigBufPos = 0;}
	quantVal = f_tmpQuantSignal&mask_events;
	sigBuf[sigBufPos].x = (hash64(quantVal, mask)<<RI_HASH_SHIFT) | span;
	if(sigBufFull && !out) rh_kv_push(mm128_t, km, *p, sigBuf[sigBufPos]);

    for (f_pos = 1; f_pos < len; ++f_pos) {
		f_tmpQuantSignal = dynamic_quantize(s_values[f_pos], fine_min, fine_max, fine_range, n_buckets)&mask_quant_bit;

		/* Quantized sig-diff: skip if |quant_diff| <= diff. diff=-1 disables filtering. */
		if(diff >= 0) {
			int qdiff = (int)f_tmpQuantSignal - (int)prev_quant_for_diff;
			if(qdiff < 0) qdiff = -qdiff;
			if(qdiff <= diff) continue;
		}

		l_sigpos = f_pos;
		prev_quant_for_diff = f_tmpQuantSignal;
		if(out) fprintf(stdout, ",%u", f_tmpQuantSignal);

		sigBuf[sigBufPos].y = id_shift | (uint32_t)f_pos<<RI_POS_SHIFT | strand;
		if(++sigBufPos == e) {sigBufFull = 1; sigBufPos = 0;}

		quantVal = (quantVal<<quant_bit|f_tmpQuantSignal)&mask_events;
		sigBuf[sigBufPos].x = (hash64(quantVal, mask)<<RI_HASH_SHIFT) | span;

		if(!sigBufFull) continue;

		if(!out)rh_kv_push(mm128_t, km, *p, sigBuf[sigBufPos]);
    }
}

void ri_sketch_reg_rev(void *km,
					   const float* s_values,
					   uint32_t id,
					   int strand,
					   uint32_t len,
					   int diff,
					   int e,
					   uint32_t quant_bit,
					   uint32_t n_buckets,
					   int k,
					   float fine_min,
				   	   float fine_max,
				   	   float fine_range,
					   mm128_v *p,
					   short out)
{
	assert(len > 0 && (uint32_t)e*quant_bit <= 64-RI_HASH_SHIFT);
	assert(n_buckets >= 1 && n_buckets <= (uint32_t)(1U<<quant_bit));

	uint32_t span = k+e-1;
	const uint64_t id_shift = (uint64_t)id<<RI_ID_SHIFT, mask = (1ULL<<32)-1, mask_events = (1ULL<<(quant_bit*e))-1, mask_quant_bit = (1ULL<<quant_bit)-1;

	int sigBufFull = 0, streak = 1;
	uint32_t f_pos = 0, r_pos = len-1, sigBufPos = 0, l_sigpos = 0; //last signal position
	uint32_t f_tmpQuantSignal = 0, r_tmpQuantSignal = 0;
	uint64_t quantVal = 0;

	if(!out)rh_kv_resize(mm128_t, km, *p, p->n + len-span-1);

	mm128_t sigBuf[e];
	memset(sigBuf, 0, e*sizeof(mm128_t));

	//First quantization is done here
	l_sigpos = f_pos;
	f_tmpQuantSignal = dynamic_quantize(s_values[f_pos], fine_min, fine_max, fine_range, n_buckets)&mask_quant_bit;
	if(out) fprintf(stdout, "%u", f_tmpQuantSignal);
	sigBuf[sigBufPos].y = id_shift | ((uint32_t)(len-f_pos-1))<<RI_POS_SHIFT | strand;
	if(++sigBufPos == e) {sigBufFull = 1; sigBufPos = 0;}
	quantVal = f_tmpQuantSignal&mask_events;
	sigBuf[sigBufPos].x = quantVal;
	if(sigBufFull && !out && streak >= e) rh_kv_push(mm128_t, km, *p, sigBuf[sigBufPos]);

    for (f_pos = 1; f_pos < len; ++f_pos) {
		f_tmpQuantSignal = dynamic_quantize(s_values[f_pos], fine_min, fine_max, fine_range, n_buckets)&mask_quant_bit;

		/* Quantized sig-diff: skip if |quant_diff| <= diff. diff=-1 disables filtering. */
		if(diff >= 0) {
			uint32_t prev_q = dynamic_quantize(s_values[l_sigpos], fine_min, fine_max, fine_range, n_buckets)&mask_quant_bit;
			int qdiff = (int)f_tmpQuantSignal - (int)prev_q;
			if(qdiff < 0) qdiff = -qdiff;
			if(qdiff <= diff) {streak = 0; continue;}
		}

		l_sigpos = f_pos;
		if(out) fprintf(stdout, ",%u", f_tmpQuantSignal);

		// sigBuf[sigBufPos].y = id_shift | ((uint32_t)(len-f_pos-1))<<RI_POS_SHIFT | strand;
		sigBuf[sigBufPos].y = id_shift | (uint32_t)f_pos<<RI_POS_SHIFT | strand;
		if(++sigBufPos == e) {sigBufFull = 1; sigBufPos = 0;}

		quantVal = (quantVal<<quant_bit|f_tmpQuantSignal)&mask_events;
		streak++;
		sigBuf[sigBufPos].x = quantVal;
		
		if(!sigBufFull) continue;

		if(!out && streak >= e)rh_kv_push(mm128_t, km, *p, sigBuf[sigBufPos]);
    }
	if(out) fprintf(stdout, "\n");
}

void ri_sketch(void *km,
               const float* s_values,
               uint32_t id,
               int strand,
               uint32_t len,
               int diff,
               int w,
               int e,
               int n,
               uint32_t quant_bit,
               uint32_t n_buckets,
               int k,
               float fine_min,
               float fine_max,
               float fine_range,
               mm128_v *p,
			   short out)
{
	if(w) ri_sketch_min(km, s_values, id, strand, len, diff, w, e, quant_bit, n_buckets, k, fine_min, fine_max, fine_range, p, out);
	else ri_sketch_reg(km, s_values, id, strand, len, diff, e, quant_bit, n_buckets, k, fine_min, fine_max, fine_range, p, out);
}

void ri_sketch_rev(void *km,
				   const float* s_values,
				   uint32_t id,
				   int strand,
				   uint32_t len,
				   int diff,
				   int w,
				   int e,
				   int n,
				   uint32_t quant_bit,
				   uint32_t n_buckets,
				   int k,
				   float fine_min,
               	   float fine_max,
               	   float fine_range,
				   mm128_v *p,
				   short out)
{
	// if(w) ri_sketch_min(km, s_values, id, strand, len, diff, w, e, q, lq, k, p, out);
	ri_sketch_reg_rev(km, s_values, id, strand, len, diff, e, quant_bit, n_buckets, k, fine_min, fine_max, fine_range, p, out);
}