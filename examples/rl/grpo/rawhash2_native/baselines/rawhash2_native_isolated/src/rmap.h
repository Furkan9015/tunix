#ifndef RMAP_H
#define RMAP_H

#include "rindex.h"
#include "rsig.h"
#include "chain.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ri_map_s{
	uint32_t c_id; //chain index
	uint32_t read_length;
	uint32_t ref_id;
	uint32_t read_start_position;
	uint32_t read_end_position;
	uint32_t fragment_start_position;
	uint32_t fragment_length;
	uint8_t mapq : 6, rev : 1, mapped : 1;
	char* tags;
} ri_map_t;

typedef struct ri_reg1_s{
	uint32_t read_id;
	// uint32_t ref_id;
	const char* read_name;
	// uint32_t read_start_position;
	// uint32_t read_end_position;
	// uint32_t fragment_start_position;
	// uint32_t fragment_length;
	// uint8_t mapq : 6, rev : 1, mapped : 1;
	// char* tags;

	ri_map_t* maps;
	uint32_t n_maps;

	uint32_t offset;
	float* events;
	mm128_t *prev_anchors;
	uint32_t n_prev_anchors;
	mm_reg1_t* creg; // This is for transition purposes.
	int n_cregs;
} ri_reg1_t;

typedef struct pipeline_ms{
	int n_processed, n_threads, io_n_threads, n_fp, cur_fp, n_f, cur_f;
	int64_t mini_batch_size;
	const ri_mapopt_t *opt;
	char **f;
	ri_sig_file_t *fp;
	const ri_idx_t *ri;
	const char **fn;
	uint32_t su_nreads, su_nestimations, ab_count, su_cur;
	float** su_estimations;
	uint32_t* su_c_estimations;
	int su_stop;
} pipeline_mt;

typedef struct step_ms{
	const pipeline_mt *p;
    int n_sig;
	ri_sig_t** sig;
	// int* n_reg, *seg_off, *n_seg, *rep_len, *frag_gap;
	ri_reg1_t** reg;
	ri_tbuf_t** buf;
	/* Per-batch external segmentation data, populated by step 0 when the
	 * pipeline's options carry an ext_peaks_index / ext_events_index. The
	 * batch_opt is a shallow copy of *p->opt with ext_peaks/ext_events
	 * overridden to point at these per-batch caches, so map_worker_for can
	 * read from &s->batch_opt instead of s->p->opt without any other code
	 * changes. Both fields are NULL when batch loading isn't active. */
	void *batch_ext_peaks;     /* ri_ext_peaks_t* (small khash, this batch only) */
	void *batch_ext_events;    /* ri_ext_events_t* (small khash, this batch only) */
	ri_mapopt_t batch_opt;     /* shallow copy of *p->opt; ext_peaks/events overridden */
} step_mt;

/**
 * Map raw nanopore signals of a single read to a reference genome
 *
 * @param idx		rindex (see rindex.h)
 * @param fn		path to the signal files
 * @param opt		mapping options
 * @param n_threads	number of threads to use in mapping
 * @param io_n_threads	number of threads to use in I/O operations (reading from a file)
 * 
 * @return		returns 0 if mapping is completed with no issues. -1, otherwise.
 */
int ri_map_file(const ri_idx_t *idx, const char *fn, const ri_mapopt_t *opt, int n_threads, int io_n_threads);

/**
 * Map raw nanopore signals of many reads to a reference genome
 *
 * @param idx		rindex (see rindex.h)
 * @param n_segs	number of signal files
 * @param fn		paths to the signal files
 * @param opt		mapping options
 * @param n_threads	number of threads to use in mapping
 * @param io_n_threads	number of threads to use in I/O operations (reading from a file)
 * 
 * @return			returns 0 if mapping is completed with no issues. -1, otherwise.
 */
int ri_map_file_frag(const ri_idx_t *idx, int n_segs, const char **fn, const ri_mapopt_t *opt, int n_threads, int io_n_threads);

/**
 * Incremental chunk-processing API.
 * Used by both the batch pipeline (map_worker_for) and live streaming (rlive.cpp).
 */

/** Process one signal chunk and check for mapping. Returns 1 if mapping found. */
int ri_map_one_chunk(const ri_idx_t *ri, const ri_mapopt_t *opt,
                     const float *chunk_sig, uint32_t chunk_len,
                     ri_reg1_t *reg, ri_tbuf_t *b,
                     double *mean_sum, double *std_dev_sum,
                     uint32_t *n_events_sum, const char *qname);

/** Finalize mapping results: generate PAF tags, handle CIGAR re-alignment. */
void ri_map_finalize(const ri_idx_t *ri, const ri_mapopt_t *opt,
                     ri_reg1_t *reg, ri_tbuf_t *b,
                     uint32_t read_id, const char *read_name,
                     uint32_t qlen, uint32_t c_count, uint32_t l_chunk,
                     double mapping_time);

/** Clean up per-read mapping state (prev_anchors, creg, events, memory pool). */
void ri_map_cleanup(ri_reg1_t *reg, ri_tbuf_t *b);

/**
 * Live streaming support: expose internal worker functions for the gRPC
 * live pipeline (rlive.cpp). Guarded by NGRPCRH to avoid unused-function
 * warnings when gRPC is disabled.
 */
#ifndef NGRPCRH
void ri_map_worker_for(void *data, long i, int tid);
ri_tbuf_t *ri_tbuf_init_live(void);
void ri_tbuf_destroy_live(ri_tbuf_t *b);
#endif

#ifdef __cplusplus
}
#endif
#endif //RMAP_H