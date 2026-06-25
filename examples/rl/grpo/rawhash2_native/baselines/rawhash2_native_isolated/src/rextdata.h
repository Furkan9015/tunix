#ifndef REXTDATA_H
#define REXTDATA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Entry for a single read's external peak positions.
 * peaks: array of segmentation boundary indices (in raw signal space)
 * n_peaks: number of peaks
 */
typedef struct ri_ext_peaks_entry_s {
	uint32_t *peaks;
	uint32_t n_peaks;
} ri_ext_peaks_entry_t;

/**
 * Entry for a single read's external event values.
 * events: array of pre-computed event values (float)
 * n_events: number of events
 */
typedef struct ri_ext_events_entry_s {
	float *events;
	uint32_t n_events;
} ri_ext_events_entry_t;

/* Opaque handle types (khash internally).
 * `ri_ext_peaks_t` and `ri_ext_events_t` are full-data hashes (read_id -> entry).
 * `ri_ext_peaks_index_t` and `ri_ext_events_index_t` are lightweight offset indexes
 * (read_id -> byte offset into the source file), used by the per-batch loader. */
typedef void ri_ext_peaks_t;
typedef void ri_ext_events_t;
typedef void ri_ext_peaks_index_t;
typedef void ri_ext_events_index_t;

/**
 * Load external peaks file.
 * Format: one read per line, tab/space separated.
 *   Column 1: read_id (UUID string)
 *   Remaining columns: uint32 peak positions (raw signal indices)
 *   Lines starting with '#' are skipped.
 *
 * @param fname  path to peaks file
 * @return       opaque handle, or NULL on failure
 */
ri_ext_peaks_t *ri_load_ext_peaks(const char *fname);

/**
 * Load move table file and convert to peak positions.
 * Format: one read per line, tab separated.
 *   Column 1: read_id
 *   Column 2: mv:B:c,STRIDE,m0,m1,...,mN  (move table from BAM)
 *   Column 3: ts:i:OFFSET  (template start from BAM)
 *   Lines starting with '#' are skipped.
 *
 * Conversion: moves are expanded using stride and template_start
 * to produce raw-signal peak positions, stored as ext_peaks entries.
 *
 * @param fname  path to moves file
 * @return       opaque ext_peaks handle, or NULL on failure
 */
ri_ext_peaks_t *ri_load_ext_moves(const char *fname);

/**
 * Load external events file.
 * Format: one read per line, tab/space separated.
 *   Column 1: read_id (UUID string)
 *   Remaining columns: float event values
 *   Lines starting with '#' are skipped.
 *
 * @param fname  path to events file
 * @return       opaque handle, or NULL on failure
 */
ri_ext_events_t *ri_load_ext_events(const char *fname);

/**
 * Look up external peaks for a read.
 * @return  pointer to entry, or NULL if read not found
 */
const ri_ext_peaks_entry_t *ri_lookup_ext_peaks(const ri_ext_peaks_t *h, const char *read_id);

/**
 * Look up external events for a read.
 * @return  pointer to entry, or NULL if read not found
 */
const ri_ext_events_entry_t *ri_lookup_ext_events(const ri_ext_events_t *h, const char *read_id);

/** Free external peaks data */
void ri_destroy_ext_peaks(ri_ext_peaks_t *h);

/** Free external events data */
void ri_destroy_ext_events(ri_ext_events_t *h);

/* -------------------------------------------------------------------------
 * Indexed batch loading (low-memory path).
 *
 * Scan the file once to record read_id -> byte_offset. The file handle is
 * kept open. Per-pipeline-batch, the mapping pipeline asks for just the
 * subset of reads it is processing; a small per-batch hash is built by
 * fseek()ing to each read's offset and re-parsing only those lines.
 *
 * The per-batch hash returned by ri_batch_load_ext_* is a full
 * ri_ext_peaks_t / ri_ext_events_t — i.e., the existing ri_lookup_ext_*
 * functions work against it transparently. Callers are responsible for
 * destroying it via ri_destroy_ext_peaks/events when the batch is done.
 * ------------------------------------------------------------------------- */

/** Build offset index by scanning a peaks file once. File handle stays open. */
ri_ext_peaks_index_t *ri_index_ext_peaks(const char *fname);

/** Build offset index by scanning a moves file once. File handle stays open.
 *  The returned index parses lines with the moves->peaks conversion at batch-load time. */
ri_ext_peaks_index_t *ri_index_ext_moves(const char *fname);

/** Build offset index by scanning an events file once. File handle stays open. */
ri_ext_events_index_t *ri_index_ext_events(const char *fname);

/**
 * Build a small full-data hash containing only the reads in `read_ids`.
 * Lookups via ri_lookup_ext_peaks() work normally on the result.
 * Reads not found in the offset index are silently omitted (lookup returns NULL).
 *
 * @param idx       offset index (must be from ri_index_ext_peaks/moves)
 * @param read_ids  array of n_ids C strings (NUL-terminated)
 * @param n_ids     number of read_ids
 * @return          opaque handle (small khash), or NULL on failure;
 *                  free with ri_destroy_ext_peaks
 */
ri_ext_peaks_t *ri_batch_load_ext_peaks(ri_ext_peaks_index_t *idx, char *const *read_ids, uint32_t n_ids);

/** Same as ri_batch_load_ext_peaks but for events files. Free with ri_destroy_ext_events. */
ri_ext_events_t *ri_batch_load_ext_events(ri_ext_events_index_t *idx, char *const *read_ids, uint32_t n_ids);

/** Free a peaks/moves offset index (closes file, frees offset hash). */
void ri_destroy_ext_peaks_index(ri_ext_peaks_index_t *idx);

/** Free an events offset index (closes file, frees offset hash). */
void ri_destroy_ext_events_index(ri_ext_events_index_t *idx);

#ifdef __cplusplus
}
#endif

#endif /* REXTDATA_H */
