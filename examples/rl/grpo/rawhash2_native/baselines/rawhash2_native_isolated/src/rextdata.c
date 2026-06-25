#include "rextdata.h"
#include "khash.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* khash instantiations for string -> peaks/events mappings */
KHASH_MAP_INIT_STR(ext_peaks, ri_ext_peaks_entry_t)
KHASH_MAP_INIT_STR(ext_events, ri_ext_events_entry_t)

/* Offset index: string read_id -> int64 byte offset into file.
 * ~92 bytes per read overhead vs the multi-KB-per-read cost of the full hash. */
KHASH_MAP_INIT_STR(ext_offsets, int64_t)

/* Maximum line length for external data files (1 MB).
 * A read with 10,000 peaks at ~7 chars each needs ~70 KB. */
#define REXTDATA_LINE_BUF_SIZE (1024 * 1024)

/* Format tag stored on the index handle so the per-batch loader knows whether
 * to apply the moves->peaks conversion. */
typedef enum { RI_EXT_FORMAT_PEAKS = 1, RI_EXT_FORMAT_MOVES = 2, RI_EXT_FORMAT_EVENTS = 3 } ri_ext_format_t;

/* Concrete struct backing the opaque ri_ext_peaks_index_t / ri_ext_events_index_t.
 * Both peaks and events indexes use the same shape — only the format tag differs. */
typedef struct ri_ext_index_s {
    FILE *fp;
    khash_t(ext_offsets) *offsets;
    ri_ext_format_t format;
    char *fname;
} ri_ext_index_t;

/* -------------------------------------------------------------------------
 * Internal line parsers — these consume a single line (whose contents may be
 * modified by strtok_r) and insert one entry into the destination khash.
 * Returns 1 on successful insert, 0 if the line was skipped (malformed/empty).
 * Both the legacy full-load functions and the new batch loader use these,
 * guaranteeing byte-identical parsing.
 *
 * NOTE: caller is responsible for stripping trailing CR/LF from `line`.
 * ------------------------------------------------------------------------- */
static int parse_peaks_line_into(khash_t(ext_peaks) *h, char *line)
{
	/* first token: read ID */
	char *saveptr = NULL;
	char *token = strtok_r(line, " \t", &saveptr);
	if (!token) return 0;

	char *read_id = strdup(token);

	/* parse peak positions into dynamic array */
	uint32_t cap = 1024, n = 0;
	uint32_t *peaks = (uint32_t*)malloc(cap * sizeof(uint32_t));

	while ((token = strtok_r(NULL, " \t", &saveptr)) != NULL) {
		if (n >= cap) {
			cap *= 2;
			peaks = (uint32_t*)realloc(peaks, cap * sizeof(uint32_t));
		}
		peaks[n++] = (uint32_t)strtoul(token, NULL, 10);
	}

	/* shrink to fit */
	if (n > 0)
		peaks = (uint32_t*)realloc(peaks, n * sizeof(uint32_t));
	else {
		free(peaks);
		peaks = NULL;
	}

	/* insert into hash map */
	int ret;
	khiter_t k = kh_put(ext_peaks, h, read_id, &ret);
	if (ret == 0) {
		/* duplicate read ID — replace old entry */
		free((char*)kh_key(h, k));
		free(kh_val(h, k).peaks);
		kh_key(h, k) = read_id;
	}
	kh_val(h, k).peaks = peaks;
	kh_val(h, k).n_peaks = n;
	return 1;
}

ri_ext_peaks_t *ri_load_ext_peaks(const char *fname)
{
	FILE *fp = fopen(fname, "r");
	if (!fp) return NULL;

	khash_t(ext_peaks) *h = kh_init(ext_peaks);
	char *line = (char*)malloc(REXTDATA_LINE_BUF_SIZE);
	uint32_t n_reads = 0;

	while (fgets(line, REXTDATA_LINE_BUF_SIZE, fp)) {
		/* skip comments and empty lines */
		if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;

		/* strip trailing newline */
		size_t len = strlen(line);
		while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
		if (len == 0) continue;

		if (parse_peaks_line_into(h, line)) n_reads++;
	}

	free(line);
	fclose(fp);
	fprintf(stderr, "[M::%s] loaded %u reads from peaks file '%s'\n", __func__, n_reads, fname);
	return (ri_ext_peaks_t*)h;
}

static int parse_moves_line_into(khash_t(ext_peaks) *h, char *line)
{
	/* Split into tab-separated columns:
	 *   col1: read_id
	 *   col2: mv:B:c,STRIDE,m0,m1,...,mN
	 *   col3: ts:i:OFFSET
	 */
	char *saveptr = NULL;
	char *col1 = strtok_r(line, "\t", &saveptr);
	char *col2 = strtok_r(NULL, "\t", &saveptr);
	char *col3 = strtok_r(NULL, "\t", &saveptr);
	if (!col1 || !col2 || !col3) return 0;

	char *read_id = strdup(col1);

	/* Parse mv:B:c,STRIDE,m0,m1,...,mN — skip "mv:B:c," prefix, first comma-value is stride */
	char *mv_data = col2;
	if (strncmp(mv_data, "mv:B:c,", 7) != 0) {
		free(read_id);
		return 0;
	}
	mv_data += 7;

	char *mv_saveptr = NULL;
	char *mv_tok = strtok_r(mv_data, ",", &mv_saveptr);
	if (!mv_tok) { free(read_id); return 0; }

	uint32_t stride = (uint32_t)strtoul(mv_tok, NULL, 10);
	if (stride == 0) { free(read_id); return 0; }

	uint32_t mv_cap = 4096, n_moves = 0;
	int8_t *moves = (int8_t*)malloc(mv_cap * sizeof(int8_t));

	while ((mv_tok = strtok_r(NULL, ",", &mv_saveptr)) != NULL) {
		if (n_moves >= mv_cap) {
			mv_cap *= 2;
			moves = (int8_t*)realloc(moves, mv_cap * sizeof(int8_t));
		}
		moves[n_moves++] = (int8_t)atoi(mv_tok);
	}

	/* Parse ts:i:OFFSET — skip "ts:i:" prefix (5 chars) */
	char *ts_data = col3;
	if (strncmp(ts_data, "ts:i:", 5) != 0) {
		free(read_id);
		free(moves);
		return 0;
	}
	uint32_t template_start = (uint32_t)strtoul(ts_data + 5, NULL, 10);

	/* Convert moves to peak positions; first peak = template_start */
	uint32_t pk_cap = 1024, n_peaks = 0;
	uint32_t *peaks = (uint32_t*)malloc(pk_cap * sizeof(uint32_t));
	peaks[n_peaks++] = template_start;
	uint32_t end = template_start + stride;
	for (uint32_t i = 1; i < n_moves; i++) {
		if (moves[i] == 1) {
			if (n_peaks >= pk_cap) {
				pk_cap *= 2;
				peaks = (uint32_t*)realloc(peaks, pk_cap * sizeof(uint32_t));
			}
			peaks[n_peaks++] = end;
		}
		end += stride;
	}

	free(moves);

	if (n_peaks > 0)
		peaks = (uint32_t*)realloc(peaks, n_peaks * sizeof(uint32_t));
	else {
		free(peaks);
		peaks = NULL;
	}

	int ret;
	khiter_t k = kh_put(ext_peaks, h, read_id, &ret);
	if (ret == 0) {
		free((char*)kh_key(h, k));
		free(kh_val(h, k).peaks);
		kh_key(h, k) = read_id;
	}
	kh_val(h, k).peaks = peaks;
	kh_val(h, k).n_peaks = n_peaks;
	return 1;
}

ri_ext_peaks_t *ri_load_ext_moves(const char *fname)
{
	FILE *fp = fopen(fname, "r");
	if (!fp) return NULL;

	khash_t(ext_peaks) *h = kh_init(ext_peaks);
	char *line = (char*)malloc(REXTDATA_LINE_BUF_SIZE);
	uint32_t n_reads = 0;

	while (fgets(line, REXTDATA_LINE_BUF_SIZE, fp)) {
		if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
		size_t len = strlen(line);
		while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
		if (len == 0) continue;

		if (parse_moves_line_into(h, line)) n_reads++;
	}

	free(line);
	fclose(fp);
	fprintf(stderr, "[M::%s] loaded %u reads from moves file '%s'\n", __func__, n_reads, fname);
	return (ri_ext_peaks_t*)h;
}

static int parse_events_line_into(khash_t(ext_events) *h, char *line)
{
	char *saveptr = NULL;
	char *token = strtok_r(line, " \t", &saveptr);
	if (!token) return 0;

	char *read_id = strdup(token);

	uint32_t cap = 1024, n = 0;
	float *events = (float*)malloc(cap * sizeof(float));

	while ((token = strtok_r(NULL, " \t", &saveptr)) != NULL) {
		if (n >= cap) {
			cap *= 2;
			events = (float*)realloc(events, cap * sizeof(float));
		}
		events[n++] = strtof(token, NULL);
	}

	if (n > 0)
		events = (float*)realloc(events, n * sizeof(float));
	else {
		free(events);
		events = NULL;
	}

	int ret;
	khiter_t k = kh_put(ext_events, h, read_id, &ret);
	if (ret == 0) {
		free((char*)kh_key(h, k));
		free(kh_val(h, k).events);
		kh_key(h, k) = read_id;
	}
	kh_val(h, k).events = events;
	kh_val(h, k).n_events = n;
	return 1;
}

ri_ext_events_t *ri_load_ext_events(const char *fname)
{
	FILE *fp = fopen(fname, "r");
	if (!fp) return NULL;

	khash_t(ext_events) *h = kh_init(ext_events);
	char *line = (char*)malloc(REXTDATA_LINE_BUF_SIZE);
	uint32_t n_reads = 0;

	while (fgets(line, REXTDATA_LINE_BUF_SIZE, fp)) {
		if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
		size_t len = strlen(line);
		while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
		if (len == 0) continue;

		if (parse_events_line_into(h, line)) n_reads++;
	}

	free(line);
	fclose(fp);
	fprintf(stderr, "[M::%s] loaded %u reads from events file '%s'\n", __func__, n_reads, fname);
	return (ri_ext_events_t*)h;
}

const ri_ext_peaks_entry_t *ri_lookup_ext_peaks(const ri_ext_peaks_t *h, const char *read_id)
{
	if (!h || !read_id) return NULL;
	const khash_t(ext_peaks) *ht = (const khash_t(ext_peaks)*)h;
	khiter_t k = kh_get(ext_peaks, ht, read_id);
	if (k == kh_end(ht)) return NULL;
	return &kh_val(ht, k);
}

const ri_ext_events_entry_t *ri_lookup_ext_events(const ri_ext_events_t *h, const char *read_id)
{
	if (!h || !read_id) return NULL;
	const khash_t(ext_events) *ht = (const khash_t(ext_events)*)h;
	khiter_t k = kh_get(ext_events, ht, read_id);
	if (k == kh_end(ht)) return NULL;
	return &kh_val(ht, k);
}

void ri_destroy_ext_peaks(ri_ext_peaks_t *h)
{
	if (!h) return;
	khash_t(ext_peaks) *ht = (khash_t(ext_peaks)*)h;
	khiter_t k;
	for (k = kh_begin(ht); k != kh_end(ht); ++k) {
		if (!kh_exist(ht, k)) continue;
		free((char*)kh_key(ht, k));
		free(kh_val(ht, k).peaks);
	}
	kh_destroy(ext_peaks, ht);
}

void ri_destroy_ext_events(ri_ext_events_t *h)
{
	if (!h) return;
	khash_t(ext_events) *ht = (khash_t(ext_events)*)h;
	khiter_t k;
	for (k = kh_begin(ht); k != kh_end(ht); ++k) {
		if (!kh_exist(ht, k)) continue;
		free((char*)kh_key(ht, k));
		free(kh_val(ht, k).events);
	}
	kh_destroy(ext_events, ht);
}

/* -------------------------------------------------------------------------
 * Indexed batch loading
 *
 * Two-phase: scan file once at startup recording (read_id -> byte offset);
 * later, given a list of read_ids, fseek+parse only those lines and return
 * a small khash of just-this-batch entries.
 * ------------------------------------------------------------------------- */

/* Build the offset index by scanning the file once.
 * Records byte offset of each non-comment, non-empty data line, keyed by the
 * first whitespace-delimited token (the read_id). The file is left open at
 * EOF — caller must use fseek before subsequent reads. */
static ri_ext_index_t *build_offset_index(const char *fname, ri_ext_format_t format)
{
	FILE *fp = fopen(fname, "r");
	if (!fp) return NULL;

	khash_t(ext_offsets) *offsets = kh_init(ext_offsets);
	char *line = (char*)malloc(REXTDATA_LINE_BUF_SIZE);
	uint32_t n_reads = 0;

	long off = ftell(fp);
	while (fgets(line, REXTDATA_LINE_BUF_SIZE, fp)) {
		/* skip comments and empty lines */
		if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') {
			off = ftell(fp);
			continue;
		}

		/* Find the read_id (first whitespace-delimited token) without using strtok
		 * (we don't want to mutate the line during indexing, since we're not parsing
		 * the rest right now). */
		size_t i = 0;
		while (line[i] && line[i] != ' ' && line[i] != '\t' && line[i] != '\n' && line[i] != '\r') i++;
		if (i == 0) { off = ftell(fp); continue; }

		char *read_id = (char*)malloc(i + 1);
		memcpy(read_id, line, i);
		read_id[i] = '\0';

		int ret;
		khiter_t k = kh_put(ext_offsets, offsets, read_id, &ret);
		if (ret == 0) {
			/* duplicate — replace */
			free((char*)kh_key(offsets, k));
			kh_key(offsets, k) = read_id;
		}
		kh_val(offsets, k) = (int64_t)off;
		n_reads++;

		off = ftell(fp);
	}

	free(line);

	ri_ext_index_t *idx = (ri_ext_index_t*)malloc(sizeof(ri_ext_index_t));
	idx->fp = fp;
	idx->offsets = offsets;
	idx->format = format;
	idx->fname = strdup(fname);

	const char *kind = (format == RI_EXT_FORMAT_PEAKS) ? "peaks" :
	                   (format == RI_EXT_FORMAT_MOVES) ? "moves" : "events";
	fprintf(stderr, "[M::%s] indexed %u reads from %s file '%s' (offset hash kept; data not loaded)\n",
	        __func__, n_reads, kind, fname);

	return idx;
}

ri_ext_peaks_index_t *ri_index_ext_peaks(const char *fname)
{
	return (ri_ext_peaks_index_t*)build_offset_index(fname, RI_EXT_FORMAT_PEAKS);
}

ri_ext_peaks_index_t *ri_index_ext_moves(const char *fname)
{
	return (ri_ext_peaks_index_t*)build_offset_index(fname, RI_EXT_FORMAT_MOVES);
}

ri_ext_events_index_t *ri_index_ext_events(const char *fname)
{
	return (ri_ext_events_index_t*)build_offset_index(fname, RI_EXT_FORMAT_EVENTS);
}

/* Read one line at the given byte offset into `line` (caller provides buffer).
 * Returns 1 on success, 0 on read failure. Strips trailing CR/LF. */
static int read_line_at_offset(FILE *fp, int64_t off, char *line)
{
	if (fseek(fp, (long)off, SEEK_SET) != 0) return 0;
	if (!fgets(line, REXTDATA_LINE_BUF_SIZE, fp)) return 0;
	size_t len = strlen(line);
	while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
	return len > 0;
}

ri_ext_peaks_t *ri_batch_load_ext_peaks(ri_ext_peaks_index_t *idx_v, char *const *read_ids, uint32_t n_ids)
{
	if (!idx_v) return NULL;
	ri_ext_index_t *idx = (ri_ext_index_t*)idx_v;
	if (idx->format != RI_EXT_FORMAT_PEAKS && idx->format != RI_EXT_FORMAT_MOVES) {
		fprintf(stderr, "[E::%s] index format mismatch (expected PEAKS/MOVES, got %d)\n", __func__, (int)idx->format);
		return NULL;
	}

	khash_t(ext_peaks) *batch = kh_init(ext_peaks);
	char *line = (char*)malloc(REXTDATA_LINE_BUF_SIZE);

	for (uint32_t i = 0; i < n_ids; i++) {
		if (!read_ids[i]) continue;
		khiter_t k = kh_get(ext_offsets, idx->offsets, read_ids[i]);
		if (k == kh_end(idx->offsets)) continue;  /* read not in file — silently skip */

		int64_t off = kh_val(idx->offsets, k);
		if (!read_line_at_offset(idx->fp, off, line)) continue;

		if (idx->format == RI_EXT_FORMAT_PEAKS)
			parse_peaks_line_into(batch, line);
		else
			parse_moves_line_into(batch, line);
	}

	free(line);
	return (ri_ext_peaks_t*)batch;
}

ri_ext_events_t *ri_batch_load_ext_events(ri_ext_events_index_t *idx_v, char *const *read_ids, uint32_t n_ids)
{
	if (!idx_v) return NULL;
	ri_ext_index_t *idx = (ri_ext_index_t*)idx_v;
	if (idx->format != RI_EXT_FORMAT_EVENTS) {
		fprintf(stderr, "[E::%s] index format mismatch (expected EVENTS, got %d)\n", __func__, (int)idx->format);
		return NULL;
	}

	khash_t(ext_events) *batch = kh_init(ext_events);
	char *line = (char*)malloc(REXTDATA_LINE_BUF_SIZE);

	for (uint32_t i = 0; i < n_ids; i++) {
		if (!read_ids[i]) continue;
		khiter_t k = kh_get(ext_offsets, idx->offsets, read_ids[i]);
		if (k == kh_end(idx->offsets)) continue;

		int64_t off = kh_val(idx->offsets, k);
		if (!read_line_at_offset(idx->fp, off, line)) continue;

		parse_events_line_into(batch, line);
	}

	free(line);
	return (ri_ext_events_t*)batch;
}

void ri_destroy_ext_peaks_index(ri_ext_peaks_index_t *idx_v)
{
	if (!idx_v) return;
	ri_ext_index_t *idx = (ri_ext_index_t*)idx_v;
	if (idx->fp) fclose(idx->fp);
	if (idx->offsets) {
		khash_t(ext_offsets) *ht = idx->offsets;
		khiter_t k;
		for (k = kh_begin(ht); k != kh_end(ht); ++k) {
			if (!kh_exist(ht, k)) continue;
			free((char*)kh_key(ht, k));
		}
		kh_destroy(ext_offsets, ht);
	}
	if (idx->fname) free(idx->fname);
	free(idx);
}

void ri_destroy_ext_events_index(ri_ext_events_index_t *idx_v)
{
	/* Same internal layout as the peaks index; reuse the destructor. */
	ri_destroy_ext_peaks_index((ri_ext_peaks_index_t*)idx_v);
}
