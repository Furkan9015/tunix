#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "rawhash.h"
#include "ketopt.h"
#include "rextdata.h"
#ifndef NGRPCRH
#include "rlive.h"
#endif

#define RH_VERSION "2.5"

static ko_longopt_t long_options[] = {
	{ (char*)"level_column",     	ko_required_argument, 	300 },
	{ (char*)"q-mid-occ",     		ko_required_argument, 	301 },
	{ (char*)"mid_occ_frac",     	ko_required_argument, 	302 },
	{ (char*)"min-events",			ko_required_argument, 	303 },
	{ (char*)"bw",					ko_required_argument, 	304 },
	{ (char*)"max-target-gap",		ko_required_argument, 	305 },
	{ (char*)"max-query-gap",		ko_required_argument, 	306 },
	{ (char*)"min-anchors",			ko_required_argument, 	307 },
	{ (char*)"min-score",			ko_required_argument, 	308 },
	{ (char*)"chain-gap-scale",		ko_required_argument, 	309 },
	{ (char*)"chain-skip-scale",	ko_required_argument, 	310 },
	{ (char*)"best-chains",			ko_required_argument, 	311 },
	{ (char*)"primary-ratio",		ko_required_argument, 	312 },
	{ (char*)"primary-length",		ko_required_argument, 	313 },
	{ (char*)"max-skips",			ko_required_argument, 	314 },
	{ (char*)"max-iterations",		ko_required_argument, 	315 },
	{ (char*)"rmq",					ko_no_argument, 	  	316 },
	{ (char*)"rmq-inner-dist",		ko_required_argument, 	317 },
	{ (char*)"rmq-size-cap",		ko_required_argument, 	318 },
	{ (char*)"bw-long",				ko_required_argument, 	319 },
	{ (char*)"max-chunks",			ko_required_argument, 	320 },
	{ (char*)"min-mapq",			ko_required_argument, 	321 },
	{ (char*)"alt-drop",			ko_required_argument, 	322 },
	{ (char*)"w-besta",				ko_required_argument, 	323 },
	{ (char*)"w-bestma",			ko_required_argument, 	324 },
	{ (char*)"w-bestq",				ko_required_argument, 	325 },
	{ (char*)"w-bestmq",			ko_required_argument, 	326 },
	{ (char*)"w-bestmc",			ko_required_argument, 	327 },
	{ (char*)"w-threshold",			ko_required_argument, 	328 },
	{ (char*)"bp-per-sec",			ko_required_argument, 	329 },
	{ (char*)"sample-rate",			ko_required_argument, 	330 },
	{ (char*)"chunk-size",			ko_required_argument, 	331 },
	{ (char*)"seg-window-length1",	ko_required_argument, 	332 },
	{ (char*)"seg-window-length2",	ko_required_argument, 	333 },
	{ (char*)"seg-threshold1",		ko_required_argument, 	334 },
	{ (char*)"seg-threshold2",		ko_required_argument, 	335 },
	{ (char*)"seg-peak-height",		ko_required_argument, 	336 },
	{ (char*)"sequence-until",     	ko_no_argument,       	337 },
	{ (char*)"threshold",			ko_required_argument, 	338 },
	{ (char*)"n-samples",			ko_required_argument, 	339 },
	{ (char*)"test-frequency",		ko_required_argument, 	340 },
	{ (char*)"min-reads",			ko_required_argument, 	341 },
	{ (char*)"occ-frac",			ko_required_argument, 	342 },
	{ (char*)"depletion",			ko_no_argument, 	  	343 },
	{ (char*)"store-sig",			ko_no_argument, 	  	344 },
	{ (char*)"sig-target",			ko_no_argument, 	  	345 },
	{ (char*)"disable-adaptive",	ko_no_argument, 	  	346 },
	{ (char*)"sig-diff",			ko_required_argument,	347 },
	{ (char*)"align",				ko_no_argument, 	  	348 },
	{ (char*)"dtw-evaluate-chains",	ko_no_argument,		  	349 },
	{ (char*)"dtw-output-cigar",	ko_no_argument,		  	350 },
	{ (char*)"dtw-border-constraint", ko_required_argument,	351 },
	{ (char*)"dtw-log-scores",		ko_no_argument,			352 },
	{ (char*)"no-chainingscore-filtering",	ko_no_argument,	353 },
	{ (char*)"dtw-match-bonus",		ko_required_argument,	354 },
	{ (char*)"output-chains",		ko_no_argument,			355 },
	{ (char*)"dtw-fill-method",		ko_required_argument,	356 },
	{ (char*)"dtw-min-score", 		ko_required_argument, 	357 },
	{ (char*)"log-anchors",			ko_no_argument,			358 },
	{ (char*)"log-num-anchors",		ko_no_argument,			359 },
	{ (char*)"rev-collision-count", ko_required_argument, 	360 },
	{ (char*)"chn-rev-bump", 		ko_required_argument, 	361 },
	{ (char*)"rev-query",			ko_no_argument, 		362 },
	{ (char*)"r10",					ko_no_argument, 		363 },
	{ (char*)"fine-min",			ko_required_argument, 	364 },
	{ (char*)"fine-max",			ko_required_argument, 	365 },
	{ (char*)"fine-range",			ko_required_argument, 	366 },
	{ (char*)"out-quantize",		ko_no_argument,  		367 },
	{ (char*)"no-event-detection",	ko_no_argument,  		368 },
	{ (char*)"io-thread",			ko_required_argument, 	369 },
	{ (char*)"min-score2",			ko_required_argument, 	370 },
	{ (char*)"version",				ko_no_argument, 	  	371 },
	{ (char*)"peaks-file",			ko_required_argument, 	372 },
	{ (char*)"events-file",			ko_required_argument, 	373 },
	{ (char*)"moves-file",			ko_required_argument, 	374 },
	{ (char*)"min-seg-length",		ko_required_argument, 	375 },
	{ (char*)"max-seg-length",		ko_required_argument, 	376 },
	{ (char*)"no-index-ext-data",	ko_no_argument,       	392 },
#ifndef NGRPCRH
	{ (char*)"live",				ko_no_argument,       	377 },
	{ (char*)"live-host",			ko_required_argument, 	378 },
	{ (char*)"live-port",			ko_required_argument, 	379 },
	{ (char*)"live-first-channel",	ko_required_argument, 	380 },
	{ (char*)"live-last-channel",	ko_required_argument, 	381 },
	{ (char*)"live-tls",			ko_no_argument,       	382 },
	{ (char*)"live-tls-cert",		ko_required_argument, 	383 },
	{ (char*)"live-duration",		ko_required_argument, 	384 },
	{ (char*)"live-debug",			ko_no_argument,       	385 },
	{ (char*)"live-no-sig-filter",	ko_no_argument,       	386 },
	{ (char*)"live-uncalibrated",	ko_no_argument,       	387 },
#endif
	{ (char*)"skip-first-events",	ko_required_argument, 	388 },
	{ (char*)"debug-read",			ko_required_argument, 	389 },
	// removed: quant-sig-diff (390) — merged into --sig-diff
	{ (char*)"n-buckets",			ko_required_argument,	391 },
	{ 0, 0, 0 }
};

static inline int64_t mm_parse_num(const char *str)
{
	double x;
	char *p;
	x = strtod(str, &p);
	if (*p == 'G' || *p == 'g') x *= 1e9, ++p;
	else if (*p == 'M' || *p == 'm') x *= 1e6, ++p;
	else if (*p == 'K' || *p == 'k') x *= 1e3, ++p;
	return (int64_t)(x + .499);
}

static inline void yes_or_no(ri_mapopt_t *opt, int64_t flag, int long_idx, const char *arg, int yes_to_set)
{
	if (yes_to_set) {
		if (strcmp(arg, "yes") == 0 || strcmp(arg, "y") == 0) opt->flag |= flag;
		else if (strcmp(arg, "no") == 0 || strcmp(arg, "n") == 0) opt->flag &= ~flag;
		else fprintf(stderr, "[WARNING]\033[1;31m option '--%s' only accepts 'yes' or 'no'.\033[0m\n", long_options[long_idx].name);
	} else {
		if (strcmp(arg, "yes") == 0 || strcmp(arg, "y") == 0) opt->flag &= ~flag;
		else if (strcmp(arg, "no") == 0 || strcmp(arg, "n") == 0) opt->flag |= flag;
		else fprintf(stderr, "[WARNING]\033[1;31m option '--%s' only accepts 'yes' or 'no'.\033[0m\n", long_options[long_idx].name);
	}
}

static void apply_r10_chemistry(ri_idxopt_t *io, ri_mapopt_t *mo)
{
	io->k = 9;

	io->window_length1 = 3; io->window_length2 = 6;
	io->threshold1 = 5.5f; io->threshold2 = 3.5f;
	io->peak_height = 0.2f;

	mo->window_length1 = 3; mo->window_length2 = 6;
	mo->threshold1 = 5.5f; mo->threshold2 = 3.5f;
	mo->peak_height = 0.2f;

	mo->chain_gap_scale = 1.2f;

	io->bp_per_sec = 400;
	mo->bp_per_sec = 400;
	io->sample_rate = 5000; io->sample_per_base = (float)io->sample_rate / io->bp_per_sec;
	mo->sample_rate = 5000; mo->sample_per_base = (float)mo->sample_rate / mo->bp_per_sec;
}

static void apply_r10_sensitive_quant(ri_idxopt_t *io)
{
	io->n_buckets = 8;
	io->q = 3;
	io->fine_min = -2.0f;
	io->fine_max = 2.0f;
	io->fine_range = 0.75f;
}

static void apply_r10fast_quant(ri_idxopt_t *io)
{
	io->n_buckets = 12;
	io->q = 4;
	io->fine_min = -1.6f;
	io->fine_max = 1.6f;
	io->fine_range = 0.6f;
}

int ri_set_opt(const char *preset, ri_idxopt_t *io, ri_mapopt_t *mo)
{
	if (preset == 0) {
		ri_idxopt_init(io);
		ri_mapopt_init(mo);
	} else if (strcmp(preset, "viral") == 0) {
		io->e = 6;
		mo->bw = 100; mo->max_target_gap_length = 500; mo->max_query_gap_length = 500;
		mo->max_num_chunk = 5, mo->min_chaining_score = 10; mo->chain_gap_scale = 1.2f; mo->chain_skip_scale = 0.3f;
	} else if (strcmp(preset, "sensitive") == 0) {
		//default
	} else if (strcmp(preset, "fast") == 0) {
		io->n_buckets = 12;
		io->q = 4;
		io->fine_min = -1.5f;
		io->fine_max = 1.5f;
		io->fine_range = 0.6f;
		mo->min_mapq = 5, mo->min_chaining_score = 10, mo->chain_gap_scale = 0.6f;
	} else if (strcmp(preset, "r10") == 0) {
		apply_r10_chemistry(io, mo);
		apply_r10_sensitive_quant(io);
	} else if (strcmp(preset, "r10fast") == 0) {
		mo->min_mapq = 5;
		mo->min_chaining_score = 10;
		apply_r10_chemistry(io, mo);
		apply_r10fast_quant(io);
	} else if (strcmp(preset, "r10ava") == 0) {
		ri_set_opt("ava", io, mo);
		apply_r10_chemistry(io, mo);
		apply_r10_sensitive_quant(io);
	} else if (strcmp(preset, "r10ava-sensitive") == 0) {
		ri_set_opt("ava-sensitive", io, mo);
		apply_r10_chemistry(io, mo);
		apply_r10_sensitive_quant(io);
	} else if (strcmp(preset, "r10ava-viral") == 0) {
		ri_set_opt("ava-viral", io, mo);
		apply_r10_chemistry(io, mo);
		apply_r10_sensitive_quant(io);
	} else if (strcmp(preset, "r10ava-large") == 0) {
		ri_set_opt("ava-large", io, mo);
		apply_r10_chemistry(io, mo);
		apply_r10_sensitive_quant(io);
	} else if (strcmp(preset, "faster") == 0) {
		io->e = 11; io->w = 3;
		mo->max_num_chunk = 5; mo->min_mapq = 5, mo->min_chaining_score = 10, mo->chain_gap_scale = 0.6f;
	} else if (strcmp(preset, "ava-viral") == 0) {
		io->e = 6;
		mo->chain_gap_scale = 1.2f; mo->chain_skip_scale = 0.3f;

		io->w = 0;
		io->diff = 0.45f;
		mo->min_chaining_score = 20;
		mo->min_chaining_score2 = 30;
		mo->min_num_anchors = 5;
		mo->min_mapq = 5;
		mo->bw = 1000;
		mo->max_target_gap_length = 2500;
		mo->max_query_gap_length = 2500;

		io->flag |= RI_I_SIG_TARGET;
		mo->flag |= RI_M_ALL_CHAINS;
		mo->flag |= RI_M_NO_ADAPTIVE;

		mo->pri_ratio = 0.0f;
	} else if (strcmp(preset, "ava") == 0) {
		//default
		io->w = 3;
		io->diff = 0.45f;
		mo->min_chaining_score = 40;
		mo->min_chaining_score2 = 75;
		mo->min_num_anchors = 5;
		mo->min_mapq = 5;
		mo->bw = 5000;
		mo->max_target_gap_length = 2500;
		mo->max_query_gap_length = 2500;

		// mo->min_mid_occ = 5000;

		io->flag |= RI_I_SIG_TARGET;
		mo->flag |= RI_M_ALL_CHAINS;
		mo->flag |= RI_M_NO_ADAPTIVE;

		mo->pri_ratio = 0.0f;
	} else if (strcmp(preset, "ava-sensitive") == 0) {
		//default
		io->w = 0;
		io->diff = 0.45f;
		mo->min_chaining_score = 75;
		mo->min_chaining_score2 = 100;
		mo->min_num_anchors = 5;
		mo->min_mapq = 5;
		mo->bw = 1000;
		mo->max_target_gap_length = 2500;
		mo->max_query_gap_length = 2500;

		// mo->min_mid_occ = 10000;

		io->flag |= RI_I_SIG_TARGET;
		mo->flag |= RI_M_ALL_CHAINS;
		mo->flag |= RI_M_NO_ADAPTIVE;

		mo->pri_ratio = 0.0f;
	} else if (strcmp(preset, "ava-large") == 0) {
		mo->chain_gap_scale = 0.6f;

		io->w = 5;
		io->diff = 0.45f;
		mo->min_chaining_score = 20;
		mo->min_chaining_score2 = 50;
		mo->min_num_anchors = 2;
		mo->min_mapq = 2;
		mo->bw = 5000;
		mo->max_target_gap_length = 2500;
		mo->max_query_gap_length = 2500;

		// mo->min_mid_occ = 10000;

		io->flag |= RI_I_SIG_TARGET;
		mo->flag |= RI_M_ALL_CHAINS;
		mo->flag |= RI_M_NO_ADAPTIVE;

		mo->pri_ratio = 0.0f;
	} else if (strcmp(preset, "sequence-until") == 0) {
		//default
	} else return -1;
	return 0;
}

int ri_mapopt_parse_dtw_border_constraint(ri_mapopt_t *opt, char* arg){
	if(strcmp(arg, "global") == 0){
		opt->dtw_border_constraint = RI_M_DTW_BORDER_CONSTRAINT_GLOBAL;
	}
	else if(strcmp(arg, "sparse") == 0){
		opt->dtw_border_constraint = RI_M_DTW_BORDER_CONSTRAINT_SPARSE;
	}
	else if(strcmp(arg, "local") == 0){
		opt->dtw_border_constraint = RI_M_DTW_BORDER_CONSTRAINT_LOCAL;
	}
	else{
		return -1;
	}
	return 0;
}

int ri_mapopt_parse_dtw_fill_method(ri_mapopt_t *opt, char* arg) {
    if (strcmp(arg, "banded") == 0) {
        opt->dtw_fill_method = RI_M_DTW_FILL_METHOD_BANDED;
    } else if (strcmp(arg, "full") == 0) {
        opt->dtw_fill_method = RI_M_DTW_FILL_METHOD_FULL;
    } else if (strncmp(arg, "banded=", 7) == 0) {
        opt->dtw_fill_method = RI_M_DTW_FILL_METHOD_BANDED;
        opt->dtw_band_radius_frac = atof(arg + 7);
    } else {
        return -1;
    }
    return 0;
}

const char* ri_maptopt_dtw_mode_to_string(uint32_t dtw_border_constraint){
	switch(dtw_border_constraint){
		case RI_M_DTW_BORDER_CONSTRAINT_GLOBAL:
			return "full";
		case RI_M_DTW_BORDER_CONSTRAINT_SPARSE:
			return "sparse";
		case RI_M_DTW_BORDER_CONSTRAINT_LOCAL:
			return "window";
		default:
			return "unknown";
	}
}

int main(int argc, char *argv[])
{
	const char *opt_str = "k:d:p:e:q:w:n:o:t:K:x:h";
	ketopt_t o = KETOPT_INIT;
	ri_mapopt_t opt;
  	ri_idxopt_t ipt;
	int c, n_threads = 3, io_n_threads = 1;
	// int n_parts;
	char *fnw = 0, *fpore = 0, *fpeaks = 0, *fevents = 0, *fmoves = 0, *s;
	int load_ext_data_full = 0; // --no-index-ext-data: 1 = legacy full upfront load; 0 = indexed batch loading (default)
	FILE *fp_help = stderr;
	ri_idx_reader_t *idx_rdr;
	ri_idx_t *ri;

	ri_verbose = 3;
	liftrlimit();
	ri_realtime0 = ri_realtime();
	ri_set_opt(0, &ipt, &opt);

#ifndef NGRPCRH
	int live_mode = 0;
	ri_live_opt_t live_opt;
	ri_live_opt_init(&live_opt);
#endif

	while ((c = ketopt(&o, argc, argv, 1, opt_str, long_options)) >= 0) {
		if (c == 'x') {
			if (ri_set_opt(o.arg, &ipt, &opt) < 0) {
				fprintf(stderr, "[ERROR] unknown preset '%s'\n", o.arg);
				return 1;
			}
		} else if (c == ':') {
			fprintf(stderr, "[ERROR] missing option argument\n");
			return 1;
		} else if (c == '?') {
			fprintf(stderr, "[ERROR] unknown option in \"%s\"\n", argv[o.i - 1]);
			return 1;
		}
	}
	o = KETOPT_INIT;

	while ((c = ketopt(&o, argc, argv, 1, opt_str, long_options)) >= 0) {
		if (c == 363) { // --r10: backward-compat alias for `-x r10`
			ri_set_opt("r10", &ipt, &opt);
		} else if (c == 374) { // --moves-file: auto-set defaults (user can override in second pass)
			opt.skip_first_events = 1;
			ipt.diff = -1; // disable sig-diff filtering: move tables give best accuracy without it
		} else if (c == 372) { // --peaks-file: auto-set defaults (user can override in second pass)
			ipt.diff = -1; // disable sig-diff filtering: peaks give 1:1 event-to-base
		}
	}
	o = KETOPT_INIT;

	int q_set_by_user = 0, n_buckets_set_by_user = 0;
	while ((c = ketopt(&o, argc, argv, 1, opt_str, long_options)) >= 0) {
		if (c == 'd') fnw = o.arg;
		else if (c == 'p') fpore = o.arg;
		else if (c == 'k') ipt.k = atoi(o.arg);
		else if (c == 'e') ipt.e = atoi(o.arg);
		else if (c == 'q') { ipt.q = atoi(o.arg); q_set_by_user = 1; }
		else if (c == 'w') ipt.w = atoi(o.arg);
		else if (c == 'n') ipt.n = atoi(o.arg);
		else if (c == 't') n_threads = atoi(o.arg);
		else if (c == 'v') ri_verbose = atoi(o.arg);
		else if (c == 'K') {opt.mini_batch_size = mm_parse_num(o.arg);}
		else if (c == 'h') fp_help = stdout;
		else if (c == 'o') {
			if (strcmp(o.arg, "-") != 0) {
				if (freopen(o.arg, "wb", stdout) == NULL) {
					fprintf(stderr, "[ERROR]\033[1;31m failed to write the output to file '%s'\033[0m: %s\n", o.arg, strerror(errno));
					exit(1);
				}
			}
		}
		else if (c == 300) ipt.lev_col = atoi(o.arg);// --level_column
		else if (c == 301) { //--q-mid-occ
			opt.min_mid_occ = strtol(o.arg, &s, 10); // min
			if (*s == ',') opt.max_mid_occ = strtol(s + 1, &s, 10); //max
			// opt.q_mid_occ = atoi(o.arg);// --q-mid-occ
		}
		else if (c == 302) opt.mid_occ_frac = atof(o.arg);// --occ-frac
		else if (c == 303) opt.min_events = (uint32_t)atoi(o.arg); // --min-events
		else if (c == 304) opt.bw = atoi(o.arg);// --bw
		else if (c == 305) opt.max_target_gap_length = atoi(o.arg);// --max-target-gap
		else if (c == 306) opt.max_query_gap_length = atoi(o.arg);// --max-query-gap
		else if (c == 307) opt.min_num_anchors = atoi(o.arg);// --min-anchors
		else if (c == 308) opt.min_chaining_score = atoi(o.arg);// --min-score
		else if (c == 309) opt.chain_gap_scale = atof(o.arg);// --chain-gap-scale
		else if (c == 310) opt.chain_skip_scale = atof(o.arg);// --chain-skip-scale
		else if (c == 311) opt.best_n = atoi(o.arg);// --best-chains
		else if (c == 312) opt.mask_level = atof(o.arg);// --primary-ratio
		else if (c == 313) opt.mask_len = atoi(o.arg);// --primary-length
		else if (c == 314) opt.max_num_skips = atoi(o.arg);// --max-skips
		else if (c == 315) opt.max_chain_iter = atoi(o.arg);// --max-iterations
		else if (c == 316) opt.flag |= RI_M_RMQ; // --rmq
		else if (c == 317) opt.rmq_inner_dist = atoi(o.arg); // --rmq-inner-dist
		else if (c == 318) opt.rmq_size_cap = atoi(o.arg); // --rmq-size-cap
		else if (c == 319) opt.bw_long = atoi(o.arg);// --bw-long
		else if (c == 320) opt.max_num_chunk = atoi(o.arg);// --max-chunks
		else if (c == 321) opt.min_mapq = atoi(o.arg);// --min-mapq
		else if (c == 322) opt.alt_drop = atof(o.arg);// --alt-drop
		else if (c == 323) opt.w_besta = atof(o.arg);// --w-besta
		else if (c == 324) opt.w_bestma = atof(o.arg);// --w-bestma
		else if (c == 325) opt.w_bestq = atof(o.arg);// --w-bestq
		else if (c == 326) opt.w_bestmq = atof(o.arg);// --w-bestmq
		else if (c == 327) opt.w_bestmc = atof(o.arg);// --w-bestmc
		else if (c == 328) opt.w_threshold = atof(o.arg);// --w-threshold
		else if (c == 329) {
			opt.bp_per_sec = atoi(o.arg); opt.sample_per_base = (float)opt.sample_rate / opt.bp_per_sec;
			ipt.bp_per_sec = atoi(o.arg); ipt.sample_per_base = (float)ipt.sample_rate / ipt.bp_per_sec;
		}// --bp-per-sec
		else if (c == 330) {
			opt.sample_rate = atoi(o.arg); opt.sample_per_base = (float)opt.sample_rate / opt.bp_per_sec;
			ipt.sample_rate = atoi(o.arg); ipt.sample_per_base = (float)ipt.sample_rate / ipt.bp_per_sec;
		}// --sample-rate
		else if (c == 331) opt.chunk_size = atoi(o.arg);// --chunk-size
		else if (c == 332) {opt.window_length1 = atoi(o.arg); ipt.window_length1 = atoi(o.arg);}// --seg-window-length1
		else if (c == 333) {opt.window_length2 = atoi(o.arg); ipt.window_length2 = atoi(o.arg);}// --seg-window-length2
		else if (c == 334) {opt.threshold1 = atof(o.arg); ipt.threshold1 = atof(o.arg);}// --seg-threshold1
		else if (c == 335) {opt.threshold2 = atof(o.arg); ipt.threshold2 = atof(o.arg);}// --seg-threshold2
		else if (c == 336) {opt.peak_height = atof(o.arg); ipt.peak_height = atof(o.arg);}// --seg-peak-height
		else if (c == 337) opt.flag |= RI_M_SEQUENCEUNTIL;// --sequence-until
		else if (c == 338) opt.t_threshold = atof(o.arg);// --threshold
		else if (c == 339) opt.tn_samples = atoi(o.arg);// --n-samples
		else if (c == 340) opt.ttest_freq = atoi(o.arg);// --test-frequency
		else if (c == 341) opt.tmin_reads = atoi(o.arg);// --min-reads
		else if (c == 342) opt.mid_occ_frac = atof(o.arg);// --occ-frac
		else if (c == 343) { // --depletion
			opt.best_n = 5; opt.min_mapq = 10; opt.w_threshold = 0.50f;
			opt.min_num_anchors = 2; opt.min_chaining_score = 15; opt.chain_skip_scale = 0.0f;
		}
		else if (c == 344) {ipt.flag |= RI_I_STORE_SIG;} // --store-sig
		else if (c == 345) {ipt.flag |= RI_I_SIG_TARGET;} // --sig-target
		else if (c == 346) {opt.flag |= RI_M_NO_ADAPTIVE;} // --disable-adaptive
		else if (c == 347) {ipt.diff = atoi(o.arg);} // --sig-diff INT
		else if (c == 348) {opt.flag |= RI_M_ALIGN;} // --align
		else if (c == 349) opt.flag |= RI_M_DTW_EVALUATE_CHAINS; // --dtw-evaluate-chains
		else if (c == 350) opt.flag |= RI_M_DTW_OUTPUT_CIGAR; // --dtw-output-cigar
		else if (c == 351) { //--dtw-border-constraint
			if(ri_mapopt_parse_dtw_border_constraint(&opt, o.arg) != 0){
				fprintf(stderr, "[ERROR] unknown DTW border constraint in \"%s\"\n", argv[o.i - 1]);
				return 1;
			}
		}
		else if (c == 352) opt.flag |= RI_M_DTW_LOG_SCORES; // --dtw-log-scores
		else if (c == 353) opt.flag |= RI_M_DISABLE_CHAININGSCORE_FILTERING; // --no-chainingscore-filtering
		else if (c == 354) opt.dtw_match_bonus = atof(o.arg); // --dtw-match-bonus
		else if (c == 355) opt.flag |= RI_M_OUTPUT_CHAINS; // --output-chains
		else if (c == 356) { //dtw-fill-method
			if(ri_mapopt_parse_dtw_fill_method(&opt, o.arg) != 0){
				fprintf(stderr, "[ERROR] unknown DTW fill method in \"%s\"\n", argv[o.i - 1]);
				return 1;
			}
		}
		else if (c == 357) opt.dtw_min_score = atof(o.arg); // --dtw-min-score
		else if (c == 358) opt.flag |= RI_M_LOG_ANCHORS; // --log-anchors
		else if (c == 359) opt.flag |= RI_M_LOG_NUM_ANCHORS; // --log-num-anchors
		else if (c == 360) opt.rev_col_limit = atoi(o.arg); // --rev-collision-count
		else if (c == 361) opt.chn_rev_bump = atof(o.arg); // --chn-rev-bump
		// else if (c == 362) {ipt.flag |= RI_I_REV_QUERY;}// --rev-query
		else if (c == 364) {ipt.fine_min = atof(o.arg);}// --fine-min
		else if (c == 365) {ipt.fine_max = atof(o.arg);}// --fine-max
		else if (c == 366) {ipt.fine_range = atof(o.arg);}// --fine-range
		else if (c == 367) {ipt.flag |= RI_I_OUT_QUANTIZE; ipt.flag |= RI_I_SIG_TARGET;}// --out-quantize
		else if (c == 368) {ipt.flag |= RI_I_NO_EVENT_DETECTION;}// --no-event-detection
		else if (c == 369) {io_n_threads = atoi(o.arg);}// --io-thread
		else if (c == 370) opt.min_chaining_score2 = atoi(o.arg);// --min-score2
		else if (c == 371) {puts(RH_VERSION); return 0;}// --version
		else if (c == 372) fpeaks = o.arg; // --peaks-file
		else if (c == 373) fevents = o.arg; // --events-file
		else if (c == 374) fmoves = o.arg; // --moves-file
		else if (c == 375) opt.min_segment_length = (uint32_t)atoi(o.arg); // --min-seg-length
		else if (c == 376) opt.max_segment_length = (uint32_t)atoi(o.arg); // --max-seg-length
		else if (c == 392) load_ext_data_full = 1; // --no-index-ext-data: legacy full upfront load
#ifndef NGRPCRH
		else if (c == 377) live_mode = 1; // --live
		else if (c == 378) live_opt.host = o.arg; // --live-host
		else if (c == 379) live_opt.port = atoi(o.arg); // --live-port
		else if (c == 380) live_opt.first_channel = (uint32_t)atoi(o.arg); // --live-first-channel
		else if (c == 381) live_opt.last_channel = (uint32_t)atoi(o.arg); // --live-last-channel
		else if (c == 382) live_opt.use_tls = 1; // --live-tls
		else if (c == 383) live_opt.tls_cert_path = o.arg; // --live-tls-cert
		else if (c == 384) live_opt.duration_seconds = atoi(o.arg); // --live-duration
		else if (c == 385) live_opt.debug = 1; // --live-debug
		else if (c == 386) live_opt.no_sig_filter = 1; // --live-no-sig-filter
		else if (c == 387) live_opt.uncalibrated = 1; // --live-uncalibrated
#endif
		else if (c == 388) opt.skip_first_events = (uint32_t)atoi(o.arg); // --skip-first-events
		else if (c == 389) opt.debug_read = o.arg; // --debug-read
		// removed: case 390 (quant-sig-diff) — merged into --sig-diff (case 347)
		else if (c == 391) { ipt.n_buckets = atoi(o.arg); n_buckets_set_by_user = 1; } // --n-buckets
		else if (c == 'V') {puts(RH_VERSION); return 0;}
	}

	/* Reconcile -q (bits) and --n-buckets (count). They cannot both be user-set. */
	if (q_set_by_user && n_buckets_set_by_user) {
		fprintf(stderr, "[ERROR] -q and --n-buckets are mutually exclusive: pick one (or neither, in which case the default n_buckets=%d is used).\n", ipt.n_buckets);
		return 1;
	}
	if (n_buckets_set_by_user) {
		if (ipt.n_buckets < 2 || ipt.n_buckets > (1<<7)) {
			fprintf(stderr, "[ERROR] --n-buckets must be in [2, %d]; got %d.\n", 1<<7, ipt.n_buckets);
			return 1;
		}
		/* Derive q = ceil(log2(n_buckets)). */
		int q = 0;
		while ((1 << q) < ipt.n_buckets) ++q;
		ipt.q = q;
	} else if (q_set_by_user) {
		if (ipt.q < 1 || ipt.q > 7) {
			fprintf(stderr, "[ERROR] -q must be in [1, 7]; got %d.\n", ipt.q);
			return 1;
		}
		ipt.n_buckets = 1 << ipt.q;
	}
	/* Hash-budget constraint: e * q must fit in (64 - RI_HASH_SHIFT) = 58 bits. */
	if (ipt.e * ipt.q > 64 - 6) {
		fprintf(stderr, "[ERROR] e * q exceeds the hash budget (64 - RI_HASH_SHIFT = 58): e=%d, q=%d, product=%d.\n", ipt.e, ipt.q, ipt.e * ipt.q);
		return 1;
	}

	if ((fpeaks != 0) + (fevents != 0) + (fmoves != 0) > 1) {
		fprintf(stderr, "[ERROR] --peaks-file, --events-file, and --moves-file are mutually exclusive. Specify only one.\n");
		return 1;
	}

	if (argc == o.ind || fp_help == stdout) {
		fprintf(fp_help, "Usage: rawhash2 [options] <target.fa>|<target.idx> [query.fast5|.pod5|.slow5] [...]\n");
		fprintf(fp_help, "Options:\n");

		fprintf(fp_help, "  General:\n");
		fprintf(fp_help, "    -h           show this help message\n");
		fprintf(fp_help, "    --version    show version number\n");

		fprintf(fp_help, "\n  Pore Model:\n");
		fprintf(fp_help, "    -p FILE      pore model file []\n");
		fprintf(fp_help, "    -k INT       k-mer size in the pore model [%d]. Typically 6 for R9.4, 9 for R10\n", ipt.k);
		fprintf(fp_help, "    --level_column INT   0-based column for mean values in pore file [%d]\n", ipt.lev_col);

		fprintf(fp_help, "\n  Indexing:\n");
		fprintf(fp_help, "    -d FILE      dump index to FILE (strongly recommended before mapping) []\n");
		fprintf(fp_help, "    -e INT       events per hash value [%d]. Also applies during mapping\n", ipt.e);
		fprintf(fp_help, "    -q INT       quantization bits [%d]. Creates 2^INT buckets. Mutually exclusive with --n-buckets.\n", ipt.q);
		fprintf(fp_help, "    --n-buckets INT   number of distinct quantization buckets [%d]. q is derived as ceil(log2(N)). Mutually exclusive with -q.\n", ipt.n_buckets);
		fprintf(fp_help, "    -w INT       minimizer window size [%d]. >0 enables minimizer seeding (faster, less accurate)\n", ipt.w);
		fprintf(fp_help, "    --sig-diff INT     skip events if quantized difference <= INT [%d]. -1 disables filtering (auto-set to -1 with --moves-file/--peaks-file)\n", ipt.diff);
		fprintf(fp_help, "    --store-sig  store reference signal in index (required for DTW alignment)\n");
		fprintf(fp_help, "    --sig-target reference contains signal values instead of bases (for overlapping)\n");

		fprintf(fp_help, "\n  Seeding:\n");
		fprintf(fp_help, "    --q-mid-occ INT1[,INT2]   k-mer occurrence bounds [%d,%d]\n", opt.min_mid_occ, opt.max_mid_occ);

		fprintf(fp_help, "\n  Chaining:\n");
		fprintf(fp_help, "    --min-events INT       min events per chunk before chaining [%u]\n", opt.min_events);
		fprintf(fp_help, "    --bw INT               max gap in chain [%d]\n", opt.bw);
		fprintf(fp_help, "    --max-target-gap INT   max reference gap in chain [%d]\n", opt.max_target_gap_length);
		fprintf(fp_help, "    --max-query-gap INT    max query gap in chain [%d]\n", opt.max_query_gap_length);
		fprintf(fp_help, "    --min-anchors INT      min anchors per chain [%d]\n", opt.min_num_anchors);
		fprintf(fp_help, "    --min-score INT        min chain score [%d]\n", opt.min_chaining_score);
		fprintf(fp_help, "    --best-chains INT      secondary chains to keep [%d]\n", opt.best_n);
		fprintf(fp_help, "    --chain-gap-scale FLOAT     gap penalty scale [%g]\n", opt.chain_gap_scale);
		fprintf(fp_help, "    --chain-skip-scale FLOAT    skip penalty scale [%g]\n", opt.chain_skip_scale);
		fprintf(fp_help, "    --primary-ratio FLOAT  [Advanced] primary chain coverage ratio [%g]\n", opt.mask_level);
		fprintf(fp_help, "    --primary-length INT   [Advanced] primary chain coverage length [%d]\n", opt.mask_len);
		fprintf(fp_help, "    --max-skips INT        [Advanced] stop after INT iterations w/o improvement [%d]\n", opt.max_num_skips);
		fprintf(fp_help, "    --max-iterations INT   [Advanced] max predecessor anchors to check [%d]\n", opt.max_chain_iter);
		fprintf(fp_help, "    --rmq                  [Advanced] use RMQ chaining (faster, less accurate)\n");
		fprintf(fp_help, "    --rmq-inner-dist INT   [Advanced] RMQ inner distance [%d]\n", opt.rmq_inner_dist);
		fprintf(fp_help, "    --rmq-size-cap INT     [Advanced] RMQ cap size [%d]\n", opt.rmq_size_cap);
		fprintf(fp_help, "    --bw-long INT          [Advanced] long gap re-chaining threshold (>--bw to enable) [%d]\n", opt.bw_long);

		fprintf(fp_help, "\n  Mapping:\n");
		fprintf(fp_help, "    --max-chunks INT       stop after INT chunks if unmapped [%u]\n", opt.max_num_chunk);
		fprintf(fp_help, "    --min-mapq INT         report mapping if MAPQ > INT [%d]\n", opt.min_mapq);
		fprintf(fp_help, "    --disable-adaptive     process entire signal (no early stopping)\n");

		fprintf(fp_help, "\n  Signal Alignment (DTW, as introduced in RawAlign):\n");
		fprintf(fp_help, "    --dtw-evaluate-chains        score chains using DTW alignment [%s]\n", opt.flag & RI_M_DTW_EVALUATE_CHAINS? "yes" : "no");
		fprintf(fp_help, "                                 (index must be built with --store-sig)\n");
		fprintf(fp_help, "    --dtw-output-cigar           include CIGAR string in output [%s]\n", opt.flag & RI_M_DTW_OUTPUT_CIGAR? "yes" : "no");
		fprintf(fp_help, "    --dtw-border-constraint STR  alignment scope: global, sparse, local [%s]\n", ri_maptopt_dtw_mode_to_string(opt.dtw_border_constraint));
		fprintf(fp_help, "    --dtw-fill-method STR        matrix computation: full, banded[=FRAC] [banded]\n");
		fprintf(fp_help, "    --dtw-match-bonus FLOAT      match score bonus [%g]\n", opt.dtw_match_bonus);
		fprintf(fp_help, "    --dtw-min-score FLOAT        min DTW score to report alignment [%g]\n", opt.dtw_min_score);

		fprintf(fp_help, "\n  Nanopore Device:\n");
		fprintf(fp_help, "    --bp-per-sec INT       translocation speed in bp/s [%u]\n", opt.bp_per_sec);
		fprintf(fp_help, "    --sample-rate INT      sampling rate in Hz [%u]\n", opt.sample_rate);
		fprintf(fp_help, "    --chunk-size INT       samples per chunk [%u]\n", opt.chunk_size);

		fprintf(fp_help, "\n  Signal Segmentation:\n");
		fprintf(fp_help, "    --seg-window-length1 INT   short detector window (ttest) [%u]\n", opt.window_length1);
		fprintf(fp_help, "    --seg-window-length2 INT   long detector window (ttest) [%u]\n", opt.window_length2);
		fprintf(fp_help, "    --seg-threshold1 FLOAT     peak threshold for short window (ttest) [%g]\n", opt.threshold1);
		fprintf(fp_help, "    --seg-threshold2 FLOAT     peak threshold for long window (ttest) [%g]\n", opt.threshold2);
		fprintf(fp_help, "    --seg-peak-height FLOAT    min peak prominence (ttest) [%g]\n", opt.peak_height);
		fprintf(fp_help, "    --min-seg-length INT       skip segments shorter than INT [%u]\n", opt.min_segment_length);
		fprintf(fp_help, "    --max-seg-length INT       skip segments longer than INT [%u]\n", opt.max_segment_length);

		fprintf(fp_help, "\n  External Segmentation:\n");
		fprintf(fp_help, "    --peaks-file FILE   use pre-computed peak positions from FILE\n");
		fprintf(fp_help, "                        Bypasses t-test peak detection; event generation still runs.\n");
		fprintf(fp_help, "                        Format: read_id<TAB>peak1 peak2 ... peakN\n");
		fprintf(fp_help, "    --events-file FILE  use pre-computed event values from FILE\n");
		fprintf(fp_help, "                        Completely bypasses the event detection pipeline.\n");
		fprintf(fp_help, "                        Format: read_id<TAB>ev1 ev2 ... evN\n");
		fprintf(fp_help, "    --moves-file FILE   use dorado move table data from FILE\n");
		fprintf(fp_help, "                        Converts move boundaries to segmentation peaks.\n");
		fprintf(fp_help, "                        Format: read_id<TAB>mv:B:c,STRIDE,0,1,...<TAB>ts:i:OFFSET\n");
		fprintf(fp_help, "                        Use test/scripts/extract_moves_from_bam.sh to generate this file.\n");
		fprintf(fp_help, "    --skip-first-events INT  skip the first INT events [%u] (auto-set to 1 with --moves-file)\n", opt.skip_first_events);
		fprintf(fp_help, "    --no-index-ext-data  load --peaks-file/--events-file/--moves-file fully into memory at startup\n");
		fprintf(fp_help, "                         instead of building a per-read offset index. The default (no flag) scans\n");
		fprintf(fp_help, "                         the file once to record (read_id -> byte offset) and loads only the reads\n");
		fprintf(fp_help, "                         needed by the current pipeline batch via fseek+parse, freeing each batch\n");
		fprintf(fp_help, "                         after use. Indexing keeps memory bounded for large external files; full\n");
		fprintf(fp_help, "                         load avoids the per-batch fseek overhead but requires the entire parsed\n");
		fprintf(fp_help, "                         dataset to fit in RAM. Both paths produce identical mapping output.\n");
		fprintf(fp_help, "    Note: --peaks-file, --events-file, and --moves-file are mutually exclusive.\n");

		fprintf(fp_help, "\n  Sequence Until (real-time abundance estimation):\n");
		fprintf(fp_help, "    --sequence-until       activate Sequence Until mode\n");
		fprintf(fp_help, "    --threshold FLOAT      outlier distance threshold [%g]\n", opt.t_threshold);
		fprintf(fp_help, "    --n-samples INT        previous estimations to compare [%u]\n", opt.tn_samples);
		fprintf(fp_help, "    --test-frequency INT   re-estimate every INT reads [%u]\n", opt.ttest_freq);
		fprintf(fp_help, "    --min-reads INT        min reads before first estimate [%u]\n", opt.tmin_reads);

#ifndef NGRPCRH
		fprintf(fp_help, "\n  Live Streaming (MinKNOW/Icarust):\n");
		fprintf(fp_help, "    --live                 enable real-time gRPC streaming from MinKNOW/Icarust\n");
		fprintf(fp_help, "    --live-host STR        gRPC server hostname [localhost]\n");
		fprintf(fp_help, "    --live-port INT        gRPC server port [10001]\n");
		fprintf(fp_help, "    --live-first-channel INT  first channel to monitor, 1-indexed [1]\n");
		fprintf(fp_help, "    --live-last-channel INT   last channel to monitor, 1-indexed [512]\n");
		fprintf(fp_help, "    --live-tls             use TLS encryption (for real MinKNOW)\n");
		fprintf(fp_help, "    --live-tls-cert FILE   path to CA certificate for TLS\n");
		fprintf(fp_help, "    --live-duration INT    run for INT seconds, 0 = until experiment ends [0]\n");
		fprintf(fp_help, "    --live-debug           print chunk metadata to stderr (no mapping)\n");
		fprintf(fp_help, "    --live-no-sig-filter   disable 30-200 pA signal filter in streaming\n");
		fprintf(fp_help, "    --live-uncalibrated    request uncalibrated data, apply per-channel cal\n");
#endif

		fprintf(fp_help, "\n  Input/Output:\n");
		fprintf(fp_help, "    -o FILE      output file [stdout]\n");
		fprintf(fp_help, "    -t INT       number of threads [%d]\n", n_threads);
		fprintf(fp_help, "    --io-thread INT   I/O threads for S/BLOW5 (must be < -t) [%d]\n", io_n_threads);
		fprintf(fp_help, "    -K NUM       minibatch size [500M]\n");

		fprintf(fp_help, "\n  Presets:\n");
		fprintf(fp_help, "    -x STR       preset (applied before other options) []\n");
		fprintf(fp_help, "      Mapping presets:\n");
		fprintf(fp_help, "        viral       small viral genomes (<10M)\n");
		fprintf(fp_help, "        sensitive   small genomes (<500M, default accuracy)\n");
		fprintf(fp_help, "        fast        large genomes (500M-5G, faster)\n");
		fprintf(fp_help, "        faster      very large genomes (>5G, uses minimizers)\n");
		fprintf(fp_help, "        r10         R10.x chemistry, sensitive shape (equivalent to -x sensitive --r10)\n");
		fprintf(fp_help, "        r10fast     R10.x chemistry, fast shape (tuned for human R10 references)\n");
		fprintf(fp_help, "      Rawsamble (overlapping) presets:\n");
		fprintf(fp_help, "        ava            all-vs-all overlapping (default for Rawsamble)\n");
		fprintf(fp_help, "        ava-sensitive  more sensitive overlapping\n");
		fprintf(fp_help, "        ava-viral      overlapping for viral genomes\n");
		fprintf(fp_help, "        ava-large      overlapping for large genomes (>10G)\n");
		fprintf(fp_help, "        r10ava            R10.x chemistry, ava shape\n");
		fprintf(fp_help, "        r10ava-sensitive  R10.x chemistry, ava-sensitive shape\n");
		fprintf(fp_help, "        r10ava-viral      R10.x chemistry, ava-viral shape\n");
		fprintf(fp_help, "        r10ava-large      R10.x chemistry, ava-large shape\n");
		fprintf(fp_help, "    --depletion  high-precision mode for contamination/abundance analysis\n");
		fprintf(fp_help, "    --r10        backward-compat alias for `-x r10`\n");

		fprintf(fp_help, "\n  Debug/Logging:\n");
		fprintf(fp_help, "    --out-quantize              output quantized signal values (no mapping performed)\n");
		fprintf(fp_help, "    --no-event-detection        skip t-test segmentation (use with external segmentation data)\n");
		fprintf(fp_help, "    --log-anchors               log seed/anchor positions to stderr\n");
		fprintf(fp_help, "    --log-num-anchors           log anchor counts per read to stderr\n");
		fprintf(fp_help, "    --output-chains             log chain details to stderr\n");
		fprintf(fp_help, "    --debug-read STR            print detailed diagnostics for a specific read\n");
		fprintf(fp_help, "    --dtw-log-scores            log DTW alignment scores to stderr\n");
		fprintf(fp_help, "    --no-chainingscore-filtering   disable chain score filtering\n");

		return fp_help == stdout? 0 : 1;
	}

	if(n_threads < io_n_threads){
		fprintf(stderr, "[ERROR] The overall number of threads (-t [%d]) must NOT be smaller than the number of IO threads (--io-thread [%d).\n", n_threads, io_n_threads);
		return 1;
	}

	if(ipt.w && ipt.n){
		fprintf(stderr, "[ERROR] minimizer window 'w' ('%d') and BLEND 'neighbor' ('%d') values cannot be set together. At least one of them must be zero to enable one of the seeding options: %s\n", ipt.w, ipt.n, strerror(errno));
		return 1;
	}

	idx_rdr = ri_idx_reader_open(argv[o.ind], &ipt, fnw);
	if (idx_rdr == 0) {
		fprintf(stderr, "[ERROR] failed to open file '%s': %s\n", argv[o.ind], strerror(errno));
		return 1;
	}

	if (!idx_rdr->is_idx && fnw == 0 && argc - o.ind < 2 && !(ipt.flag&RI_I_OUT_QUANTIZE)
#ifndef NGRPCRH
		&& !live_mode
#endif
	) {
		fprintf(stderr, "[ERROR] missing input: please specify a query FAST5/SLOW5/POD5 file(s) to map or option -d to store the index in a file before running the mapping\n");
		ri_idx_reader_close(idx_rdr);
		return 1;
	}

	
	ri_pore_t pore;
	pore.pore_vals = NULL;
	pore.pore_inds = NULL;
	pore.max_val = -5000.0;
	pore.min_val = 5000.0;
	if(!(ipt.flag&RI_I_OUT_QUANTIZE)){
		if((!idx_rdr->is_idx && fpore == 0) && !(!(ipt.flag&RI_I_REV_QUERY) && ipt.flag&RI_I_SIG_TARGET)){
			fprintf(stderr, "[ERROR] missing input: please specify a pore model file with -p when generating the index from a sequence file\n");
			ri_idx_reader_close(idx_rdr);
			return 1;
		}else if(!idx_rdr->is_idx && fpore){
			load_pore(fpore, ipt.k, ipt.lev_col, &pore);
			if(!pore.pore_vals){
				fprintf(stderr, "[ERROR] cannot parse the k-mer pore model file. Please see the example k-mer model files provided in the RawHash repository.\n");
				ri_idx_reader_close(idx_rdr);
				return 1;
			}
		}
	}

	/* External segmentation data.
	 * Default path (load_ext_data_full == 0): build a lightweight offset index at startup
	 *   (read_id -> byte offset only); the mapping pipeline loads only the per-batch subset
	 *   from disk via fseek+parse. Bounded memory on huge files (~MB-scale offset hash + 1 batch
	 *   resident at a time).
	 * Legacy path (load_ext_data_full == 1, via --no-index-ext-data): load and parse the entire
	 *   file into a full khash at startup. Faster steady-state mapping but requires the entire
	 *   parsed dataset to fit in RAM. */
	if (fpeaks) {
		if (load_ext_data_full) {
			opt.ext_peaks = ri_load_ext_peaks(fpeaks);
		} else {
			opt.ext_peaks_index = ri_index_ext_peaks(fpeaks);
		}
		if (!opt.ext_peaks && !opt.ext_peaks_index) {
			fprintf(stderr, "[ERROR] failed to %s peaks file '%s'\n",
				load_ext_data_full ? "load" : "index", fpeaks);
			ri_idx_reader_close(idx_rdr);
			return 1;
		}
	}
	if (fevents) {
		if (load_ext_data_full) {
			opt.ext_events = ri_load_ext_events(fevents);
		} else {
			opt.ext_events_index = ri_index_ext_events(fevents);
		}
		if (!opt.ext_events && !opt.ext_events_index) {
			fprintf(stderr, "[ERROR] failed to %s events file '%s'\n",
				load_ext_data_full ? "load" : "index", fevents);
			ri_idx_reader_close(idx_rdr);
			return 1;
		}
	}
	if (fmoves) {
		if (load_ext_data_full) {
			opt.ext_peaks = ri_load_ext_moves(fmoves);
		} else {
			opt.ext_peaks_index = ri_index_ext_moves(fmoves);
		}
		if (!opt.ext_peaks && !opt.ext_peaks_index) {
			fprintf(stderr, "[ERROR] failed to %s moves file '%s'\n",
				load_ext_data_full ? "load" : "index", fmoves);
			ri_idx_reader_close(idx_rdr);
			return 1;
		}
	}

	while ((ri = ri_idx_reader_read(idx_rdr, &pore, n_threads, io_n_threads)) != 0) {
		int ret;
		if (ri_verbose >= 3)
			fprintf(stderr, "[M::%s::%.3f*%.2f] loaded/built the index for %d target sequence(s)\n",
					__func__, ri_realtime() - ri_realtime0, ri_cputime() / (ri_realtime() - ri_realtime0), ri->n_seq);
		if (argc != o.ind + 1
#ifndef NGRPCRH
			|| live_mode
#endif
		) ri_mapopt_update(&opt, ri);
		if (ri_verbose >= 3) ri_idx_stat(ri);
#ifndef NGRPCRH
		if (live_mode) {
			ret = ri_map_live(ri, &opt, &live_opt, n_threads);
		} else
#endif
		if (argc - (o.ind + 1) == 0) {
			fprintf(stderr, "[INFO] No files to query index on. Only the index is constructed.\n");
			ri_idx_destroy(ri);
			continue; // no query files, just creating the index
		} else {
			ret = ri_map_file_frag(ri, argc - (o.ind + 1), (const char**)&argv[o.ind + 1], &opt, n_threads, io_n_threads);
		}
		ri_idx_destroy(ri);
		if (ret < 0) {
			fprintf(stderr, "ERROR: failed to map the query file\n");
			exit(EXIT_FAILURE);
		}
	}
	// n_parts = idx_rdr->n_parts;
	ri_idx_reader_close(idx_rdr);
	if(pore.pore_vals)free(pore.pore_vals);
	if(pore.pore_inds)free(pore.pore_inds);
	/* Free legacy full-load hashes (will be NULL when using indexed batch path) */
	if(opt.ext_peaks) ri_destroy_ext_peaks(opt.ext_peaks);
	if(opt.ext_events) ri_destroy_ext_events(opt.ext_events);
	/* Free the offset indexes (which keep file handles open) */
	if(opt.ext_peaks_index) ri_destroy_ext_peaks_index(opt.ext_peaks_index);
	if(opt.ext_events_index) ri_destroy_ext_events_index(opt.ext_events_index);

	if (fflush(stdout) == EOF) {
		perror("[ERROR] failed to write the results");
		exit(EXIT_FAILURE);
	}

	if (ri_verbose >= 3) {
		fprintf(stderr, "[M::%s] Version: %s\n", __func__, RH_VERSION);
		// fprintf(stderr, "[M::%s] CMD:", __func__);
		// for (i = 0; i < argc; ++i) fprintf(stderr, " %s", argv[i]);
		fprintf(stderr, "\n[M::%s] Real time: %.3f sec; CPU: %.3f sec; Peak RSS: %.3f GB\n", __func__, ri_realtime() - ri_realtime0, ri_cputime(), ri_peakrss() / 1024.0 / 1024.0 / 1024.0);
	}
	return 0;
}
