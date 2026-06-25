#ifndef RLIVE_H
#define RLIVE_H

#include "rsig.h"
#include "roptions.h"
#include "rindex.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Options for live MinKNOW/Icarust gRPC streaming.
 */
typedef struct ri_live_opt_s {
	char *host;              /* MinKNOW/Icarust host [localhost] */
	int port;                /* gRPC port [10001] */
	uint32_t first_channel;  /* first channel to monitor, 1-indexed [1] */
	uint32_t last_channel;   /* last channel to monitor, 1-indexed [512] */
	int use_tls;             /* 0 = insecure, 1 = TLS (for real MinKNOW) */
	char *tls_cert_path;     /* path to CA certificate for TLS */
	int duration_seconds;    /* run for N seconds, 0 = until experiment ends [0] */
	int debug;               /* 1 = print chunk metadata to stderr, no mapping */
	int no_sig_filter;       /* 1 = disable 30-200 pA signal filter [0] */
	int uncalibrated;        /* 1 = request UNCALIBRATED data and apply per-channel cal [0] */
} ri_live_opt_t;

/**
 * Initialize live options with defaults.
 */
void ri_live_opt_init(ri_live_opt_t *opt);

/**
 * Map raw nanopore signals in real-time from a MinKNOW/Icarust gRPC stream.
 *
 * Connects to the MinKNOW DataService.get_live_reads() bidirectional stream,
 * receives calibrated signal chunks per channel, accumulates them, and
 * dispatches to the existing map_worker_for() pipeline for mapping.
 * Results are output as PAF to stdout.
 *
 * @param idx        preloaded reference index
 * @param opt        mapping options (same as file-based mode)
 * @param live_opt   live streaming connection options
 * @param n_threads  total threads (1 for gRPC I/O, rest for mapping)
 * @return           0 on success, -1 on failure
 */
int ri_map_live(const ri_idx_t *idx,
                const ri_mapopt_t *opt,
                const ri_live_opt_t *live_opt,
                int n_threads);

#ifdef __cplusplus
}
#endif

#endif /* RLIVE_H */
