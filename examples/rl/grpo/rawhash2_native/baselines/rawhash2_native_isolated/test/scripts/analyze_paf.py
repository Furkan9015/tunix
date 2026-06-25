#!/usr/bin/env python3
"""
analyze_paf.py — Compute accuracy and timing stats from an annotated PAF.

Reads a PAF file annotated by pafstats.py (with rf:Z:tp/fp/fn/tn/na tags)
and reports TP/FP/FN/TN counts, precision, recall, F1, and timing statistics.

Usage:
    python3 analyze_paf.py rawhash2_ann.paf
"""

import sys
from statistics import median, mean


def parse_tags(fields):
    """Parse PAF optional fields (col 12+) into dict keyed by tag name."""
    tags = {}
    for f in fields:
        parts = f.split(":", 2)
        if len(parts) == 3:
            name, typ, val = parts
            if typ == "i":
                tags[name] = int(val)
            elif typ == "f":
                tags[name] = float(val)
            else:
                tags[name] = val
    return tags


def safe_mean(lst):
    return mean(lst) if lst else 0.0


def safe_median(lst):
    return median(lst) if lst else 0.0


def main():
    if len(sys.argv) < 2:
        print("usage: analyze_paf.py rawhash2_ann.paf")
        sys.exit(1)

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    time_per_chunk = []
    time_per_read = []
    maplast_pos = []
    maplast_chunk = []
    umaplast_pos = []
    umaplast_chunk = []

    with open(sys.argv[1]) as fh:
        for line in fh:
            cols = line.rstrip().split()
            if len(cols) < 13:
                continue

            tags = parse_tags(cols[12:])
            is_mapped = cols[4] != '*'
            lastpos = int(cols[1])

            rf = tags.get("rf", "na")

            # mt:f: = mapping time (ms), ci:i: = chunk index where mapping decision was made
            # (this is the actual number of chunks processed; nc:i: is n_cregs, not chunk count)
            mt = tags.get("mt")
            nc = tags.get("ci")

            if rf != "na":
                if mt is not None:
                    time_per_read.append(mt)
                if is_mapped:
                    maplast_pos.append(lastpos)
                elif lastpos < 100000:
                    umaplast_pos.append(lastpos)

            if nc is not None:
                if is_mapped:
                    maplast_chunk.append(nc)
                elif lastpos < 100000:
                    umaplast_chunk.append(nc)

            if mt is not None and nc is not None and nc > 0:
                time_per_chunk.append(mt / nc)

            if rf == "tp":
                tp += 1
            elif rf == "fp" or rf == "na":
                fp += 1
            elif rf == "fn":
                fn += 1
            elif rf == "tn":
                tn += 1

    print("RawHash2 TP: " + str(tp))
    print("RawHash2 FP: " + str(fp))
    print("RawHash2 FN: " + str(fn))
    print("RawHash2 TN: " + str(tn))

    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0
    print("RawHash2 precision: " + str(precision))

    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0
    print("RawHash2 recall: " + str(recall))

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    print("RawHash2 F-1 score: " + str(f1))

    print("RawHash2 Mean time per chunk : " + str(safe_mean(time_per_chunk)))
    print("RawHash2 Median time per chunk : " + str(safe_median(time_per_chunk)))
    print("RawHash2 Mean time per read : " + str(safe_mean(time_per_read)))
    print("RawHash2 Median time per read : " + str(safe_median(time_per_read)))
    print("RawHash2 Mean time per read (all) : " + str(safe_mean(time_per_read + time_per_chunk)))
    print("RawHash2 Median time per read (all) : " + str(safe_median(time_per_read + time_per_chunk)))
    print("RawHash2 Mean # of sequenced bases per read : " + str(safe_mean(maplast_pos + umaplast_pos)))
    print("RawHash2 Mean # of sequenced chunks per read : " + str(safe_mean(maplast_chunk + umaplast_chunk)))

    print("RawHash2 Mean (only mapped) # of sequenced bases per read : " + str(safe_mean(maplast_pos)))
    print("RawHash2 Mean (only mapped) # of sequenced chunks per read : " + str(safe_mean(maplast_chunk)))

    print("RawHash2 Mean (only unmapped) # of sequenced bases per read : " + str(safe_mean(umaplast_pos)))
    print("RawHash2 Mean (only unmapped) # of sequenced chunks per read : " + str(safe_mean(umaplast_chunk)))

    print("#Done with RawHash2\n")


if __name__ == "__main__":
    main()
