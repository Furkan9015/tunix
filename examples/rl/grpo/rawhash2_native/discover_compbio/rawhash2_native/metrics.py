"""Metric parsing and reward scoring for RawHash2 native runs."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import os
import re
from pathlib import Path
from statistics import mean
from typing import Any


_NUM = r"([0-9.eE+-]+)"
_CHAIN_AMBIGUITY_RATIO_THRESHOLD = 0.80
_CHAIN_CLEAR_RATIO_THRESHOLD = 0.50
_UNMAPPED_HIGH_SCORE_THRESHOLD = 100.0


def parse_time_file(path: Path) -> dict[str, float | int | str]:
    metrics: dict[str, float | int | str] = {}
    if not path.exists():
        return metrics
    elapsed_prefix = "Elapsed (wall clock) time (h:mm:ss or m:ss):"
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith(elapsed_prefix):
            value = line[len(elapsed_prefix) :].strip()
            metrics["elapsed_seconds"] = _parse_elapsed(value)
        elif line.startswith("Maximum resident set size (kbytes):"):
            value = line.split(":", 1)[1].strip()
            try:
                metrics["max_rss_kb"] = int(value)
            except ValueError:
                pass
        elif line.startswith("User time (seconds):"):
            value = line.split(":", 1)[1].strip()
            metrics["user_seconds"] = _parse_float(value)
        elif line.startswith("System time (seconds):"):
            value = line.split(":", 1)[1].strip()
            metrics["system_seconds"] = _parse_float(value)
    return metrics


def parse_throughput(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not path.exists():
        return metrics
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"BP per sec:\s+([0-9.]+)\s+([0-9.]+)", text)
    if match:
        metrics["bp_per_sec_mean"] = float(match.group(1))
        metrics["bp_per_sec_median"] = float(match.group(2))
    return metrics


def parse_comparison(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not path.exists():
        return metrics
    patterns = {
        "precision": r"RawHash2 precision:\s+([0-9.eE+-]+)",
        "recall": r"RawHash2 recall:\s+([0-9.eE+-]+)",
        "f1": r"RawHash2 F-1 score:\s+([0-9.eE+-]+)",
        "mean_time_per_read_ms": r"RawHash2 Mean time per read :\s+([0-9.eE+-]+)",
        "median_time_per_read_ms": r"RawHash2 Median time per read :\s+([0-9.eE+-]+)",
        "mean_sequenced_bases": r"RawHash2 Mean # of sequenced bases per read :\s+([0-9.eE+-]+)",
        "mean_sequenced_chunks": r"RawHash2 Mean # of sequenced chunks per read :\s+([0-9.eE+-]+)",
    }
    text = path.read_text(encoding="utf-8", errors="replace")
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def parse_profile_stderr(path: Path) -> dict[str, float]:
    """Parse RawHash2 PROFILE=1 aggregate phase timers from stderr."""
    metrics: dict[str, float] = {}
    pattern = re.compile(
        rf"File read:\s*{_NUM} sec;\s*"
        rf"Signal-to-event:\s*{_NUM} sec;\s*"
        rf"Sketching:\s*{_NUM} sec;\s*"
        rf"Seeding:\s*{_NUM} sec \(of which sorting:\s*{_NUM} sec\);\s*"
        rf"Chaining:\s*{_NUM} sec \(of which sorting:\s*{_NUM} sec\);\s*"
        rf"Mapping:\s*{_NUM} sec;\s*"
        rf"Mapping \(multi-threaded\):\s*{_NUM} sec"
    )
    for line in _iter_lines(path):
        match = pattern.search(line)
        if not match:
            continue
        values = [float(x) for x in match.groups()]
        keys = [
            "profile_file_read_seconds",
            "profile_signal_to_event_seconds",
            "profile_sketching_seconds",
            "profile_seeding_seconds",
            "profile_seed_sorting_seconds",
            "profile_chaining_seconds",
            "profile_chain_sorting_seconds",
            "profile_mapping_seconds",
            "profile_mapping_multithreaded_seconds",
        ]
        metrics.update(dict(zip(keys, values)))
    total = sum(
        metrics.get(key, 0.0)
        for key in (
            "profile_file_read_seconds",
            "profile_signal_to_event_seconds",
            "profile_sketching_seconds",
            "profile_seeding_seconds",
            "profile_chaining_seconds",
        )
    )
    if total > 0:
        metrics["profile_accounted_single_thread_seconds"] = total
        for key in (
            "profile_signal_to_event_seconds",
            "profile_sketching_seconds",
            "profile_seeding_seconds",
            "profile_chaining_seconds",
        ):
            metrics[f"{key}_fraction"] = metrics.get(key, 0.0) / total
    return metrics


def parse_paf_summary(path: Path) -> dict[str, Any]:
    """Aggregate RawHash2 PAF tags into compact feedback metrics."""
    mapped = 0
    total = 0
    tag_values: dict[str, list[float]] = {
        "mt": [],
        "ci": [],
        "sl": [],
        "cm": [],
        "nc": [],
        "s1": [],
        "sm": [],
    }
    for line in _iter_lines(path):
        if not line or line.startswith("[") or line.startswith("#"):
            continue
        cols = line.rstrip().split("\t")
        if len(cols) < 12:
            continue
        total += 1
        if cols[4] != "*":
            mapped += 1
        tags = _parse_paf_tags(cols[12:])
        for key in tag_values:
            if key in tags:
                try:
                    tag_values[key].append(float(tags[key]))
                except (TypeError, ValueError):
                    pass

    summary: dict[str, Any] = {
        "reads": total,
        "mapped_reads": mapped,
        "unmapped_reads": max(0, total - mapped),
        "mapped_fraction": mapped / total if total else 0.0,
    }
    names = {
        "mt": "map_time_ms",
        "ci": "chunks",
        "sl": "signal_length",
        "cm": "anchor_count",
        "nc": "chain_count",
        "s1": "best_chain_score",
        "sm": "mean_chain_score",
    }
    for tag, values in tag_values.items():
        prefix = names[tag]
        summary.update(_series_summary(prefix, values))
    return summary


def parse_debug_summary(path: Path, *, workers: int | None = None) -> dict[str, Any]:
    """Aggregate --output-chains and --debug-read stderr diagnostics."""
    if not path.exists():
        return _DebugAccumulator().to_summary()

    min_parallel_bytes = int(os.environ.get("RAWHASH2_NATIVE_DEBUG_PARSE_PARALLEL_MIN_BYTES", "1048576"))
    if workers is None:
        workers = int(os.environ.get("RAWHASH2_NATIVE_DEBUG_PARSE_WORKERS", str(os.cpu_count() or 1)))
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    if size >= min_parallel_bytes and workers > 1:
        return _parse_debug_summary_parallel(path, workers=workers)
    return _parse_debug_summary_stream(path)


def _parse_debug_summary_stream(path: Path) -> dict[str, Any]:
    acc = _DebugAccumulator()
    chain_block: list[str] = []
    for raw_line in _iter_lines(path):
        if raw_line.startswith("CHAINS\t"):
            if chain_block:
                acc.add_chain_record(chain_block)
            chain_block = [raw_line]
        elif chain_block and raw_line.startswith("  chain["):
            chain_block.append(raw_line)
        else:
            if chain_block:
                acc.add_chain_record(chain_block)
                chain_block = []
            acc.add_debug_line(raw_line.strip())
    if chain_block:
        acc.add_chain_record(chain_block)
    return acc.to_summary()


def _parse_debug_summary_parallel(path: Path, *, workers: int) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    acc = _DebugAccumulator()
    chain_blocks: list[list[str]] = []
    chain_block: list[str] = []
    for raw_line in text.splitlines():
        if raw_line.startswith("CHAINS\t"):
            if chain_block:
                chain_blocks.append(chain_block)
            chain_block = [raw_line]
        elif chain_block and raw_line.startswith("  chain["):
            chain_block.append(raw_line)
        else:
            if chain_block:
                chain_blocks.append(chain_block)
                chain_block = []
            acc.add_debug_line(raw_line.strip())
    if chain_block:
        chain_blocks.append(chain_block)

    if not chain_blocks:
        return acc.to_summary()

    batch_size = max(1, int(os.environ.get("RAWHASH2_NATIVE_DEBUG_PARSE_BLOCK_BATCH", "128")))
    batches = [chain_blocks[i : i + batch_size] for i in range(0, len(chain_blocks), batch_size)]
    max_workers = max(1, min(workers, len(batches)))
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for partial in pool.map(_parse_chain_batch, batches):
            acc.merge(partial)
    summary = acc.to_summary()
    summary["debug_parse_parallel_workers"] = max_workers
    summary["debug_parse_chain_batches"] = len(batches)
    summary["debug_parse_parallel_backend"] = "process"
    return summary


def _parse_chain_batch(blocks: list[list[str]]) -> "_DebugAccumulator":
    acc = _DebugAccumulator()
    for block in blocks:
        acc.add_chain_record(block)
    return acc


def make_feedback_summary(metrics: dict[str, Any]) -> str:
    """Format high-signal native feedback for audit/logging."""
    parts: list[str] = []
    if metrics.get("f1") is not None:
        parts.append(f"F1={float(metrics['f1']):.4f}")
    if metrics.get("map_elapsed_seconds") is not None:
        parts.append(f"map_elapsed={float(metrics['map_elapsed_seconds']):.2f}s")
    if metrics.get("map_max_rss_kb") is not None:
        parts.append(f"map_rss={int(metrics['map_max_rss_kb'])}KB")

    profile_keys = [
        ("profile_signal_to_event_seconds", "event"),
        ("profile_sketching_seconds", "sketch"),
        ("profile_seeding_seconds", "seed"),
        ("profile_chaining_seconds", "chain"),
        ("profile_mapping_multithreaded_seconds", "map_mt"),
    ]
    profile = [
        f"{label}={float(metrics[key]):.2f}s"
        for key, label in profile_keys
        if metrics.get(key) is not None
    ]
    if profile:
        parts.append("profile[" + ", ".join(profile) + "]")

    paf = metrics.get("paf_summary") or {}
    if paf:
        parts.append(
            "paf["
            f"reads={paf.get('reads', 0)}, mapped={paf.get('mapped_reads', 0)}, "
            f"map_ms_sum={float(paf.get('map_time_ms_sum', 0.0)):.2f}, "
            f"chains_sum={float(paf.get('chain_count_sum', 0.0)):.0f}, "
            f"anchors_sum={float(paf.get('anchor_count_sum', 0.0)):.0f}"
            "]"
        )

    debug = metrics.get("debug_summary") or (metrics.get("debug_feedback") or {}).get("debug_summary") or {}
    if debug:
        ratio = float(debug.get("chain_second_to_top_ratio_mean", 0.0))
        parts.append(
            "debug["
            f"reads={debug.get('chain_read_records', 0)}, "
            f"unmapped_with_chains={debug.get('chain_unmapped_with_chains_records', 0)}, "
            f"unmapped_no_chains={debug.get('chain_unmapped_no_chain_records', 0)}, "
            f"ambiguous={debug.get('chain_ambiguous_records', 0)}, "
            f"high_score_unmapped={debug.get('chain_unmapped_high_score_records', 0)}, "
            f"second/top_mean={ratio:.3f}"
            "]"
        )
        interpretation = metrics.get("debug_interpretation") or interpret_debug_summary(debug)
        if interpretation:
            parts.append(f"debug_hint[{interpretation}]")
    return "; ".join(parts)


def interpret_debug_summary(debug: dict[str, Any]) -> str:
    """Turn raw chain diagnostics into short model-readable guidance."""
    reads = int(debug.get("chain_read_records", 0) or 0)
    if reads <= 0:
        return ""

    unmapped = int(debug.get("chain_unmapped_records", 0) or 0)
    mapped = int(debug.get("chain_mapped_records", 0) or 0)
    unmapped_with = int(debug.get("chain_unmapped_with_chains_records", 0) or 0)
    unmapped_no = int(debug.get("chain_unmapped_no_chain_records", 0) or 0)
    ambiguous = int(debug.get("chain_ambiguous_records", 0) or 0)
    high_score_unmapped = int(debug.get("chain_unmapped_high_score_records", 0) or 0)
    top_mapq_zero = int(debug.get("chain_unmapped_top_mapq_zero_records", 0) or 0)
    detail_mean = float(debug.get("chain_detail_per_read_mean", 0.0) or 0.0)
    ratio_all = float(debug.get("chain_second_to_top_ratio_mean", 0.0) or 0.0)
    ratio_unmapped = float(debug.get("chain_unmapped_second_to_top_ratio_mean", 0.0) or 0.0)
    top_score_unmapped = float(debug.get("chain_unmapped_top_score_mean", 0.0) or 0.0)

    pieces = [
        f"{_pct(mapped, reads)} mapped in debug pass",
        f"{_pct(unmapped, reads)} unmapped",
    ]
    if unmapped:
        pieces.append(f"{_pct(unmapped_no, unmapped)} of unmapped reads had no candidate chains, suggesting seed/anchor loss")
        pieces.append(
            f"{_pct(unmapped_with, unmapped)} of unmapped reads had candidate chains, suggesting mapq/decision/ambiguity issues"
        )
        pieces.append(f"{_pct(high_score_unmapped, unmapped)} of unmapped reads had top chain score >= {_UNMAPPED_HIGH_SCORE_THRESHOLD:g}")
        pieces.append(f"{_pct(top_mapq_zero, max(1, unmapped_with))} of unmapped reads with chains had top MAPQ 0")
    pieces.append(f"{_pct(ambiguous, reads)} of reads had close top-two chains")
    pieces.append(f"mean second/top score ratio {ratio_all:.2f} overall and {ratio_unmapped:.2f} on unmapped reads")
    pieces.append(f"unmapped mean top score {top_score_unmapped:.1f}; mean candidate chains per read {detail_mean:.1f}")
    return "; ".join(pieces)


def _pct(count: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{100.0 * count / total:.1f}%"


def compact_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    """Keep only low-cardinality metrics safe to store in RL state."""
    if not metrics:
        return {}
    keys = (
        "ok",
        "reward",
        "reason",
        "f1",
        "precision",
        "recall",
        "map_elapsed_seconds",
        "map_max_rss_kb",
        "index_elapsed_seconds",
        "index_max_rss_kb",
        "profile_signal_to_event_seconds",
        "profile_sketching_seconds",
        "profile_seeding_seconds",
        "profile_seed_sorting_seconds",
        "profile_chaining_seconds",
        "profile_chain_sorting_seconds",
        "profile_mapping_seconds",
        "profile_mapping_multithreaded_seconds",
        "feedback_summary",
        "debug_interpretation",
    )
    compact = {key: metrics[key] for key in keys if key in metrics and metrics[key] is not None}
    for nested_key in ("paf_summary", "debug_summary"):
        nested = metrics.get(nested_key)
        if isinstance(nested, dict):
            compact[nested_key] = {
                key: value
                for key, value in nested.items()
                if isinstance(value, (int, float, str, bool)) or value is None
            }
    return compact


def load_baseline(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


class _RunningStats:
    __slots__ = ("count", "sum", "max")

    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.max = 0.0

    def add(self, value: float | int | None) -> None:
        if value is None:
            return
        v = float(value)
        self.count += 1
        self.sum += v
        if self.count == 1 or v > self.max:
            self.max = v

    def merge(self, other: "_RunningStats") -> None:
        if other.count == 0:
            return
        if self.count == 0 or other.max > self.max:
            self.max = other.max
        self.count += other.count
        self.sum += other.sum

    def summary(self, prefix: str) -> dict[str, float | int]:
        return {
            f"{prefix}_count": self.count,
            f"{prefix}_sum": self.sum,
            f"{prefix}_mean": self.sum / self.count if self.count else 0.0,
            f"{prefix}_max": self.max if self.count else 0.0,
        }


class _DebugAccumulator:
    def __init__(self) -> None:
        self.counters: dict[str, int] = {
            "debug_read_records": 0,
            "decision_records": 0,
            "accepted_decisions": 0,
            "chain_read_records": 0,
            "chain_detail_records": 0,
            "chain_mapped_records": 0,
            "chain_unmapped_records": 0,
            "chain_with_chains_records": 0,
            "chain_no_chain_records": 0,
            "chain_multi_chain_records": 0,
            "chain_unmapped_with_chains_records": 0,
            "chain_unmapped_no_chain_records": 0,
            "chain_ambiguous_records": 0,
            "chain_clear_best_records": 0,
            "chain_unmapped_high_score_records": 0,
            "chain_unmapped_top_mapq_zero_records": 0,
            "chain_mapped_top_mapq_zero_records": 0,
            "chain_detail_mismatch_records": 0,
        }
        self.debug_stats = {
            key: _RunningStats() for key in ("n_events", "n_seeds", "n_seed_hits", "n_chains")
        }
        self.decision_stats = {
            key: _RunningStats()
            for key in ("weighted_sum", "threshold", "bestQ", "bestC", "bestA", "meanC", "meanQ")
        }
        self.chain_stats = {
            key: _RunningStats()
            for key in (
                "n_chains",
                "n_maps",
                "offset",
                "score",
                "cnt",
                "mapq",
                "align_score",
                "detail_per_read",
                "top_score",
                "second_score",
                "score_margin",
                "second_to_top_ratio",
                "top_mapq",
                "max_mapq",
                "top_cnt",
                "total_score",
                "total_cnt",
                "mapped_top_score",
                "mapped_second_to_top_ratio",
                "mapped_top_mapq",
                "unmapped_top_score",
                "unmapped_second_to_top_ratio",
                "unmapped_top_mapq",
            )
        }

    def merge(self, other: "_DebugAccumulator") -> None:
        for key, value in other.counters.items():
            self.counters[key] = self.counters.get(key, 0) + value
        for own, theirs in (
            (self.debug_stats, other.debug_stats),
            (self.decision_stats, other.decision_stats),
            (self.chain_stats, other.chain_stats),
        ):
            for key, stats in theirs.items():
                own[key].merge(stats)

    def add_debug_line(self, line: str) -> None:
        if line.startswith("DEBUG_READ\t"):
            self.counters["debug_read_records"] += 1
            fields = _parse_tab_kv(line.split("\t")[2:])
            for key, stats in self.debug_stats.items():
                stats.add(fields.get(key))
        elif line.startswith("DEBUG_DECISION\t"):
            self.counters["decision_records"] += 1
            fields = _parse_tab_kv(line.split("\t")[2:])
            for key, stats in self.decision_stats.items():
                stats.add(fields.get(key))
            if fields.get("weighted_sum", -1.0) >= fields.get("threshold", float("inf")):
                self.counters["accepted_decisions"] += 1

    def add_chain_record(self, lines: list[str]) -> None:
        if not lines:
            return
        header = lines[0].strip()
        if not header.startswith("CHAINS\t"):
            return

        self.counters["chain_read_records"] += 1
        fields = _parse_tab_kv(header.split("\t")[2:])
        n_chains = int(fields.get("n_chains", 0.0))
        n_maps = int(fields.get("n_maps", 0.0))
        self.chain_stats["n_chains"].add(fields.get("n_chains"))
        self.chain_stats["n_maps"].add(fields.get("n_maps"))
        self.chain_stats["offset"].add(fields.get("offset"))

        if n_maps > 0:
            self.counters["chain_mapped_records"] += 1
        else:
            self.counters["chain_unmapped_records"] += 1
        if n_chains > 0:
            self.counters["chain_with_chains_records"] += 1
        else:
            self.counters["chain_no_chain_records"] += 1
        if n_chains > 1:
            self.counters["chain_multi_chain_records"] += 1
        if n_maps == 0 and n_chains > 0:
            self.counters["chain_unmapped_with_chains_records"] += 1
        if n_maps == 0 and n_chains == 0:
            self.counters["chain_unmapped_no_chain_records"] += 1

        detail_count = 0
        top_score = 0.0
        second_score = 0.0
        top_mapq = 0.0
        top_cnt = 0.0
        max_mapq = 0.0
        total_score = 0.0
        total_cnt = 0.0

        for line in lines[1:]:
            stripped = line.strip()
            if not stripped.startswith("chain["):
                continue
            detail_count += 1
            self.counters["chain_detail_records"] += 1
            detail = _parse_tab_kv(stripped.split("\t")[1:])
            score = detail.get("score", 0.0)
            cnt = detail.get("cnt", 0.0)
            mapq = detail.get("mapq", 0.0)
            total_score += score
            total_cnt += cnt
            max_mapq = max(max_mapq, mapq)
            for key in ("score", "cnt", "mapq", "align_score"):
                self.chain_stats[key].add(detail.get(key))
            if score > top_score:
                second_score = top_score
                top_score = score
                top_mapq = mapq
                top_cnt = cnt
            elif score > second_score:
                second_score = score

        self.chain_stats["detail_per_read"].add(detail_count)
        if n_chains != detail_count:
            self.counters["chain_detail_mismatch_records"] += 1

        ratio = second_score / top_score if top_score > 0 else 0.0
        margin = top_score - second_score
        self.chain_stats["top_score"].add(top_score)
        self.chain_stats["second_score"].add(second_score)
        self.chain_stats["score_margin"].add(margin)
        self.chain_stats["second_to_top_ratio"].add(ratio)
        self.chain_stats["top_mapq"].add(top_mapq)
        self.chain_stats["max_mapq"].add(max_mapq)
        self.chain_stats["top_cnt"].add(top_cnt)
        self.chain_stats["total_score"].add(total_score)
        self.chain_stats["total_cnt"].add(total_cnt)

        if top_score > 0 and ratio >= _CHAIN_AMBIGUITY_RATIO_THRESHOLD:
            self.counters["chain_ambiguous_records"] += 1
        if top_score > 0 and ratio <= _CHAIN_CLEAR_RATIO_THRESHOLD:
            self.counters["chain_clear_best_records"] += 1

        if n_maps > 0:
            self.chain_stats["mapped_top_score"].add(top_score)
            self.chain_stats["mapped_second_to_top_ratio"].add(ratio)
            self.chain_stats["mapped_top_mapq"].add(top_mapq)
            if top_mapq <= 0:
                self.counters["chain_mapped_top_mapq_zero_records"] += 1
        else:
            self.chain_stats["unmapped_top_score"].add(top_score)
            self.chain_stats["unmapped_second_to_top_ratio"].add(ratio)
            self.chain_stats["unmapped_top_mapq"].add(top_mapq)
            if top_score >= _UNMAPPED_HIGH_SCORE_THRESHOLD:
                self.counters["chain_unmapped_high_score_records"] += 1
            if top_mapq <= 0 and n_chains > 0:
                self.counters["chain_unmapped_top_mapq_zero_records"] += 1

    def to_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = dict(self.counters)
        summary["chain_ambiguity_ratio_threshold"] = _CHAIN_AMBIGUITY_RATIO_THRESHOLD
        summary["chain_clear_ratio_threshold"] = _CHAIN_CLEAR_RATIO_THRESHOLD
        summary["chain_unmapped_high_score_threshold"] = _UNMAPPED_HIGH_SCORE_THRESHOLD
        for key, stats in self.debug_stats.items():
            summary.update(stats.summary(f"debug_{key}"))
        for key, stats in self.decision_stats.items():
            summary.update(stats.summary(f"decision_{key}"))
        for key, stats in self.chain_stats.items():
            summary.update(stats.summary(f"chain_{key}"))
        return summary


def _parse_paf_tags(fields: list[str]) -> dict[str, str]:
    tags: dict[str, str] = {}
    for field in fields:
        pieces = field.split(":", 2)
        if len(pieces) == 3:
            tags[pieces[0]] = pieces[2]
    return tags


def _iter_lines(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            yield line.rstrip("\n")


def _parse_tab_kv(fields: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for field in fields:
        if "=" in field:
            key, raw = field.split("=", 1)
        elif ":" in field:
            key, raw = field.split(":", 1)
        else:
            continue
        try:
            out[key] = float(raw)
        except ValueError:
            continue
    return out


def _series_summary(prefix: str, values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_sum": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_count": len(values),
        f"{prefix}_sum": float(sum(values)),
        f"{prefix}_mean": float(mean(values)),
        f"{prefix}_max": float(max(values)),
    }


def score_metrics(
    candidate: dict[str, Any],
    baseline: dict[str, Any] | None,
    *,
    require_truth: bool = True,
) -> dict[str, Any]:
    if not candidate.get("ok"):
        return {"reward": 0.0, "reason": candidate.get("message", "candidate failed")}

    cand_f1 = _metric(candidate, "f1")
    if require_truth and cand_f1 is None:
        return {"reward": 0.0, "reason": "truth PAF is required for accuracy reward"}

    baseline_map = _metric(baseline, "map_elapsed_seconds") if baseline else None
    baseline_mem = _metric(baseline, "map_max_rss_kb") if baseline else None
    baseline_f1 = _metric(baseline, "f1") if baseline else None

    cand_map = _metric(candidate, "map_elapsed_seconds")
    cand_mem = _metric(candidate, "map_max_rss_kb")

    # F1 is a hard gate and a positive objective only for improvements over the
    # baseline. Baseline-equivalent F1 is necessary for full speed/memory credit,
    # but it should not pay out the 60% accuracy head by itself.
    if cand_f1 is None:
        accuracy = 0.0
        accuracy_gate = 1.0 if not require_truth else 0.0
    elif baseline_f1 is not None and baseline_f1 > 0:
        f1_floor = ACCURACY_GATE_FRACTION * baseline_f1
        if cand_f1 < f1_floor:
            accuracy = 0.0
            accuracy_gate = 0.0
        else:
            accuracy_gate = max(
                0.0,
                min(1.0, (cand_f1 - f1_floor) / max(baseline_f1 - f1_floor, 1e-12)),
            )
            accuracy = max(
                0.0,
                min(1.0, (cand_f1 - baseline_f1) / ACCURACY_F1_GAIN_TARGET),
            )
    else:
        accuracy = max(0.0, min(1.0, cand_f1 / 0.80))
        accuracy_gate = accuracy

    # Improvement OVER baseline: parity scores 0 on speed and memory, so those
    # axes still measure actual optimization rather than crediting a no-op for
    # matching the baseline. A SPEEDUP_TARGET (e.g. 0.50 = 50% faster) earns full
    # speed credit.
    speed = _improvement_score(baseline_map, cand_map, SPEEDUP_TARGET)
    memory = _improvement_score(baseline_mem, cand_mem, MEMSAVE_TARGET)
    perf = SPEED_WEIGHT * speed + MEMORY_WEIGHT * memory
    objective = ACCURACY_WEIGHT * accuracy + perf
    reward = accuracy_gate * (VALIDITY_FLOOR + (1.0 - VALIDITY_FLOOR) * objective)
    speedup_frac = (baseline_map / cand_map - 1.0) if (baseline_map and cand_map and cand_map > 0) else None
    memsave_frac = (baseline_mem / cand_mem - 1.0) if (baseline_mem and cand_mem and cand_mem > 0) else None
    return {
        "reward": max(0.0, min(1.0, reward)),
        "accuracy_score": accuracy,
        "accuracy_gate": accuracy_gate,
        "speed_score": speed,
        "memory_score": memory,
        "performance_score": perf,
        "objective_score": objective,
        "speedup_frac": speedup_frac,
        "memsave_frac": memsave_frac,
    }


# Reward shaping: accuracy improvement is the largest single term; speed and
# memory improvements fill the remaining range (see score_metrics). Preserving
# baseline F1 is handled by the gate and earns the validity floor, not the full
# accuracy-improvement head.
SPEEDUP_TARGET = 0.50   # fractional map-time speedup that earns full speed credit (50% faster)
MEMSAVE_TARGET = 0.30   # fractional RSS reduction that earns full memory credit
VALIDITY_FLOOR = 0.10   # reward floor for compiling, accuracy-gated candidates
ACCURACY_GATE_FRACTION = 0.70  # candidates below this fraction of baseline F1 get zero reward
ACCURACY_F1_GAIN_TARGET = 0.02 # absolute F1 gain over baseline that earns full accuracy credit
ACCURACY_WEIGHT = 0.60  # F1 improvement over baseline is the dominant objective
SPEED_WEIGHT = 0.20     # map-time improvement remains useful but secondary
MEMORY_WEIGHT = 0.20    # peak RSS improvement is weighted equally with speed


def _improvement_score(baseline: float | None, observed: float | None, target: float) -> float:
    """Normalized fractional improvement of ``observed`` vs ``baseline``.

    ``improvement = baseline/observed - 1`` is >0 when the candidate is smaller
    (faster / leaner). Parity scores 0, a ``target`` fractional gain scores 1.0,
    and regressions clamp to 0. This replaces the old ``min(1.25, ratio)/1.25``
    normalization that mapped parity itself to 0.8.
    """
    if baseline is None or observed is None or baseline <= 0 or observed <= 0 or target <= 0:
        return 0.0
    improvement = baseline / observed - 1.0
    return max(0.0, min(1.0, improvement / target))


def _metric(metrics: dict[str, Any] | None, key: str) -> float | None:
    if not metrics:
        return None
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def _parse_elapsed(value: str) -> float:
    parts = value.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        return float(value)
    except ValueError:
        return 0.0
