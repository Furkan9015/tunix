"""Compact prior-candidate context for RawHash2 optimization prompts."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from discover_compbio.rawhash2_native.evaluator import structured_edits_from_answer


_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_LOCAL_PATH_RE = re.compile(r"/(?:home|tmp|dev/shm)/[^\s\"']+")


@dataclass(frozen=True)
class ArchiveCandidate:
    patch_sha256: str
    reward: float
    ok: bool
    phase: str | None
    f1: float | None
    map_elapsed_seconds: float | None
    map_max_rss_kb: float | None
    feedback_summary: str | None
    message: str | None
    edits: Any | None


def build_prompt_archive_context(
    *,
    native_audit_path: str | Path | None,
    reward_audit_path: str | Path | None = None,
    source_files: Sequence[str] | None = None,
    max_chars: int = 6000,
    max_candidates: int = 3,
    max_failures: int = 1,
    archive_variant: int | None = None,
) -> str:
    """Build a small prompt section from previously evaluated candidates.

    The CompBio Tunix task still applies every candidate to the canonical
    baseline source tree. This context is therefore phrased as parent ideas and
    measured evidence, not as source that has already been applied.
    """
    if max_chars <= 0 or not native_audit_path:
        return ""

    native_rows = list(
        _iter_jsonl_sample(Path(native_audit_path), head_lines=768, tail_lines=768)
    )
    if not native_rows:
        return ""
    reward_edits = _load_reward_edits(reward_audit_path)
    candidates = _audit_candidates(native_rows, reward_edits)
    if not candidates:
        return ""

    source_set = {str(path) for path in source_files or ()}
    verified = [
        candidate
        for candidate in candidates
        if candidate.edits is not None and (candidate.ok or candidate.reward > 0.0)
    ]
    verified = _select_verified_candidates(
        verified,
        source_set=source_set,
        max_candidates=max_candidates,
        archive_variant=archive_variant,
    )
    failures = [
        candidate
        for candidate in candidates
        if candidate.edits is not None
        and not candidate.ok
        and candidate.reward <= 0.0
        and candidate.phase in {"apply", "build", "map", "truth", "parse"}
    ]
    if source_set:
        failures = [
            candidate
            for candidate in failures
            if _source_overlap(candidate.edits, source_set) > 0
        ]
    failures.sort(
        key=lambda candidate: (
            _source_overlap(candidate.edits, source_set),
            candidate.phase in {"apply", "build"},
        ),
        reverse=True,
    )

    sections: list[str] = []
    if verified:
        edit_max_chars = _candidate_edit_max_chars(
            max_chars=max_chars,
            candidate_count=len(verified),
            include_failures=bool(failures and max_failures > 0),
        )
        sections.append(
            "Prior verified RawHash2 candidates on this same benchmark:\n"
            "- The editable source shown below is still the baseline source. Use these as measured parent ideas; do not assume they are already applied.\n"
            "- Prefer improving or simplifying a parent edit only when it fits the shown editable files."
        )
        for index, candidate in enumerate(verified, start=1):
            sections.append(
                _format_candidate(
                    candidate,
                    f"Verified parent {index}",
                    edit_max_chars=edit_max_chars,
                )
            )

    if failures and max_failures > 0:
        sections.append("Prior failed candidate to avoid:")
        for index, candidate in enumerate(failures[:max_failures], start=1):
            sections.append(_format_candidate(candidate, f"Failure lesson {index}", include_edits=False))

    if not sections:
        return ""
    return _truncate("\n".join(sections).strip(), max_chars)


def merge_native_audit_archives(
    input_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    max_output_rows: int = 512,
    top_global_rows: int = 96,
    top_per_family_rows: int = 64,
    recent_rows: int = 128,
    failure_rows: int = 96,
) -> int:
    """Merge per-worker native audit JSONLs into a compact prompt seed archive.

    Raw per-worker audits may grow for many rounds. The merged prompt archive is
    deliberately bounded: it keeps global bests, per-family winners, recent
    candidates, and a small failure lesson pool. Prompt construction then
    selects a tiny source-aware slice from this compact archive.
    """
    max_output_rows = max(1, int(max_output_rows))
    by_hash: dict[str, tuple[dict[str, Any], int]] = {}
    order = 0
    for input_path in input_paths:
        for row in _iter_jsonl_tail(Path(input_path), max_lines=1000000):
            order += 1
            digest = row.get("patch_sha256")
            if not isinstance(digest, str) or not digest:
                continue
            previous = by_hash.get(digest)
            if previous is None or _row_rank(row) > _row_rank(previous[0]):
                by_hash[digest] = (row, order)

    rows_with_order = list(by_hash.values())
    successes = [
        item
        for item in rows_with_order
        if item[0].get("edits") is not None
        and (bool(item[0].get("ok")) or (_as_float(item[0].get("reward"), default=0.0) or 0.0) > 0.0)
    ]
    failures = [
        item
        for item in rows_with_order
        if item[0].get("edits") is not None
        and not bool(item[0].get("ok"))
        and (_as_float(item[0].get("reward"), default=0.0) or 0.0) <= 0.0
    ]

    selected: dict[str, tuple[dict[str, Any], int]] = {}

    def add(item: tuple[dict[str, Any], int] | None) -> None:
        if item is None or len(selected) >= max_output_rows:
            return
        digest = item[0].get("patch_sha256")
        if isinstance(digest, str) and digest:
            selected.setdefault(digest, item)

    ranked_successes = sorted(successes, key=lambda item: _row_rank(item[0]), reverse=True)
    for item in ranked_successes[: max(0, int(top_global_rows))]:
        add(item)

    by_family: dict[str, list[tuple[dict[str, Any], int]]] = {}
    for item in ranked_successes:
        by_family.setdefault(_row_family(item[0]), []).append(item)
    for family_items in by_family.values():
        for item in family_items[: max(0, int(top_per_family_rows))]:
            add(item)

    for item in sorted(rows_with_order, key=lambda item: item[1], reverse=True)[
        : max(0, int(recent_rows))
    ]:
        add(item)

    for item in sorted(failures, key=lambda item: item[1], reverse=True)[
        : max(0, int(failure_rows))
    ]:
        add(item)

    for item in sorted(rows_with_order, key=lambda item: _row_rank(item[0]), reverse=True):
        add(item)

    rows = [item[0] for item in selected.values()]
    rows.sort(key=_row_output_rank, reverse=True)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    tmp.replace(output)
    return len(rows)


def _iter_jsonl_tail(path: Path, *, max_lines: int) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return ()
    rows: deque[dict[str, Any]] = deque(maxlen=max_lines)
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    except OSError:
        return ()
    return rows


def _iter_jsonl_sample(
    path: Path,
    *,
    head_lines: int,
    tail_lines: int,
) -> Iterable[dict[str, Any]]:
    """Read both ranked-archive heads and append-only audit tails."""
    if not path.is_file():
        return ()
    head: list[dict[str, Any]] = []
    tail: deque[dict[str, Any]] = deque(maxlen=max(0, tail_lines))
    try:
        with path.open("r", encoding="utf-8") as f:
            for index, line in enumerate(f):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(value, dict):
                    continue
                if index < head_lines:
                    head.append(value)
                else:
                    tail.append(value)
    except OSError:
        return ()
    return head + list(tail)


def _row_rank(row: dict[str, Any]) -> tuple[float, int, int, float]:
    reward = _as_float(row.get("reward"), default=0.0) or 0.0
    ok = 1 if row.get("ok") else 0
    has_edits = 1 if row.get("edits") is not None else 0
    map_elapsed = _as_float(row.get("map_elapsed_seconds"), default=1e12) or 1e12
    return (reward, ok, has_edits, -map_elapsed)


def _row_output_rank(row: dict[str, Any]) -> tuple[int, float, int, float, str]:
    reward = _as_float(row.get("reward"), default=0.0) or 0.0
    is_failure = 0 if (bool(row.get("ok")) or reward > 0.0) else -1
    map_elapsed = _as_float(row.get("map_elapsed_seconds"), default=1e12) or 1e12
    return (is_failure, reward, 1 if row.get("ok") else 0, -map_elapsed, _row_family(row))


def _row_family(row: dict[str, Any]) -> str:
    return _edit_family(row.get("edits"))


def _load_reward_edits(reward_audit_path: str | Path | None) -> dict[str, Any]:
    if not reward_audit_path:
        return {}
    out: dict[str, Any] = {}
    for row in _iter_jsonl_tail(Path(reward_audit_path), max_lines=768):
        completion = row.get("completion")
        if not isinstance(completion, str):
            continue
        parsed = _parse_completion_answer(completion)
        if not isinstance(parsed, dict):
            continue
        edits = structured_edits_from_answer(parsed)
        if edits is None:
            continue
        digest = hashlib.sha256(json.dumps(edits, sort_keys=True).encode("utf-8")).hexdigest()
        out[digest] = edits
    return out


def _parse_completion_answer(completion: str) -> dict[str, Any] | None:
    payloads: list[str] = []
    lower = completion.lower()
    think_end = lower.rfind("</think>")
    search_spaces = []
    if think_end >= 0:
        search_spaces.append(completion[think_end + len("</think>") :])
    search_spaces.append(completion)

    for text in search_spaces:
        matches = list(_ANSWER_RE.finditer(text))
        payloads.extend(match.group(1) for match in reversed(matches))
        starts = [match.start() for match in re.finditer(r"<answer>", text, re.IGNORECASE)]
        payloads.extend(text[start + len("<answer>") :] for start in reversed(starts))
        payloads.extend(_json_object_candidates(text))

    seen: set[str] = set()
    for payload in payloads:
        parsed = _parse_json_payload(payload)
        if parsed is not None:
            return parsed
        seen.add(payload)
    if completion not in seen:
        return _parse_json_payload(completion)
    return None


def _json_object_candidates(text: str) -> list[str]:
    starts = [
        match.start()
        for match in re.finditer(r"\{\s*\"(?:edits|edit|answer)\"\s*:", text)
    ]
    return [text[start:] for start in reversed(starts)]


def _parse_json_payload(payload: str) -> dict[str, Any] | None:
    payload = payload.strip()
    if payload.startswith("```"):
        lines = payload.splitlines()
        if len(lines) >= 2 and lines[0].strip().lower() in {"```", "```json"} and lines[-1].strip() == "```":
            payload = "\n".join(lines[1:-1]).strip()
    decoder = json.JSONDecoder()
    try:
        parsed, _ = decoder.raw_decode(payload.lstrip())
    except json.JSONDecodeError:
        repaired = _repair_truncated_json_object(payload.lstrip())
        if repaired is None:
            return None
        try:
            parsed, _ = decoder.raw_decode(repaired)
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _repair_truncated_json_object(text: str) -> str | None:
    if not text.startswith("{"):
        return None
    stack: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if not stack or stack.pop() != ch:
                return None
    if in_string or not stack:
        return None
    return text + "".join(reversed(stack))


def _audit_candidates(
    native_rows: Sequence[dict[str, Any]],
    reward_edits: dict[str, Any],
) -> list[ArchiveCandidate]:
    by_hash: dict[str, ArchiveCandidate] = {}
    for row in native_rows:
        digest = row.get("patch_sha256")
        if not isinstance(digest, str) or not digest:
            continue
        reward = _as_float(row.get("reward"), default=0.0) or 0.0
        candidate = ArchiveCandidate(
            patch_sha256=digest,
            reward=reward,
            ok=bool(row.get("ok")),
            phase=row.get("phase") if isinstance(row.get("phase"), str) else None,
            f1=_as_float(row.get("f1")),
            map_elapsed_seconds=_as_float(row.get("map_elapsed_seconds")),
            map_max_rss_kb=_as_float(row.get("map_max_rss_kb")),
            feedback_summary=_as_text(row.get("feedback_summary")),
            message=_as_text(row.get("message")),
            edits=row.get("edits") if row.get("edits") is not None else reward_edits.get(digest),
        )
        previous = by_hash.get(digest)
        if previous is None or (candidate.reward, candidate.ok) >= (previous.reward, previous.ok):
            by_hash[digest] = candidate
    return list(by_hash.values())


def _select_verified_candidates(
    candidates: Sequence[ArchiveCandidate],
    *,
    source_set: set[str],
    max_candidates: int,
    archive_variant: int | None,
) -> list[ArchiveCandidate]:
    """Pick parents with source relevance, global-best share, and diversity."""
    if max_candidates <= 0:
        return []

    selected: list[ArchiveCandidate] = []
    seen: set[str] = set()

    def add(candidate: ArchiveCandidate | None) -> None:
        if candidate is None or len(selected) >= max_candidates:
            return
        if candidate.patch_sha256 in seen:
            return
        selected.append(candidate)
        seen.add(candidate.patch_sha256)

    def first_unseen(
        choices: Sequence[ArchiveCandidate],
    ) -> ArchiveCandidate | None:
        for candidate in choices:
            if candidate.patch_sha256 not in seen:
                return candidate
        return None

    ranked = sorted(candidates, key=_candidate_rank, reverse=True)
    relevant = [
        candidate
        for candidate in ranked
        if _source_overlap(candidate.edits, source_set) > 0
    ]
    non_parameter = [
        candidate for candidate in ranked if not _is_parameter_tuning(candidate)
    ]
    relevant_non_parameter = [
        candidate
        for candidate in relevant
        if not _is_parameter_tuning(candidate)
    ]

    if source_set and not relevant:
        return []

    # Always reserve the first slot for the best source-relevant parent when one
    # exists. If no source set is available, fall back to global ranking.
    add(relevant[0] if relevant else ranked[0] if ranked else None)

    variant = 0 if archive_variant is None else int(archive_variant)
    # With the 75% archive schedule this puts the top global parent into about a
    # third of archive-bearing rows where it is directly editable. It gets
    # exposure without being pasted into unrelated source-slice prompts.
    global_best = ranked[0] if ranked else None
    include_global = (
        (archive_variant is None or variant % 4 == 0)
        and global_best is not None
        and (not source_set or _source_overlap(global_best.edits, source_set) > 0)
    )
    if include_global:
        add(global_best)

    # Give non-parameter wins their own slot whenever available. Prefer local
    # non-parameter parents; unrelated full patches are not shown as parents
    # because they changed first-round generations into large copied mutations.
    add(first_unseen(relevant_non_parameter) or (None if source_set else first_unseen(non_parameter)))

    fill_pool = relevant if source_set else ranked
    for candidate in fill_pool:
        add(candidate)
        if len(selected) >= max_candidates:
            break
    return selected


def _candidate_rank(candidate: ArchiveCandidate) -> tuple[float, float]:
    map_elapsed = candidate.map_elapsed_seconds or 1e12
    return (candidate.reward, -map_elapsed)


def _is_parameter_tuning(candidate: ArchiveCandidate) -> bool:
    return _edit_family(candidate.edits) == "parameter_defaults"


def _format_candidate(
    candidate: ArchiveCandidate,
    title: str,
    *,
    include_edits: bool = True,
    edit_max_chars: int = 900,
) -> str:
    metrics = [title, f"reward={candidate.reward:.4f}"]
    if candidate.phase:
        metrics.append(f"phase={candidate.phase}")
    if candidate.f1 is not None:
        metrics.append(f"F1={candidate.f1:.4f}")
    if candidate.map_elapsed_seconds is not None:
        metrics.append(f"map={candidate.map_elapsed_seconds:.1f}s")
    if candidate.map_max_rss_kb is not None:
        metrics.append(f"RSS={candidate.map_max_rss_kb / 1024 / 1024:.1f}GiB")
    lines = ["- " + ", ".join(metrics)]
    feedback = candidate.feedback_summary or candidate.message
    if feedback:
        lines.append(f"  feedback: {_truncate(_sanitize_text(feedback), 360)}")
    if include_edits and candidate.edits is not None:
        edit_json = json.dumps({"edits": candidate.edits}, sort_keys=True, separators=(",", ":"))
        lines.append(f"  edits: {_truncate(_sanitize_text(edit_json), edit_max_chars)}")
    return "\n".join(lines)


def _candidate_edit_max_chars(
    *,
    max_chars: int,
    candidate_count: int,
    include_failures: bool,
) -> int:
    fixed_overhead = 950 if include_failures else 650
    available = max(900, max_chars - fixed_overhead)
    per_candidate = available // max(1, candidate_count)
    return max(900, min(3200, per_candidate))


def _source_overlap(edits: Any, source_set: set[str]) -> int:
    if not source_set:
        return 0
    return sum(1 for path in _edit_paths(edits) if path in source_set)


def _edit_paths(value: Any) -> set[str]:
    paths: set[str] = set()
    if isinstance(value, dict):
        path = value.get("path") or value.get("file")
        if isinstance(path, str):
            paths.add(path)
        for child in value.values():
            paths.update(_edit_paths(child))
    elif isinstance(value, list):
        for child in value:
            paths.update(_edit_paths(child))
    return paths


def _edit_family(edits: Any) -> str:
    paths = _edit_paths(edits)
    if not paths:
        return "unknown"
    if paths <= {"src/roptions.c", "src/roptions.h"}:
        return "parameter_defaults"
    if paths & {"src/rseed.c", "src/rseed.h"}:
        return "algorithmic_seed"
    if paths & {"src/dtw.c", "src/dtw.h"}:
        return "algorithmic_dtw"
    if paths & {"src/lchain.c", "src/chain.h", "src/krmq.h"}:
        return "algorithmic_chaining"
    if paths & {"src/kalloc.c", "src/kalloc.h", "src/kthread.c", "src/kthread.h"}:
        return "runtime_memory"
    if paths & {"src/rmap.c", "src/rmap.h"}:
        return "algorithmic_mapping"
    if paths & {"src/rindex.c", "src/rindex.h", "src/rawhash.h"}:
        return "algorithmic_index"
    return "other"


def _as_float(value: Any, *, default: float | None = None) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _as_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if value is None:
        return None
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return str(value)


def _sanitize_text(text: str) -> str:
    return _LOCAL_PATH_RE.sub("[path]", text).replace("\r", "\\r").replace("\n", " ")


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 24)].rstrip() + " ...[truncated]"


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge RawHash2 native audit JSONLs into a global seed archive."
    )
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--max-output-rows", type=int, default=512)
    parser.add_argument("--top-global-rows", type=int, default=96)
    parser.add_argument("--top-per-family-rows", type=int, default=64)
    parser.add_argument("--recent-rows", type=int, default=128)
    parser.add_argument("--failure-rows", type=int, default=96)
    parser.add_argument("inputs", nargs="+", help="Input native audit JSONL paths.")
    args = parser.parse_args()
    count = merge_native_audit_archives(
        args.inputs,
        args.output,
        max_output_rows=args.max_output_rows,
        top_global_rows=args.top_global_rows,
        top_per_family_rows=args.top_per_family_rows,
        recent_rows=args.recent_rows,
        failure_rows=args.failure_rows,
    )
    print(f"wrote {count} rows to {args.output}")


if __name__ == "__main__":
    _main()
