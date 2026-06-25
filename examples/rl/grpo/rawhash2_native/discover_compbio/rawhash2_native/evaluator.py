"""Reward interface for native RawHash2 patch completions."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from discover_compbio.rawhash2_native.config import NativeBenchmarkConfig
from discover_compbio.rawhash2_native.config import config_from_case_spec
from discover_compbio.rawhash2_native.patching import PatchValidationError
from discover_compbio.rawhash2_native.patching import normalize_patch
from discover_compbio.rawhash2_native.patching import patch_from_edits
from discover_compbio.rawhash2_native.runner import run_edits_benchmark
from discover_compbio.rawhash2_native.runner import run_patch_benchmark
from discover_compbio.rawhash2_native.runner import run_patch_sequence_benchmark
from discover_compbio.rawhash2_native.runner import TransientBenchmarkError


def evaluate_patch(patch: str, config: NativeBenchmarkConfig | None = None) -> dict[str, Any]:
    try:
        normalized = normalize_patch(patch)
    except PatchValidationError as exc:
        return {"ok": False, "reward": 0.0, "message": str(exc)}
    try:
        metrics = run_patch_benchmark(normalized, config=config)
    except PatchValidationError as exc:
        metrics = {"ok": False, "reward": 0.0, "message": str(exc)}
    except TransientBenchmarkError:
        raise
    except Exception as exc:
        metrics = {
            "ok": False,
            "reward": 0.0,
            "message": f"{type(exc).__name__}: {exc}",
        }
    metrics.setdefault("reward", 0.0)
    _audit_metrics(normalized, metrics)
    return metrics


def evaluate_patch_sequence(
    patches: list[str],
    config: NativeBenchmarkConfig | None = None,
) -> dict[str, Any]:
    if not patches:
        return {"ok": False, "reward": 0.0, "message": "patch sequence is empty"}
    try:
        normalized = [normalize_patch(patch) for patch in patches]
    except PatchValidationError as exc:
        return {"ok": False, "reward": 0.0, "message": str(exc)}
    try:
        metrics = run_patch_sequence_benchmark(normalized, config=config)
    except PatchValidationError as exc:
        metrics = {"ok": False, "reward": 0.0, "message": str(exc)}
    except TransientBenchmarkError:
        raise
    except Exception as exc:
        metrics = {
            "ok": False,
            "reward": 0.0,
            "message": f"{type(exc).__name__}: {exc}",
        }
    metrics.setdefault("reward", 0.0)
    _audit_metrics(json.dumps({"patches": normalized}, sort_keys=True), metrics)
    return metrics


def evaluate_edits(edits: Any, config: NativeBenchmarkConfig | None = None) -> dict[str, Any]:
    """Evaluate a structured-edits candidate by applying it directly (no git apply)."""
    try:
        metrics = run_edits_benchmark(edits, config=config)
    except PatchValidationError as exc:
        metrics = {"ok": False, "reward": 0.0, "message": str(exc)}
    except TransientBenchmarkError:
        raise
    except Exception as exc:
        metrics = {
            "ok": False,
            "reward": 0.0,
            "message": f"{type(exc).__name__}: {exc}",
        }
    metrics.setdefault("reward", 0.0)
    _audit_metrics(json.dumps(edits, sort_keys=True), metrics)
    return metrics


def score_patch(patch: str, config: NativeBenchmarkConfig | None = None) -> float:
    return float(evaluate_patch(patch, config=config).get("reward", 0.0))


def patch_from_answer(parsed: dict[str, Any], source_repo: Path | None = None) -> str | None:
    edits = structured_edits_from_answer(parsed)
    if edits is not None:
        if source_repo is None:
            raise PatchValidationError("structured edits require a source repository")
        return patch_from_edits(source_repo, edits)

    for key in ("patch", "diff", "unified_diff"):
        value = parsed.get(key)
        if isinstance(value, str):
            return value
    patches = patches_from_answer(parsed)
    if patches and len(patches) == 1:
        return patches[0]
    files = parsed.get("files")
    if isinstance(files, list):
        # The native task intentionally scores diffs, not full arbitrary files,
        # so the model cannot silently replace benchmark/build machinery.
        return None
    return None


def score_answer(parsed: dict[str, Any], expected: dict[str, Any]) -> float:
    case_spec = expected.get("rawhash2_case") if isinstance(expected.get("rawhash2_case"), dict) else expected
    cfg = config_from_case_spec(case_spec)
    # Preferred path: structured edits applied directly to the isolated source
    # tree (no unified diff, no `git apply`). This avoids the corrupt-patch
    # failure mode of hand-written diffs and gives clean per-edit feedback.
    edits = structured_edits_from_answer(parsed)
    if edits is not None:
        return float(evaluate_edits(edits, config=cfg).get("reward", 0.0))
    patches = patches_from_answer(parsed)
    if patches:
        if len(patches) == 1:
            return score_patch(patches[0], config=cfg)
        return float(evaluate_patch_sequence(patches, config=cfg).get("reward", 0.0))
    # Fallback: a raw unified diff in patch/diff/unified_diff.
    try:
        patch = patch_from_answer(parsed, source_repo=cfg.rawhash2_repo)
    except PatchValidationError:
        return 0.0
    if patch is None:
        return 0.0
    return score_patch(patch, config=cfg)


def structured_edits_from_answer(parsed: dict[str, Any]) -> Any | None:
    """Return structured edits from common model-emitted answer shapes.

    The scored representation is still the native benchmark result after
    applying edits. This normalization only repairs schema wrappers that arise
    from forced-answer prefixes, e.g. ``{"edits":[{"edits":[...]}]}``, or a
    single edit emitted as an object instead of a one-element array.
    """
    if not isinstance(parsed, dict):
        return None
    if "edits" in parsed:
        return _normalize_structured_edits_value(parsed.get("edits"))
    if "edit" in parsed:
        return _normalize_structured_edits_value(parsed.get("edit"))
    answer = parsed.get("answer")
    if isinstance(answer, dict):
        edits = structured_edits_from_answer(answer)
        if edits is not None:
            return edits
    if "patches" in parsed:
        return _structured_edits_from_patches(parsed.get("patches"))
    return None


def _normalize_structured_edits_value(value: Any) -> Any | None:
    if isinstance(value, dict):
        if "edit" in value and not _looks_like_source_edit(value):
            return _normalize_structured_edits_value(value.get("edit"))
        if "edits" in value and not _looks_like_source_edit(value):
            return _normalize_structured_edits_value(value.get("edits"))
        if "patches" in value and not _looks_like_source_edit(value):
            return _structured_edits_from_patches(value.get("patches"))
        return [value] if _looks_like_source_edit(value) else None
    if not isinstance(value, list):
        return None
    out: list[Any] = []
    for item in value:
        normalized = _normalize_structured_edits_value(item)
        if normalized is None:
            return None
        out.extend(normalized)
    return out or None


def _structured_edits_from_patches(value: Any) -> Any | None:
    if not isinstance(value, list):
        return None
    out: list[Any] = []
    for item in value:
        if isinstance(item, str):
            return None
        normalized = _normalize_structured_edits_value(item)
        if normalized is None:
            return None
        out.extend(normalized)
    return out or None


def patches_from_answer(parsed: dict[str, Any]) -> list[str] | None:
    if not isinstance(parsed, dict):
        return None
    direct: list[str] = []
    for key in ("patch", "diff", "unified_diff"):
        value = parsed.get(key)
        if isinstance(value, str):
            direct.append(value)
    if direct:
        return direct
    answer = parsed.get("answer")
    if isinstance(answer, dict):
        nested = patches_from_answer(answer)
        if nested:
            return nested
    patches = parsed.get("patches")
    if not isinstance(patches, list):
        return None
    out: list[str] = []
    for item in patches:
        if isinstance(item, str):
            out.append(item)
            continue
        if isinstance(item, dict):
            nested = patches_from_answer(item)
            if nested:
                out.extend(nested)
                continue
        return None
    return out or None


def _looks_like_source_edit(value: dict[str, Any]) -> bool:
    return any(
        key in value
        for key in (
            "path",
            "file",
            "anchor_id",
            "find",
            "find_lines",
            "after",
            "after_lines",
            "before",
            "before_lines",
        )
    )


def summarize_metrics(metrics: dict[str, Any]) -> str:
    return json.dumps(metrics, sort_keys=True)


def _audit_metrics(patch: str, metrics: dict[str, Any]) -> None:
    audit_path = os.environ.get("RAWHASH2_NATIVE_AUDIT_PATH")
    if not audit_path:
        return
    path = Path(audit_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "patch_sha256": hashlib.sha256(patch.encode("utf-8")).hexdigest(),
        "reward": metrics.get("reward", 0.0),
        "ok": metrics.get("ok", False),
        "phase": metrics.get("phase"),
        "f1": metrics.get("f1"),
        "map_elapsed_seconds": metrics.get("map_elapsed_seconds"),
        "map_max_rss_kb": metrics.get("map_max_rss_kb"),
        "profile_signal_to_event_seconds": metrics.get("profile_signal_to_event_seconds"),
        "profile_sketching_seconds": metrics.get("profile_sketching_seconds"),
        "profile_seeding_seconds": metrics.get("profile_seeding_seconds"),
        "profile_chaining_seconds": metrics.get("profile_chaining_seconds"),
        "profile_mapping_multithreaded_seconds": metrics.get("profile_mapping_multithreaded_seconds"),
        "paf_summary": metrics.get("paf_summary"),
        "feedback_summary": metrics.get("feedback_summary"),
        "message": metrics.get("message"),
        "returncode": metrics.get("returncode"),
        "out_dir": metrics.get("out_dir"),
        "workdir": metrics.get("workdir"),
        "debug_feedback": metrics.get("debug_feedback"),
    }
    audit_edits = _audit_structured_edits(patch)
    if audit_edits is not None:
        row["edits"] = audit_edits
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _audit_structured_edits(patch: str) -> Any | None:
    try:
        parsed = json.loads(patch)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, (dict, list)):
        return None
    if len(json.dumps(parsed, sort_keys=True)) > 12_000:
        return None
    return parsed
