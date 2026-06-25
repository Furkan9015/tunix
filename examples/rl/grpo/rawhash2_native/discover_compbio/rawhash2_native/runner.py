"""Native RawHash2 build/run harness."""

from __future__ import annotations

import hashlib
import functools
import json
import logging
import os
import platform
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
from typing import Any
import uuid

from discover_compbio.rawhash2_native.config import NativeBenchmarkConfig
from discover_compbio.rawhash2_native.config import default_config
from discover_compbio.rawhash2_native.metrics import load_baseline
from discover_compbio.rawhash2_native.metrics import interpret_debug_summary
from discover_compbio.rawhash2_native.metrics import make_feedback_summary
from discover_compbio.rawhash2_native.metrics import parse_comparison
from discover_compbio.rawhash2_native.metrics import parse_debug_summary
from discover_compbio.rawhash2_native.metrics import parse_paf_summary
from discover_compbio.rawhash2_native.metrics import parse_profile_stderr
from discover_compbio.rawhash2_native.metrics import parse_throughput
from discover_compbio.rawhash2_native.metrics import parse_time_file
from discover_compbio.rawhash2_native.metrics import score_metrics
from discover_compbio.rawhash2_native.patching import apply_edits_to_repo
from discover_compbio.rawhash2_native.patching import apply_patch_to_repo
from discover_compbio.rawhash2_native.patching import PatchValidationError


_BENCHMARK_SEMAPHORES_LOCK = threading.Lock()
_BENCHMARK_SEMAPHORES: dict[int, threading.BoundedSemaphore] = {}
_MAP_SEMAPHORES_LOCK = threading.Lock()
_MAP_SEMAPHORES: dict[int, threading.BoundedSemaphore] = {}
_RESULT_CACHE_VERSION = "rawhash2_native_result_cache_v3"
_RESULT_CACHE_WAIT_SECONDS = 2.0
_CACHE_VOLATILE_FIELDS = {
    "workdir",
    "out_dir",
    "paf",
    "index_command",
    "map_command",
}


class TransientBenchmarkError(RuntimeError):
    """Infrastructure backpressure exhausted before candidate evaluation."""


def run_patch_benchmark(patch: str, config: NativeBenchmarkConfig | None = None) -> dict[str, Any]:
    return _run_benchmark(
        lambda repo: apply_patch_to_repo(repo, patch),
        identity=patch,
        config=config,
    )


def run_patch_sequence_benchmark(
    patches: list[str],
    config: NativeBenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Apply multiple unified diffs sequentially, then build/run/score once."""
    return _run_benchmark(
        lambda repo: _apply_patch_sequence_to_repo(repo, patches),
        identity=json.dumps({"patches": patches}, sort_keys=True),
        config=config,
    )


def run_edits_benchmark(edits: Any, config: NativeBenchmarkConfig | None = None) -> dict[str, Any]:
    """Build/run/score a candidate produced by applying structured edits directly.

    Same isolated copy/build/run/score pipeline as :func:`run_patch_benchmark`,
    but the candidate source is produced by applying the model's ``edits`` array
    in place (no unified diff, no ``git apply``).
    """
    identity = json.dumps(edits, sort_keys=True)
    return _run_benchmark(
        lambda repo: apply_edits_to_repo(repo, edits),
        identity=identity,
        config=config,
    )


def _run_benchmark(apply_fn, identity: str, config: NativeBenchmarkConfig | None = None) -> dict[str, Any]:
    cfg = config or default_config()
    if not cfg.enable_execution:
        return {
            "ok": False,
            "message": "native execution disabled; set RAWHASH2_NATIVE_ENABLE_EXECUTION=1",
        }
    missing = _missing_inputs(cfg)
    if missing:
        return {"ok": False, "message": "missing benchmark inputs", "missing": missing}
    _wait_for_resources_or_raise(cfg, check_memory=False, check_disk=True)

    cache_key = None
    cache_owner = None
    if cfg.result_cache_path:
        cache_key = _result_cache_key(identity, cfg)
        cached, cache_owner = _result_cache_get_or_claim(
            cfg.result_cache_path,
            cache_key,
            stale_seconds=max(1, int(cfg.result_cache_stale_seconds)),
        )
        if cached is not None:
            cached["cache_hit"] = True
            return cached

    run_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    temp_parent = Path(tempfile.mkdtemp(prefix=f"rawhash2_{run_id}_", dir=str(cfg.output_root)))
    repo = temp_parent / "RawHash2"
    try:
        _copy_repo(cfg.rawhash2_repo, repo)
        try:
            apply_fn(repo)
        except PatchValidationError as exc:
            run = {
                "ok": False,
                "phase": "apply",
                "reward": 0.0,
                "message": str(exc),
                "workdir": str(temp_parent),
            }
            _maybe_store_result_cache(cfg, cache_key, cache_owner, run)
            return run
        build = _build(repo, cfg)
        if build["returncode"] != 0:
            run = {
                "ok": False,
                "phase": "build",
                "returncode": build["returncode"],
                "message": _format_failed_message(build["returncode"], build["stderr"]),
                "workdir": str(temp_parent),
            }
            _maybe_store_result_cache(cfg, cache_key, cache_owner, run)
            return run
        run = _run_rawbench_with_concurrency_limit(repo, temp_parent / "out", cfg)
        run["workdir"] = str(temp_parent)
        if run.get("ok"):
            baseline = cfg.baseline_metrics or load_baseline(cfg.baseline_json)
            run.update(score_metrics(run, baseline, require_truth=cfg.require_truth_for_reward))
        _maybe_store_result_cache(cfg, cache_key, cache_owner, run)
        return run
    except BaseException:
        _release_result_cache_claim(cfg, cache_key, cache_owner)
        raise
    finally:
        if not cfg.keep_workdir:
            shutil.rmtree(temp_parent, ignore_errors=True)


def run_baseline_benchmark(config: NativeBenchmarkConfig | None = None) -> dict[str, Any]:
    cfg = config or default_config()
    if not cfg.enable_execution:
        return {
            "ok": False,
            "message": "native execution disabled; set RAWHASH2_NATIVE_ENABLE_EXECUTION=1",
        }
    missing = _missing_inputs(cfg)
    if missing:
        return {"ok": False, "message": "missing benchmark inputs", "missing": missing}
    _wait_for_resources_or_raise(cfg, check_memory=False, check_disk=True)

    cfg.output_root.mkdir(parents=True, exist_ok=True)
    temp_parent = Path(tempfile.mkdtemp(prefix="rawhash2_baseline_", dir=str(cfg.output_root)))
    repo = temp_parent / "RawHash2"
    try:
        _copy_repo(cfg.rawhash2_repo, repo)
        build = _build(repo, cfg)
        if build["returncode"] != 0:
            return {
                "ok": False,
                "phase": "build",
                "returncode": build["returncode"],
                "message": _format_failed_message(build["returncode"], build["stderr"]),
                "workdir": str(temp_parent),
            }
        run = _run_rawbench_with_concurrency_limit(repo, temp_parent / "out", cfg)
        run["workdir"] = str(temp_parent)
        return run
    finally:
        if not cfg.keep_workdir:
            shutil.rmtree(temp_parent, ignore_errors=True)


def _copy_repo(src: Path, dst: Path) -> None:
    def ignore(dirpath: str, names: list[str]) -> set[str]:
        ignored = {".git", "build", "bin"} & set(names)
        if Path(dirpath).name == "src":
            ignored.update(name for name in names if name.endswith(".o"))
            if "rawhash2" in names:
                ignored.add("rawhash2")
        return ignored

    shutil.copytree(src, dst, ignore=ignore)
    data_dir = dst / "test/data"
    if data_dir.exists():
        shutil.rmtree(data_dir)


def _apply_patch_sequence_to_repo(repo: Path, patches: list[str]) -> None:
    if not patches:
        raise PatchValidationError("patch sequence is empty")
    for idx, patch in enumerate(patches):
        try:
            apply_patch_to_repo(repo, patch)
        except PatchValidationError as exc:
            raise PatchValidationError(f"patch {idx} failed: {exc}") from exc


def _build(repo: Path, cfg: NativeBenchmarkConfig) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("MAKEFLAGS", f"-j{cfg.build_jobs}")
    proc = subprocess.run(
        list(cfg.build_command),
        cwd=str(repo),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=cfg.timeout_seconds,
        check=False,
    )
    bin_dir = repo / "bin"
    bin_dir.mkdir(exist_ok=True)
    raw_bin = repo / "src/rawhash2"
    if raw_bin.exists():
        shutil.copy2(raw_bin, bin_dir / "rawhash2")
    return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def _benchmark_semaphore(limit: int) -> threading.BoundedSemaphore | None:
    if limit <= 0:
        return None
    with _BENCHMARK_SEMAPHORES_LOCK:
        semaphore = _BENCHMARK_SEMAPHORES.get(limit)
        if semaphore is None:
            semaphore = threading.BoundedSemaphore(limit)
            _BENCHMARK_SEMAPHORES[limit] = semaphore
        return semaphore


def _map_semaphore(limit: int) -> threading.BoundedSemaphore | None:
    if limit <= 0:
        return None
    with _MAP_SEMAPHORES_LOCK:
        semaphore = _MAP_SEMAPHORES.get(limit)
        if semaphore is None:
            semaphore = threading.BoundedSemaphore(limit)
            _MAP_SEMAPHORES[limit] = semaphore
        return semaphore


def _run_rawbench_with_concurrency_limit(
    repo: Path,
    out_dir: Path,
    cfg: NativeBenchmarkConfig,
) -> dict[str, Any]:
    semaphore = _benchmark_semaphore(int(cfg.benchmark_concurrency or 0))
    if semaphore is None:
        _wait_for_resources_or_raise(cfg, check_memory=True, check_disk=True)
        return _run_rawbench(repo, out_dir, cfg)
    start = time.monotonic()
    semaphore.acquire()
    wait_seconds = time.monotonic() - start
    if wait_seconds >= 1.0:
        logging.info(
            "RawHash2 benchmark waited %.1fs for benchmark_concurrency=%d.",
            wait_seconds,
            int(cfg.benchmark_concurrency),
        )
    try:
        _wait_for_resources_or_raise(cfg, check_memory=True, check_disk=True)
        return _run_rawbench(repo, out_dir, cfg)
    finally:
        semaphore.release()


def _run_rawbench(repo: Path, out_dir: Path, cfg: NativeBenchmarkConfig) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    binary = repo / "bin/rawhash2"
    if not binary.exists():
        binary = repo / "src/rawhash2"

    prefix = "rawbench_hsapiens_candidate"
    index_path = out_dir / f"{prefix}_rawhash2_{cfg.preset}_quant.ind"
    paf_path = out_dir / f"{prefix}_rawhash2_{cfg.preset}_quant.paf"
    index_time = out_dir / f"{prefix}_rawhash2_index_{cfg.preset}_quant.time"
    map_time = out_dir / f"{prefix}_rawhash2_map_{cfg.preset}_quant_map.time"
    index_out = out_dir / f"{prefix}_rawhash2_{cfg.preset}_quant.out"
    index_err = out_dir / f"{prefix}_rawhash2_{cfg.preset}_quant.err"
    map_out = out_dir / f"{prefix}_rawhash2_{cfg.preset}_quant_map.out"
    map_err = out_dir / f"{prefix}_rawhash2_{cfg.preset}_quant_map.err"
    feedback_paf = out_dir / f"{prefix}_rawhash2_{cfg.preset}_feedback.paf"
    feedback_time = out_dir / f"{prefix}_rawhash2_map_{cfg.preset}_feedback.time"
    feedback_out = out_dir / f"{prefix}_rawhash2_{cfg.preset}_feedback.out"
    feedback_err = out_dir / f"{prefix}_rawhash2_{cfg.preset}_feedback.err"

    common = [
        str(binary),
        "--bp-per-sec",
        str(cfg.bp_per_sec),
        "--r10",
        "-x",
        cfg.preset,
        "-t",
        str(cfg.threads),
    ]
    index_cmd = [
        "/usr/bin/time",
        "-vpo",
        str(index_time),
        *common,
        "-p",
        str(cfg.pore_model),
        "-d",
        str(index_path),
        *cfg.extra_params,
        str(cfg.reference_fasta),
    ]
    map_cmd = [
        "/usr/bin/time",
        "-vpo",
        str(map_time),
        *common,
        "-o",
        str(paf_path),
        *cfg.extra_params,
        str(index_path),
        str(cfg.fast5_dir),
    ]
    feedback_params = list(cfg.extra_params) + list(cfg.debug_feedback_params)
    if cfg.debug_feedback_read:
        feedback_params.extend(["--debug-read", cfg.debug_feedback_read])
    feedback_cmd = [
        "/usr/bin/time",
        "-vpo",
        str(feedback_time),
        *common,
        "-o",
        str(feedback_paf),
        *feedback_params,
        str(index_path),
        str(cfg.fast5_dir),
    ]

    index_proc = _run_to_files(index_cmd, index_out, index_err, cfg.timeout_seconds)
    if index_proc.returncode != 0:
        return _failed_run("index", index_proc, index_err, out_dir)

    map_proc = _run_map_to_files_with_resource_retry(map_cmd, map_out, map_err, cfg)
    if map_proc.returncode != 0:
        return _failed_run("map", map_proc, map_err, out_dir)

    metrics: dict[str, Any] = {
        "ok": True,
        "out_dir": str(out_dir),
        "paf": str(paf_path),
        "index_command": index_cmd,
        "map_command": map_cmd,
        "index_stderr_tail": _read_tail(index_err, cfg.log_tail_bytes),
        "map_stderr_tail": _read_tail(map_err, cfg.log_tail_bytes),
    }
    index_metrics = parse_time_file(index_time)
    map_metrics = parse_time_file(map_time)
    metrics.update(
        {
            "index_elapsed_seconds": index_metrics.get("elapsed_seconds"),
            "index_max_rss_kb": index_metrics.get("max_rss_kb"),
            "map_elapsed_seconds": map_metrics.get("elapsed_seconds"),
            "map_max_rss_kb": map_metrics.get("max_rss_kb"),
        }
    )
    metrics.update(parse_profile_stderr(map_err))
    metrics["paf_summary"] = parse_paf_summary(paf_path)

    if cfg.truth_paf:
        try:
            pafstats, analyze = _resolve_eval_scripts(repo, cfg)
        except FileNotFoundError as exc:
            return {"ok": False, "phase": "truth_eval", "message": str(exc), "out_dir": str(out_dir)}
        ann_paf = out_dir / f"{prefix}_ann.paf"
        throughput = out_dir / f"{prefix}.throughput"
        comparison = out_dir / f"{prefix}.comparison"
        with ann_paf.open("w", encoding="utf-8") as out, throughput.open("w", encoding="utf-8") as err:
            pafstats_proc = subprocess.run(
                [os.environ.get("PYTHON", "python3"), str(pafstats), str(paf_path), "-r", str(cfg.truth_paf), "-a"],
                cwd=str(repo),
                stdout=out,
                stderr=err,
                check=False,
                timeout=cfg.timeout_seconds,
            )
        if pafstats_proc.returncode != 0:
            return _failed_truth_eval("pafstats", pafstats_proc, throughput, out_dir)
        with comparison.open("w", encoding="utf-8") as out:
            analyze_proc = subprocess.run(
                [os.environ.get("PYTHON", "python3"), str(analyze), str(ann_paf)],
                cwd=str(repo),
                stdout=out,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=cfg.timeout_seconds,
            )
        if analyze_proc.returncode != 0:
            return _failed_truth_eval("analyze_paf", analyze_proc, comparison, out_dir)
        metrics.update(parse_throughput(throughput))
        metrics.update(parse_comparison(comparison))

    if cfg.enable_debug_feedback:
        feedback_proc = _run_to_files(feedback_cmd, feedback_out, feedback_err, cfg.timeout_seconds)
        feedback_metrics = parse_time_file(feedback_time)
        debug_summary = parse_debug_summary(feedback_err, workers=cfg.threads)
        debug_interpretation = interpret_debug_summary(debug_summary)
        metrics["debug_feedback"] = {
            "ok": feedback_proc.returncode == 0,
            "returncode": feedback_proc.returncode,
            "command": feedback_cmd,
            "paf": str(feedback_paf),
            "elapsed_seconds": feedback_metrics.get("elapsed_seconds"),
            "max_rss_kb": feedback_metrics.get("max_rss_kb"),
            "profile": parse_profile_stderr(feedback_err),
            "paf_summary": parse_paf_summary(feedback_paf),
            "debug_summary": debug_summary,
            "interpretation": debug_interpretation,
            "stdout_tail": _read_tail(feedback_out, cfg.log_tail_bytes),
            "stderr_tail": _read_tail(feedback_err, cfg.log_tail_bytes),
        }
        metrics["debug_summary"] = metrics["debug_feedback"]["debug_summary"]
        metrics["debug_interpretation"] = debug_interpretation

    metrics["feedback_summary"] = make_feedback_summary(metrics)

    return metrics


def _run_map_to_files_with_resource_retry(
    cmd: list[str],
    stdout_path: Path,
    stderr_path: Path,
    cfg: NativeBenchmarkConfig,
) -> subprocess.CompletedProcess[str]:
    semaphore = _map_semaphore(int(cfg.map_concurrency or 0))
    if semaphore is None:
        return _run_resource_sensitive_command(cmd, stdout_path, stderr_path, cfg, phase="map")

    start = time.monotonic()
    semaphore.acquire()
    wait_seconds = time.monotonic() - start
    if wait_seconds >= 1.0:
        logging.info(
            "RawHash2 map waited %.1fs for map_concurrency=%d.",
            wait_seconds,
            int(cfg.map_concurrency),
        )
    try:
        return _run_resource_sensitive_command(cmd, stdout_path, stderr_path, cfg, phase="map")
    finally:
        semaphore.release()


def _run_resource_sensitive_command(
    cmd: list[str],
    stdout_path: Path,
    stderr_path: Path,
    cfg: NativeBenchmarkConfig,
    *,
    phase: str,
) -> subprocess.CompletedProcess[str]:
    attempts = max(0, int(cfg.resource_kill_retry_attempts)) + 1
    proc: subprocess.CompletedProcess[str] | None = None
    for attempt in range(attempts):
        _wait_for_resources_or_raise(cfg, check_memory=True, check_disk=False)
        proc = _run_to_files(cmd, stdout_path, stderr_path, cfg.timeout_seconds)
        if proc.returncode == 0 or not _is_resource_kill_returncode(proc.returncode):
            return proc

        resource_error = _resource_preflight(cfg, check_memory=True, check_disk=False)
        stderr_tail = _read_tail(stderr_path, min(cfg.log_tail_bytes, 4096))
        logging.warning(
            "RawHash2 %s was killed by signal/resource pressure on attempt %d/%d "
            "(returncode=%d, resource_error=%s, stderr_tail=%r).",
            phase,
            attempt + 1,
            attempts,
            proc.returncode,
            resource_error,
            stderr_tail[-512:],
        )
        if resource_error and attempt + 1 >= attempts:
            raise TransientBenchmarkError(
                f"rawhash2 {phase} killed under host resource pressure after "
                f"{attempts} attempt(s): " + json.dumps(resource_error, sort_keys=True)
            )
        if attempt + 1 < attempts:
            _wait_for_resources_or_raise(cfg, check_memory=True, check_disk=False)
    assert proc is not None
    return proc


def _is_resource_kill_returncode(returncode: int | None) -> bool:
    if returncode is None:
        return False
    # subprocess reports direct SIGKILL as -9; /usr/bin/time commonly reports
    # child SIGKILL as shell-style 128 + 9.
    return returncode in {-9, 137}


def _run_to_files(cmd: list[str], stdout_path: Path, stderr_path: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        return subprocess.run(
            cmd,
            text=True,
            stdout=out,
            stderr=err,
            timeout=timeout,
            check=False,
        )


def _failed_run(phase: str, proc: subprocess.CompletedProcess[str], err_path: Path, out_dir: Path) -> dict[str, Any]:
    stderr = err_path.read_text(encoding="utf-8", errors="replace") if err_path.exists() else ""
    return {
        "ok": False,
        "phase": phase,
        "returncode": proc.returncode,
        "message": _format_failed_message(proc.returncode, stderr),
        "out_dir": str(out_dir),
    }


def _format_failed_message(returncode: int, stderr: str) -> str:
    diagnostic = _diagnostic_message(stderr)
    prefix = f"returncode={returncode}"
    if not diagnostic:
        return prefix
    if diagnostic.startswith(prefix):
        return diagnostic
    return f"{prefix}: {diagnostic}"


def _diagnostic_message(text: str, limit: int = 4000) -> str:
    """Prefer actionable diagnostics over warning-heavy stderr tails."""
    if not text:
        return ""
    lines = text.splitlines()
    needles = (
        " error:",
        ": error:",
        "undefined reference",
        "multiple definition",
        "segmentation fault",
        "addresssanitizer",
        "invalid free",
        "double free",
        "corrupted",
        "enters a free block",
        "assertion",
    )
    hit_indexes = [
        idx for idx, line in enumerate(lines) if any(needle in line.lower() for needle in needles)
    ]
    if not hit_indexes:
        return text[-limit:]
    selected: list[str] = []
    seen: set[int] = set()
    for hit in hit_indexes[:12]:
        for idx in range(max(0, hit - 2), min(len(lines), hit + 5)):
            if idx in seen:
                continue
            seen.add(idx)
            selected.append(lines[idx])
    message = "\n".join(selected)
    if len(message) > limit:
        return message[-limit:]
    return message


def _failed_truth_eval(
    phase: str,
    proc: subprocess.CompletedProcess[str],
    log_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    message = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    return {
        "ok": False,
        "phase": phase,
        "returncode": proc.returncode,
        "message": message[-4000:],
        "out_dir": str(out_dir),
    }


def _missing_inputs(cfg: NativeBenchmarkConfig) -> list[str]:
    checks = {
        "rawhash2_repo": cfg.rawhash2_repo,
        "reference_fasta": cfg.reference_fasta,
        "fast5_dir": cfg.fast5_dir,
        "pore_model": cfg.pore_model,
    }
    if cfg.require_truth_for_reward:
        checks["truth_paf"] = cfg.truth_paf
    missing = [name for name, path in checks.items() if path is None or not Path(path).exists()]
    if cfg.truth_paf:
        try:
            _resolve_eval_scripts(cfg.rawhash2_repo, cfg)
        except FileNotFoundError:
            missing.append("eval_scripts_dir:pafstats.py,analyze_paf.py")
    return missing


def _resolve_eval_scripts(repo: Path, cfg: NativeBenchmarkConfig) -> tuple[Path, Path]:
    """Resolve reward-side PAF evaluators outside the editable candidate tree."""
    workspace = Path(os.environ.get("COMPBIO_WORKSPACE", "/home/furka/compbio"))
    candidates = [
        cfg.eval_scripts_dir,
        repo / "test/scripts",
        workspace / "RawHash2/test/scripts",
    ]
    seen: set[Path] = set()
    checked: list[Path] = []
    for directory in candidates:
        if directory is None:
            continue
        path = Path(directory)
        if path in seen:
            continue
        seen.add(path)
        checked.append(path)
        pafstats = path / "pafstats.py"
        analyze = path / "analyze_paf.py"
        if pafstats.is_file() and analyze.is_file():
            return pafstats, analyze
    checked_text = ", ".join(str(path) for path in checked)
    raise FileNotFoundError(
        "missing RawHash2 truth-evaluation scripts "
        f"(pafstats.py and analyze_paf.py); checked: {checked_text}"
    )


def _resource_preflight(
    cfg: NativeBenchmarkConfig,
    *,
    check_memory: bool = True,
    check_disk: bool = True,
) -> dict[str, Any]:
    if cfg.allow_low_resources:
        return {}

    failures: dict[str, Any] = {}
    available_gb = _available_memory_gb() if check_memory else None
    if check_memory and available_gb is not None and available_gb < cfg.min_available_memory_gb:
        failures["available_memory_gb"] = round(available_gb, 2)
        failures["required_memory_gb"] = cfg.min_available_memory_gb

    if check_disk:
        disk_path = _existing_parent(cfg.output_root)
        free_disk_gb = shutil.disk_usage(disk_path).free / (1024**3)
        if free_disk_gb < cfg.min_output_free_disk_gb:
            failures["output_path"] = str(disk_path)
            failures["free_disk_gb"] = round(free_disk_gb, 2)
            failures["required_free_disk_gb"] = cfg.min_output_free_disk_gb

    if failures:
        failures["override"] = "set RAWHASH2_NATIVE_ALLOW_LOW_RESOURCES=1"
    return failures


def _wait_for_resources_or_raise(
    cfg: NativeBenchmarkConfig,
    *,
    check_memory: bool,
    check_disk: bool,
) -> None:
    deadline = time.monotonic() + max(0, int(cfg.resource_retry_seconds))
    interval = max(1, int(cfg.resource_retry_interval_seconds))
    while True:
        resource_error = _resource_preflight(
            cfg,
            check_memory=check_memory,
            check_disk=check_disk,
        )
        if not resource_error:
            return
        now = time.monotonic()
        if now >= deadline:
            resource_error["retry_seconds_exhausted"] = max(0, int(cfg.resource_retry_seconds))
            raise TransientBenchmarkError(
                "insufficient host resources after retry: "
                + json.dumps(resource_error, sort_keys=True)
            )
        sleep_seconds = min(interval, max(0.1, deadline - now))
        logging.info(
            "RawHash2 benchmark resource preflight failed; retrying in %.1fs: %s",
            sleep_seconds,
            resource_error,
        )
        time.sleep(sleep_seconds)


def _maybe_store_result_cache(
    cfg: NativeBenchmarkConfig,
    cache_key: str | None,
    cache_owner: str | None,
    result: dict[str, Any],
) -> None:
    if not cfg.result_cache_path or not cache_key or not cache_owner:
        return
    if _is_cacheable_result(result):
        _result_cache_store_done(cfg.result_cache_path, cache_key, cache_owner, result)
    else:
        _release_result_cache_claim(cfg, cache_key, cache_owner)


def _is_cacheable_result(result: dict[str, Any]) -> bool:
    if result.get("ok") is True:
        return True
    return result.get("phase") in {"apply", "build"}


def _release_result_cache_claim(
    cfg: NativeBenchmarkConfig,
    cache_key: str | None,
    cache_owner: str | None,
) -> None:
    if not cfg.result_cache_path or not cache_key or not cache_owner:
        return
    try:
        with _result_cache_connection(cfg.result_cache_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "DELETE FROM result_cache WHERE cache_key = ? AND owner = ? AND status = 'running'",
                (cache_key, cache_owner),
            )
            conn.commit()
    except sqlite3.Error:
        logging.exception("Failed to release RawHash2 result-cache claim.")


def _result_cache_get_or_claim(
    path: Path,
    cache_key: str,
    *,
    stale_seconds: int,
) -> tuple[dict[str, Any] | None, str | None]:
    owner = f"{os.getpid()}-{uuid.uuid4().hex}"
    while True:
        with _result_cache_connection(path) as conn:
            now = time.time()
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status, owner, updated_at, payload FROM result_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO result_cache(cache_key, status, owner, created_at, updated_at, payload)
                    VALUES (?, 'running', ?, ?, ?, NULL)
                    """,
                    (cache_key, owner, now, now),
                )
                conn.commit()
                return None, owner
            status, row_owner, updated_at, payload = row
            if status == "done" and payload:
                conn.commit()
                return json.loads(payload), None
            row_age = now - float(updated_at or 0.0)
            owner_alive = _result_cache_owner_is_alive(row_owner)
            if status == "running" and (row_age > stale_seconds or not owner_alive):
                conn.execute(
                    """
                    UPDATE result_cache
                    SET owner = ?, updated_at = ?, payload = NULL
                    WHERE cache_key = ?
                    """,
                    (owner, now, cache_key),
                )
                conn.commit()
                reason = "stale" if row_age > stale_seconds else "dead-owner"
                logging.info(
                    "Reclaimed %s RawHash2 result-cache key %s "
                    "(age=%.1fs, previous_owner=%s).",
                    reason,
                    cache_key[:12],
                    row_age,
                    row_owner,
                )
                return None, owner
            conn.commit()
        logging.info("Waiting for in-flight RawHash2 result-cache key %s.", cache_key[:12])
        time.sleep(_RESULT_CACHE_WAIT_SECONDS)


def _result_cache_owner_is_alive(owner: str | None) -> bool:
    if not owner:
        return False
    pid_text = str(owner).split("-", maxsplit=1)[0]
    if not pid_text.isdigit():
        return False
    try:
        os.kill(int(pid_text), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _result_cache_store_done(path: Path, cache_key: str, owner: str, result: dict[str, Any]) -> None:
    payload = json.dumps(_cache_payload(result), sort_keys=True)
    try:
        with _result_cache_connection(path) as conn:
            now = time.time()
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                UPDATE result_cache
                SET status = 'done', updated_at = ?, payload = ?
                WHERE cache_key = ? AND owner = ? AND status = 'running'
                """,
                (now, payload, cache_key, owner),
            )
            conn.commit()
    except sqlite3.Error:
        logging.exception("Failed to store RawHash2 result-cache row.")


def _result_cache_connection(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=60.0, isolation_level=None)
    conn.execute("PRAGMA busy_timeout = 60000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS result_cache (
            cache_key TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            owner TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            payload TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS result_cache_status_idx ON result_cache(status)")
    return conn


def _cache_payload(result: dict[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in result.items() if key not in _CACHE_VOLATILE_FIELDS}
    payload["cache_hit"] = False
    return payload


def _result_cache_key(identity: str, cfg: NativeBenchmarkConfig) -> str:
    contract = {
        "version": _RESULT_CACHE_VERSION,
        "candidate_sha256": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
        "rawhash2_source": _source_tree_fingerprint(str(cfg.rawhash2_repo)),
        "reference_fasta": _path_fingerprint(str(cfg.reference_fasta)),
        "fast5_dir": _path_fingerprint(str(cfg.fast5_dir)),
        "pore_model": _path_fingerprint(str(cfg.pore_model)),
        "truth_paf": _path_fingerprint(str(cfg.truth_paf)) if cfg.truth_paf else None,
        "baseline_json": _path_fingerprint(str(cfg.baseline_json)) if cfg.baseline_json else None,
        "baseline_metrics_sha256": _json_sha256(cfg.baseline_metrics) if cfg.baseline_metrics else None,
        "eval_scripts_dir": _path_fingerprint(str(cfg.eval_scripts_dir)) if cfg.eval_scripts_dir else None,
        "preset": cfg.preset,
        "threads": cfg.threads,
        "bp_per_sec": cfg.bp_per_sec,
        "extra_params": list(cfg.extra_params),
        "build_command": list(cfg.build_command),
        "build_dependencies": _build_dependency_fingerprints(cfg.build_command),
        "runtime_abi": _runtime_abi_fingerprint(),
        "harness_code": _harness_code_fingerprint(),
        "build_jobs": cfg.build_jobs,
        "require_truth_for_reward": cfg.require_truth_for_reward,
        "enable_debug_feedback": cfg.enable_debug_feedback,
        "debug_feedback_params": list(cfg.debug_feedback_params),
        "debug_feedback_read": cfg.debug_feedback_read,
    }
    encoded = json.dumps(contract, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_dependency_fingerprints(build_command: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """Fingerprint native build inputs that are referenced outside the repo tree."""
    deps: dict[str, Any] = {}
    for item in build_command:
        if "=" not in item:
            continue
        name, value = item.split("=", 1)
        if name == "HDF5_INCLUDE_DIR":
            include_dir = Path(value)
            deps[name] = _path_fingerprint(str(include_dir))
            deps["HDF5_H5public_h"] = _path_fingerprint(str(include_dir / "H5public.h"))
            deps["HDF5_hdf5_h"] = _path_fingerprint(str(include_dir / "hdf5.h"))
        elif name == "HDF5_LIB_DIR":
            lib_dir = Path(value)
            deps[name] = _path_fingerprint(str(lib_dir))
            deps["HDF5_libhdf5_a"] = _path_fingerprint(str(lib_dir / "libhdf5.a"))
    return deps


@functools.lru_cache(maxsize=1)
def _runtime_abi_fingerprint() -> dict[str, Any]:
    return {
        "libc": platform.libc_ver(),
        "machine": platform.machine(),
        "platform": platform.platform(),
    }


@functools.lru_cache(maxsize=1)
def _harness_code_fingerprint() -> dict[str, Any]:
    """Fingerprint the Python evaluator/applier code that interprets candidates."""
    root = Path(__file__).resolve().parent
    files = sorted(path for path in root.glob("*.py") if path.is_file())
    hasher = hashlib.sha256()
    for path in files:
        rel = path.name
        try:
            data = path.read_bytes()
        except OSError as exc:
            hasher.update(f"{rel}\0ERROR:{type(exc).__name__}:{exc}\0".encode("utf-8", errors="replace"))
            continue
        hasher.update(rel.encode("utf-8", errors="surrogateescape"))
        hasher.update(b"\0")
        hasher.update(hashlib.sha256(data).digest())
        hasher.update(b"\0")
    return {"package": "discover_compbio.rawhash2_native", "sha256": hasher.hexdigest(), "files": len(files)}


@functools.lru_cache(maxsize=64)
def _source_tree_fingerprint(root_text: str) -> dict[str, Any]:
    root = Path(root_text)
    if not root.exists():
        return {"path": root_text, "missing": True}
    hasher = hashlib.sha256()
    files: list[Path] = []
    for rel_root in ("src", "test/scripts"):
        directory = root / rel_root
        if directory.exists():
            files.extend(
                path
                for path in directory.rglob("*")
                if path.is_file() and not path.name.endswith((".o", ".a"))
            )
    for name in ("Makefile", "makefile", "GNUmakefile"):
        path = root / name
        if path.is_file():
            files.append(path)
    for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        rel = path.relative_to(root).as_posix()
        try:
            data = path.read_bytes()
        except OSError as exc:
            hasher.update(f"{rel}\0ERROR:{type(exc).__name__}:{exc}\0".encode("utf-8", errors="replace"))
            continue
        hasher.update(rel.encode("utf-8", errors="surrogateescape"))
        hasher.update(b"\0")
        hasher.update(hashlib.sha256(data).digest())
        hasher.update(b"\0")
    return {"path": str(root), "sha256": hasher.hexdigest(), "files": len(files)}


@functools.lru_cache(maxsize=512)
def _path_fingerprint(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    if not path.exists():
        return {"path": path_text, "missing": True}
    if path.is_dir():
        hasher = hashlib.sha256()
        count = 0
        for child in sorted((p for p in path.rglob("*") if p.is_file()), key=lambda p: p.relative_to(path).as_posix()):
            rel = child.relative_to(path).as_posix()
            try:
                stat = child.stat()
            except OSError as exc:
                hasher.update(f"{rel}\0ERROR:{type(exc).__name__}:{exc}\0".encode("utf-8", errors="replace"))
                continue
            count += 1
            hasher.update(rel.encode("utf-8", errors="surrogateescape"))
            hasher.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\0".encode("ascii"))
        return {"path": str(path), "kind": "dir", "sha256": hasher.hexdigest(), "files": count}
    stat = path.stat()
    details: dict[str, Any] = {
        "path": str(path),
        "kind": "file",
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if stat.st_size <= 128 * 1024 * 1024:
        details["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return details


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _available_memory_gb() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1]) / (1024**2)
    return None


def _existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _read_tail(path: Path, max_bytes: int) -> str:
    if max_bytes <= 0 or not path.exists():
        return ""
    with path.open("rb") as f:
        try:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
        except OSError:
            f.seek(0)
        return f.read(max_bytes).decode("utf-8", errors="replace")


def write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.write_text(json.dumps(metrics, sort_keys=True, indent=2) + "\n", encoding="utf-8")
