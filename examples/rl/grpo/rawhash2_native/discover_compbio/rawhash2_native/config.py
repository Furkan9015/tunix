"""Configuration for the native RawHash2 RawBench benchmark harness."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import os
from pathlib import Path
import shlex
from typing import Any


DATASET_ID = "nappenstance/rawbench_hsapiens"
DATASET_COMMIT = "2f1f08b7e43941db3857d3fabb4b46e58f82bc94"
DATASET_URL = "https://huggingface.co/datasets/nappenstance/rawbench_hsapiens/tree/main"
REFERENCE_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hs1/bigZips/hs1.fa.gz"
REFERENCE_FASTA = "hsapiens.fa"
DATASET_FILES = (
    "PAO89685_pass__2264ba8c_afee3a87_1.0_275.fast5",
    "PAO89685_pass__2264ba8c_afee3a87_9.0_207.fast5",
    "PAO89685_pass__2264ba8c_afee3a87_26.0_418.fast5",
    "PAO89685_pass__2264ba8c_afee3a87_32.0_225.fast5",
)


@dataclass(frozen=True)
class NativeBenchmarkConfig:
    rawhash2_repo: Path
    output_root: Path
    reference_fasta: Path
    fast5_dir: Path
    pore_model: Path
    truth_paf: Path | None
    baseline_json: Path | None
    eval_scripts_dir: Path | None = None
    baseline_metrics: dict[str, Any] | None = None
    preset: str = "sensitive"
    threads: int = 128
    bp_per_sec: int = 400
    extra_params: tuple[str, ...] = ("-w", "0")
    build_command: tuple[str, ...] = ("make", "PROFILE=1", "NOPOD5=1", "NOHDF5=0", "NOSLOW5=1")
    build_jobs: int = 8
    timeout_seconds: int = 24 * 60 * 60
    enable_execution: bool = False
    keep_workdir: bool = False
    require_truth_for_reward: bool = True
    enable_debug_feedback: bool = False
    debug_feedback_params: tuple[str, ...] = ("--output-chains",)
    debug_feedback_read: str | None = None
    log_tail_bytes: int = 16_384
    min_available_memory_gb: float = 64.0
    min_output_free_disk_gb: float = 32.0
    allow_low_resources: bool = False
    benchmark_concurrency: int = 0
    map_concurrency: int = 0
    result_cache_path: Path | None = None
    result_cache_stale_seconds: int = 6 * 60 * 60
    resource_retry_seconds: int = 60 * 60
    resource_retry_interval_seconds: int = 15
    resource_kill_retry_attempts: int = 1

    @property
    def rawhash2_binary_name(self) -> str:
        return "rawhash2"


def default_config() -> NativeBenchmarkConfig:
    workspace = Path(os.environ.get("COMPBIO_WORKSPACE", "/home/furka/compbio"))
    rawhash2_repo = Path(os.environ.get("RAWHASH2_NATIVE_REPO", workspace / "RawHash2"))
    output_root = Path(os.environ.get("RAWHASH2_NATIVE_OUTDIR", "/tmp/rawhash2_native_rlvr"))
    reference_fasta = Path(os.environ.get("RAWHASH2_NATIVE_REF", workspace / "refs/hsapiens.fa"))
    fast5_dir = Path(os.environ.get("RAWHASH2_NATIVE_FAST5_DIR", workspace / "fast5/hsapiens"))
    pore_model = Path(
        os.environ.get(
            "RAWHASH2_NATIVE_PORE",
            rawhash2_repo / "extern/local_kmer_models/uncalled_r1041_model_only_means.txt",
        )
    )
    truth_paf_env = os.environ.get("RAWHASH2_NATIVE_TRUTH_PAF")
    baseline_json_env = os.environ.get("RAWHASH2_NATIVE_BASELINE_JSON")
    eval_scripts_dir_env = os.environ.get("RAWHASH2_NATIVE_EVAL_SCRIPTS_DIR")
    extra_params = tuple(shlex.split(os.environ.get("RAWHASH2_NATIVE_EXTRA_PARAMS", "-w 0")))
    debug_feedback_params = tuple(
        shlex.split(os.environ.get("RAWHASH2_NATIVE_DEBUG_FEEDBACK_PARAMS", "--output-chains"))
    )
    build_command = tuple(
        shlex.split(
            os.environ.get(
                "RAWHASH2_NATIVE_BUILD_COMMAND",
                "make PROFILE=1 NOPOD5=1 NOHDF5=0 NOSLOW5=1",
            )
        )
    )

    return NativeBenchmarkConfig(
        rawhash2_repo=rawhash2_repo,
        output_root=output_root,
        reference_fasta=reference_fasta,
        fast5_dir=fast5_dir,
        pore_model=pore_model,
        truth_paf=Path(truth_paf_env) if truth_paf_env else None,
        baseline_json=Path(baseline_json_env) if baseline_json_env else None,
        eval_scripts_dir=(
            Path(eval_scripts_dir_env)
            if eval_scripts_dir_env
            else workspace / "RawHash2/test/scripts"
        ),
        baseline_metrics=None,
        preset=os.environ.get("RAWHASH2_NATIVE_PRESET", "sensitive"),
        threads=int(os.environ.get("RAWHASH2_NATIVE_THREADS", "128")),
        bp_per_sec=int(os.environ.get("RAWHASH2_NATIVE_BP_PER_SEC", "400")),
        extra_params=extra_params,
        build_command=build_command,
        build_jobs=int(os.environ.get("RAWHASH2_NATIVE_BUILD_JOBS", "8")),
        timeout_seconds=int(os.environ.get("RAWHASH2_NATIVE_TIMEOUT_SECONDS", str(24 * 60 * 60))),
        enable_execution=os.environ.get("RAWHASH2_NATIVE_ENABLE_EXECUTION", "0") == "1",
        keep_workdir=os.environ.get("RAWHASH2_NATIVE_KEEP_WORKDIR", "0") == "1",
        require_truth_for_reward=os.environ.get("RAWHASH2_NATIVE_REQUIRE_TRUTH", "1") == "1",
        enable_debug_feedback=os.environ.get("RAWHASH2_NATIVE_ENABLE_DEBUG_FEEDBACK", "0") == "1",
        debug_feedback_params=debug_feedback_params,
        debug_feedback_read=os.environ.get("RAWHASH2_NATIVE_DEBUG_READ"),
        log_tail_bytes=int(os.environ.get("RAWHASH2_NATIVE_LOG_TAIL_BYTES", "16384")),
        min_available_memory_gb=float(os.environ.get("RAWHASH2_NATIVE_MIN_AVAILABLE_MEM_GB", "64")),
        min_output_free_disk_gb=float(os.environ.get("RAWHASH2_NATIVE_MIN_OUTPUT_FREE_DISK_GB", "32")),
        allow_low_resources=os.environ.get("RAWHASH2_NATIVE_ALLOW_LOW_RESOURCES", "0") == "1",
        benchmark_concurrency=int(os.environ.get("RAWHASH2_NATIVE_BENCHMARK_CONCURRENCY", "0")),
        map_concurrency=int(os.environ.get("RAWHASH2_NATIVE_MAP_CONCURRENCY", "0")),
        result_cache_path=(
            Path(cache_path)
            if (cache_path := os.environ.get("RAWHASH2_NATIVE_RESULT_CACHE_PATH"))
            else None
        ),
        result_cache_stale_seconds=int(
            os.environ.get("RAWHASH2_NATIVE_RESULT_CACHE_STALE_SECONDS", str(6 * 60 * 60))
        ),
        resource_retry_seconds=int(os.environ.get("RAWHASH2_NATIVE_RESOURCE_RETRY_SECONDS", str(60 * 60))),
        resource_retry_interval_seconds=int(
            os.environ.get("RAWHASH2_NATIVE_RESOURCE_RETRY_INTERVAL_SECONDS", "15")
        ),
        resource_kill_retry_attempts=int(os.environ.get("RAWHASH2_NATIVE_RESOURCE_KILL_RETRY_ATTEMPTS", "1")),
    )


def config_from_case_spec(spec: dict[str, Any] | None, base: NativeBenchmarkConfig | None = None) -> NativeBenchmarkConfig:
    """Overlay task/State-injected benchmark settings on the environment config."""
    cfg = base or default_config()
    if not spec:
        return cfg

    updates: dict[str, Any] = {}
    path_fields = {
        "rawhash2_repo": "rawhash2_repo",
        "output_root": "output_root",
        "reference_fasta": "reference_fasta",
        "fast5_dir": "fast5_dir",
        "pore_model": "pore_model",
        "truth_paf": "truth_paf",
        "baseline_json": "baseline_json",
        "eval_scripts_dir": "eval_scripts_dir",
    }
    for key, field in path_fields.items():
        value = spec.get(key)
        if value:
            updates[field] = Path(str(value))

    scalar_fields = {
        "preset": str,
        "threads": int,
        "bp_per_sec": int,
        "build_jobs": int,
        "timeout_seconds": int,
        "enable_execution": _as_bool,
        "keep_workdir": _as_bool,
        "require_truth_for_reward": _as_bool,
        "enable_debug_feedback": _as_bool,
        "debug_feedback_read": str,
        "log_tail_bytes": int,
        "min_available_memory_gb": float,
        "min_output_free_disk_gb": float,
        "allow_low_resources": _as_bool,
        "benchmark_concurrency": int,
        "map_concurrency": int,
        "result_cache_stale_seconds": int,
        "resource_retry_seconds": int,
        "resource_retry_interval_seconds": int,
        "resource_kill_retry_attempts": int,
    }
    for key, caster in scalar_fields.items():
        if key in spec and spec[key] is not None:
            updates[key] = caster(spec[key])

    if "extra_params" in spec and spec["extra_params"] is not None:
        updates["extra_params"] = tuple(str(x) for x in spec["extra_params"])
    if "debug_feedback_params" in spec and spec["debug_feedback_params"] is not None:
        updates["debug_feedback_params"] = tuple(str(x) for x in spec["debug_feedback_params"])
    if "build_command" in spec and spec["build_command"] is not None:
        updates["build_command"] = tuple(str(x) for x in spec["build_command"])
    if "baseline_metrics" in spec and isinstance(spec["baseline_metrics"], dict):
        updates["baseline_metrics"] = spec["baseline_metrics"]
    if "result_cache_path" in spec and spec["result_cache_path"]:
        updates["result_cache_path"] = Path(str(spec["result_cache_path"]))

    return replace(cfg, **updates)


def case_spec_from_config(cfg: NativeBenchmarkConfig | None = None) -> dict[str, Any]:
    """Serialize the active RawHash2 benchmark contract into a task payload."""
    cfg = cfg or default_config()
    spec: dict[str, Any] = {
        "rawhash2_repo": str(cfg.rawhash2_repo),
        "output_root": str(cfg.output_root),
        "reference_fasta": str(cfg.reference_fasta),
        "fast5_dir": str(cfg.fast5_dir),
        "pore_model": str(cfg.pore_model),
        "preset": cfg.preset,
        "threads": cfg.threads,
        "bp_per_sec": cfg.bp_per_sec,
        "extra_params": list(cfg.extra_params),
        "build_command": list(cfg.build_command),
        "build_jobs": cfg.build_jobs,
        "timeout_seconds": cfg.timeout_seconds,
        "enable_execution": cfg.enable_execution,
        "keep_workdir": cfg.keep_workdir,
        "require_truth_for_reward": cfg.require_truth_for_reward,
        "enable_debug_feedback": cfg.enable_debug_feedback,
        "debug_feedback_params": list(cfg.debug_feedback_params),
        "debug_feedback_read": cfg.debug_feedback_read,
        "log_tail_bytes": cfg.log_tail_bytes,
        "min_available_memory_gb": cfg.min_available_memory_gb,
        "min_output_free_disk_gb": cfg.min_output_free_disk_gb,
        "allow_low_resources": cfg.allow_low_resources,
        "benchmark_concurrency": cfg.benchmark_concurrency,
        "map_concurrency": cfg.map_concurrency,
        "result_cache_stale_seconds": cfg.result_cache_stale_seconds,
        "resource_retry_seconds": cfg.resource_retry_seconds,
        "resource_retry_interval_seconds": cfg.resource_retry_interval_seconds,
        "resource_kill_retry_attempts": cfg.resource_kill_retry_attempts,
    }
    if cfg.result_cache_path:
        spec["result_cache_path"] = str(cfg.result_cache_path)
    if cfg.truth_paf:
        spec["truth_paf"] = str(cfg.truth_paf)
    if cfg.baseline_json:
        spec["baseline_json"] = str(cfg.baseline_json)
    if cfg.baseline_metrics:
        spec["baseline_metrics"] = cfg.baseline_metrics
    return spec


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
