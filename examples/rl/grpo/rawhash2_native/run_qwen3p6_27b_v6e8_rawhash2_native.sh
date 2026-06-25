#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TUNIX_DIR=${TUNIX_DIR:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}
WORK=${WORK:-$(dirname "$TUNIX_DIR")}
TPU_INFERENCE_DIR=${TPU_INFERENCE_DIR:-$WORK/tpu-inference}
VENV=${VENV:-$WORK/vllm_env}
RUN=${RUN:-qwen3p6_27b_v6e8_rawhash2_native}
CONFIG=${CONFIG:-$SCRIPT_DIR/configs/qwen3p6_27b_v6e8_rawhash2_native_opt.yaml}
LOG=${LOG:-$WORK/logs/rlvr_${RUN}.log}
COMPBIO_AUDIT=${COMPBIO_AUDIT:-$WORK/logs/rawhash2_compbio_reward_${RUN}.jsonl}
RAWHASH_AUDIT=${RAWHASH_AUDIT:-$WORK/logs/rawhash2_native_audit_${RUN}.jsonl}
QWEN_MODEL=${QWEN_MODEL:-Qwen/Qwen3.6-27B}
PERSISTENT_QWEN_DIR=${PERSISTENT_QWEN_DIR:-$WORK/models/qwen3p6-27b}
QWEN_DIR=${QWEN_DIR:-/dev/shm/models/qwen3p6-27b}
CLEAN_RUNTIME_CACHE=${CLEAN_RUNTIME_CACHE:-0}

if [[ "$CLEAN_RUNTIME_CACHE" == "1" ]]; then
  rm -rf \
    "$QWEN_DIR" \
    "/dev/shm/compbio_vllm_root_${RUN}" \
    "/dev/shm/compbio_vllm_xla_${RUN}" \
    /dev/shm/compbio_tmp \
    /dev/shm/rawhash2_native_rlvr
fi

mkdir -p \
  "$WORK/logs" \
  "$WORK/cache/jax_compile" \
  "$PERSISTENT_QWEN_DIR" \
  "$QWEN_DIR" \
  "/dev/shm/compbio_vllm_root_${RUN}" \
  "/dev/shm/compbio_vllm_xla_${RUN}" \
  /dev/shm/compbio_tmp \
  /dev/shm/rawhash2_native_rlvr

for required in \
  /home/furka/compbio/discover/baselines/rawhash2_native_isolated/src/main.c \
  /home/furka/compbio/discover/baselines/rawhash2_native_isolated/src/Makefile \
  /home/furka/compbio/discover/baselines/rawhash2_native_isolated/extern/local_kmer_models/uncalled_r1041_model_only_means.txt \
  /home/furka/compbio/refs/hsapiens.fa \
  /home/furka/compbio/fast5/hsapiens_subset800/subset_800.fast5 \
  /home/furka/compbio/outputs/true_mappings/hsapiens_subset800_sup_v5.2.0.paf \
  /home/furka/compbio/outputs/rawhash2_native/baseline_metrics_isolated_t128_subset800_truth.json \
  /home/furka/compbio/tools/hdf5-release/hdf5-1.10.11/build/include/hdf5.h \
  /home/furka/compbio/tools/hdf5-release/hdf5-1.10.11/build/lib/libhdf5.a; do
  if [[ ! -e "$required" ]]; then
    echo "missing required RawHash2 input: $required" >&2
    exit 1
  fi
done

set -a
[[ -f "$WORK/.hf_env" ]] && . "$WORK/.hf_env"
set +a

download_qwen_model() {
  local target="$1"
  mkdir -p "$target"
  if [[ -s "$target/model.safetensors.index.json" ]]; then
    echo "model exists, skipping: $target"
    return 0
  fi

  if [[ -x "$VENV/bin/hf" ]]; then
    "$VENV/bin/hf" download "$QWEN_MODEL" --local-dir "$target"
  else
    "$VENV/bin/python" - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${QWEN_MODEL}",
    local_dir="${target}",
    local_dir_use_symlinks=False,
)
PY
  fi
}

if [[ ! -s "$QWEN_DIR/model.safetensors.index.json" ]]; then
  download_qwen_model "$PERSISTENT_QWEN_DIR"
  if [[ "$PERSISTENT_QWEN_DIR" != "$QWEN_DIR" ]]; then
    rsync -a --delete "$PERSISTENT_QWEN_DIR"/ "$QWEN_DIR"/
  fi
fi

export HF_TOKEN="${HF_TOKEN:-}"
export PYTHONPATH="$SCRIPT_DIR:$TUNIX_DIR:$TPU_INFERENCE_DIR:${PYTHONPATH:-}"
export COMPBIO_REWARD_AUDIT_PATH="$COMPBIO_AUDIT"
export RAWHASH2_NATIVE_AUDIT_PATH="$RAWHASH_AUDIT"
export RAWHASH2_NATIVE_PROMPT_SOURCE_MODE=isolated
export RAWHASH2_NATIVE_PROMPT_SOURCE_CHAR_BUDGET=11000
export RAWHASH2_NATIVE_TRUTH_PAF=/home/furka/compbio/outputs/true_mappings/hsapiens_subset800_sup_v5.2.0.paf
export COMPBIO_REWARD_SUBPROCESS_TIMEOUT_SECONDS=${COMPBIO_REWARD_SUBPROCESS_TIMEOUT_SECONDS:-2400}
export COMPBIO_REWARD_ISOLATE_SCORE=${COMPBIO_REWARD_ISOLATE_SCORE:-1}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export MALLOC_ARENA_MAX=2
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR="$WORK/cache/jax_compile"
export JAX_COMPILATION_CACHE_MAX_SIZE=10000000000
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export VLLM_CACHE_ROOT="/dev/shm/compbio_vllm_root_${RUN}"
export VLLM_XLA_CACHE_PATH="/dev/shm/compbio_vllm_xla_${RUN}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export JAX_TRACEBACK_FILTERING=off
export WANDB_MODE=disabled
export WANDB_SILENT=true
export TOKENIZERS_PARALLELISM=false
export TMPDIR=/dev/shm/compbio_tmp

ulimit -s 2048 || true
rm -f "$LOG" "$COMPBIO_AUDIT" "$RAWHASH_AUDIT"

echo "log=$LOG"
exec "$VENV/bin/python" -u -m tunix.cli.grpo_main "$CONFIG" 2>&1 | tee "$LOG"
