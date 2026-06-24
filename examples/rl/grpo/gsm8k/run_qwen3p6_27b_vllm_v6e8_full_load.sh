#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
export VLLM_TPU_RPA_VERSION="${VLLM_TPU_RPA_VERSION:-2}"
export DISABLE_MOSAIC_ATTN="${DISABLE_MOSAIC_ATTN:-1}"
export TPU_BACKEND_TYPE="${TPU_BACKEND_TYPE:-jax}"
export HF_TOKEN="${HF_TOKEN:-}"
export TMPDIR="${TMPDIR:-/dev/shm/tmp}"
export HF_HOME="${HF_HOME:-/dev/shm/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/dev/shm/jax_compilation_cache}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-27B}"
MODEL_DIR="${MODEL_DIR:-/dev/shm/models/qwen3p6-27b}"
CONFIG_PATH="${CONFIG_PATH:-examples/rl/grpo/gsm8k/configs/qwen3p6_27b_vllm_v6e8_full_load.yaml}"

mkdir -p "${TMPDIR}" "${HF_HOME}" "${HF_HUB_CACHE}" "${JAX_COMPILATION_CACHE_DIR}"

cd "$(dirname "$0")/../../../.."

python - "${MODEL_ID}" "${MODEL_DIR}" <<'PY'
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

model_id = sys.argv[1]
model_dir = Path(sys.argv[2])
model_dir.mkdir(parents=True, exist_ok=True)

has_config = (model_dir / "config.json").is_file()
has_tokenizer = (model_dir / "tokenizer.json").is_file()
num_shards = len(list(model_dir.glob("model-*-of-00015.safetensors")))
if has_config and has_tokenizer and num_shards == 15:
  print(f"Using existing model snapshot at {model_dir}")
else:
  snapshot_download(
      repo_id=model_id,
      local_dir=str(model_dir),
      token=os.environ.get("HF_TOKEN") or None,
  )
PY

python -m tunix.cli.grpo_main \
  "${CONFIG_PATH}" \
  model_config.model_download_path="${MODEL_DIR}" \
  actor_model_config.model_download_path="${MODEL_DIR}" \
  reference_model_config.model_download_path="${MODEL_DIR}" \
  rollout_model_config.model_download_path="${MODEL_DIR}" \
  tokenizer_config.tokenizer_path="${MODEL_DIR}" \
  vllm_config.model_version="${MODEL_DIR}" \
  "$@"
