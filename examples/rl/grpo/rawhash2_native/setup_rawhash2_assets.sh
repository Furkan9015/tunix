#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TUNIX_DIR=${TUNIX_DIR:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}
WORK=${WORK:-$(dirname "$TUNIX_DIR")}
COMPBIO_ROOT=${COMPBIO_ROOT:-/home/furka/compbio}
DATA_PREFIX=${DATA_PREFIX:-gs://proust-data-euw4/compbio_rlvr/rawhash2_native/20260609}
QWEN_MODEL=${QWEN_MODEL:-Qwen/Qwen3.6-27B}
PERSISTENT_QWEN_DIR=${PERSISTENT_QWEN_DIR:-$WORK/models/qwen3p6-27b}
QWEN_DIR=${QWEN_DIR:-/dev/shm/models/qwen3p6-27b}
VENV=${VENV:-$WORK/vllm_env}
UPGRADE_FLAX_QWIX=${UPGRADE_FLAX_QWIX:-1}
FLAX_SPEC=${FLAX_SPEC:-flax}
QWIX_SPEC=${QWIX_SPEC:-qwix}

if command -v loginctl >/dev/null 2>&1 && command -v systemctl >/dev/null 2>&1; then
  sudo loginctl enable-linger "$USER" || true
  sudo mkdir -p /etc/systemd/logind.conf.d
  printf "[Login]\nRemoveIPC=no\n" | sudo tee /etc/systemd/logind.conf.d/proust.conf >/dev/null
  sudo systemctl restart systemd-logind || true
fi

sudo apt-get update
sudo apt-get install -y build-essential zlib1g-dev git curl rsync gzip ca-certificates

mkdir -p \
  "$WORK/logs" \
  "$WORK/checkpoints" \
  "$WORK/cache/jax_compile" \
  "$COMPBIO_ROOT/refs" \
  "$COMPBIO_ROOT/fast5/hsapiens_subset800" \
  "$COMPBIO_ROOT/outputs/true_mappings" \
  "$COMPBIO_ROOT/outputs/rawhash2_native" \
  "$COMPBIO_ROOT/tools/hdf5-release" \
  "$PERSISTENT_QWEN_DIR" \
  "$QWEN_DIR" \
  /dev/shm/compbio_tmp \
  /dev/shm/rawhash2_native_rlvr

if [[ -e "$COMPBIO_ROOT/discover" && ! -L "$COMPBIO_ROOT/discover" ]]; then
  echo "$COMPBIO_ROOT/discover exists and is not a symlink; refusing to replace it" >&2
  exit 1
fi
ln -sfn "$SCRIPT_DIR" "$COMPBIO_ROOT/discover"

gcs_cp_if_missing() {
  local src="$1"
  local dst="$2"
  if [[ -s "$dst" ]]; then
    echo "asset exists, skipping: $dst"
    return 0
  fi
  mkdir -p "$(dirname "$dst")"
  gcloud storage cp "$src" "$dst"
}

gcs_cp_if_missing "$DATA_PREFIX/hsapiens.fa.gz" "$COMPBIO_ROOT/refs/hsapiens.fa.gz"
gzip -dkf "$COMPBIO_ROOT/refs/hsapiens.fa.gz"
gcs_cp_if_missing "$DATA_PREFIX/subset_800.fast5" "$COMPBIO_ROOT/fast5/hsapiens_subset800/subset_800.fast5"
gcs_cp_if_missing "$DATA_PREFIX/hsapiens_subset800_sup_v5.2.0.paf" "$COMPBIO_ROOT/outputs/true_mappings/hsapiens_subset800_sup_v5.2.0.paf"
gcs_cp_if_missing "$DATA_PREFIX/baseline_metrics_isolated_t128_subset800_truth.json" "$COMPBIO_ROOT/outputs/rawhash2_native/baseline_metrics_isolated_t128_subset800_truth.json"
gcs_cp_if_missing "$DATA_PREFIX/hdf5-1.10.11.tar.gz" "$COMPBIO_ROOT/tools/hdf5-release/hdf5-1.10.11.tar.gz"

download_qwen_model() {
  local target="$1"
  mkdir -p "$target"
  if [[ -s "$target/model.safetensors.index.json" ]]; then
    echo "model exists, skipping: $target"
    return 0
  fi

  set -a
  [[ -f "$WORK/.hf_env" ]] && . "$WORK/.hf_env"
  set +a

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

cd "$COMPBIO_ROOT/tools/hdf5-release"
if [[ ! -s hdf5-1.10.11/build/lib/libhdf5.a ]] ||
   nm hdf5-1.10.11/build/lib/libhdf5.a 2>/dev/null | grep -q "__isoc23_"; then
  rm -rf hdf5-1.10.11
  tar -xzf hdf5-1.10.11.tar.gz
  cd hdf5-1.10.11
  ./configure --prefix="$PWD/build" --enable-static --disable-shared
  make -j"$(nproc)"
  make install
fi

if [[ -x "$VENV/bin/python" ]]; then
  if [[ "$UPGRADE_FLAX_QWIX" == "1" ]]; then
    "$VENV/bin/python" -m pip install --upgrade --no-deps "$FLAX_SPEC" "$QWIX_SPEC"
  fi
  "$VENV/bin/python" -m pip install --no-deps -e "$TUNIX_DIR"
  if [[ ! -s "$QWEN_DIR/model.safetensors.index.json" ]]; then
    download_qwen_model "$PERSISTENT_QWEN_DIR"
    if [[ "$PERSISTENT_QWEN_DIR" != "$QWEN_DIR" ]]; then
      rsync -a --delete "$PERSISTENT_QWEN_DIR"/ "$QWEN_DIR"/
    fi
  fi
else
  echo "venv not found at $VENV; skipping Tunix install and model download" >&2
fi

echo "rawhash2 assets ready"
echo "compbio_root=$COMPBIO_ROOT"
echo "persistent_qwen_dir=$PERSISTENT_QWEN_DIR"
echo "qwen_dir=$QWEN_DIR"
echo "tunix_dir=$TUNIX_DIR"
