# Native RawHash2 RLVR Case

This case evaluates model-generated patches against the real RawHash2 C/C++
codebase. It is intended for fair comparison to the native baseline, not for a
Python reimplementation.

The benchmark copies `RawHash2/` to an isolated work directory, applies a
candidate source change that only touches `src/*`, builds `rawhash2`, and runs
the R10.4.1 human FAST5 command. The preferred model output is a structured
exact edit such as:

```json
{
  "edits": [
    {
      "path": "src/rsketch.c",
      "find": "exact text copied from the shown source",
      "replace": "replacement text"
    }
  ]
}
```

The evaluator converts these edits into a unified diff before applying them.
Raw unified diffs are still accepted as a fallback, but malformed hunk syntax is
scored as zero before build.

The resulting candidate build/run uses:

```bash
PARAMS="-w 0"
PRESET="sensitive"
PROG="$candidate_repo/bin/rawhash2"

/usr/bin/time -vpo "$OUT/${PREFIX}_rawhash2_index_${PRESET}_quant.time" \
  "$PROG" --bp-per-sec 400 --r10 -x "$PRESET" -t 128 \
  -p "$RAWHASH2_NATIVE_PORE" \
  -d "$OUT/${PREFIX}_rawhash2_${PRESET}_quant.ind" \
  $PARAMS "$RAWHASH2_NATIVE_REF"

/usr/bin/time -vpo "$OUT/${PREFIX}_rawhash2_map_${PRESET}_quant_map.time" \
  "$PROG" --bp-per-sec 400 --r10 -x "$PRESET" -t 128 \
  -o "$OUT/${PREFIX}_rawhash2_${PRESET}_quant.paf" \
  $PARAMS "$OUT/${PREFIX}_rawhash2_${PRESET}_quant.ind" \
  "$RAWHASH2_NATIVE_FAST5_DIR"
```

The dedicated Tunix YAML pins the current high-memory TPU smoke target in
`data_config.rawhash2_case`: one FAST5 file under
`/home/furka/compbio/fast5/hsapiens_single`, CHM13 `hsapiens.fa`, `-t 128`, no
truth PAF yet, debug `--output-chains`, and baseline metrics at
`/home/furka/compbio/outputs/rawhash2_native/baseline_metrics_remote_t128_single.json`.
For the standard Discover environment path, set equivalent environment values:

```bash
export RAWHASH2_NATIVE_ENABLE_EXECUTION=1
export RAWHASH2_NATIVE_REPO=/home/furka/compbio/discover/baselines/rawhash2_native_isolated
export RAWHASH2_NATIVE_REF=/home/furka/compbio/refs/hsapiens.fa
export RAWHASH2_NATIVE_FAST5_DIR=/home/furka/compbio/fast5/hsapiens_single
export RAWHASH2_NATIVE_PORE=/home/furka/compbio/discover/baselines/rawhash2_native_isolated/extern/local_kmer_models/uncalled_r1041_model_only_means.txt
export RAWHASH2_NATIVE_BASELINE_JSON=/home/furka/compbio/outputs/rawhash2_native/baseline_metrics_remote_t128_single.json
export RAWHASH2_NATIVE_THREADS=128
export RAWHASH2_NATIVE_BUILD_JOBS=128
export RAWHASH2_NATIVE_REQUIRE_TRUTH=0
export RAWHASH2_NATIVE_ENABLE_DEBUG_FEEDBACK=1
export RAWHASH2_NATIVE_DEBUG_FEEDBACK_PARAMS="--output-chains"
```

The Hugging Face FAST5 dataset does not include basecalled reads or a truth PAF.
For accuracy reward, provide a matching externally generated PAF. For native
speed/RSS/debug probes only, set `RAWHASH2_NATIVE_REQUIRE_TRUTH=0`; this keeps
the run useful for baseline characterization but does not produce comparable F1.

Reference setup:

```bash
download_hsapiens() {
    echo "=== Downloading Human CHM13v2 ==="
    mkdir -p /home/furka/compbio/refs
    cd /home/furka/compbio/refs
    wget -O hsapiens.fa.gz \
        "https://hgdownload.soe.ucsc.edu/goldenPath/hs1/bigZips/hs1.fa.gz"
    gunzip hsapiens.fa.gz
    echo "Saved: hsapiens.fa ($(du -h hsapiens.fa | cut -f1))"
}
```

The RawBench dataset is `nappenstance/rawbench_hsapiens` at commit
`2f1f08b7e43941db3857d3fabb4b46e58f82bc94`. Download it with Hugging Face
tooling into `RAWHASH2_NATIVE_FAST5_DIR`.

Run this case explicitly with `task: "rawhash2_native_opt"` or the dedicated
config. It is registered in `TASK_NAMES`, but it is not part of the default
synthetic `mixed` rotation.

The Tunix row's `answer_json` carries a `rawhash2_case` payload with the active
benchmark settings, including execution flag, output root, thread count,
resource guards, debug flags, and optional `baseline_json` path. The scorer
overlays that payload onto the environment config, so reward workers compare
candidates against the injected baseline rather than depending only on
process-global environment variables.

Prompt construction is centralized in `prompt.py`. The highest-signal setup is
to point `rawhash2_case.rawhash2_repo` at an isolated RawHash2-compatible
baseline that exposes the same `rawhash2` CLI but keeps the editable indexing,
segmentation, and alignment code small enough to show exactly. Small source
trees are included as complete editable files, which lets the model produce
patches against real context instead of fabricating diffs from an upstream
digest.

The current isolated baseline artifact is:

```text
discover/baselines/rawhash2_native_isolated/
```

It contains the exact native standalone source files required for the benchmark
CLI, a standalone root `Makefile`, `src/Makefile`, the R10.4.1 pore model, and
the upstream license. It omits upstream git history, docs, tests, live
gRPC/proto/CMake support, previous binaries, object files, and build
directories. The dedicated Tunix config points both `rawhash2_case.rawhash2_repo`
and `rawhash2_case.pore_model` at this artifact.

For larger prompt-budget experiments that should show the model baseline source
from an isolated artifact, use:

```bash
export RAWHASH2_NATIVE_PROMPT_SOURCE_MODE=isolated
export RAWHASH2_NATIVE_PROMPT_SOURCE_CHAR_BUDGET=18000
```

For full upstream RawHash2 experiments, use the structural source reducer:

```bash
export RAWHASH2_NATIVE_PROMPT_INCLUDE_SOURCE=1
export RAWHASH2_NATIVE_PROMPT_SOURCE_REPO=/home/furka/compbio/RawHash2
export RAWHASH2_NATIVE_PROMPT_SOURCE_CHAR_BUDGET=18000
```

In automatic mode, prompt construction uses complete editable files when the
source tree fits. Otherwise it falls back to full build/API headers when
possible, then file/function indexes and hot-path excerpts from the primary
algorithm files. It avoids blind tail truncation, so compile-critical structs,
prototypes, and local code around sketching, seeding, chaining, mapping, event
extraction, signal loading, and DTW remain visible while oversized body text is
omitted.

The same isolated case is also wired as a standard Discover single-problem
environment in `examples/rawhash2_native/env.py`. That path seeds the initial
`State` with baseline metrics from `RAWHASH2_NATIVE_BASELINE_JSON`, injects the
previous patch/score into prompts with `state.to_prompt(...)`, and carries the
baseline in `State.construction["baseline_metrics"]` for subsequent rollouts.

Manual evaluation:

```bash
python -m discover_compbio.rawhash2_native.run_eval \
  --baseline \
  --metrics-json /tmp/rawhash2_baseline_metrics.json

export RAWHASH2_NATIVE_BASELINE_JSON=/tmp/rawhash2_baseline_metrics.json

python -m discover_compbio.rawhash2_native.run_eval \
  --patch-file candidate.diff \
  --metrics-json /tmp/rawhash2_candidate_metrics.json
```

For FAST5 input, build with HDF5 enabled. The default harness build command also
enables RawHash2's `PROFILERH` timers so the reward audit can aggregate file
read, signal/event, sketching, seeding, seed-sort, chaining, chain-sort, and
mapping phase times:

```bash
make PROFILE=1 NOPOD5=1 NOHDF5=0 NOSLOW5=1
```

Override it with `RAWHASH2_NATIVE_BUILD_COMMAND` if you need CMake or
system-HDF5 options. The current TPU YAML pins
`HDF5_INCLUDE_DIR`/`HDF5_LIB_DIR` under
`/home/furka/compbio/tools/hdf5-release/hdf5-1.10.11/build` to match the
validated baseline build.
The isolated source copy drops stale `bin/`, `build/`, `src/rawhash2`, and
`src/*.o` before the build. External dependency caches may still be reused if
they are present in the source tree.

The full human R10.4.1 case needs a high-memory host. By default the harness
checks for at least 64 GB available memory and 32 GB free space under
`RAWHASH2_NATIVE_OUTDIR` before launching the native run. The current
single-FAST5 TPU config raises this to 600 GB available memory and 128 GB free
output space because the validated map pass peaked near 492 GB RSS. Override
only for intentional diagnostics with `RAWHASH2_NATIVE_ALLOW_LOW_RESOURCES=1`.

Optional diagnostic feedback:

```bash
export RAWHASH2_NATIVE_ENABLE_DEBUG_FEEDBACK=1
export RAWHASH2_NATIVE_DEBUG_FEEDBACK_PARAMS="--output-chains"
# Narrow to one read when you need detailed chunk/decision diagnostics:
export RAWHASH2_NATIVE_DEBUG_READ=<read_id>
```

The harness always aggregates scored-run PAF tags such as `mt`, `ci`, `cm`,
`nc`, `s1`, and `sl` into sums/means/maxes. The optional debug feedback pass is
separate from the scored map command. This keeps candidate-vs-baseline timing/RSS
fair while still capturing aggregate `CHAINS`, `DEBUG_READ`, and
`DEBUG_DECISION` summaries plus stderr tails in the metrics JSON and optional
`RAWHASH2_NATIVE_AUDIT_PATH` JSONL.

`CHAINS` records are summarized into model-readable diagnostics rather than raw
counter dumps: unmapped reads with no chains point to seed/anchor loss, unmapped
reads with strong chains point to mapq/decision/ambiguity issues, and close
top-two chain scores point to ambiguous candidate loci. Large debug logs are
parsed with a process pool; the harness passes the benchmark thread count as the
parser worker count so post-map feedback uses the same parallelism as
indexing/mapping. Override with `RAWHASH2_NATIVE_DEBUG_PARSE_WORKERS` only for
local diagnostics.

Validated single-FAST5 baseline: index elapsed `102.05s`, index RSS
`106295372 KB`; scored map elapsed `651.11s`, map RSS `491641756 KB`, PAF reads
`4000`, mapped fraction `0.7725`; debug pass elapsed `633.27s`, debug RSS
`468664752 KB`.
