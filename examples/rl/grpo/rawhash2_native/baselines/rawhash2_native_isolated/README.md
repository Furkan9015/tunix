# RawHash2 Native Isolated Baseline

This is the isolated RawHash2-compatible source artifact for the native RLVR
case. It is not a Python wrapper and it does not shell out to another RawHash2
binary. It builds the real native indexing, signal segmentation, seeding,
chaining, DTW alignment, and PAF-emitting CLI from source.

The artifact is intentionally smaller than the upstream repository. It keeps:

- `src/`: exact standalone native source files needed by the benchmark CLI.
- `src/Makefile`: standalone RawHash2 build rules and compile flags.
- `extern/local_kmer_models/uncalled_r1041_model_only_means.txt`: R10.4.1 pore
  model used by the benchmark.
- `LICENSE`: upstream license.

It omits upstream `.git`, docs, tests, figures, live gRPC/proto/CMake support,
download scripts, previous binaries, object files, and build directories.

Build contract:

```bash
make PROFILE=1 NOPOD5=1 NOHDF5=0 NOSLOW5=1 \
  HDF5_INCLUDE_DIR=/home/furka/compbio/tools/hdf5-release/hdf5-1.10.11/build/include \
  HDF5_LIB_DIR=/home/furka/compbio/tools/hdf5-release/hdf5-1.10.11/build/lib
```

The benchmark runner expects either `bin/rawhash2` or `src/rawhash2`; this
artifact produces both during the normal build/copy flow.

Use this path in the native RLVR config:

```yaml
data_config:
  rawhash2_case:
    rawhash2_repo: "/home/furka/compbio/discover/baselines/rawhash2_native_isolated"
    pore_model: "/home/furka/compbio/discover/baselines/rawhash2_native_isolated/extern/local_kmer_models/uncalled_r1041_model_only_means.txt"
```

For model-facing source context, use:

```bash
export RAWHASH2_NATIVE_PROMPT_SOURCE_MODE=isolated
export RAWHASH2_NATIVE_PROMPT_SOURCE_CHAR_BUDGET=18000
```

The runner still evaluates candidates by copying this tree to an isolated work
directory, applying a `src/*` patch, rebuilding, indexing CHM13, and mapping
the fixed RawBench R10.4.1 FAST5 input with the same CLI contract as the
upstream RawHash2 baseline.
