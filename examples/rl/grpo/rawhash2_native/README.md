# RawHash2 Native RLVR

This example vendors the RawHash2 native optimization case used by the compbio
RLVR runs. Large benchmark assets are not checked in; stage them from GCS:

```bash
examples/rl/grpo/rawhash2_native/setup_rawhash2_assets.sh
```

The setup script uses:

```text
gs://proust-data-euw4/compbio_rlvr/rawhash2_native/20260609
```

Run the single-host v6e-8 Qwen3.6-27B job:

```bash
examples/rl/grpo/rawhash2_native/run_qwen3p6_27b_v6e8_rawhash2_native.sh
```

The config keeps the old `/home/furka/compbio` benchmark layout and uses the
vendored baseline through `/home/furka/compbio/discover`.
