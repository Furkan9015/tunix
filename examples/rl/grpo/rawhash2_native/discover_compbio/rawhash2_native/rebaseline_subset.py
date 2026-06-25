import json, sys, yaml
from pathlib import Path
sys.path.insert(0, "/home/furka/compbio_rlvr/src/discover")
from discover_compbio.rawhash2_native.config import config_from_case_spec
from discover_compbio.rawhash2_native.runner import run_baseline_benchmark

CFG = "/home/furka/compbio_rlvr/src/discover/discover_compbio/configs/qwen3p6_27b_v6e8_rawhash2_native_opt.yaml"
case = dict(yaml.safe_load(open(CFG))["data_config"]["rawhash2_case"])
case["fast5_dir"] = "/home/furka/compbio/fast5/hsapiens_subset800"
case["enable_debug_feedback"] = False
cfg = config_from_case_spec(case)
print("re-baselining on subset:", cfg.fast5_dir)
metrics = run_baseline_benchmark(config=cfg)
out = "/home/furka/compbio/outputs/rawhash2_native/baseline_metrics_isolated_t128_subset800.json"
Path(out).parent.mkdir(parents=True, exist_ok=True)
Path(out).write_text(json.dumps(metrics, sort_keys=True, indent=2) + "\n")
print("BASELINE ok=%s map_elapsed=%s index_elapsed=%s map_max_rss_kb=%s" % (
    metrics.get("ok"), metrics.get("map_elapsed_seconds"),
    metrics.get("index_elapsed_seconds"), metrics.get("map_max_rss_kb")))
print("wrote", out)
