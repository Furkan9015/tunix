"""CLI for running a native RawHash2 patch evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from discover_compbio.rawhash2_native.evaluator import evaluate_patch
from discover_compbio.rawhash2_native.runner import run_baseline_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a RawHash2 native patch on RawBench.")
    parser.add_argument("--patch-file", type=Path)
    parser.add_argument("--baseline", action="store_true", help="run the unmodified RawHash2 baseline")
    parser.add_argument("--metrics-json", type=Path)
    args = parser.parse_args()

    if args.baseline:
        metrics = run_baseline_benchmark()
    elif args.patch_file:
        metrics = evaluate_patch(args.patch_file.read_text(encoding="utf-8"))
    else:
        parser.error("provide --patch-file or --baseline")
    text = json.dumps(metrics, sort_keys=True, indent=2)
    if args.metrics_json:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
