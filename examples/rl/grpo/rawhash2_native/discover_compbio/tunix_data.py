"""Tunix data_module for Discover computational biology RLVR tasks."""

from __future__ import annotations

import json
import os
from typing import Any

import grain

from discover_compbio.tasks import make_examples
from discover_compbio.tasks import _use_rawhash2_prompt_archive
from discover_compbio.rawhash2_native.prompt_archive import build_prompt_archive_context


def create_dataset(
    task: str = "mixed",
    num_examples: int = 128,
    split: str = "train",
    seed: int = 0,
    rawhash2_case: dict[str, Any] | None = None,
    rawhash2_prompt_archive_path: str | None = None,
    rawhash2_reward_audit_path: str | None = None,
    rawhash2_prompt_archive_fraction: float = 0.0,
    rawhash2_prompt_archive_max_chars: int = 6000,
    rawhash2_prompt_archive_dynamic: bool = False,
    rawhash2_prompt_archive_warmup_examples: int = 0,
    **_: Any,
):
    rows = make_examples(
        task=task,
        num_examples=num_examples,
        split=split,
        seed=seed,
        rawhash2_case=rawhash2_case,
        rawhash2_prompt_archive_path=rawhash2_prompt_archive_path,
        rawhash2_reward_audit_path=rawhash2_reward_audit_path,
        rawhash2_prompt_archive_fraction=rawhash2_prompt_archive_fraction,
        rawhash2_prompt_archive_max_chars=rawhash2_prompt_archive_max_chars,
        rawhash2_prompt_archive_dynamic=rawhash2_prompt_archive_dynamic,
        rawhash2_prompt_archive_warmup_examples=rawhash2_prompt_archive_warmup_examples,
    )
    return grain.MapDataset.source(rows)


def batch_fn(elements: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Batch rows and inject RawHash2 prompt archive context at batch time."""
    patched = [_maybe_inject_rawhash2_archive(dict(item)) for item in elements]
    batched: dict[str, list[Any]] = {}
    for key in patched[0].keys():
        batched[key] = [item[key] for item in patched]
    return batched


def _maybe_inject_rawhash2_archive(item: dict[str, Any]) -> dict[str, Any]:
    if item.get("task") != "rawhash2_native_opt":
        return item
    prompt = item.get("prompts")
    instance_json = item.get("instance_json")
    if not isinstance(prompt, str) or not isinstance(instance_json, str):
        return item
    try:
        instance = json.loads(instance_json)
    except json.JSONDecodeError:
        return item
    archive_cfg = instance.get("rawhash2_prompt_archive")
    if not isinstance(archive_cfg, dict) or not archive_cfg.get("dynamic"):
        return item

    idx = int(archive_cfg.get("archive_variant", instance.get("example_index", 0)) or 0)
    warmup_examples = int(archive_cfg.get("warmup_examples", 0) or 0)
    if idx < warmup_examples:
        return item
    fraction = float(archive_cfg.get("fraction", 0.0) or 0.0)
    if not _use_rawhash2_prompt_archive(idx, fraction):
        return item

    source_files = archive_cfg.get("source_files")
    if not isinstance(source_files, list):
        source_files = instance.get("editable_source_files")
    context = build_prompt_archive_context(
        native_audit_path=archive_cfg.get("native_audit_path") or os.environ.get("RAWHASH2_NATIVE_AUDIT_PATH"),
        reward_audit_path=archive_cfg.get("reward_audit_path") or os.environ.get("COMPBIO_REWARD_AUDIT_PATH"),
        source_files=[str(path) for path in source_files or ()],
        max_chars=int(archive_cfg.get("max_chars", 6000) or 6000),
        archive_variant=idx,
    )
    if not context.strip():
        return item
    item["prompts"] = _insert_archive_context(prompt, context.strip())
    return item


def _insert_archive_context(prompt: str, context: str) -> str:
    if "Prior verified RawHash2 candidates on this same benchmark:" in prompt:
        return prompt
    for marker in (
        "Exact editable baseline source (complete files):",
        "Baseline RawHash2 source context (budgeted digest):",
    ):
        pos = prompt.find(marker)
        if pos >= 0:
            return f"{prompt[:pos]}{context}\n\n{prompt[pos:]}"
    return f"{prompt.rstrip()}\n\n{context}\n"
