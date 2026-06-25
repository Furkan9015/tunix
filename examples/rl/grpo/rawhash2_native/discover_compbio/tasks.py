"""Small verifiable computational biology tasks for Tunix GRPO.

The tasks in this module are intentionally dependency-light. They produce
deterministic synthetic instances and score model completions without importing
Tinker or executing generated code.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
from typing import Any


DNA = "ACGT"
TASK_NAMES = ("read_mapping", "msa", "genome_assembly", "phylogeny", "rawhash2_native_opt")
DEFAULT_MIXED_TASKS = ("read_mapping", "msa", "genome_assembly", "phylogeny")
RAWHASH2_NATIVE_SOURCE_SLICES = (
    (
        "src/lchain.c",
        "src/chain.h",
        "src/rmap.h",
    ),
    (
        "src/rseed.c",
        "src/rseed.h",
        "src/rindex.h",
        "src/rawhash.h",
    ),
    (
        "src/rmap.c",
        "src/rmap.h",
        "src/dtw.c",
        "src/dtw.h",
    ),
    (
        "src/rindex.c",
        "src/rindex.h",
        "src/kalloc.c",
        "src/kalloc.h",
        "src/kthread.c",
        "src/kthread.h",
    ),
    (
        "src/lchain.c",
        "src/chain.h",
        "src/krmq.h",
    ),
    (
        "src/rseed.c",
        "src/rseed.h",
        "src/rmap.c",
    ),
    (
        "src/rmap.c",
        "src/rmap.h",
        "src/chain.h",
        "src/lchain.c",
    ),
    (
        "src/rindex.c",
        "src/rindex.h",
        "src/roptions.c",
        "src/roptions.h",
    ),
)

SYSTEM_PROMPT = """You solve computational biology problems with verifiable outputs.
Return exactly one <answer>...</answer> block. The content inside the answer block must be valid JSON matching the requested schema.
Do not include code in the answer block."""

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class Example:
    task: str
    prompt: list[dict[str, str]]
    instance: dict[str, Any]
    answer: dict[str, Any]

    def to_row(self) -> dict[str, str | list[dict[str, str]]]:
        return {
            "prompt": self.prompt,
            "task": self.task,
            "instance_json": _json_dumps(self.instance),
            "answer_json": _json_dumps(self.answer),
        }


def make_examples(
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
) -> list[dict[str, str | list[dict[str, str]]]]:
    """Create deterministic task rows for Tunix data_module loading."""
    if task == "mixed":
        tasks = list(DEFAULT_MIXED_TASKS)
    else:
        tasks = [name.strip() for name in task.split(",") if name.strip()]
        unknown = sorted(set(tasks) - set(TASK_NAMES))
        if unknown:
            raise ValueError(f"Unknown compbio task(s): {unknown}")

    split_offset = {"train": 0, "validation": 10_000, "test": 20_000}.get(split, 30_000)
    rng = random.Random(seed + split_offset)
    rows = []
    for idx in range(num_examples):
        task_name = tasks[idx % len(tasks)]
        ex_rng = random.Random(rng.randrange(1 << 30))
        rows.append(
            _make_example(
                task_name,
                ex_rng,
                idx,
                rawhash2_case=rawhash2_case,
                rawhash2_prompt_archive_path=rawhash2_prompt_archive_path,
                rawhash2_reward_audit_path=rawhash2_reward_audit_path,
                rawhash2_prompt_archive_fraction=rawhash2_prompt_archive_fraction,
                rawhash2_prompt_archive_max_chars=rawhash2_prompt_archive_max_chars,
                rawhash2_prompt_archive_dynamic=rawhash2_prompt_archive_dynamic,
                rawhash2_prompt_archive_warmup_examples=rawhash2_prompt_archive_warmup_examples,
            ).to_row()
        )
    return rows


def score_completion(task: str, completion: str, answer_json: str) -> float:
    """Score a model completion in [0, 1]."""
    expected = json.loads(_as_str(answer_json))
    answer_payload, had_answer_tag = _extract_answer_payload(completion)
    if task == "rawhash2_native_opt":
        parsed, parse_status = _loads_top_level_json_object_with_status(answer_payload)
    return _score_rawhash2_native_completion(
        completion=completion,
        parsed=parsed,
        expected=expected,
        had_answer_tag=had_answer_tag,
        answer_payload=answer_payload,
        parse_status=parse_status,
    )
    parsed = _loads_json_object(answer_payload)
    if parsed is None:
        return 0.0

    if task == "read_mapping":
        score = _score_read_mapping(parsed, expected)
    elif task == "msa":
        score = _score_msa(parsed, expected)
    elif task == "genome_assembly":
        score = _score_genome_assembly(parsed, expected)
    elif task == "phylogeny":
        score = _score_phylogeny(parsed, expected)
    else:
        score = 0.0

    # Keep output-format pressure without overwhelming task reward.
    return max(0.0, min(1.0, score if had_answer_tag else 0.9 * score))


def _make_example(
    task: str,
    rng: random.Random,
    idx: int,
    rawhash2_case: dict[str, Any] | None = None,
    rawhash2_prompt_archive_path: str | None = None,
    rawhash2_reward_audit_path: str | None = None,
    rawhash2_prompt_archive_fraction: float = 0.0,
    rawhash2_prompt_archive_max_chars: int = 6000,
    rawhash2_prompt_archive_dynamic: bool = False,
    rawhash2_prompt_archive_warmup_examples: int = 0,
) -> Example:
    if task == "read_mapping":
        return _make_read_mapping(rng, idx)
    if task == "msa":
        return _make_msa(rng, idx)
    if task == "genome_assembly":
        return _make_genome_assembly(rng, idx)
    if task == "phylogeny":
        return _make_phylogeny(rng, idx)
    if task == "rawhash2_native_opt":
        return _make_rawhash2_native_opt(
            rng,
            idx,
            rawhash2_case=rawhash2_case,
            rawhash2_prompt_archive_path=rawhash2_prompt_archive_path,
            rawhash2_reward_audit_path=rawhash2_reward_audit_path,
            rawhash2_prompt_archive_fraction=rawhash2_prompt_archive_fraction,
            rawhash2_prompt_archive_max_chars=rawhash2_prompt_archive_max_chars,
            rawhash2_prompt_archive_dynamic=rawhash2_prompt_archive_dynamic,
            rawhash2_prompt_archive_warmup_examples=rawhash2_prompt_archive_warmup_examples,
        )
    raise ValueError(f"Unknown task: {task}")


def _make_read_mapping(rng: random.Random, idx: int) -> Example:
    reference = _rand_dna(rng, rng.randint(90, 130))
    read_len = rng.randint(16, 24)
    starts = sorted(rng.sample(range(0, len(reference) - read_len), 8))
    reads = []
    for i, start in enumerate(starts):
        read = list(reference[start : start + read_len])
        for j in range(len(read)):
            if rng.random() < 0.06:
                read[j] = rng.choice([b for b in DNA if b != read[j]])
        reads.append({"id": f"read_{idx}_{i}", "sequence": "".join(read)})

    prompt = f"""Task: read mapping.
Given a reference genome and sequencing reads with substitutions, infer the zero-based start coordinate of each read on the reference.

Reference:
{reference}

Reads, in order:
{_json_dumps(reads)}

Return JSON with schema {{"positions": [int, ...]}}. The positions list must be in the same order as the reads."""
    return Example(
        task="read_mapping",
        prompt=_messages(prompt),
        instance={"reference": reference, "reads": reads},
        answer={"positions": starts, "read_length": read_len},
    )


def _make_msa(rng: random.Random, idx: int) -> Example:
    ancestor = _rand_dna(rng, rng.randint(18, 26))
    sequences = [_mutate_for_msa(rng, ancestor) for _ in range(rng.randint(4, 5))]
    labels = [f"seq_{idx}_{i}" for i in range(len(sequences))]
    records = [{"id": label, "sequence": seq} for label, seq in zip(labels, sequences)]
    max_len = max(len(seq) for seq in sequences)
    alignment = [
        {"id": label, "aligned": seq + "-" * (max_len - len(seq))}
        for label, seq in zip(labels, sequences)
    ]

    prompt = f"""Task: multiple sequence alignment.
Align these homologous DNA sequences. Preserve every non-gap character in each input sequence and insert '-' gap characters as needed.

Sequences:
{_json_dumps(records)}

Return JSON with schema {{"alignment": [{{"id": str, "aligned": str}}, ...]}}. Use the same ids and order as the input."""
    return Example(
        task="msa",
        prompt=_messages(prompt),
        instance={"records": records},
        answer={"records": records, "alignment": alignment},
    )


def _make_genome_assembly(rng: random.Random, idx: int) -> Example:
    genome = _rand_dna(rng, rng.randint(54, 72))
    read_len = rng.randint(14, 18)
    starts = list(range(0, len(genome) - read_len + 1, rng.randint(5, 7)))
    starts.extend(rng.randrange(0, len(genome) - read_len + 1) for _ in range(8))
    reads = [genome[start : start + read_len] for start in starts]
    rng.shuffle(reads)
    records = [{"id": f"read_{idx}_{i}", "sequence": read} for i, read in enumerate(reads)]

    prompt = f"""Task: genome assembly.
Assemble the shortest plausible linear DNA contig from these overlapping reads. Reads are sampled from one forward-strand genome without indel errors.

Reads:
{_json_dumps(records)}

Return JSON with schema {{"assembly": "ACGT..."}}."""
    return Example(
        task="genome_assembly",
        prompt=_messages(prompt),
        instance={"reads": records},
        answer={"assembly": genome},
    )


def _make_phylogeny(rng: random.Random, idx: int) -> Example:
    labels = [f"taxon_{idx}_{i}" for i in range(5)]
    topology = rng.choice(_topologies(labels))
    root = _rand_dna(rng, 36)
    seqs = _simulate_tree_sequences(rng, topology, root)
    records = [{"id": label, "sequence": seqs[label]} for label in labels]
    splits = sorted(_tree_splits(topology, set(labels)), key=lambda s: sorted(s))
    newick = _to_newick(topology) + ";"

    prompt = f"""Task: phylogenetic tree construction.
Infer an unrooted tree topology from these aligned DNA sequences. Branch lengths are not required.

Aligned sequences:
{_json_dumps(records)}

Return JSON with schema {{"newick": "..."}} using the provided taxon ids as leaf names."""
    return Example(
        task="phylogeny",
        prompt=_messages(prompt),
        instance={"records": records},
        answer={"newick": newick, "splits": [list(s) for s in splits], "taxa": labels},
    )


def _make_rawhash2_native_opt(
    rng: random.Random,
    idx: int,
    rawhash2_case: dict[str, Any] | None = None,
    rawhash2_prompt_archive_path: str | None = None,
    rawhash2_reward_audit_path: str | None = None,
    rawhash2_prompt_archive_fraction: float = 0.0,
    rawhash2_prompt_archive_max_chars: int = 6000,
    rawhash2_prompt_archive_dynamic: bool = False,
    rawhash2_prompt_archive_warmup_examples: int = 0,
) -> Example:
    del rng
    from discover_compbio.rawhash2_native.config import DATASET_COMMIT
    from discover_compbio.rawhash2_native.config import DATASET_ID
    from discover_compbio.rawhash2_native.config import case_spec_from_config
    from discover_compbio.rawhash2_native.config import config_from_case_spec
    from discover_compbio.rawhash2_native.prompt import SYSTEM_PROMPT as RAWHASH2_SYSTEM_PROMPT
    from discover_compbio.rawhash2_native.prompt import native_rawhash2_prompt
    from discover_compbio.rawhash2_native.prompt_archive import build_prompt_archive_context

    cfg = config_from_case_spec(rawhash2_case)
    case_spec = case_spec_from_config(cfg)
    # Surface the baseline's per-phase compute profile in the prompt so the model
    # targets the hot paths (chaining/seeding dominate) instead of negligible phases.
    baseline_metrics = cfg.baseline_metrics
    if baseline_metrics is None and cfg.baseline_json is not None:
        from discover_compbio.rawhash2_native.metrics import load_baseline
        try:
            baseline_metrics = load_baseline(cfg.baseline_json)
        except Exception:
            baseline_metrics = None
    instance = {
        "case": "rawhash2_native_rawbench_hsapiens",
        "dataset_id": DATASET_ID,
        "dataset_commit": DATASET_COMMIT,
        "source_repo": "RawHash2",
        "allowed_patch_root": "src/",
        "benchmark": (
            f"R10.4.1 human FAST5, --bp-per-sec {case_spec['bp_per_sec']} --r10 "
            f"-x {case_spec['preset']} -t {case_spec['threads']} {' '.join(case_spec['extra_params'])}"
        ),
    }
    answer = {
        "patch_format": "edits",
        "allowed_paths": ["src/*.c", "src/*.h", "src/*.cc", "src/*.cpp", "src/*.hpp"],
        "metric": "F1-gated native RawHash2 speed/RSS reward against baseline",
        "example_index": idx,
        "rawhash2_case": case_spec,
    }
    source_files = RAWHASH2_NATIVE_SOURCE_SLICES[idx % len(RAWHASH2_NATIVE_SOURCE_SLICES)]
    instance["editable_source_files"] = list(source_files)
    if rawhash2_prompt_archive_dynamic:
        instance["rawhash2_prompt_archive"] = {
            "dynamic": True,
            "native_audit_path": rawhash2_prompt_archive_path,
            "reward_audit_path": rawhash2_reward_audit_path,
            "fraction": rawhash2_prompt_archive_fraction,
            "max_chars": rawhash2_prompt_archive_max_chars,
            "warmup_examples": max(0, int(rawhash2_prompt_archive_warmup_examples or 0)),
            "archive_variant": idx,
            "source_files": list(source_files),
        }
    archive_context = ""
    if (
        not rawhash2_prompt_archive_dynamic
        and _use_rawhash2_prompt_archive(idx, rawhash2_prompt_archive_fraction)
    ):
        archive_context = build_prompt_archive_context(
            native_audit_path=rawhash2_prompt_archive_path or os.environ.get("RAWHASH2_NATIVE_AUDIT_PATH"),
            reward_audit_path=rawhash2_reward_audit_path or os.environ.get("COMPBIO_REWARD_AUDIT_PATH"),
            source_files=source_files,
            max_chars=rawhash2_prompt_archive_max_chars,
            archive_variant=idx,
        )
    return Example(
        task="rawhash2_native_opt",
        prompt=_messages(
            native_rawhash2_prompt(
                source_repo=case_spec.get("rawhash2_repo"),
                source_files=source_files,
                baseline_metrics=baseline_metrics,
                archive_context=archive_context,
            ),
            system_prompt=RAWHASH2_SYSTEM_PROMPT,
        ),
        instance=instance,
        answer=answer,
    )


def _use_rawhash2_prompt_archive(idx: int, fraction: float) -> bool:
    fraction = max(0.0, min(1.0, float(fraction or 0.0)))
    if fraction <= 0.0:
        return False
    if fraction >= 1.0:
        return True
    # Use a prime-length schedule so archive selection does not alias with the
    # 8-way RawHash2 source-slice rotation. The old idx % 4 schedule meant
    # slices 3 and 7 never received archive context at a 0.75 fraction.
    slots = 17
    archive_slots = max(1, min(slots - 1, round(slots * fraction)))
    return (idx * 5) % slots < archive_slots


def _score_read_mapping(parsed: dict[str, Any], expected: dict[str, Any]) -> float:
    positions = parsed.get("positions")
    truth = expected["positions"]
    read_len = expected.get("read_length", 1)
    if not isinstance(positions, list) or len(positions) != len(truth):
        return 0.0
    scores = []
    for pred, gold in zip(positions, truth):
        if not isinstance(pred, int):
            return 0.0
        scores.append(max(0.0, 1.0 - abs(pred - gold) / max(1, read_len)))
    return sum(scores) / len(scores)


def _score_msa(parsed: dict[str, Any], expected: dict[str, Any]) -> float:
    alignment = parsed.get("alignment")
    records = expected["records"]
    if not isinstance(alignment, list) or len(alignment) != len(records):
        return 0.0
    aligned = []
    for got, record in zip(alignment, records):
        if not isinstance(got, dict) or got.get("id") != record["id"]:
            return 0.0
        seq = got.get("aligned")
        if not isinstance(seq, str) or not seq:
            return 0.0
        if seq.replace("-", "") != record["sequence"]:
            return 0.0
        aligned.append(seq.upper())
    lengths = {len(seq) for seq in aligned}
    if len(lengths) != 1:
        return 0.0

    expected_alignment = expected.get("alignment")
    if isinstance(expected_alignment, list) and len(expected_alignment) == len(records):
        normalized_expected = []
        for item in expected_alignment:
            if not isinstance(item, dict):
                break
            exp_id = item.get("id")
            exp_seq = item.get("aligned")
            if not isinstance(exp_id, str) or not isinstance(exp_seq, str):
                break
            normalized_expected.append({"id": exp_id, "aligned": exp_seq.upper()})
        else:
            normalized_got = [
                {"id": record["id"], "aligned": seq}
                for record, seq in zip(records, aligned)
            ]
            if normalized_got == normalized_expected:
                return 1.0

    raw = 0.0
    aln_len = lengths.pop()
    pair_count = 0
    for i in range(len(aligned)):
        for j in range(i + 1, len(aligned)):
            pair_count += 1
            for a, b in zip(aligned[i], aligned[j]):
                if a == "-" and b == "-":
                    raw += 0.0
                elif a == "-" or b == "-":
                    raw -= 0.75
                elif a == b:
                    raw += 1.0
                else:
                    raw -= 0.5
    best = pair_count * aln_len
    worst = -0.75 * pair_count * aln_len
    return (raw - worst) / (best - worst) if best > worst else 0.0


def _score_genome_assembly(parsed: dict[str, Any], expected: dict[str, Any]) -> float:
    assembly = parsed.get("assembly")
    truth = expected["assembly"]
    if not isinstance(assembly, str):
        return 0.0
    assembly = assembly.upper().replace(" ", "").replace("\n", "")
    if not assembly or set(assembly) - set(DNA):
        return 0.0
    candidates = [truth, _revcomp(truth)]
    return max(_edit_similarity(assembly, candidate) for candidate in candidates)


def _score_phylogeny(parsed: dict[str, Any], expected: dict[str, Any]) -> float:
    taxa = set(expected["taxa"])
    truth = {frozenset(s) for s in expected["splits"]}
    pred: set[frozenset[str]] = set()
    if isinstance(parsed.get("clusters"), list):
        for cluster in parsed["clusters"]:
            if isinstance(cluster, list):
                pred.add(_canonical_split(set(map(str, cluster)), taxa))
    elif isinstance(parsed.get("newick"), str):
        parsed_tree = _parse_newick(parsed["newick"], taxa)
        if parsed_tree is None:
            return 0.0
        pred = _tree_splits(parsed_tree, taxa)
    else:
        return 0.0

    pred = {s for s in pred if 1 < len(s) < len(taxa) - 1}
    if not pred and not truth:
        return 1.0
    if not pred or not truth:
        return 0.0
    tp = len(pred & truth)
    precision = tp / len(pred)
    recall = tp / len(truth)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _score_rawhash2_native_completion(
    *,
    completion: str,
    parsed: dict[str, Any] | None,
    expected: dict[str, Any],
    had_answer_tag: bool,
    answer_payload: str,
    parse_status: str = "unknown",
) -> float:
    """Score RawHash2 only from the native benchmark evaluator.

    RawHash2 optimization is an execution-gated task: a completion earns reward
    only after the candidate patch/edits are parsed, built, mapped, compared
    against the baseline, and accuracy-gated by the native evaluator. Formatting
    progress is useful for diagnostics and rollout stopping, but it must not
    become the scalar reward used by GRPO.
    """
    if not had_answer_tag or parsed is None:
        _audit_rawhash2_native_parse_failure(
            completion=completion,
            answer_payload=answer_payload,
            had_answer_tag=had_answer_tag,
            parse_status=parse_status,
        )
        return 0.0

    from discover_compbio.rawhash2_native.evaluator import score_answer

    try:
        return max(0.0, min(1.0, score_answer(parsed, expected)))
    except Exception as exc:
        if exc.__class__.__name__ != "TransientBenchmarkError":
            raise
        _audit_rawhash2_native_parse_failure(
            completion=completion,
            answer_payload=answer_payload,
            had_answer_tag=had_answer_tag,
            parse_status=f"transient_benchmark_error:{exc}",
        )
        return 0.0


def _extract_answer_payload(completion: str) -> tuple[str, bool]:
    lower = completion.lower()
    think_end = lower.rfind("</think>")
    search_spaces = []
    if think_end >= 0:
        search_spaces.append(completion[think_end + len("</think>") :])
    search_spaces.append(completion)

    for text in search_spaces:
        matches = list(ANSWER_RE.finditer(text))
        closed_candidates = [match.group(1).strip() for match in matches]
        for candidate in reversed(closed_candidates):
            if _answer_payload_starts_json_object(candidate):
                return candidate, True

        starts = [match.start() for match in re.finditer(r"<answer>", text, re.IGNORECASE)]
        unclosed_candidates = [text[start + len("<answer>") :].strip() for start in starts]
        for candidate in reversed(unclosed_candidates):
            if _answer_payload_starts_json_object(candidate):
                return candidate, True

        if closed_candidates:
            return closed_candidates[-1], True
        if unclosed_candidates:
            return unclosed_candidates[-1], True
        start = text.lower().rfind("<answer>")
        if start >= 0:
            return text[start + len("<answer>") :].strip(), True
    return completion.strip(), False


def _answer_payload_starts_json_object(text: str) -> bool:
    stripped, _ = _strip_json_fence(text)
    return stripped.lstrip().startswith("{")


def _loads_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    starts = [idx for idx, ch in enumerate(text) if ch == "{"]
    for start in starts or [0]:
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _loads_top_level_json_object(text: str) -> dict[str, Any] | None:
    parsed, _ = _loads_top_level_json_object_with_status(text)
    return parsed


def _loads_top_level_json_object_with_status(text: str) -> tuple[dict[str, Any] | None, str]:
    decoder = json.JSONDecoder()
    stripped, stripped_fence = _strip_json_fence(text)
    stripped = stripped.lstrip()
    if not stripped.startswith("{"):
        return None, "not_object_start"

    parsed = _decode_top_level_json_object(decoder, stripped)
    if parsed is not None:
        obj, tail = parsed
        if not isinstance(obj, dict):
            return None, "not_dict"
        if tail:
            return obj, "dict_with_tail_after_fence" if stripped_fence else "dict_with_tail"
        return obj, "ok_after_fence" if stripped_fence else "ok"

    control_repaired = _escape_raw_control_chars_in_json_strings(stripped)
    if control_repaired != stripped:
        parsed = _decode_top_level_json_object(decoder, control_repaired)
        if parsed is not None:
            obj, tail = parsed
            if not isinstance(obj, dict):
                return None, "not_dict"
            if tail:
                return (
                    obj,
                    "repaired_control_chars_with_tail_after_fence"
                    if stripped_fence
                    else "repaired_control_chars_with_tail",
                )
            return (
                obj,
                "repaired_control_chars_after_fence"
                if stripped_fence
                else "repaired_control_chars",
            )

        repaired = _repair_truncated_top_level_json(control_repaired)
        if repaired is not None:
            parsed = _decode_top_level_json_object(decoder, repaired)
            if parsed is not None:
                obj, tail = parsed
                if not isinstance(obj, dict) or tail:
                    return None, "json_error"
                return (
                    obj,
                    "repaired_control_chars_truncated_after_fence"
                    if stripped_fence
                    else "repaired_control_chars_truncated",
                )

    repaired = _repair_truncated_top_level_json(stripped)
    if repaired is None:
        return None, "json_error"
    parsed = _decode_top_level_json_object(decoder, repaired)
    if parsed is None:
        return None, "json_error"
    obj, tail = parsed
    if not isinstance(obj, dict) or tail:
        return None, "json_error"
    return obj, "repaired_truncated_after_fence" if stripped_fence else "repaired_truncated"


def _decode_top_level_json_object(
    decoder: json.JSONDecoder, text: str
) -> tuple[Any, str] | None:
    try:
        obj, end = decoder.raw_decode(text)
    except json.JSONDecodeError:
        return None
    return obj, text[end:].strip()


def _escape_raw_control_chars_in_json_strings(text: str) -> str:
    """Escape raw control characters that models often emit inside JSON strings.

    The repair is deliberately limited to strings inside a payload that already
    starts at the top-level object. It does not search for inner JSON objects or
    otherwise reinterpret malformed surrounding text.
    """
    out: list[str] = []
    in_string = False
    escaped = False
    changed = False
    for ch in text:
        if in_string:
            if escaped:
                out.append(ch)
                escaped = False
            elif ch == "\\":
                out.append(ch)
                escaped = True
            elif ch == '"':
                out.append(ch)
                in_string = False
            elif ch == "\n":
                out.append("\\n")
                changed = True
            elif ch == "\r":
                out.append("\\r")
                changed = True
            elif ch == "\t":
                out.append("\\t")
                changed = True
            elif ord(ch) < 0x20:
                out.append(f"\\u{ord(ch):04x}")
                changed = True
            else:
                out.append(ch)
            continue
        out.append(ch)
        if ch == '"':
            in_string = True
    return "".join(out) if changed else text


def _repair_truncated_top_level_json(text: str) -> str | None:
    """Close a top-level JSON object truncated after a complete token boundary.

    This intentionally does not search inside malformed text for inner objects.
    It only appends missing closing brackets/braces when the payload already
    starts at the top-level object and the scanner is not inside a string.
    """
    stack: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if not stack or stack.pop() != ch:
                return None
    if in_string or not stack:
        return None
    return text + "".join(reversed(stack))


def _strip_json_fence(text: str) -> tuple[str, bool]:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text, False
    lines = stripped.splitlines()
    if len(lines) < 2:
        return text, False
    first = lines[0].strip().lower()
    if first not in {"```", "```json"}:
        return text, False
    if lines[-1].strip() != "```":
        return text, False
    return "\n".join(lines[1:-1]).strip(), True


def _audit_rawhash2_native_parse_failure(
    *,
    completion: str,
    answer_payload: str,
    had_answer_tag: bool,
    parse_status: str,
) -> None:
    audit_path = os.environ.get("RAWHASH2_NATIVE_AUDIT_PATH")
    if not audit_path:
        return
    path = Path(audit_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    message = (
        f"answer_parse_failed: had_answer_tag={had_answer_tag}, "
        f"parse_status={parse_status}"
    )
    row = {
        "completion_sha256": hashlib.sha256(completion.encode("utf-8")).hexdigest(),
        "completion_chars": len(completion),
        "answer_payload_sha256": hashlib.sha256(answer_payload.encode("utf-8")).hexdigest(),
        "answer_payload_chars": len(answer_payload),
        "answer_payload_prefix": _audit_snippet(answer_payload[:512]),
        "answer_payload_suffix": _audit_snippet(answer_payload[-512:]),
        "had_answer_tag": had_answer_tag,
        "ok": False,
        "phase": "parse",
        "parse_status": parse_status,
        "reward": 0.0,
        "message": message,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _audit_snippet(text: str) -> str:
    return text.replace("\r", "\\r").replace("\n", "\\n")


def _messages(user_prompt: str, system_prompt: str = SYSTEM_PROMPT) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _rand_dna(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(DNA) for _ in range(length))


def _mutate_for_msa(rng: random.Random, seq: str) -> str:
    out = []
    for base in seq:
        if rng.random() < 0.08:
            continue
        if rng.random() < 0.10:
            out.append(rng.choice([b for b in DNA if b != base]))
        else:
            out.append(base)
        if rng.random() < 0.05:
            out.append(rng.choice(DNA))
    return "".join(out) or seq[:1]


def _mutate_substitutions(rng: random.Random, seq: str, rate: float) -> str:
    chars = list(seq)
    for i, base in enumerate(chars):
        if rng.random() < rate:
            chars[i] = rng.choice([b for b in DNA if b != base])
    return "".join(chars)


def _revcomp(seq: str) -> str:
    table = str.maketrans("ACGT", "TGCA")
    return seq.translate(table)[::-1]


def _edit_similarity(a: str, b: str) -> float:
    dist = _levenshtein(a, b)
    return max(0.0, 1.0 - dist / max(len(a), len(b), 1))


def _levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


Tree = str | tuple[Any, Any]


def _topologies(labels: list[str]) -> list[Tree]:
    a, b, c, d, e = labels
    return [
        (((a, b), (c, d)), e),
        (((a, c), (b, d)), e),
        (((a, d), (b, c)), e),
        (((a, b), e), (c, d)),
        (((a, c), e), (b, d)),
        (((a, d), e), (b, c)),
    ]


def _simulate_tree_sequences(rng: random.Random, tree: Tree, seq: str) -> dict[str, str]:
    if isinstance(tree, str):
        return {tree: seq}
    left, right = tree
    left_seq = _mutate_substitutions(rng, seq, 0.07)
    right_seq = _mutate_substitutions(rng, seq, 0.07)
    return {
        **_simulate_tree_sequences(rng, left, left_seq),
        **_simulate_tree_sequences(rng, right, right_seq),
    }


def _tree_leaves(tree: Tree) -> set[str]:
    if isinstance(tree, str):
        return {tree}
    return _tree_leaves(tree[0]) | _tree_leaves(tree[1])


def _tree_splits(tree: Tree, all_taxa: set[str]) -> set[frozenset[str]]:
    splits = set()

    def visit(node: Tree) -> None:
        leaves = _tree_leaves(node)
        if 1 < len(leaves) < len(all_taxa) - 1:
            splits.add(_canonical_split(leaves, all_taxa))
        if not isinstance(node, str):
            visit(node[0])
            visit(node[1])

    visit(tree)
    return splits


def _canonical_split(leaves: set[str], all_taxa: set[str]) -> frozenset[str]:
    clean = set(leaves) & all_taxa
    complement = all_taxa - clean
    if len(complement) < len(clean):
        clean = complement
    return frozenset(clean)


def _to_newick(tree: Tree) -> str:
    if isinstance(tree, str):
        return tree
    return f"({_to_newick(tree[0])},{_to_newick(tree[1])})"


def _parse_newick(text: str, taxa: set[str]) -> Tree | None:
    tokens = re.findall(r"[A-Za-z0-9_.-]+|[(),;]", text)
    idx = 0

    def parse_node() -> Tree | None:
        nonlocal idx
        if idx >= len(tokens):
            return None
        tok = tokens[idx]
        if tok == "(":
            idx += 1
            children = []
            while True:
                child = parse_node()
                if child is None:
                    return None
                children.append(child)
                if idx >= len(tokens):
                    return None
                if tokens[idx] == ",":
                    idx += 1
                    continue
                if tokens[idx] == ")":
                    idx += 1
                    break
                return None
            if not children:
                return None
            node = children[0]
            for child in children[1:]:
                node = (node, child)
            return node
        if tok in taxa:
            idx += 1
            return tok
        return None

    tree = parse_node()
    if tree is None:
        return None
    if idx < len(tokens) and tokens[idx] == ";":
        idx += 1
    if idx != len(tokens):
        return None
    return tree if _tree_leaves(tree) == taxa else None


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _as_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
