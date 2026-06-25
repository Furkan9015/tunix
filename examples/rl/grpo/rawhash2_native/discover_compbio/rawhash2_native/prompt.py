"""Prompt for the native RawHash2 optimization case."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Sequence

from discover_compbio.rawhash2_native.anchors import line_anchors_for_text


SYSTEM_PROMPT = """You optimize native C/C++ computational biology code.
Return exactly one final answer block delimited by <answer> and </answer>. The content inside it must be valid JSON.
Use an "edits" array of source edits; each edit may only modify files under src/.
Edits are applied directly to the source (no diff). Do NOT write unified-diff/patch text.
Do not recap the task in the final answer; the answer block must contain only JSON."""

_DEFAULT_SOURCE_CHAR_BUDGET = 11000
_SOURCE_MODE_ENV = "RAWHASH2_NATIVE_PROMPT_SOURCE_MODE"
_SOURCE_PRIORITY_ENV = "RAWHASH2_NATIVE_PROMPT_SOURCE_FILES"
_SOURCE_SUFFIXES = (".c", ".h", ".cc", ".cpp", ".hpp")
_SOURCE_PRIORITY = (
    "src/rawhash.h",
    "src/rsketch.h",
    "src/rseed.h",
    "src/chain.h",
    "src/rmap.h",
    "src/revent.h",
    "src/rsig.h",
    "src/rutils.h",
    "src/rsketch.c",
    "src/rseed.c",
    "src/lchain.c",
    "src/rmap.c",
    "src/revent.c",
    "src/rsig.c",
    "src/rutils.c",
    "src/dtw.c",
)
_FULL_HEADER_LIMIT = 9000
_IMPORTANT_WINDOWS = {
    "src/rsketch.c": ("sketch", "quant", "minimizer", "hash", "em", "rid"),
    "src/rseed.c": ("seed", "hash", "occ", "anchor", "lookup"),
    "src/lchain.c": ("chain", "rmq", "dp", "score", "anchor"),
    "src/rmap.c": ("map_worker", "align_chain", "chain", "dtw", "paf", "unmapped"),
    "src/rindex.c": ("bucket", "idx", "post", "sort", "lookup", "malloc", "thread"),
    "src/revent.c": ("event", "normalize", "seg", "mean", "threshold"),
    "src/rsig.c": ("fast5", "read_sig", "normalize", "filter", "scale"),
    "src/rutils.c": ("timer", "profile", "rss", "thread"),
    "src/dtw.c": ("dtw", "score", "band", "trace"),
    "src/kalloc.c": ("kmalloc", "krealloc", "kfree", "capacity", "core"),
    "src/kthread.c": ("thread", "worker", "pipeline", "parallel", "mutex"),
    "src/roptions.c": ("preset", "sensitive", "max", "occ", "chain", "band"),
}


_PROFILE_PHASES = (
    ("profile_chaining_seconds_fraction", "chaining", "src/lchain.c, src/chain.h"),
    ("profile_seeding_seconds_fraction", "seeding", "src/rseed.c, src/rseed.h"),
    ("profile_signal_to_event_seconds_fraction", "signal-to-event", "src/revent.c"),
    ("profile_sketching_seconds_fraction", "sketching", "src/rsketch.c"),
)


def format_baseline_profile(baseline_metrics: dict | None) -> str:
    """Render the baseline's per-phase compute breakdown for the prompt.

    Tells the model where the baseline actually spends time so it targets the hot
    paths instead of editing negligible phases. Uses only phase fractions + the
    wall-clock map time (no filesystem paths or harness plumbing)."""
    if not isinstance(baseline_metrics, dict):
        return ""
    rows = []
    for key, name, files in _PROFILE_PHASES:
        frac = baseline_metrics.get(key)
        if isinstance(frac, (int, float)):
            rows.append((float(frac), name, files))
    if not rows:
        return ""
    rows.sort(reverse=True)
    lines = [
        "Baseline compute profile (share of single-thread-accounted time per phase --",
        "spend your edits where the time actually is):",
    ]
    for frac, name, files in rows:
        lines.append(f"- {name} ({files}): ~{frac * 100:.0f}% of accounted compute")
    f1 = baseline_metrics.get("f1")
    precision = baseline_metrics.get("precision")
    recall = baseline_metrics.get("recall")
    if isinstance(f1, (int, float)):
        if isinstance(precision, (int, float)) and isinstance(recall, (int, float)):
            lines.append(
                f"Baseline accuracy: F1={f1:.4f}, precision={precision:.4f}, recall={recall:.4f}."
            )
        else:
            lines.append(f"Baseline accuracy: F1={f1:.4f}.")
    paf_summary = baseline_metrics.get("paf_summary")
    if isinstance(paf_summary, dict):
        mapped = paf_summary.get("mapped_fraction")
        chains = paf_summary.get("chain_count_mean")
        anchors = paf_summary.get("anchor_count_mean")
        if isinstance(mapped, (int, float)):
            line = f"Baseline mapped fraction: {mapped:.3f}"
            extras = []
            if isinstance(chains, (int, float)):
                extras.append(f"mean chains/read={chains:.2f}")
            if isinstance(anchors, (int, float)):
                extras.append(f"mean anchors/read={anchors:.2f}")
            if extras:
                line += " (" + ", ".join(extras) + ")"
            lines.append(line + ".")
    map_s = baseline_metrics.get("map_elapsed_seconds")
    if isinstance(map_s, (int, float)):
        lines.append(
            f"Baseline multithreaded map wall-clock: {map_s:.0f}s; the reward rises as the "
            "candidate maps faster at equal accuracy (F1)."
        )
    map_rss = baseline_metrics.get("map_max_rss_kb")
    if isinstance(map_rss, (int, float)):
        lines.append(
            f"Baseline peak map RSS: {map_rss / 1024 / 1024:.1f} GiB; lowering large "
            "seed/index/chain buffers helps only if F1 is preserved."
        )
    lines.append(
        "Edits to negligible phases (e.g. sketching, signal-to-event) will not move the "
        "reward -- concentrate on the dominant phases above."
    )
    return "\n".join(lines)


def native_rawhash2_prompt(
    *,
    source_repo: str | Path | None = None,
    max_source_chars: int | None = None,
    source_files: Sequence[str] | None = None,
    baseline_metrics: dict | None = None,
    archive_context: str = "",
) -> str:
    prompt = """Task: improve RawHash2-compatible native C/C++ read mapping on a fixed RawBench R10.4.1 human FAST5 benchmark.

The codebase under optimization is a native RawHash2-compatible source tree, not
a Python reimplementation. It may be upstream RawHash2 or an isolated baseline
that implements the same indexing, signal segmentation, and mapping/alignment
contract. Your patch is applied to an isolated copy of that source tree,
compiled, and benchmarked as a real binary. The reward compares the candidate
against a saved native baseline using the same benchmark contract.

Primary upstream algorithm files:
- src/rsketch.c / src/rsketch.h: adaptive quantization, e-mer packing, minimizers.
- src/rseed.c / src/rseed.h: hash-table seed lookup and occurrence filtering.
- src/lchain.c / src/chain.h: DP and RMQ chaining over anchors.
- src/rmap.c / src/rmap.h: chunk-level mapping, chain decisions, DTW hooks, PAF tags.
- src/rindex.c / src/rindex.h: reference index layout, bucket sorting, lookup memory.
- src/revent.c / src/revent.h and src/rsig.c / src/rsig.h: FAST5 signal/event handling.

Benchmark contract:
- RawHash2 is run in R10.4.1 mode with the sensitive preset, 400 bp/s, and -w 0.
- The scored command records mapping elapsed time, maximum RSS, PAF tags, and
  RawHash2 phase timers (signal-to-event, sketching, seeding, sorting, chaining,
  mapping), all compared against a saved baseline.

How you are scored (one reward in [0,1] vs the baseline):
- Accuracy/F1 is a gate and an improvement objective. If F1 falls below about
  70% of the baseline F1, reward is 0. F1 below baseline but above the hard gate
  reduces the credit for other improvements.
- If F1 passes the gate, reward combines F1 improvement over baseline, real
  map-time improvement, and peak RSS improvement:
  reward = accuracy_gate * (0.10 + 0.90 * (0.60 * accuracy_score + 0.20 * speed_score + 0.20 * memory_score)).
- Baseline-equivalent F1 with baseline-equivalent speed and memory earns only
  the 0.10 validity floor. The 0.60 accuracy term is for improving F1 over the
  baseline, not for merely matching it.
- A 50% map-time speedup saturates speed credit; a 30% RSS reduction saturates
  memory credit.
- Comment-only, formatting-only, or no-op edits are bad candidates even if they
  compile.
- The best edits preserve mapping behavior while reducing seeding/chaining/
  mapping work.
- Optional feedback may summarize candidate chains, ambiguity, and why reads were
  unmapped. Treat it as diagnostic guidance, not a separate output target.

Edit constraints:
- Return JSON only. Do not return prose outside JSON and do not use code fences.
- Prefer an "edits" array. Each edit is applied directly to the source file by
  anchor insertion/replacement or exact string find/replace, without `git apply`.
  A "patches" array of structured edit groups is also accepted when a candidate
  is naturally split into multiple logical changes.
- Modify only src/*.c, src/*.h, src/*.cc, src/*.cpp, or src/*.hpp.
- When exact editable source files are shown below, edit only those shown files.
  Do not invent functions, filenames, fields, or helper APIs that are not present.
- Do not alter benchmark scripts, build flags, pore models, test data, or output parsing.
- Preserve the rawhash2 CLI and PAF output format.
- Favor changes that improve speed and memory without lowering F1.
- Ambitious multi-block or multi-file changes are allowed. They are judged by
  parse/apply/build/F1/performance, not by how invasive they look.

High-value knobs to consider:
- Throughput: reduce repeated seed-hit scans, avoid unnecessary chain DP/RMQ work,
  prune dominated anchors/chains only when the downstream best-chain decision is unchanged,
  and keep inner loops branch- and allocation-light.
- Accuracy: preserve mapped/unmapped decisions, chain scores, PAF coordinates/tags,
  and DTW/alignment semantics. Do not lower occurrence filters or decision thresholds
  blindly; if recall changes, the F1 gate can zero the reward.
- Peak memory/RSS: reduce oversized transient seed/chain/index buffers, reuse per-thread
  scratch where ownership is clear, and avoid adding per-anchor/per-read allocations.
  Do not change serialized index format unless all load/dump/read paths stay compatible.
- Parallelism: the benchmark uses many threads. Avoid new shared mutable state, locks in
  hot loops, false sharing, or work imbalance across per-read/per-bucket workers.

Preferred JSON schema for one-line anchor edits:
{"edits": [{"path": "src/...", "anchor_id": "<copy-an-anchor_id-from-the-catalog-below>", "replace": "replacement for that anchored line"}]}

For insertion-only changes, prefer:
{"edits": [{"path": "src/...", "anchor_id": "<copy-an-anchor_id-from-the-catalog-below>", "insert_after": "text to insert after that line"}]}

For multi-line replacement or insertion, use JSON arrays of lines. Do not put
literal line breaks inside a JSON string:
{"edits": [{"path": "src/...", "anchor_id": "<copy-an-anchor_id-from-the-catalog-below>", "replace_lines": ["first replacement line", "second replacement line"]}]}
{"edits": [{"path": "src/...", "anchor_id": "<copy-an-anchor_id-from-the-catalog-below>", "insert_after_lines": ["first inserted line", "second inserted line"]}]}

For whole-function replacements, prefer naming the function directly. The
replacement must contain a complete C definition for that same function:
{"edits": [{"path": "src/...", "function": "function_name", "replace_lines": ["complete function signature {", "    body;", "}"]}]}

Anchor replacement edits normally target the single source line named by
anchor_id. For whole-function replacements, the function schema above is
preferred over giant fragile find blocks. If you use anchor_id + replace_lines
with a complete C function definition, the patcher can also replace the matching
function body in that same file. For non-function larger-block changes, use
exact find/replace with the full old block copied from the prompt, or use
multiple line-anchor edits that leave the surrounding original body syntactically
valid.

Exact find/replace is also accepted when you need to replace a larger block:
{"edits": [{"path": "src/...", "find": "exact text copied verbatim from the shown file", "replace": "replacement text"}]}

If the same find/after/before text appears more than once, do not give up and
do not use a giant fragile block just to make it unique. Add a named function
scope when the intended match is inside one function:
{"edits": [{"path": "src/...", "scope_function": "function_name", "find": "repeated statement", "replace": "new statement"}]}
For non-function repeated snippets, use a 1-based "occurrence" only when you
deliberately mean that exact occurrence in file order:
{"edits": [{"path": "src/...", "find": "repeated statement", "replace": "new statement", "occurrence": 2}]}

For exact insertion-only changes, use:
{"edits": [{"path": "src/...", "after": "exact anchor text", "insert": "text to insert after the anchor"}]}

For multiple coupled changes, either put all edit objects in one "edits" array
or split them into patch groups:
{"patches": [{"edits": [{"path": "src/...", "anchor_id": "<anchor>", "replace": "replacement"}]}, {"edits": [{"path": "src/...", "anchor_id": "<anchor>", "insert_after": "inserted line"}]}]}

Making edits apply cleanly:
- Prefer the line anchor IDs shown in the editable anchor catalog. They resolve
  directly to real source lines and avoid exact-string mismatch failures.
- In the anchor catalog, copy only the anchor_id token. Do not include the
  separator or source preview text after it.
- Every anchor_id must be copied verbatim from the editable anchor catalog
  below. Never emit a placeholder or invent a hash; an anchor_id that is not in
  the catalog will not be found and earns zero reward.
- Copy each "find"/"after" string VERBATIM from the shown source, character for
  character (whitespace and indentation included).
- Pick a SHORT, unique snippet -- ideally a single complete statement or a
  function-signature line. Avoid spanning comments or blank lines.
- The find/anchor text must occur exactly once in the file, once within
  scope_function, or identify an explicit occurrence/replace_all. An unscoped
  ambiguous find is rejected and earns zero reward.
- A no-op edit is rejected. Do not replace a line with identical text, and do
  not spend an edit on whitespace/comment-only changes.
- Large replacements are valid when they are necessary. Keep them parseable:
  for multi-line content, prefer replace_lines / insert_after_lines arrays or
  escaped "\\n" inside JSON strings.
- The first non-whitespace character inside the final answer block must be "{".
  Do not use doubled outer braces like "{{" or "}}"; those are invalid JSON.
  If the answer already begins with {"edits":[, continue with edit objects and
  close the array/object; do not start another {"edits":...} wrapper inside it.
  Close the answer block immediately after the JSON.
Structured edits are preferred. Raw unified diffs are accepted only in a "patch"
field or a "patches" list, and must modify allowed src/* source files only.
"""
    profile = format_baseline_profile(baseline_metrics)
    if profile:
        prompt = f"{prompt}\n{profile}\n"
    if archive_context.strip():
        prompt = f"{prompt}\n\n{archive_context.strip()}\n"
    source_context = rawhash2_source_context(
        source_repo=source_repo,
        max_chars=max_source_chars,
        source_files=source_files,
    )
    if source_context:
        prompt = f"{prompt}\n\n{source_context}"
    return prompt


def native_rawhash2_question(
    *,
    state_context: str = "",
    source_repo: str | Path | None = None,
    max_source_chars: int | None = None,
    source_files: Sequence[str] | None = None,
) -> str:
    """Builds the Discover single-problem prompt from this central module."""
    sections = [
        native_rawhash2_prompt(
            source_repo=source_repo,
            max_source_chars=max_source_chars,
            source_files=source_files,
        )
    ]
    if state_context.strip():
        sections.append(state_context.strip())
    sections.append(
        """Rules:
- Return one final answer block containing JSON with an "edits" array.
- Only modify files under the candidate source tree's src/ directory.
- Keep baseline and candidate build/run settings comparable; optimize the candidate source only."""
    )
    return "\n\n".join(sections) + "\n"


def rawhash2_source_context(
    *,
    source_repo: str | Path | None = None,
    max_chars: int | None = None,
    source_files: Sequence[str] | None = None,
) -> str:
    """Returns a budgeted baseline-source context for RawHash2 prompts.

    For an isolated RawHash2-compatible baseline, prefer complete editable files
    so the model can produce diffs against exact code. For the full upstream
    RawHash2 tree, fall back to a structural digest because the source is too
    large for normal RL prompts.
    """
    repo = _resolve_source_repo(source_repo)
    if repo is None:
        return ""
    src_dir = repo / "src"
    if not src_dir.is_dir():
        return ""

    budget = _source_budget(max_chars)
    if budget <= 0:
        return ""

    mode = _source_mode()
    priority = _source_priority(repo, source_files=source_files)
    full_source = _full_source_context(repo, budget, mode, priority)
    if full_source:
        return full_source

    header = (
        "Baseline RawHash2 source context (budgeted digest):\n"
        "- Full source is too large for this prompt. This section keeps "
        "headers/API contracts, function indexes, and hot-path excerpts.\n"
        "- Prefer narrow diffs against the shown files/functions. If a function "
        "body is omitted, do not fabricate a diff for that body.\n"
        "- Do not change external CLI/build behavior.\n"
    )
    remaining = budget - len(header)
    if remaining <= 0:
        return header[:budget]

    sections: list[str] = []
    for rel in priority:
        path = repo / rel
        if not path.is_file():
            continue
        text = _read_text(path)
        if not text:
            continue
        section = _source_section(rel, text, remaining)
        if not section:
            continue
        if len(section) > remaining:
            section = section[: max(0, remaining - 80)].rstrip()
            section += "\n/* section truncated by prompt budget */\n"
        sections.append(section)
        remaining -= len(section)
        if remaining <= 200:
            break

    if not sections:
        return header.rstrip()
    return header + "\n".join(sections).rstrip()


def _resolve_source_repo(source_repo: str | Path | None) -> Path | None:
    if source_repo is None:
        include = os.environ.get("RAWHASH2_NATIVE_PROMPT_INCLUDE_SOURCE", "0")
        if include != "1":
            return None
        source_repo = (
            os.environ.get("RAWHASH2_NATIVE_PROMPT_SOURCE_REPO")
            or os.environ.get("RAWHASH2_NATIVE_REPO")
        )
    if not source_repo:
        return None
    return Path(source_repo).expanduser()


def _source_budget(max_chars: int | None) -> int:
    if max_chars is None:
        raw = os.environ.get("RAWHASH2_NATIVE_PROMPT_SOURCE_CHAR_BUDGET")
        max_chars = int(raw) if raw else _DEFAULT_SOURCE_CHAR_BUDGET
    return max(0, int(max_chars))


def _source_mode() -> str:
    mode = os.environ.get(_SOURCE_MODE_ENV, "auto").strip().lower()
    if mode in {"auto", "digest", "full", "isolated"}:
        return mode
    return "auto"


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _source_priority(repo: Path, source_files: Sequence[str] | None = None) -> tuple[str, ...]:
    if source_files:
        return tuple(str(path).strip() for path in source_files if str(path).strip())
    requested = os.environ.get(_SOURCE_PRIORITY_ENV)
    if requested:
        files = tuple(part.strip() for part in requested.split(",") if part.strip())
        return files
    found = []
    src = repo / "src"
    if src.is_dir():
        for path in sorted(src.iterdir()):
            if path.is_file() and path.suffix in _SOURCE_SUFFIXES:
                rel = path.relative_to(repo).as_posix()
                if rel not in _SOURCE_PRIORITY:
                    found.append(rel)
    return _SOURCE_PRIORITY + tuple(found)


def _full_source_context(
    repo: Path,
    budget: int,
    mode: str,
    priority: Sequence[str],
) -> str:
    """Return complete editable source files when that fits the prompt budget."""
    if mode == "digest":
        return ""

    files = []
    for rel in priority:
        path = repo / rel
        if not path.is_file() or path.suffix not in _SOURCE_SUFFIXES:
            continue
        text = _compact_c_source(_read_text(path))
        if text:
            files.append((rel, text))
    if not files:
        return ""

    header = (
        "Exact editable baseline source (complete files):\n"
        "- Complete files are marked as complete. Large requested files may be "
        "shown as exact source excerpts/digests instead; for those files, only "
        "edit source text that is actually shown.\n"
        "- Prefer the editable line anchors below. An anchor edit replaces or "
        "inserts next to one real source line without copying fragile blocks.\n"
        "- Write find/replace edits whose find text is copied verbatim from the "
        "shown complete file or exact excerpt.\n"
        "- If a short find/after/before snippet repeats, add scope_function for "
        "the intended C function or occurrence for the deliberate nth match.\n"
        "- Keep changes small and local unless a larger edit is clearly needed "
        "for speed or memory.\n"
        "- The candidate must still build into a RawHash2-compatible binary and "
        "preserve the CLI/PAF contract used by the benchmark.\n"
    )
    anchor_catalog = _editable_anchor_catalog(
        repo, priority, max_chars=min(7200, max(2600, budget // 3))
    )
    if anchor_catalog:
        header += "\n" + anchor_catalog + "\n"
    remaining = budget - len(header)
    if remaining <= 0:
        return ""

    sections: list[str] = []
    omitted: list[str] = []
    for rel, text in files:
        section = f"\n--- {rel} (complete editable file) ---\n{text}\n"
        if len(section) <= remaining:
            sections.append(section)
            remaining -= len(section)
        else:
            excerpt_budget = min(remaining, 6500)
            if excerpt_budget >= 1400:
                excerpt = _source_section(rel, text, excerpt_budget)
                if excerpt:
                    if len(excerpt) > remaining:
                        keep = max(0, remaining - 80)
                        excerpt = excerpt[:keep].rstrip()
                        excerpt += "\n/* excerpt truncated by prompt budget */\n"
                    sections.append(excerpt)
                    remaining -= len(excerpt)
                    continue
            omitted.append(rel)
        if remaining <= 200:
            break

    if not sections:
        return ""

    all_fit = not omitted
    auto_full = mode == "auto" and all_fit
    explicit_full = mode in {"full", "isolated"}
    if not auto_full and not explicit_full:
        return ""

    if omitted:
        sections.append(
            "\nFiles omitted because of the prompt budget; do not patch omitted files: "
            + ", ".join(omitted)
            + "\n"
        )
    return header + "".join(sections).rstrip()


def _source_section(rel: str, text: str, budget: int) -> str:
    compact = _compact_c_source(text)
    if rel.endswith(".h") and len(compact) <= min(budget, _FULL_HEADER_LIMIT):
        return f"\n--- {rel} (full header) ---\n{compact}\n"
    if len(compact) <= min(budget, 6500):
        return f"\n--- {rel} (full compact source) ---\n{compact}\n"
    return _digest_source_file(rel, compact, min(budget, 6500))


def _compact_c_source(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"^[ \t]*//.*$", "", text, flags=re.MULTILINE)
    lines = [line.rstrip() for line in text.splitlines()]
    compact_lines: list[str] = []
    blank = False
    for line in lines:
        if not line.strip():
            if not blank:
                compact_lines.append("")
            blank = True
            continue
        compact_lines.append(line)
        blank = False
    return "\n".join(compact_lines).strip()


def _digest_source_file(rel: str, text: str, max_chars: int) -> str:
    lines = text.splitlines()
    contract = _contract_lines(lines)
    windows = _keyword_windows(lines, _IMPORTANT_WINDOWS.get(rel, ()))
    body = (
        f"\n--- {rel} (exact editable excerpts / source digest) ---\n"
        "/* Bodies are partially omitted by prompt budget. Use only concrete "
        "source lines from the contract or hot-path excerpts as exact edit text; "
        "prefer the line anchor catalog when possible. */\n"
    )
    if contract:
        body += "\n/* Includes/macros/types/prototypes */\n" + "\n".join(contract) + "\n"
    if windows:
        body += "\n/* Hot-path excerpts */\n" + "\n".join(windows) + "\n"
    if len(body) <= max_chars:
        return body
    keep = max(0, max_chars - 80)
    return body[:keep].rstrip() + "\n/* digest truncated by prompt budget */\n"


def _contract_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    collect_struct = False
    brace_depth = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("#include", "#define", "typedef", "enum ")):
            out.append(line)
        elif re.match(r"^(struct|typedef struct)\b", stripped):
            collect_struct = True
            brace_depth = stripped.count("{") - stripped.count("}")
            out.append(line)
            if ";" in stripped and brace_depth <= 0:
                collect_struct = False
        elif collect_struct:
            out.append(line)
            brace_depth += stripped.count("{") - stripped.count("}")
            if ";" in stripped and brace_depth <= 0:
                collect_struct = False
        if len(out) >= 90:
            out.append("/* additional contract lines omitted */")
            break
    return out


def _function_signatures(lines: list[str]) -> list[str]:
    signatures: list[str] = []
    pending: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            pending.clear()
            continue
        pending.append(stripped)
        joined = " ".join(pending)
        if "(" in joined and ")" in joined and re.search(r"\)\s*(\{|;)$", joined):
            if not joined.startswith(("#", "if ", "for ", "while ", "switch ")):
                signature = re.sub(r"\s*\{\s*$", ";", joined)
                signatures.append(signature)
            pending.clear()
        elif len(pending) > 4:
            pending.pop(0)
        if len(signatures) >= 80:
            signatures.append("/* additional function signatures omitted */")
            break
    return signatures


def _editable_anchor_catalog(repo: Path, priority: Sequence[str], max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    lines: list[str] = [
        "Editable line anchor catalog (preferred edit target):",
        "- Use anchor edits as: {\"path\":\"src/file.c\",\"anchor_id\":\"ID\",\"replace\":\"new source line for that anchor\"}.",
        "- For multi-line anchor replacements, use replace_lines; the target is still only that anchor line.",
        "- For whole-function replacements, prefer {\"path\":\"src/file.c\",\"function\":\"name\",\"replace_lines\":[\"complete function definition\"]}.",
        "- Or insert after an anchor as: {\"path\":\"src/file.c\",\"anchor_id\":\"ID\",\"insert_after\":\"new source line after that anchor\"}.",
        "- Copy only the anchor_id value, never the source preview after it.",
    ]
    remaining = max_chars - sum(len(line) + 1 for line in lines)
    if remaining <= 0:
        return ""

    file_sections: list[tuple[str, list[str]]] = []
    for rel in priority:
        path = repo / rel
        if not path.is_file() or path.suffix not in _SOURCE_SUFFIXES:
            continue
        raw = _read_text(path)
        if not raw:
            continue
        anchors = _select_line_anchors(rel, raw)
        if not anchors:
            continue
        section_items = []
        for anchor in anchors:
            text = anchor.text.replace("\t", "\\t")
            if len(text) > 140:
                text = text[:137].rstrip() + "..."
            section_items.append(f"- anchor_id={anchor.anchor_id} ; source={text}")
        file_sections.append((rel, section_items))

    files_left = len(file_sections)
    for rel, items in file_sections:
        if remaining <= 80 or files_left <= 0:
            break
        file_budget = max(0, remaining // files_left)
        block = [f"{rel}:"]
        used = len(block[0]) + 1
        for item in items:
            cost = len(item) + 1
            if used + cost > file_budget - 38:
                break
            block.append(item)
            used += cost
        if len(block) == 1 and items and remaining > len(block[0]) + len(items[0]) + 42:
            block.append(items[0])
            used += len(items[0]) + 1
        if len(block) > 1:
            if len(block) - 1 < len(items):
                block.append("- additional anchors omitted")
                used += len(block[-1]) + 1
            lines.append("\n".join(block))
            remaining -= used
        files_left -= 1
    return "\n".join(lines)


def _select_line_anchors(rel: str, text: str) -> list:
    keywords = tuple(k.lower() for k in _IMPORTANT_WINDOWS.get(rel, ()))
    anchors = [
        anchor for anchor in line_anchors_for_text(rel, text) if not _is_comment_only_line(anchor.text)
    ]
    if not keywords:
        return anchors[:18]
    selected = []
    for anchor in anchors:
        low = anchor.text.lower()
        if any(keyword in low for keyword in keywords):
            selected.append(anchor)
        if len(selected) >= 24:
            break
    if len(selected) < 8:
        for anchor in anchors:
            if anchor not in selected:
                selected.append(anchor)
            if len(selected) >= 12:
                break
    return selected


def _is_comment_only_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith(("//", "/*", "*", "*/"))


def _keyword_windows(lines: list[str], keywords: tuple[str, ...]) -> list[str]:
    if not keywords:
        return []
    lowered = tuple(k.lower() for k in keywords)
    selected: set[int] = set()
    for idx, line in enumerate(lines):
        low = line.lower()
        if any(k in low for k in lowered):
            start = max(0, idx - 10)
            end = min(len(lines), idx + 18)
            selected.update(range(start, end))
    if not selected:
        return []
    windows: list[str] = []
    prev = -2
    for idx in sorted(selected):
        if idx != prev + 1:
            windows.append(f"\n/* ... {idx + 1} ... */")
        windows.append(lines[idx])
        prev = idx
        if sum(len(w) + 1 for w in windows) > 3600:
            windows.append("/* additional hot-path excerpts omitted */")
            break
    return windows
