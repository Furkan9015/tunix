"""Patch parsing and validation for native RawHash2 candidates."""

from __future__ import annotations

import os
from pathlib import PurePosixPath
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from discover_compbio.rawhash2_native.anchors import line_anchors_for_text
from discover_compbio.rawhash2_native.anchors import normalize_code_line


DEFAULT_MAX_PATCH_BYTES = 1_000_000
DEFAULT_MAX_EDIT_BYTES = 1_000_000
DEFAULT_MAX_STRUCTURED_EDITS = 64
# Backward-compatible module constants. Runtime checks use the env-configurable
# helpers below so ambitious candidates can be evaluated without code edits.
MAX_PATCH_BYTES = DEFAULT_MAX_PATCH_BYTES
MAX_EDIT_BYTES = DEFAULT_MAX_EDIT_BYTES
MAX_STRUCTURED_EDITS = DEFAULT_MAX_STRUCTURED_EDITS
ALLOWED_SUFFIXES = {".c", ".h", ".cc", ".cpp", ".hpp"}


class PatchValidationError(ValueError):
    pass


def normalize_patch(text: str) -> str:
    patch = _strip_fence(text).strip() + "\n"
    max_patch_bytes = _max_patch_bytes()
    if len(patch.encode("utf-8")) > max_patch_bytes:
        raise PatchValidationError(f"patch exceeds {max_patch_bytes} bytes")
    if "diff --git " not in patch and "\n--- " not in patch:
        raise PatchValidationError("expected a unified diff patch")
    changed = changed_paths(patch)
    if not changed:
        raise PatchValidationError("patch does not modify any files")
    for path in changed:
        validate_changed_path(path)
    return patch


def patch_from_edits(repo: Path, edits: Any) -> str:
    """Build a unified diff from exact structured source edits.

    The model sees complete editable files in the prompt, but hand-written
    unified diff syntax is brittle. This helper accepts JSON-level edits and
    produces the diff deterministically against the baseline source tree.
    """
    if not isinstance(edits, list) or not edits:
        raise PatchValidationError("edits must be a non-empty list")
    _validate_edit_count(edits)

    originals: dict[str, str] = {}
    path_edits: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    path_order: list[str] = []
    for idx, edit in enumerate(edits):
        if not isinstance(edit, dict):
            raise PatchValidationError(f"edit {idx} must be an object")
        path = _edit_path(edit, idx)
        if path not in originals:
            originals[path] = _read_repo_text(repo, path)
            path_edits[path] = []
            path_order.append(path)
        path_edits[path].append((idx, edit))

    modified: dict[str, str] = {}
    for path in path_order:
        old_text = originals[path]
        new_text = _apply_path_edits(old_text, path_edits[path], path=path)
        if new_text != old_text:
            modified[path] = new_text

    changed_texts: dict[str, tuple[str, str]] = {}
    for path in path_order:
        new_text = modified.get(path)
        if new_text is None:
            continue
        changed_texts[path] = (originals[path], new_text)
    patch = _git_diff_from_texts(changed_texts)
    return normalize_patch(patch)


def _apply_path_edits(
    old_text: str,
    indexed_edits: list[tuple[int, dict[str, Any]]],
    *,
    path: str,
) -> str:
    if indexed_edits and all("anchor_id" in edit for _, edit in indexed_edits):
        return _apply_line_anchor_edits_from_original(old_text, indexed_edits, path=path)

    new_text = old_text
    for idx, edit in indexed_edits:
        edited = _apply_one_edit(new_text, edit, idx, path=path)
        if edited == new_text:
            continue
        new_text = edited
    return new_text


def _apply_line_anchor_edits_from_original(
    old_text: str,
    indexed_edits: list[tuple[int, dict[str, Any]]],
    *,
    path: str,
) -> str:
    """Apply same-file line-anchor edits against the original source snapshot."""
    lines = old_text.splitlines(keepends=True)
    ops_by_line: dict[int, list[tuple[int, str, str]]] = {}
    span_replacements: list[tuple[int, int, int, str]] = []
    for idx, edit in indexed_edits:
        if bool(edit.get("replace_all", False)):
            raise PatchValidationError(f"edit {idx} cannot use replace_all with anchor_id")
        anchor_id = _edit_string(edit.get("anchor_id"), idx, "anchor_id", allow_empty=False)
        if "replace" in edit or "replace_lines" in edit:
            text = _edit_text(edit, idx, "replace", "replace_lines", allow_empty=True)
            kind = "replace"
        elif "insert_after" in edit or "insert_after_lines" in edit:
            text = _edit_text(edit, idx, "insert_after", "insert_after_lines", allow_empty=False)
            kind = "after"
        elif "insert_before" in edit or "insert_before_lines" in edit:
            text = _edit_text(edit, idx, "insert_before", "insert_before_lines", allow_empty=False)
            kind = "before"
        else:
            raise PatchValidationError(
                f"edit {idx} with anchor_id must contain replace, replace_lines, "
                "insert_after, insert_after_lines, insert_before, or insert_before_lines"
            )
        if not text.endswith(("\n", "\r")):
            text += "\n"
        if kind == "replace":
            span = _function_replacement_span(old_text, text)
            if span is not None:
                span_replacements.append((idx, span[0], span[1], text))
                continue
        line_idx = _include_insert_after_line(old_text, text) if kind == "after" else None
        if line_idx is None:
            line_idx = _resolve_line_anchor(old_text, path, anchor_id)
        if line_idx is None and kind == "replace":
            line_idx = _resolve_line_anchor_from_replacement(old_text, path, anchor_id, text)
        if line_idx is None or line_idx >= len(lines):
            raise PatchValidationError(f"edit {idx} line anchor was not found: {anchor_id}")
        ops_by_line.setdefault(line_idx, []).append((idx, kind, text))

    span_replacements.sort(key=lambda item: item[1])
    previous_end = -1
    for idx, start, end, _ in span_replacements:
        if start < previous_end:
            raise PatchValidationError(f"edit {idx} overlaps another function replacement in {path}")
        for line_idx in ops_by_line:
            if start <= line_idx < end:
                raise PatchValidationError(
                    f"edit {idx} function replacement overlaps line-anchor edit in {path}"
                )
        previous_end = end
    spans_by_start = {start: (idx, end, text) for idx, start, end, text in span_replacements}

    out: list[str] = []
    line_idx = 0
    while line_idx < len(lines):
        span = spans_by_start.get(line_idx)
        if span is not None:
            _, end, text = span
            out.append(text)
            line_idx = end
            continue
        old_line = lines[line_idx]
        ops = ops_by_line.get(line_idx, [])
        for _, kind, text in ops:
            if kind == "before":
                out.append(text)
        replacements = [text for _, kind, text in ops if kind == "replace"]
        if len(replacements) > 1:
            edit_ids = [str(idx) for idx, kind, _ in ops if kind == "replace"]
            raise PatchValidationError(
                f"edits {', '.join(edit_ids)} replace the same line anchor in {path}"
            )
        out.append(replacements[0] if replacements else old_line)
        for _, kind, text in ops:
            if kind == "after":
                out.append(text)
        line_idx += 1

    new_text = "".join(out)
    return new_text


def apply_edits_to_repo(repo: Path, edits: Any) -> list[str]:
    """Apply exact structured source edits directly to files in ``repo``.

    This mirrors :func:`patch_from_edits` validation and semantics (same path
    checks, same find/replace/anchor rules, same per-edit uniqueness guards) but
    writes the modified file contents in place instead of synthesizing a unified
    diff and shelling out to ``git apply``. Applying directly removes the
    diff-roundtrip failure mode (hand-written/regenerated hunks that ``git
    apply`` rejects as "corrupt patch") and yields clear per-edit error
    messages. Returns the list of changed repo-relative paths.
    """
    if not isinstance(edits, list) or not edits:
        raise PatchValidationError("edits must be a non-empty list")
    _validate_edit_count(edits)

    originals: dict[str, str] = {}
    path_edits: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    path_order: list[str] = []
    for idx, edit in enumerate(edits):
        if not isinstance(edit, dict):
            raise PatchValidationError(f"edit {idx} must be an object")
        path = _edit_path(edit, idx)
        if path not in originals:
            originals[path] = _read_repo_text(repo, path)
            path_edits[path] = []
            path_order.append(path)
        path_edits[path].append((idx, edit))

    modified: dict[str, str] = {}
    for path in path_order:
        old_text = originals[path]
        new_text = _apply_path_edits(old_text, path_edits[path], path=path)
        if new_text != old_text:
            modified[path] = new_text

    changed: list[str] = []
    for path in path_order:
        new_text = modified.get(path)
        if new_text is None:
            continue
        if originals[path] == new_text:
            continue
        # _read_repo_text already validated the path resolves inside repo.
        (repo / path).write_text(new_text, encoding="utf-8")
        changed.append(path)
    if not changed:
        raise PatchValidationError("edits did not change any files")
    return changed


def changed_paths(patch: str) -> list[str]:
    paths: list[str] = []
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            match = re.match(r"diff --git a/(.+?) b/(.+)$", line)
            if match:
                paths.extend([match.group(1), match.group(2)])
        elif line.startswith("--- ") or line.startswith("+++ "):
            raw = line[4:].strip()
            if raw == "/dev/null":
                continue
            if raw.startswith("a/") or raw.startswith("b/"):
                raw = raw[2:]
            path = raw.split("\t", 1)[0]
            paths.append(path)
    out = []
    for path in paths:
        if path not in out:
            out.append(path)
    return out


def validate_changed_path(path: str) -> None:
    posix = PurePosixPath(path)
    if posix.is_absolute() or ".." in posix.parts:
        raise PatchValidationError(f"unsafe patch path: {path}")
    if not posix.parts or posix.parts[0] != "src":
        raise PatchValidationError(f"patch may only modify RawHash2 src/* files: {path}")
    if posix.suffix not in ALLOWED_SUFFIXES:
        raise PatchValidationError(f"unsupported source suffix in patch path: {path}")


def _edit_path(edit: dict[str, Any], idx: int) -> str:
    path = edit.get("path") or edit.get("file")
    if not isinstance(path, str) or not path.strip():
        raise PatchValidationError(f"edit {idx} missing path")
    path = path.strip()
    validate_changed_path(path)
    return path


def _read_repo_text(repo: Path, rel_path: str) -> str:
    path = repo / rel_path
    try:
        resolved = path.resolve()
        repo_resolved = repo.resolve()
    except OSError as exc:
        raise PatchValidationError(f"cannot resolve edit path {rel_path}: {exc}") from exc
    if repo_resolved not in resolved.parents:
        raise PatchValidationError(f"unsafe edit path escapes repo: {rel_path}")
    if not path.is_file():
        raise PatchValidationError(f"edit path does not exist: {rel_path}")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise PatchValidationError(f"edit path is not UTF-8 text: {rel_path}") from exc
    except OSError as exc:
        raise PatchValidationError(f"cannot read edit path {rel_path}: {exc}") from exc


def _apply_one_edit(text: str, edit: dict[str, Any], idx: int, *, path: str) -> str:
    replace_all = bool(edit.get("replace_all", False))
    has_find_replace = ("find" in edit or "find_lines" in edit) and (
        "replace" in edit or "replace_lines" in edit
    )
    has_after_insert = ("after" in edit or "after_lines" in edit) and (
        "insert" in edit or "insert_lines" in edit
    )
    has_before_insert = ("before" in edit or "before_lines" in edit) and (
        "insert" in edit or "insert_lines" in edit
    )
    if (
        ("function" in edit or "function_name" in edit)
        and not has_find_replace
        and not has_after_insert
        and not has_before_insert
    ):
        if replace_all:
            raise PatchValidationError(f"edit {idx} cannot use replace_all with function")
        function_name = _edit_function_name(
            edit.get("function", edit.get("function_name")),
            idx,
        )
        replace = _edit_text(edit, idx, "replace", "replace_lines", allow_empty=False)
        return _replace_function_by_name(text, path, function_name, replace, idx)
    if "anchor_id" in edit:
        if replace_all:
            raise PatchValidationError(f"edit {idx} cannot use replace_all with anchor_id")
        anchor_id = _edit_string(edit.get("anchor_id"), idx, "anchor_id", allow_empty=False)
        if "replace" in edit or "replace_lines" in edit:
            replace = _edit_text(edit, idx, "replace", "replace_lines", allow_empty=True)
            return _replace_line_anchor(text, path, anchor_id, replace, idx)
        if "insert_after" in edit or "insert_after_lines" in edit:
            insert = _edit_text(
                edit, idx, "insert_after", "insert_after_lines", allow_empty=False
            )
            return _insert_after_line_anchor(text, path, anchor_id, insert, idx)
        if "insert_before" in edit or "insert_before_lines" in edit:
            insert = _edit_text(
                edit, idx, "insert_before", "insert_before_lines", allow_empty=False
            )
            return _insert_before_line_anchor(text, path, anchor_id, insert, idx)
        raise PatchValidationError(
            f"edit {idx} with anchor_id must contain replace, replace_lines, "
            "insert_after, insert_after_lines, insert_before, or insert_before_lines"
        )
    scope_function = _edit_scope_function(edit, idx)
    occurrence = _edit_occurrence(edit, idx)
    if has_find_replace:
        find = _edit_text(edit, idx, "find", "find_lines", allow_empty=False)
        replace = _edit_text(edit, idx, "replace", "replace_lines", allow_empty=True)
        return _apply_find_replace(
            text,
            find,
            replace,
            idx,
            path=path,
            replace_all=replace_all,
            scope_function=scope_function,
            occurrence=occurrence,
        )
    if has_after_insert:
        anchor = _edit_text(edit, idx, "after", "after_lines", allow_empty=False)
        insert = _edit_text(edit, idx, "insert", "insert_lines", allow_empty=False)
        return _replace_exact(
            text,
            anchor,
            anchor + insert,
            idx,
            path=path,
            replace_all=replace_all,
            scope_function=scope_function,
            occurrence=occurrence,
        )
    if has_before_insert:
        anchor = _edit_text(edit, idx, "before", "before_lines", allow_empty=False)
        insert = _edit_text(edit, idx, "insert", "insert_lines", allow_empty=False)
        return _replace_exact(
            text,
            anchor,
            insert + anchor,
            idx,
            path=path,
            replace_all=replace_all,
            scope_function=scope_function,
            occurrence=occurrence,
        )
    raise PatchValidationError(
        f"edit {idx} must contain function+replace, anchor_id edit fields, "
        "find+replace, find_lines+replace_lines, after+insert, or before+insert"
    )


def _edit_string(value: Any, idx: int, key: str, *, allow_empty: bool) -> str:
    if not isinstance(value, str):
        raise PatchValidationError(f"edit {idx} field {key!r} must be a string")
    if not allow_empty and value == "":
        raise PatchValidationError(f"edit {idx} field {key!r} must be non-empty")
    max_edit_bytes = _max_edit_bytes()
    if len(value.encode("utf-8")) > max_edit_bytes:
        raise PatchValidationError(f"edit {idx} field {key!r} exceeds {max_edit_bytes} bytes")
    return value


def _edit_function_name(value: Any, idx: int) -> str:
    name = _edit_string(value, idx, "function", allow_empty=False).strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise PatchValidationError(f"edit {idx} field 'function' must be a C identifier")
    return name


def _edit_scope_function(edit: dict[str, Any], idx: int) -> str | None:
    value = (
        edit.get("scope_function")
        if "scope_function" in edit
        else edit.get("within_function", edit.get("function_scope"))
    )
    if value is None and (
        "find" in edit
        or "find_lines" in edit
        or "after" in edit
        or "after_lines" in edit
        or "before" in edit
        or "before_lines" in edit
    ):
        value = edit.get("function", edit.get("function_name"))
    if value is None:
        return None
    return _edit_function_name(value, idx)


def _edit_occurrence(edit: dict[str, Any], idx: int) -> int | None:
    value = edit.get("occurrence", edit.get("match_index"))
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise PatchValidationError(f"edit {idx} field 'occurrence' must be a positive integer")
    if value < 1:
        raise PatchValidationError(f"edit {idx} field 'occurrence' must be a positive integer")
    return value


def _edit_text(
    edit: dict[str, Any],
    idx: int,
    string_key: str,
    lines_key: str,
    *,
    allow_empty: bool,
) -> str:
    has_string = string_key in edit
    has_lines = lines_key in edit
    if has_string and has_lines:
        raise PatchValidationError(
            f"edit {idx} must use either {string_key!r} or {lines_key!r}, not both"
        )
    if has_string:
        return _edit_string(edit.get(string_key), idx, string_key, allow_empty=allow_empty)
    if not has_lines:
        raise PatchValidationError(f"edit {idx} missing field {string_key!r} or {lines_key!r}")
    lines = edit.get(lines_key)
    if not isinstance(lines, list):
        raise PatchValidationError(f"edit {idx} field {lines_key!r} must be a list of strings")
    if not lines and not allow_empty:
        raise PatchValidationError(f"edit {idx} field {lines_key!r} must be non-empty")
    out: list[str] = []
    for line_no, line in enumerate(lines):
        if not isinstance(line, str):
            raise PatchValidationError(
                f"edit {idx} field {lines_key!r}[{line_no}] must be a string"
            )
        if "\n" in line or "\r" in line:
            out.extend(line.replace("\r\n", "\n").replace("\r", "\n").split("\n"))
        else:
            out.append(line)
    value = "\n".join(out)
    max_edit_bytes = _max_edit_bytes()
    if len(value.encode("utf-8")) > max_edit_bytes:
        raise PatchValidationError(f"edit {idx} field {lines_key!r} exceeds {max_edit_bytes} bytes")
    return value


def _validate_edit_count(edits: list[Any]) -> None:
    max_structured_edits = _max_structured_edits()
    if max_structured_edits > 0 and len(edits) > max_structured_edits:
        raise PatchValidationError(
            f"too many edits; maximum is {max_structured_edits} "
            "(set RAWHASH2_NATIVE_MAX_STRUCTURED_EDITS to raise it)"
        )


def _max_patch_bytes() -> int:
    return _positive_env_int("RAWHASH2_NATIVE_MAX_PATCH_BYTES", DEFAULT_MAX_PATCH_BYTES)


def _max_edit_bytes() -> int:
    return _positive_env_int("RAWHASH2_NATIVE_MAX_EDIT_BYTES", DEFAULT_MAX_EDIT_BYTES)


def _max_structured_edits() -> int:
    return _nonnegative_env_int(
        "RAWHASH2_NATIVE_MAX_STRUCTURED_EDITS",
        DEFAULT_MAX_STRUCTURED_EDITS,
    )


def _positive_env_int(name: str, default: int) -> int:
    value = _env_int(name, default)
    return value if value > 0 else default


def _nonnegative_env_int(name: str, default: int) -> int:
    return max(0, _env_int(name, default))


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _line_anchor_index(text: str, path: str) -> dict[str, int]:
    return {anchor.anchor_id: anchor.line_no - 1 for anchor in line_anchors_for_text(path, text)}


def _normalize_anchor_id(anchor_id: str) -> str:
    anchor_id = anchor_id.strip().strip("`").strip()
    if anchor_id.startswith("anchor_id="):
        anchor_id = anchor_id[len("anchor_id=") :].strip()
    if " | " in anchor_id:
        anchor_id = anchor_id.split(" | ", 1)[0].strip()
    if " ; " in anchor_id:
        anchor_id = anchor_id.split(" ; ", 1)[0].strip()
    if anchor_id.startswith("- "):
        anchor_id = anchor_id[2:].strip()
    return anchor_id


def _anchor_preview_text(anchor_id: str) -> str | None:
    if " | " in anchor_id:
        return anchor_id.split(" | ", 1)[1].strip()
    marker = "source="
    if marker in anchor_id:
        return anchor_id.split(marker, 1)[1].strip()
    return None


def _resolve_line_anchor(text: str, path: str, anchor_id: str) -> int | None:
    preview = _anchor_preview_text(anchor_id)
    anchor_id = _normalize_anchor_id(anchor_id)
    anchors = line_anchors_for_text(path, text)
    exact = {anchor.anchor_id: anchor.line_no - 1 for anchor in anchors}
    if anchor_id in exact:
        return exact[anchor_id]

    prefix_matches = [
        anchor.line_no - 1 for anchor in anchors if anchor.anchor_id.startswith(anchor_id)
    ]
    if len(prefix_matches) == 1:
        return prefix_matches[0]

    if re.fullmatch(r"[0-9a-fA-F]{6,}", anchor_id):
        suffix_matches = [
            anchor.line_no - 1
            for anchor in anchors
            if anchor.anchor_id.endswith(f":{anchor_id}")
        ]
        if len(suffix_matches) == 1:
            return suffix_matches[0]

    line_hash_match = re.fullmatch(r"L[0-9]+:[0-9a-fA-F]{6,}", anchor_id)
    if line_hash_match:
        suffix_matches = [
            anchor.line_no - 1 for anchor in anchors if anchor.anchor_id.endswith(anchor_id)
        ]
        if len(suffix_matches) == 1:
            return suffix_matches[0]

    if preview:
        preview = preview.replace("\\t", "\t").strip()
        preview_matches = [
            anchor.line_no - 1 for anchor in anchors if anchor.text.strip() == preview
        ]
        if len(preview_matches) == 1:
            return preview_matches[0]

    match = re.fullmatch(r"(.+):L([0-9]+)(?::([0-9a-fA-F]*))?", anchor_id)
    if not match:
        return None
    anchor_path, raw_line_no, raw_digest = match.groups()
    if anchor_path != path:
        return None
    line_no = int(raw_line_no)
    if line_no < 1:
        return None
    for anchor in anchors:
        if anchor.line_no != line_no:
            continue
        if raw_digest and not anchor.anchor_id.startswith(f"{path}:L{line_no}:{raw_digest}"):
            if _env_bool("RAWHASH2_NATIVE_LINE_ANCHOR_NUMBER_FALLBACK", True):
                return line_no - 1
            return None
        return line_no - 1
    return None


def _resolve_line_anchor_from_replacement(
    text: str,
    path: str,
    anchor_id: str,
    replace: str,
) -> int | None:
    if not _env_bool("RAWHASH2_NATIVE_LINE_ANCHOR_NUMBER_FALLBACK", True):
        return None
    match = re.fullmatch(r"(.+):L([0-9]+)(?::([0-9a-fA-F]*))?", _normalize_anchor_id(anchor_id))
    if not match or match.group(1) != path:
        return None

    first_code_line = next((line for line in replace.splitlines() if _norm_code_line(line)), "")
    if not first_code_line:
        return None
    lhs_match = re.match(
        r"\s*([A-Za-z_][A-Za-z0-9_]*(?:->|\.)[A-Za-z_][A-Za-z0-9_]*)\s*=",
        first_code_line,
    )
    lhs = lhs_match.group(1) if lhs_match else None
    option_markers = re.findall(r"--[A-Za-z0-9][A-Za-z0-9_-]*", first_code_line)

    candidates: list[int] = []
    for idx, line in enumerate(text.splitlines()):
        if lhs and re.search(rf"\b{re.escape(lhs)}\s*=", line):
            candidates.append(idx)
            continue
        if option_markers and any(marker in line for marker in option_markers):
            candidates.append(idx)
    unique = sorted(set(candidates))
    return unique[0] if len(unique) == 1 else None


def _include_insert_after_line(text: str, insert: str) -> int | None:
    insert_lines = [line.strip() for line in insert.splitlines() if line.strip()]
    if not insert_lines or not all(line.startswith("#include ") for line in insert_lines):
        return None
    last_include = -1
    for idx, line in enumerate(text.splitlines()[:120]):
        stripped = line.strip()
        if stripped.startswith("#include "):
            last_include = idx
    return last_include if last_include >= 0 else None


def _replacement_function_name(replace: str) -> str | None:
    if "\n" not in replace:
        return None
    for name, _, _ in _iter_function_definitions(replace):
        return name
    return None


def _function_replacement_span(text: str, replace: str) -> tuple[int, int] | None:
    name = _replacement_function_name(replace)
    if name is None:
        return None
    return _find_function_span_by_name(text, name)


def _find_function_span_by_name(text: str, name: str) -> tuple[int, int] | None:
    spans: list[tuple[int, int]] = []
    for found_name, start_line, end_line in _iter_function_definitions(text):
        if found_name != name:
            continue
        spans.append((start_line, end_line))
    return spans[0] if len(spans) == 1 else None


def _iter_function_definitions(text: str) -> list[tuple[str, int, int]]:
    """Return ``(name, start_line, end_line)`` for C-like function definitions.

    This deliberately stays conservative: it only accepts definitions with a
    visible return/type prefix before ``name(``, a balanced parameter list, and
    an opening brace after whitespace and comments. That covers RawHash2's
    multiline signatures while avoiding fuzzy block matching.
    """
    definitions: list[tuple[str, int, int]] = []
    pattern = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
    for match in pattern.finditer(text):
        name = match.group(1)
        if name in {"if", "for", "while", "switch", "return", "sizeof"}:
            continue
        name_start = match.start(1)
        line_start = text.rfind("\n", 0, name_start) + 1
        prefix = text[line_start:name_start].strip()
        if not _looks_like_function_prefix(prefix):
            continue
        open_paren = text.find("(", match.start(1), match.end())
        if open_paren < 0:
            continue
        close_paren = _find_matching_delimiter(text, open_paren, "(", ")")
        if close_paren is None:
            continue
        open_brace = _skip_c_ws_and_comments(text, close_paren + 1)
        if open_brace >= len(text) or text[open_brace] != "{":
            continue
        close_brace = _find_matching_delimiter(text, open_brace, "{", "}")
        if close_brace is None:
            continue
        end_pos = close_brace + 1
        if end_pos < len(text) and text[end_pos] == "\n":
            end_pos += 1
        start_line = text[:line_start].count("\n")
        end_line = text[:end_pos].count("\n")
        if end_pos > 0 and text[end_pos - 1] != "\n":
            end_line += 1
        definitions.append((name, start_line, end_line))
    return definitions


def _looks_like_function_prefix(prefix: str) -> bool:
    if not prefix:
        return False
    if prefix.startswith("#") or re.search(r"\b(return|sizeof|case)\b", prefix):
        return False
    if any(token in prefix for token in ("=", ".", "->")):
        return False
    if prefix.endswith((",", "(", "[")):
        return False
    return bool(re.search(r"[A-Za-z_][A-Za-z0-9_]*|\*", prefix))


def _find_matching_delimiter(
    text: str,
    open_pos: int,
    open_ch: str,
    close_ch: str,
) -> int | None:
    depth = 0
    pos = open_pos
    while pos < len(text):
        skipped = _skip_c_ignored(text, pos)
        if skipped != pos:
            pos = skipped
            continue
        ch = text[pos]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return pos
        pos += 1
    return None


def _skip_c_ws_and_comments(text: str, pos: int) -> int:
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        skipped = _skip_c_comment(text, pos)
        if skipped == pos:
            return pos
        pos = skipped
    return pos


def _skip_c_ignored(text: str, pos: int) -> int:
    skipped = _skip_c_comment(text, pos)
    if skipped != pos:
        return skipped
    if text[pos] in {"'", '"'}:
        return _skip_c_string(text, pos)
    return pos


def _skip_c_comment(text: str, pos: int) -> int:
    if text.startswith("//", pos):
        newline = text.find("\n", pos + 2)
        return len(text) if newline < 0 else newline + 1
    if text.startswith("/*", pos):
        end = text.find("*/", pos + 2)
        return len(text) if end < 0 else end + 2
    return pos


def _skip_c_string(text: str, pos: int) -> int:
    quote = text[pos]
    pos += 1
    while pos < len(text):
        if text[pos] == "\\":
            pos += 2
            continue
        if text[pos] == quote:
            return pos + 1
        pos += 1
    return pos


def _replace_function_by_name(
    text: str,
    path: str,
    function_name: str,
    replace: str,
    idx: int,
) -> str:
    span = _find_function_span_by_name(text, function_name)
    if span is None:
        raise PatchValidationError(
            f"edit {idx} function {function_name!r} was not found uniquely in {path}"
        )
    if _find_function_span_by_name(replace, function_name) is None:
        raise PatchValidationError(
            f"edit {idx} replacement must define function {function_name!r}"
        )
    lines = text.splitlines(keepends=True)
    if not replace.endswith(("\n", "\r")):
        replace += "\n"
    lines[span[0] : span[1]] = [replace]
    return "".join(lines)


def _replace_line_anchor(
    text: str,
    path: str,
    anchor_id: str,
    replace: str,
    idx: int,
) -> str:
    span = _function_replacement_span(text, replace)
    if span is not None:
        lines = text.splitlines(keepends=True)
        if not replace.endswith(("\n", "\r")):
            replace += "\n"
        lines[span[0] : span[1]] = [replace]
        return "".join(lines)
    lines = text.splitlines(keepends=True)
    line_idx = _resolve_line_anchor(text, path, anchor_id)
    if line_idx is None:
        line_idx = _resolve_line_anchor_from_replacement(text, path, anchor_id, replace)
    if line_idx is None or line_idx >= len(lines):
        raise PatchValidationError(f"edit {idx} line anchor was not found: {anchor_id}")
    old = lines[line_idx]
    if old.endswith(("\n", "\r")) and not replace.endswith(("\n", "\r")):
        replace += "\n"
    lines[line_idx] = replace
    return "".join(lines)


def _insert_after_line_anchor(
    text: str,
    path: str,
    anchor_id: str,
    insert: str,
    idx: int,
) -> str:
    lines = text.splitlines(keepends=True)
    line_idx = _include_insert_after_line(text, insert)
    if line_idx is None:
        line_idx = _resolve_line_anchor(text, path, anchor_id)
    if line_idx is None or line_idx >= len(lines):
        raise PatchValidationError(f"edit {idx} line anchor was not found: {anchor_id}")
    if not insert.endswith(("\n", "\r")):
        insert += "\n"
    lines.insert(line_idx + 1, insert)
    return "".join(lines)


def _insert_before_line_anchor(
    text: str,
    path: str,
    anchor_id: str,
    insert: str,
    idx: int,
) -> str:
    lines = text.splitlines(keepends=True)
    line_idx = _resolve_line_anchor(text, path, anchor_id)
    if line_idx is None or line_idx >= len(lines):
        raise PatchValidationError(f"edit {idx} line anchor was not found: {anchor_id}")
    if not insert.endswith(("\n", "\r")):
        insert += "\n"
    lines.insert(line_idx, insert)
    return "".join(lines)


def _replace_exact(
    text: str,
    find: str,
    replace: str,
    idx: int,
    *,
    path: str,
    replace_all: bool,
    scope_function: str | None = None,
    occurrence: int | None = None,
) -> str:
    if scope_function is not None:
        return _apply_within_function_scope(
            text,
            path,
            scope_function,
            idx,
            lambda scoped_text: _replace_exact(
                scoped_text,
                find,
                replace,
                idx,
                path=path,
                replace_all=replace_all,
                scope_function=None,
                occurrence=occurrence,
            ),
        )
    if replace_all and occurrence is not None:
        raise PatchValidationError(f"edit {idx} cannot use occurrence with replace_all")
    matches = text.count(find)
    if matches == 0:
        raise PatchValidationError(f"edit {idx} find/anchor text was not found exactly")
    if occurrence is not None:
        return _replace_nth_exact(text, find, replace, idx, occurrence, matches)
    if matches > 1 and not replace_all:
        raise PatchValidationError(
            f"edit {idx} find/anchor text matched {matches} times; make it unique or set replace_all"
        )
    return text.replace(find, replace) if replace_all else text.replace(find, replace, 1)


def _apply_find_replace(
    text: str,
    find: str,
    replace: str,
    idx: int,
    *,
    path: str,
    replace_all: bool,
    scope_function: str | None = None,
    occurrence: int | None = None,
) -> str:
    """Apply a find/replace edit, exact first then whitespace/comment-tolerant.

    The model copies ``find`` from a budget-compacted view of the source (which
    strips comments, trailing whitespace and blank runs) and may use spaces where
    the file uses tabs, so a long verbatim block rarely matches the raw file byte
    for byte. We first try an exact match (cheap, unambiguous); if the text is not
    found we fall back to a line-based tolerant match that ignores insignificant C
    whitespace and comments but still requires the same code tokens in the same
    order and a unique location.
    """
    if scope_function is not None:
        return _apply_within_function_scope(
            text,
            path,
            scope_function,
            idx,
            lambda scoped_text: _apply_find_replace(
                scoped_text,
                find,
                replace,
                idx,
                path=path,
                replace_all=replace_all,
                scope_function=None,
                occurrence=occurrence,
            ),
        )
    if replace_all and occurrence is not None:
        raise PatchValidationError(f"edit {idx} cannot use occurrence with replace_all")
    matches = text.count(find)
    if matches == 1 or (matches > 1 and replace_all):
        return text.replace(find, replace) if replace_all else text.replace(find, replace, 1)
    if occurrence is not None:
        if matches > 0:
            return _replace_nth_exact(text, find, replace, idx, occurrence, matches)
        return _replace_find_tolerant(
            text,
            find,
            replace,
            idx,
            replace_all=replace_all,
            occurrence=occurrence,
        )
    if matches > 1:
        raise PatchValidationError(
            f"edit {idx} find/anchor text matched {matches} times; make it unique or set replace_all"
        )
    return _replace_find_tolerant(
        text,
        find,
        replace,
        idx,
        replace_all=replace_all,
        occurrence=occurrence,
    )


def _norm_code_line(line: str) -> str:
    """Normalize a C source line for tolerant matching.

    Drops ``//`` and inline ``/* ... */`` comments, collapses internal runs of
    spaces/tabs to a single space, and strips leading/trailing whitespace. This
    makes tab-vs-space, indentation and trailing-whitespace differences (and the
    prompt's comment stripping) irrelevant to matching while preserving the code
    tokens. Block comments that span multiple lines are only partially handled
    (per-line), which is acceptable: an unmatched line simply falls through.
    """
    return normalize_code_line(line)


def _replace_find_tolerant(
    text: str,
    find: str,
    replace: str,
    idx: int,
    *,
    replace_all: bool,
    occurrence: int | None = None,
) -> str:
    """Line-based whitespace/comment-tolerant find/replace.

    Matches the non-blank normalized lines of ``find`` as a contiguous block of
    non-blank normalized lines in ``text`` (interior blank/comment-only lines are
    skipped for matching but included in the replaced raw span), then replaces the
    corresponding raw line range with ``replace``. Requires a unique match unless
    ``replace_all`` is set.
    """
    raw_lines = text.split("\n")
    norm_find = [n for n in (_norm_code_line(l) for l in find.split("\n")) if n]
    if not norm_find:
        raise PatchValidationError(f"edit {idx} find/anchor text was not found exactly")

    nonblank = [(i, n) for i, n in ((j, _norm_code_line(l)) for j, l in enumerate(raw_lines)) if n]
    seq = [n for _, n in nonblank]
    raw_index = [i for i, _ in nonblank]

    m = len(norm_find)
    starts = [k for k in range(len(seq) - m + 1) if seq[k : k + m] == norm_find]
    if not starts:
        raise PatchValidationError(f"edit {idx} find/anchor text was not found exactly")
    if replace_all and occurrence is not None:
        raise PatchValidationError(f"edit {idx} cannot use occurrence with replace_all")
    if occurrence is not None:
        if occurrence > len(starts):
            raise PatchValidationError(
                f"edit {idx} occurrence {occurrence} exceeds {len(starts)} matches"
            )
        starts = [starts[occurrence - 1]]
    if len(starts) > 1 and not replace_all:
        raise PatchValidationError(
            f"edit {idx} find/anchor text matched {len(starts)} times; make it unique or set replace_all"
        )

    replacement_lines = replace.split("\n")
    new_lines = list(raw_lines)
    for k in sorted(starts, reverse=True):
        lo = raw_index[k]
        hi = raw_index[k + m - 1]
        new_lines[lo : hi + 1] = replacement_lines
        if not replace_all:
            break
    return "\n".join(new_lines)


def _replace_nth_exact(
    text: str,
    find: str,
    replace: str,
    idx: int,
    occurrence: int,
    matches: int,
) -> str:
    if occurrence > matches:
        raise PatchValidationError(f"edit {idx} occurrence {occurrence} exceeds {matches} matches")
    start = -1
    search_from = 0
    for _ in range(occurrence):
        start = text.find(find, search_from)
        search_from = start + len(find)
    if start < 0:
        raise PatchValidationError(f"edit {idx} find/anchor text was not found exactly")
    return text[:start] + replace + text[start + len(find) :]


def _apply_within_function_scope(
    text: str,
    path: str,
    function_name: str,
    idx: int,
    apply_scoped: Any,
) -> str:
    span = _find_function_span_by_name(text, function_name)
    if span is None:
        raise PatchValidationError(
            f"edit {idx} scope_function {function_name!r} was not found uniquely in {path}"
        )
    lines = text.splitlines(keepends=True)
    scoped_text = "".join(lines[span[0] : span[1]])
    new_scoped_text = apply_scoped(scoped_text)
    if new_scoped_text == scoped_text:
        return text
    lines[span[0] : span[1]] = [new_scoped_text]
    return "".join(lines)


def _git_diff_from_texts(changed_texts: dict[str, tuple[str, str]]) -> str:
    """Return a git-formatted diff for exact old/new file contents."""
    if not changed_texts:
        return ""
    with tempfile.TemporaryDirectory(prefix="rawhash2-edits-") as tmp:
        work = Path(tmp)
        init_proc = subprocess.run(
            ["git", "init", "-q"],
            cwd=str(work),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if init_proc.returncode != 0:
            msg = (init_proc.stderr or init_proc.stdout).strip()
            raise PatchValidationError(f"git init failed while building patch: {msg}")

        paths = list(changed_texts)
        for rel_path, (old_text, _) in changed_texts.items():
            path = work / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(old_text, encoding="utf-8")

        add_proc = subprocess.run(
            ["git", "add", "--", *paths],
            cwd=str(work),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if add_proc.returncode != 0:
            msg = (add_proc.stderr or add_proc.stdout).strip()
            raise PatchValidationError(f"git add failed while building patch: {msg}")

        for rel_path, (_, new_text) in changed_texts.items():
            (work / rel_path).write_text(new_text, encoding="utf-8")

        diff_proc = subprocess.run(
            ["git", "diff", "--", *paths],
            cwd=str(work),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if diff_proc.returncode not in (0, 1):
            msg = (diff_proc.stderr or diff_proc.stdout).strip()
            raise PatchValidationError(f"git diff failed while building patch: {msg}")
        return diff_proc.stdout


def apply_patch_to_repo(repo: Path, patch: str) -> None:
    normalize_patch(patch)
    if not (repo / ".git").exists():
        init_proc = subprocess.run(
            ["git", "init", "-q"],
            cwd=str(repo),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if init_proc.returncode != 0:
            msg = (init_proc.stderr or init_proc.stdout).strip()
            raise PatchValidationError(f"git init failed: {msg}")
    proc = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        cwd=str(repo),
        input=patch,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout).strip()
        raise PatchValidationError(f"git apply failed: {msg}")


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)
