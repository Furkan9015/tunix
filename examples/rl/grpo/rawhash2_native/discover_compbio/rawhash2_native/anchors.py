"""Stable source-line anchors for RawHash2 structured edits."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class LineAnchor:
    path: str
    line_no: int
    anchor_id: str
    text: str


def normalize_code_line(line: str) -> str:
    """Normalize one C/C++ source line while preserving code identity."""
    line = re.sub(r"/\*.*?\*/", " ", line)
    line = re.sub(r"//.*$", "", line)
    return re.sub(r"[ \t]+", " ", line).strip()


def make_line_anchor_id(path: str, line_no: int, line: str) -> str:
    norm = normalize_code_line(line)
    digest = hashlib.sha1(f"{path}\0{line_no}\0{norm}".encode()).hexdigest()[:10]
    return f"{path}:L{line_no}:{digest}"


def line_anchors_for_text(path: str, text: str) -> list[LineAnchor]:
    anchors: list[LineAnchor] = []
    for idx, line in enumerate(text.splitlines(), 1):
        if not normalize_code_line(line):
            continue
        anchors.append(
            LineAnchor(
                path=path,
                line_no=idx,
                anchor_id=make_line_anchor_id(path, idx, line),
                text=line.rstrip(),
            )
        )
    return anchors
