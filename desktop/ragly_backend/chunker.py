"""Sentence-aware chunking. Chunks never cross page boundaries so citations point to one page."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

_SENT = re.compile(r"(?<=[.!?;:])\s+|\n{2,}|\n(?=\s*(?:[-*•]|\d+[.)])\s)")
_HEADING = re.compile(r"^(?:[A-Z0-9][A-Z0-9 &/,.()-]{2,80}|(?:\d+(?:\.\d+)*)\s+\S.{0,80})$")


@dataclass
class Chunk:
    page: int
    ord: int
    text: str
    heading: str


def clean(text: str) -> str:
    text = text.replace("\r", "\n").replace("­", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # de-hyphenate line breaks
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"(?<![.!?:\n])\n(?!\n)", " ", text)       # join wrapped lines
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _detect_heading(line: str) -> bool:
    line = line.strip()
    return 3 <= len(line) <= 90 and bool(_HEADING.match(line)) and not line.endswith(".")


def chunk_pages(
    pages: list[tuple[int, str]],
    count_tokens: Callable[[str], int],
    split_long: Callable[[str, int, int], list[str]],
    max_tokens: int = 380,
    overlap: int = 60,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    heading = ""
    for page_no, raw in pages:
        # headings are detected on raw lines before wrapped lines are joined
        for line in raw.splitlines()[:5]:
            if _detect_heading(line):
                heading = line.strip()
                break
        text = clean(raw)
        if not text:
            continue
        sentences: list[tuple[str, int]] = []
        for s in _SENT.split(text):
            s = s.strip()
            if not s:
                continue
            n = count_tokens(s)
            if n > max_tokens:
                for piece in split_long(s, max_tokens, overlap):
                    sentences.append((piece, count_tokens(piece)))
            else:
                sentences.append((s, n))

        buf: list[tuple[str, int]] = []
        size = 0
        for s, n in sentences:
            if buf and size + n > max_tokens:
                chunks.append(Chunk(page_no, len(chunks), " ".join(x for x, _ in buf), heading))
                carry: list[tuple[str, int]] = []
                carried = 0
                for prev in reversed(buf):
                    if carried + prev[1] > overlap:
                        break
                    carry.insert(0, prev)
                    carried += prev[1]
                buf, size = carry, carried
            buf.append((s, n))
            size += n
        if buf:
            chunks.append(Chunk(page_no, len(chunks), " ".join(x for x, _ in buf), heading))
    return chunks
