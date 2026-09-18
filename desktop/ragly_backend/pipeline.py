"""Multimodal ingestion pipeline: one document in, structured knowledge out.

For every document we produce
    pages   - with a content type (text / scanned_text / table / image / diagram / mixed)
    tables  - headers + rows preserved, so arithmetic can be computed instead of guessed
    images  - extracted rasters (or a rendered page when a page is a picture)
    chunks  - text chunks, including a readable rendering of each table
keeping the relationship document -> page -> content -> chunk so every answer can
point back to exact evidence.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pymupdf as fitz

from . import tables as tbl
from .chunker import Chunk, chunk_pages
from .config import settings
from .ingest import DOCX_EXT, IMAGE_EXT, PDF_EXT, SUPPORTED, TEXT_EXT, OCR, extract
from .images import extract_images

log = logging.getLogger("ragly.pipeline")

SHEET_EXT = {".xlsx", ".xlsm", ".xltx", ".csv", ".tsv"}
ALL_SUPPORTED = SUPPORTED | SHEET_EXT


def classify_page(text: str, has_table: bool, has_image: bool, ocr_used: bool,
                  image_area_ratio: float = 0.0) -> str:
    """Decide what a page actually contains. Cheap rules, explainable in the UI."""
    chars = len((text or "").strip())
    if chars < 15 and not has_table:
        return "image" if has_image else "empty"
    if ocr_used:
        return "mixed" if has_table else "scanned_text"
    if has_table and chars < 900:
        return "table"
    if has_table:
        return "mixed"
    if has_image and image_area_ratio > 0.45 and chars < 400:
        return "diagram"
    if has_image:
        return "mixed"
    return "text"


def _pdf_page_facts(path: Path) -> dict[int, dict]:
    """Per page: does it hold images, and how much of the page do they cover."""
    facts: dict[int, dict] = {}
    try:
        with fitz.open(path) as doc:
            for i, page in enumerate(doc, start=1):
                area = 0.0
                try:
                    for info in page.get_image_info():
                        r = info.get("bbox")
                        if r:
                            area += abs((r[2] - r[0]) * (r[3] - r[1]))
                except Exception:
                    pass
                page_area = abs(page.rect.width * page.rect.height) or 1.0
                facts[i] = {"has_image": area > 0, "image_area_ratio": min(1.0, area / page_area)}
    except Exception as exc:
        log.warning("could not read image info from %s: %s", path.name, exc)
    return facts


def strip_running_headers(raw_pages: list[tuple[int, str]], min_share: float = 0.6) -> list[tuple[int, str]]:
    """Remove the header and footer a publisher repeats on every page.

    Running furniture ("Synthetic RAG Test Corpus … Page 7") lands in every chunk: it wastes
    prompt space, it matches every query a little, and it turns up as an entity. Exactly two
    candidates are considered - the line most often at the top of a page, and the line most
    often at the bottom - and each is removed only if it recurs on most pages. Page numbers
    differ per page, so lines are compared with their digits masked. A document under four
    pages is left alone, and a page is never emptied.
    """
    if len(raw_pages) < 4:
        return raw_pages
    import collections
    import re as _re

    mask = lambda ln: _re.sub(r"\d+", "#", ln.strip())
    tops: collections.Counter = collections.Counter()
    bottoms: collections.Counter = collections.Counter()
    for _page, text in raw_pages:
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        if not lines:
            continue
        if 8 <= len(lines[0]) <= 160:
            tops[mask(lines[0])] += 1
        if len(lines) > 1 and 8 <= len(lines[-1]) <= 160:
            bottoms[mask(lines[-1])] += 1

    threshold = max(3, int(len(raw_pages) * min_share))
    furniture = {line for counter in (tops, bottoms)
                 for line, count in counter.most_common(1) if count >= threshold}
    if not furniture:
        return raw_pages

    out: list[tuple[int, str]] = []
    removed = 0
    for page, text in raw_pages:
        lines = [ln for ln in (text or "").splitlines()]
        stripped = [ln.strip() for ln in lines]
        first = next((i for i, ln in enumerate(stripped) if ln), None)
        last = next((i for i in range(len(stripped) - 1, -1, -1) if stripped[i]), None)
        keep = [ln for i, ln in enumerate(lines)
                if not (i in (first, last) and mask(ln) in furniture)]
        removed += len(lines) - len(keep)
        out.append((page, "\n".join(keep).strip() or (text or "")))
    if removed:
        log.info("removed %d running header/footer line(s)", removed)
    return out


def process(path: Path, doc_id: int, images_dir: Path, count_tokens, split_long,
            display_name: str | None = None) -> dict:
    """Run the whole extraction for one file. Returns pages, tables, chunks and image records."""
    ext = path.suffix.lower()
    pages: list[dict] = []
    tables: list[dict] = []
    image_records: list[dict] = []
    ocr_pages = 0

    if ext in SHEET_EXT:
        tables = tbl.extract_sheet_tables(path)
        stem_now = Path(display_name or path.name).stem
        for t in tables:
            if not t.get("title") or t.get("title") == path.stem:
                t["title"] = stem_now
            pages.append({"page": t["page"], "text": tbl.table_to_text(t, max_chars=4000),
                          "content_type": "table", "ocr_used": False, "has_table": True,
                          "has_image": False})
    else:
        raw_pages, ocr_pages = extract(path)
        raw_pages = strip_running_headers(raw_pages)
        if ext in PDF_EXT:
            tables = tbl.extract_pdf_tables(path)
            facts = _pdf_page_facts(path)
        else:
            facts = {}
        table_pages = {t["page"] for t in tables}
        for page_no, text in raw_pages:
            f = facts.get(page_no, {})
            has_image = bool(f.get("has_image")) or ext in IMAGE_EXT
            ocr_used = ocr_pages > 0 and len((text or "").strip()) > 0 and page_no in _ocr_hint(raw_pages, facts)
            pages.append({
                "page": page_no,
                "text": text or "",
                "has_table": page_no in table_pages,
                "has_image": has_image,
                "ocr_used": ocr_used,
                "content_type": classify_page(text, page_no in table_pages, has_image, ocr_used,
                                              float(f.get("image_area_ratio", 0.0))),
            })

        # images: embedded rasters, plus a render of pages that are pictures with no raster we could pull
        render = [p["page"] for p in pages if p["content_type"] in ("image", "diagram", "scanned_text")]
        try:
            image_records = extract_images(path, images_dir, doc_id, render_pages=render)
        except Exception as exc:
            log.warning("image extraction failed for %s: %s", path.name, exc)

    # the stored copy is named by hash, so tables inherit the real document name
    stem = Path(display_name or path.name).stem
    for t in tables:
        if not t.get("title") or t.get("title") == path.stem:
            t["title"] = stem

    for p in pages:
        p["char_count"] = len((p.get("text") or "").strip())

    # ---- chunks: normal text, then a readable rendering of every table
    text_pages = [(p["page"], p["text"]) for p in pages if p["content_type"] != "table" and p["text"].strip()]
    chunks: list[Chunk] = chunk_pages(text_pages, count_tokens, split_long,
                                      settings.chunk_tokens, settings.chunk_overlap)
    table_chunk_map: list[tuple[int, int]] = []     # (index in chunks, index in tables)
    for ti, t in enumerate(tables):
        for part in tbl.table_to_chunks(t):
            table_chunk_map.append((len(chunks), ti))
            chunks.append(Chunk(page=t["page"], ord=len(chunks), text=part,
                                heading=t.get("title", "") or "table"))

    return {
        "pages": pages,
        "tables": tables,
        "chunks": chunks,
        "table_chunk_map": table_chunk_map,
        "images": image_records,
        "ocr_pages": ocr_pages,
        "page_count": len(pages),
    }


def _ocr_hint(raw_pages: list[tuple[int, str]], facts: dict) -> set[int]:
    """Pages we most likely OCR'd: they have images and their text arrived anyway."""
    return {p for p, text in raw_pages
            if facts.get(p, {}).get("has_image") and len((text or "").strip()) > 0}
