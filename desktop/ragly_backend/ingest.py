"""Text extraction: PDF text layer, OCR fallback for scanned pages and images, plain text files."""
from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path

import pymupdf as fitz
import numpy as np

from .config import settings

log = logging.getLogger("ragly.ingest")

PDF_EXT = {".pdf"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
TEXT_EXT = {".txt", ".md", ".markdown", ".csv", ".log", ".json"}
DOCX_EXT = {".docx"}
SUPPORTED = PDF_EXT | IMAGE_EXT | TEXT_EXT | DOCX_EXT
TEXT_PAGE_CHARS = 3000  # plain-text files are split into pseudo-pages of this size


class _WindowsOCR:
    """Windows built-in OCR (Windows.Media.Ocr). Offline, ships with Windows, works on x64 and ARM64."""

    name = "windows"

    def __init__(self):
        from winrt.windows.globalization import Language  # noqa: F401  (import check)
        from winrt.windows.media.ocr import OcrEngine

        engine = None
        for tag in ("en-US", "en-GB", "en-IN", "en"):
            try:
                lang = Language(tag)
                if OcrEngine.is_language_supported(lang):
                    engine = OcrEngine.try_create_from_language(lang)
                    break
            except Exception:
                continue
        if engine is None:
            engine = OcrEngine.try_create_from_user_profile_languages()
        if engine is None:
            raise RuntimeError(
                "Windows OCR has no language pack. In an admin PowerShell run: "
                'Add-WindowsCapability -Online -Name "Language.OCR~~~en-US~0.0.1.0"'
            )
        self.engine = engine
        self.max_dim = int(getattr(OcrEngine, "max_image_dimension", 10000) or 10000)

    def __call__(self, rgb: np.ndarray) -> str:
        import asyncio

        from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
        from winrt.windows.storage.streams import DataWriter

        step = int(np.ceil(max(rgb.shape[:2]) / self.max_dim))
        if step > 1:
            rgb = rgb[::step, ::step]
        h, w = rgb.shape[:2]
        rgba = np.dstack([rgb, np.full((h, w), 255, dtype=np.uint8)])
        writer = DataWriter()
        writer.write_bytes(np.ascontiguousarray(rgba).tobytes())
        bitmap = SoftwareBitmap.create_copy_from_buffer(writer.detach_buffer(), BitmapPixelFormat.RGBA8, w, h)

        async def run():
            return await self.engine.recognize_async(bitmap)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            result = asyncio.run(run())                 # ordinary worker thread
        else:
            # Called from a thread that already drives an event loop (an async API handler):
            # asyncio.run() would raise and the OCR would silently return nothing.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(lambda: asyncio.run(run())).result()
        return "\n".join(line.text for line in result.lines)


class _RapidOCR:
    """RapidOCR (ONNX models bundled in the pip package). Used on Linux/macOS or Python <= 3.12."""

    name = "rapidocr"

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR

        self.engine = RapidOCR()

    def __call__(self, rgb: np.ndarray) -> str:
        result, _ = self.engine(np.ascontiguousarray(rgb[:, :, ::-1]))  # expects BGR
        if not result:
            return ""
        items = []
        for box, text, _score in result:
            ys = [p[1] for p in box]
            xs = [p[0] for p in box]
            items.append((min(ys), max(ys), min(xs), text))
        items.sort(key=lambda t: (t[0], t[2]))
        lines: list[list[tuple]] = []
        for it in items:  # group boxes into rough lines, read left-to-right
            if lines and abs(it[0] - lines[-1][0][0]) < (it[1] - it[0]) * 0.6:
                lines[-1].append(it)
            else:
                lines.append([it])
        return "\n".join(" ".join(t[3] for t in sorted(line, key=lambda t: t[2])) for line in lines)


class OCR:
    """Picks the first OCR backend that loads. RAGLY_OCR = auto | windows | rapidocr | off."""

    _backend = None
    _error: str | None = None
    _loaded = False
    _lock = threading.Lock()

    @classmethod
    def _load(cls):
        with cls._lock:
            if cls._loaded:
                return cls._backend
            choice = os.environ.get("RAGLY_OCR", "auto").lower()
            if choice == "off":
                order = []
            elif choice == "windows":
                order = [_WindowsOCR]
            elif choice == "rapidocr":
                order = [_RapidOCR]
            else:
                order = [_WindowsOCR, _RapidOCR] if sys.platform == "win32" else [_RapidOCR, _WindowsOCR]
            errors = []
            for backend in order:
                try:
                    cls._backend = backend()
                    break
                except Exception as exc:
                    errors.append(f"{backend.name}: {exc}")
            if cls._backend is None:
                cls._error = "; ".join(errors) or "OCR disabled (RAGLY_OCR=off)"
                log.warning("No OCR backend available - scanned pages will be skipped. %s", cls._error)
            else:
                log.info("OCR backend: %s", cls._backend.name)
            cls._loaded = True
            return cls._backend

    @classmethod
    def status(cls) -> dict:
        backend = cls._load()
        return {"backend": backend.name if backend else None, "error": cls._error}

    @classmethod
    def image_to_text(cls, rgb: np.ndarray) -> str:
        backend = cls._load()
        if backend is None:
            raise RuntimeError(f"OCR unavailable ({cls._error})")
        return backend(rgb)


def _load_image(path: Path) -> np.ndarray:
    pix = fitz.Pixmap(str(path))
    if pix.alpha or pix.n not in (3, 4):
        pix = fitz.Pixmap(fitz.csRGB, pix)
    if pix.width < 1600:  # small images lose word spacing in OCR: upscale first
        scale = 1600 / pix.width
        pix = fitz.Pixmap(pix, int(pix.width * scale), int(pix.height * scale), None)
    return _pixmap_to_array(pix)


def _pixmap_to_array(pix: "fitz.Pixmap") -> np.ndarray:
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        arr = arr[:, :, :3]
    return arr.copy()  # RGB


def extract(path: Path) -> tuple[list[tuple[int, str]], int]:
    """Return ([(page_no, text), ...], ocr_page_count)."""
    ext = path.suffix.lower()
    if ext in PDF_EXT:
        return _extract_pdf(path)
    if ext in IMAGE_EXT:
        text = OCR.image_to_text(_load_image(path))
        return [(1, text)], 1
    if ext in TEXT_EXT:
        raw = path.read_text(encoding="utf-8", errors="replace")
        return _paginate(raw), 0
    if ext in DOCX_EXT:
        return _extract_docx(path), 0
    raise ValueError(f"Unsupported file type: {ext}")


def _paginate(raw: str) -> list[tuple[int, str]]:
    pages, buf, size = [], [], 0
    for para in raw.split("\n\n"):
        if buf and size + len(para) > TEXT_PAGE_CHARS:
            pages.append("\n\n".join(buf))
            buf, size = [], 0
        buf.append(para)
        size += len(para)
    if buf:
        pages.append("\n\n".join(buf))
    return [(i + 1, p) for i, p in enumerate(pages)]


def _extract_docx(path: Path) -> list[tuple[int, str]]:
    import zipfile
    import re
    from xml.etree import ElementTree as ET

    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    paras = []
    for p in root.iter(f"{{{ns['w']}}}p"):
        paras.append("".join(t.text or "" for t in p.iter(f"{{{ns['w']}}}t")))
    raw = "\n\n".join(x for x in paras if x.strip())
    return _paginate(re.sub(r"\n{3,}", "\n\n", raw))


def _extract_pdf(path: Path) -> tuple[list[tuple[int, str]], int]:
    pages: list[tuple[int, str]] = []
    ocr_pages = 0
    with fitz.open(path) as doc:
        if doc.needs_pass:
            raise ValueError("PDF is password protected")
        for i, page in enumerate(doc):
            text = page.get_text("text") or ""
            if len(text.strip()) < settings.ocr_min_chars:
                try:
                    pix = page.get_pixmap(dpi=settings.ocr_dpi)
                    ocr_text = OCR.image_to_text(_pixmap_to_array(pix))
                    if len(ocr_text.strip()) > len(text.strip()):
                        text = ocr_text
                        ocr_pages += 1
                except Exception as exc:  # OCR is best effort
                    log.warning("OCR failed on %s page %d: %s", path.name, i + 1, exc)
            pages.append((i + 1, text))
    return pages, ocr_pages


def page_count(path: Path) -> int:
    if path.suffix.lower() in PDF_EXT:
        with fitz.open(path) as doc:
            return doc.page_count
    return 1


def render_page_png(path: Path, page_no: int, highlight: str | None = None, dpi: int = 110) -> bytes:
    """Render a page for the source viewer, highlighting the cited passage when possible."""
    ext = path.suffix.lower()
    if ext in IMAGE_EXT:
        return path.read_bytes() if ext == ".png" else fitz.Pixmap(str(path)).tobytes("png")
    if ext not in PDF_EXT:
        raise ValueError("Page rendering is only available for PDFs and images")
    with fitz.open(path) as doc:
        if not 1 <= page_no <= doc.page_count:
            raise IndexError("page out of range")
        page = doc[page_no - 1]
        if highlight:
            words = highlight.split()
            # search a few short phrases from the chunk; full chunks rarely match verbatim
            for start in range(0, min(len(words), 400), 8):
                phrase = " ".join(words[start : start + 8])
                if len(phrase) < 12:
                    continue
                for quad in page.search_for(phrase, quads=True)[:3]:
                    annot = page.add_highlight_annot(quad)
                    annot.update()
        return page.get_pixmap(dpi=dpi).tobytes("png")
