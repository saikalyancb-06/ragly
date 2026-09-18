"""Local image index: extraction, perceptual hashing, optional CLIP embeddings, fused search.

Everything here is on-device. The only network access in the whole feature lives in
``scripts/download_models.py`` (setup time); at runtime the offline guard in
``ragly_backend.offline`` would block it anyway.

Three capability tiers, reported honestly by :meth:`ImageIndex.capabilities`:

* ``clip``      - the local CLIP ONNX pair is installed: real text -> image and
                  image -> image semantic search, fused with OCR keywords and pHash.
* ``ocr_phash`` - no vision model installed: text -> image still works through the OCR
                  text of each image, and image -> image works through perceptual
                  hashing (near-duplicate detection) plus OCR of the query image.

Nothing in this module raises because the CLIP model is absent.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pymupdf as fitz

from .config import MODELS_DIR, settings

log = logging.getLogger("ragly.images")

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PDF_EXT = {".pdf"}

# ---- CLIP (ViT-B/32) constants -------------------------------------------------------
CLIP_MODEL_ID = "clip-vit-base-patch32"
CLIP_DIM = 512
CLIP_INPUT = 224
CLIP_CONTEXT = 77
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
CLIP_EOT = 49407  # <|endoftext|>, also CLIP's padding token
VISUAL_FILE = "visual.onnx"
TEXTUAL_FILE = "textual.onnx"
TOKENIZER_FILE = "tokenizer.json"

PAGE_RENDER_DPI = 140
PHASH_BLOCK = 8          # low-frequency DCT block -> 64 bits
PHASH_GRID = 32          # image is reduced to 32x32 grey before the DCT
NEAR_DUPLICATE_MAX = 12  # Hamming distance still worth reporting
#: How well a text query must describe an image, relative to how well the image's own best
#: concept describes it. Measured on an indexed corpus: queries naming something really in the
#: picture scored 0.93-1.00 of that yardstick, queries naming absent objects (submarine, horse,
#: mountain, swimming pool) never exceeded 0.88. The gap is where this sits.
CONCEPT_RATIO = 0.90

# small, self-contained stop list so this module does not depend on the retriever
_STOPWORDS = set(
    """a an and are as at be by can do does for from has have how i in is it its me my of on or our
    please show tell than that the their them there these they this to was we were what when where which
    who whom why will with you your about into any all also give list find picture image photo photos
    images pictures show me like similar same looks look containing contains showing shown
    containing having include includes including something anything""".split()
)


def download_hint() -> str:
    """The exact command a user runs to install the local CLIP model."""
    return "python scripts/download_models.py --clip"


def clip_dir() -> Path:
    """Where the CLIP ONNX pair is expected (override with RAGLY_CLIP_DIR)."""
    env = os.environ.get("RAGLY_CLIP_DIR")
    return Path(env) if env else MODELS_DIR / CLIP_MODEL_ID


# ======================================================================================
# image loading / resizing (pymupdf + numpy only - no PIL, no torch)
# ======================================================================================
def _pixmap_to_array(pix: "fitz.Pixmap") -> np.ndarray:
    """RGB uint8 array from a pymupdf Pixmap (drops alpha, converts CMYK/grey)."""
    if pix.alpha or pix.n not in (1, 3):
        pix = fitz.Pixmap(fitz.csRGB, pix)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif pix.n == 4:
        arr = arr[:, :, :3]
    return np.ascontiguousarray(arr)


def load_rgb(path_or_array, min_width: int = 0) -> np.ndarray:
    """Load an image file (or pass an array through) as an RGB uint8 array.

    ``min_width`` upscales small images, which materially helps OCR word spacing.
    """
    if isinstance(path_or_array, np.ndarray):
        arr = path_or_array
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        arr = np.ascontiguousarray(arr[:, :, :3].astype(np.uint8))
    else:
        arr = _pixmap_to_array(fitz.Pixmap(str(path_or_array)))
    if min_width and arr.shape[1] < min_width:
        scale = min_width / arr.shape[1]
        arr = resize_rgb(arr, int(arr.shape[1] * scale), max(1, int(arr.shape[0] * scale)))
    return arr


def resize_rgb(arr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Resize an HxWxC uint8 array with numpy: box pre-reduction, then bilinear.

    The box step kills the aliasing that plain bilinear sampling produces on large
    downscales (which would otherwise make perceptual hashes unstable).
    """
    h, w = arr.shape[:2]
    if (h, w) == (out_h, out_w):
        return arr
    work = arr.astype(np.float32)
    fy, fx = max(1, h // max(1, out_h * 2)), max(1, w // max(1, out_w * 2))
    if fy > 1 or fx > 1:  # integer box average
        hh, ww = (h // fy) * fy, (w // fx) * fx
        work = work[:hh, :ww].reshape(hh // fy, fy, ww // fx, fx, work.shape[2]).mean(axis=(1, 3))
        h, w = work.shape[:2]
    # bilinear to the exact target size
    ys = (np.arange(out_h, dtype=np.float32) + 0.5) * (h / out_h) - 0.5
    xs = (np.arange(out_w, dtype=np.float32) + 0.5) * (w / out_w) - 0.5
    ys = np.clip(ys, 0, h - 1)
    xs = np.clip(xs, 0, w - 1)
    y0, x0 = np.floor(ys).astype(np.int64), np.floor(xs).astype(np.int64)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    wy, wx = (ys - y0)[:, None, None], (xs - x0)[None, :, None]
    top = work[y0][:, x0] * (1 - wx) + work[y0][:, x1] * wx
    bot = work[y1][:, x0] * (1 - wx) + work[y1][:, x1] * wx
    return np.clip(top * (1 - wy) + bot * wy, 0, 255).astype(np.uint8)


def to_grey(arr: np.ndarray) -> np.ndarray:
    """ITU-R 601 luminance as float32."""
    a = arr.astype(np.float32)
    if a.ndim == 2:
        return a
    return a[:, :, 0] * 0.299 + a[:, :, 1] * 0.587 + a[:, :, 2] * 0.114


# ======================================================================================
# perceptual hash (DCT-II via a numpy matrix multiply - scipy is not installed)
# ======================================================================================
_DCT_CACHE: dict[int, np.ndarray] = {}


def _dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II basis, so ``C @ X @ C.T`` is the 2-D DCT of X."""
    m = _DCT_CACHE.get(n)
    if m is None:
        k = np.arange(n, dtype=np.float64)[:, None]
        i = np.arange(n, dtype=np.float64)[None, :]
        m = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
        m *= math.sqrt(2.0 / n)
        m[0] *= math.sqrt(0.5)
        _DCT_CACHE[n] = m
    return m


def phash(img_array_or_path) -> str:
    """64-bit DCT perceptual hash as 16 hex characters.

    Grey 32x32 -> 2-D DCT -> top-left 8x8 low-frequency block -> one bit per
    coefficient against the median of the block (the DC term is excluded from the
    median so a single huge coefficient cannot bias the threshold).
    """
    grey = to_grey(load_rgb(img_array_or_path))
    small = to_grey(resize_rgb(np.repeat(grey[:, :, None], 3, axis=2).astype(np.uint8),
                               PHASH_GRID, PHASH_GRID))
    c = _dct_matrix(PHASH_GRID)
    block = (c @ small.astype(np.float64) @ c.T)[:PHASH_BLOCK, :PHASH_BLOCK]
    flat = block.flatten()
    median = float(np.median(flat[1:]))  # drop the DC coefficient
    bits = flat > median
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return f"{value:016x}"


def hamming(a_hex: str, b_hex: str) -> int:
    """Bit distance between two hex hashes; 64 (max) when either side is unusable."""
    try:
        return bin(int(a_hex, 16) ^ int(b_hex, 16)).count("1")
    except (TypeError, ValueError):
        return 64


# ======================================================================================
# extraction
# ======================================================================================
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _record(doc_id: int | None, page: int, path: Path, source: str, arr: np.ndarray, sha: str) -> dict:
    return {
        "doc_id": doc_id,
        "page": page,
        "path": str(path),
        "source": source,
        "width": int(arr.shape[1]),
        "height": int(arr.shape[0]),
        "sha256": sha,
        "phash": phash(arr),
    }


def extract_images(path: Path, out_dir: Path, doc_id: int | None,
                   min_pixels: int = 12000, render_pages: list[int] | None = None) -> list[dict]:
    """Pull every usable raster image out of ``path``.

    PDFs: embedded images per page (pymupdf ``page.get_images``), skipping anything
    smaller than ``min_pixels`` and anything whose bytes were already seen. Pages listed
    in ``render_pages`` that carry no embedded image are rendered whole at 140 dpi and
    recorded with ``source="page_render"`` - that is how a diagram drawn with vector
    strokes, or a scanned page, still ends up in the image index. The caller decides
    which pages those are (it has the page classification).

    Image files: registered in place, no copy - ``source="upload"`` when ``doc_id`` is
    None, otherwise ``"embedded"``.

    Returns a list of dicts ready for :meth:`Store.add_image`.
    """
    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext in PDF_EXT:
        return _extract_pdf_images(path, out_dir, doc_id, min_pixels, render_pages or [])
    if ext in IMAGE_EXT:
        try:
            arr = load_rgb(path)
        except Exception as exc:
            log.warning("cannot read image %s: %s", path, exc)
            return []
        if arr.shape[0] * arr.shape[1] < min_pixels:
            return []
        sha = _sha256(path.read_bytes())
        source = "upload" if doc_id is None else "embedded"
        return [_record(doc_id, 1, path, source, arr, sha)]
    return []


def _extract_pdf_images(path: Path, out_dir: Path, doc_id: int | None,
                        min_pixels: int, render_pages: list[int]) -> list[dict]:
    records: list[dict] = []
    seen: set[str] = set()
    wanted_renders: set[int] = {int(p) for p in render_pages}
    with fitz.open(path) as doc:
        if doc.needs_pass:
            raise ValueError("PDF is password protected")
        for index in range(doc.page_count):
            page_no = index + 1
            page = doc[index]
            found = 0
            skipped_small = False
            for info in page.get_images(full=True):
                xref = int(info[0])
                try:
                    pix = fitz.Pixmap(doc, xref)
                    arr = _pixmap_to_array(pix)
                except Exception as exc:  # broken or unsupported stream: skip it
                    log.debug("image xref %s on page %d unreadable: %s", xref, page_no, exc)
                    continue
                if arr.shape[0] * arr.shape[1] < min_pixels:
                    skipped_small = True
                    continue  # icon, rule, logo, decorative sliver
                png = _png_bytes(arr)
                sha = _sha256(png)
                if sha in seen:
                    found += 1  # the page does have artwork, we just stored it already
                    continue
                seen.add(sha)
                dest = out_dir / f"doc{doc_id if doc_id is not None else 'x'}_p{page_no}_{sha[:12]}.png"
                if not dest.exists():
                    dest.write_bytes(png)
                records.append(_record(doc_id, page_no, dest, "embedded", arr, sha))
                found += 1
            # A page can show artwork that `get_images` cannot hand back: vector drawings,
            # figures painted as shapes, or a photo the caller did not flag. If the page
            # paints anything substantial and nothing was extracted, render the page itself
            # so the visual is in the image index rather than lost.
            if found == 0 and page_no not in wanted_renders:
                try:
                    page_area = abs(page.rect.get_area()) or 1
                    blocks = page.get_text("dict").get("blocks", [])
                    big_picture = any(
                        b.get("type") == 1 and abs(fitz.Rect(b["bbox"]).get_area()) / page_area > 0.12
                        for b in blocks)
                    drawings = page.get_drawings()
                    drawn = sum(abs(d["rect"].get_area()) for d in drawings) if drawings else 0
                    # Only a page that is largely picture is worth rendering: a tiny logo or a
                    # rule must not turn every page of a text document into an "image".
                    if (big_picture or drawn / page_area > 0.25) and not skipped_small:
                        wanted_renders.add(page_no)
                except Exception as exc:
                    log.debug("could not inspect page %d of %s: %s", page_no, path.name, exc)
            if found == 0 and page_no in wanted_renders:
                try:
                    arr = _pixmap_to_array(page.get_pixmap(dpi=PAGE_RENDER_DPI))
                except Exception as exc:
                    log.warning("page render failed for %s page %d: %s", path.name, page_no, exc)
                    continue
                png = _png_bytes(arr)
                sha = _sha256(png)
                if sha in seen:
                    continue
                seen.add(sha)
                dest = out_dir / f"doc{doc_id if doc_id is not None else 'x'}_p{page_no}_render.png"
                dest.write_bytes(png)
                records.append(_record(doc_id, page_no, dest, "page_render", arr, sha))
    return records


def _png_bytes(arr: np.ndarray) -> bytes:
    h, w = arr.shape[:2]
    pix = fitz.Pixmap(fitz.csRGB, w, h, np.ascontiguousarray(arr[:, :, :3]).tobytes(), False)
    return pix.tobytes("png")


# ======================================================================================
# CLIP embedder - optional, degrades to "not installed" instead of raising
# ======================================================================================
QNN_OPTIONS = {"backend_path": "QnnHtp.dll", "htp_performance_mode": "burst"}


class ImageEmbedder:
    """CLIP-style image/text embedder backed by two local ONNX graphs.

    Expects ``<models>/clip-vit-base-patch32/{visual.onnx,textual.onnx,tokenizer.json}``
    (override the directory with ``RAGLY_CLIP_DIR``). Construction never touches disk
    and never raises: the model is loaded on first use, and when it is missing
    :meth:`available` returns False and :meth:`status` explains what to install.
    """

    def __init__(self, model_dir: Path | str | None = None, providers: list[str] | None = None):
        self.model_dir = Path(model_dir) if model_dir else clip_dir()
        self.providers = list(providers or ["CPUExecutionProvider"])
        self.dim = CLIP_DIM
        self._visual = None
        self._textual = None
        self._tokenizer = None
        self._provider: str | None = None
        self._error: str | None = None
        self._pad_id = CLIP_EOT
        self._loaded = False
        self._lock = threading.Lock()

    # ---- plumbing ----
    @property
    def model_id(self) -> str:
        """Stable identity stored alongside every vector.

        Deliberately not the directory name: ``image_vectors.model`` is filtered on
        this string, so moving or renaming the model directory must not orphan
        vectors that were already computed.
        """
        return CLIP_MODEL_ID

    @property
    def visual_path(self) -> Path:
        return self.model_dir / VISUAL_FILE

    @property
    def textual_path(self) -> Path:
        return self.model_dir / TEXTUAL_FILE

    @property
    def tokenizer_path(self) -> Path:
        return self.model_dir / TOKENIZER_FILE

    def _make_session(self, model_path: Path):
        available = set(ort.get_available_providers())
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        for prov in self.providers:
            if prov not in available:
                continue
            try:
                if prov == "QNNExecutionProvider":
                    sess = ort.InferenceSession(
                        str(model_path), opts, providers=[(prov, QNN_OPTIONS), "CPUExecutionProvider"])
                else:
                    sess = ort.InferenceSession(str(model_path), opts, providers=[prov])
                return sess
            except Exception as exc:  # provider present but failed to init -> try the next
                log.debug("provider %s unusable for %s: %s", prov, model_path.name, exc)
        return ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])

    def _load(self) -> bool:
        """Load the pair once. Returns True when usable; records an error otherwise."""
        with self._lock:
            if self._loaded:
                return self._visual is not None
            self._loaded = True
            missing = [p.name for p in (self.visual_path, self.textual_path, self.tokenizer_path)
                       if not p.exists()]
            if missing:
                self._error = (
                    f"CLIP model not installed: {', '.join(missing)} missing from {self.model_dir}. "
                    f"Install it with `{download_hint()}`. "
                    "Image search still works through OCR text and perceptual hashing."
                )
                return False
            try:
                from tokenizers import Tokenizer

                self._tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
                self._visual = self._make_session(self.visual_path)
                self._textual = self._make_session(self.textual_path)
                self._provider = self._visual.get_providers()[0]
                out_dim = self._visual.get_outputs()[0].shape[-1]
                if isinstance(out_dim, int) and out_dim > 0:
                    self.dim = int(out_dim)
                self._pad_id = self._tokenizer.token_to_id("<|endoftext|>") or CLIP_EOT
                log.info("CLIP ready: %s on %s (dim %d)", self.model_id, self._provider, self.dim)
                return True
            except Exception as exc:
                self._visual = self._textual = self._tokenizer = None
                self._error = f"CLIP model present but failed to load from {self.model_dir}: {exc}"
                log.warning("%s", self._error)
                return False

    # ---- capability reporting ----
    def available(self) -> bool:
        return self._load()

    def status(self) -> dict:
        ok = self._load()
        return {
            "model": self.model_id if ok else None,
            "model_dir": str(self.model_dir),
            "installed": ok,
            "dim": self.dim if ok else None,
            "provider": self._provider if ok else None,
            "available_providers": list(ort.get_available_providers()),
            "error": None if ok else self._error,
            "download_hint": None if ok else download_hint(),
        }

    def _unavailable(self) -> RuntimeError:
        return RuntimeError(self._error or f"CLIP model unavailable. Run `{download_hint()}`.")

    # ---- inference ----
    def preprocess_image(self, path_or_array) -> np.ndarray:
        """CLIP pixel values: shortest edge to 224, centre crop, mean/std, NCHW float32."""
        arr = load_rgb(path_or_array)
        h, w = arr.shape[:2]
        scale = CLIP_INPUT / max(1, min(h, w))
        nh = max(CLIP_INPUT, int(round(h * scale)))
        nw = max(CLIP_INPUT, int(round(w * scale)))
        arr = resize_rgb(arr, nw, nh)
        top, left = (nh - CLIP_INPUT) // 2, (nw - CLIP_INPUT) // 2
        arr = arr[top:top + CLIP_INPUT, left:left + CLIP_INPUT]
        x = arr.astype(np.float32) / 255.0
        x = (x - CLIP_MEAN) / CLIP_STD
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None, ...], dtype=np.float32)

    def tokenize(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """CLIP token ids padded to 77 with <|endoftext|>, plus the attention mask."""
        ids = list(self._tokenizer.encode(text or "").ids)[:CLIP_CONTEXT]
        if len(ids) == CLIP_CONTEXT:
            ids[-1] = self._pad_id  # keep the end-of-text marker the pooling looks for
        real = len(ids)
        ids = ids + [self._pad_id] * (CLIP_CONTEXT - real)
        mask = [1] * real + [0] * (CLIP_CONTEXT - real)
        return (np.asarray([ids], dtype=np.int64), np.asarray([mask], dtype=np.int64))

    @staticmethod
    def _normalise(vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        return (vec / max(float(np.linalg.norm(vec)), 1e-12)).astype(np.float32)

    def embed_image(self, path_or_array) -> np.ndarray:
        if not self._load():
            raise self._unavailable()
        feeds = {self._visual.get_inputs()[0].name: self.preprocess_image(path_or_array)}
        with self._lock:
            out = self._visual.run(None, feeds)[0]
        return self._normalise(out[0] if np.ndim(out) > 1 else out)

    def embed_text(self, text: str, ensemble: bool = True) -> np.ndarray:
        if not self._load():
            raise self._unavailable()
        raw = (text or "").strip()
        if not raw:
            return np.zeros(self.dim, dtype=np.float32)
        if not ensemble or len(raw.split()) > 8:
            return self._embed_single_text(raw)
        templates = [
            raw,
            f"a photo of a {raw}",
            f"a photo of {raw}",
            f"a picture of {raw}",
            f"an image showing {raw}",
        ]
        vecs = [self._embed_single_text(t) for t in templates]
        combined = np.sum(vecs, axis=0)
        return self._normalise(combined)

    def _embed_single_text(self, text: str) -> np.ndarray:
        ids, mask = self.tokenize(text)
        names = {i.name for i in self._textual.get_inputs()}
        feeds: dict[str, np.ndarray] = {"input_ids": ids}
        if "attention_mask" in names:
            feeds["attention_mask"] = mask
        feeds = {k: v for k, v in feeds.items() if k in names} or {
            self._textual.get_inputs()[0].name: ids}
        with self._lock:
            out = self._textual.run(None, feeds)[0]
        return self._normalise(out[0] if np.ndim(out) > 1 else out)



# ======================================================================================
# the index itself
# ======================================================================================
def fts_query(text: str, max_terms: int = 24) -> str:
    """Turn free text into a safe FTS5 MATCH expression (quoted OR-ed terms)."""
    words = [w for w in re.findall(r"\w+", (text or "").lower())
             if len(w) > 1 and w not in _STOPWORDS]
    seen: list[str] = []
    for w in words:
        if w not in seen:
            seen.append(w)
    return " OR ".join(f'"{w}"' for w in seen[:max_terms])


_VISUAL = "visual similarity"
_TEXT_IN_IMAGE = "text in image"
_BOTH = "both"
_TEXT_IN_DOC = "text in document"


class ImageIndex:
    """Adds images to the store and searches them by text, by image, or by document.

    Search always fuses every signal that is actually available with Reciprocal Rank
    Fusion, so the ranking degrades smoothly rather than disappearing when there is no
    vision model: OCR keywords cover text -> image, and perceptual hashing covers
    image -> image.
    """

    def __init__(self, store, image_embedder: ImageEmbedder | None = None, ocr_callable=None):
        from .concepts import ConceptTagger

        self.tagger = ConceptTagger(image_embedder)
        self.store = store
        self.embedder = image_embedder
        self.ocr = ocr_callable

    # ---- capabilities ----
    def clip_ready(self) -> bool:
        try:
            return bool(self.embedder) and self.embedder.available()
        except Exception:  # a broken model must never take the index down
            return False

    def mode(self) -> str:
        return "clip" if self.clip_ready() else "ocr_phash"

    def capabilities(self) -> dict:
        ready = self.clip_ready()
        st = self.embedder.status() if self.embedder else {}
        if ready:
            note = (f"{st.get('model')} runs locally on {st.get('provider')}: text and image "
                    "queries are matched semantically, then fused with OCR text and perceptual hashing.")
        else:
            note = ("No vision model installed, so there is no semantic image matching: image "
                    "search uses OCR text inside the images and perceptual hashing for "
                    f"near-duplicates. Install CLIP with `{download_hint()}`.")
        return {
            "vision_model": st.get("model") if ready else None,
            "mode": "clip" if ready else "ocr_phash",
            "provider": st.get("provider") if ready else None,
            "note": note,
            "images": self.store.image_counts() if hasattr(self.store, "image_counts") else {},
            "error": None if ready else st.get("error"),
            "download_hint": None if ready else download_hint(),
        }

    # ---- OCR ----
    def _ocr_text(self, path_or_array) -> str:
        """OCR one image, tolerating every possible failure (it is a bonus signal)."""
        if not self.ocr:
            return ""
        try:
            arr = load_rgb(path_or_array, min_width=1600)
        except Exception as exc:
            log.debug("cannot load %s for OCR: %s", path_or_array, exc)
            return ""
        try:
            return (self.ocr(arr) or "").strip()
        except Exception as exc:
            log.debug("OCR failed on %s: %s", path_or_array, exc)
            return ""

    # ---- indexing ----
    def index_document_images(self, doc_id: int | None, records: list[dict]) -> dict:
        """Store ``records`` (from :func:`extract_images`), OCR them, embed when possible."""
        use_clip = self.clip_ready()
        added = ocr_hits = embedded = skipped = described = 0
        image_ids: list[int] = []
        for rec in records:
            rec = dict(rec)
            rec.setdefault("doc_id", doc_id)
            if not rec.get("phash"):
                try:
                    rec["phash"] = phash(rec["path"])
                except Exception:
                    rec["phash"] = ""
            text = self._ocr_text(rec["path"])
            if text:
                rec["ocr_text"] = text
                rec.setdefault("caption_source", "ocr")
                ocr_hits += 1
            try:
                image_id = self.store.add_image(rec)
            except Exception as exc:
                log.warning("could not store image %s: %s", rec.get("path"), exc)
                skipped += 1
                continue
            if image_id is None:
                skipped += 1
                continue
            added += 1
            image_ids.append(image_id)
            log.info("IMAGE FOUND document=%s page=%s image_id=%s %sx%s",
                     doc_id, rec.get("page"), image_id, rec.get("width"), rec.get("height"))
            if use_clip:
                try:
                    vec = self.embedder.embed_image(rec["path"])
                    self.store.set_image_vector(image_id, vec, self.embedder.model_id)
                    embedded += 1
                    log.info("VISUAL EMBEDDING CREATED image_id=%s dim=%d", image_id, len(vec))
                    # Understanding happens once, here: the concepts and description are stored,
                    # so searching never runs a vision model over the corpus.
                    objects, description, best = self.tagger.tag(vec)
                    if objects:
                        self.store.set_image_understanding(image_id, objects, description,
                                                           self.embedder.model_id, best)
                        described += 1
                        log.info("IMAGE METADATA CREATED image_id=%s objects=%s", image_id, objects)
                except Exception as exc:  # a bad image must not fail the whole document
                    log.warning("CLIP embedding failed for %s: %s", rec.get("path"), exc)
        return {
            "doc_id": doc_id,
            "images": added,
            "skipped": skipped,
            "with_ocr_text": ocr_hits,
            "embedded": embedded,
            "described": described,
            "image_ids": image_ids,
            "mode": "clip" if use_clip else "ocr_phash",
        }

    # ---- shared ranking helpers ----
    def _vector_ranks(self, vec: np.ndarray, limit: int) -> list[tuple[int, float]]:
        """Image ids ranked by cosine similarity against the stored image vectors."""
        model = self.embedder.model_id if self.embedder else None
        try:
            mat, ids = self.store.image_matrix(model)
        except Exception as exc:
            log.warning("image matrix unavailable: %s", exc)
            return []
        if mat is None or len(ids) == 0 or mat.shape[1] != vec.shape[0]:
            return []
        sims = mat @ vec
        k = min(limit, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(int(ids[i]), float(sims[i])) for i in idx]

    def _text_ranks(self, text: str, limit: int) -> list[int]:
        match = fts_query(text)
        if not match:
            return []
        try:
            return self.store.image_text_search(match, limit)
        except Exception as exc:
            log.debug("image FTS failed for %r: %s", match, exc)
            return []

    @staticmethod
    def _rrf(fused: dict[int, float], ranked, weight: float = 1.0) -> None:
        k = settings.rrf_k
        for rank, item in enumerate(ranked):
            image_id = item[0] if isinstance(item, tuple) else item
            fused[image_id] = fused.get(image_id, 0.0) + weight / (k + rank + 1)

    def _rows(self, image_ids: list[int]) -> dict[int, dict]:
        if not image_ids:
            return {}
        try:
            rows = self.store.get_images(ids=image_ids, limit=max(len(image_ids), 1))
        except Exception as exc:
            log.warning("could not read image rows: %s", exc)
            return {}
        return {int(r["id"]): r for r in rows}

    def _hits(self, fused: dict[int, float], limit: int, reason_of, extra_of=None) -> list[dict]:
        order = sorted(fused, key=lambda i: -fused[i])[:limit]
        rows = self._rows(order)
        out = []
        for image_id in order:
            row = rows.get(image_id)
            if row is None:
                continue
            hit = {
                "image_id": image_id,
                "doc_id": row.get("doc_id"),
                "doc_name": row.get("doc_name") or "",
                "page": row.get("page") or 1,
                "path": row.get("path"),
                "score": round(float(fused[image_id]), 6),
                "match_reason": reason_of(image_id),
            }
            if extra_of:
                hit.update(extra_of(image_id, row))
            out.append(hit)
        return out

    # ---- text -> image ----
    def debug(self, query_type: str, query_vec=None, candidates: int = 0, limit: int = 0,
              hits: list[dict] | None = None) -> dict:
        """What actually happened in a search: model, dimension, pool size, scores.

        Every number here is read back from the running index, never assumed."""
        caps = self.capabilities()
        vec_len = int(np.asarray(query_vec).size) if query_vec is not None else None
        norm = float(np.linalg.norm(np.asarray(query_vec, dtype=np.float32))) if query_vec is not None else None
        return {
            "query_type": query_type,
            "embedding_model": caps.get("vision_model"),
            "embedding_provider": caps.get("provider"),
            "embedding_dimension": vec_len,
            "query_embedding_norm": round(norm, 4) if norm is not None else None,
            "indexed_images": caps.get("images", {}).get("images", 0),
            "images_with_vectors": caps.get("images", {}).get("with_vectors", 0),
            "candidate_count": candidates,
            "top_k": limit,
            "similarity_scores": [
                {"image_id": h["image_id"], "document": h.get("doc_name"), "page": h.get("page"),
                 "similarity": h.get("similarity"), "phash_distance": h.get("phash_distance"),
                 "why": h.get("match_reason")}
                for h in (hits or [])
            ],
            "mode": self.mode(),
        }

    def understand_pending(self, limit: int = 200) -> int:
        """Tag images that have a vector but no concepts yet.

        A workspace indexed before the vision model was installed, or before this feature
        existed, would otherwise stay blind until the reader re-indexed everything. One dot
        product per image is cheap, so it is done once, here.
        """
        if not self.clip_ready():
            return 0
        done = 0
        try:
            rows = self.store.get_images(limit=limit)
        except Exception as exc:
            log.debug("cannot list images for tagging: %s", exc)
            return 0
        for row in rows:
            has_objects = (row.get("objects") or "[]") != "[]"
            has_yardstick = float(row.get("concept_score") or 0) > 0
            if has_objects and has_yardstick:
                continue
            try:
                vec = self.embedder.embed_image(row["path"])
                objects, description, best = self.tagger.tag(vec)
                if objects:
                    self.store.set_image_understanding(row["id"], objects, description,
                                                       self.embedder.model_id, best)
                    done += 1
            except Exception as exc:
                log.debug("could not tag image %s: %s", row.get("id"), exc)
        if done:
            log.info("understood %d image(s) that had no concepts yet", done)
        return done

    def search_by_text(self, query: str, limit: int = 12) -> list[dict]:
        """Find images from a text query: CLIP when installed, OCR keywords always."""
        from .concepts import clean_image_query, expand_query as _expand

        clean_q = clean_image_query(query)
        pool = max(limit * 3, 24)
        vector_hits: list[tuple[int, float]] = []
        self.last_query_vector = None
        if self.clip_ready():
            try:
                self.last_query_vector = self.embedder.embed_text(clean_q or query, ensemble=True)
                vector_hits = self._vector_ranks(self.last_query_vector, pool)
            except Exception as exc:
                log.warning("CLIP text embedding failed: %s", exc)
        text_hits = self._text_ranks(clean_q or query, pool)
        concept_hits, matched_concepts = self._concept_ranks(query, pool)
        if not concept_hits and clean_q != query:
            concept_hits, matched_concepts = self._concept_ranks(clean_q, pool)

        fused: dict[int, float] = {}
        self._rrf(fused, vector_hits, weight=1.5)
        self._rrf(fused, text_hits, weight=1.0)
        self._rrf(fused, concept_hits, weight=1.3)   # what the picture contains is strong evidence
        v_ids = {i for i, _ in vector_hits}
        t_ids = set(text_hits)
        cosine = {i: s for i, s in vector_hits}

        # Filtering absent concepts and background noise:
        # A comparison inside embedding space ensures random background noise is not returned
        # for non-existent objects, while genuine visual matches remain discoverable.
        if vector_hits and not concept_hits and not text_hits:
            rows = self._rows([i for i, _ in vector_hits])
            kept: list[tuple[int, float]] = []
            for image_id, score in vector_hits:
                yardstick = float((rows.get(image_id) or {}).get("concept_score") or 0.0)
                # If yardstick exists, check score relative to yardstick or absolute visual cutoff
                if yardstick > 0 and score < 0.78 * yardstick:
                    continue
                if score < 0.18:
                    continue
                kept.append((image_id, score))
            if not kept:
                log.info("no image matches %r as well as its own description", query)
                return []
            vector_hits = kept
            v_ids = {i for i, _ in vector_hits}
            fused = {i: s for i, s in fused.items() if i in v_ids}

        asked_for = _expand(query)
        if asked_for and not concept_hits and not text_hits:
            # If the user explicitly asked for specific concepts (e.g. "car", "dog"),
            # ensure top visual score is high enough, otherwise treat as absent from corpus
            top_score = max((cosine.get(i, 0.0) for i in v_ids), default=0.0)
            if top_score < 0.22:
                log.info("no image contains %s (%d indexed images checked, top score=%.3f)",
                         ", ".join(asked_for[:3]), len(vector_hits), top_score)
                return []

        def reason(image_id: int) -> str:
            found = matched_concepts.get(image_id)
            if found:
                return "contains " + ", ".join(found[:3])
            if image_id in v_ids and image_id in t_ids:
                return _BOTH
            if image_id in v_ids:
                sim = cosine.get(image_id)
                if sim is not None:
                    return f"visual similarity ({round(sim * 100)}%)"
                return _VISUAL
            return _TEXT_IN_IMAGE

        def extra(image_id: int, row: dict) -> dict:
            import json as _json

            try:
                objects = _json.loads(row.get("objects") or "[]")
            except ValueError:
                objects = []
            return {
                "similarity": round(cosine[image_id], 4) if image_id in cosine else None,
                "ocr_text": (row.get("ocr_text") or "")[:400],
                "objects": objects,
                "description": row.get("caption") or "",
                "matched_concepts": matched_concepts.get(image_id, []),
                "matched_visually": image_id in v_ids,
                "matched_text": image_id in t_ids,
                "mode": self.mode(),
            }

        return self._hits(fused, limit, reason, extra)

    def _concept_ranks(self, query: str, limit: int) -> tuple[list[tuple[int, int]], dict[int, list[str]]]:
        """Images whose stored concepts answer the query, ranked by how many matched.

        The concepts were found by the vision model at indexing time; the query is widened with
        plain synonyms ("vehicle" also asks about car, truck, van) so a search does not depend on
        the searcher guessing the same noun the model used.
        """
        import json as _json

        from .concepts import expand_query

        wanted = expand_query(query)
        if not wanted:
            return [], {}
        scored: list[tuple[int, int, int]] = []
        matched: dict[int, list[str]] = {}
        try:
            rows = self.store.get_images(limit=500)
        except Exception as exc:
            log.debug("concept lookup failed: %s", exc)
            return [], {}
        for row in rows:
            try:
                objects = _json.loads(row.get("objects") or "[]")
            except ValueError:
                objects = []
            if not objects:
                continue
            hits = [o for o in objects if any(w == o or w in o or o in w for w in wanted)]
            if hits:
                matched[row["id"]] = hits
                scored.append((len(hits), row["id"], objects.index(hits[0])))
        scored.sort(key=lambda t: (-t[0], t[2]))
        return [(image_id, rank + 1) for rank, (_, image_id, _) in enumerate(scored[:limit])], matched

    # ---- image -> image ----
    def search_by_image(self, path: Path, limit: int = 12) -> list[dict]:
        """Find images similar to ``path``: CLIP cosine, pHash distance, and OCR text."""
        pool = max(limit * 3, 24)
        vector_hits: list[tuple[int, float]] = []
        self.last_query_vector = None
        if self.clip_ready():
            try:
                self.last_query_vector = self.embedder.embed_image(path)
                vector_hits = self._vector_ranks(self.last_query_vector, pool)
            except Exception as exc:
                log.warning("CLIP image embedding failed: %s", exc)

        phash_hits: list[tuple[int, int]] = []
        try:
            query_hash = phash(path)
        except Exception as exc:
            log.warning("could not hash %s: %s", path, exc)
            query_hash = ""
        if query_hash:
            try:
                phash_hits = self.store.images_by_phash(query_hash, NEAR_DUPLICATE_MAX, pool)
            except Exception as exc:
                log.debug("phash lookup failed: %s", exc)

        query_text = self._ocr_text(path)
        text_hits = self._text_ranks(query_text, pool) if query_text else []

        if query_hash and not vector_hits and not phash_hits and not text_hits:
            # No vision model, nothing near-duplicate and no shared text: returning "no match"
            # hides the index. Rank every indexed image by perceptual distance instead, and show
            # that distance, so the reader judges rather than a threshold deciding silently.
            try:
                phash_hits = self.store.images_by_phash(query_hash, 64, pool)
            except Exception as exc:
                log.debug("wide phash lookup failed: %s", exc)

        fused: dict[int, float] = {}
        self._rrf(fused, vector_hits)
        self._rrf(fused, phash_hits)
        self._rrf(fused, text_hits)
        cosine = {i: s for i, s in vector_hits}
        distance = {i: d for i, d in phash_hits}
        t_ids = set(text_hits)

        def reason(image_id: int) -> str:
            if image_id in distance:
                return f"near-duplicate (phash d={distance[image_id]})"
            if image_id in cosine:
                return _VISUAL
            return _TEXT_IN_IMAGE

        def extra(image_id: int, row: dict) -> dict:
            reasons = []
            if image_id in distance:
                reasons.append(f"near-duplicate (phash d={distance[image_id]})")
            if image_id in cosine:
                reasons.append(_VISUAL)
            if image_id in t_ids:
                reasons.append(_TEXT_IN_IMAGE)
            return {
                "similarity": round(cosine[image_id], 4) if image_id in cosine else None,
                "phash_distance": distance.get(image_id),
                "reasons": reasons,
                "query_ocr_text": query_text[:400],
                "mode": self.mode(),
            }

        return self._hits(fused, limit, reason, extra)

    # ---- image -> documents ----
    def search_documents_by_image(self, path: Path, limit: int = 8) -> list[dict]:
        """Which documents does this image come from / relate to."""
        hits = self.search_by_image(path, limit=max(limit * 4, 24))
        docs: dict[int, dict] = {}

        def bucket(doc_id, doc_name) -> dict:
            return docs.setdefault(int(doc_id), {
                "doc_id": int(doc_id), "doc_name": doc_name or "",
                "score": 0.0, "pages": [], "reasons": [],
            })

        for hit in hits:
            if hit.get("doc_id") is None:
                continue  # an uploaded image that belongs to no document
            entry = bucket(hit["doc_id"], hit.get("doc_name"))
            entry["score"] += float(hit["score"])
            if hit.get("page") and hit["page"] not in entry["pages"]:
                entry["pages"].append(int(hit["page"]))
            for r in (hit.get("reasons") or [hit["match_reason"]]):
                if r not in entry["reasons"]:
                    entry["reasons"].append(r)

        # the words inside the uploaded image also point at documents whose *text* says them
        query_text = self._ocr_text(path)
        match = fts_query(query_text)
        if match and hasattr(self.store, "fts_search"):
            try:
                chunk_ids = self.store.fts_search(match, max(limit * 3, 15))
                rows = self.store.get_chunks(chunk_ids)
            except Exception as exc:
                log.debug("document text search failed: %s", exc)
                chunk_ids, rows = [], {}
            for rank, chunk_id in enumerate(chunk_ids):
                row = rows.get(chunk_id)
                if not row:
                    continue
                entry = bucket(row["doc_id"], row.get("doc_name"))
                entry["score"] += 1.0 / (settings.rrf_k + rank + 1)
                if row.get("page") and row["page"] not in entry["pages"]:
                    entry["pages"].append(int(row["page"]))
                if _TEXT_IN_DOC not in entry["reasons"]:
                    entry["reasons"].append(_TEXT_IN_DOC)

        out = sorted(docs.values(), key=lambda d: -d["score"])[:limit]
        for entry in out:
            entry["pages"].sort()
            entry["score"] = round(entry["score"], 6)
        return out
