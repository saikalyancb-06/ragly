"""Image index tests. Every one of these runs with NO vision model installed.

That is the point: extraction, perceptual hashing, OCR-text search and near-duplicate
detection must all work on a machine that never downloaded CLIP, and the capability
report must say so plainly instead of pretending.
"""
from __future__ import annotations

import numpy as np
import pymupdf as fitz
import pytest

from ragly_backend import images as im
from ragly_backend.store import Store

# ---------------------------------------------------------------------------
# synthetic images. Structured content on purpose: a perceptual hash is only
# meaningful when the picture has mid-frequency detail (a flat gradient has
# almost no DCT energy above the first coefficients, so its hash is noise).
# ---------------------------------------------------------------------------
W, H = 240, 200


def diagram() -> np.ndarray:
    """Two labelled boxes joined by a connector - a stand-in for a real diagram."""
    img = np.full((H, W, 3), 255, np.uint8)
    img[30:60, 20:100] = (30, 60, 200)
    img[130:170, 140:220] = (200, 60, 30)
    img[44:47, 100:140] = (0, 0, 0)
    img[60:132, 138:141] = (0, 0, 0)
    img[90:120, 30:90] = (20, 160, 60)
    return img


def blocks(seed: int = 1) -> np.ndarray:
    """A different picture: scattered coloured rectangles."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 240, np.uint8)
    for _ in range(8):
        x0, y0 = int(rng.integers(0, W - 60)), int(rng.integers(0, H - 50))
        w, h = int(rng.integers(30, 60)), int(rng.integers(25, 50))
        img[y0:y0 + h, x0:x0 + w] = rng.integers(0, 200, 3).astype(np.uint8)
    return img


def slightly_modified(arr: np.ndarray) -> np.ndarray:
    """Same picture after a re-save: mild noise, a brightness shift, a corner stamp."""
    rng = np.random.default_rng(7)
    out = np.clip(arr.astype(np.int16) + 10, 0, 255)
    out = np.clip(out + rng.integers(-5, 6, out.shape), 0, 255).astype(np.uint8)
    out[0:12, 0:12] = 0
    return out


@pytest.fixture
def pdf_with_images(tmp_path):
    """A 3-page PDF: page 1 a diagram, page 2 a second picture + a repeat + a tiny icon,
    page 3 text only (nothing raster to extract)."""
    art_a = im._png_bytes(diagram())
    art_b = im._png_bytes(blocks())
    icon = im._png_bytes(np.full((20, 20, 3), 33, np.uint8))  # 400 px: decorative

    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), "Claim photo of the damaged laptop")
    p1.insert_image(fitz.Rect(72, 100, 312, 300), stream=art_a)

    p2 = doc.new_page()
    p2.insert_text((72, 72), "Second exhibit")
    p2.insert_image(fitz.Rect(72, 100, 312, 300), stream=art_b)
    p2.insert_image(fitz.Rect(320, 100, 560, 300), stream=art_a)   # duplicate of page 1
    p2.insert_image(fitz.Rect(72, 320, 92, 340), stream=icon)      # below min_pixels

    p3 = doc.new_page()
    p3.insert_text((72, 72), "Page three is text only, drawn with vector strokes")

    path = tmp_path / "claim.pdf"
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def absent_model_dir(tmp_path):
    """A directory where the CLIP model definitively is not."""
    return tmp_path / "no-such-model"


@pytest.fixture
def embedder(absent_model_dir):
    return im.ImageEmbedder(model_dir=absent_model_dir)


@pytest.fixture
def indexed(tmp_path, pdf_with_images, embedder):
    """Store + ImageIndex with one ready document whose images are indexed.

    The OCR callable is a stub, so these tests do not depend on a real OCR backend.
    """
    store = Store(tmp_path / "images.db")
    doc_id = store.add_document("claim.pdf", str(pdf_with_images), "sha-claim-1",
                                pdf_with_images.stat().st_size, 3)
    store.mark_ready(doc_id, pages=3)
    records = im.extract_images(pdf_with_images, tmp_path / "extracted", doc_id=doc_id)
    index = im.ImageIndex(store, embedder, lambda arr: "damaged laptop serial ABC123")
    summary = index.index_document_images(doc_id, records)
    return {"store": store, "index": index, "doc_id": doc_id,
            "records": records, "summary": summary}


# ===========================================================================
# perceptual hash
# ===========================================================================
def test_phash_is_16_hex_chars():
    value = im.phash(diagram())
    assert len(value) == 16
    int(value, 16)  # parses as hex


def test_phash_is_stable_for_the_same_image():
    """Same pixels in, same hash out - every time, and from a file too."""
    arr = diagram()
    assert im.phash(arr) == im.phash(arr) == im.phash(arr.copy())
    assert im.hamming(im.phash(arr), im.phash(arr)) == 0


def test_phash_stable_when_loaded_from_disk(tmp_path):
    arr = diagram()
    path = tmp_path / "d.png"
    path.write_bytes(im._png_bytes(arr))
    assert im.phash(path) == im.phash(arr)


def test_phash_small_distance_for_slightly_modified_image():
    base = im.phash(diagram())
    assert im.hamming(im.phash(slightly_modified(diagram())), base) <= 8


def test_phash_small_distance_after_downscale():
    """A thumbnail of the same picture is still recognisably the same picture."""
    arr = diagram()
    half = im.resize_rgb(arr, W // 2, H // 2)
    assert im.hamming(im.phash(half), im.phash(arr)) <= 8


def test_phash_large_distance_for_different_images():
    assert im.hamming(im.phash(diagram()), im.phash(blocks())) >= 16
    assert im.hamming(im.phash(blocks(1)), im.phash(blocks(9))) >= 16


def test_hamming_handles_rubbish_without_raising():
    assert im.hamming("", "abc") == 64
    assert im.hamming(None, "abc") == 64
    assert im.hamming("ffffffffffffffff", "0000000000000000") == 64
    assert im.hamming("0000000000000001", "0000000000000000") == 1


# ===========================================================================
# extraction
# ===========================================================================
def test_extract_images_from_pdf(tmp_path, pdf_with_images):
    """Two distinct pictures survive; the repeat and the tiny icon do not."""
    records = im.extract_images(pdf_with_images, tmp_path / "out", doc_id=42)
    assert len(records) == 2
    assert [r["page"] for r in records] == [1, 2]
    for rec in records:
        assert set(rec) == {"doc_id", "page", "path", "source", "width", "height",
                            "sha256", "phash"}
        assert rec["doc_id"] == 42
        assert rec["source"] == "embedded"
        assert (rec["width"], rec["height"]) == (W, H)
        assert len(rec["sha256"]) == 64
        assert len(rec["phash"]) == 16
        from pathlib import Path
        saved = Path(rec["path"])
        assert saved.exists() and saved.suffix == ".png"
        assert saved.parent == tmp_path / "out"


def test_extract_images_skips_duplicates_by_sha256(tmp_path, pdf_with_images):
    """The diagram appears on pages 1 and 2; it is stored once."""
    records = im.extract_images(pdf_with_images, tmp_path / "out", doc_id=1)
    hashes = [r["sha256"] for r in records]
    assert len(hashes) == len(set(hashes))
    page_one = next(r for r in records if r["page"] == 1)
    assert page_one["phash"] == im.phash(diagram())
    assert not any(r["page"] == 2 and r["sha256"] == page_one["sha256"] for r in records)


def test_extract_images_skips_tiny_images(tmp_path, pdf_with_images):
    """min_pixels is a real filter, not decoration."""
    assert len(im.extract_images(pdf_with_images, tmp_path / "a", doc_id=1)) == 2
    # a huge threshold rejects everything, a tiny one lets the 20x20 icon through
    assert im.extract_images(pdf_with_images, tmp_path / "b", doc_id=1, min_pixels=10 ** 7) == []
    loose = im.extract_images(pdf_with_images, tmp_path / "c", doc_id=1, min_pixels=100)
    assert len(loose) == 3
    assert any(r["width"] == 20 for r in loose)


def test_extract_images_renders_requested_pages_without_artwork(tmp_path, pdf_with_images):
    """render_pages only fires where the page has no embedded image of its own."""
    records = im.extract_images(pdf_with_images, tmp_path / "out", doc_id=1,
                                render_pages=[1, 3])
    by_source = {(r["page"], r["source"]) for r in records}
    assert (3, "page_render") in by_source        # text-only page was rendered
    assert (1, "embedded") in by_source           # page 1 kept its embedded image
    assert (1, "page_render") not in by_source    # and was NOT rendered on top
    rendered = next(r for r in records if r["source"] == "page_render")
    assert rendered["width"] > W and rendered["height"] > H   # 140 dpi full page


def test_extract_images_without_render_pages_ignores_empty_page(tmp_path, pdf_with_images):
    records = im.extract_images(pdf_with_images, tmp_path / "out", doc_id=1)
    assert all(r["source"] == "embedded" for r in records)
    assert 3 not in {r["page"] for r in records}


def test_extract_images_registers_an_image_file(tmp_path):
    """A plain image file is registered in place; doc_id decides the source label."""
    src = tmp_path / "photo.png"
    src.write_bytes(im._png_bytes(blocks()))

    upload = im.extract_images(src, tmp_path / "out", doc_id=None)
    assert len(upload) == 1
    assert upload[0]["source"] == "upload"
    assert upload[0]["doc_id"] is None
    assert upload[0]["path"] == str(src)          # registered, not copied
    assert (upload[0]["width"], upload[0]["height"]) == (W, H)

    attached = im.extract_images(src, tmp_path / "out", doc_id=5)
    assert attached[0]["source"] == "embedded"
    assert attached[0]["doc_id"] == 5
    assert attached[0]["sha256"] == upload[0]["sha256"]


def test_extract_images_ignores_unsupported_files(tmp_path):
    other = tmp_path / "notes.txt"
    other.write_text("not an image")
    assert im.extract_images(other, tmp_path / "out", doc_id=1) == []


# ===========================================================================
# the embedder with no model installed
# ===========================================================================
def test_embedder_constructs_without_a_model(absent_model_dir):
    """Constructing must never touch the network or explode."""
    assert im.ImageEmbedder(model_dir=absent_model_dir).model_id == im.CLIP_MODEL_ID


def test_embedder_is_unavailable_without_a_model(embedder):
    assert embedder.available() is False
    assert embedder.available() is False   # repeat: the failure is cached, not retried


def test_embedder_status_explains_what_is_missing(embedder, absent_model_dir):
    status = embedder.status()
    assert status["installed"] is False
    assert status["model"] is None
    assert status["provider"] is None
    assert status["dim"] is None
    error = status["error"]
    assert error, "status() must explain why the model is unavailable"
    assert "visual.onnx" in error and "textual.onnx" in error and "tokenizer.json" in error
    assert str(absent_model_dir) in error
    assert im.download_hint() in error
    assert status["download_hint"] == im.download_hint()
    assert "CPUExecutionProvider" in status["available_providers"]


def test_download_hint_is_the_exact_command():
    assert im.download_hint() == "python scripts/download_models.py --clip"


def test_embedding_without_a_model_raises_a_helpful_error(embedder, tmp_path):
    src = tmp_path / "p.png"
    src.write_bytes(im._png_bytes(diagram()))
    for call in (lambda: embedder.embed_image(src), lambda: embedder.embed_text("a laptop")):
        with pytest.raises(RuntimeError) as excinfo:
            call()
        assert im.download_hint() in str(excinfo.value)


def test_clip_dir_honours_the_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("RAGLY_CLIP_DIR", str(tmp_path / "elsewhere"))
    assert im.clip_dir() == tmp_path / "elsewhere"
    monkeypatch.delenv("RAGLY_CLIP_DIR")
    assert im.clip_dir().name == im.CLIP_MODEL_ID


# ===========================================================================
# the index, in ocr_phash mode
# ===========================================================================
def test_index_document_images_reports_the_degraded_mode(indexed):
    summary = indexed["summary"]
    assert summary["mode"] == "ocr_phash"
    assert summary["images"] == 2
    assert summary["with_ocr_text"] == 2
    assert summary["embedded"] == 0          # no vision model, so no vectors
    assert len(summary["image_ids"]) == 2
    counts = indexed["store"].image_counts()
    assert counts == {"images": 2, "with_vectors": 0}


def test_index_tolerates_a_failing_ocr_callable(tmp_path, pdf_with_images, embedder):
    """A broken OCR backend degrades the index, it does not break it."""
    store = Store(tmp_path / "broken-ocr.db")
    doc_id = store.add_document("c.pdf", str(pdf_with_images), "sha-broken", 10, 3)
    store.mark_ready(doc_id, pages=3)

    def exploding_ocr(arr):
        raise RuntimeError("no OCR backend on this machine")

    records = im.extract_images(pdf_with_images, tmp_path / "x", doc_id=doc_id)
    index = im.ImageIndex(store, embedder, exploding_ocr)
    summary = index.index_document_images(doc_id, records)
    assert summary["images"] == 2
    assert summary["with_ocr_text"] == 0
    assert store.image_counts()["images"] == 2


def test_index_works_with_no_ocr_callable_at_all(tmp_path, pdf_with_images, embedder):
    store = Store(tmp_path / "no-ocr.db")
    doc_id = store.add_document("c.pdf", str(pdf_with_images), "sha-noocr", 10, 3)
    store.mark_ready(doc_id, pages=3)
    records = im.extract_images(pdf_with_images, tmp_path / "x", doc_id=doc_id)
    index = im.ImageIndex(store, embedder, None)
    assert index.index_document_images(doc_id, records)["images"] == 2
    assert index.search_by_text("laptop") == []      # nothing to match on: honest empty


def test_capabilities_are_reported_honestly(indexed):
    caps = indexed["index"].capabilities()
    assert caps["vision_model"] is None
    assert caps["mode"] == "ocr_phash"
    assert caps["provider"] is None
    assert isinstance(caps["note"], str) and caps["note"]
    assert "OCR" in caps["note"] and im.download_hint() in caps["note"]
    assert indexed["index"].mode() == "ocr_phash"


def test_search_by_text_finds_images_through_their_ocr_text(indexed):
    hits = indexed["index"].search_by_text("damaged laptop", limit=5)
    assert hits, "OCR keyword search must work without a vision model"
    for hit in hits:
        assert set(hit) >= {"image_id", "doc_id", "doc_name", "page", "path",
                            "score", "match_reason"}
        assert hit["match_reason"] == "text in image"   # no CLIP: cannot claim visual
        assert hit["doc_id"] == indexed["doc_id"]
        assert hit["doc_name"] == "claim.pdf"
        assert hit["page"] in (1, 2)
        assert hit["similarity"] is None
        assert hit["score"] > 0
        assert "ABC123" in hit["ocr_text"]
    assert [h["score"] for h in hits] == sorted((h["score"] for h in hits), reverse=True)


def test_search_by_text_respects_the_limit(indexed):
    assert len(indexed["index"].search_by_text("damaged laptop serial", limit=1)) == 1


def test_search_by_text_returns_nothing_for_an_unrelated_query(indexed):
    assert indexed["index"].search_by_text("zzzqqq nonexistent term") == []


def test_search_by_text_survives_fts_punctuation(indexed):
    """A query full of FTS5 operators must not raise."""
    for query in ['laptop AND "', "NEAR(a b)", "* OR (", "ABC123 -- ;drop", ""]:
        assert isinstance(indexed["index"].search_by_text(query), list)


def test_search_by_image_finds_the_near_duplicate(indexed, tmp_path):
    """The extracted page-1 diagram, fed back in, ranks itself first via pHash."""
    query = indexed["records"][0]["path"]
    hits = indexed["index"].search_by_image(query, limit=5)
    assert hits
    top = hits[0]
    assert top["page"] == 1
    assert top["phash_distance"] == 0
    assert top["match_reason"] == "near-duplicate (phash d=0)"
    assert top["similarity"] is None            # no vision model: nothing to claim
    assert "near-duplicate (phash d=0)" in top["reasons"]
    assert top["doc_id"] == indexed["doc_id"]


def test_search_by_image_matches_a_re_saved_copy(indexed, tmp_path):
    """A slightly different copy of the same diagram still lands on it."""
    query = tmp_path / "resaved.png"
    query.write_bytes(im._png_bytes(slightly_modified(diagram())))
    hits = indexed["index"].search_by_image(query, limit=5)
    assert hits
    assert hits[0]["page"] == 1
    assert hits[0]["phash_distance"] is not None
    assert hits[0]["phash_distance"] <= im.NEAR_DUPLICATE_MAX
    assert hits[0]["match_reason"].startswith("near-duplicate (phash d=")


def test_search_by_image_falls_back_to_ocr_text(indexed, tmp_path):
    """An unrelated picture still reaches images whose OCR text the query shares."""
    query = tmp_path / "unrelated.png"
    query.write_bytes(im._png_bytes(blocks(seed=31)))
    hits = indexed["index"].search_by_image(query, limit=5)
    assert hits
    assert any(h["match_reason"] == "text in image" for h in hits)
    assert all(h["similarity"] is None for h in hits)


def test_search_by_image_accepts_an_array(indexed):
    assert indexed["index"].search_by_image(diagram(), limit=3)[0]["page"] == 1


def test_search_documents_by_image_aggregates_to_documents(indexed):
    docs = indexed["index"].search_documents_by_image(indexed["records"][0]["path"])
    assert docs
    top = docs[0]
    assert set(top) == {"doc_id", "doc_name", "score", "pages", "reasons"}
    assert top["doc_id"] == indexed["doc_id"]
    assert top["doc_name"] == "claim.pdf"
    assert top["score"] > 0
    assert top["pages"] == sorted(top["pages"])
    assert 1 in top["pages"]
    assert top["reasons"] and all(isinstance(r, str) for r in top["reasons"])
    assert any("near-duplicate" in r for r in top["reasons"])


def test_search_documents_by_image_respects_the_limit(indexed):
    assert len(indexed["index"].search_documents_by_image(
        indexed["records"][0]["path"], limit=1)) <= 1


def test_search_on_an_empty_index_is_empty(tmp_path, embedder):
    store = Store(tmp_path / "empty.db")
    index = im.ImageIndex(store, embedder, lambda arr: "anything")
    blank = tmp_path / "q.png"
    blank.write_bytes(im._png_bytes(diagram()))
    assert index.search_by_text("laptop") == []
    assert index.search_by_image(blank) == []
    assert index.search_documents_by_image(blank) == []
    assert index.capabilities()["mode"] == "ocr_phash"


def test_index_without_any_embedder_still_works(tmp_path, pdf_with_images):
    """image_embedder=None is a supported configuration, not a crash."""
    store = Store(tmp_path / "none.db")
    doc_id = store.add_document("c.pdf", str(pdf_with_images), "sha-none", 10, 3)
    store.mark_ready(doc_id, pages=3)
    records = im.extract_images(pdf_with_images, tmp_path / "x", doc_id=doc_id)
    index = im.ImageIndex(store, None, lambda arr: "damaged laptop serial ABC123")
    assert index.index_document_images(doc_id, records)["mode"] == "ocr_phash"
    assert index.capabilities()["vision_model"] is None
    assert index.search_by_text("damaged laptop")


# ===========================================================================
# helpers
# ===========================================================================
def test_fts_query_quotes_terms_and_drops_stopwords():
    query = im.fts_query("show me the photo of a DAMAGED laptop")
    assert '"damaged"' in query and '"laptop"' in query
    assert '"the"' not in query and '"photo"' not in query
    assert " OR " in query


def test_fts_query_is_empty_when_nothing_is_searchable():
    assert im.fts_query("") == ""
    assert im.fts_query("*** ( ) ***") == ""
    assert im.fts_query("the of a") == ""


def test_resize_rgb_hits_the_exact_size():
    out = im.resize_rgb(diagram(), 37, 61)
    assert out.shape == (61, 37, 3)
    assert out.dtype == np.uint8
    assert im.resize_rgb(diagram(), W, H).shape == (H, W, 3)


def test_load_rgb_upscales_small_images_for_ocr(tmp_path):
    src = tmp_path / "small.png"
    src.write_bytes(im._png_bytes(diagram()))
    assert im.load_rgb(src).shape == (H, W, 3)
    upscaled = im.load_rgb(src, min_width=1600)
    assert upscaled.shape[1] >= 1600
    assert upscaled.shape[2] == 3


def test_load_rgb_normalises_greyscale_arrays():
    grey = np.full((20, 30), 128, np.uint8)
    assert im.load_rgb(grey).shape == (20, 30, 3)


def test_preprocess_image_shape_and_normalisation(absent_model_dir):
    """Preprocessing is pure numpy and works with no model loaded."""
    embedder = im.ImageEmbedder(model_dir=absent_model_dir)
    pixels = embedder.preprocess_image(diagram())
    assert pixels.shape == (1, 3, im.CLIP_INPUT, im.CLIP_INPUT)
    assert pixels.dtype == np.float32
    # white pixels map to (1 - mean) / std, well outside [0, 1]
    assert pixels.max() > 1.5


def test_reindexing_twice_does_not_break_the_image_search_index(tmp_path, monkeypatch):
    """Regression: images_fts is contentless, so a plain DELETE raised
    "cannot DELETE from contentless fts5 table" and killed re-indexing."""
    import sys

    monkeypatch.setenv("RAGLY_DATA", str(tmp_path))
    for mod in [m for m in list(sys.modules) if m.startswith("ragly_backend")]:
        del sys.modules[mod]
    from ragly_backend.store import Store

    store = Store(tmp_path / "t.db")
    doc_id = store.add_document("d.pdf", str(tmp_path / "d.pdf"), "sha", 10, 1)
    for page in (1, 2):
        store.add_image({"doc_id": doc_id, "page": page, "path": str(tmp_path / f"i{page}.png"),
                         "source": "embedded", "width": 100, "height": 100, "sha256": f"s{page}",
                         "phash": "0" * 16, "ocr_text": f"invoice {page}", "caption": ""})
    assert len(store.get_images()) == 2
    store.delete_images_for_doc(doc_id)        # this raised OperationalError before the fix
    assert store.get_images() == []
    store.delete_images_for_doc(doc_id)        # and again, on an already-empty document
