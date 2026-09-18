"""Image pipeline regression tests: extraction → embedding → retrieval → provenance.

Runs against samples/Offline_Multimodal_RAG_Test_Corpus.pdf, which carries an invoice
image (page 5), an employee ID card (page 8) and a workstation photo (page 9).
The CLIP model is optional on a build machine: the visual assertions are skipped when it
is not installed, the extraction and provenance assertions always run.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CORPUS = ROOT / "samples" / "Offline_Multimodal_RAG_Test_Corpus.pdf"
WORKSTATION = ROOT / "samples" / "multimodal" / "workstation.png"
pytestmark = pytest.mark.skipif(not CORPUS.exists(), reason="test corpus not generated")


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    os.environ["RAGLY_DATA"] = str(tmp_path_factory.mktemp("imgdata"))
    for mod in [m for m in list(sys.modules) if m.startswith("ragly_backend")]:
        del sys.modules[mod]
    from ragly_backend.app import App

    a = App(start_indexer=False)
    doc, _ = a.indexer.add_file(CORPUS, CORPUS.name)
    a.indexer.index_document(doc["id"])
    yield a


def images(app):
    return app.store.get_images()


# ---------------------------------------------------------------- ingestion
def test_embedded_images_are_extracted(app):
    assert len(images(app)) >= 3, "FAIL IMAGE INGESTION: no embedded images found"


def test_every_image_has_stable_identity_and_provenance(app):
    for im in images(app):
        assert im["id"] and im["doc_id"] and im["page"], "image is missing provenance"
        assert Path(im["path"]).exists(), "extracted image file is missing on disk"
        assert im["width"] > 0 and im["height"] > 0


def test_images_are_on_the_expected_pages(app):
    pages = {im["page"] for im in images(app)}
    assert {5, 8, 9} <= pages, f"expected images on pages 5, 8 and 9, found {sorted(pages)}"


def test_image_embeddings_are_persisted(app):
    if not app.image_index.clip_ready():
        pytest.skip("vision model not installed on this machine")
    caps = app.image_index.capabilities()["images"]
    assert caps["with_vectors"] == caps["images"], "FAIL IMAGE EMBEDDING: vectors missing"


# ---------------------------------------------------------------- retrieval
def test_exact_image_retrieves_itself(app):
    if not WORKSTATION.exists():
        pytest.skip("query image not generated")
    hits = app.image_index.search_by_image(WORKSTATION, limit=3)
    assert hits, "FAIL IMAGE RETRIEVAL: nothing returned for an indexed image"
    assert hits[0]["page"] == 9, "FAIL IMAGE RANKING: the source image is not ranked first"


def test_resized_and_recompressed_image_still_retrieves_the_original(app, tmp_path):
    if not WORKSTATION.exists():
        pytest.skip("query image not generated")
    from ragly_backend.images import load_rgb, resize_rgb
    import pymupdf as fitz

    arr = load_rgb(WORKSTATION)
    small = resize_rgb(arr, arr.shape[1] // 2, arr.shape[0] // 2)
    out = tmp_path / "resized.jpg"
    h, w = small.shape[:2]
    pix = fitz.Pixmap(fitz.csRGB, w, h, np.ascontiguousarray(small[:, :, :3]).tobytes(), False)
    pix.save(str(out), "jpg")             # resized AND recompressed in one query
    hits = app.image_index.search_by_image(out, limit=3)
    assert any(h_["page"] == 9 for h_ in hits[:2]), "resized copy did not retrieve the original"


def test_text_to_image_search_uses_the_vision_model(app):
    if not app.image_index.clip_ready():
        pytest.skip("vision model not installed on this machine")
    hits = app.image_index.search_by_text("workstation with a laptop", limit=3)
    assert hits, "FAIL IMAGE RETRIEVAL: text query returned nothing"
    assert hits[0]["page"] == 9
    assert hits[0]["similarity"] is not None, "similarity score must be visible"


def test_image_to_document_search_returns_page_provenance(app):
    if not WORKSTATION.exists():
        pytest.skip("query image not generated")
    docs = app.image_index.search_documents_by_image(WORKSTATION, limit=3)
    assert docs and docs[0]["doc_name"] == CORPUS.name
    assert docs[0]["pages"], "a document hit must say which page matched"


# ---------------------------------------------------------------- debug contract
def test_debug_block_reports_real_numbers(app):
    hits = app.image_index.search_by_text("invoice", limit=3)
    dbg = app.image_index.debug("text_to_image", app.image_index.last_query_vector, len(hits), 3, hits)
    assert dbg["indexed_images"] == len(images(app))
    assert dbg["query_type"] == "text_to_image"
    assert dbg["top_k"] == 3
    if app.image_index.clip_ready():
        assert dbg["embedding_dimension"] == 512
        assert 0.99 <= dbg["query_embedding_norm"] <= 1.01, "query vector must be L2-normalised"
        assert dbg["embedding_model"]


# ---------------------------------------------------------------- image understanding
def test_every_image_gets_concepts_and_a_description(app):
    import json

    if not app.image_index.clip_ready():
        pytest.skip("vision model not installed on this machine")
    described = 0
    for im in app.store.get_images():
        objects = json.loads(im.get("objects") or "[]")
        if objects:
            described += 1
            assert im.get("caption"), "an image with concepts must also carry a description"
            assert im.get("caption_source"), "the description must name the model that wrote it"
    assert described, "FAIL IMAGE METADATA: no image was understood"


def test_a_concept_query_finds_the_image_that_contains_it(app):
    if not app.image_index.clip_ready():
        pytest.skip("vision model not installed on this machine")
    hits = app.image_index.search_by_text("Find images containing a laptop", limit=3)
    assert hits and hits[0]["page"] == 9
    assert "laptop" in " ".join(hits[0].get("matched_concepts") or []) or hits[0]["similarity"] is not None


def test_a_concept_the_corpus_does_not_contain_returns_nothing(app):
    """No blind cut-off: the vision model listed what is in every picture, and no picture has a car."""
    if not app.image_index.clip_ready():
        pytest.skip("vision model not installed on this machine")
    assert app.image_index.search_by_text("Find images containing a car", limit=3) == []
    assert app.image_index.search_by_text("Find images containing a dog", limit=3) == []


def test_synonyms_reach_the_same_concept():
    from ragly_backend.concepts import expand_query

    assert "car" in expand_query("find images containing a vehicle")
    assert "car" in expand_query("photos of an automobile")
    assert "mobile phone" in expand_query("images with a smartphone")
    assert expand_query("what is the interest rate") == []      # not an image query at all
