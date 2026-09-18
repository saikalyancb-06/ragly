"""Background indexing queue: extract -> chunk -> embed -> store, one document at a time."""
from __future__ import annotations

import hashlib
import logging
import queue
import shutil
import sqlite3
import threading
import time
from pathlib import Path

from .autotune import Pack, build_pack, dumps, extract_entities, normalise_code
from . import pipeline
from .config import DATA_DIR, FILES_DIR, settings
from .embedder import Embedder
from .ingest import page_count
from .store import Store, SCHEMA

log = logging.getLogger("ragly.indexer")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Indexer:
    def __init__(self, store: Store, embedder: Embedder, on_retune=None,
                 image_index=None, graph_builder=None, project_id: int | None = None,
                 pack_key: str = "pack"):
        self.store = store
        self.embedder = embedder
        self.on_retune = on_retune
        self.image_index = image_index
        self.graph_builder = graph_builder
        self.project_id = project_id
        self.pack_key = pack_key
        self.images_dir = DATA_DIR / "images"
        self.pack: Pack | None = None
        self.q: "queue.Queue[int]" = queue.Queue()
        self.current: dict | None = None
        self._thread = threading.Thread(target=self._loop, name="ragly-indexer", daemon=True)
        self._stop = threading.Event()

    def start(self) -> None:
        # re-index automatically if the embedding model changed since the index was built
        prev = self.store.get_meta("embed_model")
        sig = f"{self.embedder.model_id}:{self.embedder.dim}"
        if prev and prev != sig:
            log.warning("embedding model changed (%s -> %s): re-indexing everything", prev, sig)
            self.store.reset_index()
        self.store.set_meta("embed_model", sig)
        raw = self.store.get_meta(self.pack_key)
        if raw:
            import json as _json

            self.pack = Pack.from_dict(_json.loads(raw))
        for doc_id in self.store.queued_ids():
            self.q.put(doc_id)
        self._thread.start()
        # A document can end up queued with nobody working on it: the app was restarted
        # mid-index (startup puts "indexing" rows back in the queue), or a second Store
        # instance reset the row. Nothing re-enqueued it, so it sat at "queued, 0 chunks"
        # forever with half its chunks already written. This watchdog picks up anything the
        # queue has forgotten; indexing replaces a document's chunks, so re-running is safe.
        self._watchdog_thread = threading.Thread(target=self._watchdog, daemon=True,
                                                 name="ragly-indexer-watchdog")
        self._watchdog_thread.start()

    def _watchdog(self, every: float = 5.0) -> None:
        while not self._stop.wait(every):
            try:
                if self.current is not None or not self.q.empty():
                    continue                      # work is already in flight
                for doc_id in self.store.queued_ids():
                    log.info("re-queuing document %s: queued with no worker", doc_id)
                    self.q.put(doc_id)
            except Exception as exc:              # a watchdog must never kill the process
                log.debug("watchdog pass failed: %s", exc)

    def stop(self) -> None:
        self._stop.set()
        self.q.put(-1)

    def add_file(self, src: Path, original_name: str) -> tuple[dict, bool]:
        """Copy into the private store and queue it. Returns (document, created)."""
        ext = Path(original_name).suffix.lower()
        if ext not in pipeline.ALL_SUPPORTED:
            raise ValueError(f"Unsupported file type '{ext}'. "
                             f"Supported: {', '.join(sorted(pipeline.ALL_SUPPORTED))}")
        sha = sha256_file(src)
        existing = self.store.find_by_hash(sha)
        if existing:
            return existing, False
        FILES_DIR.mkdir(parents=True, exist_ok=True)
        dest = FILES_DIR / f"{sha[:16]}{ext}"
        shutil.copyfile(src, dest)
        try:
            pages = page_count(dest)
        except Exception as exc:
            dest.unlink(missing_ok=True)
            raise ValueError(f"Could not open file: {exc}") from exc
        try:
            doc_id = self.store.add_document(original_name, str(dest), sha, dest.stat().st_size, pages,
                                             project_id=self.project_id)
        except sqlite3.IntegrityError:
            # A database created by an older build still has UNIQUE(sha256) across all workspaces,
            # which blocks the same file in a second workspace. Rebuild that table once and retry.
            if not self.store._rebuild_documents_if_globally_unique():
                raise ValueError("This file is already in this workspace.") from None
            self.store.conn.executescript(SCHEMA)
            doc_id = self.store.add_document(original_name, str(dest), sha, dest.stat().st_size, pages,
                                             project_id=self.project_id)
        self.q.put(doc_id)
        return self.store.get_document(doc_id), True

    def pending(self) -> int:
        return self.q.qsize() + (1 if self.current else 0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            doc_id = self.q.get()
            if doc_id < 0:
                break
            try:
                self.index_document(doc_id)
            except Exception as exc:  # keep the worker alive
                log.exception("indexing failed for %s", doc_id)
                self.store.set_status(doc_id, "error", error=str(exc)[:500])
            finally:
                self.current = None

    def index_document(self, doc_id: int) -> None:
        """Full multimodal indexing of one document: pages, tables, images, chunks, entities, graph."""
        doc = self.store.get_document(doc_id)
        if not doc or doc["status"] == "ready":
            return
        t0 = time.perf_counter()
        path = Path(doc["path"])
        self.current = {"doc_id": doc_id, "name": doc["name"], "stage": "extracting"}
        self.store.set_status(doc_id, "indexing", error=None)

        self.images_dir.mkdir(parents=True, exist_ok=True)
        result = pipeline.process(path, doc_id, self.images_dir, self.embedder.count_tokens,
                                  self.embedder.token_windows, display_name=doc["name"])
        chunks = result["chunks"]
        if not chunks and not result["images"]:
            raise ValueError("No readable text, table or image found (even after OCR)")

        self.current["stage"] = "storing pages"
        self.store.replace_pages(doc_id, result["pages"])

        # the label goes up BEFORE the work, not after it: embedding 264 chunks takes half a
        # minute and the screen was still saying "storing pages" for all of it
        self.current["stage"] = f"embedding {len(chunks)} chunks"
        vecs = self.embedder.embed_documents([c.text for c in chunks]) if chunks else None
        if self.store.get_document(doc_id) is None:      # deleted meanwhile
            return
        if chunks:
            entities = [[(k, v, normalise_code(v)) for k, v in extract_entities(c.text)] for c in chunks]
            self.store.replace_chunks(doc_id, chunks, vecs, entities)
        chunk_ids = self.store.chunk_ids(doc_id)

        # tables keep a pointer to the chunk that carries their text form
        tables = result["tables"]
        for chunk_index, table_index in result["table_chunk_map"]:
            if chunk_index < len(chunk_ids) and tables[table_index].get("chunk_id") is None:
                tables[table_index]["chunk_id"] = chunk_ids[chunk_index]
        if tables:
            self.current["stage"] = f"storing {len(tables)} table(s)"
            self.store.replace_tables(doc_id, tables)

        image_info = {"images": 0, "embedded": 0, "mode": "disabled"}
        if self.image_index is not None and result["images"]:
            self.current["stage"] = f"indexing {len(result['images'])} image(s)"
            self.store.delete_images_for_doc(doc_id)
            try:
                image_info = self.image_index.index_document_images(doc_id, result["images"])
            except Exception as exc:
                log.warning("image indexing failed for %s: %s", doc["name"], exc)

        if self.graph_builder is not None and chunks:
            self.current["stage"] = "building the entity graph"
            try:
                self.graph_builder.index_chunks(doc_id, chunks, chunk_ids)
            except Exception as exc:
                log.warning("graph indexing failed for %s: %s", doc["name"], exc)

        secs = round(time.perf_counter() - t0, 2)
        self.store.mark_ready(doc_id, pages=result["page_count"], ocr_pages=result["ocr_pages"],
                              chunk_count=len(chunks), table_count=len(tables),
                              image_count=int(image_info.get("images", 0)),
                              index_seconds=secs, error=None)
        log.info("indexed %s: %d pages, %d chunks, %d tables, %d images, %d OCR pages in %.1fs",
                 doc["name"], result["page_count"], len(chunks), len(tables),
                 image_info.get("images", 0), result["ocr_pages"], secs)
        self.retune()

    def retune(self) -> Pack:
        """Rebuild the auto-tuned pack from everything indexed so far."""
        pack = build_pack(self.store.all_chunk_texts(), docs=self.store.counts()["documents"])
        self.store.set_meta(self.pack_key, dumps(pack))
        self.pack = pack
        if self.on_retune:
            self.on_retune(pack)
        log.info("auto-tuned pack: %s", pack.summary())
        return pack
