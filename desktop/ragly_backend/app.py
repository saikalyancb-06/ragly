"""Shared application state: models, indexes and retrieval, wired to the detected hardware."""
from __future__ import annotations

import logging
from pathlib import Path

from . import offline, registry
from .answer import Answerer
from .compare import compare_documents
from .config import MODELS_DIR, ensure_dirs, load_user_settings, save_user_settings, settings
from .embedder import Embedder
from .engine import EngineManager
from .graph import GraphBuilder, GraphRetriever
from .hardware import InferenceManager, detect
from .images import ImageEmbedder, ImageIndex, load_rgb
from .indexer import Indexer
from .ingest import OCR
from .retriever import Retriever
from .store import Store

log = logging.getLogger("ragly")


class App:
    def __init__(self, start_indexer: bool = True):
        if settings.offline_guard:
            offline.install()
        ensure_dirs()

        # ---- hardware first: everything else asks it which backend to use
        self.hardware = detect()
        self.inference = InferenceManager(self.hardware)
        log.info("hardware: %s", self.hardware.summary())

        self.engine = EngineManager()
        self.store = Store()
        self.project_id = self.store.ensure_default_project()
        saved = load_user_settings().get("project_id")
        if saved and self.store.get_project(int(saved)):
            self.project_id = int(saved)
        self.store.scope(self.project_id)

        # ---- text embeddings (int8 when an NPU is present and installed, else fp32)
        subdir, reason = registry.embedding_choice(self.hardware.npu_available)
        self.embed_reason = reason
        providers = self.inference.providers("text_embedding")
        self.embedder = Embedder(model_dir=MODELS_DIR / subdir, providers=providers)
        self.inference.record("text_embedding", self.embedder.provider)
        log.info("text embeddings: %s on %s (%s)", self.embedder.model_id, self.embedder.provider, reason)

        # ---- image embeddings (CLIP if installed; otherwise OCR + perceptual hash)
        self.image_embedder = ImageEmbedder(providers=self.inference.providers("image_embedding"))
        if self.image_embedder.available():
            self.inference.record("image_embedding", self.image_embedder.status().get("provider", ""))
        self.image_index = ImageIndex(self.store, self.image_embedder,
                                      ocr_callable=lambda p: OCR.image_to_text(load_rgb(p)))

        # ---- retrieval, graph, answering
        # Repair entity names left by an older build ("No Axis Bank" -> "Axis Bank"), once,
        # so an existing database gets a readable graph without being re-indexed.
        from . import BUILD

        if self.store.get_meta("graph_clean_build") != BUILD:
            try:
                fixed = self.store.clean_graph_nodes()
                self.store.set_meta("graph_clean_build", BUILD)
                if fixed:
                    log.info("graph tidy-up: %d entity names repaired or merged", fixed)
            except Exception as exc:              # never block startup on a cleanup
                log.warning("graph tidy-up skipped: %s", exc)

        # Images indexed before the vision model existed get understood once, in the background
        # of the first run, so an older workspace behaves like a new one without re-indexing.
        try:
            self.image_index.understand_pending()
        except Exception as exc:
            log.debug("image tagging pass skipped: %s", exc)

        self.graph_builder = GraphBuilder(self.store)
        self.graph = GraphRetriever(self.store)
        self.retriever = Retriever(self.store, self.embedder)
        self.retriever.graph = self.graph
        self.answerer = Answerer(self.retriever)
        self.answerer.store = self.store
        self.indexer = Indexer(self.store, self.embedder, on_retune=lambda _p: self.sync_pack(),
                               image_index=self.image_index, graph_builder=self.graph_builder,
                               project_id=self.project_id, pack_key=self.pack_key())
        if start_indexer:
            self.indexer.start()
        else:
            raw = self.store.get_meta(self.pack_key())
            if raw:
                import json as _json

                from .autotune import Pack

                self.indexer.pack = Pack.from_dict(_json.loads(raw))
        self.sync_pack()

    # ---------------- projects ----------------
    def pack_key(self) -> str:
        return f"pack:{self.project_id}"

    def projects(self) -> list[dict]:
        return [{**p, "active": p["id"] == self.project_id} for p in self.store.list_projects()]

    def create_project(self, name: str, description: str = "", activate: bool = True) -> dict:
        pid = self.store.create_project(name, description)
        if activate:
            self.switch_project(pid)
        return self.store.get_project(pid)

    def switch_project(self, project_id: int) -> dict:
        project = self.store.get_project(project_id)
        if not project:
            raise ValueError("project not found")
        self.project_id = project_id
        self.store.scope(project_id)
        self.indexer.project_id = project_id
        save_user_settings({"project_id": project_id})
        raw = self.store.get_meta(self.pack_key())
        if raw:
            import json as _json

            from .autotune import Pack

            self.indexer.pack = Pack.from_dict(_json.loads(raw))
        else:
            self.indexer.pack = None
        self.indexer.pack_key = self.pack_key()
        self.sync_pack()
        return project

    def delete_project(self, project_id: int) -> dict:
        if len(self.store.list_projects()) <= 1:
            raise ValueError("a workspace must always have at least one project")
        removed = self.store.delete_project(project_id)
        for d in removed:
            if not self.store.path_in_use(d["path"]):   # kept if another workspace still uses it
                Path(d["path"]).unlink(missing_ok=True)
        if self.project_id == project_id:
            self.switch_project(self.store.list_projects()[0]["id"])
        return {"deleted": project_id, "documents_removed": len(removed)}

    # ---------------- pack ----------------
    def sync_pack(self) -> None:
        self.retriever.pack = self.indexer.pack
        self.answerer.pack = self.indexer.pack

    def retune(self):
        pack = self.indexer.retune()
        self.sync_pack()
        return pack

    # ---------------- comparison ----------------
    def compare(self, left_id: int, right_id: int) -> dict:
        left, right = self.store.get_document(left_id), self.store.get_document(right_id)
        if not left or not right:
            raise ValueError("both documents must exist")
        return compare_documents(left, self.store.get_pages(left_id),
                                 right, self.store.get_pages(right_id))

    # ---------------- danger zone ----------------
    def wipe(self) -> dict:
        """Delete every project, document, index and stored file. Used by Settings → erase all data."""
        import shutil

        from .config import DATA_DIR, FILES_DIR

        counts = self.store.counts()
        projects = len(self.store.list_projects())
        for project in self.store.list_projects():
            try:
                self.store.delete_project(project["id"])
            except Exception:
                pass
        with self.store._lock:
            for table in ("graph_edges", "node_mentions", "graph_nodes", "embed_cache",
                          "benchmarks", "images", "image_vectors", "tables", "pages", "meta"):
                try:
                    self.store.conn.execute(f"DELETE FROM {table}")
                except Exception:
                    pass
            self.store.conn.commit()
        shutil.rmtree(FILES_DIR, ignore_errors=True)
        shutil.rmtree(DATA_DIR / "images", ignore_errors=True)
        FILES_DIR.mkdir(parents=True, exist_ok=True)
        self.project_id = self.store.ensure_default_project()
        self.store.scope(self.project_id)
        self.indexer.project_id = self.project_id
        self.indexer.pack = None
        self.indexer.pack_key = self.pack_key()
        self.sync_pack()
        return {"documents_removed": counts["documents"], "projects_removed": projects}

    # ---------------- diagnostics ----------------
    def diagnostics(self) -> dict:
        counts = self.store.counts()
        return {
            "project": self.store.get_project(self.project_id),
            "projects": self.projects(),
            "documents_processed": counts["documents"],
            "pages": counts["pages"],
            "text_chunks": counts["chunks"],
            "vectors": counts["chunks"],
            "content_types": self.store.content_type_counts(),
            "tables": len(self.store.list_tables()),
            "images": self.store.image_counts(),
            "graph": self.store.graph_counts(),
            "embedding_cache": self.store.cache_size(),
            "models": registry.registry_status(),
            "hardware": self.hardware.to_dict(),
            "backends": self.inference.report(),
            "embedding_model": {"id": self.embedder.model_id, "dim": self.embedder.dim,
                                "quantization": self.embedder.quantization,
                                "provider": self.embedder.provider, "reason": self.embed_reason},
            "vision": self.image_index.capabilities(),
            "ocr": OCR.status(),
            "llm": self.engine.status(),
            "offline_guard": offline.status(),
            "cloud_api_calls": 0,
            "internet_required": False,
        }

    def start_engine(self) -> None:
        try:
            self.engine.start()
        except Exception as exc:
            log.error("answer engine failed to start: %s", exc)

    def shutdown(self) -> None:
        self.indexer.stop()
        self.engine.stop()
