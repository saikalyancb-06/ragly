"""FastAPI backend. Bound to 127.0.0.1 only. Endpoints are under /api."""
from __future__ import annotations

import json
import logging
import tempfile
import time
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import BUILD, __version__, offline, registry
from .auth import Auth
from .app import App
from .config import ROOT, settings
from .suggest import recommend
from .visualize import Planner
from .hardware import bench
from .ingest import OCR, render_page_png
from .pipeline import ALL_SUPPORTED as SUPPORTED
from .router import classify, explain
from .voice import Voice, VoiceError, spoken_summary

log = logging.getLogger("ragly.api")
state: dict = {}


def app_state() -> App:
    return state["app"]


def auth_state() -> Auth:
    if "auth" not in state:
        state["auth"] = Auth()
    return state["auth"]





@asynccontextmanager
async def lifespan(_: FastAPI):
    a = App()
    state["app"] = a
    state["auth"] = Auth()
    # start the answer engine in the background so the API is usable immediately
    threading.Thread(target=a.start_engine, name="engine-start", daemon=True).start()
    yield
    a.shutdown()


api = FastAPI(title="Falcon offline document intelligence", version=__version__, lifespan=lifespan)
PUBLIC_PATHS = ("/api/auth/", "/api/health", "/assets/", "/docs", "/openapi.json", "/favicon")


@api.middleware("http")
async def require_unlock(request, call_next):
    """When a passcode is set, every API call needs the session token. The UI sends it as a
    Bearer header; the landing, auth and static routes stay open so the app can load and log in."""
    path = request.url.path
    if path.startswith("/api") and not path.startswith(PUBLIC_PATHS):
        auth = auth_state()
        if auth.enabled:
            token = (request.headers.get("authorization", "").removeprefix("Bearer ").strip()
                     or request.query_params.get("token", ""))
            if not auth.valid(token):
                return Response('{"detail":"locked: sign in first"}', status_code=401,
                                media_type="application/json")
    return await call_next(request)


api.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(https?://(localhost|127\.0\.0\.1)(:\d+)?|tauri://localhost|http://tauri\.localhost)$",
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- local passcode lock ----------------
class PasscodeRequest(BaseModel):
    passcode: str = Field(min_length=1, max_length=200)
    display_name: str = Field(default="", max_length=80)


class ChangePasscodeRequest(BaseModel):
    old: str = Field(default="", max_length=200)
    new: str = Field(min_length=1, max_length=200)


@api.get("/api/auth/status")
def auth_status():
    a = auth_state()
    return {**a.state().to_dict(), "min_length": 4,
            "note": ("A local passcode locks this app on this device. The backend already refuses "
                     "any non-localhost connection; the database itself is not encrypted.")}


@api.post("/api/auth/register")
def auth_register(req: PasscodeRequest):
    try:
        token = auth_state().register(req.passcode, req.display_name)
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    return {"token": token, **auth_state().state().to_dict()}


@api.post("/api/auth/login")
def auth_login(req: PasscodeRequest):
    a = auth_state()
    if not a.verify(req.passcode):
        raise HTTPException(401, "wrong passcode")
    return {"token": a.issue_token(), **a.state().to_dict()}


@api.post("/api/auth/check")
def auth_check(payload: dict):
    return {"valid": auth_state().valid(str(payload.get("token") or ""))}


@api.post("/api/auth/change")
def auth_change(req: ChangePasscodeRequest):
    try:
        return {"token": auth_state().change_passcode(req.old, req.new)}
    except ValueError as exc:
        raise HTTPException(401, str(exc))


@api.post("/api/auth/disable")
def auth_disable(req: PasscodeRequest):
    try:
        auth_state().disable(req.passcode)
    except ValueError as exc:
        raise HTTPException(401, str(exc))
    return {"enabled": False}


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=12)
    doc_ids: list[int] | None = None
    tuning: bool = True          # False = plain RAG, for the A/B demo


class EngineRequest(BaseModel):
    mode: str


class SettingsRequest(BaseModel):
    top_k: int | None = Field(default=None, ge=1, le=12)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    strict_grounding: bool | None = None
    max_tokens: int | None = Field(default=None, ge=64, le=2048)


def _doc_view(d: dict) -> dict:
    d = dict(d)
    d.pop("path", None)
    return d


@api.get("/api/health")
def health():
    a = app_state()
    return {"ok": True, "engine": a.engine.state, "version": __version__}


@api.get("/api/hardware")
def hardware():
    """Real hardware detection: never claims an accelerator the runtime does not report."""
    a = app_state()
    return {"device": a.hardware.to_dict(), "backends": a.inference.report(),
            "models": registry.registry_status()}


@api.get("/api/diagnostics")
def diagnostics():
    return app_state().diagnostics()


@api.get("/api/registry")
def model_registry():
    return {"models": registry.registry_status()}


@api.post("/api/route")
def route(req: dict):
    r = classify(str(req.get("question", "")), has_image=bool(req.get("has_image")),
                 doc_ids=req.get("doc_ids"))
    return {**r.to_dict(), "explanation": explain(r)}


@api.get("/api/system")
def system():
    a = app_state()
    return {
        "version": __version__,
        "build": BUILD,
        "device": a.engine.device,
        "engine": a.engine.status(),
        "embedder": {"model": a.embedder.model_id, "dim": a.embedder.dim, "provider": a.embedder.provider},
        "project": a.store.get_project(a.project_id),
        "projects": a.projects(),
        "index": a.store.counts(),
        "pack": a.indexer.pack.to_dict() if a.indexer.pack else None,
        "indexing": {"pending": a.indexer.pending(), "current": a.indexer.current},
        "offline_guard": offline.status(),
        "ocr": OCR.status(),
        "hardware": {"summary": a.hardware.summary(), "is_snapdragon": a.hardware.is_snapdragon,
                     "npu_available": a.hardware.npu_available, "gpu_available": a.hardware.gpu_available,
                     "cpu": a.hardware.cpu, "mode": a.inference.mode,
                     "backends": a.inference.report()["actual"]},
        "vision": a.image_index.capabilities(),
        "content_types": a.store.content_type_counts(),
        "graph": a.store.graph_counts(),
        "images": a.store.image_counts(),
        "tables": len(a.store.list_tables()),
        "settings": {"top_k": settings.top_k, "min_score": settings.min_score,
                     "strict_grounding": settings.strict_grounding, "max_tokens": settings.max_tokens,
                     "chunk_tokens": settings.chunk_tokens, "chunk_overlap": settings.chunk_overlap},
        "voice": Voice.status(),
        "supported_types": sorted(SUPPORTED),
    }


# ---------------- danger zone ----------------
class ResetRequest(BaseModel):
    passcode: str = Field(default="", max_length=200)
    confirm: str = Field(default="", max_length=40)


@api.post("/api/admin/reset")
def reset_everything(req: ResetRequest):
    """Erase every document, index, image, entity and project on this device."""
    a = app_state()
    auth = auth_state()
    if auth.enabled and not auth.verify(req.passcode):
        raise HTTPException(401, "wrong passcode")
    if req.confirm.strip().upper() != "DELETE":
        raise HTTPException(400, "type DELETE to confirm")
    removed = a.wipe()
    return {"reset": True, **removed}


# ---------------- projects ----------------
class ProjectRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=500)


@api.get("/api/projects")
def list_projects():
    a = app_state()
    return {"projects": a.projects(), "active": a.project_id}


@api.post("/api/projects")
def create_project(req: ProjectRequest):
    a = app_state()
    try:
        return a.create_project(req.name, req.description)
    except Exception as exc:
        raise HTTPException(409, f"could not create the project: {exc}")


@api.post("/api/projects/{project_id}/activate")
def activate_project(project_id: int):
    try:
        return app_state().switch_project(project_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc))


@api.patch("/api/projects/{project_id}")
def rename_project(project_id: int, req: ProjectRequest):
    a = app_state()
    if not a.store.get_project(project_id):
        raise HTTPException(404, "project not found")
    a.store.rename_project(project_id, req.name, req.description)
    return a.store.get_project(project_id)


@api.delete("/api/projects/{project_id}")
def delete_project(project_id: int):
    try:
        return app_state().delete_project(project_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc))


# ---------------- engine toggle ----------------
@api.get("/api/engine")
def get_engine():
    return app_state().engine.status()


@api.post("/api/engine")
def set_engine(req: EngineRequest):
    try:
        return app_state().engine.switch(req.mode)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(409, str(exc))


@api.post("/api/engine/restart")
def restart_engine():
    a = app_state()
    try:
        a.engine.stop()
        return a.engine.start()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))


# ---------------- documents ----------------
@api.post("/api/documents")
async def upload(files: list[UploadFile] = File(...)):
    a = app_state()
    out = []
    for f in files:
        name = Path(f.filename or "document").name
        suffix = Path(name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            while chunk := await f.read(1 << 20):
                tmp.write(chunk)
            tmp_path = Path(tmp.name)
        try:
            doc, created = a.indexer.add_file(tmp_path, name)
            out.append({"document": _doc_view(doc), "created": created})
        except ValueError as exc:
            out.append({"filename": name, "error": str(exc)})
        finally:
            tmp_path.unlink(missing_ok=True)
    return {"results": out}


@api.post("/api/documents/from-path")
def add_from_path(payload: dict):
    """Index a file or every supported file in a folder on this machine (no upload copy)."""
    a = app_state()
    p = Path(str(payload.get("path", ""))).expanduser()
    if not p.exists():
        raise HTTPException(404, f"Path not found: {p}")
    files = [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.suffix.lower() in SUPPORTED)
    out = []
    for f in files:
        try:
            doc, created = a.indexer.add_file(f, f.name)
            out.append({"document": _doc_view(doc), "created": created})
        except ValueError as exc:
            out.append({"filename": f.name, "error": str(exc)})
    return {"results": out}


@api.get("/api/documents")
def list_documents():
    a = app_state()
    return {"documents": [_doc_view(d) for d in a.store.list_documents()], "index": a.store.counts()}


@api.get("/api/documents/{doc_id}")
def get_document(doc_id: int):
    d = app_state().store.get_document(doc_id)
    if not d:
        raise HTTPException(404, "document not found")
    return _doc_view(d)


@api.delete("/api/documents/{doc_id}")
def delete_document(doc_id: int):
    store = app_state().store
    d = store.delete_document(doc_id)
    if not d:
        raise HTTPException(404, "document not found")
    if not store.path_in_use(d["path"]):          # another workspace may hold the same file
        Path(d["path"]).unlink(missing_ok=True)
    return {"deleted": doc_id, "name": d["name"]}


@api.post("/api/documents/{doc_id}/reindex")
def reindex(doc_id: int):
    a = app_state()
    if not a.store.get_document(doc_id):
        raise HTTPException(404, "document not found")
    a.store.set_status(doc_id, "queued")
    a.indexer.q.put(doc_id)
    return {"queued": doc_id}


@api.get("/api/documents/{doc_id}/file")
def document_file(doc_id: int):
    d = app_state().store.get_document(doc_id)
    if not d:
        raise HTTPException(404, "document not found")
    return FileResponse(d["path"], filename=d["name"])


@api.get("/api/documents/{doc_id}/pages/{page}")
def page_image(doc_id: int, page: int, chunk_id: int | None = Query(default=None), dpi: int = Query(110, ge=50, le=300)):
    a = app_state()
    d = a.store.get_document(doc_id)
    if not d:
        raise HTTPException(404, "document not found")
    highlight = None
    if chunk_id is not None:
        ch = a.store.get_chunks([chunk_id]).get(chunk_id)
        highlight = ch["text"] if ch and ch["doc_id"] == doc_id else None
    try:
        png = render_page_png(Path(d["path"]), page, highlight, dpi)
    except (ValueError, IndexError) as exc:
        raise HTTPException(400, str(exc))
    return Response(png, media_type="image/png")


@api.get("/api/settings")
def get_settings():
    return {"top_k": settings.top_k, "min_score": settings.min_score,
            "strict_grounding": settings.strict_grounding, "max_tokens": settings.max_tokens}


@api.post("/api/settings")
def set_settings(req: SettingsRequest):
    for field_name, value in req.model_dump(exclude_none=True).items():
        setattr(settings, field_name, value)
    return get_settings()


# ---------------- voice (push to talk) ----------------
class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


@api.get("/api/voice")
def voice_status():
    return Voice.status()


@api.post("/api/voice/listen")
def voice_listen(timeout_s: float = Query(8.0, ge=2.0, le=20.0)):
    a = app_state()
    try:
        return Voice.listen(timeout_s, a.indexer.pack)
    except VoiceError as exc:
        raise HTTPException(503, str(exc))


@api.post("/api/voice/speak")
def voice_speak(req: SpeakRequest):
    try:
        return Response(Voice.speak(req.text), media_type="audio/wav")
    except VoiceError as exc:
        raise HTTPException(503, str(exc))


@api.post("/api/voice/ask")
def voice_ask(timeout_s: float = Query(8.0, ge=2.0, le=20.0), speak: bool = Query(True)):
    """Listen -> answer -> (optionally) the spoken text for the UI to play."""
    a = app_state()
    llm = _llm_or_409()
    try:
        heard = Voice.listen(timeout_s, a.indexer.pack)
    except VoiceError as exc:
        raise HTTPException(503, str(exc))
    result = a.answerer.ask(llm, heard["text"])
    result["heard"] = heard
    if speak:
        result["speech_text"] = spoken_summary(result["answer"], result.get("citations", []))
    return result


# ---------------- auto-tuning ----------------
@api.get("/api/pack")
def get_pack():
    a = app_state()
    if not a.indexer.pack:
        return {"pack": None, "summary": "No documents indexed yet"}
    return {"pack": a.indexer.pack.to_dict(), "summary": a.indexer.pack.summary()}


@api.post("/api/pack/retune")
def retune():
    a = app_state()
    pack = a.retune()
    return {"pack": pack.to_dict(), "summary": pack.summary()}


@api.get("/api/entities")
def entities(kind: str = Query("code"), limit: int = Query(50, ge=1, le=500)):
    return {"kind": kind, "items": app_state().store.entity_summary(kind, limit)}


#: Recommendations are rebuilt when the workspace changes, not on every poll.
_SUGGESTIONS: dict[str, list[str]] = {}


@api.get("/api/suggestions")
def suggestions(limit: int = Query(6, ge=1, le=12)):
    """One summary per indexed document, plus an all-documents summary when there are several.

    ``items`` carries the doc_ids so each summary reads only its own document; ``questions``
    is the same list as plain strings, kept for older callers. See :mod:`ragly_backend.suggest`.
    """
    a = app_state()
    key = f"{a.project_id}:{a.pack_key()}:{len(a.store.list_documents())}:{limit}"
    if key not in _SUGGESTIONS:
        _SUGGESTIONS.clear()
        _SUGGESTIONS[key] = recommend(a, limit)
    items = _SUGGESTIONS[key]
    return {"items": items, "questions": [i["question"] for i in items]}


# ---------------- search + ask ----------------
@api.get("/api/documents/{doc_id}/pagetext")
def page_text(doc_id: int, page: int = Query(1, ge=1)):
    """Plain text of a page — used by the viewer for spreadsheets and notes, which have no page image."""
    a = app_state()
    doc = a.store.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "document not found")
    pages = a.store.get_pages(doc_id)
    row = next((p for p in pages if p["page"] == page), None)
    return {"doc_id": doc_id, "page": page, "content_type": (row or {}).get("content_type"),
            "renderable": Path(doc["path"]).suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".bmp",
                                                               ".tif", ".tiff", ".webp"},
            "pages": len(pages), "text": (row or {}).get("text", "")}


@api.get("/api/search")
def search(q: str = Query(min_length=1), top_k: int = Query(5, ge=1, le=20), tuning: bool = Query(True)):
    hits, timing = app_state().retriever.search(q, top_k, tuning=tuning)
    return {"hits": [h.__dict__ for h in hits], "timing": timing}


def _llm_or_409():
    try:
        return app_state().engine.require_client()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))


@api.post("/api/ask")
def ask(req: AskRequest):
    llm = _llm_or_409()
    try:
        return app_state().answerer.ask(llm, req.question, req.top_k, req.doc_ids, req.tuning)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        log.exception("ask failed")
        raise HTTPException(502, f"answer engine error: {exc}")


@api.post("/api/ask/stream")
def ask_stream(req: AskRequest):
    """Server-Sent Events: event types 'sources', 'token', 'done', 'error'."""
    llm = _llm_or_409()
    a = app_state()

    def gen():
        try:
            for ev in a.answerer.stream(llm, req.question, req.top_k, req.doc_ids, req.tuning):
                yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
        except Exception as exc:
            log.exception("stream failed")
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------- web UI ----------------
UI_DIR = ROOT / "ui" / "dist"
if UI_DIR.is_dir():
    api.mount("/assets", StaticFiles(directory=UI_DIR / "assets"), name="assets")

    @api.get("/", response_class=HTMLResponse)
    def ui_index():
        return (UI_DIR / "index.html").read_text(encoding="utf-8")
else:  # the UI has not been built yet

    @api.get("/", response_class=HTMLResponse)
    def ui_missing():
        return (
            "<h3>Ragly backend is running</h3>"
            "<p>The UI is not built. Run <code>npm install &amp;&amp; npm run build</code> in <code>ui/</code>, "
            "or use the API docs at <a href='/docs'>/docs</a>.</p>"
        )


# ---------------- images / multimodal search ----------------
class ImageQuery(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=12, ge=1, le=60)


@api.get("/api/images")
def list_images(doc_id: int | None = Query(default=None), limit: int = Query(120, ge=1, le=500)):
    a = app_state()
    return {"images": a.store.get_images(doc_id=doc_id, limit=limit),
            "capabilities": a.image_index.capabilities()}


@api.get("/api/images/{image_id}/file")
def image_file(image_id: int):
    a = app_state()
    rows = a.store.get_images(ids=[image_id])
    if not rows:
        raise HTTPException(404, "image not found")
    path = Path(rows[0]["path"])
    if not path.exists():
        raise HTTPException(404, "image file missing on disk")
    return FileResponse(path)


@api.post("/api/images/search")
def image_search(req: ImageQuery):
    a = app_state()
    results = a.image_index.search_by_text(req.query, req.limit)
    return {"results": results, "capabilities": a.image_index.capabilities(),
            "debug": a.image_index.debug("text_to_image", getattr(a.image_index, "last_query_vector", None),
                                         candidates=len(results), limit=req.limit, hits=results)}


@api.post("/api/images/search-by-image")
async def image_search_by_image(file: UploadFile = File(...), limit: int = Query(12, ge=1, le=60),
                                mode: str = Query("images", pattern="^(images|documents)$")):
    a = app_state()
    suffix = Path(file.filename or "query.png").suffix.lower() or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        while chunk := await file.read(1 << 20):
            tmp.write(chunk)
        tmp_path = Path(tmp.name)
    try:
        if mode == "documents":
            results = a.image_index.search_documents_by_image(tmp_path, limit)
        else:
            results = a.image_index.search_by_image(tmp_path, limit)
        return {"results": results, "capabilities": a.image_index.capabilities(),
                "debug": a.image_index.debug(
                    "image_to_document" if mode == "documents" else "image_to_image",
                    getattr(a.image_index, "last_query_vector", None),
                    candidates=len(results), limit=limit,
                    hits=results if mode != "documents" else [])}
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------- tables ----------------
@api.get("/api/tables")
def list_tables(doc_id: int | None = Query(default=None)):
    a = app_state()
    tables = a.store.list_tables([doc_id] if doc_id else None)
    return {"tables": [{**t, "rows": t["rows"][:200]} for t in tables]}


@api.get("/api/tables/{table_id}")
def get_table(table_id: int):
    t = app_state().store.get_table(table_id)
    if not t:
        raise HTTPException(404, "table not found")
    return t


@api.post("/api/tables/compute")
def table_compute(req: dict):
    """Deterministic arithmetic over the indexed tables (no model involved)."""
    a = app_state()
    res = a.answerer.table_answer(str(req.get("question", "")), req.get("doc_ids"), req.get("table_id"))
    if not res:
        return {"answered": False,
                "reason": "no table/column could be identified confidently for this question"}
    return {"answered": True, **res}


# ---------------- visualisation planner ----------------
#: One planner per workspace: the fact scan is the expensive part and it is cached inside.
_PLANNERS: dict[str, Planner] = {}
_PLANS: dict[str, dict] = {}


def _planner(a) -> Planner:
    key = f"{a.project_id}:{len(a.store.list_documents())}"
    planner = _PLANNERS.get(key)
    if planner is None:
        _PLANNERS.clear()
        planner = _PLANNERS[key] = Planner(a)
    return planner


@api.get("/api/visualize")
def visualize(q: str | None = Query(default=None, max_length=300),
              doc_ids: str | None = Query(default=None)):
    """A structured visualisation of what is indexed, chosen to fit the question.

    The knowledge graph is the representation underneath; this returns the *view* -- a
    timeline, a summary of labelled figures, a readable relationship graph, a table -
    with the document, page and wording behind every value. See :mod:`ragly_backend.visualize`.
    """
    a = app_state()
    ids = [int(x) for x in (doc_ids or "").split(",") if x.strip().isdigit()] or None
    key = f"{a.project_id}:{a.pack_key()}:{len(a.store.list_documents())}:{(q or '').strip().lower()}:{ids}"
    if key not in _PLANS:
        if len(_PLANS) > 32:
            _PLANS.clear()
        _PLANS[key] = _planner(a).plan(q, ids)
    return _PLANS[key]


# ---------------- entity graph ----------------
@api.get("/api/graph/nodes")
def graph_nodes(kind: str | None = Query(default=None), limit: int = Query(40, ge=1, le=300)):
    return {"nodes": app_state().store.top_nodes(kind, limit), "counts": app_state().store.graph_counts()}


@api.get("/api/graph/node/{node_id}")
def graph_node(node_id: int, hops: int = Query(1, ge=1, le=2)):
    return app_state().graph.expand(node_id, hops)


@api.get("/api/graph/summary")
def graph_summary(q: str = Query(min_length=1)):
    return app_state().graph.summary(q)


# ---------------- document comparison ----------------
class CompareRequest(BaseModel):
    left_id: int
    right_id: int


@api.post("/api/compare")
def compare(req: CompareRequest):
    try:
        return app_state().compare(req.left_id, req.right_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


# ---------------- benchmarks (measured, never fabricated) ----------------
@api.post("/api/benchmark")
def benchmark(iterations: int = Query(3, ge=1, le=20), include_llm: bool = Query(True)):
    """Run the same workloads on whatever backend is actually active and record the numbers."""
    a = app_state()
    results = []
    backend = a.inference.backend_label("text_embedding")
    text = ("The Ring Main Unit must be isolated before maintenance and the earth bolt "
            "torqued to 40 Nm according to the procedure. ") * 2
    r = bench("embed_text", a.embedder.model_id, backend,
              lambda: a.embedder.embed_documents([text]), iterations=iterations,
              units=a.embedder.count_tokens(text))
    results.append({**r.to_dict(), "unit": "tokens/s"})

    r = bench("embed_query", a.embedder.model_id, backend,
              lambda: a.embedder.embed_query("what is the earth bolt torque"), iterations=iterations)
    results.append({**r.to_dict(), "unit": ""})

    imgs = a.store.get_images(limit=1)
    if imgs and a.image_embedder.available():
        p = Path(imgs[0]["path"])
        r = bench("embed_image", a.image_embedder.model_id,
                  a.inference.backend_label("image_embedding"),
                  lambda: a.image_embedder.embed_image(p), iterations=max(1, iterations // 2), units=1)
        results.append({**r.to_dict(), "unit": "images/s"})
    if imgs and OCR.status()["backend"]:
        from .images import load_rgb

        arr = load_rgb(Path(imgs[0]["path"]))
        r = bench("ocr", OCR.status()["backend"], "CPU", lambda: OCR.image_to_text(arr),
                  iterations=max(1, iterations // 2), units=1)
        results.append({**r.to_dict(), "unit": "images/s"})

    if a.store.counts()["chunks"]:
        r = bench("retrieval", "hybrid BM25+vector+graph", "CPU",
                  lambda: a.retriever.search("earth bolt torque", 5), iterations=iterations)
        results.append({**r.to_dict(), "unit": ""})

    if include_llm and a.engine.state == "ready":
        llm = a.engine.require_client()
        msgs = [{"role": "user", "content": "Reply with the single word: ready"}]
        t = time.perf_counter()
        _, stats = llm.chat(msgs, max_tokens=8)
        results.append({"task": "llm_generate", "model": stats.get("model", ""),
                        "backend": "NPU (GenieX)" if a.engine.mode == "snapdragon" else "CPU (llama.cpp)",
                        "latency_ms": round((time.perf_counter() - t) * 1000, 1),
                        "throughput": stats.get("tokens_per_sec"), "memory_mb": None,
                        "iterations": 1, "unit": "tokens/s"})

    for r2 in results:
        a.store.record_benchmark(r2["task"], r2["model"], r2["backend"], r2.get("latency_ms"),
                                 r2.get("throughput"), r2.get("memory_mb"),
                                 {"iterations": r2.get("iterations"), "unit": r2.get("unit", "")})
    return {"results": results, "hardware": a.hardware.summary(), "mode": a.inference.mode,
            "note": "Every number here was measured on this machine just now."}


@api.get("/api/benchmark")
def benchmark_history(limit: int = Query(100, ge=1, le=500)):
    return {"runs": app_state().store.benchmarks(limit)}


# ---------------- RAG evaluation reports (written by scripts/rag_eval.py) ----------------
EVAL_DIR = ROOT / "eval"


def _eval_files() -> list[Path]:
    return sorted(EVAL_DIR.glob("rag_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


@api.get("/api/eval/reports")
def eval_reports():
    """Lists evaluation runs on disk. Empty until `python scripts/rag_eval.py` has been run."""
    out = []
    for f in _eval_files():
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        out.append({"file": f.name, "label": data.get("label"), "timestamp": data.get("timestamp"),
                    "model": data.get("model"), "questions": data.get("questions"),
                    "retrieval_only": data.get("retrieval_only", False),
                    "peak_ram_mb": data.get("peak_ram_mb"), "model_load_s": data.get("model_load_s"),
                    "config": data.get("config", {}), "hardware": data.get("hardware", {}),
                    "metrics": data.get("metrics", {})})
    return {"reports": out, "directory": str(EVAL_DIR),
            "command": "python scripts/rag_eval.py --ingest --label baseline"}


@api.get("/api/eval/report/{name}")
def eval_report(name: str):
    f = EVAL_DIR / name
    if f.suffix != ".json" or f.parent != EVAL_DIR or not f.exists():
        raise HTTPException(404, "No such evaluation report")
    return json.loads(f.read_text(encoding="utf-8"))
