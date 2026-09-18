# Ragly — offline multimodal knowledge engine

An offline multimodal knowledge engine, built for Qualcomm Snapdragon edge devices: it privately indexes
documents, images, tables and relationships, retrieves evidence with hybrid search, and uses local AI to answer
questions with verifiable citations. Not a "chat with PDF" wrapper.

**No cloud. No OpenAI/Gemini API. No cloud vector database. No internet needed after installation.**
The only network access in the whole project is `scripts/download_models.py` at setup time; at runtime an
in-process offline guard blocks every non-loopback socket.

```
                      ┌─ pages (text / scanned / table / image / diagram / mixed)
import ─► extract ────┼─ tables  (headers + rows preserved)
                      ├─ images  (extracted rasters, OCR, pHash, CLIP vectors)
                      └─ chunks  (text + table renderings) ─► SQLite: FTS5 + float32 vectors

ask ─► router ─► hybrid retrieval ─► evidence ─► local LLM ─► verification ─► answer + citations
        │          BM25 · dense cosine · exact codes · entity graph · metadata filter
        └─ table/numeric question ─► deterministic computation (no LLM arithmetic)
```

Retrieval is SQLite FTS5 (BM25) + NumPy cosine over float32 vectors, fused with reciprocal rank fusion —
no FAISS and no external vector store; below ~50k chunks this is faster and adds no dependency.

## Quick start (Windows)

```powershell
cd desktop
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1     # venv, packages, llama.cpp, models (~2.3 GB), tests
powershell -ExecutionPolicy Bypass -File scripts\run.ps1               # http://127.0.0.1:8765/docs
```

On the Snapdragon PC the same setup script picks the ARM64 llama.cpp build, installs the GenieX CLI and pulls
`ai-hub-models/Qwen3-4B-Instruct-2507`. Then:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run.ps1 -Mode snapdragon
```

Setup flags: `-Llm 1.5b` (smaller/faster model), `-SkipLlama`, `-SkipModels`, `-SkipGenieX`.
Python 3.10+ is required (on Snapdragon, the ARM64 installer from python.org).

## What makes it different

| Capability | How it works | Status |
|---|---|---|
| **Multimodal ingestion** | PDF, DOCX, TXT/MD, XLSX/CSV, JPG/PNG, scanned PDFs. Every page is classified (text / scanned_text / table / image / diagram / mixed) and stored as document → page → content → chunk | working |
| **Hybrid retrieval** | dense cosine + BM25 + exact code/part-number matching + entity-graph expansion + metadata filters, fused with configurable RRF weights | working |
| **Table-aware reasoning** | tables keep headers/rows; numeric questions are answered by computing locally and citing the table + page. The model never does arithmetic | working |
| **Entity/relationship graph** | local rule-based extraction (people, orgs, invoices, contracts, projects, amounts, dates, IDs) with relations (`issued_by`, `works_for`, `belongs_to`, `between`, …) in SQLite; used as a retrieval path | working |
| **Image search** | text→image, image→image, image→document. CLIP ViT-B/32 ONNX locally when installed; OCR text + perceptual hash otherwise, and the UI says which mode is live | working |
| **Evidence & verification** | structured evidence per answer + claim verification: every number, code, date and amount must exist in the retrieved text | working |
| **Document comparison** | section matching, then added/removed/modified with changed values (`60 days → 90 days`) and page references | working |
| **Snapdragon/NPU** | hardware abstraction (QNN → GPU → CPU) with real detection; int8 embeddings when an NPU is present; benchmarks measured, never fabricated | abstraction working; NPU claims only on real hardware |

## Honest capability reporting

- `GET /api/hardware` reports the CPU, RAM, GPUs and the ONNX Runtime providers actually available. On an Intel
  i5 with no accelerator it says *"Snapdragon/NPU: not detected · running in CPU compatibility mode"*.
- A backend is only ever labelled NPU/GPU when a live session reports that provider (`get_providers()[0]`).
- `GET /api/diagnostics` lists every registry model as installed or missing, the vision mode in use, OCR
  backend, offline-guard status and `cloud_api_calls: 0`.
- Image understanding is claimed only when the CLIP ONNX model is present; otherwise the mode is `ocr_phash`.

## Projects (user → project → documents)

Every document belongs to a **project**, and a project is a self-contained workspace: its own documents,
index, tables, images, entity graph, auto-tuning and answers. Switching projects re-scopes every query, so a
search in "Legal review" can never return a page from "Site manuals".

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/projects` | list projects with their document counts, and which is active |
| POST | `/api/projects` | create `{name, description}` and switch to it |
| POST | `/api/projects/{id}/activate` | switch the active project |
| PATCH/DELETE | `/api/projects/{id}` | rename, or delete the project and everything indexed in it |

The active project is remembered in `data/settings.json`, so the app reopens where you left off.

## Local passcode

A first run offers to set a passcode. It is stored as a PBKDF2-SHA256 hash (200k iterations) in
`data/auth.json`, sessions are signed with a key generated on the machine, and the UI locks behind it.
Being accurate about what this is: the backend already refuses any non-localhost connection, so the passcode
locks the app on this device — **it does not encrypt the database**, and the UI says so.
`/api/auth/status | register | login | check | change | disable`.

## The app

Start the backend and open **http://127.0.0.1:8765** — the React UI is served by the backend itself
(prebuilt in `ui/dist`, so the Snapdragon PC needs no Node.js).

- **Landing page** with a local-first pitch, then the passcode screen, then the workspace.
- **Sidebar:** project switcher, + Add Documents, and three groups — Workspace (Overview, Documents, Images,
  Tables), Intelligence (Ask, Photo Search, Compare, Knowledge Graph), System (Edge AI, Benchmarks, Models,
  Settings) — with a permanent ● LOCAL PROCESSING indicator.
- **Left:** drag in documents, watch indexing, and see what auto-tuning learned.
- **Middle:** ask by typing or by voice; citations `[1]` are clickable; every answer shows a verdict line
  (verified / blocked, match score, search ms, tokens/sec, time to first word).
- **Right:** the cited page as an image with the passage highlighted, plus retrieval sliders.
- **Header:** Normal ↔ Snapdragon engine toggle, auto-tuning on/off, strict grounding on/off.

To develop the UI with hot reload: `cd ui && npm install && npm run dev` (proxies /api to the backend).
To rebuild what the backend serves: `npm run build`.

## Auto-tuning (no manual configuration)

On import Ragly reads the documents and builds a "pack" (`GET /api/pack`):

| Learned | How it is used |
|---|---|
| Glossary — "Ring Main Unit (RMU)", frequent acronyms | query expansion: asking about "RMU" also searches "Ring Main Unit" |
| Codes, part numbers, measurements, money, dates, clause refs | stored in an `entities` table; an exact code hit ("E-47") outranks fuzzy similarity |
| Document type (manual / contract / medical report / SOP) | domain rules added to the prompt |
| Structure (numbered steps, clauses, value tables) | answer format: steps, clauses, values or prose |
| Code density | keyword vs vector weighting in the hybrid search |
| Headings + codes | suggested one-tap questions |

`POST /api/pack/retune` rebuilds it. Every ask accepts `"tuning": false` to run plain RAG for comparison.

### Accuracy test (tuning off vs on)

```powershell
.venv\Scripts\python scripts\make_manual_samples.py          # demo corpus of field manuals
.venv\Scripts\python scripts\eval.py --ingest --retrieval-only
.venv\Scripts\python scripts\eval.py                          # full: checks generated answers too
```

21 questions in `eval/questions.json`, including two that must be refused. Strict scoring (the top hit must be
the right document *and* contain the evidence) measured here: **tuning off 13/21 (62%) → on 15/21 (71%)**.
A report is written to `eval/report.json`.

## Strict grounding (no hallucinations)

Four layers, all on by default (`RAGLY_STRICT=0` or `POST /api/settings {"strict_grounding": false}` to disable):

1. **Confidence gate** — if the best match is below the threshold, Ragly answers "Not found in your documents."
   without calling the model.
2. **Prompt** — answer only from the numbered sources, cite each fact, never estimate or use outside knowledge.
3. **Claim verification** — every number, code, date, amount and duration in the answer must appear in the
   retrieved text. Anything invented is listed in `grounding.unsupported_values` and the answer is refused.
4. **Citation repair** — each sentence is matched to the source that contains its facts and cited automatically;
   a sentence whose facts appear in no source is dropped (`grounding.dropped_sentences`), and hedging words
   ("typically", "usually", "approximately") are rejected (`grounding.flag = hedged_language`).

`grounding.verified = true` means: citations valid, all values traced back to the documents.

## Voice (push-to-talk, offline)

Windows' built-in speech engines, so nothing is downloaded and it works on x64 and ARM64:

| Endpoint | Purpose |
|---|---|
| `GET /api/voice` | availability (`stt`, `tts`) |
| `POST /api/voice/listen` | record from the mic and return the recognised text |
| `POST /api/voice/ask` | listen → answer → `speech_text` for playback |
| `POST /api/voice/speak` | text → WAV audio |

Recognised speech is repaired using the auto-tuned glossary: "error e forty seven" → "E-47",
"are em you" → "RMU". Answers spoken aloud put safety warnings first and always name the source page.
Needs microphone permission in Windows Settings → Privacy → Microphone.

## Engine toggle (Normal ↔ Snapdragon)

| | Normal | Snapdragon |
|---|---|---|
| Server | `llama-server` (started by the backend) | `geniex serve` (started by the backend) |
| URL | `127.0.0.1:8080/v1` | `127.0.0.1:18181/v1` |
| Model | `models/qwen2.5-3b-instruct-q4_k_m.gguf` | `ai-hub-models/Qwen3-4B-Instruct-2507` (NPU) |

Switch at runtime: `POST /api/engine {"mode":"snapdragon"}`. The backend stops one server, starts the other,
waits for `/v1/models`, and rolls back to the previous engine if the switch fails. Snapdragon mode is only
offered when a Snapdragon CPU is detected (`RAGLY_FORCE_SNAPDRAGON=1` overrides). Edit `engines.json` to change
models, ports or the GenieX command (`geniex serve -h`). If a server is already running on the URL, the backend
just uses it. The document index is shared by both engines; no re-indexing on switch.

Optional NPU embeddings on Snapdragon: `pip uninstall onnxruntime; pip install onnxruntime-qnn` — the embedder
then tries `QNNExecutionProvider` and falls back to CPU (see `/api/system` → `embedder.provider`).

## API (all under http://127.0.0.1:8765)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | liveness + engine state |
| GET | `/api/system` | device, engine, embedder, index counts, indexing progress, offline-guard status |
| GET/POST | `/api/engine` | read / switch engine `{"mode": "normal" \| "snapdragon"}` |
| POST | `/api/engine/restart` | restart the current engine |
| POST | `/api/documents` | multipart upload (`files` field, many files) → queued for background indexing |
| POST | `/api/documents/from-path` | `{"path": "C:\\Docs"}` index a local file or folder |
| GET | `/api/documents` | list with status `queued / indexing / ready / error` |
| GET/DELETE | `/api/documents/{id}` | details / delete (removes chunks, vectors and the stored copy) |
| POST | `/api/documents/{id}/reindex` | re-index one document |
| GET | `/api/documents/{id}/file` | original file |
| GET | `/api/documents/{id}/pages/{n}?chunk_id=` | PNG of the page with the cited passage highlighted |
| GET | `/api/search?q=&tuning=` | retrieval only (debug) |
| GET/POST | `/api/pack`, `/api/pack/retune` | what auto-tuning learned / rebuild it |
| GET | `/api/entities?kind=code` | every code, part number or measurement found, with page |
| GET | `/api/suggestions` | one-tap questions for these documents |
| GET/POST | `/api/settings` | top_k, min_score, max_tokens, strict_grounding |
| GET | `/api/voice`, `/api/voice/*` | see the Voice section |
| POST | `/api/ask` | `{"question", "top_k"?, "doc_ids"?}` → answer, citations, sources, grounding, timing |
| POST | `/api/ask/stream` | same, as SSE: `route` → `sources` → `token`… → `done` (or `error`) |
| GET | `/api/hardware` | detected device, providers, selected/actual backends, model registry |
| GET | `/api/diagnostics` | the full local-first diagnostics page payload |
| POST | `/api/route` | how a question would be routed, and why |
| GET | `/api/images`, `/api/images/{id}/file` | indexed images |
| POST | `/api/images/search` | text → image search |
| POST | `/api/images/search-by-image?mode=images\|documents` | image → image / image → document |
| GET | `/api/tables`, `/api/tables/{id}` | extracted tables with headers and rows |
| POST | `/api/tables/compute` | deterministic arithmetic over tables |
| GET | `/api/graph/nodes`, `/api/graph/node/{id}`, `/api/graph/summary?q=` | entity graph |
| POST | `/api/compare` | `{left_id, right_id}` → what changed, with pages |
| GET/POST | `/api/benchmark` | run/read measured benchmarks per backend |

`done` / `/api/ask` payload: `answer`, `citations[]` (doc, page, chunk_id, snippet, scores), `cited_numbers`,
`dropped_citations`, `refused`, `grounding {best_score, min_score, flag}`, `timing` (embed/search ms, LLM
time-to-first-token, tokens/sec), `model`.

## How answers stay grounded

1. Retrieval confidence gate: if the best cosine score < `RAGLY_MIN_SCORE` (0.45; 0.40 when keyword search also
   matched) the backend answers **"Not found in your documents."** without calling the LLM.
2. The prompt allows only the numbered sources and requires `[n]` after each fact.
3. Citation numbers that don't exist are removed and reported in `dropped_citations`; answers with no citation
   are flagged `uncited`; model refusals are normalised.

## Offline / privacy guarantees

- The server binds to 127.0.0.1 only (refuses to start on other hosts).
- An in-process **offline guard** blocks every non-loopback socket connection; the LLM URL must be localhost.
- `HF_HUB_OFFLINE=1` at runtime; all models load from `models/`.
- Uploaded files are copied into `data/files`, the index lives in `data/ragly.db`. Delete `data/` to wipe everything.

## OCR (scanned pages and images)

- **Windows (x64 and Snapdragon ARM64):** the built-in Windows OCR engine (`Windows.Media.Ocr`) via the `winrt-*`
  packages. Offline, no model download, any Python 3.9+.
- **Linux/macOS dev:** RapidOCR (Python ≤ 3.12).
- `GET /api/system` → `ocr.backend` shows which one loaded. Force one with `RAGLY_OCR=windows|rapidocr|off`.
- Missing English OCR pack on Windows (admin PowerShell):
  `Add-WindowsCapability -Online -Name "Language.OCR~~~en-US~0.0.1.0"`

## Self-test

```powershell
.venv\Scripts\python scripts\selftest.py            # 60+ checks, ~1 minute, uses a throwaway data folder
.venv\Scripts\python scripts\selftest.py --with-llm # also starts the answer model and asks real questions
```

It exercises every capability through the real API — ingestion of each format, OCR, content-type
classification, auto-tuning, hybrid search, exact-code lookup, graph retrieval, the router, table arithmetic
and its refusals, image search in both directions, comparison, page rendering, settings, verification
(including an invented number being blocked), voice reporting, benchmarks, project isolation and the
passcode — and prints PASS/FAIL per check.

## CLI

```powershell
.venv\Scripts\python -m ragly_backend.cli ingest samples
.venv\Scripts\python -m ragly_backend.cli search "termination notice"
.venv\Scripts\python -m ragly_backend.cli ask "What is the termination notice period?"
```

## Settings (env vars)

`RAGLY_PORT` (8765) · `RAGLY_DATA` · `RAGLY_ENGINES` · `RAGLY_TOP_K` (5) · `RAGLY_MIN_SCORE` (0.45) ·
`RAGLY_CHUNK_TOKENS` (380) · `RAGLY_CHUNK_OVERLAP` (60) · `RAGLY_MAX_TOKENS` (512) · `RAGLY_OFFLINE_GUARD` (1) ·
`RAGLY_STRICT` (1) · `RAGLY_TEMPERATURE` (0.0) · `RAGLY_OCR` (auto) · `RAGLY_FORCE_SNAPDRAGON`.

## Layout

```
desktop/
  ragly_backend/  config · offline · ingest · chunker · embedder · store · retriever · autotune · llm · engine · answer · voice · indexer · app · api · cli
  ui/             React + Vite app (src/) and the prebuilt dist/ the backend serves
  eval/           questions.json + report.json (tuning off vs on)
  scripts/        setup_windows.ps1 · run.ps1 · run.sh · download_models.py · make_samples.py · make_manual_samples.py · eval.py
  samples/        demo docs (contract, lab report, scanned note, notes) + manuals/ (field-maintenance corpus)
  tests/          pytest (unit + end-to-end with a stubbed LLM)
  engines.json    engine toggle config
  models/  bin/  data/   created by setup / at runtime (not in git)
```

Tested here: Python 3.11, Qwen2.5-3B Q4_K_M on CPU — all sample questions answered with correct page citations,
unrelated questions refused.
