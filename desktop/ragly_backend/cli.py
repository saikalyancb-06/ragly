"""Command line:  python -m ragly_backend.cli ingest <file-or-folder> | ask "question" | status"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .app import App
from .pipeline import ALL_SUPPORTED as SUPPORTED


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ragly")
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ingest", help="index a file or a folder")
    i.add_argument("path")
    q = sub.add_parser("ask", help="ask a question (starts the answer engine if needed)")
    q.add_argument("question")
    q.add_argument("--top-k", type=int, default=None)
    s = sub.add_parser("search", help="retrieval only, no LLM")
    s.add_argument("query")
    sub.add_parser("status")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    app = App(start_indexer=False)
    try:
        if args.cmd == "ingest":
            root = Path(args.path)
            files = [root] if root.is_file() else sorted(x for x in root.rglob("*") if x.suffix.lower() in SUPPORTED)
            for f in files:
                doc, created = app.indexer.add_file(f, f.name)
                if not created and doc["status"] == "ready":
                    print(f"skip (already indexed): {f.name}")
                    continue
                t = time.perf_counter()
                try:
                    app.indexer.index_document(doc["id"])
                except Exception as exc:
                    app.store.set_status(doc["id"], "error", error=str(exc))
                    print(f"ERROR {f.name}: {exc}")
                    continue
                d = app.store.get_document(doc["id"])
                print(f"indexed {f.name}: {d['pages']} pages, {d['chunk_count']} chunks, "
                      f"{d['ocr_pages']} OCR pages, {time.perf_counter() - t:.1f}s")
            print(app.store.counts())
        elif args.cmd == "search":
            hits, timing = app.retriever.search(args.query)
            for h in hits:
                print(f"{h.vector_score:.3f} kw={h.keyword_rank} {h.doc_name} p{h.page}: {h.text[:120]}")
            print(timing)
        elif args.cmd == "ask":
            app.engine.start()
            for ev in app.answerer.stream(app.engine.require_client(), args.question, args.top_k):
                if ev["type"] == "token":
                    print(ev["token"], end="", flush=True)
                elif ev["type"] == "done":
                    print("\n\nSources:")
                    for c in ev["citations"]:
                        print(f"  [{c['n']}] {c['doc_name']} p.{c['page']} (score {c['vector_score']})")
                    print("grounding:", ev["grounding"], "\ntiming:", ev["timing"])
        elif args.cmd == "status":
            print(app.engine.device, app.store.counts(), app.embedder.provider, sep="\n")
    finally:
        app.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
