"""Accuracy test: auto-tuning OFF vs ON, on the same documents and questions.

    python scripts/eval.py --retrieval-only     # fast: did we find the right page?
    python scripts/eval.py                      # full: also checks the generated answer
    python scripts/eval.py --ingest             # index eval/questions.json's corpus first
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ragly_backend.app import App  # noqa: E402
from ragly_backend.config import NOT_FOUND  # noqa: E402
from ragly_backend.pipeline import ALL_SUPPORTED as SUPPORTED  # noqa: E402


def load_questions(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def retrieval_ok(hits, case, strict: bool = True) -> bool:
    """Strict: the FIRST hit must be the right document and contain the evidence."""
    if case.get("expect_refusal"):
        return not hits or max(h.vector_score for h in hits) < 0.45
    if not hits:
        return False
    if strict:
        h = hits[0]
        return h.doc_name == case["doc"] and all(s.lower() in h.text.lower() for s in case.get("must_include", []))
    return any(h.doc_name == case["doc"] for h in hits[:3])


def answer_ok(result, case) -> bool:
    text = (result.get("answer") or "").lower()
    if case.get("expect_refusal"):
        return result.get("refused") is True
    if result.get("refused"):
        return False
    if not any(c["doc_name"] == case["doc"] for c in result.get("citations", [])):
        return False
    return all(s.lower() in text for s in case.get("must_include", []))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "eval" / "questions.json"))
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--loose", action="store_true", help="count a hit anywhere in the top 3")
    ap.add_argument("--ingest", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "eval" / "report.json"))
    args = ap.parse_args()

    spec = load_questions(Path(args.questions))
    app = App(start_indexer=False)
    try:
        if args.ingest:
            corpus = ROOT / spec.get("corpus", "samples")
            files = [f for f in sorted(corpus.rglob("*")) if f.suffix.lower() in SUPPORTED]
            for f in files:
                doc, created = app.indexer.add_file(f, f.name)
                if created or doc["status"] != "ready":
                    app.indexer.index_document(doc["id"])
                    print("indexed", f.name)
            app.retune()
        if app.indexer.pack:
            print("Auto-tuned pack:", app.indexer.pack.summary())

        llm = None
        if not args.retrieval_only:
            app.engine.start()
            llm = app.engine.require_client()

        rows, totals = [], {"off": 0, "on": 0}
        for case in spec["questions"]:
            row = {"q": case["q"]}
            for mode, tuning in (("off", False), ("on", True)):
                t0 = time.perf_counter()
                if args.retrieval_only:
                    hits, _ = app.retriever.search(case["q"], tuning=tuning)
                    ok = retrieval_ok(hits, case, strict=not args.loose)
                    row[f"{mode}_top"] = hits[0].doc_name if hits else "-"
                else:
                    res = app.answerer.ask(llm, case["q"], tuning=tuning)
                    ok = answer_ok(res, case)
                    row[f"{mode}_answer"] = (res.get("answer") or "")[:120]
                row[f"{mode}_ok"] = ok
                row[f"{mode}_ms"] = round((time.perf_counter() - t0) * 1000)
                totals[mode] += int(ok)
            rows.append(row)
            mark = {True: "PASS", False: "FAIL"}
            print(f"{mark[row['off_ok']]:>4} -> {mark[row['on_ok']]:<4} | {case['q'][:64]}")

        n = len(spec["questions"])
        mode_name = "retrieval" if args.retrieval_only else "answers"
        print(f"\n{mode_name}: tuning OFF {totals['off']}/{n} ({totals['off'] / n:.0%})   "
              f"tuning ON {totals['on']}/{n} ({totals['on'] / n:.0%})")
        report = {"mode": mode_name, "total": n, "tuning_off": totals["off"], "tuning_on": totals["on"],
                  "pack": app.indexer.pack.to_dict() if app.indexer.pack else None, "rows": rows}
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("report written to", args.out)
        return 0
    finally:
        app.shutdown()


if __name__ == "__main__":
    sys.exit(main())
