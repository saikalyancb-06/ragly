"""RAG evaluation harness.

Runs a fixed question set against the live pipeline and measures where the system
actually fails: retrieval, generation, or grounding. Nothing here is estimated --
every number in the report comes from a real run on this machine.

    python scripts/rag_eval.py --ingest --label baseline
    python scripts/rag_eval.py --retrieval-only          # fast, no LLM needed
    python scripts/rag_eval.py --compare eval/rag_baseline.json eval/rag_after.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VALUE_CATEGORIES = {"numerical", "table", "date"}


# ----------------------------------------------------------------- helpers
def norm(text: str) -> str:
    """Lower-case, collapse whitespace, drop thousands separators inside numbers."""
    t = (text or "").lower().replace("–", "-").replace("—", "-")
    t = re.sub(r"(?<=\d),(?=\d)", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def contains(answer: str, needle: str) -> bool:
    return norm(needle) in norm(answer)


def page_of(src: dict) -> int:
    try:
        return int(src.get("page") or 0)
    except (TypeError, ValueError):
        return 0


def matches(expected: dict, doc_name: str, page: int, page_slack: int = 1) -> bool:
    """`document` may be one name or several (the same table indexed as .xlsx and .csv, say);
    `page` may be one page, several, or 0 for "any page of that document"."""
    docs = expected["document"]
    docs = [docs] if isinstance(docs, str) else list(docs)
    if not any(norm(d) == norm(doc_name) for d in docs):
        return False
    want = expected.get("page") or 0
    wants = [want] if isinstance(want, int) else list(want)
    return 0 in wants or any(abs(int(w) - page) <= page_slack for w in wants)


class RamSampler(threading.Thread):
    """Samples RSS of this process and every child (the llama.cpp server included)."""

    def __init__(self, interval: float = 0.25):
        super().__init__(daemon=True)
        self.interval, self.peak_mb, self._flag = interval, 0.0, threading.Event()

    def run(self) -> None:
        try:
            import psutil
        except ImportError:
            return
        me = psutil.Process()
        while not self._flag.is_set():
            total = 0
            try:
                total = me.memory_info().rss
                for child in me.children(recursive=True):
                    try:
                        total += child.memory_info().rss
                    except psutil.Error:
                        pass
            except psutil.Error:
                pass
            self.peak_mb = max(self.peak_mb, total / (1024 * 1024))
            self._flag.wait(self.interval)

    def stop(self) -> float:
        self._flag.set()
        self.join(timeout=2)
        return round(self.peak_mb, 1)


def pct(num: int, den: int) -> float:
    return round(100.0 * num / den, 1) if den else 0.0


# ----------------------------------------------------------------- ingestion
def ingest(app, spec: dict) -> list[str]:
    from ragly_backend.pipeline import ALL_SUPPORTED as SUPPORTED

    corpora = spec.get("corpus", "samples")
    if isinstance(corpora, str):
        corpora = [corpora]
    indexed = []
    for rel in corpora:
        folder = ROOT / rel
        for f in sorted(folder.rglob("*")):
            if f.suffix.lower() not in SUPPORTED:
                continue
            doc, created = app.indexer.add_file(f, f.name)
            if created or doc["status"] != "ready":
                app.indexer.index_document(doc["id"])
                indexed.append(f.name)
    if indexed:
        app.retune()
        app.sync_pack()
    return indexed


# ----------------------------------------------------------------- per question
def score_retrieval(case: dict, hits: list) -> dict:
    expected = case.get("expected_sources") or []
    found, ranks = [], []
    for exp in expected:
        for i, h in enumerate(hits):
            if matches(exp, h.doc_name, h.page):
                found.append(exp)
                ranks.append(i + 1)
                break
    relevant_hits = sum(1 for h in hits if any(matches(e, h.doc_name, h.page) for e in expected))
    best = max((h.vector_score for h in hits), default=0.0)
    return {
        "retrieved": [{"doc": h.doc_name, "page": h.page, "vector_score": round(h.vector_score, 4),
                       "rerank_score": round(getattr(h, "rerank_score", 0.0), 4)} for h in hits],
        "expected_found": len(found),
        "expected_total": len(expected),
        "recall": (len(found) / len(expected)) if expected else None,
        "precision": (relevant_hits / len(hits)) if hits and expected else None,
        "first_rank": min(ranks) if ranks else None,
        "top1_correct": bool(hits and expected and any(matches(e, hits[0].doc_name, hits[0].page) for e in expected)),
        "best_vector_score": round(best, 4),
    }


def score_answer(case: dict, result: dict, retrieval: dict) -> dict:
    answer = result.get("answer") or ""
    grounding = result.get("grounding") or {}
    citations = result.get("citations") or []
    refused = bool(result.get("refused")) or grounding.get("flag") == "insufficient_evidence"
    answerable = bool(case.get("answerable"))
    must = case.get("must_include") or []
    missing = [m for m in must if not contains(answer, m)]

    if answerable:
        correct = (not refused) and not missing
    else:
        correct = refused

    cited_docs = [c.get("doc_name", "") for c in citations]
    expected = case.get("expected_sources") or []
    if answerable and not refused:
        citation_correct = bool(citations) and any(
            any(matches(e, c.get("doc_name", ""), page_of(c)) for e in expected) for c in citations)
    else:
        citation_correct = None    # not applicable

    unsupported = grounding.get("unsupported_values") or []
    grounded = bool(grounding.get("verified")) if (answerable and not refused) else None

    # ---- failure attribution
    retrieval_failure = generation_failure = grounding_failure = False
    if not correct:
        if answerable:
            got_evidence = (retrieval.get("recall") or 0) > 0
            if not got_evidence:
                retrieval_failure = True
            elif refused:
                grounding_failure = True          # evidence was there, the gate refused it
            elif missing:
                generation_failure = True         # evidence was there, the model did not use it
            else:
                grounding_failure = True
        else:
            grounding_failure = True              # answered a question the corpus cannot support
            if retrieval.get("best_vector_score", 0) >= 0.45:
                retrieval_failure = True          # a confident distractor helped it along
    elif answerable and not refused and (unsupported or citation_correct is False):
        grounding_failure = True                  # right text, wrong support

    timing = result.get("timing") or {}
    return {
        "answer": answer,
        "refused": refused,
        "correct": correct,
        "missing_phrases": missing,
        "citations": cited_docs,
        "citation_correct": citation_correct,
        "grounded": grounded,
        "unsupported_values": unsupported,
        "dropped_sentences": grounding.get("dropped_sentences") or [],
        "flag": grounding.get("flag"),
        "sufficiency": {k: v for k, v in (grounding.get("sufficiency") or {}).items()
                        if k in ("sufficient", "score", "missing", "reasons")},
        "retrieval_failure": retrieval_failure,
        "generation_failure": generation_failure,
        "grounding_failure": grounding_failure,
        "latency_ms": timing.get("total_ms"),
        "retrieval_ms": timing.get("search_ms") or timing.get("total_ms"),
        "llm_ms": timing.get("llm_elapsed_ms") or timing.get("llm_ms"),
        "completion_tokens": timing.get("llm_completion_tokens"),
    }


# ----------------------------------------------------------------- aggregation
def aggregate(rows: list[dict], retrieval_only: bool) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]

    recalls = [r["retrieval"]["recall"] for r in answerable if r["retrieval"]["recall"] is not None]
    precisions = [r["retrieval"]["precision"] for r in answerable if r["retrieval"]["precision"] is not None]
    top1 = [r for r in answerable if r["retrieval"]["top1_correct"]]

    out = {
        "questions": len(rows),
        "retrieval_recall_pct": round(100 * statistics.fmean(recalls), 1) if recalls else 0.0,
        "retrieval_precision_pct": round(100 * statistics.fmean(precisions), 1) if precisions else 0.0,
        "retrieval_top1_pct": pct(len(top1), len(answerable)),
        "retrieval_full_hit_pct": pct(sum(1 for r in answerable if r["retrieval"]["recall"] == 1.0), len(answerable)),
    }
    if retrieval_only:
        return out

    correct = [r for r in rows if r["answer"]["correct"]]
    answered = [r for r in answerable if not r["answer"]["refused"]]
    cit_applicable = [r for r in answered if r["answer"]["citation_correct"] is not None]
    grounded_applicable = [r for r in answered if r["answer"]["grounded"] is not None]
    value_rows = [r for r in answerable if r["category"] in VALUE_CATEGORIES]
    multi_rows = [r for r in rows if r["category"] == "multi_document"]
    latencies = [r["answer"]["latency_ms"] for r in rows if r["answer"].get("latency_ms")]

    out.update({
        "answer_correct_pct": pct(len(correct), len(rows)),
        "answerable_correct_pct": pct(sum(1 for r in answerable if r["answer"]["correct"]), len(answerable)),
        "abstention_accuracy_pct": pct(sum(1 for r in unanswerable if r["answer"]["refused"]), len(unanswerable)),
        "over_refusal_pct": pct(sum(1 for r in answerable if r["answer"]["refused"]), len(answerable)),
        "hallucination_pct": pct(sum(1 for r in unanswerable if not r["answer"]["refused"]), len(unanswerable)),
        "citation_correct_pct": pct(sum(1 for r in cit_applicable if r["answer"]["citation_correct"]), len(cit_applicable)),
        "groundedness_pct": pct(sum(1 for r in grounded_applicable if r["answer"]["grounded"]), len(grounded_applicable)),
        "unsupported_claim_rate_pct": pct(sum(1 for r in answered if r["answer"]["unsupported_values"]), len(answered)),
        "numerical_accuracy_pct": pct(sum(1 for r in value_rows if r["answer"]["correct"]), len(value_rows)),
        "multi_document_accuracy_pct": pct(sum(1 for r in multi_rows if r["answer"]["correct"]), len(multi_rows)),
        "latency_p50_ms": round(statistics.median(latencies), 1) if latencies else None,
        "latency_p95_ms": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 1) if latencies else None,
        "latency_mean_ms": round(statistics.fmean(latencies), 1) if latencies else None,
        "failures": {
            "retrieval": sum(1 for r in rows if r["answer"]["retrieval_failure"]),
            "generation": sum(1 for r in rows if r["answer"]["generation_failure"]),
            "grounding": sum(1 for r in rows if r["answer"]["grounding_failure"]),
        },
    })
    by_cat: dict[str, dict] = {}
    for r in rows:
        c = by_cat.setdefault(r["category"], {"n": 0, "correct": 0})
        c["n"] += 1
        c["correct"] += 1 if r["answer"]["correct"] else 0
    out["by_category"] = {k: {**v, "pct": pct(v["correct"], v["n"])} for k, v in sorted(by_cat.items())}
    return out


def print_report(report: dict, retrieval_only: bool) -> None:
    a = report["metrics"]
    print("\n" + "=" * 78)
    print(f"RAG EVALUATION — {report['label']}   ({report['questions']} questions)")
    print("=" * 78)
    print(f"Model:        {report.get('model') or 'not loaded'}")
    print(f"Backend:      {report['hardware']['backend']}")
    print(f"Model load:   {report.get('model_load_s', 'n/a')} s")
    print(f"Peak RAM:     {report.get('peak_ram_mb', 'n/a')} MB")
    print("-" * 78)
    print("RETRIEVAL")
    print(f"  Recall              {a['retrieval_recall_pct']:6.1f}%")
    print(f"  Precision           {a['retrieval_precision_pct']:6.1f}%")
    print(f"  Top-1 correct       {a['retrieval_top1_pct']:6.1f}%")
    print(f"  All evidence found  {a['retrieval_full_hit_pct']:6.1f}%")
    if retrieval_only:
        print("=" * 78)
        return
    print("ANSWERS")
    print(f"  Overall correct     {a['answer_correct_pct']:6.1f}%")
    print(f"  Answerable correct  {a['answerable_correct_pct']:6.1f}%")
    print(f"  Numerical accuracy  {a['numerical_accuracy_pct']:6.1f}%")
    print(f"  Multi-document      {a['multi_document_accuracy_pct']:6.1f}%")
    print("GROUNDING")
    print(f"  Citation correct    {a['citation_correct_pct']:6.1f}%")
    print(f"  Groundedness        {a['groundedness_pct']:6.1f}%")
    print(f"  Unsupported claims  {a['unsupported_claim_rate_pct']:6.1f}%")
    print(f"  Abstention accuracy {a['abstention_accuracy_pct']:6.1f}%")
    print(f"  Hallucinated        {a['hallucination_pct']:6.1f}%")
    print(f"  Over-refusal        {a['over_refusal_pct']:6.1f}%")
    print("LATENCY")
    print(f"  p50 {a['latency_p50_ms']} ms · mean {a['latency_mean_ms']} ms · p95 {a['latency_p95_ms']} ms")
    print("FAILURE ATTRIBUTION")
    f = a["failures"]
    print(f"  retrieval {f['retrieval']} · generation {f['generation']} · grounding {f['grounding']}")
    print("-" * 78)
    print("BY CATEGORY")
    for cat, v in a["by_category"].items():
        print(f"  {cat:<18} {v['correct']}/{v['n']}  {v['pct']:5.1f}%")
    bad = [r for r in report["rows"] if not r["answer"].get("correct", True)]
    if bad:
        print("-" * 78)
        print("FAILED QUESTIONS")
        for r in bad:
            kinds = [k for k in ("retrieval", "generation", "grounding") if r["answer"][f"{k}_failure"]]
            print(f"  [{r['id']}] {r['question']}")
            print(f"      cause: {', '.join(kinds) or 'unclassified'}")
            if r["answer"].get("missing_phrases"):
                print(f"      missing: {r['answer']['missing_phrases']}")
            print(f"      got: {(r['answer'].get('answer') or '')[:160]}")
    print("=" * 78 + "\n")


def compare(old_path: Path, new_path: Path) -> int:
    old = json.loads(old_path.read_text(encoding="utf-8"))
    new = json.loads(new_path.read_text(encoding="utf-8"))
    keys = [k for k in new["metrics"] if k.endswith("_pct") or k.endswith("_ms")]
    print(f"\n{'metric':<28}{old['label']:>14}{new['label']:>14}{'delta':>10}")
    print("-" * 66)
    for k in keys:
        a, b = old["metrics"].get(k), new["metrics"].get(k)
        if a is None or b is None:
            continue
        print(f"{k:<28}{a:>14.1f}{b:>14.1f}{b - a:>+10.1f}")
    print()
    return 0


# ----------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "tests" / "rag_eval" / "questions.json"))
    ap.add_argument("--out", default="")
    ap.add_argument("--label", default="run")
    ap.add_argument("--ingest", action="store_true", help="index the corpus first")
    ap.add_argument("--retrieval-only", action="store_true", help="skip the LLM")
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--compare", nargs=2, metavar=("OLD", "NEW"))
    ap.add_argument("--only", default="", help="comma-separated question ids or categories")
    args = ap.parse_args()

    if args.compare:
        return compare(Path(args.compare[0]), Path(args.compare[1]))

    spec = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    cases = spec["questions"]
    if args.only:
        wanted = {w.strip() for w in args.only.split(",") if w.strip()}
        cases = [c for c in cases if c["id"] in wanted or c["category"] in wanted]

    from ragly_backend.app import App
    from ragly_backend.config import settings

    top_k = args.top_k or settings.top_k
    ram = RamSampler()
    ram.start()

    app = App(start_indexer=False)
    if args.ingest:
        added = ingest(app, spec)
        print(f"indexed {len(added)} file(s)" if added else "corpus already indexed")

    llm, model_load_s, model_name = None, None, None
    if not args.retrieval_only:
        t0 = time.perf_counter()
        app.engine.start()
        llm = app.engine.require_client()
        model_name = llm.model
        model_load_s = round(time.perf_counter() - t0, 2)
        print(f"engine ready in {model_load_s}s ({model_name})")

    rows = []
    for case in cases:
        hits, rtiming = app.retriever.search(case["question"], top_k=top_k)
        retrieval = score_retrieval(case, hits)
        retrieval["search_ms"] = rtiming.get("total_ms") or rtiming.get("search_ms")
        if llm is None:
            answer_row = {"correct": None, "retrieval_failure": retrieval.get("recall") == 0,
                          "generation_failure": False, "grounding_failure": False}
        else:
            try:
                result = app.answerer.ask(llm, case["question"], top_k=top_k)
            except Exception as exc:                      # the engine died mid-run: record it, keep going
                result = {"answer": "", "refused": False, "grounding": {"flag": "engine_error"},
                          "citations": [], "timing": {}, "error": str(exc)}
            answer_row = score_answer(case, result, retrieval)
            if result.get("error"):
                answer_row["error"] = result["error"]
        rows.append({"id": case["id"], "category": case["category"], "question": case["question"],
                     "answerable": bool(case.get("answerable")), "expected_sources": case.get("expected_sources") or [],
                     "retrieval": retrieval, "answer": answer_row})
        mark = "·" if llm is None else ("PASS" if answer_row["correct"] else "FAIL")
        print(f"  {mark:>4}  [{case['id']}] {case['question'][:64]}")

    peak = ram.stop()
    docs = app.store.list_documents()
    report = {
        "label": args.label,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "questions": len(rows),
        "retrieval_only": args.retrieval_only,
        "model": model_name,
        "model_load_s": model_load_s,
        "peak_ram_mb": peak,
        "documents_indexed": len(docs),
        "config": {"top_k": top_k, "pool_factor": settings.pool_factor, "min_score": settings.min_score,
                   "strict_grounding": settings.strict_grounding,
                   "require_sufficient_evidence": settings.require_sufficient_evidence,
                   "chunk_tokens": settings.chunk_tokens, "embed_model": settings.embed_model_id},
        "hardware": {"summary": app.hardware.summary(), "backend": app.inference.backend_label("llm"),
                     "embedding_backend": app.inference.backend_label("text_embedding"),
                     "platform": platform.platform(), "python": platform.python_version()},
        "metrics": aggregate(rows, args.retrieval_only),
        "rows": rows,
    }
    out = Path(args.out) if args.out else ROOT / "eval" / f"rag_{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_report(report, args.retrieval_only)
    print(f"report: {out}")
    app.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
