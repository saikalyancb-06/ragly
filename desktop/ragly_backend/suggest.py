"""Recommended questions, built from what was actually indexed.

    documents indexed
        -> one "Summarize <document>" per document, scoped to that document
        -> "Summarize all the documents" as well when there is more than one

Only summaries are offered. Everything else the recommender can build -- table totals,
entity lookups, term questions -- is still available through :func:`candidates`, and the
evidence check in :func:`answerable` still gates them, but they are not shown: a suggested
question that is merely answerable is not necessarily a question anyone wanted asked.

Nothing here is a fixed list: a gardening PDF proposes gardening questions because the
candidates are made out of that PDF's own tables, headings, codes and names. A candidate is
only offered after the same evidence check the Ask page uses has agreed there is something
to answer it with, so a recommendation is never a dead end.
"""
from __future__ import annotations

import logging
import re

from .evidence import assess

log = logging.getLogger("ragly.suggest")

#: How much each kind of question is worth. A table total is computed in Python and always
#: verifiable, a named entity is specific, a heading is broad, a bare term is the weakest.
WEIGHT = {"table": 5.0, "entity": 4.0, "code": 3.5, "heading": 2.5, "term": 1.5, "image": 3.0}

_META = {"corpus", "document", "documents", "page", "pages", "test", "tests", "query", "queries",
         "expected", "behavior", "behaviour", "answer", "answers", "abstain", "abstention",
         "retrieval", "synthetic", "reference", "ground", "truth", "section", "sections", "fictional"}

_PERSON_TITLE = re.compile(r"^(mr|mrs|ms|dr|shri|smt|prof)\.?\s", re.I)

#: A code worth asking about looks like a code: letters and digits joined by a separator
#: (NS-2048, INV-1003, P/N 88123-A), or a long alphanumeric run. "I34" out of an OCR'd
#: "₹34,869" does not qualify, and a question about it could only ever be a dead end.
_REAL_CODE = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9]{1,6}[-/][A-Za-z0-9-]{2,}$|"
                        r"^[A-Za-z]{2,}\d{3,}$")


def looks_like_a_code(name: str) -> bool:
    return bool(_REAL_CODE.match(name.strip()))


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip(" .,:;")


def candidates(app) -> list[tuple[float, str, str]]:
    """Every question this corpus suggests, as (weight, kind, question)."""
    out: list[tuple[float, str, str]] = []
    seen: set[str] = set()

    def add(kind: str, question: str, bonus: float = 0.0) -> None:
        q = _clean(question)
        key = q.lower()
        if q and key not in seen and len(q) < 90:
            seen.add(key)
            out.append((WEIGHT.get(kind, 1.0) + bonus, kind, q))

    # ---- tables: a total is arithmetic over real rows, never generated text
    for table in app.store.list_tables()[:6]:
        title = _clean(table.get("title") or "")
        where = f" in {title}" if title and len(title) < 40 else ""
        for column in (table.get("numeric_cols") or [])[:2]:
            add("table", f"What is the total {column}{where}?")

    # ---- entities: the specific things these documents are about
    try:
        nodes = app.store.top_nodes(limit=40)
    except Exception as exc:
        log.debug("entity candidates skipped: %s", exc)
        nodes = []
    for node in nodes:
        name, kind, mentions = _clean(node["name"]), node["kind"], node.get("mentions") or 0
        if len(name) < 3:
            continue
        bonus = min(1.5, mentions / 6)
        if kind == "person" or _PERSON_TITLE.match(name):
            add("entity", f"Who is {name}?", bonus)
        elif kind == "org":
            add("entity", f"What do the documents say about {name}?", bonus)
        elif kind in ("invoice", "code", "part_number"):
            if looks_like_a_code(name):
                add("code", f"What does {name} mean?", bonus)
        elif kind == "project":
            add("entity", f"What is recorded about {name}?", bonus)
        elif kind == "contract":
            add("entity", f"What are the terms of the {name.lower()}?", bonus)

    # ---- headings and vocabulary from auto-tuning
    pack = app.indexer.pack
    if pack:
        for code in (pack.sample_codes or [])[:6]:
            if looks_like_a_code(code):
                add("code", f"What does {code} mean?")
        for question in (pack.suggested_questions or []):
            kind = "heading" if question.lower().startswith("what does") else "term"
            if not any(w in question.lower() for w in _META):
                add(kind, question)
        for term in [t for t in (pack.key_terms or []) if t not in _META][:4]:
            add("term", f"What do the documents say about {term}?")

    out.sort(key=lambda row: -row[0])
    return out


def answerable(app, question: str, kind: str = "term") -> bool:
    """Does the corpus actually hold an answer? The same evidence check the Ask page uses."""
    try:
        if kind == "table":
            computed = getattr(getattr(app, "answerer", None), "table_answer", lambda q: None)(question)
            if computed:
                return True
        hits, _ = app.retriever.search(question)
        if not hits:
            return False
        return bool(assess(question, hits).sufficient)
    except Exception as exc:
        log.debug("candidate %r skipped: %s", question, exc)
        return False


def ranked_candidates(app, limit: int = 6, pool: int = 28) -> list[str]:
    """Answerable corpus questions, best first. Kept for the evaluation harness; the Ask page
    shows summaries instead (see :func:`recommend`)."""
    scored: list[tuple[float, str, str]] = []
    for weight, kind, question in candidates(app)[:pool]:
        if not answerable(app, question, kind):
            continue
        scored.append((weight + (2.0 if kind == "table" else 0.0), kind, question))
        if len(scored) >= limit * 3:
            break
    scored.sort(key=lambda row: -row[0])
    picked: list[str] = []
    used: dict[str, int] = {}
    for _, kind, question in scored:
        if used.get(kind, 0) >= 2:
            continue
        used[kind] = used.get(kind, 0) + 1
        picked.append(question)
        if len(picked) >= limit:
            break
    return picked


def recommend(app, limit: int = 6, pool: int = 28) -> list[dict]:
    """What the Ask page offers: a summary of each document, and nothing else.

    Returns dicts so each one can be scoped to its own document -- "Summarize invoice.pdf"
    reads only invoice.pdf, instead of retrieving across the whole workspace and summarising
    a mixture. An empty workspace offers nothing.
    """
    try:
        docs = list(app.store.list_documents() or [])
    except Exception as exc:
        log.debug("no document list: %s", exc)
        return []
    named: list[tuple[int, str]] = []
    for d in docs:
        try:
            name = _clean(d["name"])
            if name:
                named.append((int(d["id"]), name))
        except Exception:                       # a row without a name is simply not offered
            continue
    if not named:
        return []

    out: list[dict] = []
    if len(named) > 1:
        out.append({"question": "Summarize all the documents", "doc_ids": None})
    for doc_id, name in named:
        out.append({"question": f"Summarize {name}", "doc_ids": [doc_id]})
        if len(out) >= limit:
            break
    return out[:limit]
