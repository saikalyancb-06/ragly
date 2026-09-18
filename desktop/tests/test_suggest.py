"""Recommended questions are built from the corpus and must be answerable."""
from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ragly_backend.suggest import candidates, ranked_candidates, recommend  # noqa: E402


class FakeHit:
    def __init__(self, text, score=0.7):
        self.text, self.vector_score, self.doc_name, self.page = text, score, "d.pdf", 1
        self.heading, self.content_type = "", "text"


class FakeApp:
    """Just enough of App for the recommender: tables, entities, a pack and a retriever."""

    def __init__(self, tables, nodes, pack, text, docs=({"id": 1, "name": "garden.pdf"},)):
        self.store = types.SimpleNamespace(list_tables=lambda: tables, top_nodes=lambda limit=40: nodes,
                                           list_documents=lambda: list(docs))
        self.indexer = types.SimpleNamespace(pack=pack)
        self.retriever = types.SimpleNamespace(search=lambda q, **kw: ([FakeHit(text)], {}))
        # the table engine answers totals deterministically; the stub says "yes, computable"
        self.answerer = types.SimpleNamespace(
            table_answer=lambda q: {"answer": "18 cm"} if tables and "total" in q.lower() else None)


GARDEN_TABLE = {"title": "Basil growth log", "numeric_cols": ["Height (cm)"],
                "headers": ["Date", "Height (cm)"], "rows": [["14 September", "18"]]}
GARDEN_TEXT = ("Basil growth log. On 14 September the basil measured 18 cm. Watering is due every "
               "second day. Ananya Rao keeps the log for the Balcony Garden project.")
GARDEN_NODES = [{"name": "Ananya Rao", "kind": "person", "mentions": 4},
                {"name": "Balcony Garden", "kind": "project", "mentions": 3}]


def _pack(**kw):
    fields = {"sample_codes": [], "suggested_questions": [], "key_terms": []}
    fields.update(kw)
    return types.SimpleNamespace(**fields)


def test_questions_come_from_this_corpus():
    app = FakeApp([GARDEN_TABLE], GARDEN_NODES, _pack(key_terms=["basil", "watering"]), GARDEN_TEXT)
    asked = " ".join(q for _, _, q in candidates(app)).lower()
    assert "height (cm)" in asked and "basil growth log" in asked
    assert "ananya rao" in asked and "balcony garden" in asked
    assert "termination" not in asked and "insurance" not in asked      # no fixed template list


def test_a_different_corpus_asks_different_questions():
    claim_table = {"title": "Repair estimate", "numeric_cols": ["Cost (INR)"],
                   "headers": ["Part", "Cost (INR)"], "rows": [["Bumper", "18,400"]]}
    text = "Repair estimate for claim CLM-7781. The bumper costs ₹18,400."
    a = FakeApp([GARDEN_TABLE], GARDEN_NODES, _pack(), GARDEN_TEXT)
    b = FakeApp([claim_table], [{"name": "CLM-7781", "kind": "code", "mentions": 3}], _pack(), text)
    assert {q for _, _, q in candidates(a)} != {q for _, _, q in candidates(b)}
    assert any("Cost (INR)" in q for _, _, q in candidates(b))


def test_only_answerable_questions_are_offered():
    """Nothing retrievable and no table that computes: ranked_candidates is empty rather
    than misleading."""
    app = FakeApp([GARDEN_TABLE], GARDEN_NODES, _pack(), GARDEN_TEXT)
    app.retriever = types.SimpleNamespace(search=lambda q, **kw: ([], {}))
    app.answerer = types.SimpleNamespace(table_answer=lambda q: None)
    assert ranked_candidates(app) == []


def test_the_ranked_list_is_ranked_and_mixed():
    app = FakeApp([GARDEN_TABLE], GARDEN_NODES, _pack(key_terms=["basil"]), GARDEN_TEXT)
    picked = ranked_candidates(app, limit=4)
    assert picked, "a corpus with a table and entities must produce candidates"
    assert picked[0].startswith("What is the total"), "a computable total should lead"
    assert len(picked) == len(set(picked))


def test_the_ask_page_offers_summaries_and_nothing_else():
    """Krishna asked for exactly one kind of recommendation: summarise the document."""
    app = FakeApp([GARDEN_TABLE], GARDEN_NODES, _pack(key_terms=["basil"]), GARDEN_TEXT)
    picked = recommend(app)
    assert [p["question"] for p in picked] == ["Summarize garden.pdf"]
    assert picked[0]["doc_ids"] == [1]


def test_each_summary_is_scoped_to_its_own_document():
    docs = ({"id": 1, "name": "garden.pdf"}, {"id": 7, "name": "invoice.pdf"})
    app = FakeApp([GARDEN_TABLE], GARDEN_NODES, _pack(), GARDEN_TEXT, docs=docs)
    picked = recommend(app)
    assert picked[0] == {"question": "Summarize all the documents", "doc_ids": None}
    assert {p["question"]: p["doc_ids"] for p in picked[1:]} == {
        "Summarize garden.pdf": [1], "Summarize invoice.pdf": [7]}


def test_an_empty_workspace_recommends_nothing():
    app = FakeApp([], [], _pack(), "", docs=())
    assert recommend(app) == []


def test_ocr_noise_is_never_recommended():
    """Regression: "I34" came out of an OCR'd "₹34,869" and became a recommended question
    that could only ever answer "not found"."""
    from ragly_backend.suggest import looks_like_a_code

    assert looks_like_a_code("NS-2048") and looks_like_a_code("E-47") and looks_like_a_code("MB-884201")
    assert not looks_like_a_code("I34") and not looks_like_a_code("I34,869")
    assert not looks_like_a_code("2026") and not looks_like_a_code("ABC")


def test_a_junk_code_entity_produces_no_question():
    import types

    from ragly_backend.suggest import candidates

    app = FakeApp([], [{"name": "I34", "kind": "code", "mentions": 3}], 
                  types.SimpleNamespace(sample_codes=["I34"], suggested_questions=[], key_terms=[]),
                  "Find the image containing the total ₹34,869.")
    assert not any("I34" in q for _w, _k, q in candidates(app))
