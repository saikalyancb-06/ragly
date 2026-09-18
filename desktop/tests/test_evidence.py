"""Evidence sufficiency and the evaluation harness's failure attribution."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ragly_backend.evidence import assess, classify_question  # noqa: E402


@dataclass
class FakeHit:
    text: str
    doc_name: str = "doc.pdf"
    page: int = 1
    vector_score: float = 0.7
    heading: str = ""
    content_type: str = "text"


LOAN_RULES = FakeHit(
    "Loan transactions are categorised separately. Pre-EMI interest is treated as a finance cost "
    "and disclosed under borrowing costs in the notes to accounts.",
    doc_name="rules.pdf", page=2, vector_score=0.62)

NOTICE = FakeHit("Either party may terminate this agreement by giving 60 days written notice.",
                 doc_name="service_agreement_v1.pdf", page=2, vector_score=0.78)

TORQUE = FakeHit("Tighten the earth bolt to 40 Nm. Tighten the contact stem nut to 25 Nm.",
                 doc_name="maintenance_procedures.pdf", page=3, vector_score=0.71)


# ---------------------------------------------------------------- classification
def test_classify_value_question():
    shape = classify_question("What is the earth bolt torque?")
    assert shape.wants_value == "number"      # the answer must contain a number
    assert "torque" in " ".join(shape.attributes + shape.entities).lower()


def test_classify_summary_question():
    shape = classify_question("Summarise what the fault code manual covers.")
    assert shape.intent == "summary"
    assert shape.wants_value is None          # a summary is not required to contain a number


# ---------------------------------------------------------------- sufficiency
def test_sufficient_when_the_answer_is_on_the_page():
    s = assess("What is the termination notice period?", [NOTICE])
    assert s.sufficient is True


def test_insufficient_when_the_entity_is_absent():
    """The exact failure this layer was written for: related topic, wrong entity."""
    s = assess("What are the actual Axis Bank loan interest rates?", [LOAN_RULES])
    assert s.sufficient is False
    assert "axis bank" in s.message.lower()
    assert "rules.pdf" in s.message


def test_insufficient_when_the_attribute_is_absent():
    s = assess("What is the warranty period for the transformer?", [TORQUE])
    assert s.sufficient is False


def test_no_hits_is_insufficient():
    s = assess("Anything at all?", [])
    assert s.sufficient is False
    assert s.message


def test_message_never_invents_a_value():
    s = assess("What is the SF6 refill cost in rupees?", [TORQUE])
    assert s.sufficient is False
    assert "₹" not in s.message


# ---------------------------------------------------------------- harness scoring
def _row(case, result, retrieval):
    import rag_eval

    return rag_eval.score_answer(case, result, retrieval)


def test_retrieval_failure_is_blamed_on_retrieval():
    case = {"answerable": True, "category": "factual", "must_include": ["60 days"],
            "expected_sources": [{"document": "a.pdf", "page": 2}]}
    result = {"answer": "Not found in your documents.", "refused": True, "grounding": {"flag": "low_relevance"},
              "citations": [], "timing": {}}
    row = _row(case, result, {"recall": 0.0, "best_vector_score": 0.2})
    assert row["retrieval_failure"] and not row["generation_failure"]


def test_generation_failure_when_the_evidence_was_retrieved():
    case = {"answerable": True, "category": "factual", "must_include": ["60 days"],
            "expected_sources": [{"document": "a.pdf", "page": 2}]}
    result = {"answer": "The agreement may be terminated with notice [1].", "refused": False,
              "grounding": {"verified": True}, "citations": [{"doc_name": "a.pdf", "page": 2}], "timing": {}}
    row = _row(case, result, {"recall": 1.0, "best_vector_score": 0.8})
    assert row["generation_failure"] and not row["retrieval_failure"]


def test_over_refusal_is_a_grounding_failure():
    case = {"answerable": True, "category": "factual", "must_include": ["60 days"],
            "expected_sources": [{"document": "a.pdf", "page": 2}]}
    result = {"answer": "Not found in your documents.", "refused": True,
              "grounding": {"flag": "insufficient_evidence"}, "citations": [], "timing": {}}
    row = _row(case, result, {"recall": 1.0, "best_vector_score": 0.8})
    assert row["grounding_failure"] and not row["generation_failure"]


def test_answering_an_unanswerable_question_is_a_grounding_failure():
    case = {"answerable": False, "category": "unanswerable"}
    result = {"answer": "The rate is 9.5% [1].", "refused": False, "grounding": {"verified": True},
              "citations": [{"doc_name": "rules.pdf", "page": 2}], "timing": {}}
    row = _row(case, result, {"recall": None, "best_vector_score": 0.62})
    assert row["correct"] is False and row["grounding_failure"]


def test_correct_abstention_counts_as_correct():
    case = {"answerable": False, "category": "unanswerable"}
    result = {"answer": "I couldn't find Axis Bank in the indexed documents.", "refused": True,
              "grounding": {"flag": "insufficient_evidence"}, "citations": [], "timing": {}}
    row = _row(case, result, {"recall": None, "best_vector_score": 0.62})
    assert row["correct"] is True
    assert not any(row[k] for k in ("retrieval_failure", "generation_failure", "grounding_failure"))


def test_thousands_separators_do_not_break_matching():
    import rag_eval

    assert rag_eval.contains("The total is 2,35,500.00 rupees.", "235500")
    assert rag_eval.contains("The fee is Rs 1,75,000 per month.", "1,75,000")


# ---------------------------------------------------------------- specificity (similar-but-wrong)
def test_a_compound_subject_must_actually_appear():
    """"cable gland clamp bolt" is not answered by a page that merely contains "bolt" and a torque."""
    s = assess("What is the torque for the cable gland clamp bolt?", [TORQUE])
    assert s.sufficient is False


def test_a_named_role_must_appear_not_just_the_company():
    invoice = FakeHit("Invoice INV-1003 from Bharat Switchgear Pvt Ltd. Approved by: Ramesh Iyer.",
                      doc_name="invoice_INV-1003.pdf")
    s = assess("Who is the CEO of Bharat Switchgear Pvt Ltd?", [invoice])
    assert s.sufficient is False          # "ceo" must not be found inside "invoice order"-style text


def test_a_measure_word_is_satisfied_by_a_value_of_that_kind():
    """"notice period" is answered by "60 days" even though the word "period" never appears."""
    s = assess("What is the termination notice period?", [NOTICE])
    assert s.sufficient is True


def test_word_stems_match():
    invoice = FakeHit("Invoice INV-1001. Approver: Ramesh Iyer. Amount: Rs 62,000.",
                      doc_name="invoice_INV-1001.pdf")
    s = assess("Who approved invoice INV-1001?", [invoice])
    assert s.sufficient is True           # "approved" is answered by "Approver"


def test_document_selection_words_are_ignored():
    s = assess("What is the termination notice period in the original agreement?", [NOTICE])
    assert s.sufficient is True           # "original" only picks a document; it is not evidence


# ---------------------------------------------------------------- claim verification
def test_currency_word_inside_another_word_is_not_a_money_claim():
    """Regression: "…a qualified engineer, and…" was read as the amount "Rs ," and blocked a good answer."""
    from ragly_backend.answer import verify_claims

    sources = "FAULT E-63 Meaning: SF6 gas pressure low. Recommended action: Isolate the bay and call the gas team."
    answer = "Isolate the bay and call the gas handling team, and have a qualified engineer, inspect it."
    unsupported, _ = verify_claims(answer, sources)
    assert unsupported == []


def test_an_invented_amount_is_still_caught():
    from ragly_backend.answer import verify_claims

    unsupported, _ = verify_claims("The refill costs Rs 45,000 [1].", "SF6 gas pressure low. Isolate the bay.")
    assert any("45,000" in u for u in unsupported)


def test_a_question_that_only_names_a_code_is_answerable():
    """Regression: "What does NS-2048 mean?" was refused while "what is ns2048" was answered —
    the code leaked into the attribute list and was then looked for as a separate detail."""
    card = FakeHit("Employee ID NS-2048 belongs to Ananya Rao, Applied Intelligence department.",
                   doc_name="corpus.pdf", page=8, vector_score=0.57)
    for question in ("What does NS-2048 mean?", "What does NS-2048 refer to?", "what is ns2048"):
        assert assess(question, [card]).sufficient is True, question


def test_an_unknown_code_is_still_refused():
    card = FakeHit("Employee ID NS-2048 belongs to Ananya Rao.", doc_name="corpus.pdf", page=8)
    assert assess("What does MTR-9999 mean?", [card]).sufficient is False


# ---------------------------------------------------------------- quoting the source
def test_a_stubborn_refusal_falls_back_to_the_source_words():
    """The corpus itself says "a grounded system should abstain", and the model obeyed it.
    Document text is material, not instructions, so the source's own words are shown."""
    from ragly_backend.answer import quote_evidence
    from ragly_backend.evidence import classify_question

    hit = FakeHit("Employee ID NS-2048 Employee name Ananya Rao Employee department Applied Intelligence.",
                  doc_name="corpus.pdf", page=8)
    quoted = quote_evidence("What does NS-2048 mean?", [hit], classify_question("What does NS-2048 mean?"))
    assert quoted and "NS-2048" in quoted[0] and quoted[1] == 1


def test_a_quote_must_carry_everything_the_question_named():
    """Regression: quoting the Northstar rate for a question about Axis Bank would be a
    hallucination wearing a citation."""
    from ragly_backend.answer import quote_evidence
    from ragly_backend.evidence import classify_question

    hit = FakeHit("Principal ₹18,00,000 Annual interest rate 9.25% fixed Tenure 36 months.",
                  doc_name="corpus.pdf", page=3)
    q = "What is the interest rate on the Axis Bank loan?"
    assert quote_evidence(q, [hit], classify_question(q)) is None


def test_a_quote_needs_the_detail_not_just_the_subject():
    from ragly_backend.answer import quote_evidence
    from ragly_backend.evidence import classify_question

    hit = FakeHit("The image contains a laptop, phone, desk and mug.", doc_name="corpus.pdf", page=9)
    q = "What is the warranty period of the laptop?"
    assert quote_evidence(q, [hit], classify_question(q)) is None


def test_a_code_must_match_as_a_whole_token():
    """Regression: "I34" matched inside "₹34,869" and quoted a line from a test script."""
    from ragly_backend.answer import _token_present, quote_evidence
    from ragly_backend.evidence import classify_question

    assert _token_present("Employee ID NS-2048 name Ananya Rao", "NS-2048")
    assert not _token_present("Find the image containing the total ₹34,869.", "I34")
    assert not _token_present("Invoice INV-1003 was paid.", "INV-10")

    hit = FakeHit("9. Find the image containing the total ₹34,869. Image → Image 10.",
                  doc_name="corpus.pdf", page=12)
    assert quote_evidence("What does I34 mean?", [hit], classify_question("What does I34 mean?")) is None


def test_a_numbered_test_step_is_never_quoted():
    from ragly_backend.answer import quote_evidence
    from ragly_backend.evidence import classify_question

    hit = FakeHit("9. Upload the vehicle image and search for visually similar images NS-2048.",
                  doc_name="corpus.pdf", page=12)
    q = "What does NS-2048 mean?"
    assert quote_evidence(q, [hit], classify_question(q)) is None
