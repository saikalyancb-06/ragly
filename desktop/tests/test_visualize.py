"""The Visualization Planner: the right picture, in the document's own words, with evidence."""
from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ragly_backend.visualize import (  # noqa: E402
    Planner, classify_intent, choose_type, describe_relation, extract_facts, parse_date)

LOAN = ("LOAN AGREEMENT. This agreement is executed on 5 September 2026 between Axis Bank and "
        "NS-2048, an employee of BluePeak Analytics Pvt. Ltd. Loan amount: ₹18,00,000. "
        "Interest rate: 9.25% p.a. Tenure: 36 months. Monthly EMI is ₹57,530. "
        "Processing fee of ₹28,890 is payable. The first payment is due on 14 October 2026 "
        "and the facility expires on 31 March 2027.")


def _facts(text=LOAN):
    return {(f.label.lower(), f.value): f for f in extract_facts(text, 1, "Agreement.pdf", 1)}


# ---------------------------------------------------------------- semantic roles
def test_every_value_carries_the_documents_own_label():
    """The acceptance case: these numbers must stop being anonymous nodes."""
    facts = _facts()
    roles = {f.role for f in facts.values()}
    assert {"LOAN_AMOUNT", "MONTHLY_EMI", "INTEREST_RATE", "DURATION", "FEE"} <= roles
    by_value = {f.value: f for f in facts.values()}
    assert by_value["₹18,00,000"].label.lower().startswith("loan")
    assert by_value["₹57,530"].role == "MONTHLY_EMI"
    assert by_value["9.25% p.a."].category == "rate"
    assert by_value["36 months"].category == "duration"


def test_a_value_with_no_readable_label_is_not_shown():
    """An unexplained number on screen is worse than a shorter list."""
    assert extract_facts("Totals: 4,500 1,200 900 300", 1, "d.pdf", 1) == []


def test_a_value_the_document_did_not_name_is_not_given_a_meaning():
    """Flattened table text has no separator, so nearby words are not evidence of meaning.
    Guessing there is what turned "12 months" into PRINCIPAL and "3%" into EMI."""
    facts = extract_facts("Field Ground truth Loan amount ₹18,00,000", 1, "d.pdf", 1)
    assert [(f.role, f.label) for f in facts] == [("UNKNOWN", "Amount mentioned")]
    stated = extract_facts("Loan amount: ₹18,00,000", 1, "d.pdf", 1)
    assert [(f.role, f.label) for f in stated] == [("LOAN_AMOUNT", "Loan amount")]


def test_a_two_column_table_gives_the_label_the_prose_could_not():
    """The structure IS the meaning: the row label names the value beside it."""
    from ragly_backend.visualize import facts_from_table

    facts = facts_from_table({"id": 3, "doc_id": 1, "doc_name": "Agreement.pdf", "page": 3,
                              "title": "Key Facts", "headers": ["Term", "Value"],
                              "rows": [["Loan Amount", "₹18,00,000"], ["Interest Rate", "9.25%"],
                                       ["Tenure", "36 months"], ["EMI", "₹57,530"],
                                       ["Borrower", "NS-2048"]]})
    assert [f.role for f in facts] == ["LOAN_AMOUNT", "INTEREST_RATE", "DURATION", "MONTHLY_EMI"]
    assert facts[0].sources[0]["row_label"] == "Loan Amount"
    assert facts[0].sources[0]["column_label"] == "Value"
    assert facts[0].sources[0]["table_id"] == 3


def test_the_specific_mislabellings_that_were_reported():
    """12 months must not be PRINCIPAL, 3% must not be EMI, a charge is not GST."""
    facts = {f.value: (f.role, f.label) for f in extract_facts(
        "Outstanding principal after 12 months a foreclosure charge of 3% applies. "
        "A charge of ₹28,890 was raised.", 1, "d.pdf", 1)}
    assert facts["12 months"][0] in ("UNKNOWN", "DURATION")
    assert "principal" not in facts["12 months"][1].lower()
    assert facts["3%"][0] != "MONTHLY_EMI"
    assert "gst" not in facts["₹28,890"][1].lower()


def test_prepayment_is_not_the_loan_amount():
    facts = {f.value: f.role for f in extract_facts(
        "Loan amount: ₹18,00,000. Prepayment amount: ₹2,00,000.", 1, "d.pdf", 1)}
    assert facts["₹18,00,000"] == "LOAN_AMOUNT"
    assert facts["₹2,00,000"] == "PREPAYMENT"


def test_a_date_phrase_cannot_label_a_percentage():
    facts = {f.value: f.label for f in extract_facts("First EMI date 5 September 2026 rate 3%", 1, "d.pdf", 1)}
    assert "date" not in facts["3%"].lower()


def test_an_unnamed_value_never_reaches_the_screen():
    """It stays in the model so the debug panel can count it, and is never rendered."""
    facts = extract_facts("Expected OCR anchors ₹34,869", 1, "d.pdf", 1)
    assert [f.role for f in facts] == ["UNKNOWN"]
    plan = Planner(_app()).plan("")
    rendered = [i["label"] for section in plan["sections"] for i in (section.get("items") or [])]
    assert not any("mentioned" in label.lower() for label in rendered)


def test_dates_are_parsed_in_the_forms_indian_documents_use():
    assert parse_date("5 September 2026").isoformat() == "2026-09-05"
    assert parse_date("14 Oct 2026").isoformat() == "2026-10-14"
    assert parse_date("31/03/2027").isoformat() == "2027-03-31"
    assert parse_date("2026-09-05").isoformat() == "2026-09-05"
    assert parse_date("sometime next year") is None


# ---------------------------------------------------------------- intent -> view
def test_the_question_chooses_the_visualisation():
    have = {"dates": 4, "money": 6, "relations": 5, "tables": 2, "images": 3}
    cases = {
        "Visualize the loan terms": "financial_summary",
        "Show financial information": "financial_summary",
        "Show important dates": "timeline",
        "Show relationships": "relationship_graph",
        "Explain how NS-2048 is connected to the agreement": "focused_path",
        "Show the data in this table": "table_visualization",
        "Compare these documents": "comparison_view",
        "Visualize this document": "document_overview",
        "": "document_overview",
    }
    for question, expected in cases.items():
        assert choose_type(classify_intent(question), have) == expected, question


def test_a_view_is_not_promised_when_the_documents_cannot_fill_it():
    """No timeline for a document with one date; no graph without relationships."""
    empty = {"dates": 1, "money": 0, "relations": 0, "tables": 0, "images": 0}
    assert choose_type("timeline", empty) == "document_overview"
    assert choose_type("financial", empty) == "document_overview"
    assert choose_type("relationships", empty) == "document_entity_map"


# ---------------------------------------------------------------- relationships
def test_proximity_is_never_drawn_as_a_relationship():
    """"Axis Bank appears alongside Bengaluru" is retrieval evidence, not a fact."""
    assert describe_relation("works_at") == ("works at", "explicit")
    assert describe_relation("co_occurs_with") == ("", "")
    assert describe_relation("related_to") == ("", "")
    assert describe_relation("x7_$$") == ("", "")


def test_the_graph_holds_only_relationships_the_document_states():
    plan = Planner(_app()).plan("Show relationships")
    assert all(e["origin"] == "explicit" or e["note"] for e in plan["relationships"]["edges"])
    assert all(e["label"] != "appears alongside" for e in plan["relationships"]["edges"])


# ---------------------------------------------------------------- the plan
class FakeStore:
    def __init__(self, text=LOAN):
        self._text = text

    def list_documents(self):
        return [{"id": 1, "name": "Agreement.pdf", "status": "ready"}]

    def chunk_ids(self, doc_id):
        return [10]

    def get_chunks(self, ids):
        return {10: {"id": 10, "doc_id": 1, "page": 1, "heading": "", "text": self._text,
                     "doc_name": "Agreement.pdf", "content_type": "text"}}

    def top_nodes(self, kind=None, limit=40):
        if kind == "org":
            return [{"id": 2, "kind": "org", "name": "Axis Bank", "mentions": 4}]
        if kind == "person":
            return []
        return [{"id": 2, "kind": "org", "name": "Axis Bank", "mentions": 4}]

    def node_neighbourhood(self, node_id, limit=60):
        return {"node": {"id": 2, "kind": "org", "name": "Axis Bank", "mentions": 4},
                "edges": [{"src": 2, "dst": 3, "rel": "party_to", "src_kind": "org",
                           "dst_kind": "contract", "src_name": "Axis Bank", "dst_name": "Agreement",
                           "doc_name": "Agreement.pdf", "page": 1, "snippet": "Axis Bank and NS-2048"},
                          {"src": 2, "dst": 4, "rel": "has_amount", "src_kind": "org",
                           "dst_kind": "amount", "src_name": "Axis Bank", "dst_name": "₹18,00,000",
                           "doc_name": "Agreement.pdf", "page": 1, "snippet": ""}],
                "documents": [], "images": []}

    def list_tables(self, doc_ids=None):
        return []

    def get_images(self, limit=200, **kw):
        return []


def _app():
    return types.SimpleNamespace(store=FakeStore(),
                                 graph=types.SimpleNamespace(resolve=lambda q, limit=1: []))


def test_the_financial_view_leads_with_the_key_terms_panel_and_keeps_provenance():
    """One card per meaning, headed by the roles a reader came for — not a card per number."""
    plan = Planner(_app()).plan("Visualize the loan terms")
    assert plan["type"] == "financial_summary"
    assert plan["sections"][0]["kind"] == "kpi"
    roles = [i["role"] for i in plan["sections"][0]["items"]]
    assert roles[:4] == ["LOAN_AMOUNT", "INTEREST_RATE", "DURATION", "MONTHLY_EMI"]
    item = plan["sections"][0]["items"][0]
    assert item["evidence"]["doc_name"] == "Agreement.pdf"
    assert item["evidence"]["page"] == 1 and item["evidence"]["text"]
    assert 0 < item["evidence"]["confidence"] <= 1.0
    assert item["references"] >= 1


def test_a_document_of_comparable_values_does_get_a_chart():
    """The breakdown has to sit under its own heading — that is what separates a spending
    table from three unrelated dollar figures that share a file."""
    app = _app()
    app.store = FakeStore("Food: \u20b918,200. Travel: \u20b911,400. Shopping: \u20b98,900. "
                          "Utilities: \u20b97,100.")
    app.store.get_chunks = lambda ids: {10: {"id": 10, "doc_id": 1, "page": 1,
                                             "heading": "Monthly spending", "text": app.store._text,
                                             "doc_name": "Agreement.pdf", "content_type": "text"}}
    plan = Planner(app).plan("Show financial information")
    chart = next((s2["chart"] for s2 in plan["sections"] if s2["kind"] == "chart"), None)
    assert chart and len(chart["series"][0]["points"]) == 4


def test_the_timeline_is_in_chronological_order():
    plan = Planner(_app()).plan("Show important dates")
    assert plan["type"] == "timeline"
    dates = [i["date"] for i in plan["timeline"]]
    assert dates == sorted(dates)
    assert dates[0] == "2026-09-05" and dates[-1] == "2027-03-31"


def test_amounts_never_appear_as_nodes_in_the_relationship_graph():
    """The whole complaint: NS-2048 -> ₹57,530 is not a relationship anyone can read."""
    plan = Planner(_app()).plan("Show relationships")
    graph = plan["relationships"]
    assert graph["nodes"], "a relationship view needs entities"
    assert all(n["kind"] not in ("amount", "date", "duration") for n in graph["nodes"])
    assert all(e["label"] and e["origin"] in ("explicit", "inferred") for e in graph["edges"])


def test_the_first_screen_stays_readable():
    plan = Planner(_app()).plan("")
    assert plan["type"] == "document_overview"
    assert len(plan["relationships"]["nodes"]) <= 15
    for section in plan["sections"]:
        assert len(section.get("items") or []) <= 12


# ---------------------------------------------------------------- charts
def test_labelled_figures_of_one_kind_become_a_chart():
    from ragly_backend.visualize import chart_from_facts

    facts = extract_facts("Loan amount: ₹18,00,000. Monthly EMI is ₹57,530. "
                          "Processing fee of ₹28,890 is payable.", 1, "a.pdf", 1)
    chart = chart_from_facts(facts, "money")
    assert chart["kind"] == "bar" and chart["unit"] == "₹"
    assert [p["label"] for p in chart["series"][0]["points"]] == ["Loan amount", "EMI", "Processing fee"]
    assert chart["series"][0]["points"][0]["value"] == 1800000.0
    assert chart["series"][0]["points"][0]["display"] == "₹18,00,000"


def test_rupees_and_percentages_are_never_put_on_one_axis():
    """A chart mixing units is a picture that lies."""
    from ragly_backend.visualize import _chart_from_table

    chart = _chart_from_table(["Item", "Cost", "Share"],
                              [["A", "₹100", "10%"], ["B", "₹200", "20%"], ["C", "₹700", "70%"]])
    assert [s["name"] for s in chart["series"]] == ["Cost"]
    assert chart["unit"] == "₹"


def test_a_quantity_column_does_not_win_over_the_money_column():
    from ragly_backend.visualize import _chart_from_table

    chart = _chart_from_table(["Part", "Qty", "Total"],
                              [["Bumper", "1", "₹24,800"], ["Headlamp", "1", "₹31,500"],
                               ["Clips", "1", "₹1,650"], ["Paint", "1", "₹9,500"]])
    assert chart["series"][0]["name"] == "Total" and chart["kind"] == "bar"


def test_dates_on_the_x_axis_make_it_a_line_in_date_order():
    from ragly_backend.visualize import _chart_from_table

    chart = _chart_from_table(["Date", "Reading"],
                              [["19 Sep 2026", "25"], ["5 Sep 2026", "18"], ["12 Sep 2026", "21"]])
    assert chart["kind"] == "line"
    assert [p["value"] for p in chart["series"][0]["points"]] == [18.0, 21.0, 25.0]


def test_two_money_columns_are_grouped_bars():
    from ragly_backend.visualize import _chart_from_table

    chart = _chart_from_table(["Month", "Debit", "Credit"],
                              [["Jan", "₹1,000", "₹900"], ["Feb", "₹1,200", "₹1,500"],
                               ["Mar", "₹800", "₹1,100"]])
    assert chart["kind"] == "grouped_bar" and len(chart["series"]) == 2


def test_one_figure_alone_is_not_charted():
    from ragly_backend.visualize import chart_from_facts

    facts = extract_facts("Loan amount: ₹18,00,000.", 1, "a.pdf", 1)
    assert chart_from_facts(facts, "money") is None


def test_a_loan_document_gets_cards_and_no_chart():
    """A loan amount, a rate, a tenure and an instalment are four kinds of fact. There is
    nothing to compare, so there is no chart — the old "Amounts side by side" was nonsense."""
    plan = Planner(_app()).plan("Show financial information")
    assert [s["kind"] for s in plan["sections"] if s["kind"] == "chart"] == []
    assert plan["sections"][0]["kind"] == "kpi"


def test_the_same_fact_on_three_pages_is_one_card_with_three_sources():
    from ragly_backend.visualize import merge_facts

    pages = [extract_facts("Loan amount: ₹18,00,000.", 1, "a.pdf", p, "Loan") for p in (3, 12, 13)]
    merged = merge_facts([f for page in pages for f in page])
    assert len(merged) == 1
    assert merged[0].references_count if hasattr(merged[0], "references_count") else len(merged[0].sources) == 3
    assert [s["page"] for s in merged[0].sources] == [3, 12, 13]


def test_two_values_for_one_role_are_both_kept_and_flagged():
    """Silently picking one would be inventing an answer."""
    from ragly_backend.visualize import merge_facts

    facts = merge_facts(extract_facts("Loan amount: ₹18,00,000. Loan amount: ₹2,00,000.",
                                      1, "a.pdf", 1, "Loan"))
    assert len(facts) == 2
    assert all(f.conflict for f in facts)


def test_facts_are_grouped_by_the_thing_they_describe():
    from ragly_backend.visualize import group_by_subject, merge_facts

    facts = merge_facts(
        extract_facts("Loan amount: ₹18,00,000. Monthly EMI is ₹57,530.", 1, "a.pdf", 1, "Loan")
        + extract_facts("Invoice total: ₹34,090.20.", 1, "a.pdf", 5, "Invoice"))
    groups = dict((name, [f.label for f in items]) for name, items in group_by_subject(facts))
    assert "Loan" in groups and "Invoice" in groups
    assert "Invoice total" not in groups["Loan"]


def test_the_headline_panel_shows_one_card_per_meaning():
    from ragly_backend.visualize import key_facts, merge_facts

    facts = merge_facts(extract_facts(
        "Loan amount: ₹18,00,000. Loan amount: ₹18,00,000. Interest rate: 9.25%. "
        "Tenure: 36 months. Monthly EMI is ₹57,530.", 1, "a.pdf", 1, "Loan"))
    roles = [f.role for f in key_facts(facts)]
    assert roles == ["LOAN_AMOUNT", "INTEREST_RATE", "DURATION", "MONTHLY_EMI"]


def test_a_loan_amount_a_rate_and_a_tenure_are_never_one_bar_chart():
    """The complaint that started this: those three are different kinds of fact."""
    from ragly_backend.visualize import comparable_series, merge_facts

    facts = merge_facts(extract_facts(
        "Loan amount: ₹18,00,000. Interest rate: 9.25%. Tenure: 36 months. "
        "Monthly EMI is ₹57,530.", 1, "a.pdf", 1, "Loan"))
    assert comparable_series(facts) is None


def test_values_of_the_same_kind_do_chart():
    from ragly_backend.visualize import comparable_series, merge_facts

    facts = merge_facts(
        extract_facts("Food: ₹18,200. Travel: ₹11,400. Shopping: ₹8,900. "
                      "Utilities: ₹7,100.", 1, "a.pdf", 1, "Monthly spending"))
    chart = comparable_series(facts)
    assert chart and chart["kind"] == "bar" and len(chart["series"][0]["points"]) == 4


def test_a_query_about_loan_terms_keeps_the_rate_and_the_tenure():
    """A filter that only matched the word "loan" dropped exactly what was asked for."""
    from ragly_backend.visualize import _relevant_to, merge_facts

    facts = merge_facts(extract_facts(
        "Loan amount: ₹18,00,000. Interest rate: 9.25%. Tenure: 36 months. "
        "Invoice total: ₹34,090.20.", 1, "a.pdf", 1, "Loan"))
    kept = {f.role for f in _relevant_to(facts, "Visualize the loan terms", "financial")}
    assert {"LOAN_AMOUNT", "INTEREST_RATE", "DURATION"} <= kept


def test_a_timeline_is_never_filtered_by_the_wording_of_the_question():
    from ragly_backend.visualize import _relevant_to, merge_facts

    facts = merge_facts(extract_facts(
        "Agreement executed on 2 August 2026. First payment due 5 September 2026. "
        "Expires on 31 March 2027.", 1, "a.pdf", 1, "Loan"))
    assert len(_relevant_to(facts, "Show important dates", "timeline")) == len(facts)


def test_a_column_of_dates_or_years_is_never_charted_as_a_height():
    from ragly_backend.visualize import _chart_from_table

    chart = _chart_from_table(["Description", "Date", "Amount"],
                              [["A", "2 Aug 2026", "₹100"], ["B", "5 Sep 2026", "₹200"],
                               ["C", "1 Oct 2026", "₹300"]])
    assert [s["name"] for s in chart["series"]] == ["Amount"]
    assert _chart_from_table(["Item", "Year"], [["A", "2024"], ["B", "2025"], ["C", "2026"]]) is None


# ---------------------------------------------------------------- the acceptance list
def test_a_question_this_document_cannot_answer_says_so():
    """TEST 8: never fabricate a visualisation."""
    plan = Planner(_app()).plan("Show the vehicle registration numbers")
    assert plan["type"] == "insufficient_evidence"
    assert "Nothing in" in plan["notes"][0]


def test_a_donut_is_only_drawn_when_the_parts_make_the_stated_whole():
    from ragly_backend.visualize import donut_from_table

    good = donut_from_table(["Category", "Amount"],
                            [["Food", "₹18,200"], ["Travel", "₹11,400"],
                             ["Shopping", "₹8,900"], ["Utilities", "₹7,100"],
                             ["TOTAL", "₹45,600"]])
    assert good["kind"] == "donut" and good["total"] == 45600.0
    assert len(good["series"][0]["points"]) == 4          # the TOTAL row is not a slice

    mismatched = donut_from_table(["Category", "Amount"],
                                  [["Food", "₹18,200"], ["Travel", "₹11,400"],
                                   ["Shopping", "₹8,900"], ["TOTAL", "₹99,999"]])
    assert mismatched is None
    assert donut_from_table(["Category", "Amount"],
                            [["Food", "₹18,200"], ["Travel", "₹11,400"]]) is None


def test_asking_for_the_kind_of_view_does_not_filter_the_facts_away():
    """"Show financial information" names the view, not a topic: it must not empty the page."""
    plan = Planner(_app()).plan("Show financial information")
    assert plan["type"] == "financial_summary"
    assert plan["sections"][0]["kind"] == "kpi"


def test_asking_for_charts_finds_every_chartable_table():
    """A chart in the fourth table is still a chart: the overview used to look at the first
    table only, so a document full of chartable data appeared to have none."""
    from ragly_backend.visualize import choose_type, classify_intent

    assert classify_intent("Show me charts") == "charts"
    assert classify_intent("show the graphs") == "charts"
    assert choose_type("charts", {"tables": 2, "money": 0, "dates": 0, "relations": 0, "images": 0}) == "charts"
    # nothing chartable: say so rather than draw something meaningless
    assert choose_type("charts", {"tables": 0, "money": 0, "dates": 0, "relations": 0, "images": 0}) \
        == "document_overview"


def test_a_short_form_date_column_still_reads_as_time():
    """"02 Sep" has no year, so parse_date returns None — the column is still dates, and the
    chart must be a line in date order, not bars."""
    from ragly_backend.visualize import _chart_from_table, _looks_like_dates

    assert _looks_like_dates(["02 Sep", "09 Sep", "16 Sep"])
    chart = _chart_from_table(["Date", "Debit"],
                              [["02 Sep", "1200"], ["09 Sep", "800"], ["16 Sep", "1500"]])
    assert chart["kind"] == "line"


def test_a_sentence_fragment_is_not_a_label():
    """An annual report filled the screen with cards headed "HAD NOT YET COMMENCED" and
    "ELECTED TO OFFSET" — prose captured as though it named the figure."""
    from ragly_backend.visualize import _is_a_name

    assert _is_a_name("Loan amount") and _is_a_name("Accounts payable") and _is_a_name("Tax expense")
    assert not _is_a_name("had not yet commenced")
    assert not _is_a_name("elected to offset")
    assert not _is_a_name("price or impairments")
    assert not _is_a_name("maintain minimum liquidity")

    facts = {f.value: (f.role, f.label) for f in extract_facts(
        "Construction that had not yet commenced was $92.7 million. Income tax expense: $411 million.",
        1, "r.docx", 57)}
    assert facts["$92.7 million"][0] == "UNKNOWN"
    assert facts["$411 million"][1] == "Tax expense"


def test_the_scale_word_stays_with_the_figure():
    """"$411 million" read as "$411" is wrong by six orders of magnitude, and it sat next to
    a card reading "$8"."""
    facts = extract_facts("Income tax expense: $411 million. Revenue: ₹1.4 crore.", 1, "r.docx", 1)
    values = [f.value for f in facts]
    assert "$411 million" in values
    assert "₹1.4 crore" in values


def test_a_number_used_as_a_heading_is_not_a_subject():
    from ragly_backend.visualize import _clean_subject

    assert _clean_subject("2,460") == ""
    assert _clean_subject("12.") == ""
    assert _clean_subject("Income taxes") == "Income taxes"


def test_only_well_evidenced_facts_are_rendered():
    """A guess from a sentence stays in the model for the debug panel, never on the screen."""
    app = _app()
    app.store = FakeStore("Construction that had not yet commenced was $92.7 million. "
                          "The Company elected to offset $452 million of deferred tax assets.")
    plan = Planner(app).plan("")
    rendered = [i["label"] for s in plan["sections"] for i in (s.get("items") or [])]
    assert not any("commenced" in r.lower() or "offset" in r.lower() for r in rendered)


# ---------------------------------------------------------------- no repetition, no fake charts
def _money(label, value, number, unit, subject, doc="2025_AnnualReport.docx", role="STATED"):
    from ragly_backend.visualize import Fact

    return Fact(label=label, value=value, role=role, category="money", doc_id=1, doc_name=doc,
                page=1, snippet="", confidence=0.95, number=number, unit=unit, subject=subject,
                sources=[{"doc_id": 1, "doc_name": doc, "page": 1, "text": ""}])


def test_a_label_that_repeats_with_different_values_is_held_back():
    """Nine cards headed "Issuance" with nine different figures is not information."""
    from ragly_backend.visualize import merge_facts

    facts = merge_facts([_money("Issuance", "$0.1", 0.1, "$", "2025_AnnualReport.docx"),
                         _money("Issuance", "$3.8", 3.8, "$", "2025_AnnualReport.docx"),
                         _money("Issuance", "$23.8", 23.8, "$", "2025_AnnualReport.docx"),
                         _money("Accounts payable", "$6.9", 6.9, "$", "2025_AnnualReport.docx")])
    ambiguous = {f.label for f in facts if f.ambiguous}
    assert ambiguous == {"Issuance"}
    assert not next(f for f in facts if f.label == "Accounts payable").ambiguous


def test_values_compared_is_gone():
    """An issuance, a legal provision and a construction commitment are not one measure —
    being dollars in the same file is not a reason to put them on one axis."""
    from ragly_backend.visualize import comparable_series, merge_facts

    facts = merge_facts([_money("Issuance", "$0.1", 0.1, "$", "2025_AnnualReport.docx"),
                         _money("Aggregate legal liabilities", "$541", 541, "$", "2025_AnnualReport.docx"),
                         _money("Had not yet commenced", "$92.7", 92.7, "$", "2025_AnnualReport.docx")])
    assert comparable_series(facts) is None


def test_a_breakdown_under_a_real_heading_still_charts():
    from ragly_backend.visualize import comparable_series, merge_facts

    facts = merge_facts([_money("Food", "₹18,200", 18200, "₹", "Monthly spending"),
                         _money("Travel", "₹11,400", 11400, "₹", "Monthly spending"),
                         _money("Shopping", "₹8,900", 8900, "₹", "Monthly spending")])
    chart = comparable_series(facts)
    assert chart and chart["title"] == "Monthly spending"
    assert {p["label"] for p in chart["series"][0]["points"]} == {"Food", "Travel", "Shopping"}


def test_dollars_and_euros_are_different_units():
    from ragly_backend.visualize import _unit_of_value

    assert _unit_of_value("$23.8") == "$"
    assert _unit_of_value("€4.1") == "€"
    assert _unit_of_value("$411 million") == "$ million"
    assert _unit_of_value("$411 million") != _unit_of_value("$411")


def test_labels_read_as_sentences_not_as_shouting():
    """Sentence case everywhere: first letter capital, the rest as written — but an acronym
    keeps its shape, because "Monthly emi" and "Gst" are not improvements."""
    from ragly_backend.visualize import sentence_case

    assert sentence_case("Loan Amount") == "Loan amount"
    assert sentence_case("Accounts Payable") == "Accounts payable"
    assert sentence_case("Monthly EMI") == "Monthly EMI"
    assert sentence_case("GST payable") == "GST payable"
    assert sentence_case("NS-2048 reference") == "NS-2048 reference"


def test_the_visualisation_never_shows_an_image_grid():
    """Images belong on the Photo search page; here they only competed with the facts."""
    plan = Planner(_app()).plan("")
    assert all(section["kind"] != "images" for section in plan["sections"])
