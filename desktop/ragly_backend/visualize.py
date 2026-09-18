"""The Visualization Planner: machine-extracted knowledge -> a view a person can read.

Why this file exists
--------------------
The knowledge graph answers "what is connected to what". That is the right internal
representation and it stays exactly as it is. It is the wrong thing to put in front of a
person: a node called NS-2048 with ₹18,00,000, 9.25% and 36 months floating around it tells
the reader nothing, because the *meaning* of each number -- which one is the loan, which one
is the instalment -- lives in the sentence the number came from, not in the edge.

So this module sits between the graph and the screen:

    documents -> entities / relations / tables / images   (existing extraction)
              -> facts with a semantic role                (extract_facts, here)
              -> an intent read off the user's question    (classify_intent, here)
              -> one structured visualisation              (plan, here)
              -> the UI renders that structure

The planner picks the *form*: a timeline for dates, a summary of labelled figures for money,
a relationship graph only when the question is actually about relationships.

The label is the document's own words
-------------------------------------
A role lexicon recognises the common ones (loan amount, EMI, interest rate, tenure), but the
general rule is that whatever the document wrote next to the value becomes the label. That is
what makes this work on a lab report or a delivery challan as well as on a loan agreement,
without a per-document-type rule anywhere.

Everything here is deterministic string work: no model, no network, no LLM call. A value with
no readable label is dropped rather than shown, because an unexplained number on a screen is
worse than a shorter list.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable

log = logging.getLogger("ragly.visualize")

MAX_CHUNKS_SCANNED = 400          # a whole workspace is scanned once, then cached
MAX_FACTS_PER_CATEGORY = 12
MAX_GRAPH_NODES = 14              # "8-15 meaningful nodes": the readable ceiling
MAX_TIMELINE = 14

# --------------------------------------------------------------------------- #
# Values
# --------------------------------------------------------------------------- #
#: The scale word is part of the figure: "$411 million" read as "$411" is off by six orders of
#: magnitude, and the card said $411 next to a card saying $8.
_MONEY = re.compile(r"(?:₹|Rs\.?|INR|\$|€|£)\s?\d[\d,]*(?:\.\d+)?"
                    r"(?:\s*(?:lakhs?|lacs?|crores?|millions?|billions?|trillions?|mn|bn|tn))?", re.I)
_PERCENT = re.compile(r"\d+(?:\.\d+)?\s?%(?:\s*p\.?a\.?)?", re.I)
_DURATION = re.compile(r"\b\d+\s*(?:day|days|week|weeks|month|months|year|years|hour|hours)\b", re.I)
_MEASURE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:kg|g|mg|km|m|cm|mm|l|ml|kw|kwh|v|a|hz|nm|mpa|psi|°c|°f)\b", re.I)
_DATE_PATTERNS = (
    re.compile(r"\b(\d{1,2})[ \-](Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?,?[ \-](\d{4})\b", re.I),
    re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?[ \-](\d{1,2}),?[ \-](\d{4})\b", re.I),
    re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b"),
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
)
_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7,
           "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}


def parse_date(text: str) -> date | None:
    """A date out of the handful of forms Indian documents actually use, or None."""
    raw = (text or "").strip()
    for i, pattern in enumerate(_DATE_PATTERNS):
        m = pattern.search(raw)
        if not m:
            continue
        try:
            if i == 0:
                return date(int(m.group(3)), _MONTHS[m.group(2)[:3].lower()], int(m.group(1)))
            if i == 1:
                return date(int(m.group(3)), _MONTHS[m.group(1)[:3].lower()], int(m.group(2)))
            if i == 2:
                day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if day > 12 >= month or day <= 12:          # dd/mm/yyyy is the local convention
                    return date(year, month, day)
                return date(year, day, month)
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except (ValueError, KeyError):
            continue
    return None


#: (matcher on the document's own label, canonical role, category)
#: The canonical role is only used for ordering and for the icon; the label shown to the
#: reader is always the document's wording.
ROLE_LEXICON: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"\b(loan|principal|sanction(ed)?|disburs)\w*\s*(amount|sum)?\b", re.I), "LOAN_AMOUNT", "money"),
    (re.compile(r"\b(emi|instal?ment|monthly\s+payment|monthly\s+due)\b", re.I), "MONTHLY_EMI", "money"),
    (re.compile(r"\b(interest|rate\s+of\s+interest|roi|apr)\b", re.I), "INTEREST_RATE", "rate"),
    (re.compile(r"\b(tenure|tenor|duration|term|period|repayment\s+period)\b", re.I), "DURATION", "duration"),
    (re.compile(r"\b(prepay\w*|pre-?payment|part[- ]payment|foreclosure|pre-?closure)\s*"
                r"(amount|charge|charges|fee)?\b", re.I), "PREPAYMENT", "money"),
    (re.compile(r"\b(processing|service|late|penalty|penal|bounce)\s*(fee|charge|charges)?\b", re.I), "FEE", "money"),
    (re.compile(r"\b(grand\s+total|total|net\s+payable|amount\s+payable|balance)\b", re.I), "TOTAL", "money"),
    (re.compile(r"\b(salary|ctc|gross\s+pay|net\s+pay|remuneration)\b", re.I), "SALARY", "money"),
    (re.compile(r"\b(deposit|advance|down\s*payment|margin)\b", re.I), "DEPOSIT", "money"),
    (re.compile(r"\b(premium|sum\s+assured|cover|coverage|claim)\b", re.I), "INSURANCE", "money"),
    (re.compile(r"\b(tax|gst|igst|cgst|sgst|tds|vat)\b", re.I), "TAX", "money"),
    (re.compile(r"\b(first|next|due)\s*(payment|instal?ment|date)?\b", re.I), "FIRST_PAYMENT", "date"),
    (re.compile(r"\b(agreement|signed|execution|executed|effective|commencement|start)\b", re.I), "START_DATE", "date"),
    (re.compile(r"\b(maturity|expiry|expires?|valid\s*(till|until|upto)|end|closure|last)\b", re.I), "END_DATE", "date"),
    (re.compile(r"\b(issue|issued|invoice|billing|statement)\s*(date)?\b", re.I), "ISSUE_DATE", "date"),
    (re.compile(r"\b(notice|grace|lock[- ]?in|cooling)\s*(period)?\b", re.I), "NOTICE_PERIOD", "duration"),
)

#: Ordering inside a section: the headline number first, the small print after.
ROLE_ORDER = {"LOAN_AMOUNT": 0, "PREPAYMENT": 5.5, "TOTAL": 1, "SALARY": 1, "INSURANCE": 1, "MONTHLY_EMI": 2,
              "INTEREST_RATE": 3, "DURATION": 4, "DEPOSIT": 5, "FEE": 6, "TAX": 7,
              "START_DATE": 0, "ISSUE_DATE": 1, "FIRST_PAYMENT": 2, "END_DATE": 3,
              "NOTICE_PERIOD": 5}

CATEGORY_TITLES = {"money": "Amounts", "rate": "Rates", "duration": "Durations",
                   "date": "Dates", "measure": "Measurements", "other": "Other figures"}

#: Words that are not a label on their own. A value whose only "label" is one of these has
#: nothing readable to show, so it is dropped.
_FILLER = frozenset("""a an the of for to is are was were be been at on in by with and or as
that this these those it its his her their our your my from into per which who whom whose
shall will may can must not no than then there here such any each other same
up upon under over about above below between during within after before""".split())
#: "amount", "total" and "value" stay OUT of the filler list on purpose: "Loan amount" and
#: "Grand total" are the labels the reader needs, and stripping them left a bare "Loan".

_LABEL_TAIL = re.compile(r"[\s:=\-–—]+$")

#: Nouns that finish a label rather than start the next one: "Loan" + "amount", "Due" + "date".
_LABEL_NOUNS = frozenset("""amount amounts rate rates date dates fee fees charge charges period
payment payments total tenure tenor term price value sum instal instalment installment emi
duration months years days cost balance limit deposit premium""".split())

#: Which kind of value each closing noun belongs to. "EMI" + "date" must not become the label
#: of a percentage, so a noun is only appended when it agrees with the value it is labelling.
_NOUN_CATEGORY = {"date": "date", "dates": "date",
                  "rate": "rate", "rates": "rate",
                  "months": "duration", "years": "duration", "days": "duration",
                  "duration": "duration", "period": "duration", "tenure": "duration",
                  "tenor": "duration", "term": "duration"}

#: Words that belong to a test script or a form's scaffolding, not to a value's name. A label
#: built out of these explains nothing, so the value is dropped instead of shown.
_META_LABEL = re.compile(r"\b(expected|ground truth|anchors?|test|tests|query|queries|sample|"
                         r"example|placeholder|lorem|field name|column|row)\b", re.I)
_SENTENCE_SPLIT = re.compile(r"[.;\n•|]")


@dataclass
class Fact:
    """One labelled value, normalised: what it is, what it belongs to, and where it came from.

    ``number`` and ``unit`` are what make two facts comparable (or not). ``subject`` is the
    thing the value belongs to -- the section heading it sits under, which is what separates a
    loan's figures from an invoice's on the same page. ``sources`` accumulates every place the
    same fact was found, so the screen shows one card with "3 references", not three cards.
    """

    label: str                       # the document's own words, e.g. "Monthly EMI"
    value: str                       # the value as printed, e.g. "₹57,530"
    role: str                        # canonical role or "STATED" when only the label is known
    category: str                    # money | rate | duration | date | measure | other
    doc_id: int
    doc_name: str
    page: int
    snippet: str
    confidence: float
    method: str = "label proximity"
    sort_date: str | None = None     # ISO date for timeline ordering
    number: float | None = None      # the value as a number, for comparison and charting
    unit: str = ""                   # ₹ | % | months | kg … two facts only compare within a unit
    subject: str = ""                # the section/heading this value belongs to
    sources: list[dict] = field(default_factory=list)
    conflict: bool = False           # another value claims the same role for the same subject
    ambiguous: bool = False          # this label repeats with several values, so it names nothing

    @property
    def key(self) -> tuple:
        """Identity of the FACT, not of the sentence: same thing said twice is one fact."""
        return (self.subject.lower(), (self.role if self.role != "STATED" else self.label.lower()),
                self.number if self.number is not None else re.sub(r"\s+", "", self.value.lower()),
                self.unit)

    @property
    def role_key(self) -> tuple:
        return (self.subject.lower(), self.role if self.role != "STATED" else self.label.lower())

    def to_dict(self) -> dict[str, Any]:
        first = self.sources[0] if self.sources else {
            "doc_id": self.doc_id, "doc_name": self.doc_name, "page": self.page, "text": self.snippet}
        return {
            "label": self.label, "value": self.value, "role": self.role, "category": self.category,
            "unit": self.unit, "number": self.number, "subject": self.subject,
            "date": self.sort_date,
            "references": len(self.sources) or 1,
            "ambiguous": self.ambiguous,
            "conflict": self.conflict,
            "evidence": {**first, "confidence": round(self.confidence, 2), "method": self.method,
                         "sources": self.sources[:6]},
        }


#: Sentence case: the first letter capital, the rest as written. Acronyms and codes keep their
#: shape -- "Monthly EMI" must not become "Monthly emi", and "NS-2048" must stay itself.
_ACRONYM = re.compile(r"^[A-Z0-9][A-Z0-9&./-]*$")


def sentence_case(text: str) -> str:
    words = str(text or "").split()
    if not words:
        return ""
    out = []
    for i, word in enumerate(words):
        core = word.strip("(),.:;")
        if _ACRONYM.match(core) and len(core) > 1:
            out.append(word)                       # EMI, GST, INR, NS-2048
        elif i == 0:
            out.append(word[:1].upper() + word[1:].lower())
        else:
            out.append(word.lower())
    return " ".join(out)


def _clean_label(raw: str) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z./&'-]*", raw)
    while words and words[0].lower() in _FILLER:
        words.pop(0)
    while words and words[-1].lower() in _FILLER:
        words.pop()
    if not words or len(words) > 4:
        words = words[-3:] if words else []
        while words and words[0].lower() in _FILLER:
            words.pop(0)
    if not words:
        return ""
    label = " ".join(words)
    if len(label) < 3 or all(w.lower() in _FILLER for w in words):
        return ""
    return sentence_case(label)


#: A value is only labelled when the document *states* the label next to it. Two forms count:
#:   "Loan amount: ₹18,00,000"      a separator binds the words to the value
#:   "The loan amount is ₹18,00,000"  a linking verb does the same
#: Anything else — words that merely happen to sit nearby — is not evidence of meaning. That
#: guess is what turned "12 months" into PRINCIPAL and "3%" into EMI.
_SEP_LABEL = re.compile(r"([A-Za-z][A-Za-z ./&'()-]{2,44}?)\s*[:=\u2013-]\s*$")
_VERB_LABEL = re.compile(r"([A-Za-z][A-Za-z ./&'()-]{2,44}?)\s+(?:is|are|was|were|of|at|totall?ing|"
                         r"amounts?\s+to|comes?\s+to|shall\s+be|will\s+be)\s+(?:a\s+|an\s+|the\s+)?$", re.I)
#: Dates are written with a connector rather than a colon far more often than figures are:
#: "executed on 5 September 2026", "valid till 31 March 2027". The words before the connector
#: are the event, and they still have to survive the entity check below.
_WHEN_LABEL = re.compile(r"([A-Za-z][A-Za-z ./&'()-]{2,44}?)\s+(?:on|from|until|till|upto|up\s+to|by|"
                         r"dated|w\.e\.f\.?)\s+$", re.I)

#: A label is a noun phrase. These words mean the capture ran into the sentence instead of a
#: name: "had not yet commenced", "elected to offset", "price or impairments". A 70-page annual
#: report is full of them, and each one became a confident, meaningless card.
_NOT_A_NAME = frozenset("""had has have having was were been being is are will shall may might could
would should must do does did done make makes made take takes taken give gives given
elected commenced commence begin began expects expect expected believes believe believed
incurred recognised recognized recorded reported estimated approved agreed paid pays
maintain maintained maintains offset offsets exceed exceeds exceeded remain remains remained
include includes included excluding relating related pursuant subject whether which that
not yet also than then such any all both either neither because although however therefore
and or but if when while during after before under over between within""".split())

#: A label that is only a connective or a fragment ending in one is not a name either.
_TRAILING_JUNK = frozenset("""or and of to for with in on at by from as than the a an""".split())


def _is_a_name(label: str) -> bool:
    """Does this read as the NAME of a value, rather than part of a sentence about it?"""
    words = [w.lower().strip(".,;:") for w in label.split()]
    if not words or len(words) > 4:
        return False
    if any(w in _NOT_A_NAME for w in words):
        return False
    if words[-1] in _TRAILING_JUNK or words[0] in _TRAILING_JUNK:
        return False
    if not any(len(w) > 2 for w in words):
        return False
    return True


#: An unlabelled value still exists — it is simply not claimed to mean anything.
UNSURE_LABELS = {"money": "Amount mentioned", "rate": "Percentage mentioned",
                 "duration": "Duration mentioned", "measure": "Measurement mentioned",
                 "date": "Date mentioned in document", "other": "Value mentioned"}

#: Words that make a phrase the name of a THING rather than the name of a value. A date
#: labelled "BluePeak Analytics Pvt" is a company standing next to a date, not an event.
_ENTITY_MARKERS = re.compile(r"\b(pvt|ltd|limited|llp|inc|corp|gmbh|plc|bank|analytics|solutions|"
                             r"technologies|systems|services|industries|enterprises|mr|mrs|ms|dr|shri|smt)\b",
                             re.I)
#: …unless the phrase names an event, which is what a date is allowed to be labelled with.
_EVENT_WORDS = re.compile(r"\b(date|dated|due|signed|executed|effective|start|starts|started|begin|"
                          r"commence\w*|expir\w*|end|ends|ended|matur\w*|renew\w*|issue[d]?|payment|"
                          r"instal?ment|emi|deadline|valid|review|inspection|approval|delivery|"
                          r"completion|submission|closing|agreement)\b", re.I)


def _label_for(text: str, start: int, end: int, category: str = "") -> tuple[str, float]:
    """The label the document STATES for this value, or nothing.

    Returns ("", 0.0) when the document does not say what the value is. The caller then shows
    it as "Amount mentioned" rather than inventing a meaning for it.
    """
    left = _SENTENCE_SPLIT.split(text[max(0, start - 110):start])[-1]

    patterns = [(_SEP_LABEL, 0.95), (_VERB_LABEL, 0.85)]
    if category == "date":
        patterns.append((_WHEN_LABEL, 0.8))
    for pattern, confidence in patterns:
        m = pattern.search(left)
        if not m:
            continue
        label = _clean_label(m.group(1))
        if not label or _META_LABEL.search(label):
            continue
        if category == "date":
            # An event is named with a verb — "Agreement signed", "Payment due" — so the
            # noun-phrase test does not apply; what it must have is an event word.
            if not _EVENT_WORDS.search(label):
                continue
            label = re.sub(r"\b(is|are|was|were|has|have|had|been|be|shall|will|to)\b", " ", label, flags=re.I)
            label = re.sub(r"\s{2,}", " ", label).strip(" -–—,")
            if not label:
                continue
            label = label[:1].upper() + label[1:]
        elif not _is_a_name(label):
            continue                       # a sentence fragment is not a label
        # the label has to be a name, not half a sentence that ran into the value
        for lex, _role, lex_category in ROLE_LEXICON:
            if (lex_category == "date") != (category == "date"):
                continue
            hit = list(lex.finditer(label))
            if hit:
                label = label[hit[-1].start():].strip()
                break
        if len(label.split()) > 5:
            label = " ".join(label.split()[-4:])
        label = label[:1].upper() + label[1:]
        if category == "date" and _ENTITY_MARKERS.search(label):
            # "Signed by BluePeak Analytics Pvt. Ltd" -> "Signed". Cut the party out of the
            # event name; if nothing recognisable as an event is left, this is not an event.
            head = re.split(r"\s+(?:by|with|to|from|for|between)\s+", label, maxsplit=1)[0].strip()
            if not head or _ENTITY_MARKERS.search(head) or not _EVENT_WORDS.search(head):
                continue
            label = head
        return label, confidence

    # "₹57,530 per month" / "36 months tenure": a unit phrase immediately after the value
    right = _SENTENCE_SPLIT.split(text[end:end + 40])[0]
    m = re.match(r"\s*(per\s+(?:month|annum|year|day)|monthly|annually|per\s+unit)\b", right, re.I)
    if m and category in ("money", "measure"):
        return _clean_label(m.group(1)) or "", 0.6
    return "", 0.0


def _canonical(label: str, default_category: str) -> tuple[str, str]:
    for pattern, role, category in ROLE_LEXICON:
        if pattern.search(label):
            # a lexicon date role only applies to a date, a money role only to money
            if category in ("date",) and default_category != "date":
                continue
            if category in ("money", "rate", "duration") and default_category == "date":
                continue
            return role, (default_category if category == "money" and default_category != "money"
                          else category)
    return "STATED", default_category


def _snippet(text: str, start: int, end: int, width: int = 130) -> str:
    lo = max(0, start - width // 2)
    hi = min(len(text), end + width // 2)
    out = re.sub(r"\s+", " ", text[lo:hi]).strip()
    return ("… " if lo else "") + out + (" …" if hi < len(text) else "")


def extract_facts(text: str, doc_id: int = 0, doc_name: str = "", page: int = 1,
                  subject: str = "") -> list[Fact]:
    """Every value in this passage that the document itself gave a name to."""
    facts: list[Fact] = []
    seen: set[tuple[str, str]] = set()
    scanners: tuple[tuple[re.Pattern[str], str], ...] = (
        (_MONEY, "money"), (_PERCENT, "rate"), (_DURATION, "duration"), (_MEASURE, "measure"),
    )
    spans: list[tuple[int, int, str, str]] = []
    for pattern, category in scanners:
        for m in pattern.finditer(text):
            spans.append((m.start(), m.end(), m.group(0).strip(), category))
    for pattern in _DATE_PATTERNS:
        for m in pattern.finditer(text):
            spans.append((m.start(), m.end(), m.group(0).strip(), "date"))

    spans.sort()
    for start, end, value, category in spans:
        value = value.strip().rstrip(",;")
        label, confidence = _label_for(text, start, end, category)
        if label:
            role, category = _canonical(label, category)
        else:
            # The document never says what this value is. It is kept, plainly labelled, and
            # it never reaches the headline panel or a chart -- guessing a role here is
            # exactly the bug this whole layer exists to prevent.
            role, confidence = "UNKNOWN", 0.3
            label = UNSURE_LABELS.get(category, "Value mentioned")
        flat_value = re.sub(r"[\s,]+", "", value.lower())
        key = (label.lower(), flat_value)
        role_key = (role, flat_value)
        if key in seen or (role != "STATED" and role_key in seen):
            continue                                  # the same figure under two headings
        seen.add(key)
        seen.add(role_key)
        when = parse_date(value) if category == "date" else None
        evidence = {"doc_id": doc_id, "doc_name": doc_name, "page": page,
                    "text": _snippet(text, start, end)}
        facts.append(Fact(label=label, value=value, role=role, category=category,
                          doc_id=doc_id, doc_name=doc_name, page=page,
                          snippet=evidence["text"],
                          confidence=min(1.0, confidence + (0.1 if role not in ("STATED", "UNKNOWN") else 0.0)),
                          sort_date=when.isoformat() if when else None,
                          number=_number_of(value) if category != "date" else None,
                          unit=_unit_of_value(value) or _UNIT_OF.get(category, ""),
                          subject=subject or doc_name,
                          sources=[evidence]))
    return facts


# --------------------------------------------------------------------------- #
# Intent
# --------------------------------------------------------------------------- #
INTENTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("timeline", re.compile(r"\b(timeline|dates?|when|chronolog\w*|schedule|deadlines?|milestones?)\b", re.I)),
    ("financial", re.compile(r"\b(financial|money|amount|amounts|cost|costs|price|payment|payments|emi|"
                             r"loan|interest|rate|salary|invoice|billing|charges?|fees?|terms)\b", re.I)),
    ("relationships", re.compile(r"\b(relationship|relationships|connected|connection|related|links?|"
                                 r"who\s+is|people|parties|organi[sz]ations?|companies|network)\b", re.I)),
    ("charts", re.compile(r"\b(charts?|graphs?|plot|plots|bar|line|donut|pie|visuali[sz]ations?)\b", re.I)),
    ("table", re.compile(r"\b(table|tables|rows?|columns?|spreadsheet)\b", re.I)),
    ("comparison", re.compile(r"\b(compare|comparison|difference|differences|versus|vs\.?|changed)\b", re.I)),
    ("images", re.compile(r"\b(images?|photos?|pictures?|figures?|diagrams?|scans?)\b", re.I)),
    ("overview", re.compile(r"\b(overview|summar\w+|everything|this document|the document|visuali[sz]e this)\b", re.I)),
)

_FOCUS = re.compile(r"\b(?:how\s+(?:is|are)|explain\s+how|connection\s+between|path\s+between)\b", re.I)


def classify_intent(query: str | None) -> str:
    """What kind of picture the question is asking for."""
    q = (query or "").strip()
    if not q:
        return "overview"
    if _FOCUS.search(q):
        return "focus"
    for name, pattern in INTENTS:
        if pattern.search(q):
            return name
    return "overview"


def choose_type(intent: str, evidence: dict[str, int]) -> str:
    """The visualisation that fits, given the question AND what the documents actually hold.

    The question leads; the counts stop the planner promising a timeline for a document with
    two dates in it, or a relationship graph for a document with no relationships.
    """
    dates, money, relations, tables, images = (evidence.get(k, 0) for k in
                                               ("dates", "money", "relations", "tables", "images"))
    if intent == "timeline":
        return "timeline" if dates >= 2 else "document_overview"
    if intent == "financial":
        return "financial_summary" if money else "document_overview"
    if intent == "relationships":
        return "relationship_graph" if relations else "document_entity_map"
    if intent == "focus":
        return "focused_path" if relations else "document_entity_map"
    if intent == "charts":
        return "charts" if (tables or money) else "document_overview"
    if intent == "table":
        return "table_visualization" if tables else "document_overview"
    if intent == "comparison":
        return "comparison_view"
    if intent == "images":
        return "image_map" if images else "document_overview"
    return "document_overview"


# --------------------------------------------------------------------------- #
# Relationships, in words
# --------------------------------------------------------------------------- #
#: Edge name -> what to print on the arrow. An edge with no phrase here is not shown by
#: default: if it cannot be said in plain English it does not belong on the screen.
REL_PHRASES: dict[str, tuple[str, str]] = {
    # relation            (phrase,                    explicit | inferred)
    "works_at":           ("works at",                "explicit"),
    "employed_by":        ("works at",                "explicit"),
    "issued_by":          ("issued by",               "explicit"),
    "issued_to":          ("issued to",               "explicit"),
    "has_amount":         ("is for",                  "explicit"),
    "dated":              ("dated",                   "explicit"),
    "located_in":         ("is located in",           "explicit"),
    "part_of":            ("is part of",              "explicit"),
    "owns":               ("owns",                    "explicit"),
    "signed_by":          ("signed by",               "explicit"),
    "party_to":           ("is a party to",           "explicit"),
    "mentions":           ("mentions",                "explicit"),
    "contains":           ("contains",                "explicit"),
    "valid_for":          ("lasts for",               "explicit"),
    # NOTE: "co_occurs_with" and "related_to" are deliberately absent. Two entities sitting in
    # the same chunk is retrieval evidence, not a fact about the world -- drawing it produced
    # "Axis Bank appears alongside Bengaluru", which tells a reader nothing and is not even
    # true in the way an arrow implies. The graph backend keeps those edges; the picture does not.
}

KIND_LABELS = {"org": "Organisation", "person": "Person", "invoice": "Invoice", "contract": "Agreement",
               "project": "Project", "place": "Place", "code": "Reference", "part_number": "Part",
               "amount": "Amount", "date": "Date", "duration": "Duration", "id": "Identifier",
               "image": "Image", "document": "Document"}


def describe_relation(rel: str) -> tuple[str, str]:
    """(phrase, explicit|inferred) for an edge, or ("", "") when it should not be drawn."""
    phrase = REL_PHRASES.get(rel)
    if phrase:
        return phrase
    if rel in ("co_occurs_with", "related_to", "near", "same_page", "same_chunk"):
        return "", ""                      # proximity, not meaning
    readable = rel.replace("_", " ").strip()
    if readable and re.fullmatch(r"[a-z ]{3,30}", readable):
        return readable, "inferred"
    return "", ""


#: A relationship graph is about who and what, not about the numbers. Amounts, dates and
#: durations are facts ABOUT an entity and are shown as labelled values instead -- putting
#: them on the canvas is exactly what made the old graph unreadable.
ENTITY_KINDS = frozenset({"org", "person", "project", "place", "contract", "invoice",
                          "code", "part_number", "document"})


#: The roles that make up the headline panel of a document, in reading order. One card per
#: role -- never one card per number found.
KEY_ROLE_ORDER = ("LOAN_AMOUNT", "TOTAL", "SALARY", "INSURANCE", "DEPOSIT", "INTEREST_RATE",
                  "DURATION", "MONTHLY_EMI", "PREPAYMENT", "FEE", "TAX", "NOTICE_PERIOD")

ROLE_TITLES = {"LOAN_AMOUNT": "Loan amount", "PREPAYMENT": "Prepayment", "MONTHLY_EMI": "Monthly payment",
               "INTEREST_RATE": "Interest rate", "DURATION": "Duration", "FEE": "Fee",
               "TOTAL": "Total", "SALARY": "Salary", "DEPOSIT": "Deposit", "TAX": "Tax",
               "INSURANCE": "Cover", "NOTICE_PERIOD": "Notice period",
               "START_DATE": "Start", "END_DATE": "End", "FIRST_PAYMENT": "First payment",
               "ISSUE_DATE": "Issued"}


def _clean_subject(heading: str) -> str:
    """A section heading, tidied into something that can head a group of facts."""
    text = re.sub(r"\s+", " ", str(heading or "")).strip(" .:-—")
    text = re.sub(r"^\d+[.)]\s*", "", text)
    if not text or len(text) > 60 or _META_LABEL.search(text):
        return ""
    letters = sum(1 for c in text if c.isalpha())
    if letters < 3 or letters < len(text) / 2:
        return ""                          # "2,460" is a figure someone used as a heading
    # "Agreement date: 2 August 2026" is a fact someone wrote as a heading, not a section
    if _MONEY.search(text) or _PERCENT.search(text) or any(p.search(text) for p in _DATE_PATTERNS):
        return ""
    return sentence_case(text)


def merge_facts(facts: Iterable[Fact]) -> list[Fact]:
    """Turn a stream of extractions into a set of distinct facts.

    Three things happen here, and they are the difference between a visualisation and a
    debug dump:

    1. the same fact found on three pages becomes ONE fact with three sources;
    2. two different values claiming the same role for the same subject are both kept and
       both marked ``conflict`` -- silently picking one would be inventing an answer;
    3. facts keep the subject they belong to, so a loan's figures and an invoice's totals
       never end up in the same group.
    """
    merged: dict[tuple, Fact] = {}
    for fact in facts:
        existing = merged.get(fact.key)
        if existing is None:
            merged[fact.key] = fact
            continue
        for source in fact.sources:
            if source not in existing.sources:
                existing.sources.append(source)
        if fact.confidence > existing.confidence:
            existing.label, existing.confidence = fact.label, fact.confidence

    # One label, many different values ("Issuance $0.1, Issuance $3.8, Issuance $23.8 …") does
    # not tell a reader which is which. Nine identical cards is not information, so these are
    # held back from the screen rather than repeated.
    by_label: dict[tuple, list[Fact]] = {}
    for fact in merged.values():
        by_label.setdefault((fact.subject.lower(), fact.label.lower()), []).append(fact)
    for group in by_label.values():
        if len({f.number for f in group}) > 2:
            for fact in group:
                fact.ambiguous = True

    by_role: dict[tuple, list[Fact]] = {}
    for fact in merged.values():
        by_role.setdefault(fact.role_key, []).append(fact)
    for role_key, group in by_role.items():
        if len(group) > 1 and role_key[1] != "stated":
            for fact in group:
                fact.conflict = True
    return sorted(merged.values(),
                  key=lambda f: (ROLE_ORDER.get(f.role, 9), -f.confidence, -len(f.sources), f.label))


def key_facts(facts: Iterable[Fact]) -> list[Fact]:
    """The headline panel: one fact per semantic role, best evidence first.

    A document that states the loan amount on four pages has ONE loan amount. Where a role
    genuinely has two different values they are both kept, because they are both real.
    """
    chosen: dict[str, list[Fact]] = {}
    for fact in facts:
        if fact.role in ("STATED", "") or fact.category == "date":
            continue
        chosen.setdefault(fact.role, []).append(fact)
    out: list[Fact] = []
    for role in KEY_ROLE_ORDER:
        group = chosen.get(role)
        if not group:
            continue
        group.sort(key=lambda f: (-len(f.sources), -f.confidence))
        out.append(group[0])
        for extra in group[1:2]:                      # a second, different value is shown too
            if extra.number != group[0].number:
                out.append(extra)
    return out


def group_by_subject(facts: Iterable[Fact], limit: int = 6) -> list[tuple[str, list[Fact]]]:
    """Facts grouped by the thing they describe, largest group first."""
    groups: dict[str, list[Fact]] = {}
    for fact in facts:
        if fact.category == "date":
            continue
        groups.setdefault(fact.subject or "Other", []).append(fact)
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    return [(name, items[:MAX_FACTS_PER_CATEGORY]) for name, items in ordered[:limit]]


def comparable_series(facts: Iterable[Fact]) -> dict | None:
    """A chart ONLY when every value is the same measure, in the same unit.

    Two things may be charted:

    * several values of the SAME canonical role (four invoice totals, three salaries);
    * a labelled breakdown under one real section heading, where every label is different and
      appears exactly once — a spending list, a headcount by team.

    Everything else gets cards. The old rule ("same subject, same unit") produced the chart
    titled "values compared" that put an issuance, a legal provision and a construction
    commitment on one axis because all three were dollars in the same file.
    """
    pool = [f for f in facts
            if f.number is not None and f.category != "date" and not f.ambiguous and f.unit]
    buckets: dict[tuple, list[Fact]] = {}
    for fact in pool:
        if fact.role not in ("STATED", "UNKNOWN", ""):
            buckets.setdefault(("role", fact.role, fact.unit), []).append(fact)
        elif fact.subject and fact.subject.lower() not in (fact.doc_name or "").lower():
            # a breakdown only counts when it sits under a real heading, not under the file name
            buckets.setdefault(("breakdown", fact.subject.lower(), fact.unit), []).append(fact)

    best: list[Fact] = []
    kind = ""
    for (bucket_kind, _name, _unit), group in buckets.items():
        labels = [f.label.lower() for f in group]
        if len(group) < 3 or len(set(labels)) != len(labels):
            continue                      # a repeated label means the labels do not identify anything
        if len(group) > len(best):
            best, kind = group, bucket_kind
    if not best:
        return None

    unit = best[0].unit
    points = [{"label": f.label, "value": f.number, "display": f.value,
               "evidence": {"doc_name": f.doc_name, "page": f.page}} for f in best[:MAX_CHART_POINTS]]
    if kind == "role":
        name = ROLE_TITLES.get(best[0].role, best[0].role.replace("_", " ").title())
        title = f"{name} across this document"
    else:
        name = best[0].subject
        title = f"{best[0].subject}"
    return {
        "kind": "bar", "title": title, "unit": unit, "x": "", "y": name,
        "series": [{"name": name, "points": points}],
        "note": "Every value here is the same measure in the same unit.",
    }


#: What a topic word in the question means in terms of roles. "Loan terms" asks for the
#: amount, the rate, the tenure and the instalment -- not only for facts with "loan" in the
#: label. Without this the filter drops exactly the figures the reader asked to see.
QUERY_FAMILIES: tuple[tuple[re.Pattern[str], frozenset[str]], ...] = (
    (re.compile(r"\b(loan|emi|mortgage|borrow\w*|repay\w*|instal?ment)\b", re.I),
     frozenset({"LOAN_AMOUNT", "MONTHLY_EMI", "INTEREST_RATE", "DURATION", "FEE", "DEPOSIT",
                "FIRST_PAYMENT", "START_DATE", "END_DATE"})),
    (re.compile(r"\b(invoice|bill|billing|receipt|purchase)\b", re.I),
     frozenset({"TOTAL", "TAX", "FEE", "ISSUE_DATE"})),
    (re.compile(r"\b(salary|payroll|ctc|pay)\b", re.I), frozenset({"SALARY", "TAX", "DEPOSIT"})),
    (re.compile(r"\b(insurance|policy|claim|cover\w*)\b", re.I),
     frozenset({"INSURANCE", "TOTAL", "DURATION", "END_DATE"})),
    (re.compile(r"\b(interest|rate)\b", re.I), frozenset({"INTEREST_RATE"})),
)

#: Words that name the KIND of view being asked for, not a topic inside the document. The
#: intent already routes on these, so treating them as topic words would filter out exactly
#: the facts they ask for ("show financial information" matched no label and emptied the page).
_GENERIC_QUERY_WORDS = frozenset("""show visualize visualise these this that with about from into what
which information document documents important please give data chart charts graph graphs
terms term details detail summary summarise summarize everything overview all financial finance
money amounts figures numbers values relationship relationships connected dates timeline
tables table images picture pictures""".split())


def _table_matches(table: dict, query: str | None) -> bool:
    """Is this table about what was asked? A bank statement's chart does not belong under
    "visualize the loan terms" just because both contain rupees."""
    words = {w for w in re.findall(r"[a-z]{4,}", (query or "").lower()) if w not in _GENERIC_QUERY_WORDS}
    if not words:
        return True
    haystack = " ".join([str(table.get("title") or "")] + [str(h) for h in (table.get("headers") or [])]).lower()
    return any(w in haystack or w.rstrip("s") in haystack for w in words)


def _relevant_to(facts: list[Fact], query: str | None, intent: str = "") -> list[Fact]:
    """Drop facts the question did not ask about.

    "Visualize the loan terms" should not pull in an unrelated invoice total sitting in the
    same workspace. Two things stop this from being too blunt: a topic word brings in its
    whole family of roles, and dates are never filtered out by wording -- a timeline is about
    when things happen, not about which words the question used.
    """
    text = query or ""
    words = {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in _GENERIC_QUERY_WORDS}
    family: set[str] = set()
    for pattern, roles in QUERY_FAMILIES:
        if pattern.search(text):
            family |= set(roles)
    if not words and not family:
        return facts
    if intent in ("timeline", "images", "table", "comparison"):
        return facts

    def touches(fact: Fact) -> bool:
        if fact.category == "date" and family:
            return fact.role in family or fact.role == "STATED"
        if fact.role in family:
            return True
        haystack = f"{fact.label} {fact.subject} {fact.role}".lower().replace("_", " ")
        return any(w in haystack or w.rstrip("s") in haystack for w in words)

    return [f for f in facts if touches(f)]


#: A two-column table is a list of labelled values: the row label names the value beside it.
#: This is the most reliable label a document can give, so table facts outrank prose facts.
_VALUE_HEADERS = frozenset({"value", "values", "amount", "amounts", "figure", "details", "detail",
                            "particulars", "description", "term", "terms", "item", "field"})


def facts_from_table(table: dict) -> list[Fact]:
    """Facts from a key/value table, where the structure itself carries the meaning.

    | Loan Amount   | ₹18,00,000 |      ->  LOAN_AMOUNT = ₹18,00,000
    | Interest Rate | 9.25%      |      ->  INTEREST_RATE = 9.25%

    Only two-column tables are read this way. A wide table is data to chart, not a list of
    labelled facts, and pairing an arbitrary cell with an arbitrary header would be the same
    guessing this module exists to avoid.
    """
    headers = [str(h) for h in (table.get("headers") or [])]
    rows = [list(r) for r in (table.get("rows") or [])]
    if len(headers) != 2 or not rows:
        return []
    subject = _clean_subject(table.get("title") or "") or str(table.get("doc_name") or "")
    out: list[Fact] = []
    for row in rows:
        if len(row) < 2:
            continue
        label, printed = _clean_label(str(row[0])), str(row[1]).strip()
        if not label or not printed or _META_LABEL.search(label):
            continue
        category = ("money" if _MONEY.search(printed) else
                    "rate" if _PERCENT.search(printed) else
                    "duration" if _DURATION.search(printed) else
                    "date" if any(p.search(printed) for p in _DATE_PATTERNS) else
                    "measure" if _MEASURE.search(printed) else "")
        if not category:
            continue                                   # a row of prose is not a fact card
        role, category = _canonical(label, category)
        when = parse_date(printed) if category == "date" else None
        evidence = {"doc_id": int(table.get("doc_id") or 0), "doc_name": str(table.get("doc_name") or ""),
                    "page": int(table.get("page") or 1), "table_id": table.get("id"),
                    "row_label": str(row[0]).strip(), "column_label": headers[1],
                    "text": f"{str(row[0]).strip()} | {printed}"}
        out.append(Fact(label=label, value=printed, role=role, category=category,
                        doc_id=evidence["doc_id"], doc_name=evidence["doc_name"],
                        page=evidence["page"], snippet=evidence["text"],
                        confidence=0.98, method="table row",
                        sort_date=when.isoformat() if when else None,
                        number=_number_of(printed) if category != "date" else None,
                        unit=_unit_of_value(printed) or _UNIT_OF.get(category, ""),
                        subject=subject, sources=[evidence]))
    return out


class Planner:
    """Builds one visualisation from what is already indexed. No model, no network."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self.store = app.store
        self._facts_cache: dict[str, list[Fact]] = {}

    # ---------------- facts ----------------
    def facts(self, doc_ids: list[int] | None = None) -> list[Fact]:
        """Every distinct labelled value in scope, merged and grouped. Cached per scope."""
        key = ",".join(str(d) for d in sorted(doc_ids)) if doc_ids else "all"
        if key in self._facts_cache:
            return self._facts_cache[key]
        docs = [d for d in self.store.list_documents() if d.get("status") == "ready"]
        if doc_ids:
            wanted = set(int(i) for i in doc_ids)
            docs = [d for d in docs if int(d["id"]) in wanted]
        raw: list[Fact] = []
        budget = MAX_CHUNKS_SCANNED
        for doc in docs:
            ids = self.store.chunk_ids(int(doc["id"]))[:budget]
            budget -= len(ids)
            rows = self.store.get_chunks(ids)
            for chunk_id in ids:
                row = rows.get(chunk_id)
                if not row:
                    continue
                # the heading the passage sits under is what the values belong to; without one
                # the document itself is the subject
                subject = _clean_subject(row.get("heading") or "") or str(row["doc_name"])
                raw.extend(extract_facts(row["text"], int(row["doc_id"]), row["doc_name"],
                                         int(row["page"]), subject))
            if budget <= 0:
                break
        # a table states its labels outright, so those facts come first and win on merge
        try:
            for table in self.store.list_tables(doc_ids):
                raw = facts_from_table(table) + raw
        except Exception as exc:
            log.debug("no tables for facts: %s", exc)
        facts = merge_facts(raw)
        self._facts_cache[key] = facts
        return facts

    # ---------------- pieces of the view ----------------
    def _fact_section(self, title: str, facts: Iterable[Fact], limit: int = MAX_FACTS_PER_CATEGORY) -> dict | None:
        items = [f.to_dict() for f in list(facts)[:limit]]
        return {"title": title, "kind": "facts", "items": items} if items else None

    def _timeline(self, facts: Iterable[Fact]) -> list[dict]:
        """Events, in order. A date the document did not name is not an event."""
        dated = [f for f in facts
                 if f.category == "date" and f.sort_date and f.role not in ("UNKNOWN", "")]
        dated.sort(key=lambda f: (f.sort_date or "", f.label))
        seen: set[str] = set()
        out: list[dict] = []
        for fact in dated:
            key = f"{fact.sort_date}|{fact.label.lower()}"
            if key in seen:
                continue
            seen.add(key)
            out.append(fact.to_dict())
            if len(out) >= MAX_TIMELINE:
                break
        return out

    def _parties(self, limit: int = 10) -> list[dict]:
        people = self.store.top_nodes("person", limit)
        orgs = self.store.top_nodes("org", limit)
        out = []
        for node in sorted(list(people) + list(orgs), key=lambda n: -int(n.get("mentions") or 0))[:limit]:
            out.append({"label": KIND_LABELS.get(node["kind"], node["kind"].title()),
                        "value": node["name"], "role": node["kind"].upper(), "category": "party",
                        "node_id": int(node["id"]), "mentions": int(node.get("mentions") or 0)})
        return out

    def _relationships(self, seed: str | None = None, limit: int = MAX_GRAPH_NODES) -> dict:
        """A small, readable neighbourhood: the most-mentioned entities and the edges we can
        put into words. Never the whole graph -- that is unreadable by construction."""
        graph = getattr(self.app, "graph", None)
        if graph is None:
            return {"nodes": [], "edges": [], "seed": None}

        seeds: list[dict] = []
        if seed:
            try:
                seeds = graph.resolve(seed, limit=1)
            except Exception as exc:
                log.debug("seed %r did not resolve: %s", seed, exc)
        if not seeds:
            seeds = list(self.store.top_nodes(None, 3))
        if not seeds:
            return {"nodes": [], "edges": [], "seed": None}

        nodes: dict[int, dict] = {}
        edges: list[dict] = []
        pairs: set[tuple[int, int, str]] = set()
        for start in seeds[:3]:
            hood = self.store.node_neighbourhood(int(start["id"]), limit=40)
            node = hood.get("node") or start
            nodes.setdefault(int(node["id"]), {
                "id": int(node["id"]), "name": node["name"], "kind": node["kind"],
                "type": KIND_LABELS.get(node["kind"], node["kind"].title()),
                "mentions": int(node.get("mentions") or 0), "seed": True})
            for edge in hood.get("edges", []):
                phrase, source = describe_relation(str(edge["rel"]))
                if not phrase:
                    continue
                if str(edge["src_kind"]) not in ENTITY_KINDS or str(edge["dst_kind"]) not in ENTITY_KINDS:
                    continue                          # values belong in the cards, not on the canvas
                src, dst = int(edge["src"]), int(edge["dst"])
                if (src, dst, phrase) in pairs:
                    continue
                pairs.add((src, dst, phrase))
                for side, name, kind in ((src, edge["src_name"], edge["src_kind"]),
                                         (dst, edge["dst_name"], edge["dst_kind"])):
                    nodes.setdefault(side, {"id": side, "name": name, "kind": kind,
                                            "type": KIND_LABELS.get(kind, str(kind).title()),
                                            "mentions": 0, "seed": False})
                edges.append({
                    "source": src, "target": dst, "label": phrase, "rel": str(edge["rel"]),
                    "origin": source,
                    "confidence": 0.9 if source == "explicit" else 0.5,
                    "note": "" if source == "explicit" else "Inferred from document context",
                    "evidence": {"doc_name": edge.get("doc_name", ""), "page": edge.get("page", 0),
                                 "text": edge.get("snippet", "") or edge.get("text", "")},
                })

        # keep it readable: the best few nodes, and only the edges between those
        ranked = sorted(nodes.values(), key=lambda n: (not n["seed"], -n["mentions"], n["name"]))[:limit]
        keep = {n["id"] for n in ranked}
        edges = [e for e in edges if e["source"] in keep and e["target"] in keep]
        # what the document states comes first; what we merely noticed comes after
        edges.sort(key=lambda e: (e["origin"] != "explicit", e["label"]))
        hidden = max(0, len(nodes) - len(ranked))
        return {"nodes": ranked, "edges": edges, "seed": seeds[0]["name"] if seeds else None,
                "hidden": hidden}

    def _tables(self, doc_ids: list[int] | None, limit: int = 10) -> list[dict]:
        try:
            tables = self.store.list_tables(doc_ids)
        except Exception as exc:
            log.debug("no tables: %s", exc)
            return []
        out = []
        for t in tables[:limit]:
            headers = list(t.get("headers") or [])
            rows = [list(r) for r in (t.get("rows") or [])][:12]
            chart = donut_from_table(headers, t.get("rows") or []) \
                or _chart_from_table(headers, t.get("rows") or [])
            out.append({"title": t.get("title") or "Table", "kind": "table",
                        "headers": headers, "rows": rows,
                        "chart": chart,
                        "evidence": {"doc_id": t.get("doc_id"), "doc_name": t.get("doc_name"),
                                     "page": t.get("page"), "table_id": t.get("id"),
                                     "confidence": 1.0, "method": "table extraction"}})
        return out

    def _images(self, limit: int = 8) -> list[dict]:
        try:
            images = self.store.get_images(limit=limit)
        except Exception:
            return []
        return [{"id": int(i["id"]), "doc_name": i.get("doc_name", ""), "page": int(i.get("page") or 0),
                 "objects": (i.get("objects") or "").split(",") if i.get("objects") else [],
                 "description": i.get("description") or ""} for i in images[:limit]]

    # ---------------- the plan ----------------
    def plan(self, query: str | None = None, doc_ids: list[int] | None = None) -> dict:
        """Decide what belongs on the screen, then fill it. Extraction never decides."""
        intent = classify_intent(query)
        docs = [d for d in self.store.list_documents() if d.get("status") == "ready"]

        # ---- scope: one document unless the question asks across them -------------------
        # "Visualize this document" must not blend four unrelated PDFs into one picture.
        cross_document = intent in ("comparison", "relationships") or bool(
            re.search(r"\b(all|across|every|both|compare|documents)\b", query or "", re.I))
        if doc_ids:
            docs = [d for d in docs if int(d["id"]) in set(int(i) for i in doc_ids)]
        elif len(docs) > 1 and not cross_document:
            docs = docs[:1]                             # newest first, per list_documents()
            doc_ids = [int(docs[0]["id"])]

        all_facts = self.facts(doc_ids)
        # A value the document never named is kept in the model (the debug panel counts it)
        # but never rendered: showing "Amount mentioned: ₹34,869" is the extraction artefact
        # this layer exists to keep off the screen.
        # Prose gives a label with a separator ("Loan amount: …") at 0.95 and a table row at
        # 0.98. Anything weaker is a guess from a sentence, which is what filled an annual
        # report's screen with "HAD NOT YET COMMENCED". It stays in the model, not on screen.
        named = [f for f in all_facts
                 if f.role != "UNKNOWN" and f.confidence >= 0.9 and not f.ambiguous]
        facts = _relevant_to(named, query, intent)
        # A question about something this document does not contain gets said plainly, not
        # answered with whatever else happened to be lying around.
        asked_for_something = bool(query) and facts != named
        missing = asked_for_something and len(facts) == 0
        timeline = self._timeline(facts)
        tables = self._tables(doc_ids)
        images = self._images()
        parties = self._parties()
        headline = key_facts(facts)
        groups = group_by_subject(facts)
        counts = {"dates": len(timeline),
                  "money": sum(1 for f in facts if f.category in ("money", "rate")),
                  "relations": 0, "tables": len(tables), "images": len(images)}

        relationships: dict = {"nodes": [], "edges": []}
        if intent in ("relationships", "focus", "overview"):
            relationships = self._relationships(query if intent == "focus" else None)
            counts["relations"] = len(relationships.get("edges", []))

        kind = choose_type(intent, counts)
        if missing and intent not in ("relationships", "focus", "images", "table", "comparison"):
            kind = "insufficient_evidence"
        title = docs[0]["name"] if len(docs) == 1 else f"{len(docs)} documents"

        sections: list[dict] = []
        notes: list[str] = []

        def key_terms_section() -> dict | None:
            if len(headline) < 2:
                return None
            return {"title": "Key terms", "kind": "kpi",
                    "items": [f.to_dict() for f in headline[:8]]}

        def chart_section(chart: dict | None, evidence: dict | None = None) -> None:
            if chart:
                sections.append({"title": chart["title"], "kind": "chart", "items": [],
                                 "chart": chart, "evidence": evidence})

        if kind == "insufficient_evidence":
            notes.append(f"Nothing in {title} answers that. The document was read in full; "
                         "no value it states matches what you asked for.")
            sections.append({"title": "What this document does state", "kind": "facts",
                             "items": [f.to_dict() for f in key_facts(named)[:6]]})
        elif kind == "timeline":
            sections.append({"title": "What happens when", "kind": "timeline", "items": timeline})
        elif kind == "financial_summary":
            # the headline panel first: one card per meaning, not one per number
            panel = key_terms_section()
            if panel:
                sections.append(panel)
            for name, items in groups:
                remaining = [f for f in items if f not in headline]
                if remaining:
                    sections.append({"title": name, "kind": "facts",
                                     "items": [f.to_dict() for f in remaining]})
            # a chart only when the numbers are genuinely comparable
            chart_section(comparable_series(facts))
            shown = 0
            for table in tables:
                if table.get("chart") and _table_matches(table, query):
                    chart_section(table["chart"], table.get("evidence"))
                    shown += 1
                    if shown >= 2:
                        break
            if timeline:
                sections.append({"title": "Dates", "kind": "timeline", "items": timeline[:6]})
        elif kind in ("relationship_graph", "focused_path"):
            sections.append({"title": "How these are connected", "kind": "graph",
                             "items": [], "graph": relationships})
            if parties:
                sections.append({"title": "Who and what appears", "kind": "facts",
                                 "items": [dict(p, evidence={"confidence": 1.0,
                                                             "method": "entity extraction"}) for p in parties]})
        elif kind == "charts":
            for table in tables:
                if table.get("chart"):
                    chart_section(table["chart"], table.get("evidence"))
            chart_section(comparable_series(facts))
            if not any(sec["kind"] == "chart" for sec in sections):
                notes.append("Nothing in this document can be charted honestly: a chart needs "
                             "several values of the same kind, in the same unit. The figures it "
                             "does state are shown as cards instead.")
                panel = key_terms_section()
                if panel:
                    sections.append(panel)
        elif kind == "table_visualization":
            for table in tables:
                if table.get("chart"):
                    chart_section(table["chart"], table.get("evidence"))
                sections.append(table)
        elif kind == "image_map":
            # Images live on the Photo search page; the visualisation is about the document's
            # facts, and a grid of thumbnails here only competed with them.
            notes.append("Images are on the Photo search page.")
        elif kind == "comparison_view":
            notes.append("Pick the two documents to compare on the Compare page.")
            sections.append({"title": "Documents", "kind": "facts",
                             "items": [{"label": "Document", "value": d["name"], "role": "DOCUMENT",
                                        "category": "document",
                                        "evidence": {"doc_id": int(d["id"]), "doc_name": d["name"],
                                                     "page": 1, "confidence": 1.0,
                                                     "method": "index"}} for d in docs[:8]]})
        else:                                           # document_overview / entity map
            panel = key_terms_section()
            if panel:
                sections.append(panel)
            if timeline:
                sections.append({"title": "Important dates", "kind": "timeline", "items": timeline[:8]})
            if relationships.get("edges"):
                sections.append({"title": "How these are connected", "kind": "graph",
                                 "items": [], "graph": relationships})
            elif parties:
                sections.append({"title": "People and organisations", "kind": "facts",
                                 "items": [dict(p, evidence={"confidence": 1.0,
                                                             "method": "entity extraction"}) for p in parties]})
            chart_section(comparable_series(facts)
                          or next((t["chart"] for t in tables if t.get("chart")), None))
            for name, items in groups[:2]:
                remaining = [f for f in items if f not in headline]
                if remaining:
                    sections.append({"title": name, "kind": "facts",
                                     "items": [f.to_dict() for f in remaining[:8]]})

        conflicts = [f for f in headline if f.conflict]
        if conflicts:
            notes.append(f"{len(conflicts)} figures appear with more than one value in this "
                         "document. Both are shown — open one to see where each came from.")
        if len(self.store.list_documents()) > 1 and not cross_document:
            notes.append(f"Showing {title} only. Ask about \u201call documents\u201d to include the rest.")

        if not sections:
            notes.append("Nothing in these documents could be labelled clearly enough to show. "
                         "Ask a question on the Ask page instead.")

        inferred = sum(1 for e in relationships.get("edges", []) if e["origin"] == "inferred")
        if inferred:
            notes.append(f"{inferred} of the connections are inferred from the documents' wording, "
                         "not stated outright. They are marked with a dashed line.")

        unsure = [f for f in all_facts if f.role == "UNKNOWN"]
        debug = {
            "visualization": kind,
            "reason": _why(kind, intent, query),
            "documents_in_scope": [d["name"] for d in docs],
            "documents_ignored": max(0, len(self.store.list_documents()) - len(docs)),
            "facts_kept": len(facts),
            "facts_without_a_stated_label": len(unsure),
            "labels_that_repeat_with_different_values": len([f for f in all_facts if f.ambiguous]),
            "unlabelled_examples": [f"{f.value} ({f.label})" for f in unsure[:5]],
            "headline_roles": [f.role for f in headline],
            "chart": next((s2["chart"]["title"] for s2 in sections if s2["kind"] == "chart"), None),
            "chart_reason": ("values of one kind were found" if any(s2["kind"] == "chart" for s2 in sections)
                             else "no set of comparable values, so no chart"),
            "relationship_edges": len(relationships.get("edges", [])),
            "relationship_rule": "only relationships the document states; proximity is never an edge",
        }
        return {
            "type": kind,
            "intent": intent,
            "debug": debug,
            "query": query or "",
            "title": title,
            "summary": _summary_line(kind, counts, len(docs)),
            "sections": sections,
            "relationships": relationships,
            "timeline": timeline,
            "counts": counts,
            "notes": notes,
        }


def _why(kind: str, intent: str, query: str | None) -> str:
    asked = f'the question was "{query}"' if query else "no question was asked, so this is the overview"
    return f"{kind} chosen because {asked} (intent: {intent})."


def _summary_line(kind: str, counts: dict[str, int], docs: int) -> str:
    where = "1 document" if docs == 1 else f"{docs} documents"
    if kind == "timeline":
        return f"{counts['dates']} dated events found in {where}, in order."
    if kind == "financial_summary":
        return f"{counts['money']} labelled figures found in {where}."
    if kind in ("relationship_graph", "focused_path"):
        return f"{counts['relations']} connections that can be stated in plain English."
    if kind == "table_visualization":
        return f"{counts['tables']} tables found in {where}."
    if kind == "image_map":
        return f"{counts['images']} images found in {where}."
    verb = "contains" if docs == 1 else "contain"
    return f"What {where} {verb}, grouped by meaning."


#: A chart is only honest when everything on it is the same kind of thing. Rupees and
#: percentages on one axis is a picture that lies, so the planner charts one unit at a time.
_UNIT_OF = {"money": "₹", "rate": "%", "duration": "", "measure": "", "date": ""}
MAX_CHART_POINTS = 12
MIN_CHART_POINTS = 2


def _unit_of_value(value: str) -> str:
    """The unit, including WHICH currency: $ and € are not the same axis."""
    if re.search(r"[₹]|\b(?:inr|rs\.?|rupees?)\b", value, re.I):
        return "₹"
    for symbol in ("$", "€", "£", "¥"):
        if symbol in value:
            # a scale word changes the unit too: "$411 million" is not "$411"
            scale = re.search(r"\b(million|billion|trillion|crore|lakh|mn|bn|tn)s?\b", value, re.I)
            return f"{symbol} {scale.group(1).lower()}" if scale else symbol
    if "%" in value:
        return "%"
    m = re.search(r"\d+\s*([A-Za-z]{1,6})$", value.strip())
    return m.group(1) if m else ""


def _number_of(value: str) -> float | None:
    from .tables import parse_number

    return parse_number(value)


def chart_from_facts(facts: list[Fact], category: str) -> dict | None:
    """Compare the labelled values of one kind — the figures a reader wants side by side.

    "Loan amount ₹18,00,000 / EMI ₹57,530 / Fee ₹28,890" as three bars says *at a glance*
    which one is large, which the cards alone do not. Values of different units are never
    put on the same axis.
    """
    same = [f for f in facts if f.category == category]
    if len(same) < MIN_CHART_POINTS:
        return None
    units = {_unit_of_value(f.value) for f in same}
    unit = units.pop() if len(units) == 1 else ""
    if len(units) > 0:                                  # mixed units: pick the majority unit only
        counted: dict[str, int] = {}
        for f in same:
            u = _unit_of_value(f.value)
            counted[u] = counted.get(u, 0) + 1
        unit = max(counted, key=lambda u: counted[u])
        same = [f for f in same if _unit_of_value(f.value) == unit]
        if len(same) < MIN_CHART_POINTS:
            return None
    points = []
    for fact in same[:MAX_CHART_POINTS]:
        number = _number_of(fact.value)
        if number is None:
            continue
        points.append({"label": fact.label, "value": number,
                       "display": fact.value,
                       "evidence": {"doc_name": fact.doc_name, "page": fact.page}})
    if len(points) < MIN_CHART_POINTS:
        return None
    biggest = max(p["value"] for p in points)
    smallest = min(p["value"] for p in points)
    return {
        "kind": "bar",
        "title": f"{CATEGORY_TITLES.get(category, 'Figures')} side by side",
        "unit": unit or _UNIT_OF.get(category, ""),
        "x": "", "y": CATEGORY_TITLES.get(category, ""),
        "series": [{"name": CATEGORY_TITLES.get(category, "Value"), "points": points}],
        "note": ("Every bar is drawn to the same scale."
                 if biggest / max(smallest, 1e-9) < 500 else
                 "The largest value dwarfs the smallest, so the small bars are only just visible."),
    }


_DATEISH = re.compile(r"\b(\d{1,2}[ \-/](?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)|"
                      r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}|"
                      r"\d{1,2}[/-]\d{1,2})", re.I)


def _looks_like_dates(values: list[str]) -> bool:
    """A column of dates, including the short forms parse_date cannot turn into a real date
    ("02 Sep", "5/9"). Getting this wrong drew a bank statement as bars instead of a line."""
    usable = [v for v in values if v.strip()]
    if not usable:
        return False
    parsed = sum(1 for v in usable if parse_date(v) or _DATEISH.search(v))
    return parsed >= max(2, int(0.8 * len(usable)))


def donut_from_table(headers: list, rows: list) -> dict | None:
    """A donut only when the rows genuinely add up to a stated whole.

    A table ending in TOTAL where the parts sum to it is a composition; anything else is not,
    and drawing a donut of numbers that do not make a whole is a lie about the data.
    """
    from .tables import _is_total_row, parse_number

    if not headers or len(headers) < 2 or len(rows) < 3:
        return None
    label_col, value_col = 0, None
    for index in range(1, len(headers)):
        column = [str(r[index]) if index < len(r) else "" for r in rows]
        if sum(1 for c in column if parse_number(c) is not None) >= len(column) - 1:
            value_col = index
            break
    if value_col is None:
        return None
    parts, total = [], None
    for row in rows:
        label = str(row[label_col]).strip() if label_col < len(row) else ""
        value = parse_number(str(row[value_col])) if value_col < len(row) else None
        if not label or value is None:
            continue
        if _is_total_row(row, headers):
            total = value
        else:
            parts.append({"label": label, "value": value, "display": str(row[value_col]).strip()})
    if total is None or len(parts) < 3:
        return None
    summed = sum(p["value"] for p in parts)
    if summed <= 0 or abs(summed - total) / max(total, 1e-9) > 0.02:
        return None                                    # the parts do not make the whole
    unit = _unit_of_value(str(rows[0][value_col]))
    return {"kind": "donut", "title": f"{headers[value_col]} by {headers[label_col]}",
            "unit": unit, "x": str(headers[label_col]), "y": str(headers[value_col]),
            "total": total,
            "series": [{"name": str(headers[value_col]), "points": parts[:MAX_CHART_POINTS]}],
            "note": f"These add up to the table's stated total of {rows[-1][value_col]}."}


def _chart_from_table(headers: list, rows: list) -> dict | None:
    """A chart of a table: a line when the labels are dates, bars otherwise.

    Several number columns are drawn as grouped bars (one colour per column) rather than
    being refused — what the planner will not do is put columns of different *units* on one
    axis, or chart a column of years and call it a measurement.
    """
    if not headers or len(headers) < 2 or len(rows) < MIN_CHART_POINTS:
        return None

    label_col: int | None = None
    value_cols: list[int] = []
    for index in range(len(headers)):
        column = [str(r[index]) if index < len(r) else "" for r in rows]
        numeric = sum(1 for c in column if _number_of(c) is not None)
        header = str(headers[index]).strip().lower()
        index_like = header in ("#", "sl", "sl.", "s.no", "sno", "no", "no.", "sr", "sr.", "id")
        # a column of dates or years is a label, never a measurement: charting "Date" as a
        # height produced a bar chart of calendar numbers
        if re.search(r"\b(date|dates|day|month|months|year|years|period|quarter|fy)\b", header):
            index_like = True
        if all(re.fullmatch(r"(19|20)\d{2}", c.strip() or "x") for c in column if c.strip()):
            index_like = True
        if numeric >= max(2, int(0.8 * len(column))) and not index_like and not _looks_like_dates(column):
            value_cols.append(index)
        elif label_col is None and any(c.strip() for c in column):
            label_col = index
    if label_col is None or not value_cols:
        return None
    value_cols = value_cols[:3]                          # more than three series is unreadable

    # one unit per chart: a column of rupees and a column of counts are not comparable
    units = {}
    for index in value_cols:
        column = [str(r[index]) for r in rows if index < len(r)]
        found = {u for u in (_unit_of_value(c) for c in column) if u}
        units[index] = found.pop() if len(found) == 1 else ""
    if len(set(units.values())) > 1:
        # Money is what a reader came to compare, then percentages, then a plain count. A
        # quantity column and a rupee column on one axis would be meaningless, so only the
        # columns sharing the winning unit are drawn.
        preference = {"₹": 0, "%": 1}
        keep = sorted(set(units.values()), key=lambda u: (preference.get(u, 2 if u else 3), u))[0]
        value_cols = [i for i in value_cols if units[i] == keep]
        if not value_cols:
            return None

    labels = [str(r[label_col]).strip() if label_col < len(r) else "" for r in rows]
    dated = _looks_like_dates([l for l in labels if l])
    order = list(range(len(rows)))
    if dated:
        order.sort(key=lambda i: (parse_date(labels[i]) or date.max))

    series = []
    for index in value_cols:
        points = []
        for i in order[:MAX_CHART_POINTS]:
            row = rows[i]
            label, raw = labels[i], (str(row[index]) if index < len(row) else "")
            number = _number_of(raw)
            if label and number is not None:
                points.append({"label": label, "value": number, "display": raw.strip()})
        if len(points) >= MIN_CHART_POINTS:
            series.append({"name": str(headers[index]), "points": points})
    if not series:
        return None
    unit = units.get(value_cols[0], "")
    return {
        "kind": "line" if dated else ("grouped_bar" if len(series) > 1 else "bar"),
        "title": f"{', '.join(s['name'] for s in series)} by {headers[label_col]}",
        "unit": unit,
        "x": str(headers[label_col]),
        "y": series[0]["name"] if len(series) == 1 else "",
        "series": series,
        "note": "In date order." if dated else "",
    }
