"""Evidence sufficiency: does the retrieved text actually answer THIS question?

The failure this module exists to stop:

    Q: "What are the Axis Bank loan interest rates?"
    Retrieved: "Loan EMI transactions should be categorised as Loan EMI."   <- same topic
    Old behaviour: the model writes something plausible about loans.
    New behaviour: abstain, and say what the documents DO contain.

A relevant topic is not sufficient evidence. We check three things, in plain terms:
  1. Entity   - if the question names something specific (Axis Bank, INV-1002, Substation 4),
                that thing must appear in the evidence.
  2. Attribute- the thing being asked FOR (interest rate, tenure, notice period, torque) must
                appear, or a close variant of it.
  3. Value    - for a question that wants a number or a date, the evidence must actually contain
                one near that attribute.

Everything is rule-based and local: no extra model, a fraction of a millisecond.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .autotune import ENTITY_PATTERNS

# ---------------------------------------------------------------- question shapes
# A question only counts as numeric when it asks for a quantity, not merely when it mentions one.
NUMERIC_CUES = re.compile(
    r"\b(how much|how many|how long|how old)\b"
    # up to three words may sit between "what is the" and the measured noun:
    # "what is the earth bolt torque", "what is the maximum permissible working pressure"
    r"|\bwhat(?:'s| is| are| was| were)?\s+(?:the\s+)?(?:[a-z-]+\s+){0,3}"
    r"(rate|interest|amount|total|cost|price|fee|charges?|balance|value|torque|pressure|voltage|"
    r"current|limit|emi|tenure|duration|percentage|salary|quantity|expenditure|spend)\b", re.I)
DATE_CUES = re.compile(
    r"\b(when|what date|which date|date of|expiry|expires?|due|deadline|effective from|signed on|"
    r"valid (?:till|until)|start(?:s|ed)? on|renewal)\b", re.I)
PROCEDURE_CUES = re.compile(r"\b(how do i|how to|procedure|steps?|process for|what should i do|instructions)\b", re.I)
SUMMARY_CUES = re.compile(r"\b(summari[sz]e|summary|overview|what is this document|key points|tl;dr)\b", re.I)

#: A file name in a question names the document to read, not a fact to verify. "Summarize
#: invoice-2026.pdf" must not be refused because the words "invoice-2026.pdf" never appear
#: inside the document -- almost no document contains its own file name.
FILENAME = re.compile(r"\b[\w][\w .()\[\]&-]*?\.(?:pdf|docx?|dotx|xlsx?|xlsm|csv|tsv|txt|md|markdown|log|json|"
                      r"pptx?|png|jpe?g|bmp|tiff?|webp)\b", re.I)
LIST_CUES = re.compile(r"\b(list|which|what documents|show me all|find all)\b", re.I)

# words that are never the "thing being asked for"
STOP = set("""what which who whom whose when where why how is are was were be been do does did the a an of for
to in on at by with from as and or not no any all this that these those it its their our your my me you we
they he she please tell show give find i need want about into than then there here can could should would may
might will shall must if else per each every some other another same such only just more most less least very
document documents file files page pages say says said mention mentions mentioned contain contains
much many long old actual get got spend spent pay paid make made take took give given use used know
tell told state states stated provide provided list lists listed total overall exactly specific
original revised latest current new old previous updated earlier later first second final draft
appear appears appeared occur occurs happen happens start starts started begin begins began end ends
refer refers referring stand stands denote denotes indicate indicates signify signifies represent represents
issue issued issues cover covers covered include includes included apply applies applied require requires
allow allows mean means display displays trigger triggers arise arises come comes read reads""".split())

# Words that name the KIND of value asked for, as opposed to the subject it belongs to.
# "warranty period" is a period (measure) of a warranty (subject): the subject is what must be present.
MEASURE_WORDS = set("""rate rates interest period periods fee fees cost costs price prices amount amounts
date dates deadline value values torque pressure voltage current limit limits charge charges balance
tenure duration quantity number numbers percentage percent total headcount salary emi payment payments
name names time times""".split())

# attribute synonyms: asking for one should accept the others in the evidence
SYNONYMS: dict[str, tuple[str, ...]] = {
    "interest": ("interest rate", "rate of interest", "roi", "apr", "per annum", "p.a."),
    "rate": ("rate", "%", "per cent", "percent", "per annum"),
    "tenure": ("tenure", "term", "duration", "period", "months", "years"),
    "emi": ("emi", "instalment", "installment", "monthly payment"),
    "fee": ("fee", "charge", "charges", "processing fee", "commission"),
    "notice": ("notice", "notice period", "terminate", "termination"),
    "torque": ("torque", "nm", "n·m", "tighten"),
    "price": ("price", "amount", "cost", "value", "inr", "rs", "₹", "usd", "$"),
    "amount": ("amount", "total", "value", "inr", "rs", "₹", "sum"),
    "salary": ("salary", "ctc", "remuneration", "compensation", "pay"),
    "warranty": ("warranty", "guarantee", "warranty period"),
    "address": ("address", "located", "registered office"),
    "expiry": ("expiry", "expires", "valid till", "valid until", "end date"),
    "penalty": ("penalty", "late fee", "interest at", "liquidated damages"),
}

VALUE_PATTERNS = {
    "number": re.compile(r"\b\d+(?:[.,]\d+)*\b"),
    "percent": re.compile(r"\b\d+(?:\.\d+)?\s?%|\bper\s?cent\b|\bpercent\b", re.I),
    "money": ENTITY_PATTERNS["money"],
    "measurement": ENTITY_PATTERNS["measurement"],
    "duration": ENTITY_PATTERNS["duration"],
    "date": ENTITY_PATTERNS["date"],
}


# A measure word is also satisfied by a value of that kind: "notice period" is answered by "60 days",
# even though the word "period" never appears.
MEASURE_VALUE = {
    "period": "duration", "periods": "duration", "tenure": "duration", "duration": "duration",
    "time": "duration", "times": "duration",
    "date": "date", "dates": "date", "deadline": "date",
    "rate": "percent", "rates": "percent", "interest": "percent", "percentage": "percent", "percent": "percent",
    "fee": "money", "fees": "money", "cost": "money", "costs": "money", "price": "money", "prices": "money",
    "amount": "money", "amounts": "money", "salary": "money", "balance": "money", "emi": "money",
    "payment": "money", "payments": "money", "charge": "money", "charges": "money",
    "torque": "measurement", "pressure": "measurement", "voltage": "measurement", "current": "measurement",
    "limit": "measurement", "limits": "measurement", "value": "number", "values": "number",
    "number": "number", "numbers": "number", "quantity": "number", "total": "number", "headcount": "number",
}


@dataclass
class QuestionShape:
    intent: str                      # factual | numeric | date | procedure | summary | list
    entities: list[str] = field(default_factory=list)      # specific things named in the question
    attributes: list[str] = field(default_factory=list)    # what is being asked FOR
    wants_value: str | None = None   # number | percent | money | duration | date | None

    def to_dict(self) -> dict:
        return {"intent": self.intent, "entities": self.entities, "attributes": self.attributes,
                "wants_value": self.wants_value}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def question_entities(question: str) -> list[str]:
    """Specific things: capitalised names, codes, invoice numbers, quoted phrases."""
    found: list[str] = []
    for m in re.finditer(r"\"([^\"]{2,60})\"", question):
        found.append(m.group(1))
    # Capitalised runs that are not the first word of the sentence
    for m in re.finditer(r"(?<!^)(?<![.!?]\s)\b([A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*){0,3})\b", question):
        token = m.group(1).strip()
        if _norm(token) and _norm(token) not in STOP and len(token) > 2:
            found.append(token)
    for kind in ("code", "part_number", "clause_ref"):
        for m in ENTITY_PATTERNS[kind].finditer(question.upper()):
            found.append((m.group(1) if m.groups() else m.group(0)).strip())
    seen, out = set(), []
    for f in found:
        k = _norm(f)
        if k and k not in seen:
            seen.add(k)
            out.append(f)
    return out[:6]


def question_attributes(question: str) -> list[str]:
    """The thing being asked for, as content words (minus the entity names)."""
    entities = {_norm(e) for e in question_entities(question)}
    words = [w for w in re.findall(r"[a-zA-Z][\w-]{2,}", question.lower()) if w not in STOP]
    attrs = [w for w in words if not any(w in e.split() for e in entities)]
    # keep two-word phrases that read like attributes ("interest rate", "notice period")
    phrases = re.findall(r"\b([a-z]{3,}\s(?:rate|period|fee|amount|date|number|limit|charge|cost|value|torque))\b",
                         question.lower())
    return list(dict.fromkeys(phrases + attrs))[:8]


def classify_question(question: str) -> QuestionShape:
    q = question or ""
    # file names scope the search; they are never quoted in the text, so they are removed
    # before the question is broken into things to look for
    subject = FILENAME.sub(" ", q)
    intent = "factual"
    wants = None
    if SUMMARY_CUES.search(q):
        intent = "summary"
    elif PROCEDURE_CUES.search(q):
        intent = "procedure"
    elif DATE_CUES.search(q):
        intent, wants = "date", "date"
    elif NUMERIC_CUES.search(q):
        intent, wants = "numeric", "number"
    elif LIST_CUES.search(q):
        intent = "list"
    low = q.lower()
    if wants == "number":
        if re.search(r"\b(rate|interest|percentage|percent|%)\b", low):
            wants = "percent"
        elif re.search(r"\b(amount|price|cost|fee|salary|emi|total|spend|expenditure|paid|₹|inr|rs\.?|usd|\$)\b", low):
            wants = "money"
        elif re.search(r"\b(tenure|duration|how long|period|months|years|days)\b", low):
            wants = "duration"
    return QuestionShape(intent=intent, entities=question_entities(subject),
                         attributes=question_attributes(subject), wants_value=wants)


def _expanded(attr: str) -> tuple[str, ...]:
    for key, syns in SYNONYMS.items():
        if attr == key or attr in syns:
            return syns
    return (attr,)


def _topics(text: str, limit: int = 6) -> list[str]:
    """A few words describing what a passage is actually about (for the abstention message)."""
    words = [w for w in re.findall(r"[a-zA-Z][\w-]{3,}", text.lower()) if w not in STOP]
    counts: dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    return [w for w, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:limit]]


@dataclass
class Sufficiency:
    sufficient: bool
    score: float
    entity_found: bool
    attribute_found: bool
    value_found: bool
    missing: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    found_instead: list[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict:
        return self.__dict__


def assess(question: str, hits, shape: QuestionShape | None = None, min_score: float = 0.5) -> Sufficiency:
    """Decide whether these passages can answer this question."""
    shape = shape or classify_question(question)
    if not hits:
        return Sufficiency(False, 0.0, False, False, False, ["any matching passage"],
                           ["nothing was retrieved"], [],
                           "I couldn't find anything in the indexed documents for that question.")

    blob = "\n".join(h.text for h in hits)
    low = _norm(blob)
    compact = re.sub(r"[^a-z0-9]", "", blob.lower())

    def present(term: str) -> bool:
        """A code (E-63, INV-1003, SF6) may be written with or without punctuation; a plain word must
        appear as a word, so that "CEO" is not found inside "invoice order"."""
        t = term.strip()
        if any(ch.isdigit() for ch in t) or len(re.sub(r"[^a-z0-9]", "", t.lower())) < len(t.replace(" ", "")):
            return _norm(t) in low or re.sub(r"[^a-z0-9]", "", t.lower()) in compact
        return bool(re.search(r"\b" + re.escape(_norm(t)).replace(r"\ ", r"\s+") + r"\b", low))

    def word_present(word: str) -> bool:
        """Word-stem match: "approved" is answered by "Approver", "appears" by "appear"."""
        for variant in _expanded(word):
            v = _norm(variant)
            if not v:
                continue
            if " " in v:
                if v in low:
                    return True
                continue
            stem = v[:max(4, len(v) - 2)]
            if re.search(r"\b" + re.escape(stem) + r"[a-z]{0,4}\b", low):
                return True
            if re.sub(r"[^a-z0-9]", "", v) in compact and any(ch.isdigit() for ch in v):
                return True
        return False

    # 1. entity - EVERY named thing must be there. "Who is the CEO of Bharat Switchgear?" is not
    #    answered by a page that mentions Bharat Switchgear but never a CEO.
    entity_found = True
    missing: list[str] = []
    if shape.entities:
        absent = [e for e in shape.entities if not present(e)]
        entity_found = not absent
        if absent:
            missing.extend(absent[:2])

    # 2. attribute - the subject words carry the question's specificity, the measure words only say
    #    what kind of value is wanted. "cable gland clamp bolt torque" is not answered by a page that
    #    happens to contain "bolt" and "torque".
    entity_words = {re.sub(r"[^a-z0-9]", "", e.lower()) for e in shape.entities}
    singles = [a for a in shape.attributes
               if " " not in a and re.sub(r"[^a-z0-9]", "", a.lower()) not in entity_words]
    subjects = [a for a in singles if a not in MEASURE_WORDS]
    measures = [a for a in singles if a in MEASURE_WORDS]
    attribute_found = True
    if shape.attributes:
        def measure_present(word: str) -> bool:
            if word_present(word):
                return True
            kind = MEASURE_VALUE.get(word)
            return bool(kind and VALUE_PATTERNS[kind].search(blob))

        hit_subjects = [a for a in subjects if word_present(a)]
        hit_measures = [a for a in measures if measure_present(a)]
        coverage = (len(hit_subjects) / len(subjects)) if subjects else 1.0
        attribute_found = coverage >= 0.75 and (bool(hit_measures) or not measures)
        if not subjects and not measures:
            # Everything the question named was the entity itself ("What does NS-2048 mean?"):
            # there is no separate detail to look for, so the entity check already decided this.
            phrases = [a for a in shape.attributes if " " in a]
            attribute_found = (not phrases) or any(v in low for a in phrases for v in _expanded(a)) \
                or any(present(a) for a in shape.attributes)
        if not attribute_found:
            gaps = [a for a in subjects if a not in hit_subjects] or [a for a in measures if a not in hit_measures]
            missing.extend(gaps[:2] or shape.attributes[:2])

    # 3. value of the right kind (only when the question asks for one)
    value_found = True
    if shape.wants_value:
        pattern = VALUE_PATTERNS.get(shape.wants_value, VALUE_PATTERNS["number"])
        value_found = bool(pattern.search(blob))
        if not value_found and shape.wants_value == "money":
            # a table of amounts often has bare numbers with no currency symbol
            value_found = bool(re.search(r"\b\d{3,}(?:[.,]\d+)*\b", blob))
        if not value_found and shape.wants_value == "duration":
            value_found = bool(re.search(r"\b\d+\s*(?:day|week|month|year|hour|minute)s?\b", blob, re.I))
        if not value_found:
            missing.append({"percent": "a percentage", "money": "an amount",
                            "duration": "a duration", "date": "a date"}.get(shape.wants_value, "a number"))

    score = (0.4 * entity_found) + (0.35 * attribute_found) + (0.25 * value_found)
    best = max((h.vector_score or 0) for h in hits)
    # every applicable check must pass: a related topic is not sufficient evidence
    sufficient = entity_found and attribute_found and value_found and best > 0.3 and score >= min_score
    # summaries and "show me everything" questions only need relevant material
    if shape.intent in ("summary", "list") and best > 0.35 and entity_found:
        sufficient = True

    reasons = []
    reasons.append(("the named subject appears in the evidence" if entity_found
                    else f"the evidence never mentions {', '.join(shape.entities[:2])}"))
    reasons.append(("the requested detail appears in the evidence" if attribute_found
                    else f"the evidence does not discuss {', '.join(shape.attributes[:2])}"))
    if shape.wants_value:
        reasons.append("a value of the expected kind is present" if value_found
                       else f"no {shape.wants_value} appears in the evidence")

    found_instead = _topics(hits[0].text) if hits else []
    message = ""
    if not sufficient:
        want = ", ".join(dict.fromkeys(missing)) or "that information"
        where = f"{hits[0].doc_name}, page {hits[0].page}" if hits else "the indexed documents"
        message = (f"I couldn't find {want} in the indexed documents. "
                   f"The closest material is {where}, which covers "
                   f"{', '.join(found_instead[:4]) or 'other topics'} — it does not state it.")
    return Sufficiency(sufficient, round(score, 3), entity_found, attribute_found, value_found,
                       list(dict.fromkeys(missing)), reasons, found_instead, message)
