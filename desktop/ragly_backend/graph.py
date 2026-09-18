"""Local entity + relationship graph: rules only, no machine learning.

Why this file exists
--------------------
Keyword and vector search both answer "which passage looks like this question".
Neither answers "show me everything related to ABC Ltd", because the interesting
facts about ABC Ltd are spread over several documents and are only *connected*
through the entities they share (an invoice, an amount, a person, a project).
This module builds that connection layer:

    text -> entities -> nodes/edges/mentions in SQLite -> a third ranked list
                                                          of chunk ids

IMPORTANT - these are hand-written rules, not a trained NER model.
There is no spaCy, no transformer, no statistical model anywhere in here: just
regular expressions, a short gazetteer and same-sentence proximity cues. That
keeps it offline, CPU-cheap and deterministic, but it also means the output is
*approximate*. Expect it to miss entities it has no rule for (an organisation
without a company suffix, a person without an honorific or a role cue) and to
occasionally invent one (a capitalised phrase that merely looks like a company).
Every extraction and every relation in here is a heuristic guess, so treat the
graph as a navigation aid ("these documents mention the same things"), never as
a source of legal or financial truth. Nothing here should be presented to a user
as extracted fact without the citation that backs it.

Public surface
--------------
    extract_entities(text)                  -> list[dict]
    infer_relations(entities, text)         -> list[tuple[str, str, str]]
    entity_key(entity)                      -> "kind:norm"
    is_entity_question(question)            -> bool
    GraphBuilder(store).index_chunks(...)   -> dict   (write side)
    GraphRetriever(store).resolve/expand/chunk_ids_for/summary   (read side)

Everything is deterministic: entities come back in document order, relations in
the order the sentences are read, and every read-side list is explicitly sorted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from .autotune import ENTITY_PATTERNS, STOP, normalise_code
from .store import Store

# --------------------------------------------------------------------------- #
# Limits (the graph is a navigation aid, so it is capped rather than complete)
# --------------------------------------------------------------------------- #
MAX_ENTITIES_PER_CALL = 200      # per extract_entities() call
MAX_RELATIONS_PER_CALL = 400     # per infer_relations() call
MAX_FALLBACK_ENTITIES = 12       # entities per sentence considered for fallback
#: All pairs of the first MAX_FALLBACK_ENTITIES entities of a sentence. Capping
#: below this would truncate the pair list by position, which silently drops the
#: links of whatever is mentioned last in a long sentence.
MAX_FALLBACK_PAIRS = MAX_FALLBACK_ENTITIES * (MAX_FALLBACK_ENTITIES - 1) // 2
MAX_GRAPH_NODES = 200            # per expand() call
MAX_NEIGHBOURS = 60              # edges/mentions fetched per node
MAX_HOPS = 2

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

#: Legal-form suffixes. These are stripped from an organisation's ``norm`` so
#: that "ABC Ltd", "ABC Limited" and "ABC Pvt. Ltd." collapse onto one node.
LEGAL_SUFFIXES: tuple[str, ...] = (
    "private limited", "pvt ltd", "pvt", "ltd", "limited", "llp", "inc", "incorporated",
    "corporation", "corp", "gmbh", "plc", "co", "company",
)

#: Descriptive suffixes. They are enough to *recognise* an organisation but are
#: deliberately kept in the ``norm``: "Acme Systems" and "Acme Solutions" are
#: different companies, so stripping these would merge unrelated nodes.
DESCRIPTIVE_SUFFIXES: tuple[str, ...] = (
    "technologies", "technology", "solutions", "industries", "enterprises", "systems",
    "services", "labs", "laboratories", "hospital", "clinic", "university", "college", "bank",
)

#: Deliberately tiny gazetteer - a heuristic, not a geography database. Only
#: places common in the documents this product was built for (Indian business
#: paperwork) plus a handful of frequent international ones. A place that is not
#: in this list is simply not recognised as a place; that is expected.
PLACE_GAZETTEER: tuple[str, ...] = (
    "Bengaluru", "Bangalore", "Mumbai", "New Delhi", "Delhi", "Chennai", "Hyderabad", "Pune",
    "Kolkata", "Ahmedabad", "Jaipur", "Kochi", "Coimbatore", "Noida", "Gurugram", "Gurgaon",
    "Karnataka", "Maharashtra", "Tamil Nadu", "Kerala", "Gujarat", "Telangana", "Rajasthan",
    "India", "Singapore", "Dubai", "UAE", "USA", "UK", "London", "New York",
)

#: Honorifics that make a following capitalised phrase a person with reasonable
#: confidence ("Dr. S. Menon", "Shri R. Iyer").
HONORIFICS: tuple[str, ...] = (
    "Mr", "Mrs", "Ms", "Miss", "Dr", "Prof", "Shri", "Sri", "Smt", "Capt", "Col", "Er", "Adv",
)

#: Role cues: a capitalised phrase right after one of these is read as a person
#: ("Signed by Rajesh Kumar", "Inspector: A. Rao").
PERSON_CUES: tuple[str, ...] = (
    "signed by", "countersigned by", "inspected by", "inspector", "reviewed by", "approved by",
    "prepared by", "verified by", "checked by", "authorised by", "authorized by", "submitted by",
    "examined by", "attested by", "witness", "attn", "attention", "contact person",
)

#: Words dropped from the front of an org/contract/person/project name so that
#: "This Agreement" and "The Master Services Agreement" normalise sensibly.
DETERMINERS: frozenset[str] = frozenset(
    {"a", "an", "the", "this", "that", "these", "those", "said", "such", "present", "our", "their", "its",
     # sentence words that regularly get swept into a capitalised name and produce
     # entities like "No Axis Bank" or "Profile Northstar Edge Analytics Pvt. Ltd."
     "no", "not", "profile", "note", "notes", "see", "per", "via", "from", "for", "with", "about",
     "table", "figure", "page", "section", "annexure", "schedule", "clause", "if", "when", "then",
     "both", "each", "every", "any", "all", "some", "expected", "observation", "summary"}
)

#: Tokens that look like a sentence end but are not, so the sentence splitter
#: does not cut "Invoice No. INV-1001" in half.
_ABBREVIATIONS: frozenset[str] = frozenset(
    """no nos inc ltd pvt co corp mr mrs ms dr prof shri smt sri st rd vs etc approx dept govt
    fig eg ie jan feb mar apr may jun jul aug sep sept oct nov dec sq ext min max ref hon llp
    plc gmbh pp vol al rs inv gst pan tan cin""".split()
)

#: Cue phrases that make a question an "entity question" for the router.
ENTITY_QUESTION_CUES: tuple[str, ...] = (
    "everything related to", "everything about", "everything on", "everything we have on",
    "all related to", "anything related to", "anything about", "related to", "relating to",
    "show me all", "show me everything", "show all", "list all", "find all", "give me all",
    "documents about", "document about", "docs about", "all documents", "which documents",
    "what do we have on", "who is", "who was", "who are", "which vendor", "which supplier",
    "which company", "which client", "which customer", "connected to", "linked to",
    "associated with", "all invoices", "tell me about",
)

#: Every relation this module can produce, with the edge weight it is stored
#: with. Specific relations are worth more than a bare co-occurrence.
REL_WEIGHTS: dict[str, float] = {
    "works_for": 1.0,
    "issued_by": 1.0,
    "has_amount": 1.0,
    "dated": 1.0,
    "belongs_to": 1.0,
    "between": 1.0,
    "signed_on": 1.0,
    "has_term": 1.0,
    "located_in": 1.0,
    "co_occurs_with": 0.3,
}

# --------------------------------------------------------------------------- #
# Patterns
# --------------------------------------------------------------------------- #
_ORG_SUFFIX = (
    r"(?:Private\s+Limited|Pvt\.?\s*Ltd\.?|Ltd\.?|Limited|LLP|Inc\.?|Incorporated|Corporation"
    r"|Corp\.?|GmbH|PLC|Co\.|Company|Technologies|Technology|Solutions|Industries|Enterprises"
    r"|Systems|Services|Labs|Laboratories|Hospital|Clinic|University|College|Bank)"
)
_CAP_TOKEN = r"[A-Z][A-Za-z&.'\u2019-]{0,24}"
_NAME_CORE = r"(?:[A-Z]\.\s*){0,3}[A-Z][a-z]+(?:\s+(?:[A-Z]\.\s*)?[A-Z][a-z]+){0,3}"

# org: a run of capitalised words ending in a company suffix.
_ORG = re.compile(rf"\b((?:{_CAP_TOKEN}\s+){{1,4}}{_ORG_SUFFIX})(?![A-Za-z])")
# org: a capitalised phrase labelled with its contract role - "Kredo Automation (Provider)".
_ORG_ROLE_PAREN = re.compile(
    rf"\b((?:{_CAP_TOKEN}\s+){{0,4}}{_CAP_TOKEN})\s*\((?:the\s+)?"
    r"(?:Client|Provider|Vendor|Supplier|Customer|Buyer|Seller|Contractor|Company|Purchaser"
    r"|Licensor|Licensee)\)"
)
# org: an explicitly labelled party - "Vendor: Kredo Automation".
_ORG_LABELLED = re.compile(
    r"(?i:\b(?:vendor|supplier|seller|billed\s+by|issued\s+by|sold\s+by|billed\s+to|client|customer"
    r"|bill\s+to|party)\s*[:\-]\s*)"
    rf"((?:{_CAP_TOKEN}\s+){{0,4}}{_CAP_TOKEN})"
)

# person: honorific-led, or right after a role cue.
_PERSON_HONORIFIC = re.compile(rf"\b((?:{'|'.join(HONORIFICS)})\.?\s+{_NAME_CORE})")
_PERSON_CUED = re.compile(
    rf"(?i:\b(?:{'|'.join(re.escape(c) for c in PERSON_CUES)})\s*[:\-]?\s+)"
    rf"((?:(?:{'|'.join(HONORIFICS)})\.?\s+)?{_NAME_CORE})"
)

# contract / agreement titles.
_CONTRACT = re.compile(
    rf"\b((?:{_CAP_TOKEN}\s+){{0,5}}(?:Agreements?|Contracts?|MoU|MOU"
    r"|Memorandum\s+of\s+Understanding|Deed|Addendum|Amendment|Work\s+Order))(?![A-Za-z])"
)

# project / physical asset names.
_PROJECT = re.compile(r"\b(Project\s+[A-Z][A-Za-z0-9&'\-]*(?:\s+[A-Z][A-Za-z0-9&'\-]*){0,2})\b")
_ASSET = re.compile(
    r"\b((?:Substation|Feeder|Transformer|Unit|Block|Phase|Site|Plant|Line|Tower|Zone)"
    r"\s+(?:No\.?\s*)?[A-Z0-9][A-Za-z0-9\-]*)\b"
)

# invoice / bill numbers.
_INVOICE = re.compile(
    r"(?i:\b(?:tax\s+)?(?:invoice|bill|credit\s+note|debit\s+note)\s*(?:nos?\.?|number|#)?\s*[:#]?\s*)"
    r"([A-Z]{0,6}[-/ ]?\d{1,10}(?:[-/][A-Z0-9]{1,8})*)\b"
)
# statutory / commercial identifiers.
_GOVT_ID = re.compile(
    r"\b(GSTIN|GST\s*(?:No\.?|Number)?|PAN|TAN|CIN|UDYAM|Aadhaar|IFSC|SWIFT)\s*[:#]?\s*"
    r"([A-Z0-9]{5,25})\b"
)
_ORDER_ID = re.compile(
    r"(?i:\b(?:p\.?\s?o\.?|purchase\s+order|work\s+order)\s*(?:nos?\.?|number|#)?\s*[:#]?\s*)"
    r"([A-Z]{0,4}[-/ ]?\d{2,10}(?:[-/][A-Z0-9]{1,8})*)\b"
)

# place: gazetteer terms, longest first so "New Delhi" wins over "Delhi".
_PLACE = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in sorted(PLACE_GAZETTEER, key=len, reverse=True)) + r")\b"
)

# dates: autotune's pattern handles "Mar 12, 2026" and "12/03/2026" but not the
# "12 March 2026" form common in Indian paperwork, so two extras are added here.
_DATE_LONG = re.compile(
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?,?\s+\d{4}\b"
)
_MONTH_YEAR = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b"
)

_WS = re.compile(r"\s+")
_SENT_BREAK = re.compile(r"(?:[.!?]+[\s\u00a0]+|\n+)")
_CAP_RUN = re.compile(
    rf"\b(?:{_CAP_TOKEN})(?:\s+(?:{_CAP_TOKEN}|of|and|for|&))*"
)
_QUOTED = re.compile(r"[\"'\u201c\u2018]([^\"'\u201d\u2019]{2,60})[\"'\u201d\u2019]")
_QUERY_TAIL = re.compile(
    r"(?i:(?:everything\s+(?:related\s+to|about|on)|related\s+to|relating\s+to|all\s+about|about"
    r"|regarding|on|for|of|to)\s+)(.{2,60})$"
)


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def _norm_text(text: str) -> str:
    """Lowercase, collapse whitespace, trim surrounding punctuation."""
    return _WS.sub(" ", (text or "").strip().lower()).strip(" .,;:()[]{}\"'`\u2019-")


def _norm_org(name: str) -> str:
    """Normalise an organisation name by dropping its legal form.

    "ABC Ltd", "ABC Limited" and "M/s ABC Pvt. Ltd." all become "abc" so they
    land on one node. Descriptive words ("Solutions", "Systems") are kept - they
    distinguish real companies from each other.

    Returns "" when nothing distinguishing is left ("The Company"), which the
    junk filter then drops: a generic node like that would otherwise become a hub
    linking every contract in the corpus to every other one.
    """
    norm = _norm_text(name)
    norm = re.sub(r"^m/?s\.?\s+", "", norm)
    tokens = [t.strip(".,&") for t in norm.split()]
    tokens = [t for t in tokens if t]
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    while tokens and tokens[0] in DETERMINERS:
        tokens.pop(0)
    return " ".join(tokens)


def _norm_person(name: str) -> str:
    """Drop the honorific and the initial dots: "Dr. S. Menon" -> "s menon"."""
    norm = _norm_text(name).replace(".", " ")
    tokens = [t for t in _WS.sub(" ", norm).split() if t]
    honorifics = {h.lower() for h in HONORIFICS}
    while tokens and tokens[0] in honorifics:
        tokens.pop(0)
    return " ".join(tokens)


def _norm_for_kind(kind: str, name: str) -> str:
    if kind == "org":
        return _norm_org(name)
    if kind == "person":
        return _norm_person(name)
    if kind in ("code", "part_number", "invoice", "id"):
        return normalise_code(name)
    return _norm_text(name)


def _strip_determiners(raw: str) -> tuple[str, int]:
    """Return (name without leading determiners, characters removed from the front)."""
    offset = 0
    text = raw
    while True:
        m = re.match(r"\s*([A-Za-z']+)\s+", text)
        if not m or m.group(1).lower() not in DETERMINERS:
            return text, offset
        offset += m.end()
        text = text[m.end():]


def entity_key(entity: dict[str, Any]) -> str:
    """The stable identity of an entity inside one call: ``"kind:norm"``."""
    return f"{entity['kind']}:{entity['norm']}"


# --------------------------------------------------------------------------- #
# Extraction rules
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Rule:
    """One extraction rule.

    ``group`` selects the capture group used for both the name and the span.
    ``namer`` may rewrite the (name, norm) pair, or reject the match by
    returning ``None``.
    """

    kind: str
    pattern: re.Pattern[str]
    group: int = 0
    namer: Callable[[re.Match[str]], tuple[str, str | None] | None] | None = None


def _name_govt_id(m: re.Match[str]) -> tuple[str, str | None] | None:
    label = _WS.sub(" ", m.group(1).strip()).upper().rstrip(".")
    label = "GSTIN" if label.startswith("GST") else label
    value = m.group(2).strip()
    if not any(ch.isdigit() for ch in value) and len(value) < 8:
        return None
    return f"{label} {value}", normalise_code(value)


def _name_order_id(m: re.Match[str]) -> tuple[str, str | None] | None:
    value = _WS.sub(" ", m.group(1).strip())
    return f"PO {value}", normalise_code(f"PO{value}")


def _name_person(m: re.Match[str]) -> tuple[str, str | None] | None:
    """Reject "people" whose last word is really a company or a place."""
    name = _WS.sub(" ", m.group(m.lastindex or 1).strip())
    last = name.split()[-1].strip(".,").lower()
    if last in LEGAL_SUFFIXES or last in DESCRIPTIVE_SUFFIXES:
        return None
    if last in {p.lower() for p in PLACE_GAZETTEER}:
        return None
    return name, None


#: Rules in priority order. An earlier rule wins any span overlap, which is how
#: "Invoice No. INV-1001" stays one invoice instead of also becoming a bare code,
#: and how "Master Services Agreement" stays a contract instead of an org.
RULES: tuple[_Rule, ...] = (
    _Rule("contract", _CONTRACT, 1),
    _Rule("invoice", _INVOICE, 1),
    _Rule("id", _GOVT_ID, 0, _name_govt_id),
    _Rule("id", _ORDER_ID, 0, _name_order_id),
    _Rule("org", _ORG, 1),
    _Rule("org", _ORG_ROLE_PAREN, 1),
    _Rule("org", _ORG_LABELLED, 1),
    _Rule("person", _PERSON_HONORIFIC, 1, _name_person),
    _Rule("person", _PERSON_CUED, 1, _name_person),
    _Rule("project", _PROJECT, 1),
    _Rule("project", _ASSET, 1),
    _Rule("place", _PLACE, 1),
    _Rule("amount", ENTITY_PATTERNS["money"], 0),
    _Rule("date", _DATE_LONG, 0),
    _Rule("date", ENTITY_PATTERNS["date"], 0),
    _Rule("date", _MONTH_YEAR, 0),
    _Rule("duration", ENTITY_PATTERNS["duration"], 0),
    _Rule("measurement", ENTITY_PATTERNS["measurement"], 0),
    _Rule("clause_ref", ENTITY_PATTERNS["clause_ref"], 0),
    _Rule("part_number", ENTITY_PATTERNS["part_number"], 1),
    _Rule("code", ENTITY_PATTERNS["code"], 0),
)

_STRIP_DETERMINER_KINDS = frozenset({"org", "contract", "person", "project"})


_GENERIC_ORG_TOKENS: frozenset[str] = frozenset(LEGAL_SUFFIXES) | frozenset(DESCRIPTIVE_SUFFIXES) | DETERMINERS


def _is_junk(kind: str, name: str, norm: str) -> bool:
    """Drop 1-character names, stop-word-only names, empty norms and generic hubs."""
    if len(norm) < 2 or len(name.strip()) < 2:
        return True
    tokens = [t for t in re.split(r"[^a-z0-9]+", norm) if t]
    if not tokens:
        return True
    if kind in ("org", "person", "place", "project", "contract") and all(t in STOP for t in tokens):
        return True
    # "The Company", "the Bank": an org with no name of its own is not an entity,
    # it is a hub that would wire every document to every other one.
    if kind == "org" and all(t in _GENERIC_ORG_TOKENS for t in tokens):
        return True
    return False


def extract_entities(text: str) -> list[dict[str, Any]]:
    """Find the entities in one piece of text with regular expressions only.

    Returns ``[{"kind", "name", "norm", "span": (start, end)}, ...]`` in
    document order (earliest span first).

    The heuristics, in plain language:

    * **org** - a run of capitalised words ending in a company suffix
      ("... Ltd", "Private Limited", "... Technologies", "... Hospital"), a
      capitalised phrase tagged with its contract role ("Kredo Automation
      (Provider)"), or one after an explicit label ("Vendor: ...").
    * **person** - a name led by an honorific ("Dr. S. Menon") or one sitting
      right after a role cue ("Signed by Rajesh Kumar", "Inspector: A. Rao").
      A "name" whose last word is a company suffix or a gazetteer place is
      rejected.
    * **place** - a term from the short built-in gazetteer above. A preceding
      "in"/"at" is not required to *detect* it, but it is the cue used later for
      the ``located_in`` relation.
    * **contract** - a capitalised title ending in Agreement / Contract / MoU /
      Deed / Addendum, with leading determiners dropped ("This Agreement" ->
      "Agreement").
    * **invoice**, **id** - the identifier after a label: "Invoice No. INV-1001",
      "Bill No BL/2026/7", "GSTIN ...", "PAN ...", "PO 4501".
    * **project** - "Project <Name>" and plant-style assets ("Substation 4").
    * **amount, date, duration, measurement, clause_ref, code, part_number** -
      reuse :data:`ragly_backend.autotune.ENTITY_PATTERNS` unchanged; two extra
      date patterns are added here for the "12 March 2026" and "March 2026"
      forms that pattern does not cover.

    Normalisation: ``norm`` is lowercased and whitespace-collapsed; company
    legal forms are stripped for orgs (so "ABC Ltd" and "ABC Limited" share one
    node); honorifics are stripped for people; codes and identifiers go through
    :func:`ragly_backend.autotune.normalise_code`.

    Results are de-duplicated per call on ``(kind, norm)`` (first occurrence
    wins), 1-character and stop-word-only names are dropped, and the list is
    capped at :data:`MAX_ENTITIES_PER_CALL`.

    These are rules, **not** a trained NER model: they will miss entities they
    have no pattern for and will sometimes accept a capitalised phrase that is
    not really an entity.
    """
    if not text:
        return []

    accepted: list[dict[str, Any]] = []
    taken: list[tuple[int, int]] = []
    seen: set[tuple[str, str]] = set()

    for rule in RULES:
        for m in rule.pattern.finditer(text):
            try:
                raw = m.group(rule.group)
            except IndexError:  # pragma: no cover - defensive
                continue
            if not raw:
                continue
            start, end = m.span(rule.group)
            name = raw
            norm: str | None = None
            if rule.namer is not None:
                named = rule.namer(m)
                if named is None:
                    continue
                name, norm = named
                start, end = m.span(0) if rule.group == 0 else m.span(rule.group)
            if rule.kind in _STRIP_DETERMINER_KINDS:
                name, offset = _strip_determiners(name)
                start += offset
            name = _WS.sub(" ", name.strip()).strip(",;:")
            end = min(end, start + len(name)) if rule.namer is None else end
            if not name:
                continue
            if norm is None:
                norm = _norm_for_kind(rule.kind, name)
            if _is_junk(rule.kind, name, norm):
                continue
            if any(start < t_end and t_start < end for t_start, t_end in taken):
                continue
            key = (rule.kind, norm)
            if key in seen:
                continue
            seen.add(key)
            taken.append((start, end))
            accepted.append({"kind": rule.kind, "name": name, "norm": norm, "span": (start, end)})

    accepted.sort(key=lambda e: (e["span"][0], e["span"][1], e["kind"], e["norm"]))
    return accepted[:MAX_ENTITIES_PER_CALL]


# --------------------------------------------------------------------------- #
# Relation inference
# --------------------------------------------------------------------------- #
def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split text into sentence spans, keeping abbreviations intact.

    A full stop is only treated as a sentence end when the word before it is not
    a known abbreviation ("No.", "Ltd.", "Mr.") and is not a single initial.
    That keeps "Invoice No. INV-1001 dated 12 March 2026" in one sentence, which
    is where the proximity rules below do their work. The cost is occasional
    under-splitting, which only adds weak co-occurrence edges.
    """
    if not text:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _SENT_BREAK.finditer(text):
        if m.group(0).strip():  # a "." / "!" / "?" break - check for abbreviations
            prev = text[start:m.start()]
            tail = re.search(r"([A-Za-z][A-Za-z.]*)$", prev)
            token = tail.group(1).lower().rstrip(".") if tail else ""
            if token in _ABBREVIATIONS or len(token) == 1:
                continue
        if m.start() > start:
            spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _cue_positions(sentence_low: str, cues: Iterable[str]) -> list[int]:
    """End offsets of every cue phrase occurrence, in order."""
    out: list[int] = []
    for cue in cues:
        for m in re.finditer(re.escape(cue), sentence_low):
            out.append(m.end())
    return sorted(out)


_ISSUED_CUES = ("raised by", "issued by", "billed by", "supplied by", "sold by", "vendor", "supplier", "from")
_SIGNED_CUES = ("made on", "dated", "signed on", "executed on", "entered into on", "effective")
_TERM_WORDS = ("term", "period", "duration", "validity", "renewable")
_WORKS_FOR_GAP = re.compile(
    r"(?:\bat|\bof|\bfor|\bwith|employed\s+by|works\s+(?:for|at)|representing)\s*[,:\-]?\s*$"
)
_LOCATED_GAP = re.compile(r"(?:,\s*|\bin\s+|\bat\s+|\bnear\s+|based\s+(?:in|at)\s+)$")


def infer_relations(entities: list[dict[str, Any]], text: str) -> list[tuple[str, str, str]]:
    """Guess relations between entities from same-sentence proximity and cue words.

    Returns ``(src_key, rel, dst_key)`` triples where each key is
    ``"<kind>:<norm>"`` (see :func:`entity_key`) so the caller can resolve them
    to node ids. ``entities`` must be the output of :func:`extract_entities` for
    the same ``text``, because the spans are used to place each entity in a
    sentence.

    Implemented relations and the cue that triggers them:

    * ``person works_for org`` - the text between the person and the company
      ends in "at" / "of" / "for" / "with" / "employed by" / "works for"
      ("Rajesh Kumar, Finance Head at ABC Ltd").
    * ``invoice issued_by org`` - the sentence has an issuing cue ("raised by",
      "issued by", "vendor", "from") and the first organisation after that cue
      is taken as the issuer.
    * ``invoice has_amount amount``, ``invoice dated date``,
      ``invoice belongs_to project`` - same sentence as the invoice number.
    * ``contract between org`` - both parties after the word "between"
      (two edges, one per party).
    * ``contract signed_on date`` - "made on" / "dated" / "signed on" /
      "executed on" / "effective".
    * ``contract has_term duration`` - a duration in a sentence that also talks
      about a term, period, duration or validity.
    * ``org located_in place`` - a gazetteer place right after the company name,
      separated only by a comma or by "in" / "at" / "near" / "based in".
    * ``co_occurs_with`` - the generic fallback for two entities in the same
      sentence, stored with weight 0.3. It is only emitted when no specific
      relation was found for that pair, and never between two amounts/dates.

    This is proximity matching, **not** a trained relation extractor: a sentence
    that mentions two companies and one invoice can attribute the invoice to the
    wrong company. Weights and the ``why`` strings on the retrieval side exist
    so the UI can show the guess rather than assert it.
    """
    if not entities or not text:
        return []

    triples: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    def add(src: dict[str, Any], rel: str, dst: dict[str, Any]) -> bool:
        if src is dst:
            return False
        key = (entity_key(src), rel, entity_key(dst))
        if key[0] == key[2] or key in seen:
            return False
        seen.add(key)
        triples.append(key)
        return True

    for s_start, s_end in _sentence_spans(text):
        sentence = text[s_start:s_end]
        low = sentence.lower()
        # Entities are de-duplicated per call and only carry their first span, so
        # a name repeated in a later sentence is also looked up literally here -
        # otherwise "... at ABC Ltd." in the last sentence would see no company.
        here: list[dict[str, Any]] = []
        rel_start: dict[int, int] = {}
        rel_end: dict[int, int] = {}
        for entity in entities:
            e_start, e_end = entity["span"]
            if s_start <= e_start and e_end <= s_end:
                offset = e_start - s_start
                length = e_end - e_start
            else:
                offset = low.find(entity["name"].lower())
                if offset < 0:
                    continue
                length = len(entity["name"])
            here.append(entity)
            rel_start[id(entity)] = offset
            rel_end[id(entity)] = offset + length
        if len(here) < 2:
            continue
        here.sort(key=lambda e: (rel_start[id(e)], e["kind"], e["norm"]))
        by_kind: dict[str, list[dict[str, Any]]] = {}
        for e in here:
            by_kind.setdefault(e["kind"], []).append(e)

        paired: set[frozenset[str]] = set()

        def pair(a: dict[str, Any], b: dict[str, Any]) -> None:
            paired.add(frozenset({entity_key(a), entity_key(b)}))

        orgs = by_kind.get("org", [])
        people = by_kind.get("person", [])
        invoices = by_kind.get("invoice", [])
        contracts = by_kind.get("contract", [])
        amounts = by_kind.get("amount", [])
        dates = by_kind.get("date", [])
        durations = by_kind.get("duration", [])
        projects = by_kind.get("project", [])
        places = by_kind.get("place", [])

        # person works_for org
        for person in people:
            for org in orgs:
                gap_start, gap_end = rel_end[id(person)], rel_start[id(org)]
                if not 0 <= gap_end - gap_start <= 60:
                    continue
                if _WORKS_FOR_GAP.search(sentence[gap_start:gap_end].lower()):
                    if add(person, "works_for", org):
                        pair(person, org)

        # invoice issued_by org  (first org after an issuing cue)
        if invoices and orgs:
            cues = _cue_positions(low, _ISSUED_CUES)
            for invoice in invoices:
                issuer = None
                for cue_end in cues:
                    for org in orgs:
                        offset = rel_start[id(org)]
                        if cue_end <= offset <= cue_end + 40:
                            issuer = org
                            break
                    if issuer is not None:
                        break
                if issuer is not None and add(invoice, "issued_by", issuer):
                    pair(invoice, issuer)

        # invoice -> amount / date / project
        for invoice in invoices:
            for amount in amounts:
                if add(invoice, "has_amount", amount):
                    pair(invoice, amount)
            for date in dates:
                if add(invoice, "dated", date):
                    pair(invoice, date)
            for project in projects:
                if add(invoice, "belongs_to", project):
                    pair(invoice, project)

        # contract between org and org / signed_on date / has_term duration
        for contract in contracts:
            between = low.find("between")
            if between >= 0:
                parties = [o for o in orgs if rel_start[id(o)] > between][:4]
                for org in parties:
                    if add(contract, "between", org):
                        pair(contract, org)
            if dates:
                cues = _cue_positions(low, _SIGNED_CUES)
                target = None
                for cue_end in cues:
                    for date in dates:
                        if cue_end <= rel_start[id(date)] <= cue_end + 30:
                            target = date
                            break
                    if target is not None:
                        break
                target = target or dates[0]
                if add(contract, "signed_on", target):
                    pair(contract, target)
            if durations and any(w in low for w in _TERM_WORDS):
                for duration in durations:
                    if add(contract, "has_term", duration):
                        pair(contract, duration)

        # org located_in place
        for org in orgs:
            for place in places:
                gap_start, gap_end = rel_end[id(org)], rel_start[id(place)]
                if not 0 <= gap_end - gap_start <= 40:
                    continue
                if _LOCATED_GAP.search(sentence[gap_start:gap_end].lower()):
                    if add(org, "located_in", place):
                        pair(org, place)

        # generic fallback: co-occurrence in the same sentence
        pool = here[:MAX_FALLBACK_ENTITIES]
        emitted = 0
        for i, left in enumerate(pool):
            for right in pool[i + 1:]:
                if emitted >= MAX_FALLBACK_PAIRS:
                    break
                if left["kind"] in ("amount", "date") and right["kind"] in ("amount", "date"):
                    continue
                if frozenset({entity_key(left), entity_key(right)}) in paired:
                    continue
                first, second = sorted([left, right], key=entity_key)
                if add(first, "co_occurs_with", second):
                    emitted += 1
            if emitted >= MAX_FALLBACK_PAIRS:
                break
        if len(triples) >= MAX_RELATIONS_PER_CALL:
            break

    return triples[:MAX_RELATIONS_PER_CALL]


# --------------------------------------------------------------------------- #
# Router helper
# --------------------------------------------------------------------------- #
def is_entity_question(question: str) -> bool:
    """Cheap test for "is this a question about a *thing* rather than a passage".

    True for "show me everything related to ABC Ltd", "documents about Project
    Falcon", "who is Dr. Menon", "which vendor raised this bill", and for a
    short query that is just an entity name ("ABC Ltd"). False for ordinary
    passage questions ("what is the termination notice period?").

    Pure string matching on the cue list in :data:`ENTITY_QUESTION_CUES` plus one
    fallback that looks for an org/person/project/contract in a very short query.
    It is a router hint, not a classifier.
    """
    low = _WS.sub(" ", (question or "").strip().lower())
    if not low:
        return False
    if any(cue in low for cue in ENTITY_QUESTION_CUES):
        return True
    words = low.split()
    if len(words) <= 5:
        kinds = {e["kind"] for e in extract_entities(question)}
        if kinds & {"org", "person", "project", "contract", "invoice"}:
            return True
    return False


# --------------------------------------------------------------------------- #
# Write side
# --------------------------------------------------------------------------- #
class GraphBuilder:
    """Turns document text into graph nodes, edges and mentions in the store.

    Indexing is idempotent per document: every entry point clears the
    document's edges and mentions first (``store.clear_graph_for_doc``), so
    re-indexing the same document cannot double its edges. Nodes are shared
    across documents by ``(kind, norm)``, which is the whole point - that is how
    two unrelated files end up connected through "ABC Ltd".

    Note: ``graph_nodes.mentions`` is a counter maintained by the store's upsert
    and is not reset by a re-index, so it should be read as "how often this node
    has been seen", not as an exact count of live mentions.
    """

    def __init__(self, store: Store) -> None:
        self.store = store

    # -- public -------------------------------------------------------------
    def index_chunks(
        self,
        doc_id: int,
        chunks: Sequence[Any],
        chunk_ids: Sequence[int] | None = None,
        clear: bool = True,
    ) -> dict[str, int]:
        """Index chunk objects (anything with ``.page`` and ``.text``, e.g.
        :class:`ragly_backend.chunker.Chunk`).

        ``chunk_ids`` is an optional parallel list of the stored chunk row ids;
        when given, every mention records the chunk it came from, which is what
        makes :meth:`GraphRetriever.chunk_ids_for` able to feed the hybrid
        retriever. Returns counts for the caller to log.
        """
        if clear:
            self.store.clear_graph_for_doc(doc_id)
        before = self.store.graph_counts()
        entities = 0
        nodes: set[int] = set()
        for i, chunk in enumerate(chunks):
            page = int(getattr(chunk, "page", 1) or 1)
            text = getattr(chunk, "text", "") or ""
            chunk_id = None
            if chunk_ids is not None and i < len(chunk_ids):
                chunk_id = int(chunk_ids[i])
            key_to_id, found = self._index_text(doc_id, page, text, chunk_id)
            entities += found
            nodes.update(key_to_id.values())
        after = self.store.graph_counts()
        return {
            "chunks": len(chunks),
            "entities": entities,
            "nodes": len(nodes),
            "edges": max(0, after["edges"] - before["edges"]),
            "mentions": max(0, after["mentions"] - before["mentions"]),
        }

    def index_document(self, doc_id: int, pages: list[tuple[int, str]], clear: bool = True) -> dict[str, int]:
        """Index page-level text when the chunk row ids are not known yet.

        Mentions are recorded with a page but no chunk id, so the document and
        page still show up in :meth:`GraphRetriever.expand`; they just cannot
        contribute chunk ids to the hybrid retriever.
        """
        if clear:
            self.store.clear_graph_for_doc(doc_id)
        before = self.store.graph_counts()
        entities = 0
        nodes: set[int] = set()
        for page, text in pages:
            key_to_id, found = self._index_text(doc_id, int(page), text or "", None)
            entities += found
            nodes.update(key_to_id.values())
        after = self.store.graph_counts()
        return {
            "pages": len(pages),
            "entities": entities,
            "nodes": len(nodes),
            "edges": max(0, after["edges"] - before["edges"]),
            "mentions": max(0, after["mentions"] - before["mentions"]),
        }

    def link_image(self, image_id: int, doc_id: int, page: int) -> int:
        """Attach an image to everything the graph knows about that page.

        Every node already mentioned on ``(doc_id, page)`` gets a second mention
        row carrying ``image_id`` - the "this image belongs to this document and
        page" link - so asking about an entity can return the figures on the
        pages that mention it. Returns the number of links written. Must be
        called after the page's text has been indexed, because a re-index clears
        the document's mentions.
        """
        rows = self.store.conn.execute(
            "SELECT DISTINCT node_id FROM node_mentions WHERE doc_id=? AND page=? ORDER BY node_id",
            (doc_id, page),
        ).fetchall()
        for row in rows:
            self.store.add_mention(int(row[0]), doc_id, page, None, int(image_id))
        return len(rows)

    # -- internal -----------------------------------------------------------
    def _index_text(
        self, doc_id: int, page: int, text: str, chunk_id: int | None
    ) -> tuple[dict[str, int], int]:
        """Extract, upsert nodes/mentions, infer relations, add edges."""
        entities = extract_entities(text)
        if not entities:
            return {}, 0
        key_to_id: dict[str, int] = {}
        for entity in entities:
            node_id = self.store.upsert_node(entity["kind"], entity["name"], entity["norm"])
            key_to_id[entity_key(entity)] = node_id
            self.store.add_mention(node_id, doc_id, page, chunk_id, None)
        for src_key, rel, dst_key in infer_relations(entities, text):
            src, dst = key_to_id.get(src_key), key_to_id.get(dst_key)
            if src is None or dst is None or src == dst:
                continue
            self.store.add_edge(src, rel, dst, doc_id, page, chunk_id, REL_WEIGHTS.get(rel, 1.0))
        return key_to_id, len(entities)


# --------------------------------------------------------------------------- #
# Read side
# --------------------------------------------------------------------------- #
class GraphRetriever:
    """Answers entity-shaped questions off the graph.

    This is the third retrieval path next to BM25 and vector similarity: it does
    not score text at all, it walks entity links and hands back the chunk ids
    that mention the entities a question is about, plus a plain-language reason
    for each one.
    """

    def __init__(self, store: Store) -> None:
        self.store = store

    # -- resolve ------------------------------------------------------------
    def resolve(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Find the graph nodes a query is about.

        Candidate names come from running :func:`extract_entities` on the query
        itself, from quoted phrases, from runs of capitalised words, and from the
        tail after a cue ("everything related to <tail>"). Each candidate is
        looked up three ways, best first: an exact ``norm`` match, a
        punctuation-insensitive match (so "INV 1001" finds "INV-1001"), then a
        substring match. Returns the best few nodes, each with its kind, name,
        mention counter and which of the three matched.
        """
        out: list[dict[str, Any]] = []
        seen: set[int] = set()
        for index, (norm, kind, phrase) in enumerate(self._candidates(query)):
            for row in self.store.find_nodes(norm, limit=40):
                node_id = int(row["id"])
                if node_id in seen:
                    continue
                if row["norm"] == norm:
                    match, rank = "exact", 0
                elif normalise_code(row["norm"]) == normalise_code(norm):
                    match, rank = "normalised", 1
                elif len(norm) >= 3:
                    match, rank = "partial", 2
                else:
                    continue
                if kind is not None and row["kind"] != kind:
                    rank += 1        # a kind mismatch is a weaker match
                seen.add(node_id)
                out.append({
                    "id": node_id,
                    "kind": row["kind"],
                    "name": row["name"],
                    "norm": row["norm"],
                    "mentions": int(row["mentions"]),
                    "match": match,
                    "matched_text": phrase,
                    "_rank": (rank, index, -int(row["mentions"]), row["name"]),
                })
        out.sort(key=lambda r: r["_rank"])
        for row in out:
            row.pop("_rank", None)
        return out[:max(1, limit)]

    def _candidates(self, query: str) -> list[tuple[str, str | None, str]]:
        """(norm, kind or None, original phrase) candidates, best first."""
        cands: list[tuple[str, str | None, str]] = []
        for entity in extract_entities(query or ""):
            cands.append((entity["norm"], entity["kind"], entity["name"]))
        phrases: list[str] = []
        for m in _QUOTED.finditer(query or ""):
            phrases.append(m.group(1))
        for m in _CAP_RUN.finditer(query or ""):
            phrases.append(m.group(0))
        tail = _QUERY_TAIL.search(_WS.sub(" ", (query or "").strip()))
        if tail:
            phrases.append(tail.group(1))
        phrases.sort(key=lambda p: (-len(p.split()), -len(p)))
        for phrase in phrases:
            norm = _norm_text(phrase.strip(" ?!.,"))
            tokens = norm.split()
            if not tokens or all(t in STOP for t in tokens):
                continue
            if len(norm) < 2:
                continue
            cands.append((norm, None, phrase.strip()))
            org_norm = _norm_org(phrase)
            if org_norm and org_norm != norm:
                cands.append((org_norm, None, phrase.strip()))
        unique: list[tuple[str, str | None, str]] = []
        seen: set[tuple[str, str | None]] = set()
        for norm, kind, phrase in cands:
            if not norm or (norm, kind) in seen:
                continue
            seen.add((norm, kind))
            unique.append((norm, kind, phrase))
        return unique

    # -- expand -------------------------------------------------------------
    def expand(self, node_id: int, hops: int = 1) -> dict[str, Any]:
        """The neighbourhood of one node, as nodes / edges / documents / images.

        At most :data:`MAX_HOPS` hops and :data:`MAX_GRAPH_NODES` nodes, so a
        very common node ("India") cannot drag the whole graph into one answer.
        Documents are collapsed to one row per document with the pages that
        mention the walked nodes.
        """
        hops = max(1, min(MAX_HOPS, int(hops)))
        node_rows: dict[int, dict[str, Any]] = {}
        edge_rows: dict[tuple[Any, ...], dict[str, Any]] = {}
        mentions: list[dict[str, Any]] = []
        visited: set[int] = set()
        frontier = [int(node_id)]

        for _ in range(hops):
            nxt: set[int] = set()
            for nid in frontier:
                if nid in visited or len(visited) >= MAX_GRAPH_NODES:
                    continue
                visited.add(nid)
                hood = self.store.node_neighbourhood(nid, limit=MAX_NEIGHBOURS)
                if hood["node"]:
                    node_rows[nid] = dict(hood["node"])
                for edge in hood["edges"]:
                    src, dst = edge["src"], edge["dst"]
                    if src is None or dst is None:      # half-written edge: skip, never crash
                        continue
                    key = (src, edge["rel"], dst, edge["doc_id"], edge["page"])
                    edge_rows.setdefault(key, dict(edge))
                    nxt.add(int(src))
                    nxt.add(int(dst))
                mentions.extend(dict(m) for m in hood["mentions"])
            frontier = sorted(nxt - visited)

        # neighbours we never walked still deserve a name and a kind
        for edge in edge_rows.values():
            for side in ("src", "dst"):
                nid = int(edge[side])
                if nid not in node_rows:
                    node_rows[nid] = {
                        "id": nid,
                        "kind": edge[f"{side}_kind"],
                        "name": edge[f"{side}_name"],
                        "norm": "",
                        "mentions": 0,
                    }

        nodes = sorted(node_rows.values(), key=lambda n: (n["kind"], str(n["name"]), int(n["id"])))
        nodes = nodes[:MAX_GRAPH_NODES]
        keep = {int(n["id"]) for n in nodes}
        edges = [
            {
                "src": int(e["src"]), "rel": e["rel"], "dst": int(e["dst"]),
                "src_name": e["src_name"], "src_kind": e["src_kind"],
                "dst_name": e["dst_name"], "dst_kind": e["dst_kind"],
                "doc_id": e["doc_id"], "page": e["page"], "chunk_id": e["chunk_id"],
                "weight": float(e["weight"]),
            }
            for e in edge_rows.values()
            if int(e["src"]) in keep and int(e["dst"]) in keep
        ]
        edges.sort(key=lambda e: (str(e["src_name"]), e["rel"], str(e["dst_name"]),
                                  e["doc_id"] or 0, e["page"] or 0))

        documents = self._documents(mentions)
        images = sorted({int(m["image_id"]) for m in mentions if m.get("image_id") is not None})
        return {
            "seed": node_rows.get(int(node_id)),
            "hops": hops,
            "nodes": nodes,
            "edges": edges,
            "documents": documents,
            "images": images,
        }

    @staticmethod
    def _documents(mentions: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        docs: dict[int, dict[str, Any]] = {}
        for m in mentions:
            doc_id = int(m["doc_id"])
            entry = docs.setdefault(doc_id, {"doc_id": doc_id, "doc_name": m.get("doc_name", ""), "pages": set()})
            entry["pages"].add(int(m["page"]))
        out = [{"doc_id": d["doc_id"], "doc_name": d["doc_name"], "pages": sorted(d["pages"])}
               for d in docs.values()]
        out.sort(key=lambda d: (str(d["doc_name"]), d["doc_id"]))
        return out

    # -- chunk ids ----------------------------------------------------------
    def chunk_ids_for(self, query: str, limit: int = 20) -> tuple[list[int], list[dict[str, Any]]]:
        """Chunk ids reachable from the entities a query is about.

        Walks the resolved entities' own mentions first, then the mentions of
        their one-hop neighbours (strongest edge first), and returns
        ``(chunk_ids, explanations)``. ``explanations`` has one row per
        contributing node - ``{"node", "rel", "why", "chunk_ids"}`` - so the UI
        can say *why* a chunk is in the list ("ABC Ltd issued_by Invoice
        INV-1001"). This list is meant to be fused into the hybrid retriever as
        a third ranked list, not used on its own.
        """
        chunk_ids: list[int] = []
        explanations: list[dict[str, Any]] = []
        limit = max(1, int(limit))

        def take(node_name: str, rel: str, why: str, hood: dict[str, Any]) -> None:
            found = sorted({int(m["chunk_id"]) for m in hood["mentions"] if m.get("chunk_id") is not None})
            fresh = [c for c in found if c not in chunk_ids]
            if not fresh:
                return
            room = limit - len(chunk_ids)
            fresh = fresh[:room]
            chunk_ids.extend(fresh)
            explanations.append({"node": node_name, "rel": rel, "why": why, "chunk_ids": fresh})

        for seed in self.resolve(query):
            if len(chunk_ids) >= limit:
                break
            hood = self.store.node_neighbourhood(int(seed["id"]), limit=MAX_NEIGHBOURS)
            take(seed["name"], "mentions", f"mentions {seed['name']}", hood)

            neighbours: list[tuple[float, str, int, str]] = []
            for edge in hood["edges"]:
                if int(edge["src"]) == int(seed["id"]):
                    neighbours.append((-float(edge["weight"]), str(edge["dst_name"]),
                                       int(edge["dst"]), edge["rel"]))
                else:
                    neighbours.append((-float(edge["weight"]), str(edge["src_name"]),
                                       int(edge["src"]), edge["rel"]))
            neighbours.sort()
            done: set[int] = {int(seed["id"])}
            for _weight, name, nid, rel in neighbours:
                if len(chunk_ids) >= limit:
                    break
                if nid in done:
                    continue
                done.add(nid)
                nb = self.store.node_neighbourhood(nid, limit=MAX_NEIGHBOURS)
                take(name, rel, f"{seed['name']} {rel} {name}", nb)

        return chunk_ids, explanations

    # -- summary ------------------------------------------------------------
    def summary(self, query: str) -> dict[str, Any]:
        """"Everything related to X" in one structure.

        Resolves the query to an entity, walks one hop, and groups what it finds:
        related organisations / people / projects / places / contracts, the
        invoices with their amounts, dates and issuer, the documents and pages
        that mention any of it, and the ids of images on those pages. Empty
        ``entity`` means the query did not resolve to anything in the graph.
        """
        resolved = self.resolve(query)
        if not resolved:
            return {
                "query": query, "entity": None, "also_matched": [], "related": {},
                "invoices": [], "documents": [], "images": [], "edges": [],
            }
        seed = resolved[0]
        hood = self.expand(int(seed["id"]), hops=1)

        related: dict[str, list[str]] = {}
        group = {"org": "orgs", "person": "people", "project": "projects", "place": "places",
                 "contract": "contracts", "amount": "amounts", "date": "dates", "id": "ids",
                 "duration": "durations"}
        for node in hood["nodes"]:
            if int(node["id"]) == int(seed["id"]):
                continue
            bucket = group.get(node["kind"])
            if bucket is None:
                continue
            related.setdefault(bucket, [])
            if node["name"] not in related[bucket]:
                related[bucket].append(str(node["name"]))
        for names in related.values():
            names.sort()

        invoices: list[dict[str, Any]] = []
        invoice_nodes = [n for n in hood["nodes"] if n["kind"] == "invoice"][:10]
        for node in invoice_nodes:
            nb = self.store.node_neighbourhood(int(node["id"]), limit=MAX_NEIGHBOURS)
            amounts: list[str] = []
            dates: list[str] = []
            issuer: str | None = None
            for edge in nb["edges"]:
                other_kind = edge["dst_kind"] if int(edge["src"]) == int(node["id"]) else edge["src_kind"]
                other_name = str(edge["dst_name"] if int(edge["src"]) == int(node["id"]) else edge["src_name"])
                if edge["rel"] == "has_amount" or other_kind == "amount":
                    if other_name not in amounts:
                        amounts.append(other_name)
                elif edge["rel"] == "dated" or other_kind == "date":
                    if other_name not in dates:
                        dates.append(other_name)
                elif edge["rel"] == "issued_by" and issuer is None:
                    issuer = other_name
            invoices.append({
                "invoice": str(node["name"]),
                "amounts": sorted(amounts),
                "dates": sorted(dates),
                "issued_by": issuer,
            })
        invoices.sort(key=lambda i: i["invoice"])

        return {
            "query": query,
            "entity": {k: seed[k] for k in ("id", "kind", "name", "norm", "mentions", "match")},
            "also_matched": [{k: r[k] for k in ("id", "kind", "name")} for r in resolved[1:]],
            "related": related,
            "invoices": invoices,
            "documents": hood["documents"],
            "images": hood["images"],
            "edges": hood["edges"],
        }
