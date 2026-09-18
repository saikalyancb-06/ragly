"""Auto-tuning: read the imported documents and build a 'pack' for them.

No manual configuration. On import Ragly learns, from the text itself:
  * glossary   - abbreviations and their expansions  (RMU = Ring Main Unit)
  * patterns   - codes, part numbers, measured values, clause and section refs
  * structure  - numbered procedures / clauses / lab tables -> answer format
  * doc type   - technical manual, contract, medical report, notes
  * questions  - suggested one-tap questions built from headings and codes
The pack then improves retrieval (query expansion + exact-code boost) and the
prompt (answer format + domain rules). It can be switched off for A/B testing.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field

# ---------------------------------------------------------------- patterns
ENTITY_PATTERNS: dict[str, re.Pattern] = {
    # E-47, ERR 302, P/N 88123-A, MCB-12
    "code": re.compile(r"\b(?:[A-Z]{1,4}[-/ ]?\d{2,6}[A-Z]?)\b"),
    "part_number": re.compile(r"\b(?:P/?N|PART(?:\s*NO\.?)?)\s*[:#]?\s*([A-Z0-9][A-Z0-9-]{3,})", re.I),
    # 40 Nm, 6.9 %, 132 mg/dL, 11.2 g/dL, 230 V, 1.5 bar
    "measurement": re.compile(
        r"\b\d+(?:\.\d+)?\s?(?:Nm|N·m|kg|g|mg/dL|g/dL|mmol/L|bar|psi|kPa|kV|V|mA|A|Hz|rpm|°C|°F|%|mm|cm|m|lakh|crore)\b"
    ),
    # the currency word must stand alone ("engineers, 2" is not "Rs 2") and a digit must follow it
    "money": re.compile(r"(?<![A-Za-z])(?:INR|Rs\.?|₹|USD|\$)\s?\d[\d,]*(?:\.\d+)?"
                        r"(?:\s?(?:lakh|crore|million))?", re.I),
    "clause_ref": re.compile(r"\b(?:Clause|Section|Article|Annexure|Schedule|Step|Para(?:graph)?)\s+\d+(?:\.\d+)*[A-Z]?\b", re.I),
    # "1 March 2026", "March 1, 2026", "Mar 2026", "01/03/2026", "2026-03-01"
    "date": re.compile(
        r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b"
        r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b"
        r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b"
        r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
        r"|\b\d{4}-\d{2}-\d{2}\b"),
    "duration": re.compile(r"\b\d+\s+(?:second|minute|hour|day|week|month|year)s?\b", re.I),
}

# "Ring Main Unit (RMU)"  or  "RMU (Ring Main Unit)"
_ABBR_AFTER = re.compile(r"\b((?:[A-Z][A-Za-z]+[\s-]){1,4}[A-Z][A-Za-z]+)\s*\(([A-Z]{2,6})s?\)")
_ABBR_BEFORE = re.compile(r"\b([A-Z]{2,6})\s*\(([^)]{4,60})\)")
_ACRONYM = re.compile(r"\b[A-Z]{2,6}\b")
_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*\s+)?([A-Z][A-Z0-9 &/,'()-]{4,70})$", re.M)
_STEP_LINE = re.compile(r"^\s*(?:\d{1,2}[.)]|step\s+\d+)\s+\S", re.I | re.M)

DOC_TYPE_HINTS = {
    "technical manual": ["torque", "maintenance", "inspection", "procedure", "fault", "error code", "warning",
                         "voltage", "breaker", "lubricat", "install", "calibration", "spare part", "tool"],
    "contract": ["agreement", "party", "hereby", "termination", "confidential", "indemn", "jurisdiction",
                 "clause", "shall be deemed", "governing law", "invoice", "liability"],
    "medical report": ["patient", "haemoglobin", "hemoglobin", "diagnosis", "mg/dl", "reference range",
                       "prescription", "dosage", "lab", "sample date", "clinical"],
    "policy / SOP": ["policy", "employee", "standard operating", "compliance", "approval", "escalation", "sop"],
}

ANSWER_FORMATS = {
    "steps": "Give procedures as numbered steps, exactly in the order and wording of the source. "
             "Put any SAFETY WARNING or CAUTION first, before the steps.",
    "clauses": "Quote the operative wording of the clause, then explain it in one short sentence. "
               "Always give the clause or section number.",
    "values": "Answer with the exact value, its unit and its date or reference range when the source has one.",
    "prose": "Answer in one short paragraph.",
}

DOMAIN_RULES = {
    "technical manual": [
        "Never estimate a specification. Give the exact number and unit from the manual.",
        "If the source lists a safety warning for the step, state it first.",
    ],
    "contract": [
        "Give the clause or section number with every fact.",
        "Quote dates, amounts and notice periods exactly as written.",
    ],
    "medical report": [
        "You are not a doctor: report what the documents say, never diagnose or advise treatment.",
        "Always give the test date and the reference range next to a value.",
    ],
    "policy / SOP": ["Cite the section number and state who the rule applies to."],
}

STOP = set("""the a an and or of to in for on at by with from as is are was were be been this that these those it its
if then than when which who whom what where how all any each such other into per not no shall will may can must
your you our we they he she his her their there here about above below under over between during before after
page table figure note fig section clause step total value type name date time""".split())


@dataclass
class Pack:
    doc_type: str = "general documents"
    glossary: dict[str, str] = field(default_factory=dict)      # abbreviation -> expansion
    key_terms: list[str] = field(default_factory=list)          # frequent domain words
    entity_counts: dict[str, int] = field(default_factory=dict)
    answer_format: str = "prose"
    rules: list[str] = field(default_factory=list)
    suggested_questions: list[str] = field(default_factory=list)
    keyword_weight: float = 1.0
    vector_weight: float = 1.0
    sample_codes: list[str] = field(default_factory=list)
    docs_analysed: int = 0
    chunks_analysed: int = 0

    def to_dict(self) -> dict:
        return {
            "doc_type": self.doc_type,
            "glossary": self.glossary,
            "key_terms": self.key_terms,
            "entity_counts": self.entity_counts,
            "answer_format": self.answer_format,
            "rules": self.rules,
            "suggested_questions": self.suggested_questions,
            "keyword_weight": self.keyword_weight,
            "vector_weight": self.vector_weight,
            "sample_codes": self.sample_codes,
            "docs_analysed": self.docs_analysed,
            "chunks_analysed": self.chunks_analysed,
        }

    @staticmethod
    def from_dict(d: dict) -> "Pack":
        p = Pack()
        for k, v in (d or {}).items():
            if hasattr(p, k):
                setattr(p, k, v)
        return p

    def summary(self) -> str:
        bits = [f"{self.doc_type}", f"{len(self.glossary)} glossary terms"]
        for name, n in sorted(self.entity_counts.items(), key=lambda kv: -kv[1])[:3]:
            bits.append(f"{n} {name.replace('_', ' ')}s")
        return " · ".join(bits)


def extract_entities(text: str) -> list[tuple[str, str]]:
    """Return [(entity_type, value)] found in a chunk. Used for exact lookups."""
    out: list[tuple[str, str]] = []
    for kind, pattern in ENTITY_PATTERNS.items():
        for m in pattern.finditer(text):
            value = (m.group(1) if m.groups() else m.group(0)).strip()
            if len(value) > 60:
                continue
            out.append((kind, value))
    # de-duplicate, keep order
    seen = set()
    uniq = []
    for kind, value in out:
        key = (kind, value.lower())
        if key not in seen:
            seen.add(key)
            uniq.append((kind, value))
    return uniq


def normalise_code(text: str) -> str:
    """'e 47' / 'e-47' / 'E47' -> 'e47' so spoken or sloppy input still matches."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def build_pack(texts: list[str], docs: int = 0) -> Pack:
    """Analyse the indexed chunks and produce a tuned pack."""
    pack = Pack(docs_analysed=docs, chunks_analysed=len(texts))
    if not texts:
        return pack
    blob = "\n".join(texts)
    low = blob.lower()

    # ---- document type
    scores = {name: sum(low.count(h) for h in hints) for name, hints in DOC_TYPE_HINTS.items()}
    best, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score >= 3:
        pack.doc_type = best
        pack.rules = list(DOMAIN_RULES.get(best, []))

    # ---- glossary from "Long Form (ABC)" and "ABC (long form)"
    glossary: dict[str, str] = {}
    for m in _ABBR_AFTER.finditer(blob):
        expansion, abbr = m.group(1).strip(), m.group(2).strip()
        initials = "".join(w[0] for w in re.findall(r"[A-Za-z]+", expansion))
        if abbr.lower() in (initials.lower(), initials.lower()[: len(abbr)]):
            glossary[abbr] = expansion
    for m in _ABBR_BEFORE.finditer(blob):
        abbr, expansion = m.group(1).strip(), m.group(2).strip()
        if len(expansion.split()) <= 6 and expansion[0].isalpha() and abbr not in glossary:
            initials = "".join(w[0] for w in re.findall(r"[A-Za-z]+", expansion))
            if abbr.lower() == initials.lower():
                glossary[abbr] = expansion
    # frequent acronyms with no expansion still help keyword search
    acro = Counter(a for a in _ACRONYM.findall(blob) if a not in glossary and a.lower() not in STOP)
    for a, n in acro.most_common(15):
        if n >= 3:
            glossary.setdefault(a, "")
    pack.glossary = dict(sorted(glossary.items())[:60])

    # ---- entities
    counts: Counter = Counter()
    codes: Counter = Counter()
    for t in texts:
        for kind, value in extract_entities(t):
            counts[kind] += 1
            if kind in ("code", "part_number"):
                codes[value] += 1
    pack.entity_counts = dict(counts)
    pack.sample_codes = [c for c, _ in codes.most_common(8)]

    # ---- structure -> answer format
    steps = len(_STEP_LINE.findall(blob))
    clauses = counts.get("clause_ref", 0)
    measures = counts.get("measurement", 0)
    if steps >= max(5, len(texts) // 3):
        pack.answer_format = "steps"
    elif clauses >= 5 and pack.doc_type == "contract":
        pack.answer_format = "clauses"
    elif measures >= max(8, len(texts)):
        pack.answer_format = "values"

    # ---- retrieval weights: code-heavy corpora need exact keyword matching
    code_density = (counts.get("code", 0) + counts.get("part_number", 0)) / max(1, len(texts))
    pack.keyword_weight = round(min(2.0, 1.0 + code_density / 2), 2)

    # ---- key terms (frequent, document-specific words)
    words = Counter(w for w in re.findall(r"[a-z][a-z-]{4,}", low) if w not in STOP)
    pack.key_terms = [w for w, n in words.most_common(25) if n >= 3][:15]

    # ---- suggested questions
    headings = [h.strip().title() for h in _HEADING.findall(blob)]
    seen_h: list[str] = []
    for h in headings:
        if 3 < len(h) < 45 and h not in seen_h:
            seen_h.append(h)
    # Candidates are built from what these documents actually contain - their own codes,
    # headings and vocabulary - so a different import proposes different questions. The list is
    # deliberately longer than what the UI shows: every candidate is checked against the index
    # before it is offered, and the ones with no evidence behind them are dropped there.
    qs: list[str] = []
    for code in pack.sample_codes[:3]:
        qs.append(f"What does {code} mean?")
    for h in seen_h[:5]:
        qs.append(f"What does {h} say?")
    # Words that describe the file rather than its subject make hollow questions.
    meta = {"corpus", "document", "documents", "page", "pages", "test", "tests", "query", "queries",
            "expected", "behavior", "behaviour", "answer", "answers", "abstain", "abstention",
            "retrieval", "synthetic", "reference", "ground", "truth", "section", "sections"}
    for term in [t for t in pack.key_terms if t not in meta][:3]:
        qs.append(f"What do the documents say about {term}?")
    if pack.answer_format == "steps" and seen_h:
        qs.insert(1, f"What is the procedure for {seen_h[0]}?")
    for name in list(pack.glossary)[:3]:
        q = f"What is {name}?"
        if q not in qs:
            qs.append(q)
    seen_q: list[str] = []
    for q in qs:
        if q not in seen_q:
            seen_q.append(q)
    pack.suggested_questions = seen_q[:20]
    return pack


def expand_query(question: str, pack: Pack | None) -> tuple[list[str], list[str]]:
    """Return (extra keyword terms, code-ish tokens) for retrieval."""
    if not pack:
        return [], []
    extra: list[str] = []
    low = question.lower()
    for abbr, expansion in pack.glossary.items():
        if re.search(rf"\b{re.escape(abbr.lower())}\b", low):
            extra += re.findall(r"[A-Za-z]{3,}", expansion)
        elif expansion and expansion.lower() in low:
            extra.append(abbr)
    codes = [m.group(0) for m in ENTITY_PATTERNS["code"].finditer(question.upper())]
    # spoken forms: "error e 47" -> E47
    codes += [f"{m.group(1)}{m.group(2)}" for m in re.finditer(r"\b([A-Za-z]{1,4})[-\s]?(\d{2,6})\b", question)]
    uniq_codes = []
    for c in codes:
        if normalise_code(c) and normalise_code(c) not in {normalise_code(x) for x in uniq_codes}:
            uniq_codes.append(c)
    return list(dict.fromkeys(extra))[:12], uniq_codes[:4]


def prompt_addendum(pack: Pack | None, question: str) -> str:
    if not pack:
        return ""
    lines = [ANSWER_FORMATS.get(pack.answer_format, ANSWER_FORMATS["prose"])] + list(pack.rules)
    hits = [f"{a} = {e}" for a, e in pack.glossary.items()
            if e and re.search(rf"\b{re.escape(a.lower())}\b", question.lower())]
    if hits:
        lines.append("Glossary for this question: " + "; ".join(hits[:6]) + ".")
    return "\n".join(f"- {x}" for x in lines)


def dumps(pack: Pack) -> str:
    return json.dumps(pack.to_dict())
