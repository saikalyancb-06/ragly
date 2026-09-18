"""Query router: decide which retrieval mechanisms a question needs, before retrieving.

Rule-based on purpose - it must be instant, deterministic and explainable on an edge
device. Each route carries a plain-language reason that the UI shows to the user.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .autotune import ENTITY_PATTERNS
from .graph import is_entity_question
from .tables import is_table_question

ROUTES = ("semantic", "keyword", "table", "image", "entity", "compare", "multimodal")

_IMAGE_WORDS = re.compile(
    r"\b(photo|photos|photograph|photographs|image|images|picture|pictures|screenshot|screenshots|"
    r"scan|scans|diagram|diagrams|figure|figures|chart|charts|drawing|drawings|damaged|looks like|"
    r"visually|similar image|this image|show me the)\b", re.I)
_COMPARE_WORDS = re.compile(
    r"\b(compare|difference|differences|what changed|changed between|versus|vs\.?|diff|"
    r"redline|amended|revision|older version|newer version)\b", re.I)
_KEYWORD_SIGNALS = re.compile(r"\"[^\"]+\"|\b[A-Z]{2,6}\b|\b[A-Z]{1,4}[- ]?\d{2,6}\b|\bP/?N\b", re.I)
_DEFINITION = re.compile(r"\b(what does|what is|meaning of|define|explain)\b", re.I)
_ENTITY_EXTRA = re.compile(
    r"\b(relate[sd]?\s+to|related\s+to|linked\s+to|connected\s+to|associated\s+with|"
    r"everything\s+(?:about|related|on)|all\s+documents?\s+(?:about|for|from)|"
    r"which\s+(?:vendor|supplier|company|client|person)|who\s+is|belongs?\s+to)\b", re.I)


@dataclass
class Route:
    routes: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)
    entities: list[str] = field(default_factory=list)
    codes: list[str] = field(default_factory=list)
    has_image_query: bool = False

    @property
    def primary(self) -> str:
        for r in ("compare", "table", "image", "entity"):
            if r in self.routes:
                return r
        return "semantic"

    def to_dict(self) -> dict:
        return {"routes": self.routes, "primary": self.primary, "reasons": self.reasons,
                "entities": self.entities, "codes": self.codes, "image_query": self.has_image_query}


def classify(question: str, has_image: bool = False, doc_ids: list[int] | None = None) -> Route:
    """Classify a query into one or more retrieval routes."""
    q = (question or "").strip()
    r = Route(has_image_query=has_image)

    def add(name: str, why: str) -> None:
        if name not in r.routes:
            r.routes.append(name)
        r.reasons.setdefault(name, why)

    if has_image:
        add("image", "an image was supplied, so visual similarity and OCR of that image are used")
        add("multimodal", "image plus text retrieval are combined")

    if _COMPARE_WORDS.search(q) and (doc_ids is None or len(doc_ids or []) >= 2):
        add("compare", "the question asks what changed, so the two documents are diffed")

    if is_table_question(q):
        add("table", "numeric or aggregate question: tables are searched and the arithmetic is computed locally")

    if _IMAGE_WORDS.search(q):
        add("image", "the question mentions images, photos or diagrams")

    if is_entity_question(q) or _ENTITY_EXTRA.search(q):
        add("entity", "the question is about a named entity, so the relationship graph is expanded")

    codes = []
    for kind in ("code", "part_number", "clause_ref"):
        codes += [(m.group(1) if m.groups() else m.group(0)) for m in ENTITY_PATTERNS[kind].finditer(q.upper())]
    r.codes = list(dict.fromkeys(c.strip() for c in codes))[:6]
    if r.codes or _KEYWORD_SIGNALS.search(q):
        add("keyword", "the question contains a code, acronym or quoted phrase: exact keyword matching matters")

    # semantic search is always part of the plan unless the query is purely an image lookup
    if not r.routes or not (set(r.routes) <= {"image", "multimodal"}):
        add("semantic", "meaning-based search over document text")

    if _DEFINITION.search(q) and "keyword" not in r.routes:
        add("keyword", "definition-style question: the term itself is matched exactly as well")

    return r


def explain(route: Route) -> str:
    """One line for the UI: why these mechanisms were chosen."""
    return " · ".join(f"{name}: {route.reasons[name]}" for name in route.routes)
