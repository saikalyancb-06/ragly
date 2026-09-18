"""Local document comparison: what changed between two indexed documents.

Works on the stored page text, so it needs no model and runs instantly:
  * sections are matched by heading / clause number, then by text similarity
  * added, removed and modified sections are reported with page references
  * inside a modified section, changed numbers, dates, amounts, durations and
    entities are extracted so "60 days -> 90 days" is called out explicitly
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from .autotune import ENTITY_PATTERNS
from .graph import extract_entities

_HEADING = re.compile(
    r"^(?:(?P<num>\d+(?:\.\d+)*)[.)]?\s+)?(?P<title>[A-Z][A-Z0-9 &/,'()\-]{3,70})\s*$", re.M)
_CLAUSE = re.compile(r"^\s*(\d+(?:\.\d+)*)[.)]\s+", re.M)
SIM_SAME = 0.985      # above this AND no value change: unchanged
SIM_MATCH = 0.62      # below this, two sections are not the same section at all


@dataclass
class Section:
    key: str
    title: str
    page: int
    text: str

    def to_dict(self) -> dict:
        return {"key": self.key, "title": self.title, "page": self.page,
                "preview": self.text[:280] + ("…" if len(self.text) > 280 else "")}


@dataclass
class Change:
    kind: str                      # added | removed | modified | unchanged
    title: str
    left_page: int | None
    right_page: int | None
    similarity: float
    value_changes: list[dict] = field(default_factory=list)
    entity_changes: list[dict] = field(default_factory=list)
    text_diff: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__


def split_sections(pages: list[dict]) -> list[Section]:
    """Split page text into sections on headings / clause numbers; whole pages if neither exists."""
    sections: list[Section] = []
    for p in pages:
        text = (p.get("text") or "").strip()
        if not text:
            continue
        page_no = int(p.get("page", 1))
        marks = [(m.start(), (m.group("num") or "").strip(), m.group("title").strip())
                 for m in _HEADING.finditer(text)]
        if not marks:
            marks = [(m.start(), m.group(1), "") for m in _CLAUSE.finditer(text)]
        if not marks:
            sections.append(Section(key=f"page-{page_no}", title=f"Page {page_no}", page=page_no, text=text))
            continue
        bounds = [m[0] for m in marks] + [len(text)]
        for i, (_, num, title) in enumerate(marks):
            body = text[bounds[i]:bounds[i + 1]].strip()
            key = (num or title or f"page-{page_no}-{i}").lower()
            label = f"{num} {title}".strip() or f"Page {page_no} part {i + 1}"
            sections.append(Section(key=key, title=label, page=page_no, text=body))
    return sections


def _values(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for kind in ("measurement", "money", "duration", "date", "code", "part_number", "clause_ref"):
        found = [(m.group(1) if m.groups() else m.group(0)).strip()
                 for m in ENTITY_PATTERNS[kind].finditer(text)]
        if found:
            out[kind] = list(dict.fromkeys(found))
    return out


def _value_changes(left: str, right: str) -> list[dict]:
    lv, rv = _values(left), _values(right)
    changes: list[dict] = []
    for kind in sorted(set(lv) | set(rv)):
        a, b = lv.get(kind, []), rv.get(kind, [])
        removed = [x for x in a if x not in b]
        added = [x for x in b if x not in a]
        if not removed and not added:
            continue
        # pair them up when the counts match: that is the "60 days -> 90 days" case
        if len(removed) == len(added):
            for x, y in zip(removed, added):
                changes.append({"kind": kind, "from": x, "to": y})
        else:
            for x in removed:
                changes.append({"kind": kind, "from": x, "to": None})
            for y in added:
                changes.append({"kind": kind, "from": None, "to": y})
    return changes


def _entity_changes(left: str, right: str) -> list[dict]:
    la = {(e["kind"], e["norm"]): e["name"] for e in extract_entities(left)
          if e["kind"] in ("org", "person", "place", "project", "invoice", "contract")}
    ra = {(e["kind"], e["norm"]): e["name"] for e in extract_entities(right)
          if e["kind"] in ("org", "person", "place", "project", "invoice", "contract")}
    out = [{"kind": k[0], "from": la[k], "to": None} for k in la if k not in ra]
    out += [{"kind": k[0], "from": None, "to": ra[k]} for k in ra if k not in la]
    return out


def _word_diff(left: str, right: str, limit: int = 12) -> list[str]:
    sm = difflib.SequenceMatcher(None, left.split(), right.split())
    lines: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        a = " ".join(left.split()[i1:i2])[:160]
        b = " ".join(right.split()[j1:j2])[:160]
        if tag == "replace":
            lines.append(f"changed: “{a}” → “{b}”")
        elif tag == "delete":
            lines.append(f"removed: “{a}”")
        else:
            lines.append(f"added: “{b}”")
        if len(lines) >= limit:
            break
    return lines


def compare_documents(left_doc: dict, left_pages: list[dict],
                      right_doc: dict, right_pages: list[dict]) -> dict:
    """Compare two indexed documents. Everything comes from stored page text - no model involved."""
    left = split_sections(left_pages)
    right = split_sections(right_pages)
    right_by_key = {s.key: s for s in right}
    used: set[str] = set()
    changes: list[Change] = []

    for ls in left:
        rs = right_by_key.get(ls.key)
        if rs is None:      # match by content when the heading changed or is absent
            best, best_score = None, 0.0
            for cand in right:
                if cand.key in used:
                    continue
                score = difflib.SequenceMatcher(None, ls.text, cand.text).quick_ratio()
                if score > best_score:
                    best, best_score = cand, score
            if best is not None and best_score >= SIM_MATCH:
                rs = best
        if rs is None:
            changes.append(Change("removed", ls.title, ls.page, None, 0.0,
                                  value_changes=[], entity_changes=[],
                                  text_diff=[f"removed: “{ls.text[:200]}”"]))
            continue
        used.add(rs.key)
        sim = difflib.SequenceMatcher(None, ls.text, rs.text).ratio()
        vals = _value_changes(ls.text, rs.text)
        # a single changed number keeps similarity very high, so a value change always means "modified"
        if sim >= SIM_SAME and not vals:
            changes.append(Change("unchanged", ls.title, ls.page, rs.page, round(sim, 3)))
            continue
        changes.append(Change(
            "modified", ls.title or rs.title, ls.page, rs.page, round(sim, 3),
            value_changes=vals,
            entity_changes=_entity_changes(ls.text, rs.text),
            text_diff=_word_diff(ls.text, rs.text),
        ))

    for rs in right:
        if rs.key in used:
            continue
        if any(c.kind != "removed" and c.right_page == rs.page and c.title == rs.title for c in changes):
            continue
        changes.append(Change("added", rs.title, None, rs.page, 0.0,
                              text_diff=[f"added: “{rs.text[:200]}”"]))

    order = {"modified": 0, "added": 1, "removed": 2, "unchanged": 3}
    changes.sort(key=lambda c: (order[c.kind], -(len(c.value_changes) + len(c.entity_changes))))
    summary = {k: sum(1 for c in changes if c.kind == k) for k in ("modified", "added", "removed", "unchanged")}
    value_total = sum(len(c.value_changes) for c in changes)
    return {
        "left": {"doc_id": left_doc["id"], "name": left_doc["name"], "pages": len(left_pages),
                 "sections": len(left)},
        "right": {"doc_id": right_doc["id"], "name": right_doc["name"], "pages": len(right_pages),
                  "sections": len(right)},
        "summary": {**summary, "value_changes": value_total},
        "headline": (f"{summary['modified']} sections modified, {summary['added']} added, "
                     f"{summary['removed']} removed, {value_total} value change(s)"),
        "changes": [c.to_dict() for c in changes],
    }
