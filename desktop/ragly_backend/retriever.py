"""Hybrid retrieval: FTS5 BM25 + cosine similarity, merged with Reciprocal Rank Fusion."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np

from .autotune import Pack, expand_query, normalise_code
from .config import settings
from .evidence import QuestionShape, classify_question
from .embedder import Embedder
from .store import Store

STOPWORDS = set(
    """a an and are as at be by can do does for from has have how i in is it its me my of on or our
    please show tell than that the their them there these they this to was we were what when where which
    who whom why will with you your about into any all also give list find""".split()
)


@dataclass
class Hit:
    chunk_id: int
    doc_id: int
    doc_name: str
    page: int
    heading: str
    text: str
    score: float        # fused RRF score
    vector_score: float  # cosine similarity (-1..1)
    keyword_rank: int | None
    exact_match: bool = False   # chunk literally contains a code from the question
    graph_reason: str = ""      # why the entity graph surfaced this chunk
    content_type: str = ""      # page content type (text / table / scanned_text / ...)
    rerank_score: float = 0.0   # score after reranking the candidate pool
    why: list[str] = field(default_factory=list)   # plain-language reasons this passage was kept


def fts_query(question: str, extra: list[str] | None = None) -> str:
    words = [w for w in re.findall(r"\w+", question.lower()) if len(w) > 1 and w not in STOPWORDS]
    words += [w.lower() for w in (extra or []) if w.isalnum()]
    seen: list[str] = []
    for w in words:
        if w not in seen:
            seen.append(w)
    return " OR ".join(f'"{w}"' for w in seen[:24])


class Retriever:
    def __init__(self, store: Store, embedder: Embedder):
        self.store = store
        self.embedder = embedder
        self.pack: Pack | None = None   # set by App once the auto-tuned pack is loaded
        self.graph = None               # GraphRetriever, set by App
        self.graph_weight = 1.2

    # ------------------------------------------------------------------ reranking
    @staticmethod
    def rerank(question: str, hits: list[Hit], shape: QuestionShape, top_k: int) -> list[Hit]:
        """Score a larger candidate pool on what the question actually asks for, then keep the best few.

        Cosine similarity alone puts "same topic" above "answers the question"; these features fix that.
        """
        import re as _re

        q_terms = {w for w in _re.findall(r"[a-z0-9][\w-]{2,}", question.lower())}
        for hit in hits:
            text = hit.text
            low = text.lower()
            score = 1.6 * (hit.vector_score or 0.0) + 1.2 * (hit.score or 0.0)
            why: list[str] = []
            if hit.exact_match:
                score += 1.0
                why.append("exact code or part number match")
            if hit.keyword_rank is not None:
                score += max(0.0, 0.5 - 0.02 * hit.keyword_rank)
                why.append(f"keyword rank {hit.keyword_rank + 1}")
            if hit.graph_reason:
                score += 0.35
                why.append(hit.graph_reason)

            # A passage that lists questions or describes how a system should behave is writing
            # ABOUT the subject, not stating it. It is weaker evidence than the page that gives
            # the fact, so it drops below it rather than filling the prompt.
            meta_marks = sum(1 for marker in (
                "expected behavior", "expected behaviour", "should abstain", "should return",
                "a robust system", "evaluation questions", "adversarial test", "test corpus",
                "ground truth", "what a robust system", "use these queries") if marker in low)
            questions_in_text = low.count("?")
            if meta_marks or questions_in_text >= 4:
                score -= 0.5 * min(3, meta_marks) + 0.1 * min(5, questions_in_text)
                why.append("describes a test rather than stating the fact")

            entity_hits = [e for e in shape.entities
                           if _re.sub(r"[^a-z0-9]", "", e.lower()) in _re.sub(r"[^a-z0-9]", "", low)]
            if entity_hits:
                score += 0.8
                why.append(f"mentions {', '.join(entity_hits[:2])}")
            attr_hits = [a for a in shape.attributes if a in low]
            if attr_hits:
                score += 0.25 * min(3, len(attr_hits))
                why.append(f"discusses {', '.join(attr_hits[:2])}")
            overlap = len(q_terms & set(_re.findall(r"[a-z0-9][\w-]{2,}", low)))
            score += min(0.6, 0.05 * overlap)

            if shape.wants_value:
                from .evidence import VALUE_PATTERNS

                pattern = VALUE_PATTERNS.get(shape.wants_value, VALUE_PATTERNS["number"])
                if pattern.search(text):
                    score += 0.55
                    why.append(f"contains {shape.wants_value} values")
                else:
                    score -= 0.35
            if shape.intent == "numeric" and hit.content_type == "table":
                score += 0.3
                why.append("table page")
            if hit.heading and any(a in hit.heading.lower() for a in shape.attributes):
                score += 0.3
                why.append(f"section “{hit.heading}”")
            hit.rerank_score = round(score, 4)
            hit.why = why
        return sorted(hits, key=lambda h: -h.rerank_score)[:top_k]

    def search(self, question: str, top_k: int | None = None, doc_ids: list[int] | None = None,
               tuning: bool = True, use_graph: bool = True,
               content_types: list[str] | None = None, rerank: bool = True) -> tuple[list[Hit], dict]:
        t0 = time.perf_counter()
        top_k = top_k or settings.top_k
        shape = classify_question(question)
        pool = max(settings.candidates, top_k * settings.pool_factor) if rerank else top_k
        n = settings.candidates

        pack = self.pack if tuning else None
        extra_terms, codes = expand_query(question, pack)
        kw_w = pack.keyword_weight if pack else 1.0
        vec_w = pack.vector_weight if pack else 1.0

        qvec = self.embedder.embed_query(question)
        t_embed = time.perf_counter()

        mat, ids, docs = self.store.matrix()
        vec_rank: dict[int, int] = {}
        vec_score: dict[int, float] = {}
        if len(ids):
            sims = mat @ qvec
            if doc_ids:
                sims = np.where(np.isin(docs, doc_ids), sims, -np.inf)
            k = min(n, len(sims))
            idx = np.argpartition(-sims, k - 1)[:k]
            idx = idx[np.argsort(-sims[idx])]
            for r, i in enumerate(idx):
                if np.isfinite(sims[i]):
                    vec_rank[int(ids[i])] = r
                    vec_score[int(ids[i])] = float(sims[i])

        kw_ids: list[int] = []
        match = fts_query(question, extra_terms)
        if match:
            kw_ids = self.store.fts_search(match, n, doc_ids)
        kw_rank = {cid: r for r, cid in enumerate(kw_ids)}

        # exact code / part-number hits (tuning only): "E-47" must beat fuzzy similarity
        # third ranked list: chunks reachable from the entities named in the question
        graph_ids: list[int] = []
        graph_why: dict[int, str] = {}
        if use_graph and self.graph is not None:
            try:
                graph_ids, explanations = self.graph.chunk_ids_for(question, limit=n)
                for row in explanations:
                    for cid in row.get("chunk_ids", []) or []:
                        graph_why.setdefault(cid, f"{row.get('node', '')}: {row.get('why', 'entity match')}")
            except Exception:
                graph_ids = []

        exact_ids: list[int] = []
        if pack and codes:
            exact_ids = self.store.entity_chunks([normalise_code(c) for c in codes], limit=top_k, doc_ids=doc_ids)

        fused: dict[int, float] = {}
        for cid, r in vec_rank.items():
            fused[cid] = fused.get(cid, 0.0) + vec_w / (settings.rrf_k + r + 1)
        for cid, r in kw_rank.items():
            fused[cid] = fused.get(cid, 0.0) + kw_w / (settings.rrf_k + r + 1)
        for r, cid in enumerate(exact_ids):
            fused[cid] = fused.get(cid, 0.0) + 2.0 / (settings.rrf_k + r + 1)
        for r, cid in enumerate(graph_ids):
            fused[cid] = fused.get(cid, 0.0) + self.graph_weight / (settings.rrf_k + r + 1)

        if content_types:
            keep = self.store.chunks_with_content_type(list(fused), content_types)
            fused = {c: v for c, v in fused.items() if c in keep}

        best = sorted(fused, key=lambda c: -fused[c])[:pool]
        rows = self.store.get_chunks(best)

        # keyword-only hits have no cosine yet: compute it so the grounding guard can use it
        missing = [c for c in best if c not in vec_score]
        if missing and len(ids):
            pos = {int(c): i for i, c in enumerate(ids)}
            for c in missing:
                if c in pos:
                    vec_score[c] = float(mat[pos[c]] @ qvec)

        hits = [
            Hit(
                chunk_id=c,
                doc_id=rows[c]["doc_id"],
                doc_name=rows[c]["doc_name"],
                page=rows[c]["page"],
                heading=rows[c]["heading"],
                text=rows[c]["text"],
                score=round(fused[c], 5),
                vector_score=round(vec_score.get(c, 0.0), 4),
                keyword_rank=kw_rank.get(c),
                exact_match=c in exact_ids,
                graph_reason=graph_why.get(c, ""),
                content_type=rows[c].get("content_type", "text"),
            )
            for c in best
            if c in rows
        ]
        if rerank and hits:
            t_rr = time.perf_counter()
            hits = self.rerank(question, hits, shape, top_k)
            rerank_ms = round((time.perf_counter() - t_rr) * 1000, 2)
        else:
            hits = hits[:top_k]
            rerank_ms = 0.0

        timing = {
            "embed_ms": round((t_embed - t0) * 1000, 1),
            "search_ms": round((time.perf_counter() - t_embed) * 1000, 1),
            "searched_chunks": int(len(ids)),
            "tuning": bool(pack),
            "expanded_terms": extra_terms,
            "codes": codes,
            "exact_hits": len(exact_ids),
            "graph_hits": len(graph_ids),
            "content_types": content_types or [],
            "candidates": len(best),
            "rerank_ms": rerank_ms,
            "question_shape": shape.to_dict(),
        }
        return hits, timing
