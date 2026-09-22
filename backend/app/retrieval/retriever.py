"""Hybrid retrieval over expert evidence: dense cosine + BM25, with expert-balanced selection.

The in-memory index is rebuilt lazily whenever the corpus version changes (after refresh or a
metadata edit). For 3–100 transcripts (hundreds to a few thousand segments) a numpy matrix is
faster and simpler than an external vector DB; see README "Scaling" for the pgvector path.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..db import Database
from ..ingestion.indexer import embedding_text
from ..models.schemas import AskFilters
from ..services.store import effective_metadata
from .bm25 import BM25
from .embeddings import EmbeddingError, EmbeddingService, LocalHashEmbedding
from .text import content_terms, fold, tokenize

log = logging.getLogger(__name__)


class QueryExpander:
    """Query-side synonym expansion loaded from config (never touches evidence text)."""

    def __init__(self, groups: list[list[str]] | None = None):
        self.groups = [[fold(t).strip() for t in g if t.strip()] for g in (groups or [])]
        self._patterns = [[re.compile(rf"\b{re.escape(t)}\b") for t in g] for g in self.groups]
        # stemmed token -> stems of all synonyms in its groups (for coverage checks)
        self.alternatives: dict[str, set[str]] = {}
        for g in self.groups:
            stems = {s for t in g for s in tokenize(t)}
            for s in stems:
                self.alternatives.setdefault(s, set()).update(stems)

    @classmethod
    def from_file(cls, path: Path | None) -> "QueryExpander":
        if path is None or not path.exists():
            return cls()
        try:
            return cls(json.loads(path.read_text(encoding="utf-8")).get("groups", []))
        except (ValueError, OSError, AttributeError):
            log.warning("Ignoring invalid query expansion file %s", path.name)
            return cls()

    def expand(self, query: str) -> str:
        q = fold(query)
        extra: list[str] = []
        for group, pats in zip(self.groups, self._patterns):
            if any(p.search(q) for p in pats):
                extra.extend(t for t, p in zip(group, pats) if not p.search(q))
        return f"{query} {' '.join(extra)}" if extra else query


@dataclass
class Hit:
    evidence_id: str
    transcript_id: str
    score: float
    semantic: float
    lexical: float


@dataclass
class _Index:
    version: int
    ids: list[str]
    transcript_ids: list[str]
    markets: list[str]
    roles: list[str]
    matrix: np.ndarray | None
    has_vec: np.ndarray
    bm25: BM25
    term_sets: list[set[str]]
    doc_freq: Counter = field(default_factory=Counter)


class Retriever:
    def __init__(self, db: Database, embedder: EmbeddingService, project_id: str, semantic_weight: float = 0.6,
                 expander: QueryExpander | None = None):
        self.db, self.embedder, self.project_id = db, embedder, project_id
        self.semantic_weight = semantic_weight
        self.expander = expander or QueryExpander()
        self._index: _Index | None = None
        self._version = 0
        self._lock = threading.Lock()

    def invalidate(self) -> None:
        with self._lock:
            self._version += 1

    # ------------------------------------------------------------------------------------------
    def _build(self) -> _Index:
        rows = self.db.query(
            """SELECT e.id, e.transcript_id, e.text, e.question_context, e.embedding, e.embedding_model,
                      t.expert_name, t.role, t.market, t.metadata_overrides
               FROM evidence e JOIN transcripts t ON t.id = e.transcript_id
               WHERE e.project_id = ? AND t.status = 'indexed' AND e.speaker_type = 'expert'
               ORDER BY e.transcript_id, e.seq""",
            (self.project_id,),
        )
        metas = [effective_metadata(r) for r in rows]
        docs = [tokenize(embedding_text(m, r["question_context"], r["text"])) for m, r in zip(metas, rows)]
        vecs, has = [], []
        for r in rows:
            ok = r["embedding"] is not None and r["embedding_model"] == self.embedder.model_id
            has.append(ok)
            vecs.append(np.frombuffer(r["embedding"], dtype=np.float32) if ok else None)
        dim = next((v.shape[0] for v in vecs if v is not None), 0)
        matrix = None
        if dim:
            matrix = np.vstack([v if v is not None else np.zeros(dim, np.float32) for v in vecs])
        return _Index(
            version=self._version,
            ids=[r["id"] for r in rows],
            transcript_ids=[r["transcript_id"] for r in rows],
            markets=[m["market"] for m in metas],
            roles=[m["role"] for m in metas],
            matrix=matrix,
            has_vec=np.array(has, dtype=bool),
            bm25=BM25(docs),
            term_sets=[set(d) for d in docs],
            doc_freq=Counter(t for d in docs for t in set(d)),
        )

    def index(self) -> _Index:
        with self._lock:
            if self._index is None or self._index.version != self._version:
                self._index = self._build()
            return self._index

    # ------------------------------------------------------------------------------------------
    def _mask(self, idx: _Index, filters: AskFilters | None, transcript_ids: set[str] | None) -> np.ndarray:
        n = len(idx.ids)
        mask = np.ones(n, dtype=bool)
        if transcript_ids is not None:
            mask &= np.array([t in transcript_ids for t in idx.transcript_ids], dtype=bool)
        if filters:
            if filters.transcript_ids:
                s = set(filters.transcript_ids)
                mask &= np.array([t in s for t in idx.transcript_ids], dtype=bool)
            if filters.markets:
                s = {m.lower() for m in filters.markets}
                mask &= np.array([m.lower() in s for m in idx.markets], dtype=bool)
            if filters.roles:
                s = {r.lower() for r in filters.roles}
                mask &= np.array([r.lower() in s for r in idx.roles], dtype=bool)
        return mask

    def score(self, query: str, filters: AskFilters | None = None,
              transcript_ids: set[str] | None = None) -> tuple[_Index, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = self.index()
        n = len(idx.ids)
        mask = self._mask(idx, filters, transcript_ids) if n else np.zeros(0, dtype=bool)
        expanded = self.expander.expand(query)
        sem = np.zeros(n, dtype=np.float32)
        if n and idx.matrix is not None:
            try:
                # lexical embeddings benefit from expansion; true semantic embeddings get the question as asked
                q_text = expanded if isinstance(self.embedder, LocalHashEmbedding) else query
                q = self.embedder.embed([q_text], "query")[0]
                if q.shape[0] == idx.matrix.shape[1]:
                    sem = np.clip(idx.matrix @ q, 0, None) * idx.has_vec
            except EmbeddingError:
                pass  # degrade to keyword-only retrieval
        lex = np.asarray(idx.bm25.scores(tokenize(expanded)), dtype=np.float32) if n else np.zeros(0, np.float32)

        def norm(x: np.ndarray) -> np.ndarray:
            m = x[mask].max() if mask.any() else 0.0
            return x / m if m > 0 else np.zeros_like(x)

        w = self.semantic_weight if idx.matrix is not None else 0.0
        hybrid = w * norm(sem) + (1 - w) * norm(lex)
        hybrid = np.where(mask, hybrid, -1.0)
        return idx, hybrid, sem, lex, mask

    def retrieve(self, query: str, *, filters: AskFilters | None = None, transcript_ids: set[str] | None = None,
                 per_expert_k: int = 3, max_total: int = 40, relative_floor: float = 0.3) -> list[Hit]:
        """Top evidence per expert (so no single expert dominates), then globally ranked and capped."""
        idx, hybrid, sem, lex, mask = self.score(query, filters, transcript_ids)
        if not mask.any():
            return []
        best = float(hybrid[mask].max())
        if best <= 0:
            return []
        by_expert: dict[str, list[int]] = {}
        for i in np.argsort(-hybrid):
            if not mask[i] or hybrid[i] <= 0:
                continue
            bucket = by_expert.setdefault(idx.transcript_ids[i], [])
            if len(bucket) < per_expert_k and hybrid[i] >= relative_floor * best:
                bucket.append(int(i))
        chosen = sorted((i for b in by_expert.values() for i in b), key=lambda i: -hybrid[i])[:max_total]
        return [Hit(idx.ids[i], idx.transcript_ids[i], float(hybrid[i]), float(sem[i]), float(lex[i])) for i in chosen]

    def retrieve_for_expert(self, query: str, transcript_id: str, k: int = 4) -> list[Hit]:
        return self.retrieve(query, transcript_ids={transcript_id}, per_expert_k=k, max_total=k, relative_floor=0.0)

    def term_coverage(self, query: str, hits: list[Hit]) -> float:
        """IDF-weighted share of the question's subject terms (or configured synonyms) found in the retrieved evidence.

        Weighting matters: corpus-wide topic words ("robotic", "surgery", "hospital") are cheap to match, while a term
        that never occurs in any transcript ("company", "price", "25%") carries the most weight. A question whose
        specific subject is absent therefore scores low even though its generic words are everywhere.
        """
        terms = content_terms(query)
        if not terms:
            return 0.0
        idx = self.index()
        pos = {eid: i for i, eid in enumerate(idx.ids)}
        present: set[str] = set()
        for h in hits:
            present |= idx.term_sets[pos[h.evidence_id]]
        n_docs = max(len(idx.term_sets), 1)
        total = covered = 0.0
        for t in terms:
            alts = {t} | self.expander.alternatives.get(t, set())
            df = min((idx.doc_freq.get(a, 0) for a in alts if idx.doc_freq.get(a, 0)), default=0)
            weight = math.log(1 + n_docs / (df + 0.5))
            total += weight
            if alts & present:
                covered += weight
        return covered / total if total else 0.0
