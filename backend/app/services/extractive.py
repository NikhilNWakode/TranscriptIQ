"""Deterministic extractive answering used in mock/demo mode (no LLM configured).

Answers are assembled from verbatim source sentences, so they are grounded and qualifier-preserving by
construction — but they are not synthesised. The UI labels this mode explicitly.
"""

from __future__ import annotations

from ..models.schemas import Evidence
from ..retrieval.text import split_sentences, tokenize


def key_sentences(query: str, text: str, max_sentences: int = 2) -> str:
    """The sentences of ``text`` that best overlap the query, in original order (verbatim)."""
    sentences = split_sentences(text)
    if len(sentences) <= max_sentences:
        return text.strip()
    q = set(tokenize(query))
    scored = [(len(q & set(tokenize(s))), -i, s) for i, s in enumerate(sentences)]
    top = sorted(scored, reverse=True)[:max_sentences]
    if top[0][0] == 0:  # no overlap: lead sentences carry the answer in Q&A transcripts
        return " ".join(sentences[:max_sentences])
    keep = sorted(top, key=lambda t: -t[1])  # restore original order
    return " ".join(s for _, _, s in keep)


def expert_summary(query: str, evidence: list[Evidence]) -> str:
    parts = [key_sentences(query, ev.text) for ev in evidence]
    return " … ".join(p for p in parts if p)


def _line(query: str, ev: Evidence) -> str:
    ts = f" at {ev.timestamp}" if ev.timestamp else ""
    return f"{ev.expert_name} ({ev.market}){ts}: “{key_sentences(query, ev.text, 2)}”"


def cross_expert_summary(query: str, grouped: dict[str, list[Evidence]], total_experts: int) -> str:
    """``grouped`` is ordered best-first, so the first expert is the strongest match for the question."""
    groups = list(grouped.values())
    lines = [f"Strongest match — {_line(query, groups[0][0])}"]
    if len(groups) > 1:
        lines.append("Also related:")
        lines += [f"• {_line(query, evs[0])}" for evs in groups[1:]]
    note = (f"Extractive demo mode ({len(groups)} of {total_experts} experts matched): verbatim excerpts ranked by "
            "retrieval score — configure an LLM provider for synthesis.")
    return "\n".join(lines + ["", note])
