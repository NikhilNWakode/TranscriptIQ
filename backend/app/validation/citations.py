"""Citation validation and source resolution.

The LLM returns evidence IDs (plus an optional verbatim highlight span). Everything the user sees
— quote, timestamp, expert, market — is looked up here from the database. Anything that fails a
check is dropped, never repaired by guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..models.schemas import Evidence, Highlight, LLMEvidenceRef
from ..retrieval.text import normalise_for_match
from ..services.store import EvidenceStore


@dataclass
class ResolutionResult:
    evidence: list[Evidence] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)  # human-readable reasons


def locate_quote(quote: str, source: str) -> Highlight | None:
    """Find ``quote`` inside ``source`` modulo whitespace/quote-style/case. Returns offsets into ``source``.

    Returns None when the quote is not a verbatim substring — an unverified quote is never shown.
    """
    if not quote or not quote.strip():
        return None
    target = normalise_for_match(quote).strip(" .\"'")
    if len(target) < 3:
        return None
    # Build normalised source with a map back to original character offsets.
    norm_chars: list[str] = []
    offsets: list[int] = []
    prev_space = True
    for i, ch in enumerate(source):
        c = normalise_for_match(ch) if not ch.isspace() else " "
        if c == " ":
            if prev_space:
                continue
            prev_space = True
        else:
            prev_space = False
        for cc in c or "":
            norm_chars.append(cc)
            offsets.append(i)
    norm_source = "".join(norm_chars)
    pos = norm_source.find(target)
    if pos < 0:
        return None
    start = offsets[pos]
    end = offsets[pos + len(target) - 1] + 1
    return Highlight(start=start, end=end, text=source[start:end])


def quote_is_verbatim(quote: str, source: str) -> bool:
    return locate_quote(quote, source) is not None


_ID_RE = re.compile(r"^[a-z0-9_]{1,160}$")


def resolve_references(
    refs: list[LLMEvidenceRef] | list[str],
    store: EvidenceStore,
    *,
    allowed_ids: set[str] | None = None,
    transcript_id: str | None = None,
) -> ResolutionResult:
    """Validate references and resolve them to source evidence.

    Checks: well-formed ID · exists in this project · transcript is active · (optionally) was part of
    the retrieved context · (optionally) belongs to the expected transcript · highlight is verbatim.
    """
    result = ResolutionResult()
    normalised: list[LLMEvidenceRef] = [
        r if isinstance(r, LLMEvidenceRef) else LLMEvidenceRef(evidence_id=r, highlight=None) for r in refs
    ]
    wanted = [r.evidence_id.strip() for r in normalised]
    resolved = store.get_evidence([w for w in wanted if _ID_RE.match(w)])
    seen: set[str] = set()
    for ref, eid in zip(normalised, wanted):
        if eid in seen:
            continue
        if not _ID_RE.match(eid):
            result.dropped.append(f"{eid[:40]}: malformed evidence id")
            continue
        ev = resolved.get(eid)
        if ev is None:
            result.dropped.append(f"{eid}: unknown evidence id")
            continue
        if allowed_ids is not None and eid not in allowed_ids:
            result.dropped.append(f"{eid}: not part of the retrieved context")
            continue
        if transcript_id is not None and ev.transcript_id != transcript_id:
            result.dropped.append(f"{eid}: belongs to a different expert")
            continue
        if ev.speaker_type != "expert":
            result.dropped.append(f"{eid}: not an expert statement")
            continue
        if not store.transcript_exists(ev.transcript_id):
            result.dropped.append(f"{eid}: transcript no longer indexed")
            continue
        ev = ev.model_copy()
        if ref.highlight:
            ev.highlight = locate_quote(ref.highlight, ev.text)  # silently discard unverifiable spans
        seen.add(eid)
        result.evidence.append(ev)
    return result
