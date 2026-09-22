"""Lightweight, dependency-free text normalisation shared by BM25, local embeddings and the mock LLM."""

from __future__ import annotations

import re
import unicodedata

STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have
    having he her here hers herself him himself his how i if in into is it its itself just me more most
    my myself no nor not now of off on once only or other our ours ourselves out over own same she should
    so some such than that the their theirs them themselves then there these they this those through to
    too under until up very was we were what when where which while who whom why will with would you your
    yours yourself yourselves also get got really well yes okay ok think thing things lot quite much many
    """.split()
)

# Words that describe the *question* rather than its subject ("How do experts view ...").
QUERY_META_WORDS = frozenset(
    """
    expert experts interviewee interviewees view views say says said describe described discuss discusses
    discussed mention mentions mentioned opinion opinions perspective perspectives role tell explain
    according across compare comparison transcript transcripts interview interviews answer question
    typical main key overall general
    agree agrees agreement disagree disagrees disagreement disagreements differ differs difference differences
    different similar similarity similarities contrast contrasts consensus believe believes feel feels
    """.split()
)

# Qualifier-bearing words are kept even though they look like stopwords.
KEEP = frozenset({"some", "all", "not", "only", "no", "more", "most", "few"})

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)?%?")


def fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def stem(token: str) -> str:
    """Tiny suffix stripper — good enough to match economics/economic, timelines/timeline, trained/training."""
    if token.isdigit() or len(token) <= 3:
        return token
    for suffix in ("ational", "ations", "ation", "ically", "ities", "ity", "ings", "ing", "ical", "ics",
                   "ies", "ied", "ers", "ed", "es", "ly", "ic", "al", "er", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            break
    return token


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN_RE.findall(fold(text))
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS or t in KEEP]
    return [stem(t) for t in tokens]


def content_terms(query: str) -> list[str]:
    """Subject terms of a question (no stopwords, no question-meta words, no generic qualifiers)."""
    raw = _TOKEN_RE.findall(fold(query))
    terms = [stem(t) for t in raw if t not in STOPWORDS and t not in QUERY_META_WORDS and t not in KEEP]
    seen, out = set(), []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def normalise_for_match(text: str) -> str:
    """Whitespace/quote/dash normalisation used for exact-quote verification."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip().lower()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"“])", text.strip())
    return [p.strip() for p in parts if p.strip()]
