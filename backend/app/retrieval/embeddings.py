"""Embedding providers behind a single interface.

* ``local``             — deterministic hashed bag-of-words/bigrams. Offline, free, no API key.
* ``gemini``            — Google ``gemini-embedding-001`` (free tier available via Google AI Studio).
* ``openai_compatible`` — any ``/v1/embeddings`` endpoint (OpenAI, Ollama, LM Studio, ...).

Vectors are L2-normalised and cached by (model, kind, text) so unchanged text is never re-embedded.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Literal

import httpx
import numpy as np

from ..config import Settings
from ..db import Database
from .text import tokenize

log = logging.getLogger(__name__)
Kind = Literal["document", "query"]


class EmbeddingError(Exception):
    pass


def _normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class EmbeddingService(ABC):
    model_id: str

    @abstractmethod
    def _embed(self, texts: list[str], kind: Kind) -> np.ndarray: ...

    def __init__(self, db: Database | None = None):
        self.db = db

    def _cache_key(self, text: str, kind: Kind) -> str:
        return hashlib.sha256(f"{self.model_id}\x00{kind}\x00{text}".encode()).hexdigest()

    def embed(self, texts: list[str], kind: Kind = "document") -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        results: list[np.ndarray | None] = [None] * len(texts)
        keys = [self._cache_key(t, kind) for t in texts]
        if self.db is not None:
            for i, key in enumerate(keys):
                row = self.db.query_one("SELECT vector FROM embedding_cache WHERE key = ?", (key,))
                if row is not None:
                    results[i] = np.frombuffer(row["vector"], dtype=np.float32)
        missing = [i for i, r in enumerate(results) if r is None]
        if missing:
            fresh = _normalise(np.asarray(self._embed([texts[i] for i in missing], kind), dtype=np.float32))
            for j, i in enumerate(missing):
                results[i] = fresh[j]
            if self.db is not None:
                with self.db.tx() as conn:
                    conn.executemany(
                        "INSERT OR REPLACE INTO embedding_cache(key, model, vector) VALUES (?, ?, ?)",
                        [(keys[i], self.model_id, fresh[j].tobytes()) for j, i in enumerate(missing)],
                    )
        return np.vstack(results)  # type: ignore[arg-type]


class LocalHashEmbedding(EmbeddingService):
    """Feature-hashed unigrams + bigrams with sublinear TF. Captures lexical overlap, not deep semantics."""

    def __init__(self, db: Database | None = None, dim: int = 1024):
        super().__init__(db)
        self.dim = dim
        self.model_id = f"local-hash-{dim}"

    def _embed(self, texts: list[str], kind: Kind) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            toks = tokenize(text)
            feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
            for f in feats:
                h = int.from_bytes(hashlib.md5(f.encode()).digest()[:8], "little")
                mat[row, h % self.dim] += 1.0 if (h >> 63) == 0 else -1.0
            mat[row] = np.sign(mat[row]) * np.log1p(np.abs(mat[row]))
        return mat


class GeminiEmbedding(EmbeddingService):
    BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, api_key: str, model: str, db: Database | None = None, timeout: float = 60):
        super().__init__(db)
        if not api_key:
            raise EmbeddingError("EMBEDDING_API_KEY (or LLM_API_KEY) is required for Gemini embeddings")
        self.api_key, self.model, self.timeout = api_key, model or "gemini-embedding-001", timeout
        self.model_id = f"gemini:{self.model}"

    def _embed(self, texts: list[str], kind: Kind) -> np.ndarray:
        task = "RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT"
        out: list[list[float]] = []
        try:
            with httpx.Client(timeout=self.timeout) as client:
                for start in range(0, len(texts), 100):
                    batch = texts[start:start + 100]
                    resp = client.post(
                        f"{self.BASE}/models/{self.model}:batchEmbedContents",
                        headers={"x-goog-api-key": self.api_key},
                        json={"requests": [
                            {"model": f"models/{self.model}", "content": {"parts": [{"text": t}]}, "taskType": task}
                            for t in batch
                        ]},
                    )
                    resp.raise_for_status()
                    out.extend(e["values"] for e in resp.json()["embeddings"])
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise EmbeddingError(f"Gemini embedding request failed: {exc.__class__.__name__}") from exc
        return np.asarray(out, dtype=np.float32)


class OpenAICompatibleEmbedding(EmbeddingService):
    def __init__(self, api_key: str, model: str, base_url: str, db: Database | None = None, timeout: float = 60):
        super().__init__(db)
        if not model:
            raise EmbeddingError("EMBEDDING_MODEL is required for openai_compatible embeddings")
        self.api_key, self.model = api_key, model
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.timeout = timeout
        self.model_id = f"oai:{self.base_url}:{model}"

    def _embed(self, texts: list[str], kind: Kind) -> np.ndarray:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        out: list[list[float]] = []
        try:
            with httpx.Client(timeout=self.timeout) as client:
                for start in range(0, len(texts), 64):
                    resp = client.post(f"{self.base_url}/embeddings", headers=headers,
                                       json={"model": self.model, "input": texts[start:start + 64]})
                    resp.raise_for_status()
                    data = sorted(resp.json()["data"], key=lambda d: d["index"])
                    out.extend(d["embedding"] for d in data)
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise EmbeddingError(f"Embedding request failed: {exc.__class__.__name__}") from exc
        return np.asarray(out, dtype=np.float32)


def build_embedding_service(settings: Settings, db: Database | None) -> EmbeddingService:
    provider = settings.embedding_provider
    key = settings.embedding_api_key or settings.llm_api_key
    try:
        if provider == "gemini":
            return GeminiEmbedding(key, settings.embedding_model, db)
        if provider in {"openai", "openai_compatible"}:
            return OpenAICompatibleEmbedding(key, settings.embedding_model, settings.embedding_base_url, db)
    except EmbeddingError as exc:
        log.warning("Embedding provider '%s' unavailable (%s); falling back to local embeddings.", provider, exc)
    return LocalHashEmbedding(db)
