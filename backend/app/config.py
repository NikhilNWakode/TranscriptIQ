"""Application settings, loaded from environment variables / the repo-level .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(REPO_ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value.strip() if value is not None and value.strip() != "" else default


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


@dataclass(frozen=True)
class Settings:
    project_id: str = field(default_factory=lambda: _env("PROJECT_ID", "default"))
    transcripts_dir: Path = field(default_factory=lambda: _resolve(_env("TRANSCRIPTS_DIR", "transcripts")))
    interview_guide_path: Path = field(
        default_factory=lambda: _resolve(_env("INTERVIEW_GUIDE_PATH", "config/interview_guide.json"))
    )
    query_expansion_path: Path = field(
        default_factory=lambda: _resolve(_env("QUERY_EXPANSION_PATH", "config/query_expansion.json"))
    )
    database_path: Path = field(default_factory=lambda: _resolve(_env("DATABASE_PATH", "backend/data/transcriptiq.db")))

    # LLM: anthropic | gemini | openai_compatible | mock
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "mock").lower())
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", ""))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", ""))
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", ""))
    llm_effort: str = field(default_factory=lambda: _env("LLM_EFFORT", ""))
    llm_timeout_s: float = field(default_factory=lambda: float(_env("LLM_TIMEOUT_S", "120")))

    # Embeddings: local | gemini | openai_compatible
    embedding_provider: str = field(default_factory=lambda: _env("EMBEDDING_PROVIDER", "local").lower())
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", ""))
    embedding_api_key: str = field(default_factory=lambda: _env("EMBEDDING_API_KEY", ""))
    embedding_base_url: str = field(default_factory=lambda: _env("EMBEDDING_BASE_URL", ""))

    # Retrieval tuning
    semantic_weight: float = field(default_factory=lambda: float(_env("SEMANTIC_WEIGHT", "0.6")))
    per_expert_k: int = field(default_factory=lambda: int(_env("PER_EXPERT_K", "3")))
    max_context_segments: int = field(default_factory=lambda: int(_env("MAX_CONTEXT_SEGMENTS", "40")))
    guide_batch_size: int = field(default_factory=lambda: int(_env("GUIDE_BATCH_SIZE", "8")))

    # Tokens that are document-conversion artifacts, never transcript content.
    artifact_tokens: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            t.strip().lower() for t in _env("ARTIFACT_TOKENS", "canvas").split(",") if t.strip()
        )
    )

    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            o.strip() for o in _env("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
        )
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
