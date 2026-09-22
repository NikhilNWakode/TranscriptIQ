"""Persistent LLM response cache keyed by provider, model, prompt version and exact prompt content.

Because prompts embed the retrieved evidence text, any transcript change produces a new key automatically.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import TypeVar

from pydantic import BaseModel

from ..db import Database

T = TypeVar("T", bound=BaseModel)


class LLMCache:
    def __init__(self, db: Database):
        self.db = db

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()

    def get(self, key: str, schema: type[T]) -> T | None:
        row = self.db.query_one("SELECT value FROM llm_cache WHERE key = ?", (key,))
        if row is None:
            return None
        try:
            return schema.model_validate_json(row["value"])
        except Exception:
            return None

    def set(self, key: str, value: BaseModel) -> None:
        with self.db.tx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO llm_cache(key, value, created_at) VALUES (?, ?, ?)",
                (key, value.model_dump_json(), datetime.now(timezone.utc).isoformat(timespec="seconds")),
            )

    def clear(self) -> None:
        with self.db.tx() as conn:
            conn.execute("DELETE FROM llm_cache")
