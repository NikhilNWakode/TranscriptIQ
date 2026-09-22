"""SQLite storage. Evidence records (the source of truth) and their embeddings live here."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS transcripts (
    id               TEXT PRIMARY KEY,
    project_id       TEXT NOT NULL,
    filename         TEXT NOT NULL,
    file_hash        TEXT NOT NULL,
    parser_version   TEXT NOT NULL,
    expert_name      TEXT NOT NULL,
    role             TEXT NOT NULL,
    market           TEXT NOT NULL,
    metadata_overrides TEXT,
    raw_text         TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'indexed',
    warnings         TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    indexed_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence (
    id                TEXT PRIMARY KEY,
    project_id        TEXT NOT NULL,
    transcript_id     TEXT NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
    seq               INTEGER NOT NULL,
    speaker           TEXT NOT NULL,
    speaker_label     TEXT NOT NULL,
    speaker_type      TEXT NOT NULL,
    timestamp         TEXT,
    timestamp_seconds INTEGER,
    text              TEXT NOT NULL,
    question_context  TEXT,
    embedding         BLOB,
    embedding_model   TEXT
);
CREATE INDEX IF NOT EXISTS idx_evidence_transcript ON evidence(transcript_id, seq);
CREATE INDEX IF NOT EXISTS idx_evidence_project ON evidence(project_id, speaker_type);

CREATE TABLE IF NOT EXISTS embedding_cache (
    key     TEXT PRIMARY KEY,
    model   TEXT NOT NULL,
    vector  BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_cache (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class Database:
    """Thin wrapper around a SQLite file with a process-wide write lock."""

    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL") if self.path != ":memory:" else None
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def query(self, sql: str, params: tuple | list = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: tuple | list = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
