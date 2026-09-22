"""Idempotent folder → index synchronisation.

unchanged file → skip · new file → parse + embed + index · modified file → re-parse + re-embed ·
deleted file → removed from the active index. File identity is the SHA-256 of its bytes plus the
parser version, so parser upgrades also trigger a re-index.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from ..db import Database
from ..models.schemas import IngestReport
from ..retrieval.embeddings import EmbeddingError, EmbeddingService
from .loaders import TranscriptLoadError, discover_transcripts, load_text
from .parser import PARSER_VERSION, ParsedTranscript, parse_transcript

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def evidence_ids(transcript_id: str, parsed: ParsedTranscript) -> list[str]:
    """Deterministic, human-readable evidence IDs: <transcript>_<MM>_<SS> (suffixed on collision)."""
    ids, used = [], set()
    for seg in parsed.segments:
        base = f"{transcript_id}_{seg.timestamp.replace(':', '_')}" if seg.timestamp else f"{transcript_id}_s{seg.seq:03d}"
        if seg.speaker_type == "interviewer":
            base += "_q"
        candidate, n = base, 2
        while candidate in used:
            candidate, n = f"{base}_{n}", n + 1
        used.add(candidate)
        ids.append(candidate)
    return ids


def embedding_text(meta: dict[str, str], question_context: str | None, text: str) -> str:
    header = f"Expert: {meta.get('expert_name', '')} | Market: {meta.get('market', '')} | Role: {meta.get('role', '')}"
    q = f"\nQuestion: {question_context}" if question_context else ""
    return f"{header}{q}\nAnswer: {text}"


class Indexer:
    def __init__(self, db: Database, embedder: EmbeddingService, project_id: str,
                 artifact_tokens: tuple[str, ...] = ("canvas",)):
        self.db, self.embedder, self.project_id = db, embedder, project_id
        self.artifact_tokens = artifact_tokens

    # ------------------------------------------------------------------------------------------
    def sync(self, folder: Path, force: bool = False) -> IngestReport:
        report = IngestReport()
        folder.mkdir(parents=True, exist_ok=True)
        discovered = discover_transcripts(folder)
        report.total = len(discovered)
        existing = {
            r["id"]: r for r in self.db.query(
                "SELECT id, file_hash, parser_version, status FROM transcripts WHERE project_id = ?",
                (self.project_id,),
            )
        }

        for f in discovered:
            prev = existing.get(f.transcript_id)
            if (not force and prev is not None and prev["file_hash"] == f.file_hash
                    and prev["parser_version"] == PARSER_VERSION):
                report.unchanged.append(f.transcript_id)
                continue
            try:
                self._index_file(f.transcript_id, f.path, f.filename, f.file_hash, report)
                (report.updated if prev is not None else report.added).append(f.transcript_id)
            except TranscriptLoadError as exc:
                self._record_failure(f.transcript_id, f.filename, f.file_hash, str(exc))
                report.failed.append({"transcript_id": f.transcript_id, "filename": f.filename, "error": str(exc)})

        discovered_ids = {f.transcript_id for f in discovered}
        removed = [tid for tid in existing if tid not in discovered_ids]
        if removed:
            with self.db.tx() as conn:
                conn.executemany("DELETE FROM evidence WHERE transcript_id = ?", [(t,) for t in removed])
                conn.executemany("DELETE FROM transcripts WHERE id = ?", [(t,) for t in removed])
            report.removed.extend(removed)

        self.ensure_embeddings(report)
        return report

    # ------------------------------------------------------------------------------------------
    def _index_file(self, tid: str, path: Path, filename: str, fhash: str, report: IngestReport) -> None:
        raw = load_text(path)
        if not raw.strip():
            raise TranscriptLoadError("File is empty")
        parsed = parse_transcript(raw, self.artifact_tokens)
        if not parsed.segments:
            raise TranscriptLoadError("No transcript content could be parsed")
        for w in parsed.warnings:
            report.warnings.append(f"{filename}: {w}")

        prev = self.db.query_one("SELECT created_at, metadata_overrides FROM transcripts WHERE id = ?", (tid,))
        overrides = json.loads(prev["metadata_overrides"]) if prev and prev["metadata_overrides"] else {}
        meta = {**parsed.metadata, **{k: v for k, v in overrides.items() if v}}

        ids = evidence_ids(tid, parsed)
        vectors, model = self._embed_segments(meta, parsed, report, filename)
        now = _now()
        with self.db.tx() as conn:
            conn.execute("DELETE FROM evidence WHERE transcript_id = ?", (tid,))
            conn.execute(
                """INSERT INTO transcripts(id, project_id, filename, file_hash, parser_version, expert_name, role,
                       market, metadata_overrides, raw_text, status, warnings, created_at, updated_at, indexed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'indexed', ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET filename=excluded.filename, file_hash=excluded.file_hash,
                       parser_version=excluded.parser_version, expert_name=excluded.expert_name,
                       role=excluded.role, market=excluded.market, raw_text=excluded.raw_text,
                       status='indexed', warnings=excluded.warnings, updated_at=excluded.updated_at,
                       indexed_at=excluded.indexed_at""",
                (tid, self.project_id, filename, fhash, PARSER_VERSION, parsed.metadata["expert_name"],
                 parsed.metadata["role"], parsed.metadata["market"], json.dumps(overrides) if overrides else None,
                 raw, json.dumps(parsed.warnings), prev["created_at"] if prev else now, now, now),
            )
            conn.executemany(
                """INSERT INTO evidence(id, project_id, transcript_id, seq, speaker, speaker_label, speaker_type,
                       timestamp, timestamp_seconds, text, question_context, embedding, embedding_model)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (eid, self.project_id, tid, s.seq, s.speaker, s.speaker_label, s.speaker_type, s.timestamp,
                     s.timestamp_seconds, s.text, s.question_context,
                     vectors[i].tobytes() if vectors is not None and vectors[i] is not None else None,
                     model if vectors is not None and vectors[i] is not None else None)
                    for i, (eid, s) in enumerate(zip(ids, parsed.segments))
                ],
            )

    def _embed_segments(self, meta, parsed: ParsedTranscript, report: IngestReport, filename: str):
        idx = [i for i, s in enumerate(parsed.segments) if s.speaker_type == "expert"]
        if not idx:
            return None, None
        texts = [embedding_text(meta, parsed.segments[i].question_context, parsed.segments[i].text) for i in idx]
        try:
            mat = self.embedder.embed(texts, "document")
        except EmbeddingError as exc:
            report.warnings.append(f"{filename}: embeddings unavailable ({exc}); keyword retrieval only.")
            return None, None
        vectors = [None] * len(parsed.segments)
        for j, i in enumerate(idx):
            vectors[i] = mat[j]
        return vectors, self.embedder.model_id

    def _record_failure(self, tid: str, filename: str, fhash: str, error: str) -> None:
        now = _now()
        with self.db.tx() as conn:
            conn.execute("DELETE FROM evidence WHERE transcript_id = ?", (tid,))
            conn.execute(
                """INSERT INTO transcripts(id, project_id, filename, file_hash, parser_version, expert_name, role,
                       market, raw_text, status, warnings, created_at, updated_at, indexed_at)
                   VALUES (?, ?, ?, ?, ?, 'Unknown', 'Unknown', 'Unknown', '', 'failed', ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET file_hash=excluded.file_hash, status='failed',
                       parser_version=excluded.parser_version, warnings=excluded.warnings,
                       updated_at=excluded.updated_at, indexed_at=excluded.indexed_at""",
                (tid, self.project_id, filename, fhash, PARSER_VERSION, json.dumps([error]), now, now, now),
            )

    # ------------------------------------------------------------------------------------------
    def ensure_embeddings(self, report: IngestReport | None = None, transcript_id: str | None = None) -> int:
        """(Re-)embed expert evidence whose vectors are missing or were produced by another model."""
        sql = """SELECT e.id, e.text, e.question_context, t.expert_name, t.role, t.market, t.metadata_overrides
                 FROM evidence e JOIN transcripts t ON t.id = e.transcript_id
                 WHERE e.project_id = ? AND e.speaker_type = 'expert'
                   AND (e.embedding IS NULL OR e.embedding_model IS NULL OR e.embedding_model != ?)"""
        params: list = [self.project_id, self.embedder.model_id]
        if transcript_id:
            sql = sql.replace("WHERE e.project_id = ?", "WHERE e.project_id = ? AND e.transcript_id = ?")
            params.insert(1, transcript_id)
        rows = self.db.query(sql, params)
        if not rows:
            return 0
        from ..services.store import effective_metadata

        texts = [embedding_text(effective_metadata(r), r["question_context"], r["text"]) for r in rows]
        try:
            mat = self.embedder.embed(texts, "document")
        except EmbeddingError as exc:
            if report is not None:
                report.warnings.append(f"Embeddings unavailable ({exc}); keyword retrieval only.")
            return 0
        with self.db.tx() as conn:
            conn.executemany(
                "UPDATE evidence SET embedding = ?, embedding_model = ? WHERE id = ?",
                [(mat[i].tobytes(), self.embedder.model_id, r["id"]) for i, r in enumerate(rows)],
            )
        return len(rows)

    def update_metadata(self, transcript_id: str, updates: dict[str, str | None]) -> bool:
        row = self.db.query_one("SELECT metadata_overrides FROM transcripts WHERE id = ? AND project_id = ?",
                                (transcript_id, self.project_id))
        if row is None:
            return False
        overrides = json.loads(row["metadata_overrides"]) if row["metadata_overrides"] else {}
        for k, v in updates.items():
            if v is None:
                continue
            if v.strip():
                overrides[k] = v.strip()
            else:
                overrides.pop(k, None)
        with self.db.tx() as conn:
            conn.execute("UPDATE transcripts SET metadata_overrides = ?, updated_at = ? WHERE id = ?",
                         (json.dumps(overrides) if overrides else None, _now(), transcript_id))
            # metadata is part of the embedded text → force re-embedding for this transcript
            conn.execute("UPDATE evidence SET embedding_model = NULL WHERE transcript_id = ?", (transcript_id,))
        self.ensure_embeddings(transcript_id=transcript_id)
        return True
