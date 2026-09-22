"""Read-side repository. This is the single place source data (quotes, timestamps, experts) comes from."""

from __future__ import annotations

import json
import sqlite3

from ..db import Database
from ..models.schemas import Evidence, ExpertRef, TranscriptDetail, TranscriptSummary

_EVIDENCE_SQL = """
SELECT e.id, e.transcript_id, e.speaker, e.speaker_type, e.timestamp, e.timestamp_seconds,
       e.text, e.question_context, e.seq,
       t.expert_name, t.role, t.market, t.metadata_overrides
FROM evidence e JOIN transcripts t ON t.id = e.transcript_id
WHERE e.project_id = ? AND t.project_id = ? AND t.status = 'indexed'
"""


def effective_metadata(row: sqlite3.Row) -> dict[str, str]:
    meta = {"expert_name": row["expert_name"], "role": row["role"], "market": row["market"]}
    if row["metadata_overrides"]:
        meta.update({k: v for k, v in json.loads(row["metadata_overrides"]).items() if v})
    return meta


def _evidence_from_row(row: sqlite3.Row) -> Evidence:
    meta = effective_metadata(row)
    speaker = meta["expert_name"] if row["speaker_type"] == "expert" else row["speaker"]
    return Evidence(
        id=row["id"],
        transcript_id=row["transcript_id"],
        expert_name=meta["expert_name"],
        role=meta["role"],
        market=meta["market"],
        speaker=speaker,
        speaker_type=row["speaker_type"],
        timestamp=row["timestamp"],
        timestamp_seconds=row["timestamp_seconds"],
        text=row["text"],
        question_context=row["question_context"],
    )


class EvidenceStore:
    def __init__(self, db: Database, project_id: str):
        self.db, self.project_id = db, project_id

    # ---- evidence -------------------------------------------------------------------------
    def get_evidence(self, ids: list[str]) -> dict[str, Evidence]:
        if not ids:
            return {}
        out: dict[str, Evidence] = {}
        unique = list(dict.fromkeys(ids))
        for start in range(0, len(unique), 500):
            chunk = unique[start:start + 500]
            rows = self.db.query(
                _EVIDENCE_SQL + f" AND e.id IN ({','.join('?' * len(chunk))})",
                (self.project_id, self.project_id, *chunk),
            )
            out.update({r["id"]: _evidence_from_row(r) for r in rows})
        return out

    def get_one(self, evidence_id: str) -> Evidence | None:
        return self.get_evidence([evidence_id]).get(evidence_id)

    def transcript_segments(self, transcript_id: str) -> list[Evidence]:
        rows = self.db.query(_EVIDENCE_SQL + " AND e.transcript_id = ? ORDER BY e.seq",
                             (self.project_id, self.project_id, transcript_id))
        return [_evidence_from_row(r) for r in rows]

    # ---- transcripts ----------------------------------------------------------------------
    def _transcript_rows(self, transcript_id: str | None = None) -> list[sqlite3.Row]:
        sql = """
        SELECT t.*, (SELECT COUNT(*) FROM evidence e WHERE e.transcript_id = t.id) AS n_seg,
               (SELECT COUNT(*) FROM evidence e WHERE e.transcript_id = t.id AND e.speaker_type = 'expert') AS n_exp
        FROM transcripts t WHERE t.project_id = ?
        """
        params: list = [self.project_id]
        if transcript_id:
            sql += " AND t.id = ?"
            params.append(transcript_id)
        rows = self.db.query(sql + " ORDER BY t.filename", params)
        import re

        def natural(r):
            return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", r["filename"].lower())]

        return sorted(rows, key=natural)

    def _summary(self, row: sqlite3.Row) -> TranscriptSummary:
        meta = effective_metadata(row)
        return TranscriptSummary(
            id=row["id"], filename=row["filename"], status=row["status"],
            segments=row["n_seg"], expert_segments=row["n_exp"],
            warnings=json.loads(row["warnings"] or "[]"), indexed_at=row["indexed_at"],
            metadata_edited=bool(row["metadata_overrides"] and json.loads(row["metadata_overrides"])),
            **meta,
        )

    def list_transcripts(self, include_failed: bool = True) -> list[TranscriptSummary]:
        rows = self._transcript_rows()
        return [self._summary(r) for r in rows if include_failed or r["status"] == "indexed"]

    def get_transcript(self, transcript_id: str) -> TranscriptDetail | None:
        rows = self._transcript_rows(transcript_id)
        if not rows:
            return None
        summary = self._summary(rows[0])
        return TranscriptDetail(**summary.model_dump(), segments_list=self.transcript_segments(transcript_id))

    def experts(self) -> list[ExpertRef]:
        return [
            ExpertRef(transcript_id=t.id, expert_name=t.expert_name, role=t.role, market=t.market)
            for t in self.list_transcripts(include_failed=False)
            if t.expert_segments > 0
        ]

    def transcript_exists(self, transcript_id: str) -> bool:
        return self.db.query_one(
            "SELECT 1 FROM transcripts WHERE id = ? AND project_id = ? AND status = 'indexed'",
            (transcript_id, self.project_id),
        ) is not None
