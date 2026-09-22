"""Application container: wires settings, storage, providers and services together."""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone

from ..config import Settings
from ..db import Database
from ..ingestion.indexer import Indexer
from ..llm.providers import build_llm
from ..models.schemas import IngestReport, Stats
from ..retrieval.embeddings import build_embedding_service
from ..retrieval.retriever import QueryExpander, Retriever
from .cache import LLMCache
from .research import ResearchService, load_interview_guide
from .store import EvidenceStore

log = logging.getLogger(__name__)


class AppState:
    def __init__(self, settings: Settings, db: Database | None = None):
        self.settings = settings
        self.db = db or Database(settings.database_path)
        self.embedder = build_embedding_service(settings, self.db)
        self.llm, self.llm_warning = build_llm(settings)
        if self.llm_warning:
            log.warning(self.llm_warning)
        self.store = EvidenceStore(self.db, settings.project_id)
        self.indexer = Indexer(self.db, self.embedder, settings.project_id, settings.artifact_tokens)
        self.retriever = Retriever(self.db, self.embedder, settings.project_id, settings.semantic_weight,
                                   QueryExpander.from_file(settings.query_expansion_path))
        self.cache = LLMCache(self.db)
        self.research = ResearchService(settings, self.store, self.retriever, self.llm, self.cache)
        self._sync_lock = threading.Lock()
        self.last_report: IngestReport | None = None
        self.last_sync_at: str | None = None

    def refresh(self, force: bool = False) -> IngestReport:
        with self._sync_lock:
            report = self.indexer.sync(self.settings.transcripts_dir, force=force)
            self.retriever.invalidate()
            self.last_report = report
            self.last_sync_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            return report

    def stats(self) -> Stats:
        transcripts = self.store.list_transcripts(include_failed=False)
        try:
            n_questions = len(load_interview_guide(self.settings.interview_guide_path)[1])
        except Exception:
            n_questions = 0
        known = lambda v: v and v != "Unknown"  # noqa: E731
        markets = sorted({t.market for t in transcripts if known(t.market)})
        return Stats(
            transcripts=len(transcripts),
            experts=len({t.expert_name for t in transcripts if known(t.expert_name)}),
            markets=len(markets),
            roles=len({t.role for t in transcripts if known(t.role)}),
            interview_questions=n_questions,
            evidence_segments=sum(t.expert_segments for t in transcripts),
            total_segments=sum(t.segments for t in transcripts),
            markets_list=markets,
            last_indexed_at=max((t.indexed_at for t in transcripts), default=None),
            llm_mode="extractive" if self.llm.is_mock else "llm",
            llm_model=f"{self.llm.provider}:{self.llm.model}",
            embedding_model=self.embedder.model_id,
        )
