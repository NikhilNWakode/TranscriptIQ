"""HTTP API. Input is validated by Pydantic; errors are returned as short, safe messages."""

from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException, Query, Request

from ..models.schemas import (AskRequest, AskResponse, Evidence, GuideAnswerRequest, GuideQuestionResult,
                              IngestReport, InsightsResponse, MetadataUpdate, Stats, TranscriptDetail,
                              TranscriptSummary)
from ..services.research import GuideConfigError
from ..services.state import AppState

router = APIRouter(prefix="/api")
_ID_RE = re.compile(r"^[a-z0-9_]{1,160}$")


def state(request: Request) -> AppState:
    return request.app.state.app_state


def _check_id(value: str, what: str) -> str:
    if not _ID_RE.match(value):
        raise HTTPException(status_code=400, detail=f"Invalid {what}")
    return value


@router.get("/health")
def health(request: Request) -> dict:
    s = state(request)
    return {
        "status": "ok",
        "llm_mode": "extractive" if s.llm.is_mock else "llm",
        "llm_provider": s.llm.provider,
        "llm_model": s.llm.model,
        "llm_warning": s.llm_warning,
        "embedding_model": s.embedder.model_id,
        "transcripts_dir": s.settings.transcripts_dir.name,
        "last_sync_at": s.last_sync_at,
    }


@router.get("/stats", response_model=Stats)
def stats(request: Request) -> Stats:
    return state(request).stats()


@router.post("/refresh", response_model=IngestReport)
def refresh(request: Request) -> IngestReport:
    return state(request).refresh(force=False)


@router.post("/ingest", response_model=IngestReport)
def ingest(request: Request, force: bool = Query(default=False)) -> IngestReport:
    return state(request).refresh(force=force)


@router.get("/transcripts", response_model=list[TranscriptSummary])
def list_transcripts(request: Request) -> list[TranscriptSummary]:
    return state(request).store.list_transcripts()


@router.get("/transcripts/{transcript_id}", response_model=TranscriptDetail)
def get_transcript(transcript_id: str, request: Request) -> TranscriptDetail:
    detail = state(request).store.get_transcript(_check_id(transcript_id, "transcript id"))
    if detail is None:
        raise HTTPException(status_code=404, detail="Transcript not found")
    return detail


@router.patch("/transcripts/{transcript_id}/metadata", response_model=TranscriptSummary)
def update_metadata(transcript_id: str, body: MetadataUpdate, request: Request) -> TranscriptSummary:
    s = state(request)
    if not s.indexer.update_metadata(_check_id(transcript_id, "transcript id"), body.model_dump()):
        raise HTTPException(status_code=404, detail="Transcript not found")
    s.retriever.invalidate()
    detail = s.store.get_transcript(transcript_id)
    assert detail is not None
    return TranscriptSummary(**detail.model_dump(exclude={"segments_list"}))


@router.get("/filters")
def filters(request: Request) -> dict:
    experts = state(request).store.experts()
    known = lambda v: v != "Unknown"  # noqa: E731
    return {
        "experts": [{"transcript_id": e.transcript_id, "expert_name": e.expert_name, "market": e.market,
                     "role": e.role} for e in experts],
        "markets": sorted({e.market for e in experts if known(e.market)}),
        "roles": sorted({e.role for e in experts if known(e.role)}),
    }


@router.get("/evidence/{evidence_id}", response_model=Evidence)
def get_evidence(evidence_id: str, request: Request) -> Evidence:
    ev = state(request).store.get_one(_check_id(evidence_id, "evidence id"))
    if ev is None:
        raise HTTPException(status_code=404, detail="Evidence not found")
    return ev


@router.get("/interview-guide")
def interview_guide(request: Request) -> dict:
    try:
        data, questions = state(request).research.guide_questions()
    except GuideConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"title": data.get("title", "Interview guide"), "objective": data.get("objective", ""),
            "questions": [q.model_dump() for q in questions],
            "example_questions": [str(q) for q in data.get("example_questions", [])][:12]}


@router.post("/interview-guide/answer", response_model=list[GuideQuestionResult])
def interview_guide_answer(body: GuideAnswerRequest, request: Request) -> list[GuideQuestionResult]:
    s = state(request)
    try:
        _, questions = s.research.guide_questions()
    except GuideConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if body.question_id:
        questions = [q for q in questions if q.id == body.question_id]
        if not questions:
            raise HTTPException(status_code=404, detail="Interview-guide question not found")
    return [s.research.answer_guide_question(q) for q in questions]


@router.get("/insights", response_model=InsightsResponse)
@router.get("/themes", response_model=InsightsResponse)
def insights(request: Request) -> InsightsResponse:
    try:
        return state(request).research.insights()
    except GuideConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request) -> AskResponse:
    return state(request).research.ask(body)
