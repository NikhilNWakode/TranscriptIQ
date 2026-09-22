"""Pydantic models.

Two families:
* ``LLM*`` models are the *only* things the language model is allowed to produce. They contain
  synthesis text and evidence IDs — never timestamps, speakers or quotes as authoritative data.
* API models are what the backend returns. Every source field on them is resolved from the database.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

NO_EVIDENCE_MESSAGE = "The provided transcripts do not contain sufficient evidence to answer this question."

Classification = Literal["common_view", "difference_in_emphasis", "disagreement"]
# "evidence_map" = demo mode: evidence grouped by topic, deliberately not classified without an LLM.
FindingClass = Literal["common_view", "difference_in_emphasis", "disagreement", "evidence_map"]


# ---------------------------------------------------------------------------------------------
# LLM structured output
# ---------------------------------------------------------------------------------------------
class LLMEvidenceRef(BaseModel):
    evidence_id: str = Field(description="ID of a supplied evidence record, e.g. 'expert_1_02_18'. Never invent IDs.")
    highlight: Optional[str] = Field(
        description="Optional: a short span copied VERBATIM from that evidence record's text that best supports "
        "the claim. Use null if unsure. Never paraphrase here."
    )


class LLMExpertAnswer(BaseModel):
    transcript_id: str = Field(description="transcript_id of the expert, exactly as supplied")
    answer: str = Field(description="1-3 sentence answer for this expert, attributed to them, preserving qualifiers")
    evidence: list[LLMEvidenceRef]


class LLMCrossExpertAnswer(BaseModel):
    insufficient_evidence: bool = Field(description="true if the evidence does not support an answer")
    answer: str = Field(description="Concise grounded synthesis across experts")
    expert_answers: list[LLMExpertAnswer]
    evidence: list[LLMEvidenceRef]


class LLMGuideAnswers(BaseModel):
    expert_answers: list[LLMExpertAnswer]


class LLMFinding(BaseModel):
    title: str
    summary: str = Field(description="Grounded explanation, attributed to experts/markets, preserving qualifiers")
    classification: Classification
    transcript_ids: list[str] = Field(description="transcript_ids of the experts this finding is about")
    evidence: list[LLMEvidenceRef]


class LLMFindings(BaseModel):
    findings: list[LLMFinding]


# ---------------------------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------------------------
class Highlight(BaseModel):
    start: int
    end: int
    text: str


class Evidence(BaseModel):
    id: str
    transcript_id: str
    expert_name: str
    role: str
    market: str
    speaker: str
    speaker_type: str
    timestamp: Optional[str]
    timestamp_seconds: Optional[int]
    text: str
    question_context: Optional[str] = None
    highlight: Optional[Highlight] = None


class ExpertRef(BaseModel):
    transcript_id: str
    expert_name: str
    role: str
    market: str


class ExpertAnswer(ExpertRef):
    answer: str
    supported: bool
    evidence: list[Evidence]


class QualifierWarning(BaseModel):
    claim: str
    qualifier: str
    evidence_id: str
    message: str


class AskFilters(BaseModel):
    transcript_ids: list[str] = Field(default_factory=list, max_length=500)
    markets: list[str] = Field(default_factory=list, max_length=200)
    roles: list[str] = Field(default_factory=list, max_length=200)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    filters: AskFilters = Field(default_factory=AskFilters)

    @field_validator("question")
    @classmethod
    def strip_question(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("Question is too short")
        return v


class AnswerMeta(BaseModel):
    mode: str
    model: str
    cached: bool = False
    retrieved: int = 0
    experts_considered: int = 0
    dropped_citations: list[str] = Field(default_factory=list)
    notices: list[str] = Field(default_factory=list)


class AskResponse(BaseModel):
    question: str
    answer: str
    insufficient_evidence: bool
    expert_answers: list[ExpertAnswer]
    sources: list[Evidence]
    qualifier_warnings: list[QualifierWarning] = Field(default_factory=list)
    meta: AnswerMeta


class GuideQuestion(BaseModel):
    id: str
    text: str


class GuideAnswerRequest(BaseModel):
    question_id: Optional[str] = Field(default=None, max_length=64)
    refresh: bool = False


class GuideQuestionResult(BaseModel):
    question: GuideQuestion
    expert_answers: list[ExpertAnswer]
    qualifier_warnings: list[QualifierWarning] = Field(default_factory=list)
    meta: AnswerMeta


class Finding(BaseModel):
    id: str
    title: str
    summary: str
    classification: FindingClass
    question_id: Optional[str] = None
    experts: list[ExpertRef]
    evidence: list[Evidence]


class InsightsResponse(BaseModel):
    experts_analyzed: int
    markets_represented: int
    transcripts_analyzed: int
    themes: list[Finding]
    differences: list[Finding]
    disagreements: list[Finding]
    by_question: list[Finding]
    meta: AnswerMeta


class TranscriptSummary(BaseModel):
    id: str
    filename: str
    expert_name: str
    role: str
    market: str
    status: str
    segments: int
    expert_segments: int
    warnings: list[str]
    indexed_at: str
    metadata_edited: bool = False


class TranscriptDetail(TranscriptSummary):
    segments_list: list[Evidence]


class MetadataUpdate(BaseModel):
    expert_name: Optional[str] = Field(default=None, max_length=200)
    role: Optional[str] = Field(default=None, max_length=200)
    market: Optional[str] = Field(default=None, max_length=200)


class IngestReport(BaseModel):
    added: list[str] = Field(default_factory=list)
    updated: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    unchanged: list[str] = Field(default_factory=list)
    failed: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    total: int = 0


class Stats(BaseModel):
    transcripts: int
    experts: int
    markets: int
    roles: int
    interview_questions: int
    evidence_segments: int
    total_segments: int
    markets_list: list[str]
    last_indexed_at: Optional[str]
    llm_mode: str
    llm_model: str
    embedding_model: str
