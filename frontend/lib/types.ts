// Mirrors backend/app/models/schemas.py

export interface Highlight {
  start: number;
  end: number;
  text: string;
}

export interface Evidence {
  id: string;
  transcript_id: string;
  expert_name: string;
  role: string;
  market: string;
  speaker: string;
  speaker_type: "expert" | "interviewer" | "other";
  timestamp: string | null;
  timestamp_seconds: number | null;
  text: string;
  question_context?: string | null;
  highlight?: Highlight | null;
}

export interface ExpertRef {
  transcript_id: string;
  expert_name: string;
  role: string;
  market: string;
}

export interface ExpertAnswer extends ExpertRef {
  answer: string;
  supported: boolean;
  evidence: Evidence[];
}

export interface QualifierWarning {
  claim: string;
  qualifier: string;
  evidence_id: string;
  message: string;
}

export interface AnswerMeta {
  mode: "llm" | "extractive";
  model: string;
  cached: boolean;
  retrieved: number;
  experts_considered: number;
  dropped_citations: string[];
  notices: string[];
}

export interface AskFilters {
  transcript_ids: string[];
  markets: string[];
  roles: string[];
}

export interface AskResponse {
  question: string;
  answer: string;
  insufficient_evidence: boolean;
  expert_answers: ExpertAnswer[];
  sources: Evidence[];
  qualifier_warnings: QualifierWarning[];
  meta: AnswerMeta;
}

export interface GuideQuestion {
  id: string;
  text: string;
}

export interface InterviewGuide {
  title: string;
  objective: string;
  questions: GuideQuestion[];
  example_questions: string[];
}

export interface GuideQuestionResult {
  question: GuideQuestion;
  expert_answers: ExpertAnswer[];
  qualifier_warnings: QualifierWarning[];
  meta: AnswerMeta;
}

export type FindingClass = "common_view" | "difference_in_emphasis" | "disagreement" | "evidence_map";

export interface Finding {
  id: string;
  title: string;
  summary: string;
  classification: FindingClass;
  question_id: string | null;
  experts: ExpertRef[];
  evidence: Evidence[];
}

export interface InsightsResponse {
  experts_analyzed: number;
  markets_represented: number;
  transcripts_analyzed: number;
  themes: Finding[];
  differences: Finding[];
  disagreements: Finding[];
  by_question: Finding[];
  meta: AnswerMeta;
}

export interface TranscriptSummary {
  id: string;
  filename: string;
  expert_name: string;
  role: string;
  market: string;
  status: "indexed" | "failed";
  segments: number;
  expert_segments: number;
  warnings: string[];
  indexed_at: string;
  metadata_edited: boolean;
}

export interface TranscriptDetail extends TranscriptSummary {
  segments_list: Evidence[];
}

export interface IngestReport {
  added: string[];
  updated: string[];
  removed: string[];
  unchanged: string[];
  failed: { transcript_id: string; filename: string; error: string }[];
  warnings: string[];
  total: number;
}

export interface Stats {
  transcripts: number;
  experts: number;
  markets: number;
  roles: number;
  interview_questions: number;
  evidence_segments: number;
  total_segments: number;
  markets_list: string[];
  last_indexed_at: string | null;
  llm_mode: "llm" | "extractive";
  llm_model: string;
  embedding_model: string;
}

export interface Health {
  status: string;
  llm_mode: "llm" | "extractive";
  llm_provider: string;
  llm_model: string;
  llm_warning: string | null;
  embedding_model: string;
  transcripts_dir: string;
  last_sync_at: string | null;
}

export interface Filters {
  experts: ExpertRef[];
  markets: string[];
  roles: string[];
}
