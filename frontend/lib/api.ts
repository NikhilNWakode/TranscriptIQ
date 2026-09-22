import type {
  AskFilters, AskResponse, Evidence, Filters, GuideQuestionResult, Health, IngestReport, InsightsResponse,
  InterviewGuide, Stats, TranscriptDetail, TranscriptSummary,
} from "./types";

export class ApiError extends Error {
  constructor(message: string, public status: number) {
    super(message);
  }
}

const BACKEND_DOWN = "Backend unavailable. Start the API (see README: uvicorn on port 8000) and retry.";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
      cache: "no-store",
    });
  } catch {
    throw new ApiError(BACKEND_DOWN, 0);
  }
  const text = await res.text();
  let body: unknown = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    // Next's proxy returns an HTML/text error page when the backend is down
    throw new ApiError(res.status >= 500 ? BACKEND_DOWN : `Unexpected response (${res.status})`, res.status);
  }
  if (!res.ok) {
    const detail = (body as { detail?: unknown })?.detail;
    throw new ApiError(typeof detail === "string" ? detail : `Request failed (${res.status})`, res.status);
  }
  return body as T;
}

export const api = {
  health: () => request<Health>("/api/health"),
  stats: () => request<Stats>("/api/stats"),
  refresh: () => request<IngestReport>("/api/refresh", { method: "POST" }),
  transcripts: () => request<TranscriptSummary[]>("/api/transcripts"),
  transcript: (id: string) => request<TranscriptDetail>(`/api/transcripts/${encodeURIComponent(id)}`),
  updateMetadata: (id: string, body: Partial<Pick<TranscriptSummary, "expert_name" | "role" | "market">>) =>
    request<TranscriptSummary>(`/api/transcripts/${encodeURIComponent(id)}/metadata`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  filters: () => request<Filters>("/api/filters"),
  evidence: (id: string) => request<Evidence>(`/api/evidence/${encodeURIComponent(id)}`),
  guide: () => request<InterviewGuide>("/api/interview-guide"),
  guideAnswer: (questionId: string) =>
    request<GuideQuestionResult[]>("/api/interview-guide/answer", {
      method: "POST",
      body: JSON.stringify({ question_id: questionId }),
    }),
  insights: () => request<InsightsResponse>("/api/insights"),
  ask: (question: string, filters: AskFilters) =>
    request<AskResponse>("/api/ask", { method: "POST", body: JSON.stringify({ question, filters }) }),
};

export function sourceHref(ev: Pick<Evidence, "transcript_id" | "id" | "highlight">): string {
  const params = new URLSearchParams({ id: ev.transcript_id, evidence: ev.id });
  if (ev.highlight) params.set("hl", `${ev.highlight.start}-${ev.highlight.end}`);
  return `/transcripts?${params.toString()}`;
}
