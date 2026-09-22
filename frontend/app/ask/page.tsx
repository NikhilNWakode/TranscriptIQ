"use client";

import * as React from "react";
import { ArrowUp, Filter, Loader2, SearchX, Trash2 } from "lucide-react";
import { useApi, useData } from "@/components/data-context";
import {
  EmptyTranscripts, ErrorNote, EvidenceList, ExpertAnswerCard, MetaFooter, PageHeader, QualifierWarnings, SectionLabel,
  SynthesisLabel,
} from "@/components/research";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/input";
import { api, ApiError } from "@/lib/api";
import type { AskFilters, AskResponse } from "@/lib/types";
import { cn } from "@/lib/utils";

interface Turn {
  id: number;
  question: string;
  response?: AskResponse;
  error?: string;
}

function Chip({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "cursor-pointer rounded-full border px-2.5 py-0.5 text-xs transition-colors",
        active ? "border-primary bg-primary text-primary-foreground" : "bg-card hover:bg-muted",
      )}
    >
      {children}
    </button>
  );
}

function AnswerView({ r }: { r: AskResponse }) {
  if (r.insufficient_evidence) {
    return (
      <Card className="flex items-start gap-3 border-dashed px-5 py-4">
        <SearchX className="mt-0.5 size-5 text-muted-foreground" />
        <div>
          <p className="text-sm font-medium">{r.answer}</p>
          <p className="mt-1 text-xs text-muted-foreground">
            The assistant does not use outside knowledge. Try rephrasing, or check which topics the transcripts cover.
          </p>
        </div>
      </Card>
    );
  }
  return (
    <div className="space-y-4">
      <Card className="border-ai-border bg-ai/50 px-5 py-4">
        <SynthesisLabel mode={r.meta.mode} />
        <p className="mt-2 whitespace-pre-line text-[15px] leading-relaxed">{r.answer}</p>
      </Card>
      <QualifierWarnings warnings={r.qualifier_warnings} />
      {r.meta.mode === "llm" && r.expert_answers.length > 0 && (
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {r.expert_answers.map((a) => (
            <ExpertAnswerCard key={a.transcript_id} a={a} mode={r.meta.mode} />
          ))}
        </div>
      )}
      <div className="space-y-2">
        <SectionLabel kind="source">Sources · {r.sources.length} verbatim quotes</SectionLabel>
        <EvidenceList evidence={r.sources} />
      </div>
      <MetaFooter meta={r.meta} sources={r.sources.length} />
    </div>
  );
}

export default function AskPage() {
  const { stats } = useData();
  const filtersData = useApi(() => api.filters());
  const guide = useApi(() => api.guide());
  const examples = guide.data?.example_questions ?? [];
  const [question, setQuestion] = React.useState("");
  const [turns, setTurns] = React.useState<Turn[]>([]);
  const [busy, setBusy] = React.useState(false);
  const [filters, setFilters] = React.useState<AskFilters>({ transcript_ids: [], markets: [], roles: [] });
  const [showFilters, setShowFilters] = React.useState(false);
  const bottomRef = React.useRef<HTMLDivElement>(null);

  const toggle = (key: keyof AskFilters, value: string) =>
    setFilters((f) => ({
      ...f,
      [key]: f[key].includes(value) ? f[key].filter((v) => v !== value) : [...f[key], value],
    }));
  const activeFilters = filters.transcript_ids.length + filters.markets.length + filters.roles.length;

  const submit = async (q?: string) => {
    const text = (q ?? question).trim();
    if (text.length < 3 || busy) return;
    const id = Date.now();
    setTurns((t) => [...t, { id, question: text }]);
    setQuestion("");
    setBusy(true);
    setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), 50);
    try {
      const response = await api.ask(text, filters);
      setTurns((t) => t.map((x) => (x.id === id ? { ...x, response } : x)));
    } catch (e) {
      setTurns((t) => t.map((x) => (x.id === id ? { ...x, error: e instanceof ApiError ? e.message : "Request failed" } : x)));
    } finally {
      setBusy(false);
    }
  };

  if (stats && stats.transcripts === 0) {
    return (
      <div>
        <PageHeader title="Ask AI" />
        <EmptyTranscripts />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-5xl">
      <PageHeader
        title="Ask AI"
        description="Ask anything across all indexed transcripts. Answers cite evidence IDs that the backend resolves to exact quotes and timestamps; if the transcripts don’t support an answer, you’ll be told so."
      >
        {turns.length > 0 && (
          <Button variant="ghost" size="sm" onClick={() => setTurns([])}>
            <Trash2 /> Clear
          </Button>
        )}
      </PageHeader>

      <Card className="p-3">
        <div className="flex items-end gap-2">
          <Textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            placeholder="e.g. How do experts view the role of economics in purchasing?"
            className="min-h-12 resize-none border-0 shadow-none focus-visible:ring-0"
            maxLength={1000}
          />
          <Button variant="ghost" size="icon" onClick={() => setShowFilters(!showFilters)} aria-label="Filters" className="relative">
            <Filter />
            {activeFilters > 0 && (
              <span className="absolute -right-0.5 -top-0.5 grid size-4 place-items-center rounded-full bg-primary text-[10px] text-primary-foreground">
                {activeFilters}
              </span>
            )}
          </Button>
          <Button size="icon" onClick={() => submit()} disabled={busy || question.trim().length < 3} aria-label="Ask">
            {busy ? <Loader2 className="animate-spin" /> : <ArrowUp />}
          </Button>
        </div>
        {showFilters && filtersData.data && (
          <div className="mt-3 space-y-2 border-t px-1 pt-3 text-xs">
            <FilterRow label="Expert">
              {filtersData.data.experts.map((e) => (
                <Chip key={e.transcript_id} active={filters.transcript_ids.includes(e.transcript_id)} onClick={() => toggle("transcript_ids", e.transcript_id)}>
                  {e.expert_name}
                </Chip>
              ))}
            </FilterRow>
            <FilterRow label="Market">
              {filtersData.data.markets.map((m) => (
                <Chip key={m} active={filters.markets.includes(m)} onClick={() => toggle("markets", m)}>
                  {m}
                </Chip>
              ))}
            </FilterRow>
            <FilterRow label="Role">
              {filtersData.data.roles.map((r) => (
                <Chip key={r} active={filters.roles.includes(r)} onClick={() => toggle("roles", r)}>
                  {r}
                </Chip>
              ))}
            </FilterRow>
          </div>
        )}
      </Card>

      {turns.length === 0 && examples.length > 0 && (
        <div className="mt-5">
          <p className="mb-2 text-xs font-medium text-muted-foreground">Suggested questions (config/interview_guide.json)</p>
          <div className="flex flex-wrap gap-2">
            {examples.map((q) => (
              <button key={q} onClick={() => submit(q)} className="cursor-pointer rounded-lg border bg-card px-3 py-1.5 text-left text-sm hover:border-primary/40 hover:bg-accent/40">
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="mt-6 space-y-8">
        {turns.map((t) => (
          <section key={t.id} className="space-y-3">
            <div className="flex justify-end">
              <p className="max-w-2xl rounded-2xl rounded-br-sm bg-foreground px-4 py-2 text-sm text-background">{t.question}</p>
            </div>
            {t.error && <ErrorNote message={t.error} />}
            {!t.response && !t.error && (
              <p className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="size-4 animate-spin" /> Retrieving evidence across experts and validating citations…
              </p>
            )}
            {t.response && <AnswerView r={t.response} />}
          </section>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}

function FilterRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <span className="w-14 shrink-0 font-medium text-muted-foreground">{label}</span>
      {children}
    </div>
  );
}
