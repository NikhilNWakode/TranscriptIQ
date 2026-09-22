"use client";

import * as React from "react";
import { CheckCircle2, Circle, ListChecks, Loader2, PlayCircle } from "lucide-react";
import { useApi, useData } from "@/components/data-context";
import {
  EmptyTranscripts, ErrorNote, ExpertAnswerCard, LoadingCards, MetaFooter, PageHeader, QualifierWarnings,
} from "@/components/research";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { api, ApiError } from "@/lib/api";
import type { GuideQuestionResult } from "@/lib/types";
import { cn } from "@/lib/utils";

type Status = "idle" | "loading" | "done" | "error";

export default function GuidePage() {
  const { version, stats } = useData();
  const guide = useApi(() => api.guide());
  const questions = guide.data?.questions ?? [];
  const [selected, setSelected] = React.useState<string | null>(null);
  const [results, setResults] = React.useState<Record<string, GuideQuestionResult>>({});
  const [status, setStatus] = React.useState<Record<string, Status>>({});
  const [errors, setErrors] = React.useState<Record<string, string>>({});
  const [market, setMarket] = React.useState<string>("all");
  const inflight = React.useRef<Set<string>>(new Set());

  React.useEffect(() => {
    // index changed → previous answers are stale
    setResults({});
    setStatus({});
    setErrors({});
    inflight.current.clear();
  }, [version]);

  React.useEffect(() => {
    if (!selected && questions.length) setSelected(questions[0].id);
  }, [questions, selected]);

  const load = React.useCallback(async (qid: string) => {
    if (inflight.current.has(qid)) return;
    inflight.current.add(qid);
    setStatus((s) => ({ ...s, [qid]: "loading" }));
    try {
      const [res] = await api.guideAnswer(qid);
      setResults((r) => ({ ...r, [qid]: res }));
      setStatus((s) => ({ ...s, [qid]: "done" }));
    } catch (e) {
      setErrors((er) => ({ ...er, [qid]: e instanceof ApiError ? e.message : "Failed to answer question" }));
      setStatus((s) => ({ ...s, [qid]: "error" }));
    } finally {
      inflight.current.delete(qid);
    }
  }, []);

  React.useEffect(() => {
    if (selected && !status[selected] && (stats?.transcripts ?? 0) > 0) load(selected);
  }, [selected, status, load, stats?.transcripts]);

  const loadAll = async () => {
    for (const q of questions) if (status[q.id] !== "done") await load(q.id);
  };

  const current = selected ? results[selected] : undefined;
  const currentQ = questions.find((q) => q.id === selected);
  const markets = Array.from(new Set(current?.expert_answers.map((a) => a.market) ?? [])).sort();
  const answers = (current?.expert_answers ?? []).filter((a) => market === "all" || a.market === market);
  const doneCount = questions.filter((q) => status[q.id] === "done").length;
  const n = stats?.experts ?? 0;

  return (
    <div>
      <PageHeader
        title="Interview Guide"
        description={
          guide.data
            ? `${guide.data.objective} ${questions.length} configured questions × ${stats?.transcripts ?? 0} transcripts = ${questions.length * (stats?.transcripts ?? 0)} expert answers.`
            : "Configured in config/interview_guide.json and applied to every discovered transcript."
        }
      >
        <Button variant="outline" onClick={loadAll} disabled={!questions.length || doneCount === questions.length || !n}>
          <PlayCircle /> Answer all questions ({doneCount}/{questions.length})
        </Button>
      </PageHeader>

      {guide.error && <ErrorNote message={guide.error} />}
      {stats && stats.transcripts === 0 ? (
        <EmptyTranscripts />
      ) : (
        <div className="grid gap-6 lg:grid-cols-[300px_1fr]">
          <Card className="h-fit p-2 lg:sticky lg:top-20">
            <p className="flex items-center gap-1.5 px-3 pb-1 pt-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              <ListChecks className="size-3.5" /> {guide.data?.title ?? "Questions"}
            </p>
            {questions.map((q, i) => {
              const st = status[q.id];
              return (
                <button
                  key={q.id}
                  onClick={() => setSelected(q.id)}
                  className={cn(
                    "flex w-full cursor-pointer items-start gap-2.5 rounded-md px-3 py-2.5 text-left text-sm transition-colors",
                    selected === q.id ? "bg-accent text-accent-foreground" : "hover:bg-muted",
                  )}
                >
                  <span className="mt-0.5 shrink-0">
                    {st === "loading" ? (
                      <Loader2 className="size-4 animate-spin text-primary" />
                    ) : st === "done" ? (
                      <CheckCircle2 className="size-4 text-emerald-600" />
                    ) : (
                      <Circle className="size-4 text-muted-foreground/60" />
                    )}
                  </span>
                  <span>
                    <span className="mr-1 font-semibold">Q{i + 1}.</span>
                    {q.text}
                  </span>
                </button>
              );
            })}
          </Card>

          <div className="min-w-0 space-y-4">
            {currentQ && (
              <div className="flex flex-wrap items-center justify-between gap-3">
                <h2 className="text-lg font-semibold tracking-tight">{currentQ.text}</h2>
                {markets.length > 1 && (
                  <div className="flex flex-wrap gap-1">
                    {["all", ...markets].map((m) => (
                      <button
                        key={m}
                        onClick={() => setMarket(m)}
                        className={cn(
                          "cursor-pointer rounded-full border px-2.5 py-0.5 text-xs transition-colors",
                          market === m ? "border-primary bg-primary text-primary-foreground" : "bg-card hover:bg-muted",
                        )}
                      >
                        {m === "all" ? "All markets" : m}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
            {selected && errors[selected] && <ErrorNote message={errors[selected]} />}
            {selected && status[selected] === "loading" && !current && (
              <LoadingCards n={Math.min(Math.max(n, 1), 3)} className="md:grid-cols-2 xl:grid-cols-3" />
            )}
            {current && (
              <>
                <QualifierWarnings warnings={current.qualifier_warnings} />
                <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                  {answers.map((a) => (
                    <ExpertAnswerCard key={a.transcript_id} a={a} mode={current.meta.mode} />
                  ))}
                </div>
                <MetaFooter
                  meta={current.meta}
                  sources={current.expert_answers.reduce((acc, a) => acc + a.evidence.length, 0)}
                />
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
