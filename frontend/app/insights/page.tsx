"use client";

import * as React from "react";
import { GitCompareArrows, Layers, Scale, Swords } from "lucide-react";
import { useApi, useData } from "@/components/data-context";
import {
  ClassificationBadge, EmptyTranscripts, ErrorNote, EvidenceList, LoadingCards, MarketBadge, MetaFooter, PageHeader,
  SectionLabel, SynthesisLabel,
} from "@/components/research";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/misc";
import { api } from "@/lib/api";
import type { Finding, InsightsResponse } from "@/lib/types";

function FindingCard({ f, mode }: { f: Finding; mode: InsightsResponse["meta"]["mode"] }) {
  const [open, setOpen] = React.useState(false);
  const shown = open ? f.evidence : f.evidence.slice(0, 3);
  return (
    <Card className="flex flex-col">
      <div className="space-y-3 px-5 py-4">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <h3 className="text-[15px] font-semibold leading-snug tracking-tight">{f.title}</h3>
          <ClassificationBadge value={f.classification} />
        </div>
        <div
          className={
            mode === "llm"
              ? "space-y-1.5 rounded-lg border border-ai-border/60 bg-ai/60 px-3.5 py-2.5"
              : "space-y-1.5 rounded-lg border bg-muted/50 px-3.5 py-2.5"
          }
        >
          {mode === "llm" ? <SectionLabel kind="ai">AI Insight</SectionLabel> : <SynthesisLabel mode="extractive" />}
          <p className="text-sm leading-relaxed">{f.summary}</p>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {f.experts.map((e) => (
            <span key={e.transcript_id} className="flex items-center gap-1 text-xs text-muted-foreground">
              <span className="font-medium text-foreground">{e.expert_name}</span>
              <MarketBadge market={e.market} />
            </span>
          ))}
        </div>
        <div className="space-y-1.5">
          <SectionLabel kind="source">
            Source evidence · {f.evidence.map((e) => `${e.market} ${e.timestamp ?? ""}`.trim()).join(" · ")}
          </SectionLabel>
          <EvidenceList evidence={shown} compact />
          {f.evidence.length > 3 && (
            <button onClick={() => setOpen(!open)} className="cursor-pointer text-xs font-medium text-primary hover:underline">
              {open ? "Show less" : `Show all ${f.evidence.length} quotes`}
            </button>
          )}
        </div>
      </div>
    </Card>
  );
}

function FindingGrid({ items, mode, empty }: { items: Finding[]; mode: InsightsResponse["meta"]["mode"]; empty: string }) {
  if (!items.length) return <Card className="px-5 py-8 text-center text-sm text-muted-foreground">{empty}</Card>;
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      {items.map((f) => (
        <FindingCard key={f.id} f={f} mode={mode} />
      ))}
    </div>
  );
}

export default function InsightsPage() {
  const { stats } = useData();
  const { data, error, loading } = useApi(() => api.insights());
  const mode = data?.meta.mode ?? "extractive";
  const common = data?.by_question.filter((f) => f.classification === "common_view") ?? [];

  return (
    <div>
      <PageHeader
        title="Cross-Expert Insights"
        description="Themes and comparisons across every indexed transcript. A finding is only kept if at least one cited evidence ID resolves to a real transcript statement. Differences in emphasis are kept distinct from genuine disagreements."
      />
      {stats && stats.transcripts === 0 ? (
        <EmptyTranscripts />
      ) : (
        <>
          <div className="mb-6 grid grid-cols-3 gap-3 md:max-w-xl">
            {[
              ["Experts analyzed", data?.experts_analyzed],
              ["Markets represented", data?.markets_represented],
              ["Transcripts analyzed", data?.transcripts_analyzed],
            ].map(([label, v]) => (
              <Card key={label as string} className="px-4 py-3">
                <p className="text-xs text-muted-foreground">{label}</p>
                <p className="text-2xl font-semibold tabular-nums">{v ?? "–"}</p>
              </Card>
            ))}
          </div>
          {error && <ErrorNote message={error} />}
          {loading && !data && (
            <>
              <p className="mb-3 text-sm text-muted-foreground">Analysing evidence across experts… (LLM runs are cached after the first time)</p>
              <LoadingCards n={4} className="lg:grid-cols-2" />
            </>
          )}
          {data && mode === "extractive" && (
            <div className="mb-4 rounded-lg border bg-muted/50 px-4 py-3 text-sm text-muted-foreground">
              Extractive demo mode shows an <span className="font-medium text-foreground">evidence map</span> per interview-guide
              topic. Theme discovery and classification (common view / difference in emphasis / disagreement) require an LLM
              provider — set <code className="rounded bg-card px-1">LLM_PROVIDER</code> in <code className="rounded bg-card px-1">.env</code>.
            </div>
          )}
          {data && (
            <Tabs defaultValue={mode === "llm" ? "themes" : "topics"}>
              <TabsList>
                {mode === "llm" && (
                  <>
                    <TabsTrigger value="themes">
                      <Layers className="size-3.5" /> Themes ({data.themes.length})
                    </TabsTrigger>
                    <TabsTrigger value="differences">
                      <Scale className="size-3.5" /> Differences in emphasis ({data.differences.length})
                    </TabsTrigger>
                    <TabsTrigger value="disagreements">
                      <Swords className="size-3.5" /> Disagreements ({data.disagreements.length})
                    </TabsTrigger>
                  </>
                )}
                <TabsTrigger value="topics">
                  <GitCompareArrows className="size-3.5" /> By interview question ({data.by_question.length})
                </TabsTrigger>
              </TabsList>
              <TabsContent value="themes">
                <FindingGrid items={data.themes} mode={mode} empty="No cross-cutting themes were supported by the evidence." />
                {common.length > 0 && (
                  <p className="mt-3 text-xs text-muted-foreground">
                    Plus {common.length} topic-level common views under “By interview question”.
                  </p>
                )}
              </TabsContent>
              <TabsContent value="differences">
                <FindingGrid items={data.differences} mode={mode} empty="No differences in emphasis identified." />
              </TabsContent>
              <TabsContent value="disagreements">
                <FindingGrid
                  items={data.disagreements}
                  mode={mode}
                  empty="No direct contradictions found. Differences between experts concern emphasis, magnitude or scope — see “Differences in emphasis”."
                />
              </TabsContent>
              <TabsContent value="topics">
                <FindingGrid items={data.by_question} mode={mode} empty="No topic-level findings." />
              </TabsContent>
            </Tabs>
          )}
          {data && (
            <div className="mt-5">
              <MetaFooter
                meta={data.meta}
                sources={new Set([...data.themes, ...data.by_question].flatMap((f) => f.evidence.map((e) => e.id))).size}
              />
            </div>
          )}
        </>
      )}
    </div>
  );
}
