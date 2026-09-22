"use client";

import Link from "next/link";
import {
  ArrowRight, BookOpenText, Boxes, FileSearch, FileText, Globe2, ListChecks, MessageSquareText, Quote, ShieldCheck, Users,
} from "lucide-react";
import { useApi, useData } from "@/components/data-context";
import { EmptyTranscripts, ErrorNote, LoadingCards, PageHeader } from "@/components/research";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/misc";
import { api } from "@/lib/api";
import { formatDate } from "@/lib/utils";

const PIPELINE = [
  { title: "Transcript folder", body: "/transcripts scanned; SHA-256 per file" },
  { title: "Parser", body: "metadata, speakers, timestamps, Q→A context" },
  { title: "Evidence store", body: "one timestamped expert turn = one record" },
  { title: "Embeddings", body: "cached per text; hybrid with BM25" },
  { title: "Retriever", body: "balanced across experts, metadata filters" },
  { title: "LLM", body: "reasons over evidence, returns evidence IDs" },
  { title: "Validation", body: "IDs resolved in DB; quotes & timestamps from source" },
  { title: "UI", body: "synthesis clearly separated from verbatim evidence" },
];

function Metric({ label, value, icon: Icon }: { label: string; value: number | undefined; icon: React.ElementType }) {
  return (
    <Card className="p-4">
      <div className="flex items-center justify-between text-muted-foreground">
        <p className="text-xs font-medium">{label}</p>
        <Icon className="size-4" />
      </div>
      {value === undefined ? (
        <Skeleton className="mt-2 h-8 w-12" />
      ) : (
        <p className="mt-1.5 text-3xl font-semibold tabular-nums tracking-tight">{value}</p>
      )}
    </Card>
  );
}

export default function OverviewPage() {
  const { stats, health } = useData();
  const transcripts = useApi(() => api.transcripts());
  const indexed = transcripts.data?.filter((t) => t.status === "indexed") ?? [];
  const failed = transcripts.data?.filter((t) => t.status === "failed") ?? [];

  return (
    <div className="space-y-6">
      <PageHeader
        title="Research overview"
        description="Traceable insights from expert interview transcripts. Every answer is grounded in timestamped, verbatim evidence resolved by the backend — the model never authors a quote or a timestamp."
      />

      <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
        <Metric label="Transcripts analyzed" value={stats?.transcripts} icon={FileText} />
        <Metric label="Experts" value={stats?.experts} icon={Users} />
        <Metric label="Markets" value={stats?.markets} icon={Globe2} />
        <Metric label="Interview questions" value={stats?.interview_questions} icon={ListChecks} />
        <Metric label="Evidence segments" value={stats?.evidence_segments} icon={Quote} />
      </div>

      {transcripts.error && <ErrorNote message={transcripts.error} />}
      {transcripts.loading && !transcripts.data ? (
        <LoadingCards n={1} />
      ) : indexed.length === 0 && !transcripts.error ? (
        <EmptyTranscripts />
      ) : (
        <Card>
          <CardHeader className="flex-row items-center justify-between">
            <div>
              <CardTitle>Experts in this project</CardTitle>
              <CardDescription>Discovered automatically from /transcripts — add a file and refresh to extend.</CardDescription>
            </div>
            <Button asChild variant="outline" size="sm">
              <Link href="/transcripts">
                Browse transcripts <ArrowRight />
              </Link>
            </Button>
          </CardHeader>
          <CardContent className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-left text-xs text-muted-foreground">
                  <th className="py-2 pr-4 font-medium">Expert</th>
                  <th className="py-2 pr-4 font-medium">Role</th>
                  <th className="py-2 pr-4 font-medium">Market</th>
                  <th className="py-2 pr-4 text-right font-medium">Evidence</th>
                  <th className="py-2 pr-4 font-medium">Source file</th>
                  <th className="py-2 font-medium">Parser notes</th>
                </tr>
              </thead>
              <tbody>
                {indexed.map((t) => (
                  <tr key={t.id} className="border-b last:border-0 hover:bg-muted/50">
                    <td className="py-2.5 pr-4 font-medium">
                      <Link href={`/transcripts?id=${t.id}`} className="hover:underline">
                        {t.expert_name}
                      </Link>
                    </td>
                    <td className="py-2.5 pr-4 text-muted-foreground">{t.role}</td>
                    <td className="py-2.5 pr-4">
                      <Badge variant="outline">{t.market}</Badge>
                    </td>
                    <td className="py-2.5 pr-4 text-right tabular-nums">{t.expert_segments}</td>
                    <td className="py-2.5 pr-4 font-mono text-xs text-muted-foreground">{t.filename}</td>
                    <td className="py-2.5 text-xs text-muted-foreground">
                      {t.warnings.length ? t.warnings.join(" · ") : <span className="text-emerald-700">Clean parse</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {failed.length > 0 && (
              <div className="mt-3 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
                Skipped {failed.length} file(s): {failed.map((f) => `${f.filename} (${f.warnings[0] ?? "unreadable"})`).join(", ")}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Evidence-first pipeline</CardTitle>
            <CardDescription>The LLM reasons over evidence. The backend owns source truth.</CardDescription>
          </CardHeader>
          <CardContent>
            <ol className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
              {PIPELINE.map((step, i) => (
                <li key={step.title} className="relative rounded-lg border bg-muted/40 px-3 py-2.5">
                  <p className="text-[11px] font-semibold text-primary">Step {i + 1}</p>
                  <p className="text-sm font-medium">{step.title}</p>
                  <p className="mt-0.5 text-xs text-muted-foreground">{step.body}</p>
                </li>
              ))}
            </ol>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>System</CardTitle>
            <CardDescription>Configured through .env — no code changes.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2.5 text-sm">
            <Row label="Answer mode" value={health ? (health.llm_mode === "llm" ? "LLM synthesis" : "Extractive (demo)") : "…"} />
            <Row label="LLM" value={health ? `${health.llm_provider}:${health.llm_model}` : "…"} />
            <Row label="Embeddings" value={health?.embedding_model ?? "…"} />
            <Row label="Markets" value={stats?.markets_list.join(", ") || "—"} />
            <Row label="Last indexed" value={formatDate(stats?.last_indexed_at)} />
            <div className="flex items-center gap-1.5 pt-1 text-xs text-emerald-700">
              <ShieldCheck className="size-3.5" /> Citations validated server-side on every answer
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-3 md:grid-cols-3">
        {[
          { href: "/guide", icon: BookOpenText, title: "Interview Guide", body: "Every configured question answered for every expert, with timestamped quotes." },
          { href: "/insights", icon: Boxes, title: "Cross-Expert Insights", body: "Themes, differences in emphasis and genuine disagreements — each backed by evidence." },
          { href: "/ask", icon: MessageSquareText, title: "Ask AI", body: "Ask anything across all transcripts; unsupported questions get an explicit no-evidence answer." },
        ].map(({ href, icon: Icon, title, body }) => (
          <Link key={href} href={href}>
            <Card className="h-full p-5 transition-colors hover:border-primary/40 hover:bg-accent/30">
              <Icon className="size-5 text-primary" />
              <p className="mt-3 text-sm font-semibold">{title}</p>
              <p className="mt-1 text-sm text-muted-foreground">{body}</p>
            </Card>
          </Link>
        ))}
      </div>
      <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <FileSearch className="size-3.5" /> Tip: every quote in the app is clickable and opens the transcript at the exact timestamp.
      </p>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-4">
      <span className="text-muted-foreground">{label}</span>
      <span className="truncate text-right font-medium" title={value}>
        {value}
      </span>
    </div>
  );
}
