"use client";

import Link from "next/link";
import {
  AlertTriangle, ArrowUpRight, Database, FolderOpen, Info, Quote, ShieldCheck, Sparkles,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Skeleton, Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/misc";
import { sourceHref } from "@/lib/api";
import type { AnswerMeta, Evidence, ExpertAnswer, FindingClass, QualifierWarning } from "@/lib/types";
import { cn } from "@/lib/utils";

/* ------------------------------------------------------------------ page scaffolding */
export function PageHeader({ title, description, children }: { title: string; description?: string; children?: React.ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
        {description && <p className="mt-1 max-w-3xl text-sm text-muted-foreground">{description}</p>}
      </div>
      {children}
    </div>
  );
}

export function EmptyTranscripts() {
  return (
    <Card className="flex flex-col items-center gap-3 px-6 py-14 text-center">
      <div className="grid size-11 place-items-center rounded-full bg-muted">
        <FolderOpen className="size-5 text-muted-foreground" />
      </div>
      <p className="font-medium">No transcripts found.</p>
      <p className="max-w-md text-sm text-muted-foreground">
        Add transcript files (.txt, .md, .pdf, .docx) to the <code className="rounded bg-muted px-1">/transcripts</code>{" "}
        folder and click <span className="font-medium text-foreground">“Refresh Transcripts”</span>.
      </p>
    </Card>
  );
}

export function ErrorNote({ message }: { message: string }) {
  return (
    <div className="flex items-center gap-2 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
      <AlertTriangle className="size-4 shrink-0" /> {message}
    </div>
  );
}

export function LoadingCards({ n = 3, className }: { n?: number; className?: string }) {
  return (
    <div className={cn("grid gap-4", className)}>
      {Array.from({ length: n }).map((_, i) => (
        <Card key={i} className="space-y-3 p-5">
          <Skeleton className="h-4 w-1/3" />
          <Skeleton className="h-3 w-full" />
          <Skeleton className="h-3 w-5/6" />
          <Skeleton className="h-14 w-full" />
        </Card>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ labels */
export function SectionLabel({ kind, children }: { kind: "ai" | "source"; children?: React.ReactNode }) {
  return kind === "ai" ? (
    <p className="flex items-center gap-1.5 text-[10.5px] font-semibold uppercase tracking-[0.08em] text-ai-foreground">
      <Sparkles className="size-3" /> {children ?? "AI Synthesis"}
    </p>
  ) : (
    <p className="flex items-center gap-1.5 text-[10.5px] font-semibold uppercase tracking-[0.08em] text-source-foreground">
      <Quote className="size-3" /> {children ?? "Source Evidence · verbatim"}
    </p>
  );
}

export function SynthesisLabel({ mode }: { mode: AnswerMeta["mode"] }) {
  return mode === "llm" ? (
    <SectionLabel kind="ai" />
  ) : (
    <p className="flex items-center gap-1.5 text-[10.5px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
      <Database className="size-3" /> Extractive answer · verbatim excerpts
    </p>
  );
}

const CLASS_META: Record<FindingClass, { label: string; variant: "common" | "emphasis" | "disagree" | "muted"; help: string }> = {
  common_view: { label: "Common view", variant: "common", help: "Experts express substantially the same position." },
  difference_in_emphasis: {
    label: "Difference in emphasis",
    variant: "emphasis",
    help: "Same direction, different weight, magnitude or scope — not a contradiction.",
  },
  disagreement: { label: "Direct disagreement", variant: "disagree", help: "Evidence shows genuinely contradictory positions." },
  evidence_map: { label: "Evidence map", variant: "muted", help: "Demo mode: evidence grouped by topic, not classified." },
};

export function ClassificationBadge({ value }: { value: FindingClass }) {
  const m = CLASS_META[value];
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span>
          <Badge variant={m.variant} className="cursor-default">
            {m.label}
          </Badge>
        </span>
      </TooltipTrigger>
      <TooltipContent>{m.help}</TooltipContent>
    </Tooltip>
  );
}

export function MarketBadge({ market }: { market: string }) {
  return <Badge variant="outline">{market}</Badge>;
}

/* ------------------------------------------------------------------ evidence */
function HighlightedText({ ev }: { ev: Evidence }) {
  const h = ev.highlight;
  if (!h || h.start < 0 || h.end > ev.text.length || h.start >= h.end) return <>{ev.text}</>;
  return (
    <>
      {ev.text.slice(0, h.start)}
      <mark className="rounded-sm bg-amber-200/70 px-0.5 text-inherit">{ev.text.slice(h.start, h.end)}</mark>
      {ev.text.slice(h.end)}
    </>
  );
}

export function TimestampChip({ ev }: { ev: Evidence }) {
  return (
    <span className="rounded bg-foreground/[0.06] px-1.5 py-0.5 font-mono text-[11px] font-medium tabular-nums text-foreground/80">
      {ev.timestamp ?? "—"}
    </span>
  );
}

/** One verbatim source quote. Everything shown here was resolved by the backend from the transcript. */
export function EvidenceQuote({ ev, showExpert = true, compact = false }: { ev: Evidence; showExpert?: boolean; compact?: boolean }) {
  return (
    <Link
      href={sourceHref(ev)}
      className="group block rounded-lg border border-source-border/70 bg-source px-3.5 py-2.5 transition-colors hover:border-source-border hover:bg-amber-50"
    >
      <div className="mb-1 flex flex-wrap items-center gap-2 text-xs">
        <TimestampChip ev={ev} />
        {showExpert && (
          <span className="font-medium text-foreground">
            {ev.expert_name} <span className="font-normal text-muted-foreground">— {ev.market}</span>
          </span>
        )}
        <span className="ml-auto flex items-center gap-0.5 text-[11px] text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100">
          View in transcript <ArrowUpRight className="size-3" />
        </span>
      </div>
      <p className={cn("text-[13px] leading-relaxed text-foreground/90", compact && "line-clamp-3")}>
        “<HighlightedText ev={ev} />”
      </p>
    </Link>
  );
}

export function EvidenceList({ evidence, showExpert = true, compact = false }: { evidence: Evidence[]; showExpert?: boolean; compact?: boolean }) {
  if (!evidence.length) return null;
  return (
    <div className="space-y-2">
      {evidence.map((ev) => (
        <EvidenceQuote key={ev.id} ev={ev} showExpert={showExpert} compact={compact} />
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ answers */
export function ExpertAnswerCard({ a, mode }: { a: ExpertAnswer; mode: AnswerMeta["mode"] }) {
  return (
    <Card className={cn("flex flex-col", !a.supported && "bg-muted/40")}>
      <div className="flex items-start justify-between gap-3 border-b px-5 py-3.5">
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold">{a.expert_name}</p>
          <p className="truncate text-xs text-muted-foreground">{a.role}</p>
        </div>
        <MarketBadge market={a.market} />
      </div>
      <div className="flex flex-1 flex-col gap-3 px-5 py-4">
        <div className="space-y-1.5">
          <SynthesisLabel mode={mode} />
          <p className={cn("text-sm leading-relaxed", !a.supported && "italic text-muted-foreground")}>{a.answer}</p>
        </div>
        {a.evidence.length > 0 && (
          <div className="space-y-1.5">
            <SectionLabel kind="source" />
            <EvidenceList evidence={a.evidence} showExpert={false} />
          </div>
        )}
      </div>
    </Card>
  );
}

export function QualifierWarnings({ warnings }: { warnings: QualifierWarning[] }) {
  if (!warnings.length) return null;
  return (
    <div className="rounded-lg border border-amber-200 bg-amber-50/70 px-4 py-3 text-xs text-amber-900">
      <p className="mb-1 flex items-center gap-1.5 font-semibold">
        <AlertTriangle className="size-3.5" /> Qualifier check — review before quoting
      </p>
      <ul className="list-disc space-y-0.5 pl-4">
        {warnings.map((w, i) => (
          <li key={i}>{w.message}</li>
        ))}
      </ul>
    </div>
  );
}

export function MetaFooter({ meta, sources }: { meta: AnswerMeta; sources?: number }) {
  const dropped = meta.dropped_citations.length;
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
      <span className="flex items-center gap-1">
        <ShieldCheck className="size-3.5 text-emerald-600" />
        {sources ?? 0} citation{sources === 1 ? "" : "s"} verified against source
      </span>
      <span>{meta.retrieved} segments retrieved · {meta.experts_considered} experts considered</span>
      <span>{meta.model}{meta.cached ? " · cached" : ""}</span>
      {dropped > 0 && (
        <Tooltip>
          <TooltipTrigger asChild>
            <span className="flex cursor-default items-center gap-1 text-amber-700">
              <Info className="size-3" /> {dropped} invalid citation{dropped === 1 ? "" : "s"} removed
            </span>
          </TooltipTrigger>
          <TooltipContent>
            <ul className="space-y-0.5">
              {meta.dropped_citations.slice(0, 10).map((d) => (
                <li key={d}>{d}</li>
              ))}
            </ul>
          </TooltipContent>
        </Tooltip>
      )}
      {meta.notices.map((n) => (
        <span key={n} className="text-amber-700">{n}</span>
      ))}
    </div>
  );
}
