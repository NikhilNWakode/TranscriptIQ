"use client";

import * as React from "react";
import { Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Check, FileText, Pencil, Search, X } from "lucide-react";
import { useApi, useData } from "@/components/data-context";
import { EmptyTranscripts, ErrorNote, LoadingCards, MarketBadge, PageHeader } from "@/components/research";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { api, ApiError } from "@/lib/api";
import type { Evidence, TranscriptDetail } from "@/lib/types";
import { cn } from "@/lib/utils";

function escapeRegExp(s: string) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Renders text with (a) the cited highlight span and (b) search matches marked. */
function SegmentText({ text, search, hl }: { text: string; search: string; hl?: [number, number] | null }) {
  const marks: { start: number; end: number; kind: "hl" | "search" }[] = [];
  if (hl && hl[0] >= 0 && hl[1] <= text.length && hl[0] < hl[1]) marks.push({ start: hl[0], end: hl[1], kind: "hl" });
  if (search.trim().length >= 2) {
    const rx = new RegExp(escapeRegExp(search.trim()), "gi");
    for (const m of text.matchAll(rx)) {
      const start = m.index ?? 0;
      const end = start + m[0].length;
      if (!marks.some((x) => start < x.end && end > x.start)) marks.push({ start, end, kind: "search" });
    }
  }
  marks.sort((a, b) => a.start - b.start);
  const out: React.ReactNode[] = [];
  let pos = 0;
  marks.forEach((m, i) => {
    if (m.start > pos) out.push(text.slice(pos, m.start));
    out.push(
      <mark key={i} className={cn("rounded-sm px-0.5 text-inherit", m.kind === "hl" ? "bg-amber-300/80" : "bg-sky-200")}>
        {text.slice(m.start, m.end)}
      </mark>,
    );
    pos = m.end;
  });
  if (pos < text.length) out.push(text.slice(pos));
  return <>{out}</>;
}

function MetadataEditor({ t, onSaved }: { t: TranscriptDetail; onSaved: () => void }) {
  const [editing, setEditing] = React.useState(false);
  const [form, setForm] = React.useState({ expert_name: t.expert_name, role: t.role, market: t.market });
  const [error, setError] = React.useState<string | null>(null);
  React.useEffect(() => setForm({ expert_name: t.expert_name, role: t.role, market: t.market }), [t]);

  if (!editing) {
    return (
      <Button variant="ghost" size="sm" onClick={() => setEditing(true)}>
        <Pencil /> Edit metadata
      </Button>
    );
  }
  const save = async () => {
    try {
      await api.updateMetadata(t.id, form);
      setEditing(false);
      onSaved();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Save failed");
    }
  };
  return (
    <div className="flex w-full flex-wrap items-end gap-2 rounded-lg border bg-muted/40 p-3">
      {(["expert_name", "role", "market"] as const).map((k) => (
        <label key={k} className="flex min-w-40 flex-1 flex-col gap-1 text-xs text-muted-foreground">
          {k === "expert_name" ? "Expert" : k[0].toUpperCase() + k.slice(1)}
          <Input value={form[k]} onChange={(e) => setForm({ ...form, [k]: e.target.value })} className="h-8" maxLength={200} />
        </label>
      ))}
      <Button size="sm" onClick={save}>
        <Check /> Save
      </Button>
      <Button size="sm" variant="ghost" onClick={() => setEditing(false)}>
        <X /> Cancel
      </Button>
      {error && <p className="w-full text-xs text-rose-700">{error}</p>}
      <p className="w-full text-[11px] text-muted-foreground">
        Overrides are stored separately from the source file and survive re-indexing. Leave a field empty to revert to the parsed value.
      </p>
    </div>
  );
}

function Viewer() {
  const router = useRouter();
  const params = useSearchParams();
  const { reload } = useData();
  const list = useApi(() => api.transcripts());
  const indexed = React.useMemo(() => (list.data ?? []).filter((t) => t.status === "indexed"), [list.data]);

  const selectedId = params.get("id") ?? indexed[0]?.id ?? null;
  const evidenceId = params.get("evidence");
  const hlParam = params.get("hl");
  const hl = React.useMemo<[number, number] | null>(() => {
    const m = hlParam?.match(/^(\d+)-(\d+)$/);
    return m ? [Number(m[1]), Number(m[2])] : null;
  }, [hlParam]);

  const [market, setMarket] = React.useState("all");
  const [role, setRole] = React.useState("all");
  const [search, setSearch] = React.useState("");
  const [expertOnly, setExpertOnly] = React.useState(false);

  const detail = useApi(
    () => (selectedId ? api.transcript(selectedId) : Promise.resolve(null as TranscriptDetail | null)),
    [selectedId],
  );

  const markets = Array.from(new Set(indexed.map((t) => t.market))).sort();
  const roles = Array.from(new Set(indexed.map((t) => t.role))).sort();
  const visible = indexed.filter((t) => (market === "all" || t.market === market) && (role === "all" || t.role === role));

  // scroll to + flash the cited evidence
  React.useEffect(() => {
    if (!evidenceId || !detail.data) return;
    const el = document.getElementById(`seg-${evidenceId}`);
    if (!el) return;
    // Instant, direct scroll: smooth scrolling and requestAnimationFrame are both paused while a tab is in the
    // background (e.g. a citation opened in a new tab). The timed retry covers Next.js scroll restoration.
    el.scrollIntoView({ block: "center" });
    const retry = window.setTimeout(() => el.scrollIntoView({ block: "center" }), 200);
    el.classList.remove("evidence-flash");
    void el.offsetWidth;
    el.classList.add("evidence-flash");
    return () => window.clearTimeout(retry);
  }, [evidenceId, detail.data]);

  const select = (id: string) => router.push(`/transcripts?id=${encodeURIComponent(id)}`, { scroll: false });

  if (list.loading && !list.data) return <LoadingCards n={2} />;
  if (list.error) return <ErrorNote message={list.error} />;
  if (!indexed.length) return <EmptyTranscripts />;

  const t = detail.data;
  const segments: Evidence[] = (t?.segments_list ?? []).filter((s) => !expertOnly || s.speaker_type === "expert");
  const q = search.trim().toLowerCase();
  const matchCount = q.length >= 2 ? segments.filter((s) => s.text.toLowerCase().includes(q)).length : 0;

  return (
    <div className="grid gap-6 lg:grid-cols-[280px_1fr]">
      <div className="space-y-3 lg:sticky lg:top-20 lg:h-fit">
        <Card className="space-y-2 p-3 text-xs">
          <FilterSelect label="Market" value={market} options={markets} onChange={setMarket} />
          <FilterSelect label="Role" value={role} options={roles} onChange={setRole} />
        </Card>
        <Card className="max-h-[60vh] overflow-y-auto p-1.5">
          {visible.length === 0 && <p className="p-3 text-xs text-muted-foreground">No transcripts match these filters.</p>}
          {visible.map((x) => (
            <button
              key={x.id}
              onClick={() => select(x.id)}
              className={cn(
                "flex w-full cursor-pointer flex-col items-start gap-0.5 rounded-md px-3 py-2 text-left transition-colors",
                x.id === selectedId ? "bg-accent" : "hover:bg-muted",
              )}
            >
              <span className="text-sm font-medium">{x.expert_name}</span>
              <span className="text-xs text-muted-foreground">
                {x.role} · {x.market}
              </span>
            </button>
          ))}
        </Card>
      </div>

      <div className="min-w-0 space-y-4">
        {detail.error && <ErrorNote message={detail.error} />}
        {!t && detail.loading && <LoadingCards n={1} />}
        {t && (
          <>
            <Card className="space-y-3 p-5">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <h2 className="text-lg font-semibold tracking-tight">{t.expert_name}</h2>
                  <p className="text-sm text-muted-foreground">{t.role}</p>
                </div>
                <div className="flex items-center gap-2">
                  <MarketBadge market={t.market} />
                  {t.metadata_edited && <Badge variant="muted">edited</Badge>}
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
                <span className="flex items-center gap-1 font-mono">
                  <FileText className="size-3.5" /> {t.filename}
                </span>
                <span>
                  {t.segments} segments · {t.expert_segments} expert statements
                </span>
                {t.warnings.map((w) => (
                  <span key={w} className="text-amber-700">
                    {w}
                  </span>
                ))}
              </div>
              <MetadataEditor t={t} onSaved={reload} />
            </Card>

            <div className="sticky top-14 z-10 -mx-1 flex flex-wrap items-center gap-3 bg-background/90 px-1 py-2 backdrop-blur">
              <div className="relative min-w-60 flex-1">
                <Search className="absolute left-2.5 top-2.5 size-4 text-muted-foreground" />
                <Input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search within transcript…" className="pl-8" />
              </div>
              {q.length >= 2 && <span className="text-xs text-muted-foreground">{matchCount} matching segments</span>}
              <label className="flex cursor-pointer items-center gap-1.5 text-xs text-muted-foreground">
                <input type="checkbox" checked={expertOnly} onChange={(e) => setExpertOnly(e.target.checked)} />
                Expert statements only
              </label>
            </div>

            <Card className="divide-y">
              {segments.map((s) => {
                const cited = s.id === evidenceId;
                const dim = q.length >= 2 && !s.text.toLowerCase().includes(q);
                return (
                  <div
                    key={s.id}
                    id={`seg-${s.id}`}
                    className={cn(
                      "grid scroll-mt-32 grid-cols-[64px_1fr] gap-4 px-5 py-3.5 transition-colors",
                      s.speaker_type !== "expert" && "bg-muted/30",
                      cited && "rounded-md bg-amber-50 ring-2 ring-amber-300",
                      dim && "opacity-40",
                    )}
                  >
                    <span className="pt-0.5 font-mono text-xs font-medium tabular-nums text-muted-foreground">{s.timestamp ?? "—"}</span>
                    <div>
                      <p className={cn("mb-0.5 text-xs font-semibold", s.speaker_type === "expert" ? "text-primary" : "text-muted-foreground")}>
                        {s.speaker}
                        {cited && (
                          <Badge variant="source" className="ml-2">
                            Cited evidence · {s.id}
                          </Badge>
                        )}
                      </p>
                      <p className={cn("text-sm leading-relaxed", s.speaker_type !== "expert" && "text-muted-foreground")}>
                        <SegmentText text={s.text} search={search} hl={cited ? hl : null} />
                      </p>
                    </div>
                  </div>
                );
              })}
            </Card>
          </>
        )}
      </div>
    </div>
  );
}

function FilterSelect({ label, value, options, onChange }: { label: string; value: string; options: string[]; onChange: (v: string) => void }) {
  return (
    <label className="flex items-center justify-between gap-2">
      <span className="font-medium text-muted-foreground">{label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)} className="h-7 max-w-44 flex-1 cursor-pointer rounded-md border bg-card px-2 text-xs">
        <option value="all">All ({options.length})</option>
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </label>
  );
}

export default function TranscriptsPage() {
  return (
    <div>
      <PageHeader
        title="Transcripts"
        description="Source of truth. Every citation in the app links here, scrolls to the exact timestamp and highlights the evidence."
      />
      <Suspense fallback={<LoadingCards n={2} />}>
        <Viewer />
      </Suspense>
    </div>
  );
}
