"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  AlertTriangle, BookOpenText, Check, FileText, LayoutDashboard, Lightbulb, Loader2, MessageSquareText, RefreshCw, X,
} from "lucide-react";
import { useData } from "@/components/data-context";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/misc";
import { cn, plural } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/guide", label: "Interview Guide", icon: BookOpenText },
  { href: "/insights", label: "Cross-Expert Insights", icon: Lightbulb },
  { href: "/ask", label: "Ask AI", icon: MessageSquareText },
  { href: "/transcripts", label: "Transcripts", icon: FileText },
];

function ModeBadge() {
  const { health } = useData();
  if (!health) return null;
  const llm = health.llm_mode === "llm";
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span>
          <Badge variant={llm ? "ai" : "muted"} className="cursor-default">
            <span className={cn("size-1.5 rounded-full", llm ? "bg-ai-foreground" : "bg-muted-foreground")} />
            {llm ? `${health.llm_provider} · ${health.llm_model}` : "Extractive demo mode"}
          </Badge>
        </span>
      </TooltipTrigger>
      <TooltipContent>
        {llm
          ? `Synthesis by ${health.llm_model}. Embeddings: ${health.embedding_model}.`
          : health.llm_warning ??
            "No LLM configured: answers are verbatim transcript excerpts. Set LLM_PROVIDER in .env (Gemini has a free tier)."}
      </TooltipContent>
    </Tooltip>
  );
}

function RefreshReport() {
  const { lastReport, dismissReport } = useData();
  if (!lastReport) return null;
  const r = lastReport;
  const changed = r.added.length + r.updated.length + r.removed.length;
  return (
    <div className="mx-6 mt-4 flex items-start gap-3 rounded-lg border bg-card px-4 py-3 text-sm shadow-sm">
      {r.failed.length ? (
        <AlertTriangle className="mt-0.5 size-4 text-amber-600" />
      ) : (
        <Check className="mt-0.5 size-4 text-emerald-600" />
      )}
      <div className="flex-1">
        <p className="font-medium">
          Scanned {plural(r.total, "file")} — {changed === 0 ? "no changes" : `${changed} change${changed === 1 ? "" : "s"}`}
        </p>
        <p className="text-muted-foreground">
          {r.added.length} new · {r.updated.length} updated · {r.removed.length} removed · {r.unchanged.length} unchanged
          {r.failed.length ? ` · ${r.failed.length} failed (${r.failed.map((f) => f.filename).join(", ")})` : ""}
        </p>
        {r.warnings.length > 0 && (
          <ul className="mt-1 list-disc pl-4 text-xs text-muted-foreground">
            {r.warnings.slice(0, 4).map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        )}
      </div>
      <button onClick={dismissReport} className="cursor-pointer text-muted-foreground hover:text-foreground" aria-label="Dismiss">
        <X className="size-4" />
      </button>
    </div>
  );
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { refresh, refreshing, stats, backendError } = useData();

  return (
    <TooltipProvider delayDuration={200}>
      <div className="flex min-h-screen">
        <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col border-r bg-card md:flex">
          <div className="flex items-center gap-2.5 px-5 py-5">
            <div className="grid size-8 place-items-center rounded-lg bg-primary text-sm font-bold text-primary-foreground">
              T
            </div>
            <div className="leading-tight">
              <p className="text-sm font-semibold tracking-tight">TranscriptIQ</p>
              <p className="text-[11px] text-muted-foreground">Expert research intelligence</p>
            </div>
          </div>
          <nav className="flex flex-col gap-0.5 px-3">
            {NAV.map(({ href, label, icon: Icon }) => {
              const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
              return (
                <Link
                  key={href}
                  href={href}
                  className={cn(
                    "flex items-center gap-2.5 rounded-md px-3 py-2 text-sm transition-colors",
                    active ? "bg-accent font-medium text-accent-foreground" : "text-muted-foreground hover:bg-muted hover:text-foreground",
                  )}
                >
                  <Icon className="size-4" />
                  {label}
                </Link>
              );
            })}
          </nav>
          <div className="mt-auto border-t px-5 py-4 text-xs text-muted-foreground">
            {stats ? (
              <>
                <p>
                  <span className="font-medium text-foreground">{stats.transcripts}</span> transcripts ·{" "}
                  <span className="font-medium text-foreground">{stats.evidence_segments}</span> evidence segments
                </p>
                <p className="mt-1 truncate" title={stats.embedding_model}>
                  Embeddings: {stats.embedding_model}
                </p>
              </>
            ) : (
              <p>Connecting…</p>
            )}
          </div>
        </aside>

        <div className="flex min-w-0 flex-1 flex-col">
          <header className="sticky top-0 z-20 flex h-14 items-center gap-3 border-b bg-background/85 px-6 backdrop-blur">
            <nav className="flex gap-1 overflow-x-auto md:hidden">
              {NAV.map(({ href, label }) => (
                <Link
                  key={href}
                  href={href}
                  className={cn(
                    "whitespace-nowrap rounded px-2 py-1 text-xs hover:bg-muted",
                    (href === "/" ? pathname === "/" : pathname.startsWith(href)) ? "bg-accent font-medium text-accent-foreground" : "text-muted-foreground",
                  )}
                >
                  {label}
                </Link>
              ))}
            </nav>
            <div className="ml-auto flex shrink-0 items-center gap-3">
              <span className="hidden sm:inline">
                <ModeBadge />
              </span>
              <Button variant="outline" size="sm" onClick={refresh} disabled={refreshing} aria-label="Refresh Transcripts">
                {refreshing ? <Loader2 className="animate-spin" /> : <RefreshCw />}
                <span className="hidden sm:inline">Refresh Transcripts</span>
              </Button>
            </div>
          </header>
          {backendError && (
            <div className="mx-6 mt-4 flex items-center gap-2 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
              <AlertTriangle className="size-4" /> {backendError}
            </div>
          )}
          <RefreshReport />
          <main className="mx-auto w-full max-w-7xl flex-1 px-6 py-6">{children}</main>
        </div>
      </div>
    </TooltipProvider>
  );
}
