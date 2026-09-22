"use client";

import * as React from "react";
import { api, ApiError } from "@/lib/api";
import type { Health, IngestReport, Stats } from "@/lib/types";

interface DataContextValue {
  /** Increments whenever the index changes; pages include it in effect deps to refetch. */
  version: number;
  stats: Stats | null;
  health: Health | null;
  backendError: string | null;
  refreshing: boolean;
  lastReport: IngestReport | null;
  refresh: () => Promise<void>;
  reload: () => void;
  dismissReport: () => void;
}

const DataContext = React.createContext<DataContextValue | null>(null);

export function DataProvider({ children }: { children: React.ReactNode }) {
  const [version, setVersion] = React.useState(0);
  const [stats, setStats] = React.useState<Stats | null>(null);
  const [health, setHealth] = React.useState<Health | null>(null);
  const [backendError, setBackendError] = React.useState<string | null>(null);
  const [refreshing, setRefreshing] = React.useState(false);
  const [lastReport, setLastReport] = React.useState<IngestReport | null>(null);

  const load = React.useCallback(async () => {
    try {
      const [s, h] = await Promise.all([api.stats(), api.health()]);
      setStats(s);
      setHealth(h);
      setBackendError(null);
    } catch (e) {
      setBackendError(e instanceof ApiError ? e.message : "Backend unavailable");
    }
  }, []);

  React.useEffect(() => {
    load();
  }, [load, version]);

  const refresh = React.useCallback(async () => {
    setRefreshing(true);
    try {
      const report = await api.refresh();
      setLastReport(report);
      setVersion((v) => v + 1);
    } catch (e) {
      setBackendError(e instanceof ApiError ? e.message : "Refresh failed");
    } finally {
      setRefreshing(false);
    }
  }, []);

  const value: DataContextValue = {
    version,
    stats,
    health,
    backendError,
    refreshing,
    lastReport,
    refresh,
    reload: () => setVersion((v) => v + 1),
    dismissReport: () => setLastReport(null),
  };
  return <DataContext.Provider value={value}>{children}</DataContext.Provider>;
}

export function useData() {
  const ctx = React.useContext(DataContext);
  if (!ctx) throw new Error("useData must be used inside DataProvider");
  return ctx;
}

/** Small fetch hook that re-runs when the index version (or deps) change. */
export function useApi<T>(fn: () => Promise<T>, deps: React.DependencyList = []) {
  const { version } = useData();
  const [data, setData] = React.useState<T | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [loading, setLoading] = React.useState(true);
  React.useEffect(() => {
    let alive = true;
    setLoading(true);
    fn()
      .then((d) => alive && (setData(d), setError(null)))
      .catch((e) => alive && setError(e instanceof ApiError ? e.message : "Something went wrong"))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [version, ...deps]);
  return { data, error, loading, setData };
}
