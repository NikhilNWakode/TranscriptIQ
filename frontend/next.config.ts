import type { NextConfig } from "next";

// The browser only ever talks to /api on this origin; Next proxies to the FastAPI backend.
// API keys therefore never reach the client.
const BACKEND_URL = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${BACKEND_URL}/api/:path*` }];
  },
  // LLM-backed endpoints (insights, guide) can take longer than the default 30s proxy timeout.
  experimental: { proxyTimeout: 300_000 },
};

export default nextConfig;
