import type { Metadata } from "next";
import "./globals.css";
import { AppShell } from "@/components/app-shell";
import { DataProvider } from "@/components/data-context";

export const metadata: Metadata = {
  title: "TranscriptIQ — Expert Research Intelligence",
  description: "Traceable, evidence-grounded insights from expert interview transcripts.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <DataProvider>
          <AppShell>{children}</AppShell>
        </DataProvider>
      </body>
    </html>
  );
}
