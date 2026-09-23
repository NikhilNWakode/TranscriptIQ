# TranscriptIQ — Expert Interview Research Intelligence

TranscriptIQ turns expert interview transcripts into **traceable research insights**. It answers a configurable
interview guide for every expert, surfaces cross-expert themes and differences, and lets analysts ask free-form
questions — with every claim backed by a timestamped, verbatim quote.

> **Core principle — the LLM reasons over evidence; the backend owns source truth.**
> The model never generates a quote, a timestamp, an expert name or a market. It returns *evidence IDs*; the backend
> resolves those IDs against the original transcript and drops anything it cannot verify.

---

## Contents
- [Features](#features)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Setup](#setup) · [Environment variables](#environment-variables) · [Running](#running-the-application)
- [RAG pipeline](#rag-pipeline) · [Citation strategy](#citation-strategy) · [Hallucination prevention](#hallucination-prevention)
- [Dynamic transcript ingestion](#dynamic-transcript-ingestion) · [Scaling 3 → 30+](#scaling-from-3-to-30-transcripts)
- [Evaluation](#evaluation) · [Limitations](#limitations) · [Deployment](#deployment)
- [API](#api) · [Project structure](#project-structure)

---

## Features

| Area | What it does |
|---|---|
| **Dynamic ingestion** | Scans `/transcripts` on startup and on **Refresh Transcripts**. New → indexed, modified → re-indexed, deleted → removed, unchanged → skipped (SHA-256). Works for 3, 30 or 100 files with no code change. `.txt`, `.md`, `.pdf`, `.docx`. |
| **Parsing** | Extracts expert name / role / market (`Unknown` if absent — never guessed), speakers, timestamps, interviewer vs expert turns and the interviewer question preceding each answer. Strips document artifacts such as a stray `canvas` token. |
| **Interview Guide** | The questions in `config/interview_guide.json` are answered for **every** expert (6 × N answers), each with exact quotes and timestamps. |
| **Cross-Expert Insights** | Themes, *differences in emphasis* and *direct disagreements* are kept distinct; every finding must cite resolvable evidence or it is discarded. Counts (experts, markets, transcripts) are computed live. |
| **Ask AI** | Free-form questions across all transcripts with expert/market/role filters. Unsupported questions return: *“The provided transcripts do not contain sufficient evidence to answer this question.”* |
| **Source navigation** | Every quote is clickable → transcript viewer scrolls to the exact timestamp and highlights the cited evidence. |
| **Transcript viewer** | Expert selector, dynamic market/role filters, speaker labels, timestamps, in-transcript search, metadata editing (stored as overrides, survives re-indexing). |
| **Qualifier preservation** | A deterministic validator flags answers that drop qualifiers (“in some of the stronger centres”, “if funding is already available”, “rather than … across the whole market”) or add universal wording (“alone”, “all purchases”); in LLM mode it triggers one automatic repair pass. |
| **Provider-agnostic** | Any OpenAI-compatible endpoint (Groq, OpenRouter, Ollama…), Gemini, Claude, or an offline extractive mode — switched in `.env`. |

---

## Architecture

```text
 /transcripts ──► Folder scanner ──► Parser ──► Evidence store (SQLite) ──► Embeddings (cached)
   (txt/md/pdf/docx)  (SHA-256 diff)   (metadata,    one timestamped expert          │
                                        speakers,    turn = one evidence record      ▼
                                        Q→A context)                         Hybrid retriever
                                                                    (dense + BM25, balanced per expert,
                                                                     metadata filters, query expansion)
                                                                                     │
                                                                                     ▼
          UI  ◄── Qualifier check ◄── Citation validation ◄── Evidence IDs ◄──── LLM (structured output)
  (synthesis vs     (repair pass)     (exists? right project? right expert?
   verbatim source)                    retrieved? expert statement? verbatim highlight?)
```

**Separation of responsibilities**

| LLM — “What does the evidence mean?” | Backend — “What exactly was said, by whom, where, when?” |
|---|---|
| synthesis, comparison, interpretation, classification | source text, timestamps, expert metadata, transcript IDs, evidence IDs |

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| API | **FastAPI** + Pydantic v2 | Typed request/response validation; Pydantic models double as LLM output schemas. |
| Storage | **SQLite** (stdlib) | Zero-ops, single file, transactional. Evidence, embeddings, embedding cache and LLM cache in one DB. |
| Vector search | **numpy** matrix in memory + **BM25** | For 3–100 transcripts (≤ a few thousand segments) brute-force cosine is sub-millisecond; no extra service to run in an interview. Swappable for pgvector (see Scaling). |
| LLM | `LLMService` abstraction: **OpenAI-compatible** (Groq / OpenRouter / Ollama), **Gemini**, **Claude** (`anthropic` SDK, `messages.parse` structured outputs), **mock** | Provider and model are swapped through `.env`; adding one means implementing a single `generate(system, user, schema)` method. Transient failures (429/5xx) retry with the provider's own reset hint. |
| Embeddings | `EmbeddingService`: **local hashed vectors** (offline), **gemini-embedding-001**, **OpenAI-compatible** | Local mode needs no key or quota and pairs with any LLM provider; Gemini adds semantic matching for free. |
| Frontend | **Next.js 15**, **TypeScript**, **Tailwind CSS v4**, **shadcn/ui** components (Radix primitives, CVA) | Professional dashboard; `/api` is proxied by Next so API keys never reach the browser. |

---

## Setup

Prerequisites: **Python 3.11+**, **Node 18+** (tested with Python 3.11.9, Node 24).

```bash
# 1. configuration
cp .env.example .env          # edit LLM_PROVIDER / LLM_API_KEY (see below)

# 2. backend
cd backend
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

# 3. frontend
cd ../frontend
npm install
```

Put the case transcripts in `/transcripts` (e.g. `expert_1.txt`, `expert_2.txt`, `expert_3.txt`). Any number of
files is supported.

### Free LLM in 2 minutes (Groq)

1. Create a free API key at <https://console.groq.com/keys>.
2. In `.env`:
   ```ini
   LLM_PROVIDER=openai_compatible
   LLM_MODEL=openai/gpt-oss-120b
   LLM_BASE_URL=https://api.groq.com/openai/v1
   LLM_API_KEY=<your key>
   EMBEDDING_PROVIDER=local        # Groq has no embeddings API; local vectors need no key and no quota
   ```
3. Restart the backend. The header badge switches from “Extractive demo mode” to
   `openai_compatible · openai/gpt-oss-120b`.
4. Check it: `cd backend && .venv/Scripts/python -m app.eval.golden` → **12/12** (see [Evaluation](#evaluation)).

Model note: `openai/gpt-oss-120b` answers in ~2.5s and, on this evaluation, refused all seven unanswerable
questions. Confirm the name against your key (`curl -H "Authorization: Bearer $KEY" https://api.groq.com/openai/v1/models`);
available models differ per account.

**Free-tier rate limits matter.** A free Groq key allows ~8k tokens/minute, while a cold Cross-Expert Insights load
issues seven calls. The client honours the provider's `Retry-After`/reset hints, so the page completes — it just
takes ~2 minutes the first time. Results are cached in SQLite, so every later load is instant: **open Insights once
before a demo.** Interview Guide and Ask AI are one call each and unaffected.

Alternatives, same `.env` shape: **OpenRouter** `:free` models or a local **Ollama**
(`LLM_BASE_URL=http://localhost:11434/v1`) via `openai_compatible`; **Gemini** (`LLM_PROVIDER=gemini`,
`LLM_MODEL=gemini-3.6-flash`, free key at <https://aistudio.google.com/apikey>) additionally provides embeddings
(`EMBEDDING_PROVIDER=gemini`). With no key at all the app runs in deterministic **extractive mode**: retrieval,
quotes, timestamps and citations all work offline, without synthesis.

---

## Environment variables

All configuration lives in `.env` (git-ignored; `.env.example` documents every key). Keys are read by the backend only.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `mock` | `openai_compatible` (Groq / OpenRouter / Ollama) · `gemini` · `anthropic` · `mock` |
| `LLM_MODEL` | provider default | e.g. `openai/gpt-oss-120b`, `gemini-3.6-flash`, `claude-opus-5` |
| `LLM_API_KEY` | – | Provider key. Missing/invalid → graceful fallback to extractive mode with a visible notice. |
| `LLM_BASE_URL` | – | Required for `openai_compatible` (Groq: `https://api.groq.com/openai/v1`) |
| `LLM_EFFORT` | – | Claude only (`low`…`max`) |
| `EMBEDDING_PROVIDER` | `local` | `local` (offline, no key) · `gemini` · `openai_compatible`. Groq has no embeddings API — keep `local` with it. |
| `EMBEDDING_MODEL` / `EMBEDDING_API_KEY` / `EMBEDDING_BASE_URL` | – | Embedding key falls back to `LLM_API_KEY` |
| `TRANSCRIPTS_DIR` | `transcripts` | Folder that is scanned |
| `INTERVIEW_GUIDE_PATH` | `config/interview_guide.json` | Guide questions (+ suggested Ask AI questions) |
| `QUERY_EXPANSION_PATH` | `config/query_expansion.json` | Optional synonym groups for keyword retrieval |
| `DATABASE_PATH` | `backend/data/transcriptiq.db` | SQLite file (created automatically) |
| `ARTIFACT_TOKENS` | `canvas` | Comma-separated document artifacts to strip |
| `SEMANTIC_WEIGHT`, `PER_EXPERT_K`, `MAX_CONTEXT_SEGMENTS`, `GUIDE_BATCH_SIZE` | 0.6, 3, 40, 8 | Retrieval tuning |

---

## Running the application

Two terminals from the repository root:

```bash
# Terminal 1 — API on http://127.0.0.1:8000  (docs at /docs)
cd backend
.venv/Scripts/python -m uvicorn app.main:app --port 8000        # macOS/Linux: .venv/bin/python ...
```

```bash
# Terminal 2 — UI on http://localhost:3000
cd frontend
npm run dev
```

Tests:

```bash
cd backend
.venv/Scripts/python -m pytest -q
```

Golden regression set against the real transcripts (uses the configured LLM):

```bash
cd backend
.venv/Scripts/python -m app.eval.golden
```

---

## RAG pipeline

1. **Evidence units.** For structured interviews the natural unit is *one timestamped expert turn*, not an arbitrary
   token chunk. Each record stores the verbatim text, speaker, timestamp, seconds offset and the interviewer question
   that preceded it (`question_context`).
2. **Indexing.** Each expert turn is embedded as `Expert | Market | Role` + `Question` + `Answer` — so a guide
   question matches both the expert’s words and the question they were answering. Embeddings are cached by
   `(model, text)`; unchanged text is never re-embedded, and switching embedding model re-embeds automatically.
3. **Retrieval.** Hybrid score = `w·cosine + (1−w)·BM25` (each normalised). Selection is **balanced per expert**
   (top-k per transcript, relative floor, then global cap) so one talkative expert cannot crowd out others.
   Metadata filters (expert, market, role) are applied before scoring. Optional config-driven query expansion makes
   keyword retrieval robust to paraphrase (“main barriers” ↔ “what is holding adoption back?”).
4. **Synthesis.** Only retrieved evidence goes to the LLM — never whole transcripts. Output is a Pydantic schema
   (`LLMCrossExpertAnswer`, `LLMGuideAnswers`, `LLMFindings`) containing text + evidence IDs.
5. **Validation → response.** See below. Responses are cached (SQLite) keyed by provider, model, prompt version and
   the exact prompt (which embeds the evidence text) — so any transcript change invalidates naturally.

Interview-guide answers: for each question, per-expert retrieval (top 4), one LLM call per batch of ≤ 8 experts.
Insights: per-question findings (one call each), then one call that synthesises cross-cutting themes from those
findings — this map-reduce shape keeps prompts bounded as the number of experts grows.

## Citation strategy

```json
// LLM output (the only thing the model controls)
{ "answer": "…", "evidence": [{ "evidence_id": "expert_1_02_18", "highlight": "finance team wants to understand utilisation" }] }
```

Evidence IDs are deterministic and human-readable: `<transcript_id>_<MM>_<SS>` (collision-suffixed; interviewer turns
end in `_q`). The backend then, for every ID:

1. checks the ID is well-formed and **exists** in the database,
2. checks it belongs to the **current project** and an **active** transcript,
3. checks it was part of the **retrieved context** for this request (the model cannot cite what it wasn’t shown),
4. for per-expert answers, checks it belongs to **that expert**,
5. rejects **interviewer** turns,
6. resolves **expert, market, role, timestamp and exact text** from the database,
7. verifies the optional `highlight` is a **verbatim substring** (whitespace/quote-style/case-normalised) and returns
   character offsets; an unverifiable highlight is silently dropped,
8. if no valid evidence remains → the no-evidence response.

Dropped citations are reported in `meta.dropped_citations` and shown in the UI (“N invalid citations removed”).

## Hallucination prevention

| Layer | Mechanism |
|---|---|
| Retrieval grounding | The LLM sees only retrieved evidence records; prompts forbid outside knowledge. |
| Structured output | Pydantic schemas (Claude `messages.parse`; Gemini JSON-schema mode; validated + one corrective retry for others). |
| Evidence IDs | The model references evidence; it never authors source metadata. |
| Backend-controlled timestamps | Timestamps come from the parser, not the model. |
| Source-controlled quotes | The UI renders `Evidence.text` from the DB; highlights must be verbatim. |
| Validation | Existence, project, retrieved-set, expert-ownership and speaker-type checks. |
| Qualifier check | Numeric claims must keep the qualifiers of their source sentence; universal wording must be supported. |
| Classification guard | A “disagreement” citing only one expert is downgraded to “difference in emphasis”; findings with no valid evidence are discarded. |
| No-evidence response | Returned when retrieval finds nothing relevant, the model flags insufficiency, or all citations fail validation. |
| IDF-weighted evidence gate | The question’s subject terms are weighted by rarity in the corpus, so generic topic words (“robotic surgery”) cannot make an unanswerable question (“which company sells the most systems?”) look answered. Extractive mode requires ≥ 0.75 weighted coverage; measured on the case data, answerable questions score 1.0 and attack probes ≤ 0.62. |

---

## Dynamic transcript ingestion

```text
/transcripts ──► discover (non-recursive, natural sort, .txt .md .pdf .docx)
             ──► SHA-256 + parser version vs. stored hash
                   unchanged → skip · new → parse+embed+index · modified → re-parse+re-embed · missing → delete
             ──► per-file errors recorded (empty / unreadable / unparseable) without affecting other files
```

Nothing in the code refers to specific experts, markets or file counts — `transcript_id` is derived from the
filename, metadata from the file header, filters and statistics from the database. Add `transcripts/expert_4.txt`,
click **Refresh Transcripts**, and the new expert appears in every view.

Parser conventions (all optional): header lines such as `Name:`, `Role:`, `Market:` (value on the same or the next
line; Markdown bold is fine) and dialogue lines like `01:20 Dr. Martin:` / `[01:20] Speaker: text` /
`Speaker (01:20): text`. Short speaker labels (“Dr. Martin”) are matched to the header name. Interviewers are
detected by label or, failing that, by who asks the questions. Files without speaker labels fall back to paragraph
evidence with a warning.

---

## Scaling from 3 to 30+ transcripts

The MVP already handles 30–100 transcripts on one machine (6 × 30 = 180 guide answers in 24 batched LLM calls,
cached after the first run). Beyond that — or for many concurrent projects — the production path is:

```text
Object storage (S3/GCS) ──► async ingestion workers (queue) ──► transcript parser
   ──► evidence/chunk generation ──► embedding service (batched, cached)
   ──► PostgreSQL + pgvector (HNSW) ──► metadata filtering (project, market, role, date)
   ──► hybrid BM25 (tsvector) + semantic retrieval ──► cross-encoder reranker
   ──► LLM (map-reduce for many experts) ──► evidence validation ──► UI
```

Production improvements: asynchronous ingestion and background jobs; document versioning (keep old evidence IDs
resolvable for past reports); project/user isolation (the `project_id` column is already on every row); hybrid
retrieval in Postgres; reranking; embedding and LLM caching (already implemented locally); rate limiting;
observability (traces per request: retrieved IDs, dropped citations, qualifier warnings); cost monitoring per
project; continuous evaluation datasets; citation-accuracy monitoring in production.

## Evaluation

**Automated tests** (`backend/tests`, 94 tests, offline, run in seconds):

| Suite | Covers |
|---|---|
| `test_parser.py` | timestamps (MM:SS, HH:MM:SS), speaker extraction, interviewer/expert detection, short-label → full-name, question context, artifact removal (and keeping the same word mid-sentence), missing metadata → `Unknown`, malformed files |
| `test_ingestion.py` | new, unchanged, modified, deleted, invalid/empty files, empty folder, forced re-index, metadata overrides surviving re-index |
| `test_retrieval.py` | relevant evidence, cross-expert balance, market/transcript filters, expert-only retrieval, query expansion |
| `test_citations.py` | valid/invalid IDs, retrieved-set and expert-ownership checks, interviewer turns, deleted transcripts, timestamp/text resolution, verbatim vs fabricated highlights |
| `test_grounding.py` | supported/unsupported questions, no-evidence behaviour, LLM fabrications dropped, insufficient flag, qualifier repair loop, disagreement guard, HTTP API |
| `test_qualifiers.py` | the case’s WRONG/CORRECT pairs: *some stronger centres*, *some areas*, *across the whole market*, *if funding is already available*, *can take much longer*, *finance alone* |
| `test_providers.py` | Gemini / OpenAI-compatible request shape, schema fallback, JSON retry, 401/429 mapping, Claude structured output + refusal/truncation, fallback to extractive on LLM failure |
| `test_golden.py` | the 12-case golden set against the real `/transcripts` (5 regression questions + 7 no-evidence / interviewer-attack probes; auto-skips if the case transcripts are absent) |

Test fixtures in `backend/tests/fixtures` are **synthetic** (fictional experts and markets) and are never placed in
`/transcripts`.

**Golden set** (`config/golden_eval.json`, run with `python -m app.eval.golden`): the five required regression
questions plus seven no-evidence probes (e.g. *“Which company sells the most robotic surgery systems?”*, *“Which expert
believes robotic surgery saves exactly 25% of hospital costs?”*). Current result: **12/12**, citation accuracy 100%,
quote accuracy 100%, 0 qualifier warnings. Metrics:

| Metric | Definition |
|---|---|
| Retrieval accuracy | expected markets / timestamps / key phrases present in the resolved sources |
| Citation accuracy | share of returned sources that resolve to active expert statements |
| Quote accuracy | share of source quotes found verbatim in the original file |
| Answer groundedness | every answer sentence traceable to a cited source (LLM-judge extension; see limitations) |
| No-evidence accuracy | unsupported questions return the no-evidence message |
| Qualifier preservation | qualifier warnings raised on answers (target 0 after repair) |
| Cross-expert comparison accuracy | growth question is not labelled a disagreement; economics question distinguishes emphasis |

---

## Limitations

- **Extractive mode is not synthesis.** Without an LLM, answers are verbatim excerpts and insights show an evidence
  map rather than classified themes. Keyword retrieval (local embeddings) can miss paraphrases not covered by
  `config/query_expansion.json`; semantic embeddings fix this.
- **Qualifier checking is heuristic.** It targets numeric/scoped claims and common universalisers; it will not catch
  every possible distortion of meaning. Warnings are surfaced rather than silently blocking answers.
- **Theme quality depends on the model.** Validation guarantees every theme *cites real evidence*, not that the
  interpretation is optimal. Groundedness scoring by an LLM judge is documented but not automated.
- **Parser assumptions.** Speaker-labelled, timestamped dialogue works best. Scanned PDFs (images) need OCR, which is
  not included.
- **Single-node storage.** SQLite + in-memory vectors are right for an MVP, not for many concurrent users.
- **No authentication.** A single-project local tool; `project_id` isolation is in the schema but not exposed.

## Deployment

Two services. The browser only ever talks to the frontend: Next.js proxies `/api/*` to the backend server-side, so
the API key never reaches the client and no CORS configuration is needed.

**Backend — Render** (`render.yaml` is a ready blueprint: Dashboard → New → Blueprint → this repo). Set `LLM_API_KEY`
in the dashboard; everything else is in the blueprint. Manual setup is equivalent:

| Setting | Value |
|---|---|
| Root directory | `backend` |
| Build | `pip install -r requirements.txt` |
| Start | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| Health check | `/api/health` |
| Env | `LLM_PROVIDER=openai_compatible`, `LLM_MODEL=openai/gpt-oss-120b`, `LLM_BASE_URL=https://api.groq.com/openai/v1`, `LLM_API_KEY=…`, `EMBEDDING_PROVIDER=local`, `DATABASE_PATH=/tmp/transcriptiq.db`, `TRANSCRIPTS_DIR=/opt/render/project/src/transcripts` |

**Frontend — Vercel**: import the same repo, root directory `frontend`, and set one variable:
`BACKEND_URL=https://<your-render-service>.onrender.com`.

Notes for a hosted deployment:

* **Transcripts ship with the repo** and the SQLite index is rebuilt from them at startup, so no persistent disk is
  required. The trade-off: **“Refresh Transcripts” cannot pick up new files in production** — adding a transcript
  means committing it and redeploying. (Mount a persistent disk and point `TRANSCRIPTS_DIR` at it to change that.)
* **Render's free tier sleeps** after ~15 minutes idle; the first request then takes ~50s. For a live demo, warm it
  first, use a paid instance, or run locally.
* Answers are cached in the database, so a cold boot loses the cache and the first question per topic re-calls the LLM.

---

## API

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | mode, provider, model, embedding model |
| GET | `/api/stats` | dynamic counts |
| POST | `/api/refresh` | scan folder (idempotent) |
| POST | `/api/ingest?force=true` | force full re-index |
| GET | `/api/transcripts` · `/api/transcripts/{id}` | list · detail with all segments |
| PATCH | `/api/transcripts/{id}/metadata` | expert/role/market overrides |
| GET | `/api/filters` | experts, markets, roles from data |
| GET | `/api/evidence/{id}` | one resolved evidence record |
| GET | `/api/interview-guide` | configured questions |
| POST | `/api/interview-guide/answer` | `{question_id?}` → per-expert grounded answers |
| GET | `/api/insights` (alias `/api/themes`) | themes, differences, disagreements, per-question findings |
| POST | `/api/ask` | `{question, filters}` → grounded answer + sources |

Interactive docs: <http://127.0.0.1:8000/docs>.

## Project structure

```text
├── backend/
│   ├── app/
│   │   ├── api/routes.py            # HTTP endpoints
│   │   ├── ingestion/               # loaders (txt/md/pdf/docx), parser, idempotent indexer
│   │   ├── retrieval/               # embeddings, BM25, hybrid retriever, text utils
│   │   ├── llm/                     # provider adapters, prompts
│   │   ├── validation/              # citation resolution, qualifier checks
│   │   ├── services/                # research workflows, store, cache, app state, extractive mode
│   │   ├── models/schemas.py        # Pydantic API + LLM schemas
│   │   ├── eval/golden.py           # golden-set runner
│   │   ├── config.py · db.py · main.py
│   └── tests/                       # 94 tests + synthetic fixtures
├── frontend/                        # Next.js 15 + Tailwind v4 + shadcn/ui
│   ├── app/{page,guide,insights,ask,transcripts}
│   ├── components/{ui,research,app-shell,data-context}.tsx
│   └── lib/{api,types,utils}.ts
├── transcripts/                     # input data (drop files here)
├── config/                          # interview_guide.json, golden_eval.json, query_expansion.json
└── .env.example
```
