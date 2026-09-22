"""Prompt templates. Kept provider-neutral; versioned so cache keys change when prompts change."""

from __future__ import annotations

from ..models.schemas import Evidence, ExpertRef

PROMPT_VERSION = "v3"

GROUNDING_RULES = """You are a senior market-research analyst working ONLY from expert interview transcripts.

Grounding rules (non-negotiable):
1. Use ONLY the evidence records supplied below. Do not use outside or world knowledge. Do not guess.
2. Every claim must be supported by at least one cited evidence_id. Cite IDs exactly as given; never invent IDs.
3. Never write quotes, timestamps or speaker names as sources — the system attaches those from the database.
   You may put a short span copied VERBATIM from a record's text into `highlight`, or null.
4. Attribute statements to the specific expert (and market) who made them. Do not generalise one expert's view
   to a whole market or to all experts.
5. Preserve qualifiers exactly: scope ("in some areas", "some of the stronger centres", "selected procedures",
   "larger institutions"), magnitude ("high single digits", "low double digits"), conditions ("if funding is
   already available", "once the hospital becomes serious"), hedges ("could", "can take much longer") and
   contrasts ("rather than ... across the whole market"). Never turn a qualified statement into a universal claim.
6. Distinguish carefully:
   - common_view: experts express substantially the same position.
   - difference_in_emphasis: experts agree on direction but weigh factors differently, or differ in magnitude/scope.
   - disagreement: experts make genuinely contradictory claims about the same thing. Use this ONLY when the
     evidence shows incompatible positions. Different numbers for different markets are NOT a disagreement.
7. If the evidence does not answer the question, say so: set insufficient_evidence=true (where available),
   leave evidence empty, and do not speculate.
8. Be concise and specific. Prefer the experts' own terms."""


def format_evidence(evidence: list[Evidence]) -> str:
    lines = []
    for ev in evidence:
        q = f"\n  interviewer_question: {ev.question_context}" if ev.question_context else ""
        lines.append(
            f"- evidence_id: {ev.id}\n  transcript_id: {ev.transcript_id}\n  expert: {ev.expert_name}"
            f"\n  role: {ev.role}\n  market: {ev.market}{q}\n  text: {ev.text}"
        )
    return "\n".join(lines)


def format_experts(experts: list[ExpertRef]) -> str:
    return "\n".join(f"- transcript_id: {e.transcript_id} | expert: {e.expert_name} | role: {e.role} | market: {e.market}"
                     for e in experts)


def ask_prompt(question: str, experts: list[ExpertRef], evidence: list[Evidence]) -> str:
    return f"""Research question: {question}

Experts in scope:
{format_experts(experts)}

Evidence records (retrieved for this question, balanced across experts):
{format_evidence(evidence)}

Task:
- `answer`: a concise synthesis (2-6 sentences) that directly answers the question, attributing views to experts
  and markets, preserving every qualifier, and stating clearly whether differences are differences in emphasis or
  genuine disagreements.
- `expert_answers`: one entry per expert whose evidence is relevant (skip experts with no relevant evidence).
- `evidence`: every evidence_id that supports the synthesis.
- If nothing in the evidence addresses the question: insufficient_evidence=true, empty lists."""


def guide_prompt(question: str, experts: list[ExpertRef], evidence: list[Evidence]) -> str:
    return f"""Interview-guide question: {question}

Answer this question separately for EACH expert below, using only that expert's own evidence.

Experts:
{format_experts(experts)}

Evidence records (grouped by transcript_id):
{format_evidence(evidence)}

Rules for each `expert_answers` entry:
- transcript_id must be one of the experts above; include every expert exactly once.
- `answer`: 1-3 sentences in the third person ("Dr. X says ..."), preserving qualifiers and numbers exactly.
- `evidence`: only evidence_ids from THAT expert's transcript.
- If that expert's evidence does not address the question, answer exactly
  "No sufficient evidence in this transcript." with empty evidence."""


def findings_prompt(question: str, evidence: list[Evidence], experts: list[ExpertRef]) -> str:
    return f"""Topic (interview-guide question): {question}

Experts:
{format_experts(experts)}

Evidence records:
{format_evidence(evidence)}

Identify 1-4 findings comparing the experts on this topic. For each finding:
- `title`: short (max 8 words).
- `summary`: 1-3 sentences, attributed, qualifiers preserved.
- `classification`: common_view | difference_in_emphasis | disagreement (see rule 6 — be conservative).
- `transcript_ids`: the experts the finding concerns.
- `evidence`: supporting evidence_ids (at least one per expert mentioned).
Only produce findings the evidence supports."""


def themes_prompt(findings_block: str, experts: list[ExpertRef]) -> str:
    return f"""Experts analysed:
{format_experts(experts)}

Per-topic findings already extracted from the transcripts (each lists its supporting evidence records):
{findings_block}

Synthesise 3-6 cross-cutting THEMES that recur across several topics and experts. For each theme:
- `title`: short (max 6 words); `summary`: 2-3 sentences, attributed, qualifiers preserved.
- `classification`: common_view if experts broadly share it; difference_in_emphasis if they share direction but
  differ in weight/magnitude/scope; disagreement only for genuine contradiction.
- `transcript_ids` and `evidence`: reuse evidence_ids from the findings above (never invent new ones)."""


def repair_prompt(original_user_prompt: str, previous_json: str, warnings: list[str]) -> str:
    joined = "\n".join(f"- {w}" for w in warnings)
    return f"""{original_user_prompt}

A validator reviewed your previous answer and found qualifier problems:
{joined}

Previous answer (JSON):
{previous_json}

Rewrite the answer so every quantitative or scoped claim keeps the exact qualifiers of its source evidence and no
universal wording is added. Keep the same evidence IDs unless one is wrong. Return the full JSON object."""
