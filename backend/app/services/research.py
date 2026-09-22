"""Research workflows: Ask AI, interview-guide answers and cross-expert insights.

Pipeline for every workflow:
    retrieve evidence → (LLM reasons over it → evidence IDs) → backend resolves + validates IDs →
    qualifier check (→ one repair pass) → response built only from resolved source records.
"""

from __future__ import annotations

import json
import logging
import re
from collections import OrderedDict
from pathlib import Path

from ..config import Settings
from ..llm.prompts import (GROUNDING_RULES, PROMPT_VERSION, ask_prompt, findings_prompt, guide_prompt,
                           repair_prompt, themes_prompt)
from ..llm.providers import LLMError, LLMService
from ..models.schemas import (NO_EVIDENCE_MESSAGE, AnswerMeta, AskFilters, AskRequest, AskResponse, Evidence,
                              ExpertAnswer, ExpertRef, Finding, GuideQuestion, GuideQuestionResult,
                              InsightsResponse, LLMCrossExpertAnswer, LLMEvidenceRef, LLMFindings,
                              LLMGuideAnswers, QualifierWarning)
from ..retrieval.embeddings import LocalHashEmbedding
from ..retrieval.retriever import Retriever
from ..validation.citations import resolve_references
from ..validation.qualifiers import check_qualifiers
from .cache import LLMCache
from .extractive import cross_expert_summary, expert_summary
from .store import EvidenceStore

log = logging.getLogger(__name__)

NO_EXPERT_EVIDENCE = "No sufficient evidence in this transcript."
# Extractive mode has no model to judge answerability, so it requires the (IDF-weighted) subject of the
# question to be present in the evidence. Measured on the case data: answerable questions score 1.0,
# unanswerable probes ("Which company sells the most systems?") score <= 0.62.
EXTRACTIVE_MIN_COVERAGE = 0.75
# In LLM mode the model judges answerability (prompt rule 7); this gate only catches questions whose subject is
# essentially absent from the corpus, so no tokens are spent on them. It is deliberately far below the
# extractive threshold: with semantic embeddings, a legitimate question can score ~0.25 on lexical coverage.
LLM_MIN_COVERAGE = 0.2


class GuideConfigError(Exception):
    pass


def load_interview_guide(path: Path) -> tuple[dict, list[GuideQuestion]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        questions = [GuideQuestion(id=str(q["id"]), text=str(q["text"]).strip()) for q in data["questions"]]
    except FileNotFoundError as exc:
        raise GuideConfigError(f"Interview guide not found at {path.name}") from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise GuideConfigError("Interview guide JSON is invalid (expected {questions: [{id, text}]})") from exc
    if len({q.id for q in questions}) != len(questions):
        raise GuideConfigError("Interview guide question ids must be unique")
    return data, questions


def _sort_evidence(evs: list[Evidence]) -> list[Evidence]:
    return sorted(evs, key=lambda e: (e.transcript_id, e.timestamp_seconds if e.timestamp_seconds is not None else 0))


class ResearchService:
    def __init__(self, settings: Settings, store: EvidenceStore, retriever: Retriever, llm: LLMService,
                 cache: LLMCache):
        self.settings, self.store, self.retriever, self.llm, self.cache = settings, store, retriever, llm, cache
        self._last_scores: dict[str, float] = {}

    # ------------------------------------------------------------------------------------------
    @property
    def mode(self) -> str:
        return "extractive" if self.llm.is_mock else "llm"

    def _meta(self, **kw) -> AnswerMeta:
        return AnswerMeta(mode=self.mode, model=f"{self.llm.provider}:{self.llm.model}", **kw)

    def _call(self, kind: str, user: str, schema):
        """Cached structured LLM call. Returns (parsed, cached_flag)."""
        key = LLMCache.key(self.llm.provider, self.llm.model, PROMPT_VERSION, kind, GROUNDING_RULES, user)
        hit = self.cache.get(key, schema)
        if hit is not None:
            return hit, True
        out = self.llm.generate(GROUNDING_RULES, user, schema)
        self.cache.set(key, out)
        return out, False

    def _experts(self, filters: AskFilters | None = None) -> list[ExpertRef]:
        experts = self.store.experts()
        if filters:
            if filters.transcript_ids:
                experts = [e for e in experts if e.transcript_id in set(filters.transcript_ids)]
            if filters.markets:
                experts = [e for e in experts if e.market.lower() in {m.lower() for m in filters.markets}]
            if filters.roles:
                experts = [e for e in experts if e.role.lower() in {r.lower() for r in filters.roles}]
        return experts

    def _expert_answer(self, expert: ExpertRef, answer: str, evidence: list[Evidence]) -> ExpertAnswer:
        supported = bool(evidence)
        return ExpertAnswer(**expert.model_dump(), answer=answer if supported else NO_EXPERT_EVIDENCE,
                            supported=supported, evidence=_sort_evidence(evidence))

    # ==========================================================================================
    # Ask AI
    # ==========================================================================================
    def _no_evidence(self, question: str, **meta) -> AskResponse:
        return AskResponse(question=question, answer=NO_EVIDENCE_MESSAGE, insufficient_evidence=True,
                           expert_answers=[], sources=[], meta=self._meta(**meta))

    def ask(self, req: AskRequest) -> AskResponse:
        question = req.question
        experts = self._experts(req.filters)
        if not experts:
            return self._no_evidence(question)
        hits = self.retriever.retrieve(question, filters=req.filters, per_expert_k=self.settings.per_expert_k,
                                       max_total=self.settings.max_context_segments)
        # Evidence gate. With lexical (local) embeddings, a question whose subject terms never occur in the
        # transcripts cannot be answered. Extractive mode has no LLM to judge relevance, so it requires most
        # subject terms to be present; LLM mode lets the model decide (it can return insufficient_evidence).
        coverage = self.retriever.term_coverage(question, hits)
        lexical_embedder = isinstance(self.retriever.embedder, LocalHashEmbedding)
        floor = EXTRACTIVE_MIN_COVERAGE if self.llm.is_mock else LLM_MIN_COVERAGE
        if not hits or (lexical_embedder and coverage == 0) or coverage < floor:
            return self._no_evidence(question, retrieved=len(hits), experts_considered=len(experts))

        by_id = self.store.get_evidence([h.evidence_id for h in hits])
        context = [by_id[h.evidence_id] for h in hits if h.evidence_id in by_id]
        allowed = set(by_id)
        expert_by_id = {e.transcript_id: e for e in experts}

        if self.llm.is_mock:
            return self._ask_extractive(question, context, experts, len(hits))

        user = ask_prompt(question, experts, context)
        try:
            raw, cached = self._call("ask", user, LLMCrossExpertAnswer)
        except LLMError as exc:
            log.warning("LLM failure in ask: %s", exc)
            resp = self._ask_extractive(question, context, experts, len(hits))
            resp.meta.notices.append(f"LLM unavailable ({exc}); showing extractive answer.")
            return resp

        resp = self._validate_ask(question, raw, allowed, expert_by_id, len(hits), len(experts), cached)
        if resp.qualifier_warnings and not resp.insufficient_evidence:
            try:
                fixed_raw, _ = self._call(
                    "ask-repair",
                    repair_prompt(user, raw.model_dump_json(), [w.message for w in resp.qualifier_warnings]),
                    LLMCrossExpertAnswer,
                )
                fixed = self._validate_ask(question, fixed_raw, allowed, expert_by_id, len(hits), len(experts), cached)
                if not fixed.insufficient_evidence and len(fixed.qualifier_warnings) < len(resp.qualifier_warnings):
                    resp = fixed
            except LLMError:
                pass
        return resp

    def _validate_ask(self, question: str, raw: LLMCrossExpertAnswer, allowed: set[str],
                      expert_by_id: dict[str, ExpertRef], retrieved: int, n_experts: int, cached: bool) -> AskResponse:
        dropped: list[str] = []
        top = resolve_references(raw.evidence, self.store, allowed_ids=allowed)
        dropped += top.dropped
        expert_answers: list[ExpertAnswer] = []
        for ea in raw.expert_answers:
            expert = expert_by_id.get(ea.transcript_id)
            if expert is None:
                dropped.append(f"{ea.transcript_id}: unknown expert in answer")
                continue
            res = resolve_references(ea.evidence, self.store, allowed_ids=allowed, transcript_id=ea.transcript_id)
            dropped += res.dropped
            if res.evidence:
                expert_answers.append(self._expert_answer(expert, ea.answer.strip(), res.evidence))
        sources: OrderedDict[str, Evidence] = OrderedDict()
        for ev in top.evidence + [ev for ea in expert_answers for ev in ea.evidence]:
            sources.setdefault(ev.id, ev)
        meta = self._meta(cached=cached, retrieved=retrieved, experts_considered=n_experts, dropped_citations=dropped)
        if raw.insufficient_evidence or not sources:
            return AskResponse(question=question, answer=NO_EVIDENCE_MESSAGE, insufficient_evidence=True,
                               expert_answers=[], sources=[], meta=meta)
        all_sources = _sort_evidence(list(sources.values()))
        warnings = check_qualifiers(raw.answer, all_sources)
        for ea in expert_answers:
            warnings += check_qualifiers(ea.answer, ea.evidence)
        return AskResponse(question=question, answer=raw.answer.strip(), insufficient_evidence=False,
                           expert_answers=expert_answers, sources=all_sources, qualifier_warnings=warnings, meta=meta)

    def _ask_extractive(self, question: str, context: list[Evidence], experts: list[ExpertRef],
                        retrieved: int) -> AskResponse:
        grouped: OrderedDict[str, list[Evidence]] = OrderedDict()
        for ev in context:  # context is already ranked best-first
            grouped.setdefault(ev.transcript_id, []).append(ev)
        expert_by_id = {e.transcript_id: e for e in experts}
        answers = [
            self._expert_answer(expert_by_id[tid], expert_summary(self.retriever.expander.expand(question), evs[:1]), evs[:1])
            for tid, evs in grouped.items() if tid in expert_by_id
        ]
        sources = _sort_evidence([ev for evs in grouped.values() for ev in evs[:2]])
        return AskResponse(
            question=question, answer=cross_expert_summary(self.retriever.expander.expand(question), grouped, len(experts)),
            insufficient_evidence=False, expert_answers=answers, sources=sources,
            meta=self._meta(retrieved=retrieved, experts_considered=len(experts)),
        )

    # ==========================================================================================
    # Interview guide
    # ==========================================================================================
    def guide_questions(self) -> tuple[dict, list[GuideQuestion]]:
        return load_interview_guide(self.settings.interview_guide_path)

    def _guide_context(self, question: str, experts: list[ExpertRef], k: int = 4) -> dict[str, list[Evidence]]:
        out: dict[str, list[Evidence]] = {}
        for e in experts:
            hits = self.retriever.retrieve_for_expert(question, e.transcript_id, k=k)
            if not hits or self.retriever.term_coverage(question, hits) == 0:
                out[e.transcript_id] = []
                continue
            by_id = self.store.get_evidence([h.evidence_id for h in hits])
            out[e.transcript_id] = [by_id[h.evidence_id] for h in hits if h.evidence_id in by_id]
            self._last_scores.update({h.evidence_id: h.score for h in hits})
        return out

    def answer_guide_question(self, q: GuideQuestion) -> GuideQuestionResult:
        experts = self._experts()
        context = self._guide_context(q.text, experts)
        if self.llm.is_mock:
            answers = []
            for e in experts:
                evs = context.get(e.transcript_id, [])
                top = evs[:1]
                # without an LLM to choose, also show a close runner-up (answers often span two turns)
                if len(evs) > 1 and self._last_scores.get(evs[1].id, 0) >= 0.75 * self._last_scores.get(evs[0].id, 1):
                    top = evs[:2]
                answers.append(self._expert_answer(e, expert_summary(self.retriever.expander.expand(q.text), top), top))
            return GuideQuestionResult(question=q, expert_answers=answers,
                                       meta=self._meta(retrieved=sum(len(v) for v in context.values()),
                                                       experts_considered=len(experts)))

        answers_by_tid: dict[str, ExpertAnswer] = {}
        dropped: list[str] = []
        notices: list[str] = []
        cached_all = True
        batch = max(1, self.settings.guide_batch_size)
        for start in range(0, len(experts), batch):
            group = [e for e in experts[start:start + batch] if context.get(e.transcript_id)]
            if not group:
                continue
            evidence = [ev for e in group for ev in context[e.transcript_id]]
            allowed = {ev.id for ev in evidence}
            try:
                raw, cached = self._call("guide", guide_prompt(q.text, group, evidence), LLMGuideAnswers)
            except LLMError as exc:
                log.warning("LLM failure in guide answer: %s", exc)
                notices.append(f"LLM unavailable ({exc}); extractive answers shown.")
                for e in group:
                    top = context[e.transcript_id][:1]
                    answers_by_tid[e.transcript_id] = self._expert_answer(e, expert_summary(q.text, top), top)
                continue
            cached_all &= cached
            group_ids = {e.transcript_id: e for e in group}
            for ea in raw.expert_answers:
                expert = group_ids.get(ea.transcript_id)
                if expert is None or ea.transcript_id in answers_by_tid:
                    continue
                res = resolve_references(ea.evidence, self.store, allowed_ids=allowed, transcript_id=ea.transcript_id)
                dropped += res.dropped
                answers_by_tid[ea.transcript_id] = self._expert_answer(expert, ea.answer.strip(), res.evidence)

        answers = [answers_by_tid.get(e.transcript_id) or self._expert_answer(e, "", []) for e in experts]
        warnings: list[QualifierWarning] = []
        for a in answers:
            if a.supported:
                warnings += check_qualifiers(a.answer, a.evidence)
        return GuideQuestionResult(
            question=q, expert_answers=answers, qualifier_warnings=warnings,
            meta=self._meta(cached=cached_all, retrieved=sum(len(v) for v in context.values()),
                            experts_considered=len(experts), dropped_citations=dropped, notices=notices),
        )

    # ==========================================================================================
    # Cross-expert insights
    # ==========================================================================================
    def _finding(self, fid: str, title: str, summary: str, classification: str, refs: list[LLMEvidenceRef],
                 allowed: set[str], experts: dict[str, ExpertRef], question_id: str | None,
                 dropped: list[str]) -> Finding | None:
        res = resolve_references(refs, self.store, allowed_ids=allowed)
        dropped += res.dropped
        if not res.evidence:
            dropped.append(f"finding '{title}': removed (no valid evidence)")
            return None
        involved = list(dict.fromkeys(ev.transcript_id for ev in res.evidence))
        if classification == "disagreement" and len(involved) < 2:
            classification = "difference_in_emphasis"  # a contradiction needs at least two sides
            dropped.append(f"finding '{title}': downgraded from disagreement (evidence from one expert only)")
        return Finding(id=fid, title=title.strip(), summary=summary.strip(), classification=classification,
                       question_id=question_id, experts=[experts[t] for t in involved if t in experts],
                       evidence=_sort_evidence(res.evidence))

    def insights(self) -> InsightsResponse:
        experts = self._experts()
        _, questions = self.guide_questions()
        expert_map = {e.transcript_id: e for e in experts}
        markets = {e.market for e in experts if e.market != "Unknown"}
        base = dict(experts_analyzed=len({e.expert_name for e in experts}), markets_represented=len(markets),
                    transcripts_analyzed=len(experts))
        if not experts:
            return InsightsResponse(**base, themes=[], differences=[], disagreements=[], by_question=[],
                                    meta=self._meta())

        per_question: list[tuple[GuideQuestion, list[Evidence]]] = []
        for q in questions:
            ctx = self._guide_context(q.text, experts, k=2)  # 2 per expert keeps each per-question call inside small free-tier budgets
            per_question.append((q, [ev for tid in ctx for ev in ctx[tid]]))

        if self.llm.is_mock:
            maps = []
            for q, evs in per_question:
                if not evs:
                    continue
                firsts = list(OrderedDict((ev.transcript_id, ev) for ev in reversed(evs)).values())[::-1]
                n = len({ev.transcript_id for ev in evs})
                maps.append(Finding(
                    id=f"map_{q.id}", title=q.text, question_id=q.id, classification="evidence_map",
                    summary=f"{n} of {len(experts)} experts address this topic. Compare their statements below; "
                            "configure an LLM provider to classify common views, differences in emphasis and "
                            "disagreements automatically.",
                    experts=[expert_map[ev.transcript_id] for ev in firsts],
                    evidence=_sort_evidence(firsts),
                ))
            return InsightsResponse(**base, themes=[], differences=[], disagreements=[], by_question=maps,
                                    meta=self._meta(experts_considered=len(experts)))

        dropped: list[str] = []
        notices: list[str] = []
        findings: list[Finding] = []
        cached_all, calls = True, 0
        for q, evs in per_question:
            if not evs:
                continue
            allowed = {ev.id for ev in evs}
            try:
                raw, cached = self._call("findings", findings_prompt(q.text, evs, experts), LLMFindings)
            except LLMError as exc:
                notices.append(f"{q.id}: LLM unavailable ({exc})")
                continue
            cached_all &= cached
            calls += 1
            for i, f in enumerate(raw.findings):
                fnd = self._finding(f"{q.id}_f{i + 1}", f.title, f.summary, f.classification, f.evidence,
                                    allowed, expert_map, q.id, dropped)
                if fnd:
                    findings.append(fnd)

        themes: list[Finding] = []
        if findings:
            allowed_all = {ev.id for f in findings for ev in f.evidence}
            # The themes pass reasons over the findings, not the raw transcript: it needs each finding's meaning
            # and the evidence IDs it may reuse, not the quotes again (they were supplied in the per-question
            # pass). Resending them makes this the largest request of the page and, on small per-minute budgets,
            # it is rejected outright (HTTP 413).
            block = "\n".join(
                f"[{f.id}] ({f.classification}) {f.title}: {f.summary}\n"
                f"  evidence: " + ", ".join(f"{ev.id} ({ev.expert_name}, {ev.market})" for ev in f.evidence)
                for f in findings
            )
            try:
                raw_t, cached = self._call("themes", themes_prompt(block, experts), LLMFindings)
                cached_all &= cached
                calls += 1
                for i, t in enumerate(raw_t.findings):
                    fnd = self._finding(f"theme_{i + 1}", t.title, t.summary, t.classification, t.evidence,
                                        allowed_all, expert_map, None, dropped)
                    if fnd:
                        themes.append(fnd)
            except LLMError as exc:
                notices.append(f"themes: LLM unavailable ({exc})")

        return InsightsResponse(
            **base,
            themes=themes,
            differences=[f for f in findings if f.classification == "difference_in_emphasis"],
            disagreements=[f for f in findings if f.classification == "disagreement"],
            by_question=findings,
            meta=self._meta(cached=cached_all and calls > 0, experts_considered=len(experts),
                            dropped_citations=dropped, notices=notices),
        )


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
