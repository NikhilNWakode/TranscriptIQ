"""Golden-set evaluation.

    python -m app.eval.golden            # uses the configured LLM (or extractive mode)
    python -m app.eval.golden --json out.json

Metrics reported per case and overall:
* retrieval_ok        – expected markets/timestamps/phrases are present in the resolved sources
* citation_accuracy   – share of sources whose ID resolves to an active expert statement
* quote_accuracy      – share of sources whose text occurs verbatim in the original transcript file
* no_evidence_ok      – unsupported questions return the no-evidence response
* qualifier_warnings  – qualifier-preservation warnings raised on the answer
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field

from ..config import REPO_ROOT, get_settings
from ..models.schemas import NO_EVIDENCE_MESSAGE, AskRequest, AskResponse, Evidence
from ..retrieval.text import normalise_for_match
from ..services.state import AppState

GOLDEN_PATH = REPO_ROOT / "config" / "golden_eval.json"


@dataclass
class CaseResult:
    id: str
    question: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    citation_accuracy: float = 1.0
    quote_accuracy: float = 1.0
    qualifier_warnings: int = 0
    answer: str = ""
    sources: list[str] = field(default_factory=list)


def load_golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def canonical_market(market: str, aliases: dict[str, list[str]]) -> str | None:
    m = market.strip().lower()
    for canon, names in aliases.items():
        if m == canon.lower() or m in names or any(n in m.split() or n == m for n in names):
            return canon
    return None


def dataset_matches(state: AppState, golden: dict) -> bool:
    markets = {canonical_market(e.market, golden["market_aliases"]) for e in state.store.experts()}
    return set(golden["market_aliases"]).issubset(markets)


def _sources_for(sources: list[Evidence], market: str, aliases) -> list[Evidence]:
    if market == "*":
        return sources
    return [s for s in sources if canonical_market(s.market, aliases) == market]


def evaluate_case(state: AppState, case: dict, aliases: dict, raw_files: dict[str, str]) -> CaseResult:
    resp: AskResponse = state.research.ask(AskRequest(question=case["question"]))
    checks = case["checks"]
    res = CaseResult(id=case["id"], question=case["question"], passed=True, answer=resp.answer,
                     sources=[f"{s.expert_name} ({s.market}) {s.timestamp}" for s in resp.sources],
                     qualifier_warnings=len(resp.qualifier_warnings))
    fail = res.failures.append

    if checks.get("insufficient_evidence"):
        if not resp.insufficient_evidence or resp.answer != NO_EVIDENCE_MESSAGE:
            fail("expected the no-evidence response")
        res.passed = not res.failures
        return res
    if resp.insufficient_evidence:
        fail("unexpected no-evidence response")

    # citation + quote accuracy (backend truth)
    if resp.sources:
        valid = [s for s in resp.sources if state.store.get_one(s.id) is not None and s.speaker_type == "expert"]
        res.citation_accuracy = len(valid) / len(resp.sources)
        verbatim = [s for s in resp.sources
                    if normalise_for_match(s.text) in normalise_for_match(raw_files.get(s.transcript_id, ""))]
        res.quote_accuracy = len(verbatim) / len(resp.sources)
        if res.citation_accuracy < 1:
            fail("invalid citation returned")
        if res.quote_accuracy < 1:
            fail("quote not verbatim in source file")

    got_markets = {canonical_market(s.market, aliases) for s in resp.sources}
    for m in checks.get("source_markets", []):
        if m not in got_markets:
            fail(f"no source from {m}")
    for spec in checks.get("source_timestamps", []):
        if not any(s.timestamp == spec["timestamp"] for s in _sources_for(resp.sources, spec["market"], aliases)):
            fail(f"missing {spec['market']} source at {spec['timestamp']}")
    for spec in checks.get("source_text_any", []):
        texts = " ".join(s.text.lower() for s in _sources_for(resp.sources, spec["market"], aliases))
        if not any(p.lower() in texts for p in spec["any"]):
            fail(f"{spec['market']} sources lack any of {spec['any']}")
    for spec in checks.get("source_text_all", []):
        texts = " ".join(s.text.lower() for s in _sources_for(resp.sources, spec["market"], aliases))
        # each entry is a phrase, or a list of alternatives (e.g. ["6", "six"])
        alts = [p if isinstance(p, list) else [p] for p in spec["all"]]
        missing = [a for a in alts if not any(x.lower() in texts for x in a)]
        if missing:
            fail(f"{spec['market']} sources lack {missing}")
    if checks.get("min_experts") and len({s.transcript_id for s in resp.sources}) < checks["min_experts"]:
        fail(f"fewer than {checks['min_experts']} experts cited")
    # forbidden phrases apply to the answer's own claims, not to verbatim source quotes inside it
    own_words = re.sub(r"“[^”]*”", " ", resp.answer).lower()
    for phrase in checks.get("answer_forbidden", []):
        if phrase.lower() in own_words:
            fail(f"answer contains forbidden phrase '{phrase}'")
    if resp.meta.mode == "llm":
        for phrase in checks.get("answer_required_if_llm", []):
            if phrase.lower() not in resp.answer.lower():
                fail(f"answer lacks '{phrase}'")
    res.passed = not res.failures
    return res


def run(state: AppState | None = None) -> list[CaseResult]:
    state = state or AppState(get_settings())
    state.refresh()
    golden = load_golden()
    if not dataset_matches(state, golden):
        raise SystemExit("The indexed transcripts do not cover the golden-set markets; add the case transcripts first.")
    raw_files = {t.id: state.db.query_one("SELECT raw_text FROM transcripts WHERE id = ?", (t.id,))["raw_text"]
                 for t in state.store.list_transcripts(include_failed=False)}
    return [evaluate_case(state, c, golden["market_aliases"], raw_files) for c in golden["cases"]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", help="write results to this file")
    args = parser.parse_args(argv)
    results = run()
    for r in results:
        print(f"[{'PASS' if r.passed else 'FAIL'}] {r.id:22s} citations={r.citation_accuracy:.0%} "
              f"quotes={r.quote_accuracy:.0%} qualifier_warnings={r.qualifier_warnings}")
        for f in r.failures:
            print(f"        - {f}")
    passed = sum(r.passed for r in results)
    print(f"\n{passed}/{len(results)} golden cases passed")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump([r.__dict__ for r in results], fh, indent=2, ensure_ascii=False)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
