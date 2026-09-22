"""Regression questions against the real case transcripts in /transcripts.

Runs automatically once the case transcripts are present (skipped otherwise). Uses deterministic
extractive mode unless GOLDEN_USE_LLM=1, in which case the configured provider from .env is used.
"""

import dataclasses
import os

import pytest

from app.config import get_settings
from app.eval.golden import dataset_matches, evaluate_case, load_golden
from app.services.state import AppState

GOLDEN = load_golden()


@pytest.fixture(scope="module")
def real_state(tmp_path_factory):
    base = get_settings()
    overrides = {"database_path": tmp_path_factory.mktemp("golden") / "golden.db"}
    if os.getenv("GOLDEN_USE_LLM") != "1":
        overrides.update(llm_provider="mock", embedding_provider="local")
    state = AppState(dataclasses.replace(base, **overrides))
    state.refresh()
    if not dataset_matches(state, GOLDEN):
        pytest.skip("Case transcripts (France/Germany/UK) not present in /transcripts")
    raw = {t.id: state.db.query_one("SELECT raw_text FROM transcripts WHERE id = ?", (t.id,))["raw_text"]
           for t in state.store.list_transcripts(include_failed=False)}
    yield state, raw
    state.db.close()


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=[c["id"] for c in GOLDEN["cases"]])
def test_golden_case(real_state, case):
    state, raw = real_state
    result = evaluate_case(state, case, GOLDEN["market_aliases"], raw)
    assert result.passed, f"{case['id']}: {result.failures}\nanswer: {result.answer}\nsources: {result.sources}"
