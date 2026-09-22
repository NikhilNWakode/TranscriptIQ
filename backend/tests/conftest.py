from __future__ import annotations

import dataclasses
import shutil
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.config import get_settings  # noqa: E402
from app.llm.providers import LLMService  # noqa: E402
from app.services.state import AppState  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


def make_settings(tmp_path: Path, **overrides):
    base = get_settings()
    return dataclasses.replace(
        base,
        transcripts_dir=tmp_path / "transcripts",
        database_path=tmp_path / "test.db",
        llm_provider="mock",
        embedding_provider="local",
        **overrides,
    )


@pytest.fixture
def transcripts_dir(tmp_path: Path) -> Path:
    d = tmp_path / "transcripts"
    d.mkdir()
    for name in ("alpha.txt", "beta.md"):
        shutil.copy(FIXTURES / name, d / name)
    return d


@pytest.fixture
def app_state(tmp_path: Path, transcripts_dir: Path) -> AppState:
    state = AppState(make_settings(tmp_path))
    state.refresh()
    yield state
    state.db.close()


class FakeLLM(LLMService):
    """Returns pre-programmed structured outputs; records prompts for assertions."""

    provider, model, is_mock = "fake", "fake-1", False

    def __init__(self, responder):
        self.responder = responder
        self.prompts: list[str] = []

    def generate(self, system, user, schema):
        self.prompts.append(user)
        return schema.model_validate(self.responder(user, schema))


@pytest.fixture
def with_fake_llm(app_state: AppState):
    def install(responder) -> FakeLLM:
        fake = FakeLLM(responder)
        app_state.llm = fake
        app_state.research.llm = fake
        app_state.cache.clear()
        return fake

    return install
