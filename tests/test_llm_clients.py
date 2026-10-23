"""The three OpenAI clients carry the D54 per-call limits.

Constructing a client makes no network call, so this runs with the test key.
Each module caches its client; the cache is cleared before and restored after.
"""

from collections.abc import Iterator

import pytest

from src.query import planner
from src.rag import chain
from src.storage import vector_store
from src.utils.config import settings


@pytest.fixture
def fresh_clients(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Build each client from current settings; leave the module caches as they were."""
    monkeypatch.setattr(planner, "_llm", None)
    monkeypatch.setattr(chain, "_llm", None)
    vector_store._embeddings.cache_clear()
    yield
    vector_store._embeddings.cache_clear()


def test_planner_client_has_the_per_call_limits(fresh_clients: None) -> None:
    llm = planner._get_llm()
    assert llm.request_timeout == settings.OPENAI_TIMEOUT_SECONDS
    assert llm.max_retries == settings.OPENAI_MAX_RETRIES


def test_chain_client_has_the_per_call_limits(fresh_clients: None) -> None:
    llm = chain._get_llm()
    assert llm.request_timeout == settings.OPENAI_TIMEOUT_SECONDS
    assert llm.max_retries == settings.OPENAI_MAX_RETRIES


def test_embedding_client_has_the_per_call_limits(fresh_clients: None) -> None:
    embeddings = vector_store._embeddings()
    assert embeddings.request_timeout == settings.OPENAI_TIMEOUT_SECONDS
    assert embeddings.max_retries == settings.OPENAI_MAX_RETRIES


def test_limits_do_not_change_the_prompt_versions() -> None:
    """Timeouts change how long we wait, not what the model is asked (5C versions)."""
    assert planner.planner_version().startswith("d4bc8e36")
    assert chain.chain_version().startswith("b1d8ee5c")
