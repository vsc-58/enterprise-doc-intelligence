"""The three OpenAI clients carry the per-call limits: 10 s timeout, 1 retry.

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


def test_limits_do_not_change_the_prompt_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Timeouts change how long we wait, not what the model is asked.

    Compared with and without different limits rather than against fixed hashes:
    which hash is the measured one is test_prompt_versions' job.
    """
    before = (planner.planner_version(), chain.chain_version())
    monkeypatch.setattr(settings, "OPENAI_TIMEOUT_SECONDS", 99.0)
    monkeypatch.setattr(settings, "OPENAI_MAX_RETRIES", 7)
    assert (planner.planner_version(), chain.chain_version()) == before


def test_query_model_is_part_of_both_prompt_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    """A different query model must read as a different version, so a result file
    measured on one snapshot cannot vouch for another."""
    before = (planner.planner_version(), chain.chain_version())
    monkeypatch.setattr(settings, "QUERY_MODEL", "some-other-snapshot")
    after = (planner.planner_version(), chain.chain_version())
    assert after[0] != before[0] and after[1] != before[1]


def test_extraction_model_is_not_part_of_the_query_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The extractor has its own model setting; changing it leaves the query path alone."""
    before = (planner.planner_version(), chain.chain_version())
    monkeypatch.setattr(settings, "EXTRACTION_MODEL", "some-other-model")
    assert (planner.planner_version(), chain.chain_version()) == before


def test_query_clients_use_the_pinned_query_model(fresh_clients: None) -> None:
    assert planner._get_llm().model_name == settings.QUERY_MODEL
    assert chain._get_llm().model_name == settings.QUERY_MODEL
