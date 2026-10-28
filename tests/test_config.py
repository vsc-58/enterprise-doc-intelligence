"""The OpenAI key is hidden when settings are printed, and still reaches the clients.

Settings are built from explicit values with no .env file, so nothing here
reads the developer's real key.
"""

import pytest

from src.query import planner
from src.rag import chain
from src.storage import vector_store
from src.utils.config import Settings, settings

FAKE_KEY = "sk-test-0123456789abcdef"


def test_key_never_appears_when_settings_are_printed() -> None:
    printed = Settings(_env_file=None, OPENAI_API_KEY=FAKE_KEY, EDGAR_IDENTITY="Test test@example.com")
    for text in (repr(printed), str(printed), str(printed.model_dump()), printed.model_dump_json()):
        assert FAKE_KEY not in text
    assert printed.OPENAI_API_KEY.get_secret_value() == FAKE_KEY


def test_clients_receive_the_real_key_not_the_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(planner, "_llm", None)
    monkeypatch.setattr(chain, "_llm", None)
    vector_store._embeddings.cache_clear()
    try:
        expected = settings.OPENAI_API_KEY.get_secret_value()
        for client in (planner._get_llm(), chain._get_llm(), vector_store._embeddings()):
            assert client.openai_api_key.get_secret_value() == expected
            assert expected != "**********"
    finally:
        vector_store._embeddings.cache_clear()


def test_default_extraction_strategy_is_a_known_strategy() -> None:
    from src.extraction.prompts import STRATEGIES

    assert settings.EXTRACTION_STRATEGY in STRATEGIES
    assert settings.EXTRACTION_STRATEGY == "S3"   # the strategy the stored figures came from
