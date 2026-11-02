"""The served-model guard and the model settings it protects.

The probe call is always stubbed: no network, no API key needed beyond the
test placeholder.
"""

import httpx
import openai
import pytest
from pydantic import ValidationError

from src.eval import served_model as sm
from src.utils.config import Settings, settings


def _connection_error() -> openai.APIConnectionError:
    """An OpenAI error of the kind a failed probe call raises."""
    return openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com"))


def test_require_served_returns_the_name_when_it_matches() -> None:
    assert sm.require_served("snap-2024", probe=lambda name: name) == "snap-2024"


def test_require_served_refuses_an_alias() -> None:
    with pytest.raises(sm.ServedModelMismatch) as info:
        sm.require_served("gpt-4o-mini", probe=lambda name: "gpt-4o-mini-2024-07-18")
    assert info.value.served == "gpt-4o-mini-2024-07-18"
    assert "alias" in str(info.value)


@pytest.mark.parametrize("reported", [None, ""])
def test_require_served_refuses_a_response_without_a_model_name(reported: str | None) -> None:
    with pytest.raises(sm.ServedModelError):
        sm.require_served("snap", probe=lambda name: reported)


def test_require_served_wraps_a_failed_call() -> None:
    def failing(name: str) -> str:
        raise _connection_error()

    with pytest.raises(sm.ServedModelError) as info:
        sm.require_served("snap", probe=failing)
    assert not isinstance(info.value, sm.ServedModelMismatch)


def test_check_served_or_log_gives_none_instead_of_raising() -> None:
    assert sm.check_served_or_log("query", "a", probe=lambda name: "b") is None
    assert sm.check_served_or_log("query", "a", probe=lambda name: "a") == "a"


def test_served_model_reads_the_name_from_response_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        response_metadata = {"model_name": "snap-served"}

    class FakeChat:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def invoke(self, prompt: str) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(sm, "ChatOpenAI", FakeChat)
    assert sm.served_model("snap") == "snap-served"


def test_default_query_model_is_a_dated_snapshot_and_extraction_stays_the_alias() -> None:
    # The query path is pinned; extraction keeps the name its cache and records use.
    assert settings.QUERY_MODEL == "gpt-4o-mini-2024-07-18"
    assert settings.EXTRACTION_MODEL == "gpt-4o-mini"


def test_retired_model_setting_fails_with_a_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    with pytest.raises(ValidationError) as info:
        Settings()
    assert "QUERY_MODEL" in str(info.value) and "EXTRACTION_MODEL" in str(info.value)


def test_retired_model_setting_in_a_dotenv_file_fails_with_a_message(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=x\nEDGAR_IDENTITY=a b@c.d\nOPENAI_MODEL=gpt-4o\n",
                   encoding="utf-8")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with pytest.raises(ValidationError) as info:
        Settings(_env_file=env)
    assert "OPENAI_MODEL is no longer used" in str(info.value)


def test_new_model_settings_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUERY_MODEL", "gpt-4o-mini-2099-01-01")
    monkeypatch.setenv("EXTRACTION_MODEL", "other")
    loaded = Settings()
    assert loaded.QUERY_MODEL == "gpt-4o-mini-2099-01-01"
    assert loaded.EXTRACTION_MODEL == "other"


@pytest.mark.parametrize("extra", ["", "OPENAI_MODEL=gpt-4o\n"])
def test_settings_errors_never_print_the_api_key(
    tmp_path, monkeypatch: pytest.MonkeyPatch, extra: str
) -> None:
    # A missing required setting, or a retired one: either error must not echo
    # the other settings, which include the key.
    env = tmp_path / ".env"
    env.write_text(f"OPENAI_API_KEY=sk-should-not-appear\n{extra}", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("EDGAR_IDENTITY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with pytest.raises(ValidationError) as info:
        Settings(_env_file=env)
    assert "sk-should-not-appear" not in str(info.value)
