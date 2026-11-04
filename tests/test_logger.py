"""Tests for the two log formats.

The console format stays the default, so local output is unchanged. The json
format must give Google Cloud Logging one parseable object per line, with the
"severity" and "message" keys it reads. The processor chain is run directly on
an event, so the global logging configuration is not touched.
"""

import json
import logging

import pytest
import structlog
from pydantic import ValidationError

from src.utils.config import Settings
from src.utils.logger import build_processors


def _render(log_format: str, method: str, event: dict[str, object]) -> str:
    """Run one event through a format's processors, as structlog would.

    Args:
        log_format: "console" or "json".
        method: The log method name ("info", "warning", ...).
        event: The event dict, with "event" holding the event name.

    Returns:
        The rendered line.
    """
    result: object = dict(event)
    logger = logging.getLogger("tests.logger")       # add_logger_name reads .name
    for processor in build_processors(log_format):
        result = processor(logger, method, result)   # type: ignore[arg-type]
    assert isinstance(result, str)
    return result


# Required settings passed explicitly: with _env_file=None no .env is read, and
# the tests must not depend on the environment the developer or CI happens to set.
_REQUIRED = {"OPENAI_API_KEY": "sk-test-not-a-key", "EDGAR_IDENTITY": "Test test@example.com"}


def test_console_is_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    settings = Settings(_env_file=None, **_REQUIRED)   # type: ignore[call-arg, arg-type]
    assert settings.LOG_FORMAT == "console"


def test_unknown_format_is_rejected_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_FORMAT", "xml")
    with pytest.raises(ValidationError) as caught:
        Settings(_env_file=None, **_REQUIRED)   # type: ignore[call-arg, arg-type]
    # The error must be about LOG_FORMAT, not about something else missing.
    assert [error["loc"] for error in caught.value.errors()] == [("LOG_FORMAT",)]


def test_json_line_carries_severity_and_message() -> None:
    line = _render("json", "warning", {"event": "rate_limited", "client": "203.0.113.7"})
    record = json.loads(line)
    assert record["severity"] == "WARNING"
    assert record["message"] == "rate_limited"
    assert record["client"] == "203.0.113.7"
    assert "event" not in record
    assert "\x1b[" not in line            # no colour codes


def test_json_line_keeps_the_request_id() -> None:
    structlog.contextvars.bind_contextvars(request_id="abc123")
    try:
        record = json.loads(_render("json", "info", {"event": "http_request"}))
    finally:
        structlog.contextvars.clear_contextvars()
    assert record["request_id"] == "abc123"
    assert record["severity"] == "INFO"


def test_console_line_is_not_json() -> None:
    line = _render("console", "info", {"event": "api_started"})
    assert "api_started" in line
    with pytest.raises(json.JSONDecodeError):
        json.loads(line)


def test_build_processors_rejects_unknown_format() -> None:
    with pytest.raises(ValueError):
        build_processors("xml")
