"""Tests for telling "OpenAI credit used up" apart from "OpenAI busy".

OpenAI answers both with HTTP 429. A request-rate 429 is temporary (503
llm_unavailable, retry in 5 s). An exhausted credit or spend limit is not: it
gets 503 llm_quota_exhausted, its own sentence and Retry-After 3600, so a
visitor is not told to "try again shortly" when no retry can succeed.
No network: the errors are built locally.
"""

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from src.api.errors import ERROR_MESSAGES
from src.api.main import create_app, is_quota_exhausted
from src.api.models import ErrorCode

_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _rate_limit_error(code: str | None, error_type: str | None = None) -> openai.RateLimitError:
    """A 429 error as the OpenAI client builds it from a response body.

    Args:
        code: The body's error code.
        error_type: The body's error type.

    Returns:
        The exception the planner call would raise.
    """
    body = {"message": "limit", "code": code, "type": error_type}
    return openai.RateLimitError(
        "limit", response=httpx.Response(429, request=_REQUEST), body=body
    )


def _client_raising(exc: Exception) -> TestClient:
    """A client for a fresh app (no lifespan) whose /boom route raises exc.

    Args:
        exc: The exception to raise.

    Returns:
        A test client that does not re-raise server exceptions.
    """
    app = create_app()

    @app.get("/boom")
    async def boom() -> None:
        raise exc

    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize(
    "code",
    [
        "insufficient_quota",
        "credit_balance_exhausted",
        "project_spend_limit_exceeded",
        "organization_spend_limit_exceeded",
        "organization_usage_limit_exceeded",
    ],
)
def test_spent_credit_or_limit_gets_its_own_code(code: str) -> None:
    response = _client_raising(_rate_limit_error(code)).get("/boom")
    body = response.json()
    assert response.status_code == 503
    assert body["error"] == ErrorCode.LLM_QUOTA_EXHAUSTED.value
    assert body["message"] == ERROR_MESSAGES[ErrorCode.LLM_QUOTA_EXHAUSTED]
    assert response.headers["Retry-After"] == "3600"


def test_quota_recognised_by_type_when_code_is_missing() -> None:
    """Older responses carried the condition in "type" only."""
    assert is_quota_exhausted(_rate_limit_error(None, "insufficient_quota"))


def test_request_rate_429_stays_llm_unavailable() -> None:
    """Too many requests per minute is temporary: retry soon."""
    response = _client_raising(_rate_limit_error("rate_limit_exceeded")).get("/boom")
    assert response.json()["error"] == ErrorCode.LLM_UNAVAILABLE.value
    assert response.headers["Retry-After"] == "5"


def test_only_rate_limit_errors_can_be_quota() -> None:
    """A connection error is never read as a spent credit."""
    assert not is_quota_exhausted(openai.APIConnectionError(request=_REQUEST))


def test_quota_message_points_to_what_still_works() -> None:
    """The endpoints that need no OpenAI call are named, so the visitor has a next step."""
    message = ERROR_MESSAGES[ErrorCode.LLM_QUOTA_EXHAUSTED]
    assert "/documents" in message and "/export" in message
    assert "shortly" not in message
