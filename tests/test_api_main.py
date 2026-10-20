"""The app shell: startup that never crashes, request ids, error envelope, CORS.

No database, no vector store, no API key: startup runs against empty temporary
paths or a replaced warm-up, and errors are raised from throwaway routes added
to a fresh app.
"""

import re
from pathlib import Path

import httpx
import openai
import pytest
import structlog
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.errors import (
    ERROR_MESSAGES,
    NotReadyError,
    QueryTimeoutError,
    RateLimitedError,
    ServiceDegradedError,
)
from src.api.main import REQUEST_ID_HEADER, create_app, timeouts_consistent
from src.api.models import ErrorCode, QueryRequest
from src.api.readiness import Readiness
from src.rag.figure_guard import FigureGuardError
from src.storage.vector_store import VectorStoreError
from src.utils.config import settings

# --- Startup ------------------------------------------------------------------
# What readiness checks is tested in test_readiness.py. Here: the lifespan runs
# it once, stores it on app.state, and never stops the server.


def test_lifespan_stores_a_warmed_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    warmed: list[Readiness] = []

    async def fake_warm(self: Readiness) -> None:
        self.failures = {}
        warmed.append(self)

    monkeypatch.setattr(Readiness, "warm", fake_warm)
    app = create_app()
    with TestClient(app):                       # "with" runs the lifespan
        assert app.state.readiness is warmed[0]
        assert app.state.readiness.ready


def test_server_starts_when_nothing_is_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The real warm-up against empty paths: the app starts, not ready."""
    monkeypatch.setattr(settings, "SQLITE_DB_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setattr(settings, "CHROMA_DB_PATH", str(tmp_path / "chroma"))
    app = create_app()
    with TestClient(app) as client:
        assert not app.state.readiness.ready
        assert client.get("/no-such-path").status_code == 404     # still serving


@pytest.mark.parametrize(
    ("request_s", "per_call_s", "retries", "expected"),
    [(30, 10, 1, True), (30, 30, 0, False), (30, 10, 2, False), (30, 14.9, 1, True)],
)
def test_timeout_rule(request_s: float, per_call_s: float, retries: int, expected: bool) -> None:
    """Equal timeouts fail the rule: the per-call limit could never fire first."""
    assert timeouts_consistent(request_s, per_call_s, retries) is expected


# --- Request id middleware -------------------------------------------------------


def _app_with_probe() -> FastAPI:
    """A fresh app (no lifespan run) with a route reporting the logging context."""
    app = create_app()

    @app.get("/probe")
    async def probe() -> dict[str, str | None]:
        return {"bound": structlog.contextvars.get_contextvars().get("request_id")}

    return app


def test_request_id_is_generated_and_returned() -> None:
    response = TestClient(_app_with_probe()).get("/probe")
    request_id = response.headers[REQUEST_ID_HEADER]
    assert re.fullmatch(r"[0-9a-f]{32}", request_id)
    assert response.json()["bound"] == request_id      # every log line carries it


def test_safe_incoming_request_id_is_reused() -> None:
    response = TestClient(_app_with_probe()).get(
        "/probe", headers={REQUEST_ID_HEADER: "demo-run.42"}
    )
    assert response.headers[REQUEST_ID_HEADER] == "demo-run.42"


@pytest.mark.parametrize("bad_id", ["has spaces", "x" * 65, "semi;colon"])
def test_unsafe_incoming_request_id_is_replaced(bad_id: str) -> None:
    """The id is written into logs, so a client cannot choose arbitrary text."""
    response = TestClient(_app_with_probe()).get("/probe", headers={REQUEST_ID_HEADER: bad_id})
    assert response.headers[REQUEST_ID_HEADER] != bad_id


def test_logging_context_does_not_leak_between_requests() -> None:
    client = TestClient(_app_with_probe())
    first = client.get("/probe").headers[REQUEST_ID_HEADER]
    second = client.get("/probe")
    assert second.json()["bound"] == second.headers[REQUEST_ID_HEADER] != first


def test_unknown_path_is_404_with_request_id() -> None:
    response = TestClient(create_app()).get("/no-such-path")
    assert response.status_code == 404
    assert REQUEST_ID_HEADER in response.headers


# --- Error envelope ---------------------------------------------------------------

_CONNECTION_ERROR = openai.APIConnectionError(
    request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
)
_TIMEOUT_ERROR = openai.APITimeoutError(
    request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
)

_CASES: list[tuple[Exception, int, ErrorCode]] = [
    (NotReadyError("index missing"), 503, ErrorCode.NOT_READY),
    (ServiceDegradedError("all tasks failed"), 503, ErrorCode.SERVICE_DEGRADED),
    (QueryTimeoutError("30 s"), 504, ErrorCode.TIMEOUT),
    (RateLimitedError("budget"), 429, ErrorCode.RATE_LIMITED),
    (VectorStoreError("chroma closed"), 503, ErrorCode.NOT_READY),
    (FigureGuardError("db unreadable"), 503, ErrorCode.NOT_READY),
    (_CONNECTION_ERROR, 503, ErrorCode.LLM_UNAVAILABLE),
    (_TIMEOUT_ERROR, 503, ErrorCode.LLM_UNAVAILABLE),
    (RuntimeError("unexpected"), 500, ErrorCode.INTERNAL_ERROR),
]


def _client_raising(exc: Exception) -> TestClient:
    """A client for a fresh app whose /boom route raises exc."""
    app = create_app()

    @app.get("/boom")
    async def boom() -> None:
        raise exc

    # Starlette re-raises unhandled exceptions after answering, for server
    # logs; the client must not turn that into a test failure.
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize(("exc", "status_code", "code"), _CASES)
def test_each_error_maps_to_one_envelope(exc: Exception, status_code: int, code: ErrorCode) -> None:
    response = _client_raising(exc).get("/boom")
    body = response.json()
    assert response.status_code == status_code
    assert set(body) == {"error", "message", "request_id"}
    assert body["error"] == code.value
    assert body["message"] == ERROR_MESSAGES[code]
    assert body["request_id"] == response.headers[REQUEST_ID_HEADER]


@pytest.mark.parametrize(("exc", "status_code", "code"), _CASES)
def test_only_503_tells_the_client_when_to_retry(
    exc: Exception, status_code: int, code: ErrorCode
) -> None:
    """Retry-After is the readiness re-try interval, the earliest possible recovery."""
    response = _client_raising(exc).get("/boom")
    if status_code == 503:
        assert response.headers["Retry-After"] == "5"
    else:
        assert "Retry-After" not in response.headers


def test_exception_text_never_reaches_the_client() -> None:
    """Paths, SQL and keys in an exception message stay in the logs."""
    secret = "sk-test-123 at /home/user/db/metadata.db"
    response = _client_raising(RuntimeError(secret)).get("/boom")
    assert "sk-test" not in response.text
    assert "metadata.db" not in response.text
    assert "Traceback" not in response.text


def test_validation_error_keeps_fastapi_format() -> None:
    """422 is not wrapped: Swagger and clients understand FastAPI's format."""
    app = create_app()

    @app.post("/echo")
    async def echo(body: QueryRequest) -> dict[str, str]:
        return {"question": body.question}

    response = TestClient(app).post("/echo", json={"question": "   "})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "question"]
    assert REQUEST_ID_HEADER in response.headers


# --- CORS -------------------------------------------------------------------------


def test_no_cross_origin_access_by_default() -> None:
    """CORS_ALLOW_ORIGINS is empty, so a page on another site cannot read responses."""
    response = TestClient(_app_with_probe()).get(
        "/probe", headers={"Origin": "http://other-site.example"}
    )
    assert "access-control-allow-origin" not in response.headers
