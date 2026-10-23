"""FastAPI application: app factory, startup, middleware and error handling.

Builds the app: lifespan startup, request-id and timing middleware, CORS, and
the exception handlers that turn every error into the one error envelope
(docs/api_contract.md §7). Routes live in src/api/routes/ and are registered
here.

Run locally:  uvicorn src.api.main:app --reload            (port 8000)
One worker only: Chroma's local store is single-process and the caches the
readiness warm-up fills live per process (docs/api_contract.md §1).

Startup never crashes on a missing store. The lifespan warms the data
parts through src/api/readiness.py and stores the state on app.state.readiness;
a missing part is logged and the server starts anyway. Routes declare what
they need through src/api/dependencies.py, which re-tries readiness while not
ready and otherwise answers 503 not_ready with Retry-After. On a hosted Space
this shows the cause in a browser instead of a crash loop.
"""

import math
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import openai
import structlog
from fastapi import APIRouter, FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError

from src.api.context import REQUEST_ID_HEADER, elapsed_ms
from src.api.errors import ERROR_MESSAGES, ApiError
from src.api.limits import UsageLimiter
from src.api.models import OPENAPI_SCHEMA_DESCRIPTIONS, ErrorBody, ErrorCode
from src.api.readiness import Readiness
from src.api.routes import demo as demo_routes
from src.api.routes import documents as documents_routes
from src.api.routes import export as export_routes
from src.api.routes import health as health_routes
from src.api.routes import query as query_routes
from src.rag.figure_guard import FigureGuardError
from src.storage.vector_store import VectorStoreError
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# An incoming request id is reused only if it is short and plain. Anything else
# is replaced: the id is written into every log line, so an attacker-chosen
# value with newlines or control characters would let a client forge log lines.
_SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")

# OpenAI failures that mean "the model service is unavailable" -> 503. A
# BadRequestError is deliberately absent: it means we sent something invalid,
# which is our bug -> 500. APITimeoutError is a subclass of APIConnectionError.
_LLM_UNAVAILABLE: tuple[type[openai.OpenAIError], ...] = (
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.InternalServerError,
)

# Store failures that mean "a data store is unavailable" -> 503 not_ready.
# OperationalError only (file missing, locked, unreadable): other SQLAlchemy
# errors are programming errors and stay 500.
_STORE_UNAVAILABLE: tuple[type[Exception], ...] = (
    VectorStoreError,
    FigureGuardError,
    OperationalError,
)

# One router per route file, in the order Swagger lists them.
_ROUTERS: list[APIRouter] = [
    query_routes.router,
    documents_routes.router,
    export_routes.router,
    health_routes.router,
    demo_routes.router,
]

_DESCRIPTION = (
    "Question answering over 20 SEC 10-K filings. Each question is planned into "
    "1-3 tasks: exact figures come from a SQL lookup over LLM-extracted values, "
    "narrative answers from cited retrieval over the filing text. Refusals are "
    "normal answers (HTTP 200) with a typed reason."
)


# --- Startup ------------------------------------------------------------------


def timeouts_consistent(request_s: float, per_call_s: float, retries: int) -> bool:
    """Check that one OpenAI call, with all its retries, ends before the request does.

    The planner call runs in a worker thread that a request timeout cannot stop,
    so only the per-call limit bounds it. If the rule fails, a timed-out request
    can leave that thread running, and paying, after the user got a 504.

    Args:
        request_s: QUERY_TIMEOUT_SECONDS.
        per_call_s: OPENAI_TIMEOUT_SECONDS.
        retries: OPENAI_MAX_RETRIES.

    Returns:
        True when per_call_s * (retries + 1) < request_s.
    """
    return per_call_s * (retries + 1) < request_s


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Run startup before the first request and shutdown after the last.

    Args:
        app: The application.

    Yields:
        Control to the server while it serves requests.
    """
    if not timeouts_consistent(
        settings.QUERY_TIMEOUT_SECONDS,
        settings.OPENAI_TIMEOUT_SECONDS,
        settings.OPENAI_MAX_RETRIES,
    ):
        logger.warning(
            "timeout_rule_violated",
            query_timeout_s=settings.QUERY_TIMEOUT_SECONDS,
            openai_timeout_s=settings.OPENAI_TIMEOUT_SECONDS,
            openai_max_retries=settings.OPENAI_MAX_RETRIES,
            consequence="a timed-out request can leave a planner call running",
        )

    readiness = Readiness()
    await readiness.warm()
    app.state.readiness = readiness
    limiter: UsageLimiter = app.state.limiter
    logger.info(
        "api_started", ready=readiness.ready, unavailable=sorted(readiness.failures),
        rate_limit_per_minute=limiter.per_minute, daily_query_budget=limiter.daily_budget,
    )
    yield
    logger.info("api_stopped")


# --- Middleware ---------------------------------------------------------------


async def request_context(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Give every request an id and a timer, and log one line when it ends.

    The id is reused from an incoming X-Request-ID header when it is safe, else
    generated. It is bound to structlog's context so every log line written
    while serving the request carries it, stored on request.state for handlers
    and error bodies, and returned in the X-Request-ID response header.

    Args:
        request: The incoming request.
        call_next: The rest of the app.

    Returns:
        The response, with the X-Request-ID header set.
    """
    incoming = request.headers.get(REQUEST_ID_HEADER, "")
    request_id = incoming if _SAFE_REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex
    request.state.request_id = request_id
    request.state.started_at = time.perf_counter()
    structlog.contextvars.bind_contextvars(request_id=request_id)

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR   # if call_next raises
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
    finally:
        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status=status_code,
            latency_ms=elapsed_ms(request),
        )
        structlog.contextvars.clear_contextvars()


# --- Exception handlers -------------------------------------------------------


def error_response(
    request: Request, code: ErrorCode, status_code: int, retry_after_s: int | None = None
) -> JSONResponse:
    """Build the one error envelope (docs/api_contract.md §7).

    The message is the fixed sentence for the code, never exception text.

    Args:
        request: The failed request.
        code: The error code.
        status_code: The HTTP status.
        retry_after_s: Seconds until a retry can succeed, when the error knows
            (429: until a slot frees or the budget resets). Otherwise 503s get
            the readiness re-try interval and other statuses no header.

    Returns:
        A JSON response carrying ErrorBody and the X-Request-ID header.
    """
    request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex
    body = ErrorBody(error=code, message=ERROR_MESSAGES[code], request_id=request_id)
    headers = {REQUEST_ID_HEADER: request_id}
    if retry_after_s is not None:
        headers["Retry-After"] = str(retry_after_s)
    elif status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
        # When to try again: the readiness re-try interval, the earliest moment
        # a not-ready server can have recovered. Whole seconds, per HTTP.
        headers["Retry-After"] = str(math.ceil(settings.READINESS_RETRY_SECONDS))
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json"),
        headers=headers,
    )


async def handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
    """Map an error the API raised on purpose to its status and code.

    Args:
        request: The failed request.
        exc: An ApiError subclass; registered only for ApiError.

    Returns:
        The error envelope.
    """
    logger.warning(
        "api_error", error=exc.code.value, status=exc.status_code, detail=str(exc),
        request_id=getattr(request.state, "request_id", None),
    )
    return error_response(request, exc.code, exc.status_code, exc.retry_after_s)


async def handle_llm_unavailable(request: Request, exc: Exception) -> JSONResponse:
    """OpenAI unreachable, timed out, rate-limited or rejecting the key -> 503.

    Reaches here only from the planner call: failures inside the executor are
    caught per task and reported as failed parts (docs/api_contract.md §7).

    Args:
        request: The failed request.
        exc: An OpenAI error from _LLM_UNAVAILABLE.

    Returns:
        The error envelope, 503 llm_unavailable.
    """
    logger.error(
        "llm_unavailable", error_type=type(exc).__name__, error=str(exc),
        request_id=getattr(request.state, "request_id", None),
    )
    return error_response(
        request, ErrorCode.LLM_UNAVAILABLE, status.HTTP_503_SERVICE_UNAVAILABLE
    )


async def handle_store_unavailable(request: Request, exc: Exception) -> JSONResponse:
    """SQLite or Chroma unavailable during a request -> 503 not_ready.

    Args:
        request: The failed request.
        exc: An exception from _STORE_UNAVAILABLE.

    Returns:
        The error envelope, 503 not_ready.
    """
    logger.error(
        "store_unavailable", error_type=type(exc).__name__, error=str(exc),
        request_id=getattr(request.state, "request_id", None),
    )
    return error_response(request, ErrorCode.NOT_READY, status.HTTP_503_SERVICE_UNAVAILABLE)


async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    """Anything else -> 500 with a fixed message; the traceback goes to the log.

    Starlette runs this outside the request middleware, so the request id is
    passed explicitly rather than read from the logging context.

    Args:
        request: The failed request.
        exc: The unhandled exception.

    Returns:
        The error envelope, 500 internal_error.
    """
    logger.error(
        "unhandled_exception", error_type=type(exc).__name__,
        request_id=getattr(request.state, "request_id", None), exc_info=exc,
    )
    return error_response(
        request, ErrorCode.INTERNAL_ERROR, status.HTTP_500_INTERNAL_SERVER_ERROR
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach every handler. 422 keeps FastAPI's own validation format.

    Args:
        app: The application.
    """
    app.add_exception_handler(ApiError, handle_api_error)
    for error_type in _LLM_UNAVAILABLE:
        app.add_exception_handler(error_type, handle_llm_unavailable)
    for error_type in _STORE_UNAVAILABLE:
        app.add_exception_handler(error_type, handle_store_unavailable)
    app.add_exception_handler(Exception, handle_unexpected)


# --- App ----------------------------------------------------------------------


def use_public_schema_descriptions(app: FastAPI) -> None:
    """Serve API-owned descriptions for planner enums in the OpenAPI schema.

    The API reuses planner enums (Intent, UnsupportedReason) whose docstrings
    are planner prompt text and stay frozen at the measured prompt version.
    FastAPI copies a class docstring into the schema, so Swagger would show
    that text. This replaces the description of each listed component once,
    when the schema is first built; the classes themselves are not changed.

    Args:
        app: The application whose openapi() is wrapped.
    """
    build_schema = app.openapi

    def openapi() -> dict[str, Any]:
        """Build the schema once, with the public descriptions applied, and cache it."""
        schema = app.openapi_schema
        if schema is None:
            schema = build_schema()   # FastAPI caches this dict on the app
            components = schema.get("components", {}).get("schemas", {})
            for name, text in OPENAPI_SCHEMA_DESCRIPTIONS.items():
                if name in components:
                    components[name]["description"] = text
        return schema

    app.openapi = openapi  # type: ignore[method-assign]


def create_app() -> FastAPI:
    """Build the application.

    A factory rather than only a module-level object so tests can build a fresh
    app, add throwaway routes, and choose whether the lifespan runs.

    Middleware order: the last added runs first. CORS is outermost, so a
    browser's preflight OPTIONS request is answered before anything else.

    Returns:
        The configured FastAPI app.
    """
    app = FastAPI(
        title="Enterprise Document Intelligence API",
        description=_DESCRIPTION,
        lifespan=lifespan,
    )
    app.middleware("http")(request_context)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ALLOW_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", REQUEST_ID_HEADER],
        expose_headers=[REQUEST_ID_HEADER],
    )
    register_exception_handlers(app)
    for router in _ROUTERS:
        app.include_router(router)
    use_public_schema_descriptions(app)
    # Built here, not in the lifespan, so tests that skip the lifespan still
    # have it. Both limits are off unless set.
    app.state.limiter = UsageLimiter.from_settings()
    return app


app = create_app()
