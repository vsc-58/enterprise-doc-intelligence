# src/api/main.py
# Module: FastAPI application
# Purpose: Build the app: lifespan startup, request-id and timing middleware,
#          CORS, and the exception handlers that turn every error into the one
#          error envelope (docs/api_contract.md §7). Routes live in
#          src/api/routes/ and are registered here.
# Depends on: fastapi, starlette, structlog, openai, sqlalchemy, src.api.*,
#             src.query.resolve, src.rag.figure_guard, src.storage.vector_store,
#             src.utils.*
#
# Run locally:  uvicorn src.api.main:app --reload            (port 8000)
# One worker only: Chroma's local store is single-process and the caches warmed
# below live per process (contract §1).
#
# Startup never crashes on a missing store (D54). Each part is warmed on its
# own; a failure is logged and recorded in app.state.startup_errors, and the
# server starts anyway. /health reports which part failed and /query refuses
# with 503 not_ready (both in later steps). On a hosted Space this shows the
# cause in a browser instead of a crash loop.
#
# The lifespan WARMS caches that already exist at module level — load_index()'s
# _cached_index, get_store()'s locked singleton, load_guard_values()'s
# lru_cache — rather than building a second engine or client. The query path
# keeps using its own accessors; warming moves the slow first call to startup.

import asyncio
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import openai
import structlog
from fastapi import APIRouter, FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from src.api.errors import ERROR_MESSAGES, ApiError
from src.api.models import ErrorBody, ErrorCode
from src.query.resolve import load_index
from src.rag.figure_guard import FigureGuardError, load_guard_values
from src.storage.vector_store import VectorStoreError, collection_count, get_store
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

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

# Routers are added as the route files are built (steps 6 and 7).
_ROUTERS: list[APIRouter] = []

_DESCRIPTION = (
    "Question answering over 20 SEC 10-K filings. Each question is planned into "
    "1-3 tasks: exact figures come from a SQL lookup over LLM-extracted values, "
    "narrative answers from cited retrieval over the filing text. Refusals are "
    "normal answers (HTTP 200) with a typed reason."
)


# --- Startup ------------------------------------------------------------------


def timeouts_consistent(request_s: float, per_call_s: float, retries: int) -> bool:
    """
    Check the D54 timeout rule: one OpenAI call, with all its retries, must end
    before the request times out.

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


def _record_failure(app: FastAPI, part: str, error_type: str, detail: str) -> None:
    """
    Log one failed startup part and remember it for /health.

    Args:
        app: The application.
        part: Which part failed: "index", "vector_store" or "figure_guard".
        error_type: Exception class name, or a short label for a failed check.
        detail: What went wrong, for the log only.
    """
    app.state.startup_errors[part] = error_type
    logger.error("startup_part_unavailable", part=part, error_type=error_type, error=detail)


async def warm_up(app: FastAPI) -> None:
    """
    Load the corpus index and open the stores once, off the event loop.

    Each part is tried on its own, so one failure does not hide another, and
    none of them stops the server (D54).

    Args:
        app: The application. Sets app.state.index (CorpusIndex | None) and
            app.state.startup_errors (part -> exception type name).
    """
    app.state.index = None
    app.state.startup_errors = {}

    try:
        app.state.index = await asyncio.to_thread(load_index)
    except (RuntimeError, SQLAlchemyError) as exc:   # RuntimeError: empty corpus
        _record_failure(app, "index", type(exc).__name__, str(exc))

    # Opening is not enough: on a machine with no store, get_store() CREATES an
    # empty collection and succeeds (found running this step against an empty
    # checkout). The count is what proves there is something to retrieve.
    # collection_count() returns -1 when the collection cannot be read.
    try:
        await asyncio.to_thread(get_store)
        chunks = await asyncio.to_thread(collection_count)
    except VectorStoreError as exc:
        _record_failure(app, "vector_store", type(exc).__name__, str(exc))
    else:
        if chunks <= 0:
            _record_failure(
                app, "vector_store", "EmptyVectorStore",
                f"collection holds {chunks} chunks; run scripts.embed_documents",
            )

    try:
        await asyncio.to_thread(load_guard_values)
    except FigureGuardError as exc:
        _record_failure(app, "figure_guard", type(exc).__name__, str(exc))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Run startup before the first request and shutdown after the last.

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

    await warm_up(app)
    index = app.state.index
    logger.info(
        "api_started",
        ready=not app.state.startup_errors,
        documents=len(index.entries) if index is not None else None,
        unavailable=sorted(app.state.startup_errors),
    )
    yield
    logger.info("api_stopped")


# --- Middleware ---------------------------------------------------------------


def elapsed_ms(request: Request) -> int:
    """
    Milliseconds since the middleware started handling this request.

    Args:
        request: The current request.

    Returns:
        Elapsed whole milliseconds; 0 if the middleware did not run (tests that
        call a handler directly).
    """
    started = getattr(request.state, "started_at", None)
    if started is None:
        return 0
    return round((time.perf_counter() - started) * 1000)


async def request_context(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """
    Give every request an id and a timer, and log one line when it ends.

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


def error_response(request: Request, code: ErrorCode, status_code: int) -> JSONResponse:
    """
    Build the one error envelope (contract §7).

    The message is the fixed sentence for the code, never exception text.

    Args:
        request: The failed request.
        code: The error code.
        status_code: The HTTP status.

    Returns:
        A JSON response carrying ErrorBody and the X-Request-ID header.
    """
    request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex
    body = ErrorBody(error=code, message=ERROR_MESSAGES[code], request_id=request_id)
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json"),
        headers={REQUEST_ID_HEADER: request_id},
    )


async def handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
    """
    Map an error the API raised on purpose to its status and code.

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
    return error_response(request, exc.code, exc.status_code)


async def handle_llm_unavailable(request: Request, exc: Exception) -> JSONResponse:
    """
    OpenAI unreachable, timed out, rate-limited or rejecting the key -> 503.

    Reaches here only from the planner call: failures inside the executor are
    caught per task and reported as failed parts (contract §7).

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
    """
    SQLite or Chroma unavailable during a request -> 503 not_ready.

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
    """
    Anything else -> 500 with a fixed message; the traceback goes to the log.

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
    """
    Attach every handler. 422 keeps FastAPI's own validation format.

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


def create_app() -> FastAPI:
    """
    Build the application.

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
    return app


app = create_app()
