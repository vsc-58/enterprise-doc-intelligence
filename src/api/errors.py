# src/api/errors.py
# Module: API error types
# Purpose: The exceptions the API raises on purpose, each tied to one status code
#          and one ErrorCode, plus the fixed user-facing sentence for every code.
# Depends on: fastapi (status constants), src.api.models
#
# Raising a typed error and letting one handler in main.py build the response
# keeps every error body identical in shape (contract §7). The message a user
# sees is chosen here by code, never taken from the exception: exception text
# can carry paths, SQL or library internals, and goes to the logs only.

from fastapi import status

from src.api.models import ErrorCode

ERROR_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.NOT_READY: (
        "The service is not ready: its data stores are unavailable. Try again shortly."
    ),
    ErrorCode.LLM_UNAVAILABLE: (
        "The language model service is unavailable. Try again shortly."
    ),
    ErrorCode.SERVICE_DEGRADED: (
        "No part of the question could be answered because of an internal fault. "
        "Try again shortly."
    ),
    ErrorCode.TIMEOUT: "The question took too long to answer. Try a narrower question.",
    ErrorCode.RATE_LIMITED: "Too many requests. Try again later.",
    ErrorCode.INTERNAL_ERROR: "Something went wrong on our side.",
}


class ApiError(Exception):
    """
    Base class for errors the API turns into the error envelope.

    Subclasses fix the status code and error code. The exception's own message
    is a detail for the logs and is never sent to the client.

    Attributes:
        code: The ErrorCode in the response body.
        status_code: The HTTP status of the response.
    """

    code: ErrorCode = ErrorCode.INTERNAL_ERROR
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR


class NotReadyError(ApiError):
    """A data store or the corpus index is unavailable (startup or mid-request)."""

    code = ErrorCode.NOT_READY
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


class ServiceDegradedError(ApiError):
    """Every task in the plan failed: nothing was answered (contract §7)."""

    code = ErrorCode.SERVICE_DEGRADED
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


class QueryTimeoutError(ApiError):
    """The request passed QUERY_TIMEOUT_SECONDS."""

    code = ErrorCode.TIMEOUT
    status_code = status.HTTP_504_GATEWAY_TIMEOUT


class RateLimitedError(ApiError):
    """A per-client limit or the daily budget was hit (Phase 8; off by default)."""

    code = ErrorCode.RATE_LIMITED
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
