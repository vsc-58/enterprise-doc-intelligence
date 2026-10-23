# src/api/context.py
# Module: Per-request context helpers
# Purpose: The request-id header name, the request timer and the client
#          address, shared by the middleware in main.py and the routes.
# Depends on: fastapi
#
# Separate from main.py because main.py imports the routes: a route importing
# main.py would be a circular import.

import time

from fastapi import Request

REQUEST_ID_HEADER = "X-Request-ID"


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


def request_id_of(request: Request) -> str:
    """
    The id the middleware gave this request.

    Args:
        request: The current request.

    Returns:
        The request id, or "" if the middleware did not run.
    """
    return getattr(request.state, "request_id", "")


def client_address(request: Request) -> str:
    """
    The client's address, as the server sees it.

    Behind a proxy this is the proxy's address unless uvicorn runs with
    --proxy-headers (see src/api/limits.py). Never read from X-Forwarded-For
    here: any client can set that header.

    Args:
        request: The current request.

    Returns:
        The IP address, or "unknown" when the server did not record one.
    """
    return request.client.host if request.client else "unknown"
