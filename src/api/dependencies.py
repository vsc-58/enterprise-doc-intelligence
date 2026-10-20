# src/api/dependencies.py
# Module: Route dependencies
# Purpose: What each endpoint needs to run, and the readiness check that guards
#          it. A needed part that is missing raises NotReadyError (-> 503
#          not_ready, Retry-After) after re-trying readiness if due.
# Depends on: fastapi, src.api.errors, src.api.readiness, src.query.resolve
#
# Per-endpoint needs (contract §6):
#   /query      database, index, vector_store, figure_guard -> query_ready_index
#   /documents  index                                       -> documents_index
#   /export     database                                    -> database_ready
#   /health     nothing: it reports readiness, so it never depends on it
#
# ORDER MATTERS FOR /query. FastAPI runs a route's dependencies BEFORE it
# validates the request body. Found in step 6 against a server with no data:
# an empty question got 503 "try again in 5 s" instead of 422, and the readiness
# re-try touched the disk for a request that was malformed anyway. So /query
# takes only get_readiness (which never raises) as a dependency and calls
# query_ready_index inside the handler, after validation: 422 always wins.
# Routes without a body use the raising dependencies directly.
#
# Every endpoint reaches readiness through Depends(get_readiness) — directly or
# as a sub-dependency — so one app.dependency_overrides[get_readiness] swaps it
# for a hand-built Readiness everywhere, and route tests need no database.
# (Calling get_readiness(request) directly would bypass the override: found
# when the step-7 route tests first ran.)

from typing import Annotated

from fastapi import Depends, Request

from src.api.errors import NotReadyError
from src.api.readiness import DATABASE, FIGURE_GUARD, INDEX, VECTOR_STORE, Readiness
from src.query.resolve import CorpusIndex

QUERY_NEEDS = frozenset({DATABASE, INDEX, VECTOR_STORE, FIGURE_GUARD})
DOCUMENTS_NEEDS = frozenset({INDEX})
EXPORT_NEEDS = frozenset({DATABASE})


def get_readiness(request: Request) -> Readiness:
    """
    The readiness state the lifespan created. Never raises.

    Args:
        request: The current request.

    Returns:
        app.state.readiness.
    """
    return request.app.state.readiness


async def require(readiness: Readiness, needs: frozenset[str]) -> Readiness:
    """
    Re-try readiness if due, then fail fast if a needed part is missing.

    Args:
        readiness: The app's readiness state.
        needs: Parts the endpoint cannot run without.

    Returns:
        The same state, with every needed part usable.

    Raises:
        NotReadyError: naming the missing parts (logged; the client sees only
            the fixed not_ready message).
    """
    await readiness.refresh_if_due()
    missing = readiness.missing(needs)
    if missing:
        raise NotReadyError(f"unavailable: {', '.join(missing)}")
    return readiness


def _loaded_index(readiness: Readiness) -> CorpusIndex:
    """
    The corpus index, which a usable INDEX part guarantees is loaded.

    Args:
        readiness: A state whose INDEX part is usable.

    Returns:
        The index.

    Raises:
        NotReadyError: if the index is absent despite INDEX being usable —
            an inconsistency that must not reach the query path as None.
    """
    if readiness.index is None:
        raise NotReadyError("index part usable but no index loaded")
    return readiness.index


async def query_ready_index(readiness: Readiness) -> CorpusIndex:
    """
    For /query, called inside the handler after the body is validated.

    Args:
        readiness: From Depends(get_readiness).

    Returns:
        The corpus index for execute_plan.

    Raises:
        NotReadyError: if any of the four parts is missing.
    """
    return _loaded_index(await require(readiness, QUERY_NEEDS))


async def documents_index(
    readiness: Annotated[Readiness, Depends(get_readiness)],
) -> CorpusIndex:
    """
    Dependency for /documents: the corpus index only.

    Args:
        readiness: From get_readiness, as a sub-dependency so a test override
            of get_readiness reaches every endpoint the same way.

    Returns:
        The corpus index.

    Raises:
        NotReadyError: if the index is missing.
    """
    return _loaded_index(await require(readiness, DOCUMENTS_NEEDS))


async def database_ready(readiness: Annotated[Readiness, Depends(get_readiness)]) -> None:
    """
    Dependency for /export: the database file is present.

    Args:
        readiness: From get_readiness, as a sub-dependency.

    Raises:
        NotReadyError: if the database is missing.
    """
    await require(readiness, EXPORT_NEEDS)
