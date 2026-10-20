# src/api/dependencies.py
# Module: Route dependencies
# Purpose: What each endpoint declares it needs, via FastAPI's Depends. Each
#          dependency re-tries readiness if due, then either returns what the
#          route needs or raises NotReadyError (-> 503 not_ready).
# Depends on: fastapi, src.api.errors, src.api.readiness, src.query.resolve
#
# Per-endpoint needs (contract §6):
#   /query      database, index, vector_store, figure_guard -> query_index
#   /documents  index                                       -> documents_index
#   /export     database                                    -> database_ready
#   /health     nothing: it reports readiness, so it never depends on it
#
# Tests replace these through app.dependency_overrides, so route tests need no
# database: override query_index with a fixture index.

from fastapi import Request

from src.api.errors import NotReadyError
from src.api.readiness import DATABASE, FIGURE_GUARD, INDEX, VECTOR_STORE, Readiness
from src.query.resolve import CorpusIndex

QUERY_NEEDS = frozenset({DATABASE, INDEX, VECTOR_STORE, FIGURE_GUARD})
DOCUMENTS_NEEDS = frozenset({INDEX})
EXPORT_NEEDS = frozenset({DATABASE})


def get_readiness(request: Request) -> Readiness:
    """
    The readiness state the lifespan created.

    Args:
        request: The current request.

    Returns:
        app.state.readiness.
    """
    return request.app.state.readiness


async def _require(request: Request, needs: frozenset[str]) -> Readiness:
    """
    Re-try readiness if due, then fail fast if a needed part is missing.

    Args:
        request: The current request.
        needs: Parts the endpoint cannot run without.

    Returns:
        The readiness state, with every needed part usable.

    Raises:
        NotReadyError: naming the missing parts (logged; the client sees only
            the fixed not_ready message).
    """
    readiness = get_readiness(request)
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


async def query_index(request: Request) -> CorpusIndex:
    """
    Dependency for /query: every part ready; returns the corpus index.

    Args:
        request: The current request.

    Returns:
        The corpus index for execute_plan.
    """
    return _loaded_index(await _require(request, QUERY_NEEDS))


async def documents_index(request: Request) -> CorpusIndex:
    """
    Dependency for /documents: the corpus index only.

    Args:
        request: The current request.

    Returns:
        The corpus index.
    """
    return _loaded_index(await _require(request, DOCUMENTS_NEEDS))


async def database_ready(request: Request) -> None:
    """
    Dependency for /export: the database file is present.

    Args:
        request: The current request.
    """
    await _require(request, EXPORT_NEEDS)
