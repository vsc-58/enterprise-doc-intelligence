# src/api/routes/health.py
# Module: GET /health
# Purpose: Say whether the API can answer, and if not, which part is missing
#          (docs/api_contract.md §6). 200 when ready, 503 when not — same body.
# Depends on: fastapi, sqlalchemy, src.api.*, src.query.planner, src.rag.chain,
#             src.storage.*
#
# /health never depends on readiness: it REPORTS it. It first lets readiness
# re-try if due, so polling /health is enough for a not-ready server to
# recover. The database and store checks are live, so they also catch a store
# that broke after startup — which the cached readiness state would not.
#
# No LLM call: hosts poll this endpoint, and a health check that spends money
# or depends on OpenAI would report our server down whenever OpenAI is slow.
#
# Live checks never create a store: each looks for the file or folder first,
# because opening a missing SQLite file or Chroma folder creates an empty one.

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.api.dependencies import get_readiness
from src.api.models import CheckState, HealthChecks, HealthResponse, HealthState
from src.api.readiness import Readiness, database_present, vector_store_present
from src.query.planner import planner_version
from src.rag.chain import chain_version
from src.storage.metadata_store import get_session
from src.storage.vector_store import collection_count
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


def database_ok() -> bool:
    """
    Whether the SQLite file exists and answers a trivial query.

    Returns:
        True if SELECT 1 succeeds; False if the file is missing (not opened)
        or the query fails (logged).
    """
    if not database_present(settings.SQLITE_DB_PATH):
        return False
    try:
        with get_session() as session:
            session.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        logger.error("health_database_failed", error_type=type(exc).__name__, error=str(exc))
        return False
    return True


def chunk_count() -> int | None:
    """
    Chunks in the vector store, without creating a missing store.

    Returns:
        The count, or None if the folder is missing or the collection cannot be
        read (collection_count() returns -1 and logs).
    """
    if not vector_store_present(settings.CHROMA_DB_PATH):
        return None
    count = collection_count()
    return count if count >= 0 else None


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "Not ready; `checks` names the part."}},
    summary="Readiness of the database, vector store and corpus index",
)
async def health(
    response: Response,
    readiness: Annotated[Readiness, Depends(get_readiness)],
) -> HealthResponse:
    """
    Report readiness; 503 with the same body when any part is not ready.

    Args:
        response: Used only to set the status code.
        readiness: The app's readiness state.

    Returns:
        Overall status, each check, document and chunk counts, prompt versions.
    """
    await readiness.refresh_if_due()
    db_ok, chunks = await asyncio.gather(
        asyncio.to_thread(database_ok), asyncio.to_thread(chunk_count)
    )
    index = readiness.index
    checks = HealthChecks(
        database=CheckState.OK if db_ok else CheckState.UNAVAILABLE,
        vector_store=CheckState.OK if chunks else CheckState.UNAVAILABLE,  # 0 is not ready
        index=CheckState.OK if index is not None else CheckState.UNAVAILABLE,
    )
    all_ok = readiness.ready and all(
        state is CheckState.OK for state in (checks.database, checks.vector_store, checks.index)
    )
    if not all_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        logger.warning("health_not_ready", checks=checks.model_dump(mode="json"),
                       readiness_failures=sorted(readiness.failures))
    return HealthResponse(
        status=HealthState.OK if all_ok else HealthState.NOT_READY,
        checks=checks,
        documents=len(index.entries) if index is not None else None,
        chunks=chunks,
        planner_version=planner_version(),
        chain_version=chain_version(),
    )
