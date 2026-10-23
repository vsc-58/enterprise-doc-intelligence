"""GET /health: whether the API can answer, and if not, which part is missing.

Says whether the API can answer, and if not, which part is missing
(docs/api_contract.md §6). 200 when ready, 503 when not — same body.

/health never depends on readiness: it REPORTS it. It first lets readiness
re-try if due, so polling /health is enough for a not-ready server to
recover. The database and store checks are live, so they also catch a store
that broke after startup — which the cached readiness state would not.

One field per readiness part. database and vector_store also run a live
check; index and figure_guard follow the readiness state. A 503 body therefore
always names the part that is down. An earlier version had no figure_guard
field, so a guard failure gave 503 with every check "ok".

No LLM call: hosts poll this endpoint, and a health check that spends money
or depends on OpenAI would report our server down whenever OpenAI is slow.

Live checks never create a store: each looks for the file or folder first,
because opening a missing SQLite file or Chroma folder creates an empty one.
"""

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.api.dependencies import get_readiness
from src.api.models import CheckState, HealthChecks, HealthResponse, HealthState
from src.api.readiness import (
    DATABASE,
    FIGURE_GUARD,
    INDEX,
    VECTOR_STORE,
    Readiness,
    database_present,
    vector_store_present,
)
from src.query.planner import planner_version
from src.rag.chain import chain_version
from src.storage.metadata_store import get_session
from src.storage.vector_store import collection_count
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


def database_ok() -> bool:
    """Whether the SQLite file exists and answers a trivial query.

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


def check_state(readiness: Readiness, part: str, live_ok: bool = True) -> CheckState:
    """One check in the /health body.

    Ok only when the readiness state counts the part as usable AND its live
    check, if it has one, passes. Reading both means the body can never say
    "ok" for every part while the status is 503: whichever side failed, the
    part is named.

    Args:
        readiness: The app's readiness state, after any due re-try.
        part: A readiness part name (DATABASE, INDEX, VECTOR_STORE, FIGURE_GUARD).
        live_ok: The live check's result. Defaults to True for a part with no
            live check (figure_guard), which then follows the readiness state.

    Returns:
        CheckState.OK or CheckState.UNAVAILABLE.
    """
    usable = part not in readiness.failures
    return CheckState.OK if usable and live_ok else CheckState.UNAVAILABLE


def chunk_count() -> int | None:
    """Chunks in the vector store, without creating a missing store.

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
    """Report readiness; 503 with the same body when any part is not ready.

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
        database=check_state(readiness, DATABASE, live_ok=db_ok),
        vector_store=check_state(readiness, VECTOR_STORE, live_ok=bool(chunks)),  # 0 is not ready
        index=check_state(readiness, INDEX, live_ok=index is not None),
        figure_guard=check_state(readiness, FIGURE_GUARD),
    )
    # readiness.ready is redundant while every part has a field (a test pins
    # that); kept so a part added without a field still cannot report ok.
    all_ok = readiness.ready and all(
        state is CheckState.OK for state in checks.model_dump().values()
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
