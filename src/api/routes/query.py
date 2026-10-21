# src/api/routes/query.py
# Module: POST /query
# Purpose: Answer one question through the existing query path and return the
#          contract's response (docs/api_contract.md §2-§3, §7).
# Depends on: fastapi, src.api.*, src.query.*, src.rag.chain, src.utils.*
#
# The query path is called, not rebuilt:
#   plan_query()      sync, one LLM call      -> asyncio.to_thread (never blocks the loop)
#   execute_plan()    async, tasks concurrent -> awaited directly
#   assemble()        pure Python, fast       -> called directly
#
# Timeout: both awaited calls run inside asyncio.timeout(QUERY_TIMEOUT_SECONDS).
# On expiry the async chain calls are cancelled; a planner thread already running
# cannot be stopped and ends within its own per-call limit (D54).
#
# Status codes this route produces (the rest come from main.py's handlers):
#   200  any answer, refusal or partial answer
#   429  usage limit reached: rate_limited or daily_budget_exhausted
#        (off unless configured, D55)
#   503  every task failed (ServiceDegradedError)
#   504  timeout (QueryTimeoutError)
# Refusals are 200 because they are correct answers about the corpus.

import asyncio
from typing import Annotated

from fastapi import APIRouter, Body, Depends, Request

from src.api.context import client_address, elapsed_ms, request_id_of
from src.api.dependencies import get_limiter, get_readiness, query_ready_index
from src.api.errors import QueryTimeoutError, ServiceDegradedError
from src.api.limits import UsageLimiter
from src.api.mapping import build_query_response, nothing_answered
from src.api.models import QUERY_EXAMPLES, ErrorBody, QueryRequest, QueryResponse
from src.api.readiness import Readiness
from src.query.executor import execute_plan
from src.query.planner import plan_query, planner_version
from src.query.synthesizer import assemble
from src.rag.chain import chain_version
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["query"])

_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    503: {
        "model": ErrorBody,
        "description": "Not ready, language model unavailable, or every part of "
        "the question failed. Retry-After says when to try again.",
    },
    504: {"model": ErrorBody, "description": "The question took too long to answer."},
    429: {"model": ErrorBody, "description": "Usage limit reached (off unless "
          "configured). Retry-After says when to try again."},
    500: {"model": ErrorBody, "description": "Unexpected error."},
}


@router.post(
    "/query",
    response_model=QueryResponse,
    responses=_ERROR_RESPONSES,
    summary="Answer a question about the corpus",
    description=(
        "Plans the question into 1-3 tasks, answers figures from the figures "
        "table and narrative from cited filing text, and returns one answer. "
        "A refusal (company or year not held, section not captured, ...) is a "
        "200 with a typed `refusal_reason` per task."
    ),
)
async def query(
    request: Request,
    body: Annotated[QueryRequest, Body(openapi_examples=QUERY_EXAMPLES)],
    readiness: Annotated[Readiness, Depends(get_readiness)],
    limiter: Annotated[UsageLimiter, Depends(get_limiter)],
) -> QueryResponse:
    """
    Answer one question.

    Args:
        request: The current request (for its id and timer).
        body: The validated question; malformed bodies never reach here (422).
        readiness: The app's readiness state. Checked here, after validation,
            so a malformed request gets 422 even while the server is not ready.
        limiter: Usage limits. Checked after readiness and immediately before
            the planner, so only a request about to spend money is counted.

    Returns:
        The answer, its tasks, figures with quality flags, and sources.

    Raises:
        NotReadyError: a needed data part is missing (-> 503 not_ready).
        RateLimitedError / DailyBudgetError: a usage limit is reached
            (-> 429 rate_limited / daily_budget_exhausted, with Retry-After).
        QueryTimeoutError: QUERY_TIMEOUT_SECONDS passed (-> 504).
        ServiceDegradedError: every task failed (-> 503).
        openai errors on the planner call propagate to main.py's handlers (-> 503).
    """
    index = await query_ready_index(readiness)
    limiter.check(client_address(request))
    try:
        async with asyncio.timeout(settings.QUERY_TIMEOUT_SECONDS):
            plan = await asyncio.to_thread(plan_query, body.question)
            outcomes = await execute_plan(plan, index)
    except TimeoutError as exc:
        raise QueryTimeoutError(
            f"no answer within {settings.QUERY_TIMEOUT_SECONDS} s"
        ) from exc

    answer = assemble(plan, outcomes)
    if nothing_answered(answer):
        raise ServiceDegradedError(f"all {len(answer.statuses)} tasks failed")

    response = build_query_response(
        body.question,
        plan,
        outcomes,
        answer,
        planner_version=planner_version(),
        chain_version=chain_version(),
        request_id=request_id_of(request),
        latency_ms=elapsed_ms(request),
    )
    logger.info(
        "query_answered",
        plan_type=plan.plan_type.value,
        statuses=[status.value for status in answer.statuses],
        refusals=[t.refusal_reason.value for t in response.tasks if t.refusal_reason],
        sources=len(response.sources),
    )
    return response
