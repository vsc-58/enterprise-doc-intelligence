# src/query/planner.py
# Module: Query planner
# Purpose: One LLM call turning a question into a validated QueryPlan.
# Depends on: langchain_openai, src.query.prompts, src.query.schemas,
#             src.utils.config, src.utils.logger
#
# Plans only. No retrieval, no SQL, no entity resolution, no answer. The planner
# is the only LLM call in the query path that sees the user's question, and no
# LLM in that path ever sees a stored figure (D27).

import hashlib

from langchain_openai import ChatOpenAI

from src.query.prompts import PLANNER_PROMPT, PLANNER_SYSTEM
from src.query.schemas import (
    PlanType,
    QueryPlan,
    UnsupportedReason,
)
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_llm: ChatOpenAI | None = None

import json
...
def planner_version() -> str:
    """
    Version identifier for the planner prompt, schema and model.

    The bound schema is prompt surface, not just a contract: with_structured_output
    sends QueryPlan's JSON schema — including every class and enum docstring — as a
    function definition, and the probe measured it at roughly 1,500 of the 2,450
    input tokens per call. Hashing only the system prompt would let a docstring edit
    in schemas.py change planner behaviour while two eval results claimed the same
    version.

    Returns:
        32-character hex digest.
    """
    schema = json.dumps(QueryPlan.model_json_schema(), sort_keys=True)
    payload = f"{PLANNER_SYSTEM}|{settings.OPENAI_MODEL}|{schema}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

def _get_llm() -> ChatOpenAI:
    """
    Return the process-wide chat model, constructing it once.

    api_key is passed explicitly: pydantic-settings loads .env into Settings, not
    into os.environ, so a client relying on ambient environment variables finds
    nothing (Phase 2 bug 5).

    Returns:
        The configured ChatOpenAI client.
    """
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )
    return _llm


def _fallback(reason: str) -> QueryPlan:
    """
    Build the plan returned when planning itself fails.

    Clarification rather than unsupported: a failure to plan says nothing about
    whether the system could answer, and claiming a capability limit would be a
    false statement about the corpus.

    Args:
        reason: Short description for the rationale, logged not displayed.

    Returns:
        A clarification plan.
    """
    return QueryPlan(
        plan_type=PlanType.CLARIFICATION,
        clarification_question=(
            "I couldn't interpret that question. Could you rephrase it, naming the "
            "company and what you'd like to know?"
        ),
        rationale=f"planner fallback: {reason}",
    )


def plan_query(question: str) -> QueryPlan:
    """
    Plan one question.

    Single-item operation, so a transport failure raises rather than being
    swallowed; only a malformed model response degrades to a clarification plan,
    because that is a statement about the question rather than about the system.

    Args:
        question: The user's question, verbatim.

    Returns:
        A validated QueryPlan.

    Raises:
        ValueError: if the question is empty or whitespace.
        Exception: transport, authentication and rate-limit errors propagate.
    """
    if not question or not question.strip():
        raise ValueError("question must not be empty")

    chain = PLANNER_PROMPT | _get_llm().with_structured_output(
        QueryPlan, include_raw=True
    )

    logger.info("planner_call", planner_version=planner_version())
    response = chain.invoke({"question": question})

    parsed = response.get("parsed")
    parsing_error = response.get("parsing_error")

    if parsed is None:
        logger.warning(
            "planner_validation_failed",
            error=str(parsing_error),
            question=question[:120],
        )
        return _fallback("schema validation failed")

    usage = getattr(response.get("raw"), "usage_metadata", None) or {}
    logger.info(
        "planner_plan",
        plan_type=parsed.plan_type.value,
        intents=[task.intent.value for task in parsed.tasks],
        unsupported_reason=(
            parsed.unsupported_reason.value if parsed.unsupported_reason else None
        ),
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
    )
    return parsed