"""API models: request validation, enum coverage, and the examples Swagger shows.

No FastAPI app, no database, no API key. These are the rules that turn bad input
into a 422 before any planner call, so they are tested on the models directly.
"""

import pytest
from pydantic import ValidationError

from src.api.models import (
    QUERY_EXAMPLES,
    ErrorBody,
    QueryRequest,
    QueryResponse,
    RefusalReason,
)
from src.query.schemas import NarrativeRefusal, ResolutionFailure
from src.utils.config import settings

# --- QueryRequest ------------------------------------------------------------


def test_valid_question_is_accepted_and_stripped() -> None:
    """Outer whitespace is removed before the question goes anywhere."""
    request = QueryRequest(question="  What was Apple's total revenue in fiscal 2021?  ")
    assert request.question == "What was Apple's total revenue in fiscal 2021?"


@pytest.mark.parametrize("question", ["", "   ", "\n\t "])
def test_empty_or_whitespace_question_is_rejected(question: str) -> None:
    """Stripping runs before the length check, so whitespace counts as empty."""
    with pytest.raises(ValidationError) as excinfo:
        QueryRequest(question=question)
    assert excinfo.value.errors()[0]["type"] == "string_too_short"


def test_question_at_the_limit_is_accepted() -> None:
    QueryRequest(question="x" * settings.QUERY_MAX_CHARS)


def test_question_over_the_limit_is_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        QueryRequest(question="x" * (settings.QUERY_MAX_CHARS + 1))
    assert excinfo.value.errors()[0]["type"] == "string_too_long"


def test_missing_question_is_rejected() -> None:
    with pytest.raises(ValidationError):
        QueryRequest.model_validate({})


def test_dropped_filters_are_rejected_not_ignored() -> None:
    """D54: a company/year filter fails loudly instead of being silently dropped."""
    with pytest.raises(ValidationError) as excinfo:
        QueryRequest.model_validate({"question": "What was its revenue?", "company": "AAPL"})
    assert excinfo.value.errors()[0]["type"] == "extra_forbidden"


def test_non_string_question_is_rejected() -> None:
    with pytest.raises(ValidationError):
        QueryRequest.model_validate({"question": 2021})


# --- RefusalReason stays in step with its sources ------------------------------


def test_every_resolution_failure_is_a_refusal_reason() -> None:
    """A new resolver outcome must not reach the API without a code."""
    reasons = {member.value for member in RefusalReason}
    assert {failure.value for failure in ResolutionFailure} <= reasons


def test_every_narrative_refusal_except_failed_is_a_refusal_reason() -> None:
    """FAILED is a fault, reported as status failed — never as a refusal."""
    reasons = {member.value for member in RefusalReason}
    narrative = {r.value for r in NarrativeRefusal if r is not NarrativeRefusal.FAILED}
    assert narrative <= reasons
    assert NarrativeRefusal.FAILED.value not in reasons


def test_refusal_reason_has_no_unexplained_members() -> None:
    """Everything beyond the two source enums is one of the two API-owned codes."""
    sourced = {f.value for f in ResolutionFailure} | {r.value for r in NarrativeRefusal}
    extra = {member.value for member in RefusalReason} - sourced
    assert extra == {"threshold_unparsed", "sql_not_runnable"}


# --- Examples shown in Swagger must be valid --------------------------------------


@pytest.mark.parametrize("name", sorted(QUERY_EXAMPLES))
def test_every_swagger_request_example_validates(name: str) -> None:
    """A demo example that 422s in front of an interviewer is the worst bug."""
    QueryRequest.model_validate(QUERY_EXAMPLES[name]["value"])


def test_query_response_schema_example_validates() -> None:
    example = QueryResponse.model_config["json_schema_extra"]["examples"][0]
    QueryResponse.model_validate(example)


def test_error_body_schema_example_validates() -> None:
    example = ErrorBody.model_config["json_schema_extra"]["examples"][0]
    ErrorBody.model_validate(example)


# --- Settings the contract depends on -------------------------------------------


def test_default_timeouts_satisfy_the_d54_rule() -> None:
    """A hung planner call, with its retry, must finish before the request times out."""
    per_call_worst = settings.OPENAI_TIMEOUT_SECONDS * (settings.OPENAI_MAX_RETRIES + 1)
    assert per_call_worst < settings.QUERY_TIMEOUT_SECONDS


def test_cost_protection_is_off_by_default() -> None:
    """Phase 8 switches these on; locally they must not get in the way."""
    assert settings.RATE_LIMIT_PER_MINUTE is None
    assert settings.DAILY_QUERY_BUDGET is None
