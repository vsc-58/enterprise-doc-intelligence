"""POST /query end to end through FastAPI, with the planner and executor replaced.

Everything between them is real: validation, the readiness dependency (overridden
with the fixture index), the timeout, assemble(), mapping, the error handlers and
the middleware. No API key, network, database or vector store.

The planner and executor are patched where the route looks them up —
src.api.routes.query — not where they are defined: the route imported the names,
so patching src.query.planner would leave the route's copy untouched.
"""

import asyncio
from collections.abc import Callable, Iterator

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from src.api.context import REQUEST_ID_HEADER
from src.api.dependencies import get_readiness
from src.api.main import create_app
from src.api.readiness import VECTOR_STORE, Readiness
from src.api.routes import query as query_route
from src.query.resolve import CorpusIndex
from src.query.schemas import QueryPlan, TaskOutcome
from src.utils.config import settings
from tests.query_samples import (
    APPLE_2022_TASK,
    APPLE_OUTCOME,
    APPLE_SQL_FAILED,
    APPLE_TASK,
    INTEL_FIGURE,
    INTEL_FIGURE_TASK,
    INTEL_MARGIN_TASK,
    INTEL_RAG_FAILED,
    INTEL_SECTION_ABSENT,
    YEAR_REFUSED,
    YEAR_SENTENCE,
    hybrid,
    single,
)

APPLE_Q = {"question": "What was Apple's total revenue in fiscal 2021?"}


def ready_state(index: CorpusIndex) -> Readiness:
    """A readiness state with every part usable, as after a successful warm-up."""
    state = Readiness()
    state.failures = {}
    state.index = index
    return state


class Stub:
    """Replaces plan_query and execute_plan in the route; records what was called."""

    def __init__(self, plan: QueryPlan, outcomes: list[TaskOutcome]) -> None:
        self.plan, self.outcomes = plan, outcomes
        self.planned: list[str] = []
        self.executed_with: CorpusIndex | None = None
        self.delay_s = 0.0
        self.plan_error: Exception | None = None

    def plan_query(self, question: str) -> QueryPlan:
        self.planned.append(question)
        if self.plan_error is not None:
            raise self.plan_error
        return self.plan

    async def execute_plan(self, plan: QueryPlan, index: CorpusIndex) -> list[TaskOutcome]:
        self.executed_with = index
        await asyncio.sleep(self.delay_s)
        return self.outcomes


@pytest.fixture
def client_for(
    monkeypatch: pytest.MonkeyPatch, index: CorpusIndex
) -> Iterator[Callable[[Stub], TestClient]]:
    """Build a client for a fresh app whose route uses the given stub."""
    def make(stub: Stub) -> TestClient:
        monkeypatch.setattr(query_route, "plan_query", stub.plan_query)
        monkeypatch.setattr(query_route, "execute_plan", stub.execute_plan)
        app = create_app()
        app.dependency_overrides[get_readiness] = lambda: ready_state(index)
        return TestClient(app)        # no "with": the lifespan does not run
    yield make


# --- 200 ------------------------------------------------------------------------------


def test_figure_question_returns_200_answer(client_for, index: CorpusIndex) -> None:
    stub = Stub(single(APPLE_TASK), [APPLE_OUTCOME])
    response = client_for(stub).post("/query", json=APPLE_Q)
    body = response.json()
    assert response.status_code == 200
    assert "$365.82 billion" in body["answer"]
    assert body["tasks"][0]["figures"][0]["quality"]["verdict"] == "verified"
    assert body["request_id"] == response.headers[REQUEST_ID_HEADER]
    assert len(body["planner_version"]) == 32 and len(body["chain_version"]) == 32
    assert stub.executed_with is index                       # the dependency's index
    assert stub.planned == [APPLE_Q["question"]]


def test_refusal_is_200_with_typed_reason(client_for) -> None:
    stub = Stub(single(APPLE_2022_TASK), [YEAR_REFUSED])
    response = client_for(stub).post("/query", json={"question": "What was Apple's revenue in 2022?"})
    body = response.json()
    assert response.status_code == 200
    assert body["answer"] == YEAR_SENTENCE
    assert body["tasks"][0]["status"] == "refused"
    assert body["tasks"][0]["refusal_reason"] == "year_not_held"


def test_partial_hybrid_is_200(client_for) -> None:
    """R38: figure answered, MD&A question refused."""
    stub = Stub(hybrid(INTEL_FIGURE_TASK, INTEL_MARGIN_TASK), [INTEL_FIGURE, INTEL_SECTION_ABSENT])
    body = client_for(stub).post("/query", json={"question": "Intel net income and margins?"}).json()
    assert [t["status"] for t in body["tasks"]] == ["success", "refused"]
    assert body["tasks"][1]["refusal_reason"] == "section_absent"


def test_refused_plus_failed_is_200(client_for) -> None:
    refused = YEAR_REFUSED.model_copy(update={"task_id": "t1"})
    stub = Stub(hybrid(INTEL_FIGURE_TASK, INTEL_MARGIN_TASK), [refused, INTEL_RAG_FAILED])
    response = client_for(stub).post("/query", json={"question": "x"})
    assert response.status_code == 200
    assert [t["status"] for t in response.json()["tasks"]] == ["refused", "failed"]


def test_question_is_stripped_before_planning(client_for) -> None:
    stub = Stub(single(APPLE_TASK), [APPLE_OUTCOME])
    client_for(stub).post("/query", json={"question": "   What was Apple's revenue?  "})
    assert stub.planned == ["What was Apple's revenue?"]


# --- 422: no planner call ---------------------------------------------------------------


@pytest.mark.parametrize("payload", [
    {"question": ""},
    {"question": "   "},
    {"question": "x" * (settings.QUERY_MAX_CHARS + 1)},
    {"question": "What was Apple's revenue?", "company": "AAPL"},
    {},
])
def test_malformed_body_is_422_and_never_planned(client_for, payload: dict) -> None:
    stub = Stub(single(APPLE_TASK), [APPLE_OUTCOME])
    response = client_for(stub).post("/query", json=payload)
    assert response.status_code == 422
    assert stub.planned == []                                # no LLM call, no cost


# --- 5xx ---------------------------------------------------------------------------------


def test_every_task_failed_is_503_service_degraded(client_for) -> None:
    stub = Stub(hybrid(INTEL_FIGURE_TASK, INTEL_MARGIN_TASK), [APPLE_SQL_FAILED, INTEL_RAG_FAILED])
    response = client_for(stub).post("/query", json={"question": "x"})
    assert response.status_code == 503
    assert response.json()["error"] == "service_degraded"


def test_slow_question_is_504(client_for, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "QUERY_TIMEOUT_SECONDS", 0.05)
    stub = Stub(single(APPLE_TASK), [APPLE_OUTCOME])
    stub.delay_s = 1.0
    response = client_for(stub).post("/query", json=APPLE_Q)
    assert response.status_code == 504
    assert response.json()["error"] == "timeout"


def test_openai_down_on_planner_is_503_llm_unavailable(client_for) -> None:
    stub = Stub(single(APPLE_TASK), [APPLE_OUTCOME])
    stub.plan_error = openai.APIConnectionError(
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
    response = client_for(stub).post("/query", json=APPLE_Q)
    assert response.status_code == 503
    assert response.json()["error"] == "llm_unavailable"
    assert stub.executed_with is None                        # nothing executed


def test_not_ready_is_503_before_planning(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real dependency, no override: a missing store stops the request first."""
    stub = Stub(single(APPLE_TASK), [APPLE_OUTCOME])
    monkeypatch.setattr(query_route, "plan_query", stub.plan_query)

    async def no_warm(self: Readiness) -> None:
        return None

    monkeypatch.setattr(Readiness, "warm", no_warm)
    app = create_app()
    state = Readiness()
    state.failures = {VECTOR_STORE: "EmptyVectorStore"}
    app.state.readiness = state
    response = TestClient(app).post("/query", json=APPLE_Q)
    assert response.status_code == 503
    assert response.json()["error"] == "not_ready"
    assert stub.planned == []


def test_malformed_body_is_422_even_when_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """Found running the real server in step 6: an empty question got 503 "try
    again in 5 s". FastAPI runs dependencies before body validation, so the
    readiness check now runs inside the handler and 422 wins."""
    async def no_warm(self: Readiness) -> None:
        return None

    monkeypatch.setattr(Readiness, "warm", no_warm)
    app = create_app()
    state = Readiness()
    state.failures = {VECTOR_STORE: "EmptyVectorStore"}
    app.state.readiness = state
    response = TestClient(app).post("/query", json={"question": "  "})
    assert response.status_code == 422


# --- OpenAPI ------------------------------------------------------------------------------


def test_swagger_shows_the_demo_examples(client_for) -> None:
    spec = client_for(Stub(single(APPLE_TASK), [])).get("/openapi.json").json()
    operation = spec["paths"]["/query"]["post"]
    examples = operation["requestBody"]["content"]["application/json"]["examples"]
    assert examples["figure"]["summary"] == "Exact figure (SQL)"
    assert {"200", "422", "503", "504"} <= set(operation["responses"])
