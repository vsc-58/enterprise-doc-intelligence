"""Usage limits: per-client sliding window, daily budget, and the 429 /query returns.

Both clocks are fake and moved by hand, so a "minute" or a "day" passes
instantly. No API key, network, database or vector store.
"""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import get_readiness
from src.api.errors import DailyBudgetError, RateLimitedError
from src.api.limits import WINDOW_SECONDS, UsageLimiter
from src.api.main import create_app
from src.api.readiness import VECTOR_STORE, Readiness
from src.api.routes import query as query_route
from src.query.resolve import CorpusIndex
from tests.query_samples import APPLE_OUTCOME, APPLE_TASK, single


class Clocks:
    """A monotonic clock and a UTC clock that move together."""

    def __init__(self) -> None:
        self.mono = 1000.0
        self.utc = datetime(2026, 10, 21, 23, 59, 0, tzinfo=UTC)   # 60 s before midnight

    def monotonic(self) -> float:
        return self.mono

    def utc_now(self) -> datetime:
        return self.utc

    def advance(self, seconds: float) -> None:
        self.mono += seconds
        self.utc += timedelta(seconds=seconds)


@pytest.fixture
def clocks() -> Clocks:
    return Clocks()


def _limiter(clocks: Clocks, per_minute: int | None = None, budget: int | None = None) -> UsageLimiter:
    return UsageLimiter(per_minute=per_minute, daily_budget=budget,
                        monotonic=clocks.monotonic, utc_now=clocks.utc_now)


# --- Off by default -------------------------------------------------------------------


def test_no_limits_never_blocks(clocks: Clocks) -> None:
    limiter = _limiter(clocks)
    assert not limiter.enabled
    for _ in range(1000):
        limiter.check("1.2.3.4")


def test_app_limiter_is_off_with_default_settings() -> None:
    """Locally nothing changes until Phase 8 sets the two values."""
    assert not create_app().state.limiter.enabled


# --- Per-client sliding window ----------------------------------------------------------


def test_per_client_limit_blocks_the_next_request(clocks: Clocks) -> None:
    limiter = _limiter(clocks, per_minute=2)
    limiter.check("a")
    clocks.advance(10)
    limiter.check("a")
    with pytest.raises(RateLimitedError) as caught:
        limiter.check("a")
    # The oldest request (t=0) leaves the window at t=60; we are at t=10.
    assert caught.value.retry_after_s == 50


def test_clients_are_counted_separately(clocks: Clocks) -> None:
    limiter = _limiter(clocks, per_minute=1)
    limiter.check("a")
    limiter.check("b")                                         # a different address
    with pytest.raises(RateLimitedError):
        limiter.check("a")


def test_window_slides_rather_than_resetting(clocks: Clocks) -> None:
    """A slot frees when that request is 60 s old, not at a fixed minute boundary."""
    limiter = _limiter(clocks, per_minute=2)
    limiter.check("a")                                         # t = 0
    clocks.advance(30)
    limiter.check("a")                                         # t = 30
    clocks.advance(WINDOW_SECONDS - 30)                        # t = 60: first one expired
    limiter.check("a")
    with pytest.raises(RateLimitedError) as caught:            # t = 30 and 60 still inside
        limiter.check("a")
    assert caught.value.retry_after_s == 30


def test_retry_after_is_never_zero(clocks: Clocks) -> None:
    limiter = _limiter(clocks, per_minute=1)
    limiter.check("a")
    clocks.advance(WINDOW_SECONDS - 0.2)
    with pytest.raises(RateLimitedError) as caught:
        limiter.check("a")
    assert caught.value.retry_after_s == 1


def test_idle_clients_are_forgotten(clocks: Clocks) -> None:
    """Every address ever seen must not stay in memory forever."""
    limiter = _limiter(clocks, per_minute=5)
    for n in range(100):
        limiter.check(f"10.0.0.{n}")
    clocks.advance(WINDOW_SECONDS + 1)
    limiter.check("10.0.1.1")                                  # triggers the sweep
    assert set(limiter._recent) == {"10.0.1.1"}


# --- Daily budget --------------------------------------------------------------------------


def test_daily_budget_covers_all_clients(clocks: Clocks) -> None:
    limiter = _limiter(clocks, budget=3)
    for client in ("a", "b", "c"):
        limiter.check(client)
    with pytest.raises(DailyBudgetError) as caught:
        limiter.check("d")
    assert caught.value.retry_after_s == 60                    # until UTC midnight
    assert limiter.used_today == 3


def test_daily_budget_resets_at_utc_midnight(clocks: Clocks) -> None:
    limiter = _limiter(clocks, budget=1)
    limiter.check("a")
    with pytest.raises(DailyBudgetError):
        limiter.check("a")
    clocks.advance(61)                                          # past midnight UTC
    limiter.check("a")
    assert limiter.used_today == 1


def test_request_refused_per_client_does_not_spend_the_budget(clocks: Clocks) -> None:
    limiter = _limiter(clocks, per_minute=1, budget=10)
    limiter.check("a")
    for _ in range(5):
        with pytest.raises(RateLimitedError):
            limiter.check("a")
    assert limiter.used_today == 1


# --- Through /query --------------------------------------------------------------------------


@pytest.fixture
def app_with_limit(monkeypatch: pytest.MonkeyPatch, index: CorpusIndex, clocks: Clocks):
    """A ready app answering Apple's revenue, with a per-client limit of 1."""
    planned: list[str] = []

    def plan_query(question: str):
        planned.append(question)
        return single(APPLE_TASK)

    async def execute_plan(plan, idx):
        return [APPLE_OUTCOME]

    monkeypatch.setattr(query_route, "plan_query", plan_query)
    monkeypatch.setattr(query_route, "execute_plan", execute_plan)
    app = create_app()
    state = Readiness()
    state.failures, state.index = {}, index
    app.dependency_overrides[get_readiness] = lambda: state
    app.state.limiter = _limiter(clocks, per_minute=1)
    return app, state, planned


QUESTION = {"question": "What was Apple's total revenue in fiscal 2021?"}


def test_query_over_the_limit_is_429_with_retry_after(app_with_limit) -> None:
    app, _, planned = app_with_limit
    client = TestClient(app)
    assert client.post("/query", json=QUESTION).status_code == 200
    response = client.post("/query", json=QUESTION)
    assert response.status_code == 429
    assert response.json()["error"] == "rate_limited"
    assert response.headers["Retry-After"] == "60"
    assert len(planned) == 1                                   # the second never reached the planner


def test_daily_budget_has_its_own_code_and_message(app_with_limit, clocks: Clocks) -> None:
    """An interviewer must read "come back tomorrow", not "slow down"."""
    app, _, _ = app_with_limit
    app.state.limiter = _limiter(clocks, budget=1)
    client = TestClient(app)
    client.post("/query", json=QUESTION)
    response = client.post("/query", json=QUESTION)
    body = response.json()
    assert response.status_code == 429
    assert body["error"] == "daily_budget_exhausted"
    assert body["message"] == (
        "This demo's daily question budget is used up. It resets at 00:00 UTC."
    )
    assert response.headers["Retry-After"] == "60"            # fake clock: 60 s to midnight


def test_malformed_request_is_not_counted(app_with_limit) -> None:
    """A 422 spends no money, so it must not use up the client's allowance."""
    app, _, _ = app_with_limit
    client = TestClient(app)
    assert client.post("/query", json={"question": ""}).status_code == 422
    assert client.post("/query", json=QUESTION).status_code == 200


def test_not_ready_request_is_not_counted(app_with_limit, monkeypatch: pytest.MonkeyPatch) -> None:
    app, state, _ = app_with_limit

    async def no_warm(self: Readiness) -> None:
        return None

    monkeypatch.setattr(Readiness, "warm", no_warm)
    client = TestClient(app)
    state.failures = {VECTOR_STORE: "EmptyVectorStore"}
    assert client.post("/query", json=QUESTION).status_code == 503
    state.failures = {}
    assert client.post("/query", json=QUESTION).status_code == 200


def test_other_endpoints_are_not_limited(app_with_limit) -> None:
    """Only /query spends money; /health and /documents stay usable for monitoring."""
    app, _, _ = app_with_limit
    client = TestClient(app)
    client.post("/query", json=QUESTION)
    assert client.post("/query", json=QUESTION).status_code == 429
    assert client.get("/documents").status_code == 200
