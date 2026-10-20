"""Route dependencies: each endpoint gets what it needs, or a 503 naming nothing secret.

No database: a Readiness is built by hand and put on app.state, and its warm-up
is replaced so a re-try cannot reach a real store.
"""

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import database_ready, documents_index, query_index
from src.api.main import create_app
from src.api.readiness import DATABASE, FIGURE_GUARD, INDEX, VECTOR_STORE, Readiness
from src.query.resolve import CorpusIndex


@pytest.fixture(autouse=True)
def no_real_warm_up(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record re-tries instead of touching stores; the state stays as the test set it."""
    calls: list[int] = []

    async def fake_warm(self: Readiness) -> None:
        calls.append(1)

    monkeypatch.setattr(Readiness, "warm", fake_warm)
    return calls


def _app(failures: dict[str, str], index: CorpusIndex | None) -> FastAPI:
    """A fresh app with a hand-set readiness and one probe route per dependency."""
    app = create_app()
    state = Readiness(retry_seconds=5)
    state.failures = dict(failures)
    state.index = index
    app.state.readiness = state

    @app.get("/q")
    async def q(idx: CorpusIndex = Depends(query_index)) -> dict[str, int]:
        return {"documents": len(idx.entries)}

    @app.get("/d")
    async def d(idx: CorpusIndex = Depends(documents_index)) -> dict[str, int]:
        return {"documents": len(idx.entries)}

    @app.get("/e", dependencies=[Depends(database_ready)])
    async def e() -> dict[str, bool]:
        return {"ok": True}

    return app


def test_ready_app_passes_the_index_to_the_route(index: CorpusIndex) -> None:
    client = TestClient(_app({}, index))
    assert client.get("/q").json() == {"documents": len(index.entries)}
    assert client.get("/d").status_code == 200
    assert client.get("/e").status_code == 200


def test_query_needs_every_part(index: CorpusIndex) -> None:
    """With the vector store missing, /query is 503 but /documents and /export still work."""
    client = TestClient(_app({VECTOR_STORE: "EmptyVectorStore"}, index))
    response = client.get("/q")
    assert response.status_code == 503
    assert response.json()["error"] == "not_ready"
    assert response.headers["Retry-After"] == "5"
    assert client.get("/d").status_code == 200
    assert client.get("/e").status_code == 200


def test_missing_database_blocks_every_endpoint() -> None:
    failures = {DATABASE: "DatabaseNotFound", INDEX: "DatabaseNotFound",
                FIGURE_GUARD: "DatabaseNotFound"}
    client = TestClient(_app(failures, None))
    assert [client.get(p).status_code for p in ("/q", "/d", "/e")] == [503, 503, 503]


def test_not_ready_body_does_not_name_the_missing_part(index: CorpusIndex) -> None:
    """Which part is missing is for /health and the logs; the error body stays fixed."""
    response = TestClient(_app({VECTOR_STORE: "EmptyVectorStore"}, index)).get("/q")
    assert "vector" not in response.text.lower()


def test_not_ready_request_triggers_a_retry(index: CorpusIndex, no_real_warm_up: list[int]) -> None:
    TestClient(_app({VECTOR_STORE: "EmptyVectorStore"}, index)).get("/q")
    assert no_real_warm_up == [1]


def test_ready_request_never_retries(index: CorpusIndex, no_real_warm_up: list[int]) -> None:
    client = TestClient(_app({}, index))
    for _ in range(3):
        client.get("/q")
    assert no_real_warm_up == []


def test_usable_index_part_without_an_index_is_still_not_ready() -> None:
    """An inconsistent state must become a 503, never a None index in the query path."""
    response = TestClient(_app({}, None)).get("/q")
    assert response.status_code == 503
