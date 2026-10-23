"""Readiness: never creates a store, records every missing part, re-tries while not ready.

No database or vector store: paths point into pytest's tmp_path, and the four
loading calls are replaced. One test runs with nothing replaced, to prove that
checking readiness against missing paths writes nothing to disk.
"""

from pathlib import Path

import anyio
import pytest

from src.api import readiness as readiness_module
from src.api.readiness import (
    DATABASE,
    DATABASE_NOT_FOUND,
    EMPTY_VECTOR_STORE,
    FIGURE_GUARD,
    INDEX,
    VECTOR_STORE,
    VECTOR_STORE_NOT_FOUND,
    Readiness,
)
from src.query.resolve import CorpusIndex
from src.rag.figure_guard import FigureGuardError
from src.storage.vector_store import VectorStoreError
from src.utils.config import settings


class Calls:
    """Counts calls to the replaced loaders."""

    def __init__(self) -> None:
        self.load_index = 0
        self.get_store = 0


@pytest.fixture
def data_present(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, index: CorpusIndex) -> Calls:
    """Both stores present on disk; every loader succeeds."""
    db_file = tmp_path / "metadata.db"
    db_file.write_bytes(b"")
    chroma = tmp_path / "chroma"
    chroma.mkdir()
    (chroma / "chroma.sqlite3").write_bytes(b"")
    monkeypatch.setattr(settings, "SQLITE_DB_PATH", str(db_file))
    monkeypatch.setattr(settings, "CHROMA_DB_PATH", str(chroma))

    calls = Calls()

    def load_index() -> CorpusIndex:
        calls.load_index += 1
        return index

    def get_store() -> object:
        calls.get_store += 1
        return object()

    monkeypatch.setattr(readiness_module, "load_index", load_index)
    monkeypatch.setattr(readiness_module, "get_store", get_store)
    monkeypatch.setattr(readiness_module, "collection_count", lambda: 1750)
    monkeypatch.setattr(readiness_module, "load_guard_values", lambda: {})
    return calls


def _warm(state: Readiness) -> Readiness:
    """Run the async warm-up to completion and return the state."""
    anyio.run(state.warm)
    return state


# --- warm() ---------------------------------------------------------------------


def test_all_parts_present_is_ready(data_present: Calls, index: CorpusIndex) -> None:
    state = _warm(Readiness())
    assert state.ready
    assert state.failures == {}
    assert state.index is index


def test_not_ready_before_the_first_check() -> None:
    """A state nobody has warmed must not claim to be ready."""
    assert not Readiness().ready


def test_missing_database_is_reported_without_opening_it(
    data_present: Calls, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(settings, "SQLITE_DB_PATH", str(tmp_path / "absent.db"))
    state = _warm(Readiness())
    assert state.failures == {
        DATABASE: DATABASE_NOT_FOUND, INDEX: DATABASE_NOT_FOUND, FIGURE_GUARD: DATABASE_NOT_FOUND,
    }
    assert data_present.load_index == 0                       # never touched
    assert not (tmp_path / "absent.db").exists()


@pytest.mark.parametrize("make_folder", [False, True])
def test_missing_or_empty_chroma_folder_is_reported_without_opening_it(
    data_present: Calls, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, make_folder: bool
) -> None:
    folder = tmp_path / "no_chroma"
    if make_folder:
        folder.mkdir()                                        # exists, but holds nothing
    monkeypatch.setattr(settings, "CHROMA_DB_PATH", str(folder))
    state = _warm(Readiness())
    assert state.failures == {VECTOR_STORE: VECTOR_STORE_NOT_FOUND}
    assert data_present.get_store == 0


@pytest.mark.parametrize("chunks", [0, -1])
def test_empty_or_unreadable_collection_is_not_ready(
    data_present: Calls, monkeypatch: pytest.MonkeyPatch, chunks: int
) -> None:
    monkeypatch.setattr(readiness_module, "collection_count", lambda: chunks)
    assert _warm(Readiness()).failures == {VECTOR_STORE: EMPTY_VECTOR_STORE}


def test_empty_corpus_is_an_index_failure_only(
    data_present: Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    def empty() -> CorpusIndex:
        raise RuntimeError("corpus index is empty")
    monkeypatch.setattr(readiness_module, "load_index", empty)
    state = _warm(Readiness())
    assert state.failures == {INDEX: "RuntimeError"}
    assert state.index is None


def test_every_failed_part_is_recorded(data_present: Calls, monkeypatch: pytest.MonkeyPatch) -> None:
    """One failure must not hide another."""
    def no_store() -> object:
        raise VectorStoreError("cannot open")
    def no_guard() -> dict:
        raise FigureGuardError("cannot read")
    monkeypatch.setattr(readiness_module, "get_store", no_store)
    monkeypatch.setattr(readiness_module, "load_guard_values", no_guard)
    assert _warm(Readiness()).failures == {
        VECTOR_STORE: "VectorStoreError", FIGURE_GUARD: "FigureGuardError",
    }


def test_missing_lists_only_needed_parts(data_present: Calls, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(readiness_module, "collection_count", lambda: 0)
    state = _warm(Readiness())
    assert state.missing({INDEX}) == []                       # /documents can still run
    assert state.missing({DATABASE, INDEX, VECTOR_STORE}) == [VECTOR_STORE]


def test_checking_a_fresh_checkout_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Nothing replaced: the step-4 finding was two empty stores created on start."""
    db_file, chroma = tmp_path / "db" / "metadata.db", tmp_path / "db" / "chroma"
    monkeypatch.setattr(settings, "SQLITE_DB_PATH", str(db_file))
    monkeypatch.setattr(settings, "CHROMA_DB_PATH", str(chroma))
    state = _warm(Readiness())
    assert not state.ready
    assert not (tmp_path / "db").exists()


# --- refresh_if_due() ------------------------------------------------------------


class FakeClock:
    """A monotonic clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_recovers_without_restart_once_data_appears(
    data_present: Calls, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clock = FakeClock()
    real_db = settings.SQLITE_DB_PATH
    monkeypatch.setattr(settings, "SQLITE_DB_PATH", str(tmp_path / "later.db"))
    state = _warm(Readiness(retry_seconds=5, clock=clock))
    assert not state.ready

    monkeypatch.setattr(settings, "SQLITE_DB_PATH", real_db)  # the data arrives
    clock.now += 2
    anyio.run(state.refresh_if_due)
    assert not state.ready                                    # too soon to re-try

    clock.now += 3
    anyio.run(state.refresh_if_due)
    assert state.ready                                        # re-tried and recovered


def test_ready_state_is_not_rechecked(data_present: Calls) -> None:
    clock = FakeClock()
    state = _warm(Readiness(retry_seconds=5, clock=clock))
    clock.now += 3600
    anyio.run(state.refresh_if_due)
    assert data_present.load_index == 1                       # startup only


def test_a_burst_of_requests_shares_one_retry(
    data_present: Calls, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(readiness_module, "collection_count", lambda: 0)
    state = _warm(Readiness(retry_seconds=5, clock=clock))
    first = data_present.get_store
    clock.now += 10

    async def burst() -> None:
        """Ten concurrent requests trying to re-check readiness at the same moment."""
        async with anyio.create_task_group() as group:
            for _ in range(10):
                group.start_soon(state.refresh_if_due)

    anyio.run(burst)
    assert data_present.get_store == first + 1                # ten requests, one re-try
