# src/api/readiness.py
# Module: API readiness
# Purpose: Know which data parts the API can use right now, warm them once at
#          startup, and re-try while any part is missing — so the server
#          recovers without a restart when data appears after it started.
# Depends on: sqlalchemy, src.query.resolve, src.rag.figure_guard,
#             src.storage.vector_store, src.utils.*
#
# Four parts, each needed by different endpoints (contract §6, step 5):
#   database      the SQLite file exists        /export, and everything below
#   index         corpus index loaded           /query, /documents
#   vector_store  Chroma holds chunks           /query
#   figure_guard  stored figures readable       /query (masking needs them, D41)
#
# NEVER CREATES A STORE. Opening a missing SQLite file or Chroma folder creates
# an empty one, which then looks present (found in step 4 against an empty
# checkout: two empty files appeared). Both paths are checked before anything
# is opened.
#
# ONCE READY, STAYS READY. A store that breaks mid-request raises inside the
# query path and the exception handlers in main.py answer 503 there. While NOT
# ready, every request may re-try the warm-up, at most once per
# READINESS_RETRY_SECONDS, under a lock so a burst of requests runs it once.
#
# The caches being warmed already exist at module level (load_index's
# _cached_index, get_store's singleton, load_guard_values' lru_cache). None of
# them caches a failure, so a re-try reads the stores again.
#
# Not for rebuilding the corpus under a running server: embed_documents would
# write to Chroma while the API reads it, and the local store is single-process.
# Stop the server, rebuild, start.

import asyncio
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError

from src.query.resolve import CorpusIndex, load_index
from src.rag.figure_guard import FigureGuardError, load_guard_values
from src.storage.vector_store import VectorStoreError, collection_count, get_store
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

DATABASE = "database"
INDEX = "index"
VECTOR_STORE = "vector_store"
FIGURE_GUARD = "figure_guard"
ALL_PARTS: frozenset[str] = frozenset({DATABASE, INDEX, VECTOR_STORE, FIGURE_GUARD})

# Labels for failed checks that are not exceptions.
DATABASE_NOT_FOUND = "DatabaseNotFound"
VECTOR_STORE_NOT_FOUND = "VectorStoreNotFound"
EMPTY_VECTOR_STORE = "EmptyVectorStore"


def database_present(path: str) -> bool:
    """
    Whether the SQLite file exists, checked without opening it.

    Args:
        path: settings.SQLITE_DB_PATH.

    Returns:
        True if it is a regular file.
    """
    return Path(path).is_file()


def vector_store_present(path: str) -> bool:
    """
    Whether the Chroma folder exists and holds anything, checked without opening it.

    Args:
        path: settings.CHROMA_DB_PATH.

    Returns:
        True if it is a non-empty directory.
    """
    folder = Path(path)
    return folder.is_dir() and any(folder.iterdir())


class Readiness:
    """
    Which parts are usable, and the corpus index once it has loaded.

    Attributes:
        index: The corpus index, or None until it loads.
        failures: Part -> exception class name or check label, for parts that
            are not usable. Empty means ready.
    """

    def __init__(
        self,
        retry_seconds: float = settings.READINESS_RETRY_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """
        Create an un-warmed state. Nothing is checked until warm() runs.

        Args:
            retry_seconds: Minimum gap between re-tries while not ready.
            clock: Monotonic time source; replaceable in tests.
        """
        self.index: CorpusIndex | None = None
        self.failures: dict[str, str] = {part: "NotChecked" for part in ALL_PARTS}
        self._retry_seconds = retry_seconds
        self._clock = clock
        self._last_attempt: float | None = None
        self._lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        """Whether every part is usable."""
        return not self.failures

    def missing(self, needs: Iterable[str]) -> list[str]:
        """
        The needed parts that are not usable.

        Args:
            needs: Parts an endpoint requires.

        Returns:
            Those that failed, sorted; empty when the endpoint can run.
        """
        return sorted(part for part in needs if part in self.failures)

    def _fail(self, part: str, error_type: str, detail: str) -> None:
        """
        Record one unusable part and log it.

        Args:
            part: Which part.
            error_type: Exception class name or check label.
            detail: What went wrong, for the log only.
        """
        self.failures[part] = error_type
        logger.error("readiness_part_unavailable", part=part, error_type=error_type, error=detail)

    async def warm(self) -> None:
        """
        Check every part from scratch and load what is present, off the event loop.

        Each part is tried on its own, so one failure does not hide another, and
        none of them raises: a missing part is recorded, never fatal (D54).
        """
        self._last_attempt = self._clock()
        self.failures = {}

        if not database_present(settings.SQLITE_DB_PATH):
            detail = f"no database file at {settings.SQLITE_DB_PATH}"
            for part in (DATABASE, INDEX, FIGURE_GUARD):
                self._fail(part, DATABASE_NOT_FOUND, detail)
        else:
            try:
                self.index = await asyncio.to_thread(load_index)
            except (RuntimeError, SQLAlchemyError) as exc:   # RuntimeError: empty corpus
                self._fail(INDEX, type(exc).__name__, str(exc))
            try:
                await asyncio.to_thread(load_guard_values)
            except FigureGuardError as exc:
                self._fail(FIGURE_GUARD, type(exc).__name__, str(exc))

        if not vector_store_present(settings.CHROMA_DB_PATH):
            self._fail(
                VECTOR_STORE, VECTOR_STORE_NOT_FOUND,
                f"no vector store at {settings.CHROMA_DB_PATH}; run scripts.embed_documents",
            )
        else:
            try:
                await asyncio.to_thread(get_store)
                chunks = await asyncio.to_thread(collection_count)   # -1 when unreadable
            except VectorStoreError as exc:
                self._fail(VECTOR_STORE, type(exc).__name__, str(exc))
            else:
                if chunks <= 0:
                    self._fail(
                        VECTOR_STORE, EMPTY_VECTOR_STORE,
                        f"collection holds {chunks} chunks; run scripts.embed_documents",
                    )

        logger.info(
            "readiness_checked",
            ready=self.ready,
            documents=len(self.index.entries) if self.index is not None else None,
            unavailable=sorted(self.failures),
        )

    async def refresh_if_due(self) -> None:
        """
        Re-try the warm-up when not ready and the retry interval has passed.

        Called by every request that needs a part. When ready it returns at
        once. The lock makes a burst of requests share one re-try; the second
        check inside it skips a re-try that another request just finished.
        """
        if self.ready or not self._due():
            return
        async with self._lock:
            if self.ready or not self._due():
                return
            await self.warm()

    def _due(self) -> bool:
        """Whether a re-try is allowed now."""
        return (
            self._last_attempt is None
            or self._clock() - self._last_attempt >= self._retry_seconds
        )
