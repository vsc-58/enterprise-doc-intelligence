# src/storage/metadata_store.py
# Module: Metadata store (SQLite via SQLAlchemy 2.x)
# Purpose: Defines the Document tracking table and the engine/session machinery
#          used across the project to record and look up acquired filings.
# Depends on: sqlalchemy>=2.0, src.utils.config, src.utils.logger

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    Boolean,
    DateTime,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Declarative base for all ORM models in the metadata store."""

    pass


class Document(Base):
    """
    Tracking record for one acquired 10-K filing.

    One row per (company, fiscal year). The (cik, filing_year) pair is the
    domain key — a company files exactly one 10-K per fiscal year — and is
    enforced as a unique constraint so duplicate acquisition is rejected at
    the database level regardless of which code path writes.

    The two boolean flags (is_extracted, is_embedded) track two independent
    downstream processes — extraction (Phase 3) and embedding (Phase 4) —
    that both read the acquired text. They are deliberately separate flags
    rather than a single linear status, because a document can be embedded
    but not yet extracted, or vice versa.

    Attributes:
        id: Surrogate primary key.
        company_name: Human-readable company name from the filing.
        ticker: Stock ticker used to request the filing.
        cik: EDGAR Central Index Key, stored as a string to preserve the
             canonical 10-digit zero-padded form (int would drop leading zeros).
        filing_year: Fiscal year the 10-K covers.
        accession_number: EDGAR's unique submission ID. Stored for reference
             only — not uniquely constrained, since the access pattern is by
             (cik, filing_year).
        local_path: Path to the saved raw filing text on disk.
        is_extracted: Set True by the Phase 3 extraction pipeline.
        is_embedded: Set True by the Phase 4 embedding pipeline.
        created_at: UTC timestamp when the row was created.
    """

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("cik", "filing_year", name="uq_document_cik_year"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    company_name: Mapped[str] = mapped_column(String, nullable=False)
    ticker: Mapped[str] = mapped_column(String, nullable=False)
    cik: Mapped[str] = mapped_column(String, nullable=False)
    filing_year: Mapped[int] = mapped_column(Integer, nullable=False)
    accession_number: Mapped[str | None] = mapped_column(String, nullable=True)
    local_path: Mapped[str | None] = mapped_column(String, nullable=True)
    is_extracted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    is_embedded: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    def __repr__(self) -> str:
        return (
            f"Document(id={self.id}, ticker={self.ticker!r}, "
            f"cik={self.cik!r}, filing_year={self.filing_year})"
        )


# Engine: one per process, pointed at the configured SQLite file.
# echo=False — SQL logging is handled by structlog at the call sites, not by
# SQLAlchemy's own logger.
_engine = create_engine(
    f"sqlite:///{settings.SQLITE_DB_PATH}",
    echo=False,
)

# Session factory. expire_on_commit=False keeps attributes accessible on
# returned objects after the session commits and closes, so callers can read
# e.g. doc.id without triggering a lazy reload against a closed session.
_SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False)


def init_db() -> None:
    """
    Create the metadata database and all tables if they do not exist.

    Idempotent: SQLAlchemy issues CREATE TABLE IF NOT EXISTS, so repeated
    calls never drop or modify existing data. Also ensures the parent
    directory of the SQLite file exists, so a fresh clone (where db/ is
    gitignored) works without manual setup.

    Must be called explicitly once at startup by the caller that owns
    process lifecycle — the acquisition script in Phase 1, the FastAPI
    lifespan manager in Phase 6. It is deliberately NOT called at import
    time: importing this module must have no side effects (no file writes).

    Returns:
        None
    """
    db_path = Path(settings.SQLITE_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(_engine)
    logger.info("metadata_db_initialised", path=settings.SQLITE_DB_PATH)


def document_exists(ticker: str, year: int) -> bool:
    """
    Return True if a Document row already exists for this (ticker, year).

    The pipeline's pre-flight skip check: a local SQLite read run BEFORE any
    network fetch, so an already-acquired document costs a query, not a filing
    download. Keyed on (ticker, filing_year) — the key the pipeline holds at
    the top of its loop. The hard uniqueness guarantee remains the
    (cik, filing_year) table constraint underneath.

    Args:
        ticker: Stock ticker symbol.
        year: Filing year (period-end convention).

    Returns:
        True if a matching row exists, else False.
    """
    with get_session() as session:
        stmt = select(Document.id).where(
            Document.ticker == ticker,
            Document.filing_year == year,
        )
        return session.execute(stmt).first() is not None


def create_document(
    *,
    company_name: str,
    ticker: str,
    cik: str,
    filing_year: int,
    accession_number: str | None,
    local_path: str,
) -> int:
    """
    Persist one Document tracking row and return its primary key.

    Called only after BOTH text acquisition and ground-truth capture succeed —
    the committed row is the success signal under the implicit-completeness
    model (no acquisition-status column; the row's existence means the
    document is fully acquired). is_extracted and is_embedded stay at their
    defaults (False); Phases 3 and 4 set them. Keyword-only arguments prevent
    positional mistakes among the several string fields.

    Args:
        company_name: Company name from the filing.
        ticker: Requested ticker.
        cik: Zero-padded 10-digit CIK.
        filing_year: Period-end year.
        accession_number: EDGAR accession, or None.
        local_path: Path to the saved raw-text file.

    Returns:
        The new row's integer id.

    Raises:
        IntegrityError: If a row for (cik, filing_year) already exists — the
                        unique constraint surfacing a skip-check miss or a
                        concurrent insert, rather than silently duplicating.
    """
    with get_session() as session:
        doc = Document(
            company_name=company_name,
            ticker=ticker,
            cik=cik,
            filing_year=filing_year,
            accession_number=accession_number,
            local_path=local_path,
        )
        session.add(doc)
        session.flush()  # assigns the autoincrement id within the transaction
        return doc.id


@contextmanager
def get_session() -> Iterator[Session]:
    """
    Provide a transactional database session as a context manager.

    Scope is one unit of work — in Phase 1, one document. On clean exit the
    transaction is committed; on any exception it is rolled back and the
    exception re-raised for the caller to handle. The session is always
    closed. This per-unit transaction boundary is what lets the acquisition
    pipeline isolate failures: one document's failed insert rolls back only
    that document, leaving already-committed documents untouched.

    Yields:
        Session: an open SQLAlchemy 2.x session.

    Raises:
        Exception: re-raises whatever was raised inside the with-block,
                   after rolling back the transaction.
    """
    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()