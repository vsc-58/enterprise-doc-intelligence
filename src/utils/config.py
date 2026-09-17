# src/utils/config.py
# Module: Application configuration
# Purpose: Single source of truth for all settings loaded from environment variables
# Depends on: pydantic-settings, python-dotenv, .env file in project root

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.

    All configuration in the project flows through this class.
    Never hardcode values that belong here — always import from this module.

    Raises:
        ValidationError: If required fields (OPENAI_API_KEY, EDGAR_IDENTITY)
                         are missing from the environment or .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # Required — no defaults, will raise ValidationError at startup if missing
    OPENAI_API_KEY: str
    EDGAR_IDENTITY: str

    # Models
    OPENAI_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Logging
    LOG_LEVEL: str = "INFO"

    # Storage paths
    CHROMA_DB_PATH: str = "db/chroma"
    SQLITE_DB_PATH: str = "db/metadata.db"
    RAW_DATA_PATH: str = "data/raw"
    EVAL_DATA_PATH: str = "data/eval"

    # Secion characters upper and lower limits
    ITEM8_MIN_CHARS: int = 26136             # edgartools' floor; below = truncated heading
    # removed in full extraction to incorporate GS, BAC and PFE
    # ITEM8_MAX_TOKENS: int = 60000            # above = over-capture; GS hit 119591
    MODEL_INPUT_TOKEN_BUDGET: int = 120000   # leave headroom under gpt-4o-mini's 128k for output
    COVER_MAX_CHARS: int = 12000

    # error tolerance threshold
    NUMERIC_REL_TOLERANCE: float = 1e-5   # tight: catches digit-level misreads, absorbs true rounding

    # Tolerance for matching an extracted value against the numbers printed in
    # its cited line. Looser than NUMERIC_REL_TOLERANCE (1e-5) on purpose: that
    # one compares two full-precision figures, this one compares a full value
    # against a figure the filing printed rounded to millions, so a legitimate
    # rounding gap of a few parts per million is expected.
    EVIDENCE_MATCH_REL_TOLERANCE: float = 1e-3

    # Floor rejecting collapsed item slices. Set from measurement, not judgment:
    # observed collapses are 45-157 tokens; the smallest real section is 596.
    # Nothing sits between, so the threshold is in a genuine gap rather than
    # tuned until the alarm stopped (the Phase 3 D14 lesson). Also rejects
    # legitimate cross-references — BAC/GS/JPM incorporate Item 7A into Item 7,
    # so their 45-63 token 7A is a pointer, not a truncation. Same handling,
    # different meaning.
    NARRATIVE_MIN_TOKENS: int = 200

    # Mid-slice sample size for the section overlap test. Sampled from the
    # middle because adjacent items legitimately share a boundary sentence — an
    # edge match proves nothing, a mid-slice match means real containment.
    # Fires on 3/20: QCOM Item 1, META Item 1, TSLA Item 7A.
    SECTION_OVERLAP_PROBE_CHARS: int = 400

    # --- Phase 4: narrative sections, chunking & vector store -----------------
    # Section artifacts written by scripts/build_narrative_sections.py. Kept
    # separate from RAW_DATA_PATH because these are the item view's rendering of
    # the filing, not filing.text() — the two differ (Phase 3 D14) and must not
    # be confused for each other.
    PROCESSED_DATA_PATH: str = "data/processed"

    # Chunk target and overlap, in TOKENS under cl100k_base — the encoding
    # text-embedding-3-small uses. Note this differs from the o200k_base used in
    # sections.py / narrative_sections.py, which budget against gpt-4o-mini's
    # context window. Different consumers, different encodings.
    # RecursiveCharacterTextSplitter counts CHARACTERS by default, so the
    # chunker must build it via from_tiktoken_encoder or 500 silently means 500
    # chars (~125 tokens).
    CHUNK_SIZE_TOKENS: int = 500
    CHUNK_OVERLAP_TOKENS: int = 50

    # Length floor dropping fragments too short to retrieve usefully (heading
    # remnants, stray table rows). LENGTH ONLY, deliberately: any content-based
    # filter — digit density, symbol ratio — is the filter that would delete
    # real financial text.
    CHUNK_MIN_CHARS: int = 100

    # Embedding batch size and inter-batch pause. Batching bounds the blast
    # radius of a failed call; the pause keeps a ~1,750-chunk run inside rate
    # limits.
    EMBED_BATCH_SIZE: int = 100
    EMBED_BATCH_SLEEP_SECONDS: float = 0.5

# Singleton instance — import this everywhere
# from src.utils.config import settings
settings = Settings()