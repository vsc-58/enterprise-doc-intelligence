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

    # Minimum number of fields with a checkable citation before the scale-coherence
    # check runs (D29). Below this, one anomalous field could define the document
    # scale by itself, so the check abstains rather than asserting. Three is the
    # smallest count where a single minority field is still a minority; INTC, the
    # thinnest record in the corpus, has exactly three.
    SCALE_COHERENCE_MIN_FIELDS: int = 3

    # Hard ceiling on rows a ranking or filter may return. The corpus holds 20
    # companies, so this is not a performance guard — it stops a malformed plan
    # from rendering an answer longer than anyone will read.
    SQL_RESULT_LIMIT_CAP: int = 20

    # --- Phase 5B: narrative retrieval ----------------------------------------
    # Chunks per single-company narrative task. Fixed across intents: an
    # intent's sections are a union ranked by distance alone, so k does not grow
    # with section count (D38). At ~402 tokens/chunk this is ~2k tokens of
    # context, about a quarter of a typical company's Item 1 — the constraint is
    # distraction, not cost.
    RAG_TOP_K: int = 5

    # Chunks per company for corpus-wide narrative tasks ("which companies...").
    # Each eligible company is searched separately so the largest filers cannot
    # crowd the rest out: plain top-50 reached 16 companies with BAC+GS holding
    # 28 of the 50 slots; per-company reached every company holding the
    # sections (D39).
    RAG_PER_COMPANY_K: int = 2

    # Ceiling on concurrent per-company verdict calls. Twenty unbounded calls
    # work today; the cap is what keeps a larger corpus inside OpenAI's
    # per-minute limits rather than failing on them.
    RAG_VERDICT_CONCURRENCY: int = 5

    # Relative tolerance for recognising a stored metric value in text (D41).
    # Wide enough for an answer's rounding ("$365.8 billion" vs 365,817,000,000
    # is 0.005%); narrow enough that a different figure rarely falls inside it,
    # since matching is scoped to one company's five values.
    FIGURE_GUARD_REL_TOLERANCE: float = 0.005
    
# Singleton instance — import this everywhere
# from src.utils.config import settings
settings = Settings()