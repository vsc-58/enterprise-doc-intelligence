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
    ITEM8_MAX_TOKENS: int = 60000            # above = over-capture; GS hit 119591
    MODEL_INPUT_TOKEN_BUDGET: int = 120000   # leave headroom under gpt-4o-mini's 128k for output
    COVER_MAX_CHARS: int = 12000

    # error tolerance threshold
    NUMERIC_REL_TOLERANCE: float = 1e-5   # tight: catches digit-level misreads, absorbs true rounding


# Singleton instance — import this everywhere
# from src.utils.config import settings
settings = Settings()