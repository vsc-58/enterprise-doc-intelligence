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
        ValidationError: If required fields (OPENAI_API_KEY, EDGAR_USER_AGENT)
                         are missing from the environment or .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # Required — no defaults, will raise ValidationError if missing
    OPENAI_API_KEY: str
    EDGAR_USER_AGENT: str

    # Optional — defaults provided
    LOG_LEVEL: str = "INFO"
    CHROMA_DB_PATH: str = "db/chroma"
    SQLITE_DB_PATH: str = "db/metadata.db"
    PROCESSED_DATA_PATH: str = "data/processed"
    RAW_DATA_PATH: str = "data/raw"


# Singleton instance — import this everywhere
# from src.utils.config import settings
settings = Settings()