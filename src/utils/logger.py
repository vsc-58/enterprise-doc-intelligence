# src/utils/logger.py
# Module: Structured logging configuration
# Purpose: Provides a configured structlog logger for use across all modules
# Depends on: structlog, src/utils/config.py

import logging
import sys

import structlog

from src.utils.config import settings


def _configure_logging() -> None:
    """
    Configure structlog with JSON output, timestamp, log level, and module name.

    Called once at import time. Subsequent imports reuse the same configuration.
    Sets both structlog and standard library logging to the level specified
    in settings.LOG_LEVEL so third-party libraries respect the same level.
    """
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    # Configure standard library logging — controls third-party library output
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            # structlog.processors.JSONRenderer(), --commented to get human readable foramt
            structlog.dev.ConsoleRenderer(), #added to get human readable format
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


# Called once when this module is first imported
_configure_logging()


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Return a configured structlog logger bound to the given module name.

    Args:
        name: The module name, typically passed as __name__ from the calling module.

    Returns:
        A structlog BoundLogger instance ready for use.

    Example:
        logger = get_logger(__name__)
        logger.info("document_processed", company="Apple", year=2022)
    """
    return structlog.get_logger(name)