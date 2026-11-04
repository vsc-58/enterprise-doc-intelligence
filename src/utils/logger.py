"""Structured logging: one structlog configuration and get_logger() for every module.

Log events are key-value pairs, not formatted sentences, so they can be
searched and filtered. Values bound with structlog.contextvars (the API's
request id) appear in every line written while serving that request.

Two output formats, chosen by settings.LOG_FORMAT:
  console  coloured, human-readable lines for a terminal (the default)
  json     one JSON object per line for a log collector. Google Cloud Logging
           reads "severity" (INFO, WARNING, ...) and "message" from such a line;
           the colour codes of the console format would show up there as text.
"""

import logging
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog
from structlog.typing import Processor

from src.utils.config import settings

# Processors shared by both formats, in order. Everything the event carries is
# added here; the format only decides how the finished event is written.
_SHARED: list[Processor] = [
    # Adds values bound with structlog.contextvars.bind_contextvars — the API
    # binds request_id once per request, so every log line written while
    # serving it carries the id. A no-op for scripts.
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
]


def add_severity(
    logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Copy the log level into "severity", upper-case, for Google Cloud Logging.

    Cloud Logging sets an entry's severity from a top-level "severity" key in
    a JSON line (INFO, WARNING, ERROR, ...). Without it every line would show
    as "default" severity, and filtering for errors would find nothing.

    Args:
        logger: The wrapped logger (unused; part of the processor signature).
        method_name: The log method called (unused).
        event_dict: The event so far; must already carry "level".

    Returns:
        The event with "severity" added.
    """
    event_dict["severity"] = str(event_dict.get("level", "info")).upper()
    return event_dict


def build_processors(log_format: str) -> list[Processor]:
    """The processor chain for one output format.

    Args:
        log_format: "console" or "json" (settings.LOG_FORMAT).

    Returns:
        The shared processors followed by the format's own renderer.

    Raises:
        ValueError: If the format is not one of the two.
    """
    if log_format == "console":
        return [*_SHARED, structlog.dev.ConsoleRenderer()]
    if log_format == "json":
        return [
            *_SHARED,
            add_severity,
            # Cloud Logging shows "message" as the entry's summary line.
            structlog.processors.EventRenamer("message"),
            structlog.processors.JSONRenderer(),
        ]
    raise ValueError(f"unknown log format {log_format!r}; use 'console' or 'json'")


def _configure_logging() -> None:
    """Configure structlog and the standard library logging level.

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

    structlog.configure(
        processors=build_processors(settings.LOG_FORMAT),
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


# Called once when this module is first imported
_configure_logging()


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a configured structlog logger bound to the given module name.

    Args:
        name: The module name, typically passed as __name__ from the calling module.

    Returns:
        A structlog BoundLogger instance ready for use.

    Example:
        logger = get_logger(__name__)
        logger.info("document_processed", company="Apple", year=2022)
    """
    return structlog.get_logger(name)
