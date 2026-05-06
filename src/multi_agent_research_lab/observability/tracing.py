"""Tracing hooks.

Supports LangSmith tracing when LANGSMITH_API_KEY is configured,
otherwise falls back to simple perf_counter-based spans.
"""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)


def configure_tracing() -> None:
    """Configure LangSmith tracing if API key is available."""
    settings = get_settings()

    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        logger.info("LangSmith tracing enabled for project: %s", settings.langsmith_project)
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        logger.info("LangSmith tracing disabled (no API key). Using local spans only.")


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal span context with timing.

    When LangSmith is configured, LangGraph will automatically create trace spans.
    This context manager provides additional local timing for custom instrumentation.
    """
    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}

    logger.debug("Span started: %s | attrs=%s", name, attributes)
    try:
        yield span
    finally:
        span["duration_seconds"] = perf_counter() - started
        logger.debug("Span ended: %s | duration=%.3fs", name, span["duration_seconds"])
