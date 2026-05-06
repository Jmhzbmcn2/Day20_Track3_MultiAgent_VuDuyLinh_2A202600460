"""Benchmark skeleton for single-agent vs multi-agent."""

import logging
from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]


def run_benchmark(run_name: str, query: str, runner: Runner) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, estimate cost, and return metrics.

    Collects:
    - Latency (wall-clock time)
    - Estimated cost from trace events
    - Quality score placeholder
    - Notes about the run
    """
    logger.info("Benchmark | run=%s | query=%s", run_name, query[:80])

    started = perf_counter()
    try:
        state = runner(query)
        error_occurred = False
    except Exception as e:
        logger.exception("Benchmark runner failed: %s", e)
        state = ResearchState(
            request={"query": query, "max_sources": 5, "audience": "technical learners"}
        )
        state.errors.append(f"Runner failed: {e}")
        error_occurred = True

    latency = perf_counter() - started

    # Estimate cost from trace events
    total_cost = 0.0
    for event in state.trace:
        payload = event.get("payload", {})
        if "cost_usd" in payload and payload["cost_usd"] is not None:
            total_cost += payload["cost_usd"]

    # Count total tokens
    total_input_tokens = 0
    total_output_tokens = 0
    for event in state.trace:
        payload = event.get("payload", {})
        if "input_tokens" in payload and payload["input_tokens"] is not None:
            total_input_tokens += payload["input_tokens"]
        if "output_tokens" in payload and payload["output_tokens"] is not None:
            total_output_tokens += payload["output_tokens"]

    # Build notes
    notes_parts = []
    if state.route_history:
        notes_parts.append(f"Route: {' -> '.join(state.route_history)}")
    notes_parts.append(f"Sources: {len(state.sources)}")
    notes_parts.append(f"Tokens: {total_input_tokens}+{total_output_tokens}")
    if state.errors:
        notes_parts.append(f"Errors: {len(state.errors)}")
    if error_occurred:
        notes_parts.append("FAILED")

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=total_cost if total_cost > 0 else None,
        notes=" | ".join(notes_parts),
    )

    logger.info("Benchmark | run=%s | latency=%.2fs | cost=$%.6f",
                run_name, latency, total_cost)

    return state, metrics
