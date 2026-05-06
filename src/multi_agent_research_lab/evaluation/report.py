"""Benchmark report rendering."""

from datetime import datetime, timezone

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState


def render_markdown_report(
    metrics: list[BenchmarkMetrics],
    baseline_state: ResearchState | None = None,
    multi_state: ResearchState | None = None,
    query: str = "",
) -> str:
    """Render benchmark metrics to a rich markdown report.

    Includes:
    - Metrics comparison table
    - Trace analysis
    - Quality analysis
    - Failure modes and observations
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Benchmark Report: Single-Agent vs Multi-Agent",
        "",
        f"> Generated: {now}",
        "",
        "## Query",
        "",
        f"```\n{query}\n```" if query else "_No query specified._",
        "",
        "## Metrics Comparison",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Notes |",
        "|---|---:|---:|---:|---|",
    ]

    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"${item.estimated_cost_usd:.6f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        lines.append(f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} | {item.notes} |")

    lines.append("")

    # Latency comparison
    if len(metrics) >= 2:
        baseline_latency = metrics[0].latency_seconds
        multi_latency = metrics[1].latency_seconds
        speedup = baseline_latency / multi_latency if multi_latency > 0 else 0
        lines.extend([
            "## Latency Analysis",
            "",
            f"- **Single-Agent**: {baseline_latency:.2f}s",
            f"- **Multi-Agent**: {multi_latency:.2f}s",
            f"- **Ratio**: {speedup:.2f}x {'(baseline faster)' if speedup > 1 else '(multi-agent faster)'}",
            "",
        ])

    # Trace analysis for multi-agent
    if multi_state and multi_state.route_history:
        lines.extend([
            "## Multi-Agent Trace",
            "",
            f"- **Iterations**: {multi_state.iteration}",
            f"- **Route History**: {' -> '.join(multi_state.route_history)}",
            f"- **Sources Found**: {len(multi_state.sources)}",
            f"- **Agent Results**: {len(multi_state.agent_results)}",
            "",
        ])

        if multi_state.sources:
            lines.append("### Sources Used")
            lines.append("")
            for i, src in enumerate(multi_state.sources, 1):
                lines.append(f"{i}. [{src.title}]({src.url})" if src.url else f"{i}. {src.title}")
            lines.append("")

    # Errors
    all_errors = []
    if baseline_state and baseline_state.errors:
        all_errors.extend([f"[Baseline] {e}" for e in baseline_state.errors])
    if multi_state and multi_state.errors:
        all_errors.extend([f"[Multi-Agent] {e}" for e in multi_state.errors])

    if all_errors:
        lines.extend([
            "## Errors & Failure Modes",
            "",
        ])
        for err in all_errors:
            lines.append(f"- {err}")
        lines.append("")

    # Answer previews
    lines.extend([
        "## Answer Comparison",
        "",
        "### Single-Agent Answer",
        "",
    ])
    if baseline_state and baseline_state.final_answer:
        # Truncate for report
        answer = baseline_state.final_answer[:1500]
        lines.append(answer)
        if len(baseline_state.final_answer) > 1500:
            lines.append("\n_...truncated..._")
    else:
        lines.append("_No answer produced._")

    lines.extend([
        "",
        "### Multi-Agent Answer",
        "",
    ])
    if multi_state and multi_state.final_answer:
        answer = multi_state.final_answer[:1500]
        lines.append(answer)
        if len(multi_state.final_answer) > 1500:
            lines.append("\n_...truncated..._")
    else:
        lines.append("_No answer produced._")

    # Observations
    lines.extend([
        "",
        "## Observations & Failure Mode Analysis",
        "",
        "### When to use Multi-Agent?",
        "",
        "- Complex research queries requiring multiple sources",
        "- Tasks needing different skill sets (search, analysis, writing)",
        "- When quality and citation coverage matter more than speed",
        "",
        "### When NOT to use Multi-Agent?",
        "",
        "- Simple factual questions that a single LLM call can answer",
        "- Latency-critical applications",
        "- When cost is a major constraint (multi-agent uses more tokens)",
        "",
        "### Failure Modes Observed",
        "",
        "- **Routing loops**: Supervisor may cycle if state checks are imprecise",
        "- **Search failures**: Tavily API may timeout or return no results",
        "- **Token limits**: Long research notes may exceed model context",
        "- **Quality variance**: LLM output quality varies between runs",
        "",
    ])

    return "\n".join(lines) + "\n"
