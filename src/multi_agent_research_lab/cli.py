"""Command-line entrypoint for the lab starter."""

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import configure_tracing
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_tracing()


def _run_baseline(query: str) -> ResearchState:
    """Run a single-agent baseline: one LLM call to answer the query directly."""
    llm = LLMClient()
    state = ResearchState(request=ResearchQuery(query=query))

    system_prompt = """You are a research assistant. Answer the user's research query thoroughly
and accurately. Provide a well-structured response of approximately 500 words with clear
sections and factual information. If you're unsure about something, say so."""

    response = llm.complete(system_prompt, query)
    state.final_answer = response.content
    state.add_trace_event("baseline", {
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "cost_usd": response.cost_usd,
    })
    return state


def _run_multi_agent(query: str) -> ResearchState:
    """Run the multi-agent workflow."""
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    return workflow.run(state)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a minimal single-agent baseline with real LLM."""

    _init()
    console.print("[bold blue]Running Single-Agent Baseline...[/bold blue]")

    state = _run_baseline(query)

    console.print(Panel.fit(state.final_answer or "No answer produced.", title="Single-Agent Baseline"))

    # Print trace info
    if state.trace:
        trace_info = state.trace[0].get("payload", {})
        console.print(f"\n[dim]Tokens: in={trace_info.get('input_tokens', 'N/A')}, "
                      f"out={trace_info.get('output_tokens', 'N/A')} | "
                      f"Cost: ${trace_info.get('cost_usd', 'N/A')}[/dim]")


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""

    _init()
    console.print("[bold green]Running Multi-Agent Workflow...[/bold green]")

    state = _run_multi_agent(query)

    console.print(Panel.fit(state.final_answer or "No answer produced.", title="Multi-Agent Result"))

    # Print summary
    console.print(f"\n[dim]Iterations: {state.iteration} | "
                  f"Route: {' -> '.join(state.route_history)} | "
                  f"Sources: {len(state.sources)} | "
                  f"Errors: {len(state.errors)}[/dim]")


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    output: Annotated[str, typer.Option("--output", "-o", help="Output report path")] = "reports/benchmark_report.md",
) -> None:
    """Run both baseline and multi-agent, then generate a comparison benchmark report."""

    _init()
    console.print("[bold yellow]Running Benchmark: Single-Agent vs Multi-Agent...[/bold yellow]")

    all_metrics: list[BenchmarkMetrics] = []

    # Run baseline
    console.print("\n[bold blue]>>> Running Baseline...[/bold blue]")
    baseline_state, baseline_metrics = run_benchmark("single-agent-baseline", query, _run_baseline)
    all_metrics.append(baseline_metrics)
    console.print(f"  Baseline done. Latency: {baseline_metrics.latency_seconds:.2f}s")

    # Run multi-agent
    console.print("\n[bold green]>>> Running Multi-Agent...[/bold green]")
    multi_state, multi_metrics = run_benchmark("multi-agent-workflow", query, _run_multi_agent)
    all_metrics.append(multi_metrics)
    console.print(f"  Multi-Agent done. Latency: {multi_metrics.latency_seconds:.2f}s")

    # Generate report
    report = render_markdown_report(all_metrics, baseline_state, multi_state, query)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    console.print(f"\n[bold]Benchmark report saved to: {output_path}[/bold]")
    console.print(Panel.fit(report[:2000], title="Benchmark Report Preview"))


if __name__ == "__main__":
    app()
