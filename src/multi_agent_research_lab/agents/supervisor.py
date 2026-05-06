"""Supervisor / router agent."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self) -> None:
        self._llm = LLMClient()
        self._settings = get_settings()

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route.

        Routing policy:
        - If no research_notes → route to researcher
        - If research_notes but no analysis_notes → route to analyst
        - If analysis_notes but no final_answer → route to writer
        - If final_answer exists → done
        - Enforce max iterations
        """
        logger.info("SupervisorAgent.run | iteration=%d | history=%s",
                     state.iteration, state.route_history)

        # Guardrail: max iterations
        if state.iteration >= self._settings.max_iterations:
            logger.warning("Max iterations reached (%d). Forcing done.", state.iteration)
            state.record_route("done")
            state.add_trace_event("supervisor", {"decision": "done", "reason": "max_iterations"})
            return state

        # Use LLM to decide the next step based on current state
        system_prompt = """You are a Supervisor agent that routes tasks in a multi-agent research system.
Your job is to decide which agent should act next based on the current state.

Available agents:
- researcher: Searches the web and gathers source material. Use when we need information.
- analyst: Analyzes research notes to extract key claims and insights. Use when we have research notes but need analysis.
- writer: Writes the final answer from research and analysis notes. Use when we have both research and analysis.
- done: The workflow is complete. Use when we already have a final answer.

Respond with ONLY one word: researcher, analyst, writer, or done."""

        state_summary = f"""Current state:
- Query: {state.request.query}
- Iteration: {state.iteration}/{self._settings.max_iterations}
- Route history: {state.route_history}
- Has research notes: {state.research_notes is not None}
- Has analysis notes: {state.analysis_notes is not None}
- Has final answer: {state.final_answer is not None}
- Number of sources: {len(state.sources)}
- Errors: {state.errors}"""

        try:
            response = self._llm.complete(system_prompt, state_summary)
            decision = response.content.strip().lower().split()[0] if response.content.strip() else "done"

            # Validate decision
            valid_routes = {"researcher", "analyst", "writer", "done"}
            if decision not in valid_routes:
                logger.warning("Invalid route '%s', falling back to heuristic.", decision)
                decision = self._heuristic_route(state)
        except Exception as e:
            logger.exception("LLM routing failed, using heuristic fallback.")
            state.errors.append(f"Supervisor LLM error: {e}")
            decision = self._heuristic_route(state)

        state.record_route(decision)
        state.add_trace_event("supervisor", {
            "decision": decision,
            "iteration": state.iteration,
        })

        state.agent_results.append(AgentResult(
            agent=AgentName.SUPERVISOR,
            content=f"Routed to: {decision}",
            metadata={"iteration": state.iteration},
        ))

        logger.info("SupervisorAgent.run | decision=%s", decision)
        return state

    @staticmethod
    def _heuristic_route(state: ResearchState) -> str:
        """Deterministic fallback routing when LLM fails."""
        if state.research_notes is None:
            return "researcher"
        if state.analysis_notes is None:
            return "analyst"
        if state.final_answer is None:
            return "writer"
        return "done"
