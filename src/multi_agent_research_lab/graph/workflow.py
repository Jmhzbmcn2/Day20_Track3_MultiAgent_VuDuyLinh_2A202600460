"""LangGraph workflow for the multi-agent research system."""

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


def _state_to_dict(state: ResearchState) -> dict[str, Any]:
    """Convert ResearchState to dict for LangGraph."""
    return state.model_dump()


def _dict_to_state(d: dict[str, Any]) -> ResearchState:
    """Convert dict back to ResearchState."""
    return ResearchState.model_validate(d)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(self) -> None:
        self._supervisor = SupervisorAgent()
        self._researcher = ResearcherAgent()
        self._analyst = AnalystAgent()
        self._writer = WriterAgent()

    def build(self) -> StateGraph:
        """Create a LangGraph graph with supervisor routing to worker agents.

        Graph structure:
            START -> supervisor -> {researcher, analyst, writer, END}
            researcher -> supervisor
            analyst -> supervisor
            writer -> supervisor (if supervisor says done -> END)
        """
        # Define the graph using a dict-based state (LangGraph requirement)
        graph = StateGraph(dict)

        # ----- Add nodes -----
        graph.add_node("supervisor", self._supervisor_node)
        graph.add_node("researcher", self._researcher_node)
        graph.add_node("analyst", self._analyst_node)
        graph.add_node("writer", self._writer_node)

        # ----- Set entry point -----
        graph.set_entry_point("supervisor")

        # ----- Conditional edges from supervisor -----
        graph.add_conditional_edges(
            "supervisor",
            self._route_decision,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "done": END,
            },
        )

        # ----- Worker agents always route back to supervisor -----
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")

        return graph

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""
        logger.info("MultiAgentWorkflow.run | query=%s", state.request.query[:80])

        graph = self.build()
        compiled = graph.compile()

        # Convert state to dict for LangGraph
        initial_state = _state_to_dict(state)

        # Run the graph
        result = compiled.invoke(initial_state)

        # Convert result back to ResearchState
        final_state = _dict_to_state(result)

        logger.info("MultiAgentWorkflow.run | completed | iterations=%d | route_history=%s",
                     final_state.iteration, final_state.route_history)
        return final_state

    # ----- Node functions -----

    def _supervisor_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """Supervisor node: decides next route."""
        research_state = _dict_to_state(state)
        updated_state = self._supervisor.run(research_state)
        return _state_to_dict(updated_state)

    def _researcher_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """Researcher node: gathers sources and writes notes."""
        research_state = _dict_to_state(state)
        updated_state = self._researcher.run(research_state)
        return _state_to_dict(updated_state)

    def _analyst_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """Analyst node: analyzes research notes."""
        research_state = _dict_to_state(state)
        updated_state = self._analyst.run(research_state)
        return _state_to_dict(updated_state)

    def _writer_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """Writer node: produces final answer."""
        research_state = _dict_to_state(state)
        updated_state = self._writer.run(research_state)
        return _state_to_dict(updated_state)

    # ----- Routing function -----

    @staticmethod
    def _route_decision(state: dict[str, Any]) -> str:
        """Extract the last routing decision from the supervisor."""
        route_history = state.get("route_history", [])
        if route_history:
            last_route = route_history[-1]
            logger.info("Routing decision: %s", last_route)
            return last_route
        return "done"
