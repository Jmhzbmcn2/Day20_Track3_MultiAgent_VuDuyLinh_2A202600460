"""Analyst agent."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`.

        Steps:
        1. Extract key claims from research notes.
        2. Compare viewpoints and identify consensus/disagreements.
        3. Flag weak evidence or gaps.
        """
        logger.info("AnalystAgent.run | has_research_notes=%s", state.research_notes is not None)

        if not state.research_notes:
            state.errors.append("Analyst has no research notes to analyze.")
            state.analysis_notes = "No research notes available for analysis."
            state.add_trace_event("analyst", {"status": "no_input"})
            return state

        system_prompt = """You are an Analyst Agent. Your task is to critically analyze research notes
and produce structured analytical insights. Your analysis should include:

1. **Key Claims**: List the main claims/findings from the research.
2. **Evidence Strength**: Rate each claim's evidence as Strong/Moderate/Weak.
3. **Consensus vs Disagreement**: Note where sources agree or disagree.
4. **Gaps & Limitations**: Identify missing information or research gaps.
5. **Synthesis**: Provide an overall analytical summary.

Be objective, precise, and cite sources using [N] format where applicable."""

        user_prompt = f"""Research query: {state.request.query}

Research notes to analyze:
{state.research_notes}

Number of sources: {len(state.sources)}
Source titles: {[s.title for s in state.sources]}

Provide your structured analysis."""

        try:
            response = self._llm.complete(system_prompt, user_prompt)
            state.analysis_notes = response.content
        except Exception as e:
            logger.exception("Analyst LLM analysis failed.")
            state.errors.append(f"Analyst LLM error: {e}")
            state.analysis_notes = "Analysis failed. Using research notes as fallback."

        state.agent_results.append(AgentResult(
            agent=AgentName.ANALYST,
            content=state.analysis_notes or "",
            metadata={
                "research_notes_length": len(state.research_notes or ""),
            },
        ))

        state.add_trace_event("analyst", {
            "analysis_length": len(state.analysis_notes or ""),
        })

        logger.info("AnalystAgent.run | analysis_len=%d", len(state.analysis_notes or ""))
        return state
