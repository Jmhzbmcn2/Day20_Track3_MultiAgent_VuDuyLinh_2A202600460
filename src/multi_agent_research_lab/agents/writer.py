"""Writer agent."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`.

        Steps:
        1. Synthesize research notes and analysis into a coherent response.
        2. Include source references/citations.
        3. Structure for the target audience.
        """
        logger.info("WriterAgent.run | has_research=%s | has_analysis=%s",
                     state.research_notes is not None, state.analysis_notes is not None)

        system_prompt = f"""You are a Writer Agent. Your task is to produce a well-structured, 
comprehensive final answer to the user's research question.

Guidelines:
1. Synthesize information from both research notes and analysis.
2. Include source citations in [N] format.
3. Write for the target audience: {state.request.audience}.
4. Structure with clear headings and paragraphs.
5. Aim for approximately 500 words.
6. End with a brief conclusion.

Write clearly, accurately, and engagingly."""

        # Build context from available notes
        context_parts = [f"Original query: {state.request.query}"]

        if state.research_notes:
            context_parts.append(f"\n--- RESEARCH NOTES ---\n{state.research_notes}")
        if state.analysis_notes:
            context_parts.append(f"\n--- ANALYSIS ---\n{state.analysis_notes}")
        if state.sources:
            source_list = "\n".join(
                f"[{i}] {s.title} - {s.url}" for i, s in enumerate(state.sources, 1)
            )
            context_parts.append(f"\n--- SOURCES ---\n{source_list}")

        user_prompt = "\n".join(context_parts) + "\n\nWrite the final comprehensive answer."

        try:
            response = self._llm.complete(system_prompt, user_prompt)
            state.final_answer = response.content
        except Exception as e:
            logger.exception("Writer LLM synthesis failed.")
            state.errors.append(f"Writer LLM error: {e}")
            # Fallback: combine available notes
            state.final_answer = (
                f"## Answer (fallback)\n\n"
                f"Research Notes:\n{state.research_notes or 'N/A'}\n\n"
                f"Analysis:\n{state.analysis_notes or 'N/A'}"
            )

        state.agent_results.append(AgentResult(
            agent=AgentName.WRITER,
            content=state.final_answer or "",
            metadata={
                "word_count": len((state.final_answer or "").split()),
            },
        ))

        state.add_trace_event("writer", {
            "answer_length": len(state.final_answer or ""),
            "word_count": len((state.final_answer or "").split()),
        })

        logger.info("WriterAgent.run | answer_len=%d", len(state.final_answer or ""))
        return state
