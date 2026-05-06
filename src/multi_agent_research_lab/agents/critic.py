"""Optional critic agent for fact-checking and quality review."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings.

        Checks:
        1. Fact-check claims against source material.
        2. Verify citation coverage.
        3. Identify potential hallucinations.
        """
        logger.info("CriticAgent.run | has_final_answer=%s", state.final_answer is not None)

        if not state.final_answer:
            state.errors.append("Critic has no final answer to review.")
            state.add_trace_event("critic", {"status": "no_answer"})
            return state

        system_prompt = """You are a Critic Agent. Your task is to review a final answer for:

1. **Factual Accuracy**: Are claims supported by the provided sources?
2. **Citation Coverage**: Are key claims properly cited?
3. **Hallucination Risk**: Any claims that appear unsupported?
4. **Completeness**: Does the answer address the original query fully?
5. **Quality Score**: Rate the answer from 0-10.

Provide a structured review with a final quality score (0-10) on the last line as:
QUALITY_SCORE: X"""

        source_text = "\n".join(
            f"[{i}] {s.title}: {s.snippet[:200]}" for i, s in enumerate(state.sources, 1)
        )

        user_prompt = f"""Original query: {state.request.query}

Final answer to review:
{state.final_answer}

Available sources:
{source_text}

Provide your critical review."""

        try:
            response = self._llm.complete(system_prompt, user_prompt)
            review = response.content

            state.agent_results.append(AgentResult(
                agent=AgentName.CRITIC,
                content=review,
                metadata={"review_length": len(review)},
            ))

            state.add_trace_event("critic", {
                "review_length": len(review),
            })

        except Exception as e:
            logger.exception("Critic LLM review failed.")
            state.errors.append(f"Critic LLM error: {e}")

        logger.info("CriticAgent.run | review complete")
        return state
