"""Researcher agent."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self) -> None:
        self._llm = LLMClient()
        self._search = SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`.

        Steps:
        1. Generate optimized search queries from the user query.
        2. Search via Tavily.
        3. Summarize results into research notes.
        """
        logger.info("ResearcherAgent.run | query=%s", state.request.query[:80])

        # Step 1: Search for sources
        sources = self._search.search(
            query=state.request.query,
            max_results=state.request.max_sources,
        )
        state.sources.extend(sources)

        if not sources:
            state.errors.append("Researcher found no sources.")
            state.research_notes = "No sources found for the given query."
            state.add_trace_event("researcher", {"status": "no_sources"})
            return state

        # Step 2: Compile source material
        source_text = ""
        for i, src in enumerate(sources, 1):
            source_text += f"\n[{i}] {src.title}\n    URL: {src.url}\n    Content: {src.snippet}\n"

        # Step 3: Use LLM to synthesize research notes
        system_prompt = """You are a Research Agent. Your task is to synthesize search results into 
structured, comprehensive research notes. Include:
1. Key findings from each source
2. Important facts, data points, and claims
3. Source citations in [N] format
4. Areas where more research might be needed

Write clearly and concisely. Focus on factual information."""

        user_prompt = f"""Research query: {state.request.query}

Sources found:
{source_text}

Write detailed research notes based on these sources."""

        try:
            response = self._llm.complete(system_prompt, user_prompt)
            state.research_notes = response.content
        except Exception as e:
            logger.exception("Researcher LLM summarization failed.")
            state.errors.append(f"Researcher LLM error: {e}")
            # Fallback: use raw source snippets
            state.research_notes = f"Raw sources:\n{source_text}"

        state.agent_results.append(AgentResult(
            agent=AgentName.RESEARCHER,
            content=state.research_notes or "",
            metadata={
                "num_sources": len(sources),
                "source_titles": [s.title for s in sources],
            },
        ))

        state.add_trace_event("researcher", {
            "num_sources": len(sources),
            "notes_length": len(state.research_notes or ""),
        })

        logger.info("ResearcherAgent.run | sources=%d | notes_len=%d",
                     len(sources), len(state.research_notes or ""))
        return state
