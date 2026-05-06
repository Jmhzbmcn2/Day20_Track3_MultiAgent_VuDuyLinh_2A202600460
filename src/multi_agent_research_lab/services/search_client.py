"""Search client abstraction for ResearcherAgent."""

import logging

from tavily import TavilyClient as _TavilyClient

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Provider-agnostic search client using Tavily."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = _TavilyClient(api_key=settings.tavily_api_key)

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query using Tavily API."""

        logger.info("SearchClient.search | query=%s | max_results=%d", query[:80], max_results)

        try:
            response = self._client.search(query=query, max_results=max_results)
        except Exception:
            logger.exception("Tavily search failed for query: %s", query[:80])
            return []

        results: list[SourceDocument] = []
        for item in response.get("results", []):
            results.append(
                SourceDocument(
                    title=item.get("title", "Untitled"),
                    url=item.get("url"),
                    snippet=item.get("content", ""),
                    metadata={"score": item.get("score", 0.0)},
                )
            )

        logger.info("SearchClient.search | found %d results", len(results))
        return results
