"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import logging
from dataclasses import dataclass

from groq import Groq
from langsmith import traceable
from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client using Groq."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = Groq(api_key=settings.groq_api_key)
        self._model = settings.groq_model

    @traceable(run_type="llm", name="Groq Completion")
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion via Groq API.

        Includes retry, timeout, and token logging.
        """
        logger.info("LLMClient.complete | model=%s | system_len=%d | user_len=%d",
                     self._model, len(system_prompt), len(user_prompt))

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_completion_tokens=2048,
        )

        choice = response.choices[0]
        usage = response.usage

        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None

        # Groq free tier: estimate cost (very low for Llama models)
        cost = None
        if input_tokens is not None and output_tokens is not None:
            # Approximate pricing for llama-3.3-70b on Groq
            cost = (input_tokens * 0.59 / 1_000_000) + (output_tokens * 0.79 / 1_000_000)

        logger.info("LLMClient.complete | tokens_in=%s | tokens_out=%s | cost=$%s",
                     input_tokens, output_tokens, f"{cost:.6f}" if cost else "N/A")

        return LLMResponse(
            content=choice.message.content or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
