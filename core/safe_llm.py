"""Safe LLM router with OpenRouter primary models and Groq fallback."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)


class SafeLLM:
    """Try OpenRouter candidates first, then fallback to the existing Groq model."""

    def __init__(self, agent_name: str, candidates: list[str], groq_fallback: Any):
        self.agent_name = agent_name
        self.candidates = candidates
        self.groq_fallback = groq_fallback

    def _build_openrouter_llm(self, model_name: str):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("langchain-openai is required for OpenRouter support") from exc

        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

        headers = {
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:8000"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "SportNews AI"),
        }
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            default_headers=headers,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            max_retries=1,
            timeout=float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "120")),
        )

    def invoke(self, prompt: Any):
        if not os.getenv("OPENROUTER_API_KEY", "").strip():
            return self.groq_fallback.invoke(prompt)

        for model_name in self.candidates:
            try:
                logger.info("LLM agent=%s provider=openrouter model=%s", self.agent_name, model_name)
                return self._build_openrouter_llm(model_name).invoke(prompt)
            except Exception as exc:
                logger.warning(
                    "OpenRouter model failed for agent=%s model=%s: %s",
                    self.agent_name,
                    model_name,
                    exc,
                )
                time.sleep(0.5)

        logger.warning("All OpenRouter candidates failed for agent=%s. Falling back to Groq.", self.agent_name)
        return self.groq_fallback.invoke(prompt)
