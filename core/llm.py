"""LLM factory functions for SportNews AI agents."""

from __future__ import annotations

import os

from langchain_groq import ChatGroq

from core.model_candidates import MODEL_CANDIDATES, PRIMARY_MODEL
from core.safe_llm import SafeLLM


def get_agent_llm():
    """Return the existing Groq LLM fallback."""
    return ChatGroq(
        model=os.getenv("GROQ_LLM_MODEL", "llama-3.1-8b-instant"),
        api_key=os.getenv("GROQ_API_KEY"),
    )


def get_safe_llm(agent_name: str):
    """Return OpenRouter-backed SafeLLM when configured, otherwise Groq."""
    groq_fallback = get_agent_llm()
    if not os.getenv("OPENROUTER_API_KEY", "").strip():
        return groq_fallback

    candidates = MODEL_CANDIDATES.get(agent_name, [PRIMARY_MODEL])
    return SafeLLM(agent_name=agent_name, candidates=candidates, groq_fallback=groq_fallback)
