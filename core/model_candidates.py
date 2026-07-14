"""Model candidates for each SportNews AI agent."""

from __future__ import annotations

import os


DEFAULT_DEEP_RESEARCH_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "openai/gpt-oss-120b:free",
    "google/gemma-4-31b-it:free",
    "openai/gpt-oss-20b:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "openrouter/free",
]

DEFAULT_LIGHTWEIGHT_MODELS = [
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "poolside/laguna-xs-2.1:free",
    "openai/gpt-oss-20b:free",
    "liquid/lfm-2.5-1.2b-instruct:free",
    "openrouter/free",
]


def _models_from_env(env_name: str, default: list[str]) -> list[str]:
    raw_value = os.getenv(env_name, "").strip()
    if not raw_value:
        return list(default)

    models = [model.strip() for model in raw_value.split(",") if model.strip()]
    return models or list(default)


DEEP_RESEARCH_MODELS = _models_from_env(
    "OPENROUTER_MODEL_CANDIDATES",
    DEFAULT_DEEP_RESEARCH_MODELS,
)
LIGHTWEIGHT_MODELS = _models_from_env(
    "OPENROUTER_LIGHT_MODEL_CANDIDATES",
    DEFAULT_LIGHTWEIGHT_MODELS,
)

PRIMARY_MODEL = DEEP_RESEARCH_MODELS[0]

MODEL_CANDIDATES = {
    "planner": DEEP_RESEARCH_MODELS,
    "researcher": DEEP_RESEARCH_MODELS,
    "ranker": DEEP_RESEARCH_MODELS,
    "writer": DEEP_RESEARCH_MODELS,
    "reviewer": DEEP_RESEARCH_MODELS,
    "clarifier": LIGHTWEIGHT_MODELS,
    "fast_chat": LIGHTWEIGHT_MODELS,
}
