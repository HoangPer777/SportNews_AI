"""Model candidates for each SportNews AI agent."""

from __future__ import annotations

import os


PRIMARY_MODEL = os.getenv("OPENROUTER_PRIMARY_MODEL", "openai/gpt-oss-120b:free")
FALLBACK_MODEL = os.getenv("OPENROUTER_FALLBACK_MODEL", "z-ai/glm-4.5-air:free")

MODEL_CANDIDATES = {
    "planner": [PRIMARY_MODEL, FALLBACK_MODEL],
    "ranker": [PRIMARY_MODEL, FALLBACK_MODEL],
    "writer": [PRIMARY_MODEL, FALLBACK_MODEL],
    "reviewer": [PRIMARY_MODEL, FALLBACK_MODEL],
}
