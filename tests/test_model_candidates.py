from __future__ import annotations

import importlib

import core.model_candidates as model_candidates


def test_deep_research_agents_use_stable_free_candidates() -> None:
    expected_first_model = "nvidia/nemotron-3-super-120b-a12b:free"
    removed_model = "z-ai/glm-4.5-air:free"

    for agent_name in ("planner", "researcher", "ranker", "writer", "reviewer"):
        candidates = model_candidates.MODEL_CANDIDATES[agent_name]

        assert candidates[0] == expected_first_model
        assert len(candidates) > 2
        assert removed_model not in candidates


def test_lightweight_agents_use_lightweight_candidates() -> None:
    expected_first_model = "nvidia/nemotron-3-nano-30b-a3b:free"

    for agent_name in ("clarifier", "fast_chat"):
        candidates = model_candidates.MODEL_CANDIDATES[agent_name]

        assert candidates[0] == expected_first_model
        assert "openrouter/free" in candidates


def test_model_candidates_can_be_overridden_by_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_MODEL_CANDIDATES", "model-a, model-b,,model-c")

    reloaded = importlib.reload(model_candidates)

    assert reloaded.MODEL_CANDIDATES["writer"] == ["model-a", "model-b", "model-c"]

    monkeypatch.delenv("OPENROUTER_MODEL_CANDIDATES")
    importlib.reload(model_candidates)
