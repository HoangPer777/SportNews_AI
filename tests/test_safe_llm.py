from __future__ import annotations

from unittest.mock import MagicMock, patch

from core.llm import get_safe_llm
from core.safe_llm import SafeLLM


def test_get_safe_llm_returns_groq_when_openrouter_key_missing() -> None:
    mock_groq = MagicMock()

    with (
        patch.dict("os.environ", {"OPENROUTER_API_KEY": ""}, clear=False),
        patch("core.llm.get_agent_llm", return_value=mock_groq),
    ):
        llm = get_safe_llm("writer")

    assert llm is mock_groq


def test_safe_llm_tries_next_openrouter_candidate() -> None:
    groq = MagicMock()
    first = MagicMock()
    second = MagicMock()
    expected = MagicMock()
    first.invoke.side_effect = RuntimeError("rate limited")
    second.invoke.return_value = expected

    router = SafeLLM("writer", ["model-a", "model-b"], groq)

    with (
        patch.dict("os.environ", {"OPENROUTER_API_KEY": "key"}, clear=False),
        patch.object(router, "_build_openrouter_llm", side_effect=[first, second]),
        patch("core.safe_llm.time.sleep"),
    ):
        result = router.invoke("prompt")

    assert result is expected
    first.invoke.assert_called_once_with("prompt")
    second.invoke.assert_called_once_with("prompt")
    groq.invoke.assert_not_called()


def test_safe_llm_falls_back_to_groq_when_all_openrouter_models_fail() -> None:
    groq = MagicMock()
    expected = MagicMock()
    groq.invoke.return_value = expected

    router = SafeLLM("reviewer", ["model-a", "model-b"], groq)

    with (
        patch.dict("os.environ", {"OPENROUTER_API_KEY": "key"}, clear=False),
        patch.object(router, "_build_openrouter_llm", side_effect=RuntimeError("provider down")),
        patch("core.safe_llm.time.sleep"),
    ):
        result = router.invoke("prompt")

    assert result is expected
    groq.invoke.assert_called_once_with("prompt")
