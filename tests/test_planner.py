# Feature: sports-weekly-intelligence-agent, Property 12: Planner state contains all required sub-goals
"""Property tests for agents/planner.py."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agents.planner import REQUIRED_SUB_GOALS, planner_node
from models.schemas import ArticleSchema, ReportState


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

article_strategy = st.builds(
    ArticleSchema,
    title=st.text(min_size=1, max_size=50),
    content=st.text(min_size=1, max_size=200),
    source=st.sampled_from(["VnExpress", "Thanh Nien", "Tuoi Tre"]),
    url=st.from_regex(r"https://[a-z]{3,10}\.[a-z]{2,4}/[a-z0-9]{1,20}", fullmatch=True),
    published_at=st.just(datetime.utcnow()),
    category=st.just("sports"),
)

extra_sub_goals_strategy = st.lists(
    st.text(min_size=1, max_size=40),
    min_size=0,
    max_size=5,
)


def _make_llm_response(extra_sub_goals: list[str]) -> str:
    """Build a JSON string the mock LLM will return, with optional extra sub-goals."""
    sub_goals = list(REQUIRED_SUB_GOALS) + extra_sub_goals
    return json.dumps(
        {
            "date_range": "2024-01-01 to 2024-01-07",
            "sub_goals": sub_goals,
            "corpus_summary": "Test corpus summary.",
        }
    )


def _make_llm_response_missing_some(extra_sub_goals: list[str]) -> str:
    """Build a JSON string that intentionally omits some required sub-goals."""
    # Only include the extra goals — planner must add the required ones back
    return json.dumps(
        {
            "date_range": "2024-01-01 to 2024-01-07",
            "sub_goals": extra_sub_goals,
            "corpus_summary": "Partial corpus summary.",
        }
    )


def _build_state(articles: list[ArticleSchema]) -> ReportState:
    return ReportState(
        articles=articles,
        plan=None,  # type: ignore[arg-type]
        retrieved_articles=[],
        report=None,
        review_status="pending",
        rewrite_count=0,
        error=None,
    )


# ---------------------------------------------------------------------------
# Property 12: Planner state contains all required sub-goals
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    articles=st.lists(article_strategy, min_size=0, max_size=10),
    extra_sub_goals=extra_sub_goals_strategy,
)
def test_planner_always_contains_required_sub_goals(
    articles: list[ArticleSchema],
    extra_sub_goals: list[str],
) -> None:
    """Property 12: Planner state contains all required sub-goals.

    Validates: Requirements 6.1, 6.3
    """
    state = _build_state(articles)
    mock_content = _make_llm_response(extra_sub_goals)

    mock_message = MagicMock()
    mock_message.content = mock_content

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_message

    with patch("agents.planner.ChatGoogleGenerativeAI", return_value=mock_llm):
        result = planner_node(state)

    assert result.get("error") is None, f"Unexpected error: {result.get('error')}"
    plan = result["plan"]
    assert plan is not None

    lower_sub_goals = [g.lower() for g in plan.sub_goals]
    for required in REQUIRED_SUB_GOALS:
        assert required.lower() in lower_sub_goals, (
            f"Required sub-goal '{required}' missing from plan.sub_goals: {plan.sub_goals}"
        )


@settings(max_examples=100)
@given(
    articles=st.lists(article_strategy, min_size=0, max_size=10),
    extra_sub_goals=extra_sub_goals_strategy,
)
def test_planner_adds_missing_required_sub_goals(
    articles: list[ArticleSchema],
    extra_sub_goals: list[str],
) -> None:
    """Property 12 (variant): Planner merges required sub-goals even when LLM omits them.

    Validates: Requirements 6.1, 6.3
    """
    state = _build_state(articles)
    # LLM response intentionally omits all required sub-goals
    mock_content = _make_llm_response_missing_some(extra_sub_goals)

    mock_message = MagicMock()
    mock_message.content = mock_content

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_message

    with patch("agents.planner.ChatGoogleGenerativeAI", return_value=mock_llm):
        result = planner_node(state)

    assert result.get("error") is None, f"Unexpected error: {result.get('error')}"
    plan = result["plan"]
    assert plan is not None

    lower_sub_goals = [g.lower() for g in plan.sub_goals]
    for required in REQUIRED_SUB_GOALS:
        assert required.lower() in lower_sub_goals, (
            f"Required sub-goal '{required}' missing after merge. sub_goals: {plan.sub_goals}"
        )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_planner_sets_error_on_llm_failure() -> None:
    """On LLM failure, planner sets state['error'] and does not raise."""
    state = _build_state([])

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("API unavailable")

    with patch("agents.planner.ChatGoogleGenerativeAI", return_value=mock_llm):
        result = planner_node(state)

    assert result["error"] is not None
    assert "Planner failed" in result["error"]


def test_planner_strips_markdown_fences() -> None:
    """Planner correctly strips ```json ... ``` fences from LLM response."""
    state = _build_state([])
    payload = json.dumps(
        {
            "date_range": "2024-01-01 to 2024-01-07",
            "sub_goals": list(REQUIRED_SUB_GOALS),
            "corpus_summary": "Summary.",
        }
    )
    fenced = f"```json\n{payload}\n```"

    mock_message = MagicMock()
    mock_message.content = fenced

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_message

    with patch("agents.planner.ChatGoogleGenerativeAI", return_value=mock_llm):
        result = planner_node(state)

    assert result.get("error") is None
    assert result["plan"] is not None
    lower_sub_goals = [g.lower() for g in result["plan"].sub_goals]
    for required in REQUIRED_SUB_GOALS:
        assert required.lower() in lower_sub_goals
