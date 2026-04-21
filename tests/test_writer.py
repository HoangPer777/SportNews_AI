# Feature: sports-weekly-intelligence-agent, Property 15: Writer report contains exactly three sections
# Feature: sports-weekly-intelligence-agent, Property 17: Writer saves Markdown report to disk
"""Property tests for agents/writer.py."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agents.writer import writer_node
from models.schemas import ArticleSchema, PlanSchema, ReportState


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@st.composite
def article_strategy(draw):
    """Generate a single ArticleSchema with unique-ish URL."""
    idx = draw(st.integers(min_value=0, max_value=10_000))
    return ArticleSchema(
        title=draw(st.text(min_size=1, max_size=80)),
        content=draw(st.text(min_size=1, max_size=300)),
        source=draw(st.sampled_from(["VnExpress", "Thanh Nien", "Tuoi Tre"])),
        url=f"https://example.com/article-{idx}",
        published_at=datetime.utcnow(),
        category="sports",
    )


@st.composite
def article_list_strategy(draw):
    """Generate a list of 0–10 articles."""
    n = draw(st.integers(min_value=0, max_value=10))
    return [draw(article_strategy()) for _ in range(n)]


def _make_llm_response(articles) -> str:
    """Build a valid JSON LLM response referencing the provided articles."""
    highlighted = []
    for a in articles[:3]:
        highlighted.append({
            "headline": f"Headline: {a.title[:40]}",
            "summary": f"Summary of {a.title[:40]}",
            "source": a.source,
            "url": a.url,
        })
    # Ensure at least one highlighted item even with empty articles
    if not highlighted:
        highlighted.append({
            "headline": "Top Sports Story",
            "summary": "A major sports event occurred this week.",
            "source": "VnExpress",
            "url": "https://example.com/top-story",
        })

    return json.dumps({
        "executive_summary": "This week saw exciting developments across multiple sports disciplines.",
        "trending_keywords": ["football", "Vietnam", "championship"],
        "highlighted_news": highlighted,
    })


def _build_state(articles: list[ArticleSchema]) -> ReportState:
    return ReportState(
        articles=articles,
        plan=PlanSchema(
            date_range="2024-01-01 to 2024-01-07",
            sub_goals=["retrieve relevant stories", "identify trending topics"],
            corpus_summary="Test corpus.",
        ),
        retrieved_articles=articles,
        report=None,
        review_status="pending",
        rewrite_count=0,
        error=None,
    )


# ---------------------------------------------------------------------------
# Property 15: Writer report contains exactly three sections
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(articles=article_list_strategy())
def test_writer_report_contains_three_sections(articles: list[ArticleSchema]) -> None:
    """Property 15: Writer report contains exactly three sections.

    Validates: Requirements 8.1
    """
    state = _build_state(articles)
    mock_content = _make_llm_response(articles)

    mock_message = MagicMock()
    mock_message.content = mock_content

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_message

    with (
        patch("agents.writer.ChatGoogleGenerativeAI", return_value=mock_llm),
        tempfile.TemporaryDirectory() as tmpdir,
        patch.dict(os.environ, {"REPORT_OUTPUT_PATH": os.path.join(tmpdir, "report.md")}),
    ):
        result = writer_node(state)

    assert result.get("error") is None, f"Unexpected error: {result.get('error')}"
    report = result["report"]
    assert report is not None, "state['report'] should not be None"

    # All three sections must be non-empty
    assert report.executive_summary, "executive_summary must be non-empty"
    assert report.trending_keywords, "trending_keywords must be non-empty"
    assert report.highlighted_news, "highlighted_news must be non-empty"


# ---------------------------------------------------------------------------
# Property 17: Writer saves Markdown report to disk
# ---------------------------------------------------------------------------

@settings(max_examples=50)
@given(articles=article_list_strategy())
def test_writer_saves_markdown_to_disk(articles: list[ArticleSchema]) -> None:
    """Property 17: Writer saves Markdown report to disk.

    Validates: Requirements 8.5
    """
    state = _build_state(articles)
    mock_content = _make_llm_response(articles)

    mock_message = MagicMock()
    mock_message.content = mock_content

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_message

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "weekly_report.md")
        with (
            patch("agents.writer.ChatGoogleGenerativeAI", return_value=mock_llm),
            patch.dict(os.environ, {"REPORT_OUTPUT_PATH": output_path}),
        ):
            result = writer_node(state)

        assert result.get("error") is None, f"Unexpected error: {result.get('error')}"

        # File must exist and be non-empty
        assert os.path.exists(output_path), (
            f"Expected Markdown report at {output_path} but file does not exist"
        )
        assert os.path.getsize(output_path) > 0, (
            f"Markdown report at {output_path} is empty"
        )
