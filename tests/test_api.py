# Feature: sports-weekly-intelligence-agent, Property 1: API success response is valid Report JSON
"""Property and unit tests for main.py FastAPI application."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from main import app
from models.schemas import (
    ArticleSchema,
    HighlightedNewsItem,
    PlanSchema,
    ReportResponse,
    ReportSchema,
    ReportState,
)


# ---------------------------------------------------------------------------
# Helpers / strategies
# ---------------------------------------------------------------------------

def _make_report_state() -> ReportState:
    """Build a minimal valid ReportState with a completed report."""
    report = ReportSchema(
        executive_summary="A great week in sports.",
        trending_keywords=["football", "Vietnam", "championship"],
        highlighted_news=[
            HighlightedNewsItem(
                headline="Big Match Result",
                summary="Team A beat Team B 3-0.",
                source="VnExpress",
                url="https://vnexpress.net/article-1",
            )
        ],
        generated_at=datetime.utcnow(),
    )
    return ReportState(
        articles=[],
        plan=PlanSchema(
            date_range="2024-01-01 to 2024-01-07",
            sub_goals=["retrieve relevant stories"],
            corpus_summary="Test corpus.",
        ),
        retrieved_articles=[],
        report=report,
        review_status="approved",
        rewrite_count=0,
        error=None,
    )


@st.composite
def report_state_strategy(draw):
    """Generate a ReportState with varied report content."""
    n_keywords = draw(st.integers(min_value=1, max_value=10))
    keywords = [draw(st.text(min_size=1, max_size=30)) for _ in range(n_keywords)]

    n_news = draw(st.integers(min_value=1, max_value=5))
    news_items = [
        HighlightedNewsItem(
            headline=draw(st.text(min_size=1, max_size=80)),
            summary=draw(st.text(min_size=1, max_size=200)),
            source=draw(st.sampled_from(["VnExpress", "Thanh Nien", "Tuoi Tre"])),
            url=f"https://example.com/article-{draw(st.integers(min_value=0, max_value=100_000))}",
        )
        for _ in range(n_news)
    ]

    report = ReportSchema(
        executive_summary=draw(st.text(min_size=1, max_size=500)),
        trending_keywords=keywords,
        highlighted_news=news_items,
        generated_at=datetime.utcnow(),
    )

    return ReportState(
        articles=[],
        plan=PlanSchema(
            date_range="2024-01-01 to 2024-01-07",
            sub_goals=["retrieve relevant stories"],
            corpus_summary="Test corpus.",
        ),
        retrieved_articles=[],
        report=report,
        review_status="approved",
        rewrite_count=0,
        error=None,
    )


# ---------------------------------------------------------------------------
# Property 1: API success response is valid Report JSON
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(state=report_state_strategy())
def test_api_success_response_is_valid_report_json(state: ReportState) -> None:
    """Property 1: API success response is valid Report JSON.

    Validates: Requirements 1.3
    """
    with patch("main.run_pipeline", return_value=state):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/generate-report")

    assert response.status_code == 202

    # Must deserialize into ReportResponse without error
    body = ReportResponse.model_validate(response.json())

    assert body.status == "success"
    assert body.report is not None, "report must not be null on success"

    # All three sections must be present and non-empty
    assert body.report.executive_summary, "executive_summary must be non-empty"
    assert body.report.trending_keywords, "trending_keywords must be non-empty"
    assert body.report.highlighted_news, "highlighted_news must be non-empty"


# ---------------------------------------------------------------------------
# Error case: pipeline raises → HTTP 500 with status="error"
# ---------------------------------------------------------------------------

def test_api_error_response_on_pipeline_failure() -> None:
    """When run_pipeline() raises, the API returns HTTP 500 with status='error'."""
    with patch("main.run_pipeline", side_effect=RuntimeError("pipeline exploded")):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/generate-report")

    assert response.status_code == 500

    body = ReportResponse.model_validate(response.json())
    assert body.status == "error"
    assert body.error is not None
    assert "pipeline exploded" in body.error
