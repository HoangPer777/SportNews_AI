# Feature: sports-weekly-intelligence-agent, Property 13: Retriever result is deduplicated and bounded
# Feature: sports-weekly-intelligence-agent, Property 14: Retriever uses all required query topics
"""Property tests for agents/retriever.py."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agents.retriever import REQUIRED_QUERY_TOPICS, retriever_node
from models.schemas import ArticleSchema, PlanSchema, ReportState


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def _unique_url(i: int) -> str:
    return f"https://example.com/article-{i}"


@st.composite
def article_pool_strategy(draw):
    """Generate a list of articles with unique URLs."""
    n = draw(st.integers(min_value=0, max_value=30))
    articles = []
    for i in range(n):
        articles.append(
            ArticleSchema(
                title=draw(st.text(min_size=1, max_size=50)),
                content=draw(st.text(min_size=1, max_size=100)),
                source=draw(st.sampled_from(["VnExpress", "Thanh Nien", "Tuoi Tre"])),
                url=_unique_url(i),
                published_at=datetime.utcnow(),
                category="sports",
            )
        )
    return articles


@st.composite
def plan_strategy(draw):
    """Generate a PlanSchema with varying sub-goals."""
    extra = draw(st.lists(st.text(min_size=1, max_size=40), min_size=0, max_size=5))
    sub_goals = list(REQUIRED_QUERY_TOPICS) + extra
    return PlanSchema(
        date_range="2024-01-01 to 2024-01-07",
        sub_goals=sub_goals,
        corpus_summary="Test corpus.",
    )


def _build_state(articles: list[ArticleSchema], plan: PlanSchema) -> ReportState:
    return ReportState(
        articles=articles,
        plan=plan,
        retrieved_articles=[],
        report=None,
        review_status="pending",
        rewrite_count=0,
        error=None,
    )


def _make_mock_index(n_articles: int, dim: int = 4):
    """Return a mock FAISS index that returns sequential indices."""
    mock_index = MagicMock()
    mock_index.ntotal = n_articles

    def fake_search(vec, k):
        # Return first min(k, n_articles) indices
        count = min(k, n_articles)
        indices = np.array([[i for i in range(count)]], dtype=np.int64)
        distances = np.zeros((1, count), dtype=np.float32)
        return distances, indices

    mock_index.search.side_effect = fake_search
    return mock_index


def _make_embed_query_mock(dim: int = 4):
    """Return a mock embed_query that returns a fixed-dim vector."""
    def fake_embed(query: str) -> np.ndarray:
        return np.zeros((1, dim), dtype=np.float32)
    return fake_embed


# ---------------------------------------------------------------------------
# Property 13: Retriever result is deduplicated and bounded
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    articles=article_pool_strategy(),
    plan=plan_strategy(),
)
def test_retriever_result_bounded_and_deduplicated(
    articles: list[ArticleSchema],
    plan: PlanSchema,
) -> None:
    """Property 13: Retriever result is deduplicated and bounded.

    Validates: Requirements 7.3
    """
    state = _build_state(articles, plan)
    mock_index = _make_mock_index(len(articles))

    with (
        patch("agents.retriever.load_faiss_index", return_value=mock_index),
        patch("agents.retriever.embed_query", side_effect=_make_embed_query_mock()),
    ):
        result = retriever_node(state)

    retrieved = result["retrieved_articles"]

    # Bounded: at most 10 articles
    assert len(retrieved) <= 10, (
        f"Expected at most 10 retrieved articles, got {len(retrieved)}"
    )

    # Deduplicated: all URLs unique
    urls = [a.url for a in retrieved]
    assert len(urls) == len(set(urls)), (
        f"Duplicate URLs found in retrieved_articles: {urls}"
    )


# ---------------------------------------------------------------------------
# Property 14: Retriever uses all required query topics
# ---------------------------------------------------------------------------

@st.composite
def plan_with_varying_sub_goals_strategy(draw):
    """Generate a PlanSchema with arbitrary sub-goals (may or may not include required topics)."""
    extra = draw(st.lists(st.text(min_size=1, max_size=40), min_size=0, max_size=8))
    return PlanSchema(
        date_range="2024-01-01 to 2024-01-07",
        sub_goals=extra,
        corpus_summary="Test corpus.",
    )


@settings(max_examples=100)
@given(
    articles=article_pool_strategy(),
    plan=plan_with_varying_sub_goals_strategy(),
)
def test_retriever_always_uses_required_query_topics(
    articles: list[ArticleSchema],
    plan: PlanSchema,
) -> None:
    """Property 14: Retriever uses all required query topics.

    Validates: Requirements 7.4
    """
    state = _build_state(articles, plan)
    mock_index = _make_mock_index(len(articles))

    captured_queries: list[str] = []

    def capturing_embed_query(query: str) -> np.ndarray:
        captured_queries.append(query)
        return np.zeros((1, 4), dtype=np.float32)

    with (
        patch("agents.retriever.load_faiss_index", return_value=mock_index),
        patch("agents.retriever.embed_query", side_effect=capturing_embed_query),
    ):
        retriever_node(state)

    lower_captured = [q.lower() for q in captured_queries]
    for required_topic in REQUIRED_QUERY_TOPICS:
        assert required_topic.lower() in lower_captured, (
            f"Required query topic '{required_topic}' was not passed to embed_query. "
            f"Captured queries: {captured_queries}"
        )
