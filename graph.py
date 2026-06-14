"""LangGraph pipeline for the Sports Weekly Intelligence Agent."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Callable, TypeVar

from langgraph.graph import END, START, StateGraph

from agents.planner import planner_node
from agents.ranker import ranker_node
from agents.retriever import retriever_node
from agents.reviewer import reviewer_node, should_rewrite
from agents.writer import writer_node
from models.schemas import ReportMetadata, ReportState
from tools.crawler import crawl_all_sources
from tools.db import get_articles_by_lookback_days, get_engine, save_articles
from tools.embeddings import build_faiss_index, embed_articles
from tools.preprocess import clean_text, deduplicate_articles, filter_recent_articles

logger = logging.getLogger(__name__)

T = TypeVar("T")


class PipelineStageError(RuntimeError):
    """Error raised with the pipeline stage that failed."""

    def __init__(self, stage: str, cause: Exception):
        self.stage = stage
        self.cause = cause
        super().__init__(f"Pipeline failed at stage '{stage}': {cause}")


def build_graph():
    """Build and compile the LangGraph state machine.

    Graph topology:
        START → planner → retriever → writer → reviewer
        reviewer --[approved or max rewrites]--> END
        reviewer --[rejected, rewrite_count < 2]--> writer
    """
    graph = StateGraph(ReportState)

    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("ranker", ranker_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "ranker")
    graph.add_edge("ranker", "writer")
    graph.add_edge("writer", "reviewer")

    graph.add_conditional_edges(
        "reviewer",
        should_rewrite,
        {
            "writer": "writer",
            "end": END,
        },
    )

    return graph.compile()


def _build_metadata(
    period_type: str,
    lookback_days: int,
    source_count: int = 0,
    ranked_count: int = 0,
    stage: str | None = None,
) -> ReportMetadata:
    today = datetime.now(tz=timezone(timedelta(hours=7))).date()
    if period_type == "weekly":
        period_start = today - timedelta(days=today.weekday())
        period_end = period_start + timedelta(days=6)
    else:
        period_end = today
        period_start = today - timedelta(days=lookback_days - 1)
    return ReportMetadata(
        period_type=period_type,
        period_start=period_start,
        period_end=period_end,
        lookback_days=lookback_days,
        source_count=source_count,
        ranked_count=ranked_count,
        stage=stage,
    )


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning("Invalid %s=%r. Using default %d.", name, raw_value, default)
        return default
    return value


def article_limit_for_period(period_type: str) -> int:
    """Return the max article count for the requested report period."""
    fallback = _env_int("MAX_REPORT_ARTICLES", 25 if period_type == "daily" else 60)
    if period_type == "daily":
        return _env_int("MAX_DAILY_REPORT_ARTICLES", fallback)
    if period_type == "weekly":
        return _env_int("MAX_WEEKLY_REPORT_ARTICLES", fallback)
    return fallback


def limit_articles_for_report(articles, period_type: str):
    """Limit articles before embedding/ranking to keep report generation bounded."""
    limit = article_limit_for_period(period_type)
    if limit <= 0 or len(articles) <= limit:
        return articles
    logger.info("Limiting %s report articles from %d to %d.", period_type, len(articles), limit)
    return articles[:limit]


def _run_stage(stage: str, action: Callable[[], T]) -> T:
    logger.info("Pipeline stage started: %s", stage)
    try:
        result = action()
    except Exception as exc:
        logger.error("Pipeline stage failed: %s: %s", stage, exc, exc_info=True)
        raise PipelineStageError(stage, exc) from exc
    logger.info("Pipeline stage completed: %s", stage)
    return result


def run_pipeline(period_type: str = "weekly", lookback_days: int = 7) -> ReportState:
    """Bootstrap tools and run the full LangGraph pipeline.

    Bootstrap steps:
        1. Crawl all sources
        2. Clean article content
        3. Deduplicate and filter to last 7 days
        4. Persist to DB
        5. Reload from DB (last 7 days)
        6. Embed articles and build FAISS index (if articles exist)
        7. Run the LangGraph state machine
    """
    logger.info("Starting pipeline bootstrap...")

    # Step 1: Crawl
    raw_articles = _run_stage("crawl", lambda: crawl_all_sources(lookback_days=lookback_days))
    logger.info("Crawled %d raw articles.", len(raw_articles))

    # Step 2: Clean content in-place
    def clean_articles():
        for article in raw_articles:
            article.content = clean_text(article.content)

    _run_stage("clean", clean_articles)

    # Step 3: Deduplicate and filter
    articles = _run_stage(
        "dedup_filter",
        lambda: deduplicate_articles(filter_recent_articles(raw_articles, lookback_days=lookback_days)),
    )
    logger.info("%d articles after dedup/filter.", len(articles))

    # Step 4: Persist
    engine = _run_stage("db_connect", get_engine)
    _run_stage("db_save_articles", lambda: save_articles(articles, engine))

    # Step 5: Reload from DB
    articles = _run_stage(
        "db_load_articles",
        lambda: get_articles_by_lookback_days(engine, lookback_days=lookback_days),
    )
    logger.info("Loaded %d articles from DB before report limit.", len(articles))
    articles = limit_articles_for_report(articles, period_type)
    logger.info("Using %d articles for %s report.", len(articles), period_type)

    # Steps 6a-b: Embed and index (only when there are articles)
    if articles:
        embeddings = _run_stage("embedding", lambda: embed_articles(articles))
        _run_stage("faiss_index", lambda: build_faiss_index(embeddings))
        logger.info("FAISS index built with %d vectors.", len(articles))

    # Step 7: Initialise state and run graph
    initial_state: ReportState = {
        "articles": articles,
        "plan": None,  # type: ignore[typeddict-item]
        "retrieved_articles": [],
        "ranked_articles": [],
        "report": None,
        "metadata": _build_metadata(period_type, lookback_days, source_count=len(articles), stage="graph"),
        "review_status": "pending",
        "rewrite_count": 0,
        "error": None,
    }

    compiled = build_graph()
    final_state: ReportState = _run_stage("langgraph", lambda: compiled.invoke(initial_state))
    metadata = final_state.get("metadata") or _build_metadata(period_type, lookback_days)
    metadata.source_count = len(articles)
    metadata.ranked_count = len(final_state.get("ranked_articles", []))
    metadata.stage = "completed"
    final_state["metadata"] = metadata
    return final_state
