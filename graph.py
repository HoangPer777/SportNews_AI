"""LangGraph pipeline for the Sports Weekly Intelligence Agent."""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from agents.planner import planner_node
from agents.retriever import retriever_node
from agents.reviewer import reviewer_node, should_rewrite
from agents.writer import writer_node
from models.schemas import ReportState
from tools.crawler import crawl_all_sources
from tools.db import get_articles_last_7_days, get_engine, save_articles
from tools.embeddings import build_faiss_index, embed_articles
from tools.preprocess import clean_text, deduplicate_articles, filter_recent_articles

logger = logging.getLogger(__name__)


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
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "writer")
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


def run_pipeline() -> ReportState:
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
    raw_articles = crawl_all_sources()
    logger.info("Crawled %d raw articles.", len(raw_articles))

    # Step 2: Clean content in-place
    for article in raw_articles:
        article.content = clean_text(article.content)

    # Step 3: Deduplicate and filter
    articles = deduplicate_articles(filter_recent_articles(raw_articles))
    logger.info("%d articles after dedup/filter.", len(articles))

    # Step 4: Persist
    engine = get_engine()
    save_articles(articles, engine)

    # Step 5: Reload from DB
    articles = get_articles_last_7_days(engine)
    logger.info("Loaded %d articles from DB.", len(articles))

    # Steps 6a-b: Embed and index (only when there are articles)
    if articles:
        embeddings = embed_articles(articles)
        build_faiss_index(embeddings)
        logger.info("FAISS index built with %d vectors.", len(articles))

    # Step 7: Initialise state and run graph
    initial_state: ReportState = {
        "articles": articles,
        "plan": None,  # type: ignore[typeddict-item]
        "retrieved_articles": [],
        "report": None,
        "review_status": "pending",
        "rewrite_count": 0,
        "error": None,
    }

    compiled = build_graph()
    final_state: ReportState = compiled.invoke(initial_state)
    return final_state
