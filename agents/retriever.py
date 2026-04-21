"""Retriever agent for the Sports Weekly Intelligence Agent pipeline."""

from __future__ import annotations

import logging
import os

from models.schemas import ArticleSchema, ReportState
from tools.embeddings import embed_query, load_faiss_index

logger = logging.getLogger(__name__)

REQUIRED_QUERY_TOPICS = [
    "sports highlights of the week",
    "football top news",
    "Vietnam sports achievements",
    "international sports trends",
]

RETRIEVER_MAX = 30  # FAISS pre-filters to this many before Ranker selects top 10


def retriever_node(state: ReportState) -> ReportState:
    """LangGraph node: use FAISS to pre-filter ~30 relevant articles for the Ranker."""
    articles: list[ArticleSchema] = state.get("articles", [])
    plan = state.get("plan")

    if not articles:
        state["retrieved_articles"] = []
        return state

    # Build query list from plan sub-goals + required topics
    queries: list[str] = []
    if plan and plan.sub_goals:
        queries.extend(plan.sub_goals)
    lower_queries = [q.lower() for q in queries]
    for topic in REQUIRED_QUERY_TOPICS:
        if topic.lower() not in lower_queries:
            queries.append(topic)

    top_k = int(os.environ.get("TOP_K_RETRIEVAL", "10"))

    try:
        index = load_faiss_index()
    except FileNotFoundError as exc:
        logger.warning("FAISS index not found (%s), passing all articles to Ranker.", exc)
        state["retrieved_articles"] = articles
        return state

    seen_urls: set[str] = set()
    retrieved: list[ArticleSchema] = []

    # FAISS search: each query contributes top_k results, deduplicated
    for query in queries:
        if len(retrieved) >= RETRIEVER_MAX:
            break
        try:
            query_vec = embed_query(query)
            k = min(top_k, index.ntotal)
            if k == 0:
                continue
            _distances, indices = index.search(query_vec, k)
            for idx in indices[0]:
                if idx < 0 or idx >= len(articles):
                    continue
                article = articles[idx]
                if article.url not in seen_urls:
                    seen_urls.add(article.url)
                    retrieved.append(article)
                    if len(retrieved) >= RETRIEVER_MAX:
                        break
        except Exception as exc:
            logger.warning("Query '%s' failed: %s", query, exc)

    # Ensure source diversity: guarantee at least 1 article per source
    sources_present = {a.source for a in retrieved}
    for source in {a.source for a in articles} - sources_present:
        for article in articles:
            if article.source == source and article.url not in seen_urls:
                seen_urls.add(article.url)
                retrieved.append(article)
                break

    logger.info(
        "Retriever → Ranker: %d articles (sources: %s)",
        len(retrieved),
        {a.source for a in retrieved},
    )
    state["retrieved_articles"] = retrieved
    return state
