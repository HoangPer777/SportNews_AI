"""Retriever agent for the Sports Weekly Intelligence Agent pipeline."""

from __future__ import annotations

import logging
import os

import numpy as np

from models.schemas import ArticleSchema, ReportState
from tools.embeddings import embed_query, load_faiss_index

logger = logging.getLogger(__name__)

REQUIRED_QUERY_TOPICS = [
    "sports highlights of the week",
    "football top news",
    "Vietnam sports achievements",
    "international sports trends",
]

MAX_RETRIEVED_ARTICLES = 10


def retriever_node(state: ReportState) -> ReportState:
    """LangGraph node: retrieve the most relevant articles using FAISS semantic search."""
    articles: list[ArticleSchema] = state.get("articles", [])
    plan = state.get("plan")

    # Derive queries from plan sub-goals, then always append the 4 required topics
    queries: list[str] = []
    if plan and plan.sub_goals:
        queries.extend(plan.sub_goals)

    # Ensure all required query topics are included
    lower_queries = [q.lower() for q in queries]
    for topic in REQUIRED_QUERY_TOPICS:
        if topic.lower() not in lower_queries:
            queries.append(topic)

    top_k = int(os.environ.get("TOP_K_RETRIEVAL", "5"))

    try:
        index = load_faiss_index()
    except FileNotFoundError as exc:
        logger.error("FAISS index not found: %s", exc)
        state["error"] = f"Retriever failed: {exc}"
        state["retrieved_articles"] = []
        return state

    seen_urls: set[str] = set()
    retrieved: list[ArticleSchema] = []

    for query in queries:
        if len(retrieved) >= MAX_RETRIEVED_ARTICLES:
            break

        try:
            query_vec = embed_query(query)
            # query_vec shape: (1, D)
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
                    if len(retrieved) >= MAX_RETRIEVED_ARTICLES:
                        break

        except Exception as exc:
            logger.warning("Query '%s' failed: %s", query, exc)
            continue

    state["retrieved_articles"] = retrieved
    return state
