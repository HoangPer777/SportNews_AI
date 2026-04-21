"""Ranker agent: uses LLM to select the most newsworthy articles before writing."""

from __future__ import annotations

import json
import logging
import os
import re

from langchain_groq import ChatGroq

from models.schemas import ArticleSchema, ReportState

logger = logging.getLogger(__name__)

TOP_N = 8


def ranker_node(state: ReportState) -> ReportState:
    """LangGraph node: rank retrieved articles by newsworthiness and select top N."""
    articles: list[ArticleSchema] = state.get("retrieved_articles", [])

    if not articles:
        state["ranked_articles"] = []
        return state

    # If already small enough, skip LLM call
    if len(articles) <= TOP_N:
        state["ranked_articles"] = articles
        return state

    # Build a compact list: index, source, title, first sentence of content
    candidates = ""
    for i, a in enumerate(articles):
        first_sentence = a.content.split(".")[0].strip()[:150] if a.content else ""
        candidates += f"[{i}] ({a.source}) {a.title}. {first_sentence}\n"

    prompt = (
        "You are a sports news editor. Below is a numbered list of sports articles from this week.\n"
        "Select the indices of the most newsworthy and important articles.\n"
        "Criteria: national/international significance, broad audience interest, unique story, recency.\n"
        f"Pick exactly {TOP_N} indices. You MUST include at least 2 articles from EACH source present in the list.\n\n"
        f"Articles:\n{candidates}\n"
        f"Return ONLY a JSON array of {TOP_N} integer indices, e.g.: [0, 3, 5, 7, 2, 8, 1, 4, 6, 9]\n"
        "No explanation. No markdown. Just the JSON array."
    )

    model_name = os.getenv("GROQ_LLM_MODEL", "llama-3.1-8b-instant")
    try:
        llm = ChatGroq(model=model_name, api_key=os.getenv("GROQ_API_KEY"))
        response = llm.invoke(prompt)
        raw = response.content.strip()

        # Extract JSON array
        match = re.search(r"\[[\d,\s]+\]", raw)
        if not match:
            raise ValueError(f"No JSON array found in response: {raw[:200]}")

        indices = json.loads(match.group())
        ranked = []
        seen = set()
        for idx in indices:
            if isinstance(idx, int) and 0 <= idx < len(articles) and idx not in seen:
                ranked.append(articles[idx])
                seen.add(idx)

        # Fallback: fill up to TOP_N if LLM returned fewer valid indices
        if len(ranked) < TOP_N:
            for a in articles:
                if a.url not in {r.url for r in ranked}:
                    ranked.append(a)
                if len(ranked) >= TOP_N:
                    break

        logger.info(
            "Ranker selected %d articles from sources: %s",
            len(ranked),
            {a.source for a in ranked},
        )
        state["ranked_articles"] = ranked

    except Exception as exc:
        logger.warning("Ranker LLM failed (%s), falling back to retrieved_articles.", exc)
        state["ranked_articles"] = articles[:TOP_N]

    return state
