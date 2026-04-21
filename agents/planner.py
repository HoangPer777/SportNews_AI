"""Planner agent for the Sports Weekly Intelligence Agent pipeline."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta

from langchain_groq import ChatGroq

from models.schemas import PlanSchema, ReportState

logger = logging.getLogger(__name__)

REQUIRED_SUB_GOALS = [
    "retrieve relevant stories",
    "identify trending topics",
    "summarize findings",
    "review report quality",
]


def _get_week_date_range() -> str:
    """Return the current week date range as a string."""
    today = datetime.utcnow().date()
    week_start = today - timedelta(days=today.weekday())
    week_end = week_start + timedelta(days=6)
    return f"{week_start.isoformat()} to {week_end.isoformat()}"


def _ensure_required_sub_goals(sub_goals: list[str]) -> list[str]:
    """Merge required sub-goals into the list, appending any that are missing."""
    lower_existing = [g.lower() for g in sub_goals]
    result = list(sub_goals)
    for required in REQUIRED_SUB_GOALS:
        if required.lower() not in lower_existing:
            result.append(required)
    return result


def planner_node(state: ReportState) -> ReportState:
    """LangGraph node: generate a structured plan for the weekly report."""
    articles = state.get("articles", [])
    sources = list({a.source for a in articles})
    article_count = len(articles)
    date_range = _get_week_date_range()

    prompt = (
        f"You are a sports news editor planning a weekly intelligence report.\n"
        f"Current week: {date_range}\n"
        f"Article corpus: {article_count} articles from sources: {', '.join(sources) if sources else 'none'}.\n\n"
        "Return ONLY a JSON object (no markdown, no extra text) with this exact structure:\n"
        "{\n"
        '  "date_range": "<week date range>",\n'
        '  "sub_goals": ["retrieve relevant stories", "identify trending topics", "summarize findings", "review report quality", ...],\n'
        '  "corpus_summary": "<brief summary of the corpus>"\n'
        "}\n\n"
        "The sub_goals list MUST include at minimum:\n"
        '- "retrieve relevant stories"\n'
        '- "identify trending topics"\n'
        '- "summarize findings"\n'
        '- "review report quality"\n'
        "You may add additional sub-goals as appropriate."
    )

    model_name = os.getenv("GROQ_LLM_MODEL", "grok-3-mini")

    try:
        llm = ChatGroq(model=model_name, api_key=os.getenv("GROQ_API_KEY"))
        response = llm.invoke(prompt)
        raw = response.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(
                line for line in lines if not line.startswith("```")
            ).strip()

        data = json.loads(raw)

        sub_goals = data.get("sub_goals", [])
        sub_goals = _ensure_required_sub_goals(sub_goals)

        plan = PlanSchema(
            date_range=data.get("date_range", date_range),
            sub_goals=sub_goals,
            corpus_summary=data.get("corpus_summary", ""),
        )
        state["plan"] = plan

    except Exception as exc:
        logger.error("Planner LLM call failed: %s", exc)
        state["error"] = f"Planner failed: {exc}"

    return state
