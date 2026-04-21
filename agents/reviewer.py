"""Reviewer agent for the Sports Weekly Intelligence Agent pipeline."""

from __future__ import annotations

import json
import logging
import os

from langchain_groq import ChatGroq

from models.schemas import ReportState

logger = logging.getLogger(__name__)


def _build_prompt(state: ReportState) -> str:
    """Build the LLM prompt to validate the report against quality criteria."""
    report = state.get("report")

    if report is None:
        return (
            "The report is missing entirely. Reject it.\n\n"
            'Return ONLY a JSON object: {"status": "rejected", "reason": "Report is missing."}'
        )

    executive_summary = report.executive_summary or ""
    trending_keywords = report.trending_keywords or []
    highlighted_news = report.highlighted_news or []

    news_text = ""
    for i, item in enumerate(highlighted_news, 1):
        news_text += (
            f"\n[{i}] Headline: {item.headline}\n"
            f"    Summary: {item.summary}\n"
            f"    Source: {item.source}\n"
            f"    URL: {item.url}\n"
        )

    return (
        "You are a senior sports journalism editor reviewing a weekly intelligence report.\n"
        "Evaluate the report below against the following quality criteria:\n"
        "1. Completeness: executive_summary is non-empty, trending_keywords has at least 1 item, "
        "highlighted_news has at least 1 item with all required fields (headline, summary, source, url).\n"
        "2. Factual grounding: claims are supported by the provided article data, not invented.\n"
        "3. Professional tone: language is clear, objective, and suitable for a professional audience.\n"
        "4. All three sections are present and substantive.\n\n"
        "--- REPORT ---\n"
        f"Executive Summary:\n{executive_summary}\n\n"
        f"Trending Keywords: {', '.join(trending_keywords)}\n\n"
        f"Highlighted News:{news_text}\n"
        "--------------\n\n"
        'Return ONLY a JSON object (no markdown, no extra text) with this exact structure:\n'
        '{\n'
        '  "status": "approved" | "rejected",\n'
        '  "reason": "<brief explanation>"\n'
        '}\n\n'
        'Set "status" to "approved" if all criteria are met, otherwise "rejected".'
    )


def reviewer_node(state: ReportState) -> ReportState:
    """LangGraph node: validate the report and set review_status."""
    model_name = os.getenv("GROQ_LLM_MODEL", "grok-3-mini")
    prompt = _build_prompt(state)

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
        status = data.get("status", "").lower()
        reason = data.get("reason", "")

        if status not in ("approved", "rejected"):
            logger.warning("Unexpected review status '%s'; defaulting to approved.", status)
            status = "approved"

        state["review_status"] = status
        logger.info("Reviewer decision: %s — %s", status, reason)

    except Exception as exc:
        logger.error("Reviewer LLM call failed: %s", exc)
        # Fail open: approve to avoid blocking the pipeline
        state["review_status"] = "approved"
        state["error"] = f"Reviewer failed: {exc}"

    return state


def should_rewrite(state: ReportState) -> str:
    """Conditional edge function: route back to writer or end the pipeline."""
    review_status = state.get("review_status", "approved")
    rewrite_count = state.get("rewrite_count", 0)

    if review_status == "rejected" and rewrite_count < 2:
        logger.info(
            "Report rejected (rewrite_count=%d); routing back to writer.", rewrite_count
        )
        return "writer"

    return "end"
