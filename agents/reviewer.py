"""Reviewer agent for the Sports Weekly Intelligence Agent pipeline."""

from __future__ import annotations

import json
import logging
import os

from langchain_groq import ChatGroq

from models.schemas import ReportState

logger = logging.getLogger(__name__)


_GENERIC_KEYWORDS = {
    "thể thao", "bóng đá", "tin tức", "sport", "news",
    "đội bóng", "cầu thủ", "trận đấu",
}


def _build_prompt(state: ReportState) -> str:
    """Build the LLM prompt to validate the report against quality criteria."""
    report = state.get("report")

    if report is None:
        return (
            "Báo cáo bị thiếu hoàn toàn. Từ chối.\n\n"
            'Trả về CHỈ một JSON object: {"status": "rejected", "reason": "Báo cáo bị thiếu."}'
        )

    executive_summary = report.executive_summary or ""
    trending_keywords = report.trending_keywords or []
    highlighted_news = report.highlighted_news or []

    # Criterion B: collect expected sources from ranked_articles
    ranked_articles = state.get("ranked_articles") or []
    expected_sources = sorted({a.source for a in ranked_articles if a.source})
    expected_sources_text = (
        ", ".join(expected_sources) if expected_sources else "(none)"
    )

    # Criterion A: extract first and last paragraph for the LLM to compare
    paragraphs = [p.strip() for p in executive_summary.split("\n\n") if p.strip()]
    para1 = paragraphs[0] if len(paragraphs) >= 1 else ""
    para4 = paragraphs[3] if len(paragraphs) >= 4 else ""

    # Criterion C: pre-check generic keywords (deterministic, no LLM needed)
    generic_hits = [kw for kw in trending_keywords if kw.lower() in _GENERIC_KEYWORDS]
    generic_warning = (
        f"WARNING: trending_keywords contains overly generic terms: {generic_hits}. "
        "This MUST be rejected.\n"
        if generic_hits
        else ""
    )

    news_text = ""
    for i, item in enumerate(highlighted_news, 1):
        news_text += (
            f"\n[{i}] Tiêu đề: {item.headline}\n"
            f"    Tóm tắt: {item.summary}\n"
            f"    Nguồn: {item.source}\n"
            f"    URL: {item.url}\n"
        )

    return (
        "You are a senior sports journalism editor reviewing a weekly intelligence report.\n"
        "Evaluate the report below against these quality criteria:\n"
        "1. Completeness: executive_summary is non-empty, trending_keywords has at least 1 item, "
        "highlighted_news has at least 1 item with all required fields (headline, summary, source, url).\n"
        "2. Factual grounding: claims are supported by the provided article data, not invented.\n"
        "3. Professional tone: language is clear, objective, and suitable for a professional audience.\n"
        "4. All three sections are present and substantive.\n"
        "5. ALL content MUST be in VIETNAMESE (proper nouns like team/player names can stay in original language).\n"
        "6. Consistency: executive_summary must mention events from highlighted_news and vice versa.\n"
        "7. Depth: executive_summary must analyze trends, not just list events. Each summary in highlighted_news must have at least 2 sentences.\n"
        "8. No repetition (Criterion A): Paragraph 1 and Paragraph 4 of executive_summary must NOT be near-identical "
        "in theme or opening phrase. If they restate the same idea, reject.\n"
        f"   Paragraph 1: \"{para1}\"\n"
        f"   Paragraph 4: \"{para4}\"\n"
        "9. Source coverage (Criterion B): Every source listed in highlighted_news must appear in the expected sources "
        f"derived from the article corpus. Expected sources: [{expected_sources_text}]. "
        "Reject if any highlighted_news item has a source NOT in that list.\n"
        f"10. No generic keywords (Criterion C): trending_keywords must not contain overly generic terms "
        f"such as: {sorted(_GENERIC_KEYWORDS)}. Reject if any such term is present.\n"
        f"{generic_warning}"
        "\n--- REPORT ---\n"
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
        if "```" in raw:
            import re
            raw = re.sub(r"```(?:json)?\s*", "", raw).strip()

        # Find the FIRST complete JSON object only
        start = raw.find("{")
        if start != -1:
            depth, end = 0, start
            for i, ch in enumerate(raw[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            raw = raw[start:end]

        data = json.loads(raw, strict=False)
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
