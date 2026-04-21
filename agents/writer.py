"""Writer agent for the Sports Weekly Intelligence Agent pipeline."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone, timedelta

from langchain_groq import ChatGroq

from models.schemas import HighlightedNewsItem, ReportSchema, ReportState

logger = logging.getLogger(__name__)

DEFAULT_REPORT_OUTPUT_PATH = "outputs/weekly_report.md"


def _build_summary_prompt(articles) -> str:
    """Prompt for call 1: executive summary + trending keywords only."""
    articles_text = ""
    for i, article in enumerate(articles, 1):
        articles_text += (
            f"\n[{i}] Title: {article.title}\n"
            f"    Source: {article.source}\n"
            f"    Content: {article.content[:300]}\n"
        )

    return (
        "You are a professional sports journalist. Write the OVERVIEW section of a weekly sports report in VIETNAMESE.\n\n"
        f"Articles:\n{articles_text if articles_text else '(No articles available)'}\n\n"
        "Write ONLY these two fields:\n"
        "1. executive_summary: Exactly 4 paragraphs in Vietnamese analyzing the week in sports.\n"
        "   - Paragraph 1: Overall sports landscape (2-3 sentences)\n"
        "   - Paragraph 2: Most prominent trend with analysis (2-3 sentences)\n"
        "   - Paragraph 3: Second major development (2-3 sentences)\n"
        "   - Paragraph 4: Closing outlook (2 sentences)\n"
        "2. trending_keywords: 8-12 significant keywords as a JSON array.\n\n"
        "Return ONLY a valid JSON object:\n"
        '{"executive_summary": "<4 paragraphs separated by \\n\\n>", "trending_keywords": ["kw1", "kw2", ...]}\n'
        "CRITICAL: executive_summary MUST NOT be empty."
    )


def _build_news_prompt(articles) -> str:
    """Prompt for call 2: highlighted news items only."""
    articles_text = ""
    for i, article in enumerate(articles, 1):
        articles_text += (
            f"\n[{i}] Title: {article.title}\n"
            f"    Source: {article.source}\n"
            f"    URL: {article.url}\n"
            f"    Content: {article.content[:200]}\n"
        )

    return (
        "You are a sports journalist. Write highlighted news items in VIETNAMESE.\n\n"
        f"Articles:\n{articles_text}\n\n"
        "For EACH article write one JSON object with:\n"
        "- headline: Vietnamese title\n"
        "- summary: 2 sentences in Vietnamese\n"
        "- source: source name as given\n"
        "- url: full URL as given\n\n"
        "Return ONLY a JSON array. No markdown. No extra text.\n"
        '[{"headline":"...","summary":"...","source":"...","url":"..."}, ...]'
    )


def _build_markdown(report: ReportSchema) -> str:
    """Convert a ReportSchema into a Markdown string."""
    generated_at_str = report.generated_at.strftime("%d/%m/%Y %H:%M (GMT+7)")

    lines = ["# Báo Cáo Thể Thao Tuần", ""]
    lines += [f"*Ngày tạo: {generated_at_str}*", ""]
    lines += ["---", ""]

    # Executive summary: ensure each paragraph is separated by a blank line
    lines += ["## Tổng Quan", ""]
    paragraphs = [p.strip() for p in report.executive_summary.split("\n") if p.strip()]
    for para in paragraphs:
        lines.append(para)
        lines.append("")

    lines += ["---", ""]
    lines += ["## Từ Khóa Nổi Bật", ""]
    for kw in report.trending_keywords:
        lines.append(f"- {kw}")
    lines.append("")

    lines += ["---", ""]
    lines += ["## Tin Tức Nổi Bật", ""]
    for item in report.highlighted_news:
        lines.append(f"### {item.headline}")
        sentences = [s.strip() for s in item.summary.split(". ") if s.strip()]
        summary = ". ".join(sentences)
        if summary and not summary.endswith("."):
            summary += "."
        lines.append(summary)
        lines.append(f"**Nguồn:** {item.source}  ")
        lines.append(f"**URL:** {item.url}")
        lines.append("")

    return "\n".join(lines)


def writer_node(state: ReportState) -> ReportState:
    """LangGraph node: generate the weekly sports report using 2 LLM calls."""
    articles = state.get("ranked_articles") or state.get("retrieved_articles", [])
    model_name = os.getenv("GROQ_LLM_MODEL", "grok-3-mini")
    output_path = os.getenv("REPORT_OUTPUT_PATH", DEFAULT_REPORT_OUTPUT_PATH)

    state["rewrite_count"] = state.get("rewrite_count", 0) + 1

    try:
        llm = ChatGroq(model=model_name, api_key=os.getenv("GROQ_API_KEY"))

        # --- Call 1: Executive Summary + Trending Keywords ---
        summary_prompt = _build_summary_prompt(articles)
        resp1 = llm.invoke(summary_prompt)
        raw1 = resp1.content.strip()
        logger.info("Writer call 1 (summary) response: %s", raw1[:300])

        if "```" in raw1:
            import re
            raw1 = re.sub(r"```(?:json)?\s*", "", raw1).strip()
        s1, e1 = raw1.find("{"), raw1.rfind("}") + 1
        if s1 != -1 and e1 > s1:
            raw1 = raw1[s1:e1]
        data1 = json.loads(raw1, strict=False)

        executive_summary = data1.get("executive_summary", "").strip()
        if not executive_summary:
            raise ValueError("Call 1 returned empty executive_summary")
        trending_keywords = data1.get("trending_keywords", [])

        # --- Call 2: Highlighted News (with retry) ---
        import re
        news_data = None
        for attempt in range(1, 3):
            news_prompt = _build_news_prompt(articles)
            resp2 = llm.invoke(news_prompt)
            raw2 = resp2.content.strip()
            logger.info("Writer call 2 attempt %d: %s", attempt, raw2[:200])
            if "```" in raw2:
                raw2 = re.sub(r"```(?:json)?\s*", "", raw2).strip()
            s2, e2 = raw2.find("["), raw2.rfind("]") + 1
            if s2 != -1 and e2 > s2:
                raw2 = raw2[s2:e2]
            try:
                news_data = json.loads(raw2, strict=False)
                break
            except Exception as exc:
                logger.warning("Call 2 attempt %d parse failed: %s", attempt, exc)

        if not news_data:
            raise ValueError("Failed to parse highlighted news after retries")

        highlighted_news = [
            HighlightedNewsItem(
                headline=item["headline"],
                summary=item["summary"],
                source=item["source"],
                url=item["url"],
            )
            for item in news_data
        ]

        VN_TZ = timezone(timedelta(hours=7))
        report = ReportSchema(
            executive_summary=executive_summary,
            trending_keywords=trending_keywords,
            highlighted_news=highlighted_news,
            generated_at=datetime.now(tz=VN_TZ),
        )

        markdown = _build_markdown(report)
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        state["report"] = report

    except Exception as exc:
        logger.error("Writer LLM call failed: %s", exc)
        state["error"] = f"Writer failed: {exc}"

    return state
