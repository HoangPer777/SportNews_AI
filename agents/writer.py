"""Writer agent for the Sports Weekly Intelligence Agent pipeline."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone

from core.llm import get_safe_llm
from models.schemas import HighlightedNewsItem, ReportSchema, ReportState

logger = logging.getLogger(__name__)

DEFAULT_REPORT_OUTPUT_PATH = "outputs/weekly_report.md"


def _build_summary_prompt(articles, period_type: str = "weekly") -> str:
    """Prompt for call 1: executive summary + trending keywords only."""
    articles_text = ""
    for i, article in enumerate(articles, 1):
        articles_text += (
            f"\n[{i}] Title: {article.title}\n"
            f"    Source: {article.source}\n"
            f"    Content: {article.content[:300]}\n"
        )

    report_label = "daily" if period_type == "daily" else "weekly"
    period_word = "trong ngày" if period_type == "daily" else "trong tuần"
    return (
        f"You are a senior Vietnamese sports journalist. Write the OVERVIEW section of a {report_label} sports report "
        "in polished VIETNAMESE with full diacritics.\n\n"
        f"Articles:\n{articles_text if articles_text else '(No articles available)'}\n\n"
        "Write ONLY these two fields:\n"
        f"1. executive_summary: Exactly 4 paragraphs in Vietnamese analyzing sports events {period_word}.\n"
        "   - Paragraph 1: Overall sports landscape (2-3 sentences).\n"
        "   - Paragraph 2: Most prominent trend with analysis (2-3 sentences).\n"
        "   - Paragraph 3: Second major development (2-3 sentences).\n"
        "   - Paragraph 4: Closing outlook (2 sentences).\n"
        "   STRICT RULES:\n"
        "   - Each paragraph MUST cover a DIFFERENT topic or event.\n"
        "   - Paragraph 4 MUST NOT restate, paraphrase, or echo Paragraph 1.\n"
        "   - Do NOT repeat the same idea, phrase, or event across paragraphs.\n"
        "   - Do NOT invent events, scores, quotes, names, or facts not present in the articles.\n"
        "   - If this is a daily report, do not write 'trong tuần qua'.\n"
        "2. trending_keywords: 8-12 specific keywords as a JSON array.\n"
        "   - Avoid generic keywords such as 'Thể thao', 'Tin tức', 'Bóng đá' unless paired with a specific event/team/person.\n\n"
        "Return ONLY a valid JSON object:\n"
        '{"executive_summary": "<4 paragraphs separated by \\n\\n>", "trending_keywords": ["kw1", "kw2", ...]}\n'
        "CRITICAL: executive_summary MUST NOT be empty."
    )


def _build_news_prompt(articles, period_type: str = "weekly") -> str:
    """Prompt for call 2: highlighted news items only."""
    articles_text = ""
    for i, article in enumerate(articles, 1):
        articles_text += (
            f"\n[{i}] Title: {article.title}\n"
            f"    Source: {article.source}\n"
            f"    URL: {article.url}\n"
            f"    Content: {article.content[:200]}\n"
        )

    report_label = "daily" if period_type == "daily" else "weekly"
    return (
        f"You are a Vietnamese sports journalist. Write highlighted news items for a {report_label} report "
        "in natural Vietnamese with full diacritics.\n\n"
        f"Articles:\n{articles_text}\n\n"
        "For EACH article write one JSON object with:\n"
        "- headline: Vietnamese title\n"
        "- summary: exactly 2 sentences in Vietnamese\n"
        "- source: source name exactly as given\n"
        "- url: full URL exactly as given\n\n"
        "Rules:\n"
        "- Keep the exact source and URL from the article list.\n"
        "- Do not invent new facts, scores, quotes, names, or events.\n"
        "- Prefer clear journalistic language over promotional language.\n\n"
        "Return ONLY a JSON array. No markdown. No extra text.\n"
        '[{"headline":"...","summary":"...","source":"...","url":"..."}, ...]'
    )


def _build_markdown(report: ReportSchema, period_type: str = "weekly") -> str:
    """Convert a ReportSchema into a Markdown string."""
    generated_at_str = report.generated_at.strftime("%d/%m/%Y %H:%M (GMT+7)")

    title = "Báo Cáo Thể Thao Hôm Nay" if period_type == "daily" else "Báo Cáo Thể Thao Tuần"
    lines = [f"# {title}", ""]
    lines += [f"*Ngày tạo: {generated_at_str}*", ""]
    lines += ["---", ""]

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
    """LangGraph node: generate the sports report using 2 LLM calls."""
    articles = state.get("ranked_articles") or state.get("retrieved_articles", [])
    metadata = state.get("metadata")
    period_type = metadata.period_type if metadata else "weekly"
    default_output_path = "outputs/daily_report.md" if period_type == "daily" else DEFAULT_REPORT_OUTPUT_PATH
    output_path = os.getenv("REPORT_OUTPUT_PATH", default_output_path)

    state["rewrite_count"] = state.get("rewrite_count", 0) + 1

    try:
        llm = get_safe_llm("writer")

        summary_prompt = _build_summary_prompt(articles, period_type=period_type)
        resp1 = llm.invoke(summary_prompt)
        raw1 = resp1.content.strip()
        logger.info("Writer call 1 (summary) response: %s", raw1[:300])

        if "```" in raw1:
            raw1 = re.sub(r"```(?:json)?\s*", "", raw1).strip()
        s1, e1 = raw1.find("{"), raw1.rfind("}") + 1
        if s1 != -1 and e1 > s1:
            raw1 = raw1[s1:e1]
        data1 = json.loads(raw1, strict=False)

        executive_summary = data1.get("executive_summary", "").strip()
        if not executive_summary:
            raise ValueError("Call 1 returned empty executive_summary")
        trending_keywords = data1.get("trending_keywords", [])

        news_data = None
        for attempt in range(1, 3):
            news_prompt = _build_news_prompt(articles, period_type=period_type)
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

        vn_tz = timezone(timedelta(hours=7))
        report = ReportSchema(
            executive_summary=executive_summary,
            trending_keywords=trending_keywords,
            highlighted_news=highlighted_news,
            generated_at=datetime.now(tz=vn_tz),
        )

        markdown = _build_markdown(report, period_type=period_type)
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        state["report"] = report

    except Exception as exc:
        logger.error("Writer LLM call failed: %s", exc)
        state["error"] = f"Writer failed: {exc}"

    return state
