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


def _build_prompt(articles) -> str:
    """Build the LLM prompt from retrieved articles."""
    articles_text = ""
    for i, article in enumerate(articles, 1):
        articles_text += (
            f"\n[{i}] Title: {article.title}\n"
            f"    Source: {article.source}\n"
            f"    URL: {article.url}\n"
            f"    Content: {article.content[:500]}\n"
        )

    return (
        "You are a professional sports journalist. Write a weekly sports intelligence report.\n"
        "ALL output text (summaries, headlines, executive summary) MUST be written in VIETNAMESE.\n"
        "Only keep proper nouns (team names, player names) in their original language.\n\n"
        f"Articles:\n{articles_text if articles_text else '(No articles available)'}\n\n"
        "Requirements:\n"
        "1. executive_summary: EXACTLY 4 paragraphs in Vietnamese:\n"
        "   - Paragraph 1: Overall landscape of the week in sports (2-3 sentences)\n"
        "   - Paragraph 2: Most prominent trend or storyline with analysis (2-3 sentences)\n"
        "   - Paragraph 3: Second major trend or notable development (2-3 sentences)\n"
        "   - Paragraph 4: Closing outlook or summary of what to watch next (2 sentences)\n"
        "2. trending_keywords: 8-12 significant keywords.\n"
        "3. highlighted_news: Select the 8-10 MOST IMPORTANT articles. Must include articles from ALL available sources (VnExpress, Thanh Nien, Tuoi Tre). Write 2-3 sentence summary in Vietnamese for each.\n"
        "4. The executive_summary must reference events from highlighted_news.\n\n"
        "Return ONLY a valid JSON object (no markdown, no extra text):\n"
        "{\n"
        '  "executive_summary": "<exactly 4 paragraphs separated by \\n\\n>",\n'
        '  "trending_keywords": ["keyword1", "keyword2"],\n'
        '  "highlighted_news": [\n'
        '    {"headline": "<Vietnamese headline>", "summary": "<2-3 sentences in Vietnamese>", "source": "...", "url": "..."}\n'
        "  ]\n"
        "}\n"
        "IMPORTANT: Return ONLY the JSON. No markdown fences. No text before or after."
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
    """LangGraph node: generate the weekly sports report from retrieved articles."""
    articles = state.get("retrieved_articles", [])
    model_name = os.getenv("GROQ_LLM_MODEL", "grok-3-mini")
    output_path = os.getenv("REPORT_OUTPUT_PATH", DEFAULT_REPORT_OUTPUT_PATH)

    # Increment rewrite count on every invocation
    state["rewrite_count"] = state.get("rewrite_count", 0) + 1

    prompt = _build_prompt(articles)

    try:
        llm = ChatGroq(model=model_name, api_key=os.getenv("GROQ_API_KEY"))
        response = llm.invoke(prompt)
        raw = response.content.strip()
        logger.info("Writer LLM raw response (first 500 chars): %s", raw[:500])

        # Strip markdown code fences if present (```json ... ``` or ``` ... ```)
        if "```" in raw:
            import re
            raw = re.sub(r"```(?:json)?\s*", "", raw).strip()

        # Find the JSON object boundaries in case there's surrounding text
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            raw = raw[start:end]

        data = json.loads(raw, strict=False)

        highlighted_news = [
            HighlightedNewsItem(
                headline=item["headline"],
                summary=item["summary"],
                source=item["source"],
                url=item["url"],
            )
            for item in data.get("highlighted_news", [])
        ]

        VN_TZ = timezone(timedelta(hours=7))
        report = ReportSchema(
            executive_summary=data["executive_summary"],
            trending_keywords=data.get("trending_keywords", []),
            highlighted_news=highlighted_news,
            generated_at=datetime.now(tz=VN_TZ),
        )

        # Save Markdown to disk
        markdown = _build_markdown(report)
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        state["report"] = report

    except Exception as exc:
        logger.error("Writer LLM call failed: %s", exc)
        state["error"] = f"Writer failed: {exc}"

    return state
