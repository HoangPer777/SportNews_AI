"""Writer agent for the Sports Weekly Intelligence Agent pipeline."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime

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
        "You are a professional sports journalist writing a weekly intelligence report.\n"
        "Based on the following articles, produce a structured JSON report.\n"
        "If no articles are provided, write a general sports overview.\n\n"
        f"Articles:\n{articles_text if articles_text else '(No articles available)'}\n\n"
        "Return ONLY a valid JSON object (no markdown, no extra text) with this exact structure:\n"
        "{\n"
        '  "executive_summary": "<2-3 paragraph summary of the week in sports>",\n'
        '  "trending_keywords": ["keyword1", "keyword2", "keyword3"],\n'
        '  "highlighted_news": [\n'
        '    {"headline": "...", "summary": "...", "source": "General", "url": "https://example.com"}\n'
        "  ]\n"
        "}\n"
        "IMPORTANT: Return ONLY the JSON object, nothing else."
    )


def _build_markdown(report: ReportSchema) -> str:
    """Convert a ReportSchema into a Markdown string."""
    lines = ["# Weekly Sports Report", ""]

    lines += ["## Executive Summary", "", report.executive_summary, ""]

    lines += ["## Trending Keywords", ""]
    for kw in report.trending_keywords:
        lines.append(f"- {kw}")
    lines.append("")

    lines += ["## Highlighted News", ""]
    for item in report.highlighted_news:
        lines.append(f"### {item.headline}")
        lines.append(f"**Source:** {item.source}  ")
        lines.append(f"**URL:** {item.url}  ")
        lines.append("")
        lines.append(item.summary)
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

        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(
                line for line in lines if not line.startswith("```")
            ).strip()

        data = json.loads(raw)

        highlighted_news = [
            HighlightedNewsItem(
                headline=item["headline"],
                summary=item["summary"],
                source=item["source"],
                url=item["url"],
            )
            for item in data.get("highlighted_news", [])
        ]

        report = ReportSchema(
            executive_summary=data["executive_summary"],
            trending_keywords=data.get("trending_keywords", []),
            highlighted_news=highlighted_news,
            generated_at=datetime.utcnow(),
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
