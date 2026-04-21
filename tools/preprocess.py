"""
Preprocessor for crawled sports articles.

Provides:
  - clean_text: strip HTML, normalize whitespace
  - deduplicate_articles: remove duplicate URLs then duplicate titles
  - filter_recent_articles: keep only sports articles from the last 7 days
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from bs4 import BeautifulSoup

from models.schemas import ArticleSchema


def clean_text(text: str) -> str:
    """Strip HTML tags, normalize whitespace, and strip leading/trailing whitespace.

    Requirements: 3.1
    """
    # Remove HTML tags via BeautifulSoup
    soup = BeautifulSoup(text, "html.parser")
    cleaned = soup.get_text(separator=" ")

    # Remove any residual angle-bracket characters (e.g. bare '<' or '>')
    cleaned = cleaned.replace("<", "").replace(">", "")

    # Normalize all whitespace sequences (spaces, tabs, newlines) to a single space
    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned.strip()


def deduplicate_articles(articles: list[ArticleSchema]) -> list[ArticleSchema]:
    """Deduplicate articles by URL first, then by title (preserve first occurrence).

    Requirements: 3.2
    """
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    result: list[ArticleSchema] = []

    for article in articles:
        if article.url in seen_urls:
            continue
        if article.title in seen_titles:
            continue
        seen_urls.add(article.url)
        seen_titles.add(article.title)
        result.append(article)

    return result


def filter_recent_articles(articles: list[ArticleSchema]) -> list[ArticleSchema]:
    """Keep only articles published within the last 7 days with category == 'sports'.

    Requirements: 3.3, 3.4
    """
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=7)
    result: list[ArticleSchema] = []

    for article in articles:
        pub = article.published_at
        # Ensure timezone-aware comparison
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        if pub >= cutoff and article.category == "sports":
            result.append(article)

    return result
