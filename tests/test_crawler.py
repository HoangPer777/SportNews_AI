"""
Property-based tests for tools/crawler.py.

Properties covered:
  - Property 2: Crawled articles have all required fields
  - Property 3: Crawled articles are within the 7-day window
  - Property 4: crawl_all_sources returns union of all three sources
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from models.schemas import ArticleSchema
from tools.crawler import crawl_all_sources

# ---------------------------------------------------------------------------
# HTML template helpers
# ---------------------------------------------------------------------------

def _make_section_html(article_urls: list[str], url_pattern: str) -> str:
    """Build a minimal section-page HTML with links matching the given pattern."""
    links = "\n".join(f'<a href="{u}">Article</a>' for u in article_urls)
    return f"<html><body>{links}</body></html>"


def _make_article_html(title: str, content: str, published_iso: str) -> str:
    """Build a minimal article-page HTML that the crawler can parse."""
    return (
        f"<html><head>"
        f'<meta property="article:published_time" content="{published_iso}"/>'
        f"</head><body>"
        f"<h1>{title}</h1>"
        f'<div class="fck_detail">{content}</div>'
        f"</body></html>"
    )


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_non_empty_text = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters=" "),
    min_size=3,
    max_size=80,
).filter(lambda s: s.strip())

# Dates within the last 6 days (safely inside the 7-day window)
_recent_date_strategy = st.integers(min_value=0, max_value=5).map(
    lambda days_ago: (datetime.now(tz=timezone.utc) - timedelta(days=days_ago)).isoformat()
)

# Dates older than 8 days (safely outside the 7-day window)
_stale_date_strategy = st.integers(min_value=8, max_value=30).map(
    lambda days_ago: (datetime.now(tz=timezone.utc) - timedelta(days=days_ago)).isoformat()
)


def _build_mock_responses(
    section_url: str,
    article_urls: list[str],
    url_pattern: str,
    titles: list[str],
    contents: list[str],
    dates: list[str],
) -> dict[str, MagicMock]:
    """Return a mapping of URL → mock Response for use with side_effect."""
    mapping: dict[str, MagicMock] = {}

    section_resp = MagicMock()
    section_resp.text = _make_section_html(article_urls, url_pattern)
    section_resp.raise_for_status = MagicMock()
    mapping[section_url] = section_resp

    for url, title, content, date in zip(article_urls, titles, contents, dates):
        resp = MagicMock()
        resp.text = _make_article_html(title, content, date)
        resp.raise_for_status = MagicMock()
        mapping[url] = resp

    return mapping


def _side_effect_factory(mapping: dict[str, MagicMock]):
    """Return a side_effect callable that dispatches on the first positional arg (url)."""
    def _side_effect(url, **kwargs):
        if url in mapping:
            return mapping[url]
        # Unknown URL → return empty section page
        resp = MagicMock()
        resp.text = "<html><body></body></html>"
        resp.raise_for_status = MagicMock()
        return resp
    return _side_effect


# ---------------------------------------------------------------------------
# Property 2: Crawled articles have all required fields
# ---------------------------------------------------------------------------

# Feature: sports-weekly-intelligence-agent, Property 2: Crawled articles have all required fields

@settings(max_examples=50)
@given(
    titles=st.lists(_non_empty_text, min_size=1, max_size=3),
    contents=st.lists(_non_empty_text, min_size=1, max_size=3),
)
def test_crawled_articles_have_all_required_fields(titles, contents):
    """Property 2: Crawled articles have all required fields.

    **Validates: Requirements 2.2, 2.4**

    Mocks requests.get to return synthetic HTML for each source.
    Asserts every article from crawl_all_sources() has non-empty title,
    content, source, url, published_at, and category == "sports".
    """
    # Use the minimum length across the three lists so we can zip them safely
    n = min(len(titles), len(contents))
    titles = titles[:n]
    contents = contents[:n]

    # A recent date that passes the 7-day filter
    recent_date = (datetime.now(tz=timezone.utc) - timedelta(days=1)).isoformat()
    dates = [recent_date] * n

    # --- VnExpress ---
    vne_section = "https://vnexpress.net/the-thao"
    vne_urls = [f"https://vnexpress.net/the-thao/article-{i}.html" for i in range(n)]
    vne_map = _build_mock_responses(vne_section, vne_urls, r"/the-thao/.*\.html", titles, contents, dates)

    # --- Thanh Nien ---
    tn_section = "https://thanhnien.vn/the-thao/"
    tn_urls = [f"https://thanhnien.vn/the-thao/article-{i}.htm" for i in range(n)]
    tn_map = _build_mock_responses(tn_section, tn_urls, r"/the-thao/.*\.htm", titles, contents, dates)

    # --- Tuoi Tre ---
    tt_section = "https://tuoitre.vn/the-thao.htm"
    tt_urls = [f"https://tuoitre.vn/the-thao/article-{i}.htm" for i in range(n)]
    tt_map = _build_mock_responses(tt_section, tt_urls, r"/the-thao/.*\.htm", titles, contents, dates)

    combined_map = {**vne_map, **tn_map, **tt_map}

    with patch("tools.crawler.requests.get", side_effect=_side_effect_factory(combined_map)):
        articles = crawl_all_sources()

    for article in articles:
        assert article.title and article.title.strip(), (
            f"Article has empty title: {article}"
        )
        assert article.content and article.content.strip(), (
            f"Article has empty content: {article}"
        )
        assert article.source and article.source.strip(), (
            f"Article has empty source: {article}"
        )
        assert article.url and article.url.strip(), (
            f"Article has empty url: {article}"
        )
        assert article.published_at is not None, (
            f"Article has None published_at: {article}"
        )
        assert article.category == "sports", (
            f"Article category is not 'sports': {article.category!r}"
        )


# ---------------------------------------------------------------------------
# Property 3: Crawled articles are within the 7-day window
# ---------------------------------------------------------------------------

# Feature: sports-weekly-intelligence-agent, Property 3: Crawled articles are within the 7-day window

@settings(max_examples=50)
@given(
    recent_count=st.integers(min_value=1, max_value=3),
    stale_count=st.integers(min_value=1, max_value=3),
    recent_days_ago=st.lists(st.integers(min_value=0, max_value=5), min_size=1, max_size=3),
    stale_days_ago=st.lists(st.integers(min_value=8, max_value=30), min_size=1, max_size=3),
)
def test_crawled_articles_within_7_day_window(
    recent_count, stale_count, recent_days_ago, stale_days_ago
):
    """Property 3: Crawled articles are within the 7-day window.

    **Validates: Requirements 2.3**

    Mocks requests.get with articles at varying dates (some within 7 days,
    some older). Asserts all returned articles have published_at >= now() - 7 days.
    """
    now = datetime.now(tz=timezone.utc)
    cutoff = now - timedelta(days=7)

    # Build recent dates (within window) and stale dates (outside window)
    recent_dates = [
        (now - timedelta(days=d)).isoformat()
        for d in (recent_days_ago * recent_count)[:recent_count]
    ]
    stale_dates = [
        (now - timedelta(days=d)).isoformat()
        for d in (stale_days_ago * stale_count)[:stale_count]
    ]

    all_dates = recent_dates + stale_dates
    n = len(all_dates)
    titles = [f"Article {i}" for i in range(n)]
    contents = ["Some sports content here"] * n

    # --- VnExpress ---
    vne_section = "https://vnexpress.net/the-thao"
    vne_urls = [f"https://vnexpress.net/the-thao/article-{i}.html" for i in range(n)]
    vne_map = _build_mock_responses(vne_section, vne_urls, r"/the-thao/.*\.html", titles, contents, all_dates)

    # --- Thanh Nien ---
    tn_section = "https://thanhnien.vn/the-thao/"
    tn_urls = [f"https://thanhnien.vn/the-thao/article-{i}.htm" for i in range(n)]
    tn_map = _build_mock_responses(tn_section, tn_urls, r"/the-thao/.*\.htm", titles, contents, all_dates)

    # --- Tuoi Tre ---
    tt_section = "https://tuoitre.vn/the-thao.htm"
    tt_urls = [f"https://tuoitre.vn/the-thao/article-{i}.htm" for i in range(n)]
    tt_map = _build_mock_responses(tt_section, tt_urls, r"/the-thao/.*\.htm", titles, contents, all_dates)

    combined_map = {**vne_map, **tn_map, **tt_map}

    with patch("tools.crawler.requests.get", side_effect=_side_effect_factory(combined_map)):
        articles = crawl_all_sources()

    for article in articles:
        pub = article.published_at
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        assert pub >= cutoff, (
            f"Article published_at={pub} is older than cutoff={cutoff} "
            f"but was returned by crawl_all_sources()"
        )


# ---------------------------------------------------------------------------
# Property 4: crawl_all_sources returns union of all three sources
# ---------------------------------------------------------------------------

# Feature: sports-weekly-intelligence-agent, Property 4: crawl_all_sources returns union of all three sources

def _make_articles(source: str, n: int) -> list[ArticleSchema]:
    """Build a list of n ArticleSchema objects for the given source."""
    now = datetime.now(tz=timezone.utc)
    return [
        ArticleSchema(
            title=f"{source} Article {i}",
            content=f"Content for {source} article {i}",
            source=source,
            url=f"https://example.com/{source.lower().replace(' ', '-')}/article-{i}",
            published_at=now - timedelta(hours=i),
            category="sports",
        )
        for i in range(n)
    ]


@settings(max_examples=100)
@given(
    vne_count=st.integers(min_value=0, max_value=5),
    tn_count=st.integers(min_value=0, max_value=5),
    tt_count=st.integers(min_value=0, max_value=5),
)
def test_crawl_all_sources_returns_union(vne_count, tn_count, tt_count):
    """Property 4: crawl_all_sources returns union of all three sources.

    **Validates: Requirements 2.1**

    Mocks each individual crawler function and asserts crawl_all_sources()
    result contains all articles from all three mocked sources.
    """
    vne_articles = _make_articles("VnExpress", vne_count)
    tn_articles = _make_articles("Thanh Nien", tn_count)
    tt_articles = _make_articles("Tuoi Tre", tt_count)

    with (
        patch("tools.crawler.crawl_vnexpress", return_value=vne_articles) as _mock_vne,
        patch("tools.crawler.crawl_thanhnien", return_value=tn_articles) as _mock_tn,
        patch("tools.crawler.crawl_tuoitre", return_value=tt_articles) as _mock_tt,
    ):
        result = crawl_all_sources()

    expected_urls = (
        {a.url for a in vne_articles}
        | {a.url for a in tn_articles}
        | {a.url for a in tt_articles}
    )
    result_urls = {a.url for a in result}

    assert result_urls == expected_urls, (
        f"crawl_all_sources() returned {result_urls!r}, "
        f"expected union {expected_urls!r}"
    )

    assert len(result) == vne_count + tn_count + tt_count, (
        f"Expected {vne_count + tn_count + tt_count} articles total, got {len(result)}"
    )
