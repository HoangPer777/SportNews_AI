"""
Property-based tests for tools/preprocess.py.

Properties covered:
  - Property 5: clean_text produces HTML-free, normalized output
  - Property 6: deduplicate_articles produces unique URL and title set
  - Property 7: filter_recent_articles removes stale and non-sports articles
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from hypothesis import given, settings
from hypothesis import strategies as st

from models.schemas import ArticleSchema
from tools.preprocess import clean_text, deduplicate_articles, filter_recent_articles


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# HTML tag fragments to embed in generated text
_html_tags = st.sampled_from([
    "<b>", "</b>", "<p>", "</p>", "<div>", "</div>",
    "<span>", "</span>", "<a href='x'>", "</a>",
    "<br/>", "<img src='x'/>", "<h1>", "</h1>",
])

_text_with_html = st.builds(
    lambda base, tag, pos: base[:pos] + tag + base[pos:],
    base=st.text(min_size=0, max_size=100),
    tag=_html_tags,
    pos=st.integers(min_value=0, max_value=100),
)

_article_text = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters=" "),
    min_size=1,
    max_size=50,
).filter(lambda s: s.strip())


def _make_article(
    title: str,
    url: str,
    category: str = "sports",
    days_ago: float = 1.0,
) -> ArticleSchema:
    pub = datetime.now(tz=timezone.utc) - timedelta(days=days_ago)
    return ArticleSchema(
        title=title,
        content="Some content",
        source="TestSource",
        url=url,
        published_at=pub,
        category=category,
    )


# ---------------------------------------------------------------------------
# Property 5: clean_text produces HTML-free, normalized output
# ---------------------------------------------------------------------------

# Feature: sports-weekly-intelligence-agent, Property 5: clean_text produces HTML-free, normalized output

@settings(max_examples=100)
@given(text=st.one_of(st.text(), _text_with_html))
def test_clean_text_html_free_normalized(text):
    """Property 5: clean_text produces HTML-free, normalized output.

    **Validates: Requirements 3.1**

    For any input string (including strings with HTML tags, multiple spaces,
    tabs, and leading/trailing whitespace), clean_text(text) must return a
    string with no HTML tags, no consecutive whitespace, and no leading or
    trailing whitespace.
    """
    result = clean_text(text)

    # No HTML angle brackets remain
    assert "<" not in result, f"Output contains '<': {result!r}"
    assert ">" not in result, f"Output contains '>': {result!r}"

    # No consecutive whitespace
    assert not re.search(r"\s{2,}", result), (
        f"Output contains consecutive whitespace: {result!r}"
    )

    # No leading or trailing whitespace
    assert result == result.strip(), (
        f"Output has leading/trailing whitespace: {result!r}"
    )


# ---------------------------------------------------------------------------
# Property 6: deduplicate_articles produces unique URL and title set
# ---------------------------------------------------------------------------

# Feature: sports-weekly-intelligence-agent, Property 6: deduplicate_articles produces unique URL and title set

@settings(max_examples=100)
@given(
    base_titles=st.lists(_article_text, min_size=1, max_size=5),
    base_urls=st.lists(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=3, max_size=20),
        min_size=1,
        max_size=5,
    ),
    duplicate_factor=st.integers(min_value=1, max_value=3),
)
def test_deduplicate_articles_unique_urls_and_titles(base_titles, base_urls, duplicate_factor):
    """Property 6: deduplicate_articles produces unique URL and title set.

    **Validates: Requirements 3.2**

    For any list of articles (including lists with duplicate URLs and duplicate
    titles), deduplicate_articles(articles) must return a list where every URL
    is unique and every title is unique.
    """
    # Build articles with intentional duplicates
    articles: list[ArticleSchema] = []
    for i, (title, url_suffix) in enumerate(zip(base_titles, base_urls)):
        full_url = f"https://example.com/{url_suffix}"
        for _ in range(duplicate_factor):
            articles.append(_make_article(title=title, url=full_url))

    result = deduplicate_articles(articles)

    urls = [a.url for a in result]
    titles = [a.title for a in result]

    assert len(urls) == len(set(urls)), (
        f"Duplicate URLs found in result: {urls}"
    )
    assert len(titles) == len(set(titles)), (
        f"Duplicate titles found in result: {titles}"
    )


# ---------------------------------------------------------------------------
# Property 7: filter_recent_articles removes stale and non-sports articles
# ---------------------------------------------------------------------------

# Feature: sports-weekly-intelligence-agent, Property 7: filter_recent_articles removes stale and non-sports articles

@settings(max_examples=100)
@given(
    recent_sports=st.lists(
        st.integers(min_value=0, max_value=6),  # days ago, within window
        min_size=0,
        max_size=5,
    ),
    stale_sports=st.lists(
        st.integers(min_value=8, max_value=30),  # days ago, outside window
        min_size=0,
        max_size=5,
    ),
    recent_non_sports=st.lists(
        st.integers(min_value=0, max_value=6),
        min_size=0,
        max_size=5,
    ),
    stale_non_sports=st.lists(
        st.integers(min_value=8, max_value=30),
        min_size=0,
        max_size=5,
    ),
    non_sports_category=st.sampled_from(["politics", "tech", "entertainment", "business"]),
)
def test_filter_recent_articles_removes_stale_and_non_sports(
    recent_sports,
    stale_sports,
    recent_non_sports,
    stale_non_sports,
    non_sports_category,
):
    """Property 7: filter_recent_articles removes stale and non-sports articles.

    **Validates: Requirements 3.3, 3.4**

    For any list of articles containing a mix of recent and old articles, and
    sports and non-sports categories, filter_recent_articles(articles) must
    return only articles where published_at >= now() - 7 days and
    category == 'sports'.
    """
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=7)
    articles: list[ArticleSchema] = []
    idx = 0

    for days_ago in recent_sports:
        articles.append(_make_article(
            title=f"Recent sports {idx}",
            url=f"https://example.com/rs/{idx}",
            category="sports",
            days_ago=days_ago,
        ))
        idx += 1

    for days_ago in stale_sports:
        articles.append(_make_article(
            title=f"Stale sports {idx}",
            url=f"https://example.com/ss/{idx}",
            category="sports",
            days_ago=days_ago,
        ))
        idx += 1

    for days_ago in recent_non_sports:
        articles.append(_make_article(
            title=f"Recent non-sports {idx}",
            url=f"https://example.com/rns/{idx}",
            category=non_sports_category,
            days_ago=days_ago,
        ))
        idx += 1

    for days_ago in stale_non_sports:
        articles.append(_make_article(
            title=f"Stale non-sports {idx}",
            url=f"https://example.com/sns/{idx}",
            category=non_sports_category,
            days_ago=days_ago,
        ))
        idx += 1

    result = filter_recent_articles(articles)

    for article in result:
        pub = article.published_at
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)

        assert pub >= cutoff, (
            f"Stale article passed filter: published_at={pub}, cutoff={cutoff}"
        )
        assert article.category == "sports", (
            f"Non-sports article passed filter: category={article.category!r}"
        )
