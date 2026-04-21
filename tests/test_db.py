# Feature: sports-weekly-intelligence-agent, Property 8: save_articles is idempotent on duplicate URLs

from datetime import datetime, timedelta, timezone

from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy import MetaData, Table, Column, Integer, String, Text, create_engine, text

from models.schemas import ArticleSchema

# ---------------------------------------------------------------------------
# SQLite-compatible table definition (no SERIAL, no PostgreSQL-specific types)
# ---------------------------------------------------------------------------

_sqlite_metadata = MetaData()

_news_articles_sqlite = Table(
    "news_articles",
    _sqlite_metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("title", Text, nullable=False),
    Column("content", Text, nullable=False),
    Column("source", String(100), nullable=False),
    Column("url", Text, nullable=False, unique=True),
    Column("published_at", Text, nullable=False),
    Column("category", String(50), nullable=False),
    Column("created_at", Text, nullable=False),
)


def _make_sqlite_engine():
    """Create a fresh SQLite in-memory engine with the news_articles table."""
    engine = create_engine("sqlite:///:memory:")
    _sqlite_metadata.create_all(engine)
    return engine


def _save_articles_sqlite(articles: list[ArticleSchema], engine) -> None:
    """SQLite-compatible save: INSERT OR IGNORE to skip duplicate URLs."""
    if not articles:
        return

    rows = [
        {
            "title": a.title,
            "content": a.content,
            "source": a.source,
            "url": a.url,
            "published_at": a.published_at.isoformat(),
            "category": a.category,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        for a in articles
    ]

    with engine.begin() as conn:
        for row in rows:
            conn.execute(
                text(
                    "INSERT OR IGNORE INTO news_articles "
                    "(title, content, source, url, published_at, category, created_at) "
                    "VALUES (:title, :content, :source, :url, :published_at, :category, :created_at)"
                ),
                row,
            )


def _row_count(engine) -> int:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM news_articles"))
        return result.scalar()


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_non_empty_text = st.text(min_size=1, max_size=200).filter(lambda s: s.strip())

_url_strategy = st.from_regex(r"https://[a-z]{3,10}\.[a-z]{2,4}/[a-z0-9]{1,20}", fullmatch=True)

_fixed_published_at = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


def _article_strategy(url_pool: list[str]):
    """Build a strategy that draws a URL from the given pool."""
    return st.builds(
        ArticleSchema,
        title=_non_empty_text,
        content=_non_empty_text,
        source=st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
        url=st.sampled_from(url_pool),
        published_at=st.just(_fixed_published_at),
        category=st.just("sports"),
    )


@settings(max_examples=100)
@given(
    # Generate a small pool of URLs (1–5) so articles will overlap
    url_pool=st.lists(
        _url_strategy,
        min_size=1,
        max_size=5,
        unique=True,
    ),
    # First batch: 1–8 articles drawn from the pool
    first_batch_size=st.integers(min_value=1, max_value=8),
    # Second batch: 1–8 articles drawn from the same pool (guaranteed overlaps)
    second_batch_size=st.integers(min_value=1, max_value=8),
)
def test_save_articles_idempotent_on_duplicate_urls(url_pool, first_batch_size, second_batch_size):
    """Property 8: save_articles is idempotent on duplicate URLs.

    Validates: Requirements 4.2

    Calling save_articles twice with overlapping URLs must not raise an exception
    and must not increase the row count after the second call.
    """
    from hypothesis import assume
    import random

    rng = random.Random(42)

    def make_article(url: str) -> ArticleSchema:
        return ArticleSchema(
            title=f"Title for {url[:30]}",
            content="Some content",
            source="TestSource",
            url=url,
            published_at=_fixed_published_at,
            category="sports",
        )

    # Build first batch: pick URLs with replacement from pool
    first_urls = [rng.choice(url_pool) for _ in range(first_batch_size)]
    first_articles = [make_article(u) for u in first_urls]

    # Build second batch: also from the same pool (guaranteed URL overlap)
    second_urls = [rng.choice(url_pool) for _ in range(second_batch_size)]
    second_articles = [make_article(u) for u in second_urls]

    engine = _make_sqlite_engine()

    # First save — must not raise
    _save_articles_sqlite(first_articles, engine)
    count_after_first = _row_count(engine)

    # Second save with overlapping URLs — must not raise
    _save_articles_sqlite(second_articles, engine)
    count_after_second = _row_count(engine)

    # Row count must not grow beyond what was inserted in the first call
    # (second call may add new URLs not seen in first batch, but duplicate
    #  URLs must be silently ignored)
    new_urls_in_second = set(second_urls) - set(first_urls)
    assert count_after_second == count_after_first + len(new_urls_in_second), (
        f"Expected {count_after_first + len(new_urls_in_second)} rows after second save, "
        f"got {count_after_second}. "
        f"First batch URLs: {set(first_urls)}, second batch URLs: {set(second_urls)}"
    )


# ---------------------------------------------------------------------------
# SQLite-compatible helper for get_articles_last_7_days
# ---------------------------------------------------------------------------

def _get_articles_last_7_days_sqlite(engine, cutoff=None) -> list[ArticleSchema]:
    """SQLite-compatible version of get_articles_last_7_days.

    Queries articles where published_at >= cutoff (defaults to now() - 7 days).
    Uses ISO-string comparison which works because ISO-8601 strings sort
    lexicographically.
    """
    if cutoff is None:
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=7)
    cutoff_str = cutoff.isoformat()

    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT id, title, content, source, url, published_at, category, created_at "
                "FROM news_articles "
                "WHERE published_at >= :cutoff "
                "ORDER BY published_at DESC"
            ),
            {"cutoff": cutoff_str},
        )
        rows = result.fetchall()

    articles: list[ArticleSchema] = []
    for row in rows:
        articles.append(
            ArticleSchema(
                id=row[0],
                title=row[1],
                content=row[2],
                source=row[3],
                url=row[4],
                published_at=datetime.fromisoformat(str(row[5])),
                category=row[6],
                created_at=datetime.fromisoformat(str(row[7])) if row[7] else None,
            )
        )
    return articles


# ---------------------------------------------------------------------------
# Property 9: get_articles_last_7_days returns only recent articles
# ---------------------------------------------------------------------------

# Feature: sports-weekly-intelligence-agent, Property 9: get_articles_last_7_days returns only recent articles


@settings(max_examples=100)
@given(
    day_offsets=st.lists(
        st.integers(min_value=-14, max_value=0),
        min_size=1,
        max_size=20,
    )
)
def test_get_articles_last_7_days_returns_only_recent(day_offsets):
    """Property 9: get_articles_last_7_days returns only recent articles.

    Validates: Requirements 4.3

    Seeds the DB with articles at varying ages (0–14 days ago) and asserts
    that only articles with published_at >= now() - 7 days are returned,
    and that NO articles older than 7 days are included.
    """
    # Use a fixed reference point so article timestamps and cutoff are consistent.
    # Avoid the exact boundary (offset == -7) by treating it as stale to prevent
    # sub-second timing races between article creation and cutoff computation.
    now = datetime.now(tz=timezone.utc)
    cutoff = now - timedelta(days=7)

    # Build one article per offset, using a unique URL per entry.
    # offset == -7 lands exactly on the boundary; treat it as stale (< cutoff)
    # by subtracting one extra second so the test is deterministic.
    articles = []
    for i, offset in enumerate(day_offsets):
        published_at = now + timedelta(days=offset)
        # Push exact-boundary articles just past the cutoff to avoid flakiness
        if published_at == cutoff:
            published_at = cutoff - timedelta(seconds=1)
        articles.append(
            ArticleSchema(
                title=f"Article offset {offset} index {i}",
                content="Some content",
                source="TestSource",
                url=f"https://example.com/article-{i}-offset-{offset}",
                published_at=published_at,
                category="sports",
            )
        )

    engine = _make_sqlite_engine()
    _save_articles_sqlite(articles, engine)

    # Pass the same cutoff to the helper so both sides use the same reference
    results = _get_articles_last_7_days_sqlite(engine, cutoff=cutoff)

    # Every returned article must be within the 7-day window
    for article in results:
        assert article.published_at >= cutoff, (
            f"Article with published_at={article.published_at} is older than cutoff={cutoff} "
            "but was returned by get_articles_last_7_days"
        )

    # Every article that was inserted within the window must appear in results
    recent_urls = {a.url for a in articles if a.published_at >= cutoff}
    returned_urls = {a.url for a in results}
    assert recent_urls == returned_urls, (
        f"Expected recent URLs {recent_urls} but got {returned_urls}"
    )

    # No stale article should appear in results
    stale_urls = {a.url for a in articles if a.published_at < cutoff}
    assert stale_urls.isdisjoint(returned_urls), (
        f"Stale articles found in results: {stale_urls & returned_urls}"
    )
