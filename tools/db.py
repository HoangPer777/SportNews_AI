from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

from sqlalchemy import (
    Column,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

from models.schemas import ArticleSchema

metadata = MetaData()

news_articles = Table(
    "news_articles",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("title", Text, nullable=False),
    Column("content", Text, nullable=False),
    Column("source", String(100), nullable=False),
    Column("url", Text, nullable=False, unique=True),
    Column("published_at", Text, nullable=False),
    Column("category", String(50), nullable=False, server_default="sports"),
    Column("created_at", Text, nullable=False, server_default=text("NOW()")),
    Column("embedding", Text, nullable=True),  # JSON-serialized float list
    Index("idx_news_articles_published_at", "published_at"),
    Index("idx_news_articles_category", "category"),
)


def get_engine() -> Engine:
    """Build and return a SQLAlchemy engine from DATABASE_URL env var."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise ValueError(
            "DATABASE_URL environment variable is not set. "
            "Please set it to a valid SQLAlchemy connection string, "
            "e.g. postgresql://user:pass@localhost:5432/sportsdb"
        )
    engine = create_engine(url)
    metadata.create_all(engine)
    # Add embedding column if it doesn't exist yet (for existing DBs)
    with engine.begin() as conn:
        conn.execute(text(
            "ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS embedding TEXT"
        ))
    return engine


def save_articles(articles: list[ArticleSchema], engine: Engine | None = None) -> None:
    """Bulk-insert articles, silently skipping rows with duplicate URLs."""
    if not articles:
        return
    if engine is None:
        engine = get_engine()

    rows = [
        {
            "title": a.title,
            "content": a.content,
            "source": a.source,
            "url": a.url,
            "published_at": a.published_at.isoformat(),
            "category": a.category,
        }
        for a in articles
    ]

    with engine.begin() as conn:
        stmt = pg_insert(news_articles).values(rows)
        stmt = stmt.on_conflict_do_nothing(index_elements=["url"])
        conn.execute(stmt)


def get_articles_last_7_days(engine: Engine | None = None) -> list[ArticleSchema]:
    """Return all articles whose published_at is within the last 7 days."""
    if engine is None:
        engine = get_engine()

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=7)

    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT id, title, content, source, url, published_at, category, created_at, embedding "
                "FROM news_articles "
                "WHERE published_at >= :cutoff "
                "ORDER BY published_at DESC"
            ),
            {"cutoff": cutoff.isoformat()},
        )
        rows = result.fetchall()

    articles: list[ArticleSchema] = []
    for row in rows:
        a = ArticleSchema(
            id=row[0],
            title=row[1],
            content=row[2],
            source=row[3],
            url=row[4],
            published_at=datetime.fromisoformat(str(row[5])),
            category=row[6],
            created_at=datetime.fromisoformat(str(row[7])) if row[7] else None,
        )
        # Attach cached embedding if present
        if row[8]:
            try:
                a.embedding = json.loads(row[8])
            except Exception:
                pass
        articles.append(a)
    return articles


def save_embeddings(url_embedding_pairs: list[tuple[str, list[float]]], engine: Engine | None = None) -> None:
    """Persist computed embeddings back to DB by URL."""
    if not url_embedding_pairs:
        return
    if engine is None:
        engine = get_engine()

    with engine.begin() as conn:
        for url, embedding in url_embedding_pairs:
            conn.execute(
                text("UPDATE news_articles SET embedding = :emb WHERE url = :url"),
                {"emb": json.dumps(embedding), "url": url},
            )
