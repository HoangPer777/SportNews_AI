from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

from graph import article_limit_for_period, limit_articles_for_report
from models.schemas import ArticleSchema


def _article(index: int) -> ArticleSchema:
    return ArticleSchema(
        title=f"Article {index}",
        content="content",
        source="VnExpress",
        url=f"https://example.com/{index}",
        published_at=datetime(2026, 1, 1),
        category="sports",
    )


def test_daily_report_article_limit_uses_env_value() -> None:
    articles = [_article(i) for i in range(10)]

    with patch.dict("os.environ", {"MAX_DAILY_REPORT_ARTICLES": "3"}, clear=False):
        limited = limit_articles_for_report(articles, "daily")

    assert len(limited) == 3
    assert [article.url for article in limited] == [articles[0].url, articles[1].url, articles[2].url]


def test_weekly_report_article_limit_uses_env_value() -> None:
    articles = [_article(i) for i in range(10)]

    with patch.dict("os.environ", {"MAX_WEEKLY_REPORT_ARTICLES": "7"}, clear=False):
        limited = limit_articles_for_report(articles, "weekly")

    assert len(limited) == 7


def test_article_limit_falls_back_when_env_invalid() -> None:
    with patch.dict("os.environ", {"MAX_REPORT_ARTICLES": "bad-value"}, clear=False):
        assert article_limit_for_period("daily") == 25
