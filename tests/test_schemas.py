# Feature: sports-weekly-intelligence-agent, Property 16: Highlighted news items contain all required fields

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from models.schemas import HighlightedNewsItem

non_empty_text = st.text(min_size=1)


@settings(max_examples=100)
@given(
    headline=non_empty_text,
    summary=non_empty_text,
    source=non_empty_text,
    url=non_empty_text,
)
def test_highlighted_news_item_required_fields(headline, summary, source, url):
    """Property 16: Highlighted news items contain all required fields.

    Validates: Requirements 8.4
    """
    item = HighlightedNewsItem(headline=headline, summary=summary, source=source, url=url)

    assert isinstance(item.headline, str) and len(item.headline) > 0
    assert isinstance(item.summary, str) and len(item.summary) > 0
    assert isinstance(item.source, str) and len(item.source) > 0
    assert isinstance(item.url, str) and len(item.url) > 0
