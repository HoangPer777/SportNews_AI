"""Property-based tests for tools/embeddings.py."""
import os
import tempfile
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from models.schemas import ArticleSchema
from tools.embeddings import build_faiss_index, embed_articles, load_faiss_index

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

FIXED_DIM = 768  # fixed embedding dimension used in mocked embed_text


def article_strategy():
    """Generate a minimal valid ArticleSchema."""
    return st.builds(
        ArticleSchema,
        title=st.text(min_size=1, max_size=50),
        content=st.text(min_size=1, max_size=200),
        source=st.text(min_size=1, max_size=20),
        url=st.text(min_size=1, max_size=100),
        published_at=st.just(datetime(2024, 1, 1)),
        category=st.just("sports"),
    )


def fixed_embedding(_text: str) -> list[float]:
    """Mock embed_text that always returns a fixed-dimension vector."""
    return [0.1] * FIXED_DIM


# ---------------------------------------------------------------------------
# Property 10: embed_articles returns one embedding per article
# ---------------------------------------------------------------------------

# Feature: sports-weekly-intelligence-agent, Property 10: embed_articles returns one embedding per article
@settings(max_examples=100)
@given(st.lists(article_strategy(), min_size=0, max_size=20))
def test_embed_articles_shape(articles):
    """**Validates: Requirements 5.1**

    For any list of N articles, embed_articles must return an array of shape (N, D).
    """
    with patch("tools.embeddings.embed_text", side_effect=fixed_embedding):
        result = embed_articles(articles)

    n = len(articles)
    if n == 0:
        # numpy stacks an empty list to shape (0,) — handle gracefully
        assert result.shape[0] == 0
    else:
        assert result.shape == (n, FIXED_DIM), (
            f"Expected shape ({n}, {FIXED_DIM}), got {result.shape}"
        )
        # All rows must have the same dimension
        assert result.shape[1] == FIXED_DIM


# ---------------------------------------------------------------------------
# Property 11: FAISS index build-load round trip
# ---------------------------------------------------------------------------

def _random_embeddings(draw, min_n=1, max_n=20, dim=32):
    """Draw a random float32 embedding matrix of shape (N, dim)."""
    n = draw(st.integers(min_value=min_n, max_value=max_n))
    data = draw(
        st.lists(
            st.lists(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=dim, max_size=dim),
            min_size=n,
            max_size=n,
        )
    )
    return np.array(data, dtype=np.float32)


# Feature: sports-weekly-intelligence-agent, Property 11: FAISS index build-load round trip
@settings(max_examples=50)
@given(st.data())
def test_faiss_build_load_round_trip(data):
    """**Validates: Requirements 5.2, 5.3**

    For any set of embeddings, build_faiss_index followed by load_faiss_index must
    return an index that produces the same nearest neighbors as the original.
    """
    dim = 32
    embeddings = _random_embeddings(data.draw, min_n=1, max_n=10, dim=dim)
    n = embeddings.shape[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = os.path.join(tmpdir, "test_faiss.index")
        with patch.dict(os.environ, {"FAISS_INDEX_PATH": index_path}):
            original_index = build_faiss_index(embeddings)

            # Query with the first embedding
            query = embeddings[:1]
            k = min(n, 3)
            _, original_neighbors = original_index.search(query, k)

            loaded_index = load_faiss_index()
            _, loaded_neighbors = loaded_index.search(query, k)

        assert np.array_equal(original_neighbors, loaded_neighbors), (
            f"Neighbors differ after round-trip: original={original_neighbors}, loaded={loaded_neighbors}"
        )
