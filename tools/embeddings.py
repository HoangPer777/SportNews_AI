"""Embedding and FAISS index utilities for the Sports Weekly Intelligence Agent."""

from __future__ import annotations

import logging
import os
import time

import faiss
import numpy as np
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "data/faiss.index")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "models/gemini-embedding-001")

# Retry config
_MAX_RETRIES = 5
_INITIAL_BACKOFF = 10  # seconds


def _get_client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")
    return genai.Client(api_key=api_key)


def _embed_batch(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Embed a batch of texts in a single API call with exponential backoff on 429."""
    client = _get_client()
    model = os.environ.get("EMBEDDING_MODEL", EMBEDDING_MODEL)

    backoff = _INITIAL_BACKOFF
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            result = client.models.embed_content(
                model=model,
                contents=texts,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return [e.values for e in result.embeddings]
        except Exception as exc:
            err_str = str(exc)
            is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
            if is_rate_limit and attempt < _MAX_RETRIES:
                logger.warning(
                    "Rate limit hit (attempt %d/%d). Waiting %ds before retry...",
                    attempt, _MAX_RETRIES, backoff,
                )
                time.sleep(backoff)
                backoff *= 2  # exponential backoff
            else:
                raise

    raise RuntimeError("Exceeded max retries for embedding API call.")


def embed_text(text: str) -> list[float]:
    """Embed a single text string using the Gemini embedding model."""
    return _embed_batch([text], task_type="RETRIEVAL_DOCUMENT")[0]


def embed_articles(articles, batch_size: int = 5) -> np.ndarray:
    """Embed a list of articles in batches, returning array of shape (N, D)."""
    texts = [a.title + " " + a.content for a in articles]
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        logger.info("Embedding batch %d-%d of %d articles...", i + 1, i + len(batch), len(texts))
        vecs = _embed_batch(batch, task_type="RETRIEVAL_DOCUMENT")
        all_embeddings.extend(vecs)
        # Small pause between batches to stay within rate limits
        if i + batch_size < len(texts):
            time.sleep(2)

    return np.array(all_embeddings, dtype=np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a FAISS IndexFlatL2 and save to FAISS_INDEX_PATH."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))
    index_path = os.environ.get("FAISS_INDEX_PATH", FAISS_INDEX_PATH)
    os.makedirs(os.path.dirname(index_path) if os.path.dirname(index_path) else ".", exist_ok=True)
    faiss.write_index(index, index_path)
    return index


def load_faiss_index() -> faiss.Index:
    """Load FAISS index from disk. Raises FileNotFoundError if absent."""
    index_path = os.environ.get("FAISS_INDEX_PATH", FAISS_INDEX_PATH)
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index file not found at path: {index_path}")
    return faiss.read_index(index_path)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string, return shape (1, D)."""
    vec = _embed_batch([query], task_type="RETRIEVAL_QUERY")[0]
    return np.array([vec], dtype=np.float32)
