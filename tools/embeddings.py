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
    """Embed a batch of texts, one API call per text to ensure correct results."""
    client = _get_client()
    model = os.environ.get("EMBEDDING_MODEL", EMBEDDING_MODEL)

    results = []
    for text in texts:
        backoff = _INITIAL_BACKOFF
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                result = client.models.embed_content(
                    model=model,
                    contents=text,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                results.append(result.embeddings[0].values)
                break
            except Exception as exc:
                err_str = str(exc)
                is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
                if is_rate_limit and attempt < _MAX_RETRIES:
                    logger.warning(
                        "Rate limit hit (attempt %d/%d). Waiting %ds before retry...",
                        attempt, _MAX_RETRIES, backoff,
                    )
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    raise
    return results


def embed_text(text: str) -> list[float]:
    """Embed a single text string using the Gemini embedding model."""
    return _embed_batch([text], task_type="RETRIEVAL_DOCUMENT")[0]


def embed_articles(articles, batch_size: int = 5) -> np.ndarray:
    """Embed articles, using cached embeddings from DB where available."""
    from tools.db import save_embeddings

    result_map: dict[int, list[float]] = {}
    to_embed: list[tuple[int, str, str]] = []

    for i, article in enumerate(articles):
        emb = article.embedding
        if emb and isinstance(emb, list) and len(emb) > 10:  # valid embedding has many dimensions
            result_map[i] = emb
        else:
            to_embed.append((i, article.url, article.title + " " + article.content))

    logger.info(
        "Embedding: %d cached, %d need API calls (%d batches of %d)",
        len(result_map), len(to_embed),
        (len(to_embed) + batch_size - 1) // batch_size if to_embed else 0,
        batch_size,
    )

    new_url_embeddings: list[tuple[str, list[float]]] = []

    for batch_start in range(0, len(to_embed), batch_size):
        batch = to_embed[batch_start: batch_start + batch_size]
        texts = [t for _, _, t in batch]
        logger.info(
            "Embedding batch %d-%d of %d new articles...",
            batch_start + 1, batch_start + len(batch), len(to_embed),
        )
        try:
            vecs = _embed_batch(texts, task_type="RETRIEVAL_DOCUMENT")
            if len(vecs) != len(batch):
                logger.warning("API returned %d vectors for %d texts in batch.", len(vecs), len(batch))
            for (orig_idx, url, _), vec in zip(batch, vecs):
                result_map[orig_idx] = vec
                new_url_embeddings.append((url, vec))
            logger.debug("Batch done: %d/%d articles embedded so far.", len(new_url_embeddings), len(to_embed))
        except Exception as exc:
            logger.error("Batch %d-%d failed: %s. Using zero vectors.", batch_start + 1, batch_start + len(batch), exc)
            # Determine fallback dimension from any existing embedding
            fallback_dim = len(next(iter(result_map.values()))) if result_map else 768
            for (orig_idx, url, _) in batch:
                result_map[orig_idx] = [0.0] * fallback_dim

    if new_url_embeddings:
        try:
            save_embeddings(new_url_embeddings)
            logger.info("Saved %d new embeddings to DB.", len(new_url_embeddings))
        except Exception as exc:
            logger.warning("Failed to save embeddings to DB: %s", exc)

    # Safety check: fill any missing indices with zeros
    if len(result_map) < len(articles):
        fallback_dim = len(next(iter(result_map.values()))) if result_map else 768
        for i in range(len(articles)):
            if i not in result_map:
                logger.warning("Missing embedding for article index %d, using zero vector.", i)
                result_map[i] = [0.0] * fallback_dim

    ordered = [result_map[i] for i in range(len(articles))]
    return np.array(ordered, dtype=np.float32)


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
