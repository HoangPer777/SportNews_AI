import os

import faiss
import numpy as np
from google import genai
from google.genai import types

FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "data/faiss.index")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "models/gemini-embedding-001")


def _get_client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")
    return genai.Client(api_key=api_key)


def embed_text(text: str) -> list[float]:
    """Embed a single text string using the Gemini embedding model."""
    client = _get_client()
    model = os.environ.get("EMBEDDING_MODEL", EMBEDDING_MODEL)
    result = client.models.embed_content(
        model=model,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return result.embeddings[0].values


def embed_articles(articles) -> np.ndarray:
    """Embed a list of articles, returning array of shape (N, D)."""
    embeddings = []
    for article in articles:
        text = article.title + " " + article.content
        vec = embed_text(text)
        embeddings.append(vec)
    return np.array(embeddings, dtype=np.float32)


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
    vec = embed_text(query)
    return np.array([vec], dtype=np.float32)
