"""
Embedding engine for the RAG pipeline.
Uses sentence-transformers (all-MiniLM-L6-v2) for fast, high-quality semantic embeddings.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List


class EmbeddingEngine:
    """Manages embedding generation and semantic similarity search."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding engine.

        Args:
            model_name: HuggingFace model name for sentence embeddings.
        """
        self.model_name = model_name
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the model to avoid loading at import time."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        Encode a list of texts into dense vector embeddings.

        Args:
            texts: List of text strings to encode.
            normalize: Whether to L2-normalize embeddings (recommended for cosine similarity).

        Returns:
            Numpy array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.array([])

        # Sanitize: the Rust tokenizer crashes on surrogates & control chars
        # common in PDF-extracted text
        clean_texts = []
        for t in texts:
            if t is None:
                clean_texts.append("empty")
                continue
            s = str(t)
            # Remove surrogates that crash the Rust tokenizer
            s = s.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
            # Strip null bytes and control characters (keep newlines/tabs)
            s = "".join(ch if ch in ("\n", "\t") or (ord(ch) >= 32) else " " for ch in s)
            s = s.strip()
            clean_texts.append(s if len(s) > 0 else "empty")

        embeddings = self.model.encode(
            clean_texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=32,
        )

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            embeddings = embeddings / norms

        return embeddings

    def compute_similarity(
        self, query_embedding: np.ndarray, doc_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between a query and document embeddings.

        Args:
            query_embedding: 1D or 2D array of the query embedding.
            doc_embeddings: 2D array of document embeddings.

        Returns:
            1D array of similarity scores.
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Cosine similarity (assumes normalized vectors → dot product)
        similarities = np.dot(doc_embeddings, query_embedding.T).flatten()
        return similarities
