"""
Local embedding implementation using SentenceTransformers.

This module provides the concrete Embedder implementation using
the SentenceTransformers library for local embedding generation.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class LocalEmbedder:
    """
    Embedder implementation using SentenceTransformers.

    This class implements the Embedder protocol for local embedding generation.
    It can be swapped with OpenAI, Cohere, or other providers without changing
    the service layer.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialize the embedding model.

        Args:
            model_name: HuggingFace model identifier for SentenceTransformers.
        """
        self._model = SentenceTransformer(model_name)
        self._dimension: int = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        """Return the embedding dimension size."""
        return self._dimension

    def encode(self, text: str) -> list[float]:
        """
        Generate embedding vector for a single text.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embedding vectors for multiple texts.

        Args:
            texts: List of input texts to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def cosine_similarity(self, emb1: list[float], emb2: list[float]) -> float:
        """
        Calculate cosine similarity between two embedding vectors.

        Args:
            emb1: First embedding vector.
            emb2: Second embedding vector.

        Returns:
            Cosine similarity score between -1 and 1.
        """
        emb1_np = np.array(emb1)
        emb2_np = np.array(emb2)

        dot_product = np.dot(emb1_np, emb2_np)
        norm1 = np.linalg.norm(emb1_np)
        norm2 = np.linalg.norm(emb2_np)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))
