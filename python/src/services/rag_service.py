"""
RAG (Retrieval-Augmented Generation) service.

This module provides the core RAG orchestration logic for building
context from relevant documents and generating responses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.domain import SearchResult

if TYPE_CHECKING:
    from src.core.interfaces import Embedder, VectorStore


class RAGService:
    """
    Service for RAG operations.

    Orchestrates retrieval and context building for RAG queries.
    All dependencies are injected via constructor.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
    ) -> None:
        """
        Initialize the RAG service.

        Args:
            vector_store: Store for vector similarity search.
            embedder: Embedder for query vectorization.
        """
        self._vector_store = vector_store
        self._embedder = embedder

    async def retrieve_relevant_chunks(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query: The search query text.
            top_k: Maximum number of results to return.

        Returns:
            List of SearchResult DTOs ordered by relevance.
        """
        query_embedding = self._embedder.encode(query)
        return await self._vector_store.search_similar(query_embedding, top_k)

    async def build_context(
        self,
        query: str,
        top_k: int = 5,
    ) -> str:
        """
        Build context string from relevant chunks for RAG.

        Args:
            query: The search query text.
            top_k: Maximum number of chunks to include.

        Returns:
            Formatted context string, empty if no results.
        """
        relevant_chunks = await self.retrieve_relevant_chunks(query, top_k)

        if not relevant_chunks:
            return ""

        context_parts = ["Retrieved context:"]
        for i, chunk in enumerate(relevant_chunks, 1):
            context_parts.append(
                f"\n[Context {i}] (Relevance: {chunk.similarity:.3f})"
            )
            context_parts.append(chunk.content)

        return "\n".join(context_parts)
