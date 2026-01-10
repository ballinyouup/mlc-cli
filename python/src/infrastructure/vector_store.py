"""
Prisma-based vector store implementation.

This module provides vector storage and similarity search using Prisma/SQLite.
All database-specific logic is encapsulated here.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from src.core.domain import Chunk, SearchResult

if TYPE_CHECKING:
    from prisma import Prisma
    from src.core.interfaces import Embedder


class PrismaVectorStore:
    """
    VectorStore implementation using Prisma/SQLite.

    Implements the VectorStore protocol for chunk storage and similarity search.
    The cosine similarity calculation is done in-memory since SQLite doesn't
    have native vector operations.
    """

    def __init__(self, prisma: Prisma, embedder: Embedder) -> None:
        """
        Initialize the vector store.

        Args:
            prisma: Connected Prisma client instance.
            embedder: Embedder for similarity calculations.
        """
        self._prisma = prisma
        self._embedder = embedder

    async def store_chunk(
        self,
        document_id: str,
        content: str,
        embedding: list[float],
        position: int,
    ) -> Chunk:
        """
        Store a chunk with its embedding.

        Args:
            document_id: ID of the parent document.
            content: Text content of the chunk.
            embedding: Vector embedding of the content.
            position: Position of chunk within the document.

        Returns:
            The created Chunk entity.
        """
        embedding_str = json.dumps(embedding)

        chunk_record = await self._prisma.chunk.create(
            data={
                "documentId": document_id,
                "content": content,
                "embedding": embedding_str,
                "position": position,
            }
        )

        return Chunk(
            id=chunk_record.id,
            document_id=chunk_record.documentId,
            content=chunk_record.content,
            embedding=embedding,
            position=chunk_record.position,
            created_at=chunk_record.createdAt,
        )

    async def search_similar(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Find chunks most similar to the query embedding.

        Args:
            query_embedding: The query vector to search against.
            top_k: Maximum number of results to return.

        Returns:
            List of SearchResult DTOs ordered by similarity (descending).
        """
        all_chunks = await self._prisma.chunk.find_many(include={"document": True})

        scored_results: list[tuple[float, SearchResult]] = []

        for chunk in all_chunks:
            chunk_embedding = json.loads(chunk.embedding)
            similarity = self._embedder.cosine_similarity(query_embedding, chunk_embedding)

            metadata = None
            if chunk.document and chunk.document.metadata:
                metadata = json.loads(chunk.document.metadata)

            result = SearchResult(
                chunk_id=chunk.id,
                document_id=chunk.documentId,
                content=chunk.content,
                position=chunk.position,
                similarity=similarity,
                metadata=metadata,
            )
            scored_results.append((similarity, result))

        scored_results.sort(key=lambda x: x[0], reverse=True)

        return [result for _, result in scored_results[:top_k]]
