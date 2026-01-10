"""
Document management service.

This module provides business logic for document operations including
chunking and embedding generation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.domain import Document, DocumentSummary

if TYPE_CHECKING:
    from src.core.interfaces import DocumentRepository, Embedder, VectorStore


class DocumentService:
    """
    Service for document management operations.

    Handles document creation with chunking and embedding,
    deletion, and listing. All dependencies are injected.
    """

    def __init__(
        self,
        document_repo: DocumentRepository,
        vector_store: VectorStore,
        embedder: Embedder,
    ) -> None:
        """
        Initialize the document service.

        Args:
            document_repo: Repository for document CRUD operations.
            vector_store: Store for chunk embeddings.
            embedder: Embedder for generating vectors.
        """
        self._document_repo = document_repo
        self._vector_store = vector_store
        self._embedder = embedder

    async def add_document(
        self,
        content: str,
        metadata: dict | None = None,
        chunk_size: int = 500,
    ) -> Document:
        """
        Add a document to the database with chunking and embeddings.

        Args:
            content: Full text content of the document.
            metadata: Optional metadata dictionary.
            chunk_size: Target size for text chunks.

        Returns:
            The created Document entity.
        """
        document = await self._document_repo.create(content, metadata)

        chunks = self._split_text(content, chunk_size)

        for position, chunk_text in enumerate(chunks):
            embedding = self._embedder.encode(chunk_text)
            await self._vector_store.store_chunk(
                document_id=document.id,
                content=chunk_text,
                embedding=embedding,
                position=position,
            )

        return document

    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all its chunks.

        Args:
            document_id: The document's unique identifier.

        Returns:
            True if deleted, False if not found.
        """
        return await self._document_repo.delete(document_id)

    async def list_documents(self) -> list[DocumentSummary]:
        """
        List all documents with summary information.

        Returns:
            List of DocumentSummary DTOs.
        """
        return await self._document_repo.list_all()

    def _split_text(self, text: str, chunk_size: int = 500) -> list[str]:
        """
        Split text into chunks of approximately chunk_size characters.

        Args:
            text: Text to split.
            chunk_size: Target chunk size in characters.

        Returns:
            List of text chunks.
        """
        words = text.split()
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1
            if current_length + word_length > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
