"""
Prisma-based document repository implementation.

This module provides document CRUD operations using Prisma/SQLite.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional

from src.core.domain import Document, DocumentSummary

if TYPE_CHECKING:
    from prisma import Prisma


class PrismaDocumentRepository:
    """
    DocumentRepository implementation using Prisma/SQLite.

    Implements the DocumentRepository protocol for document management.
    """

    def __init__(self, prisma: Prisma) -> None:
        """
        Initialize the document repository.

        Args:
            prisma: Connected Prisma client instance.
        """
        self._prisma = prisma

    async def create(self, content: str, metadata: Optional[dict] = None) -> Document:
        """
        Create a new document.

        Args:
            content: Full text content of the document.
            metadata: Optional metadata dictionary.

        Returns:
            The created Document entity.
        """
        metadata_str = json.dumps(metadata) if metadata else None

        doc_record = await self._prisma.document.create(
            data={
                "content": content,
                "metadata": metadata_str,
            }
        )

        return Document(
            id=doc_record.id,
            content=doc_record.content,
            metadata=metadata,
            created_at=doc_record.createdAt,
        )

    async def get_by_id(self, document_id: str) -> Optional[Document]:
        """
        Retrieve a document by its ID.

        Args:
            document_id: The document's unique identifier.

        Returns:
            The Document if found, None otherwise.
        """
        doc_record = await self._prisma.document.find_unique(
            where={"id": document_id}
        )

        if doc_record is None:
            return None

        metadata = json.loads(doc_record.metadata) if doc_record.metadata else None

        return Document(
            id=doc_record.id,
            content=doc_record.content,
            metadata=metadata,
            created_at=doc_record.createdAt,
        )

    async def delete(self, document_id: str) -> bool:
        """
        Delete a document and all its chunks.

        Args:
            document_id: The document's unique identifier.

        Returns:
            True if deleted, False if not found.
        """
        try:
            await self._prisma.document.delete(where={"id": document_id})
            return True
        except Exception:
            return False

    async def list_all(self) -> list[DocumentSummary]:
        """
        List all documents with summary information.

        Returns:
            List of DocumentSummary DTOs.
        """
        documents = await self._prisma.document.find_many(include={"chunks": True})

        return [
            DocumentSummary(
                id=doc.id,
                content_preview=(
                    doc.content[:100] + "..." if len(doc.content) > 100 else doc.content
                ),
                metadata=json.loads(doc.metadata) if doc.metadata else None,
                chunk_count=len(doc.chunks) if doc.chunks else 0,
                created_at=doc.createdAt,
            )
            for doc in documents
        ]
