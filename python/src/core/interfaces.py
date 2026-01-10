"""
Protocol definitions (Interfaces).

This module defines the contracts that infrastructure implementations must fulfill.
Business logic depends on these abstractions, not concrete implementations.

Analogous to Java interfaces or Go interfaces.
"""

from __future__ import annotations

from typing import AsyncIterator, Optional, Protocol, runtime_checkable

from src.core.domain import (
    ChatMessage,
    Chunk,
    Document,
    DocumentSummary,
    LaMPSample,
    LaMPTaskSummary,
    LaMPSampleSummary,
    ProfileDocument,
    SearchResult,
)


@runtime_checkable
class Embedder(Protocol):
    """
    Interface for text embedding generation.

    Implementations may use SentenceTransformers, OpenAI, Cohere, etc.
    The service layer depends only on this interface, allowing easy swapping
    of embedding providers without touching business logic.
    """

    @property
    def dimension(self) -> int:
        """Return the embedding dimension size."""
        ...

    def encode(self, text: str) -> list[float]:
        """
        Generate embedding vector for a single text.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        ...

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embedding vectors for multiple texts.

        Args:
            texts: List of input texts to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        ...

    def cosine_similarity(self, emb1: list[float], emb2: list[float]) -> float:
        """
        Calculate cosine similarity between two embedding vectors.

        Args:
            emb1: First embedding vector.
            emb2: Second embedding vector.

        Returns:
            Cosine similarity score between -1 and 1.
        """
        ...


@runtime_checkable
class VectorStore(Protocol):
    """
    Interface for vector storage and similarity search.

    Implementations may use Prisma/SQLite, Qdrant, Pinecone, etc.
    All database-specific logic is encapsulated here.
    """

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
        ...

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
        ...


@runtime_checkable
class DocumentRepository(Protocol):
    """
    Interface for document CRUD operations.

    Separates document management from vector search concerns.
    """

    async def create(self, content: str, metadata: Optional[dict] = None) -> Document:
        """
        Create a new document.

        Args:
            content: Full text content of the document.
            metadata: Optional metadata dictionary.

        Returns:
            The created Document entity.
        """
        ...

    async def get_by_id(self, document_id: str) -> Optional[Document]:
        """
        Retrieve a document by its ID.

        Args:
            document_id: The document's unique identifier.

        Returns:
            The Document if found, None otherwise.
        """
        ...

    async def delete(self, document_id: str) -> bool:
        """
        Delete a document and all its chunks.

        Args:
            document_id: The document's unique identifier.

        Returns:
            True if deleted, False if not found.
        """
        ...

    async def list_all(self) -> list[DocumentSummary]:
        """
        List all documents with summary information.

        Returns:
            List of DocumentSummary DTOs.
        """
        ...


@runtime_checkable
class LaMPRepository(Protocol):
    """
    Interface for LaMP dataset operations.

    Handles all LaMP-specific data access including tasks, samples, and profiles.
    """

    async def get_sample(
        self,
        sample_id: str,
        task_number: Optional[int] = None,
        split: Optional[str] = None,
        variant: Optional[str] = None,
    ) -> Optional[LaMPSample]:
        """
        Retrieve a LaMP sample by its original ID.

        Args:
            sample_id: The original sample ID from the dataset.
            task_number: Optional task number filter (1-7).
            split: Optional split filter (train/validation/test).
            variant: Optional variant filter (user_based/time_based).

        Returns:
            The LaMPSample if found, None otherwise.
        """
        ...

    async def search_profile_documents(
        self,
        query_embedding: list[float],
        sample_id: Optional[str] = None,
        top_k: int = 5,
    ) -> list[ProfileDocument]:
        """
        Search profile documents by embedding similarity.

        Args:
            query_embedding: The query vector to search against.
            sample_id: Optional sample ID to scope the search.
            top_k: Maximum number of results to return.

        Returns:
            List of ProfileDocument DTOs ordered by similarity.
        """
        ...

    async def list_tasks(self) -> list[LaMPTaskSummary]:
        """
        List all LaMP tasks.

        Returns:
            List of LaMPTaskSummary DTOs.
        """
        ...

    async def list_samples(
        self,
        task_number: Optional[int] = None,
        split: Optional[str] = None,
        variant: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
        include_profiles: bool = True,
    ) -> list[LaMPSampleSummary]:
        """
        List LaMP samples with optional filters.

        Args:
            task_number: Optional task number filter.
            split: Optional split filter.
            variant: Optional variant filter.
            limit: Maximum number of results.
            offset: Number of results to skip.
            include_profiles: Whether to include profile counts.

        Returns:
            List of LaMPSampleSummary DTOs.
        """
        ...

    async def count_samples(self) -> int:
        """
        Count total number of LaMP samples.

        Returns:
            Total sample count.
        """
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """
    Interface for Large Language Model inference.

    Implementations may use MLC-LLM, OpenAI, Anthropic, etc.
    Supports both streaming and non-streaming generation.
    """

    def generate(
        self,
        messages: list[ChatMessage],
        model: str,
        stream: bool = True,
    ) -> AsyncIterator[str]:
        """
        Generate a response from the LLM.

        Args:
            messages: List of chat messages forming the conversation.
            model: Model identifier to use.
            stream: Whether to stream the response.

        Yields:
            Response text chunks if streaming, or full response if not.
        """
        ...

    def terminate(self) -> None:
        """
        Clean up LLM resources.

        Should be called when done using the provider.
        """
        ...
