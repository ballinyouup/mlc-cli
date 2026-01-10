"""
Prisma-based LaMP repository implementation.

This module provides LaMP dataset operations using Prisma/SQLite.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional

from src.core.domain import (
    LaMPSample,
    LaMPSampleSummary,
    LaMPTask,
    LaMPTaskSummary,
    ProfileDocument,
)

if TYPE_CHECKING:
    from prisma import Prisma
    from src.core.interfaces import Embedder


class PrismaLaMPRepository:
    """
    LaMPRepository implementation using Prisma/SQLite.

    Implements the LaMPRepository protocol for LaMP dataset access.
    """

    def __init__(self, prisma: Prisma, embedder: Embedder) -> None:
        """
        Initialize the LaMP repository.

        Args:
            prisma: Connected Prisma client instance.
            embedder: Embedder for similarity calculations.
        """
        self._prisma = prisma
        self._embedder = embedder

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
        where_clause: dict = {"sampleId": sample_id}

        if split:
            where_clause["split"] = split
        if variant:
            where_clause["variant"] = variant

        if task_number:
            task = await self._prisma.lamptask.find_first(
                where={"taskNumber": task_number}
            )
            if task:
                where_clause["taskId"] = task.id

        sample = await self._prisma.lampsample.find_first(
            where=where_clause,
            include={"task": True, "profileItems": True},
        )

        if not sample:
            return None

        return LaMPSample(
            id=sample.id,
            sample_id=sample.sampleId,
            input=sample.input,
            output=sample.output,
            split=sample.split,
            variant=sample.variant,
            task=LaMPTask(
                id=sample.task.id,
                task_number=sample.task.taskNumber,
                name=sample.task.name,
                description=sample.task.description,
            ),
            profile_count=len(sample.profileItems) if sample.profileItems else 0,
        )

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
        where_clause: dict = {}

        if sample_id:
            sample = await self._prisma.lampsample.find_first(
                where={"sampleId": sample_id}
            )
            if sample:
                where_clause["sampleId"] = sample.id

        profile_docs = await self._prisma.profiledocument.find_many(
            where=where_clause if where_clause else None,
            include={"sample": {"include": {"task": True}}},
        )

        scored_docs: list[tuple[float, ProfileDocument]] = []

        for doc in profile_docs:
            doc_embedding = json.loads(doc.embedding)
            similarity = self._embedder.cosine_similarity(query_embedding, doc_embedding)

            profile = ProfileDocument(
                profile_id=doc.profileId,
                title=doc.title,
                content=doc.content,
                similarity=similarity,
                sample_id=doc.sample.sampleId if doc.sample else None,
                task_name=doc.sample.task.name if doc.sample and doc.sample.task else None,
            )
            scored_docs.append((similarity, profile))

        scored_docs.sort(key=lambda x: x[0], reverse=True)

        return [doc for _, doc in scored_docs[:top_k]]

    async def list_tasks(self) -> list[LaMPTaskSummary]:
        """
        List all LaMP tasks.

        Returns:
            List of LaMPTaskSummary DTOs.
        """
        tasks = await self._prisma.lamptask.find_many(include={"samples": True})

        return [
            LaMPTaskSummary(
                id=task.id,
                number=task.taskNumber,
                name=task.name,
                description=task.description,
                sample_count=len(task.samples) if task.samples else 0,
            )
            for task in tasks
        ]

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
        where_clause: dict = {}

        if task_number:
            task = await self._prisma.lamptask.find_first(
                where={"taskNumber": task_number}
            )
            if task:
                where_clause["taskId"] = task.id

        if split:
            where_clause["split"] = split
        if variant:
            where_clause["variant"] = variant

        include_clause: dict = {"task": True}
        if include_profiles:
            include_clause["profileItems"] = True

        samples = await self._prisma.lampsample.find_many(
            where=where_clause if where_clause else None,
            include=include_clause,
            skip=offset,
            take=limit,
        )

        return [
            LaMPSampleSummary(
                id=sample.id,
                sample_id=sample.sampleId,
                input_preview=(
                    sample.input[:100] + "..."
                    if len(sample.input) > 100
                    else sample.input
                ),
                output=sample.output,
                split=sample.split,
                variant=sample.variant,
                task_name=sample.task.name,
                task_number=sample.task.taskNumber,
                profile_count=(
                    len(sample.profileItems)
                    if include_profiles and sample.profileItems
                    else 0
                ),
            )
            for sample in samples
        ]

    async def count_samples(self) -> int:
        """
        Count total number of LaMP samples.

        Returns:
            Total sample count.
        """
        return await self._prisma.lampsample.count()
