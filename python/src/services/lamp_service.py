"""
LaMP personalization service.

This module provides business logic for LaMP benchmark tasks
including profile retrieval and context building.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from src.core.domain import (
    LaMPContext,
    LaMPContextError,
    LaMPSampleSummary,
    LaMPTaskSummary,
    ProfileDocument,
)

if TYPE_CHECKING:
    from src.core.interfaces import Embedder, LaMPRepository


class LaMPService:
    """
    Service for LaMP personalization operations.

    Handles LaMP task management, profile retrieval, and
    personalized context building. All dependencies are injected.
    """

    TASK_SYSTEM_PROMPTS: dict[str, str] = {
        "Citation_Identification": (
            "You are an expert academic citation analyzer. Analyze the researcher's "
            "publication history to determine which reference is most relevant to their "
            "research domain. Respond with ONLY the reference number (e.g., [1] or [2]) "
            "without any explanation."
        ),
        "Movie_Tagging": (
            "You are a movie recommendation expert. Based on the user's movie watching "
            "history and tag preferences, predict the most appropriate tag for the given "
            "movie. Respond with ONLY the tag name from the provided list, without any "
            "explanation."
        ),
        "Product_Rating": (
            "You are a product review analyst. Based on the user's review history and "
            "rating patterns, predict their rating for the given product. Respond with "
            "ONLY the numerical rating without any explanation."
        ),
        "News_Headline_Generation": (
            "You are a news headline writer. Based on the user's article writing style "
            "and topics, generate an appropriate headline for the given article. Respond "
            "with ONLY the headline text without any explanation."
        ),
        "Scholarly_Title_Generation": (
            "You are an academic title generator. Based on the user's academic writing "
            "style and research focus, generate an appropriate scholarly title. Respond "
            "with ONLY the title text without any explanation."
        ),
        "Tweet_Paraphrasing": (
            "You are a social media writing expert. Based on the user's tweet writing "
            "style and language patterns, paraphrase the given tweet in their style. "
            "Respond with ONLY the paraphrased tweet without any explanation."
        ),
    }

    DEFAULT_SYSTEM_PROMPT: str = (
        "You are a helpful assistant. Analyze the user's historical data and complete "
        "the task. Respond with only the answer without explanation."
    )

    def __init__(
        self,
        lamp_repo: LaMPRepository,
        embedder: Embedder,
    ) -> None:
        """
        Initialize the LaMP service.

        Args:
            lamp_repo: Repository for LaMP data access.
            embedder: Embedder for query vectorization.
        """
        self._lamp_repo = lamp_repo
        self._embedder = embedder

    async def retrieve_profile_documents(
        self,
        query: str,
        sample_id: str | None = None,
        top_k: int = 5,
    ) -> list[ProfileDocument]:
        """
        Retrieve most relevant profile documents for a query.

        Args:
            query: The search query text.
            sample_id: Optional sample ID to scope the search.
            top_k: Maximum number of results to return.

        Returns:
            List of ProfileDocument DTOs ordered by relevance.
        """
        query_embedding = self._embedder.encode(query)
        return await self._lamp_repo.search_profile_documents(
            query_embedding=query_embedding,
            sample_id=sample_id,
            top_k=top_k,
        )

    async def build_lamp_context(
        self,
        sample_id: str,
        task_number: int | None = None,
        top_k: int = 5,
        split: str | None = None,
        variant: str | None = None,
    ) -> Union[LaMPContext, LaMPContextError]:
        """
        Build personalized context for a LaMP sample.

        Args:
            sample_id: The original sample ID from the dataset.
            task_number: Optional task number filter.
            top_k: Number of profile documents to retrieve.
            split: Optional split filter.
            variant: Optional variant filter.

        Returns:
            LaMPContext on success, LaMPContextError on failure.
        """
        sample = await self._lamp_repo.get_sample(
            sample_id=sample_id,
            task_number=task_number,
            split=split,
            variant=variant,
        )

        if not sample:
            return LaMPContextError(error=f"Sample {sample_id} not found")

        relevant_profiles = await self.retrieve_profile_documents(
            query=sample.input,
            sample_id=sample_id,
            top_k=top_k,
        )

        profile_context = self._format_profile_context(relevant_profiles)

        return LaMPContext(
            sample=sample,
            input=sample.input,
            profile_context=profile_context,
            expected_output=sample.output,
            retrieved_profiles=len(relevant_profiles),
        )

    def _format_profile_context(self, profiles: list[ProfileDocument]) -> str:
        """
        Format profile documents into a context string.

        Args:
            profiles: List of profile documents.

        Returns:
            Formatted context string.
        """
        if not profiles:
            return ""

        context_parts = ["User's historical data:"]
        for i, profile in enumerate(profiles, 1):
            context_parts.append(
                f"\n[Profile {i}] (Relevance: {profile.similarity:.3f})"
            )
            if profile.title:
                context_parts.append(f"Title: {profile.title}")
            context_parts.append(profile.content)

        return "\n".join(context_parts)

    def get_system_prompt(self, task_name: str) -> str:
        """
        Get the system prompt for a specific LaMP task.

        Args:
            task_name: Name of the LaMP task.

        Returns:
            Task-specific system prompt.
        """
        return self.TASK_SYSTEM_PROMPTS.get(task_name, self.DEFAULT_SYSTEM_PROMPT)

    def build_system_message(self, task_name: str, profile_context: str) -> str:
        """
        Build the full system message for LLM inference.

        Args:
            task_name: Name of the LaMP task.
            profile_context: Formatted profile context.

        Returns:
            Complete system message with instructions and context.
        """
        system_prompt = self.get_system_prompt(task_name)
        return f"""{system_prompt}

User's Historical Data:
{profile_context}

Analyze the patterns, preferences, and style from the user's history above to personalize your response."""

    async def list_tasks(self) -> list[LaMPTaskSummary]:
        """
        List all LaMP tasks.

        Returns:
            List of LaMPTaskSummary DTOs.
        """
        return await self._lamp_repo.list_tasks()

    async def list_samples(
        self,
        task_number: int | None = None,
        split: str | None = None,
        variant: str | None = None,
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
        return await self._lamp_repo.list_samples(
            task_number=task_number,
            split=split,
            variant=variant,
            limit=limit,
            offset=offset,
            include_profiles=include_profiles,
        )

    async def count_samples(self) -> int:
        """
        Count total number of LaMP samples.

        Returns:
            Total sample count.
        """
        return await self._lamp_repo.count_samples()
