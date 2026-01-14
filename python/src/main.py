"""
Composition Root - Application Entry Point.

This module acts as the dependency injection container, wiring together
all concrete implementations and injecting them into services.

This is the ONLY place where concrete implementations are instantiated.
All other modules depend only on interfaces.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys

from src.core.domain import ChatMessage, ChatRole, LaMPContext, LaMPContextError
from src.infrastructure.database import PrismaConnection
from src.infrastructure.document_repository import PrismaDocumentRepository
from src.infrastructure.embedder import LocalEmbedder
from src.infrastructure.lamp_repository import PrismaLaMPRepository
from src.infrastructure.llm_provider import MLCLLMProvider
from src.infrastructure.vector_store import PrismaVectorStore
from src.services.document_service import DocumentService
from src.services.lamp_service import LaMPService
from src.services.rag_service import RAGService

os.environ["MLC_JIT_POLICY"] = "REDO"


class Application:
    """
    Main application class that orchestrates all services.

    This class is responsible for:
    - Initializing infrastructure components
    - Wiring dependencies via constructor injection
    - Providing high-level operations that combine services
    """

    def __init__(
        self,
        rag_service: RAGService,
        lamp_service: LaMPService,
        document_service: DocumentService,
        llm_provider: MLCLLMProvider | None = None,
        model: str = "",
    ) -> None:
        """
        Initialize the application with injected services.

        Args:
            rag_service: Service for RAG operations.
            lamp_service: Service for LaMP personalization.
            document_service: Service for document management.
            llm_provider: Optional LLM provider for inference.
            model: Model identifier for LLM.
        """
        self.rag_service = rag_service
        self.lamp_service = lamp_service
        self.document_service = document_service
        self.llm_provider = llm_provider
        self.model = model

    async def run_generic_rag(self, query: str) -> None:
        """
        Run generic RAG query with LLM generation.

        Args:
            query: User's question.
        """
        print("Retrieving relevant context...")
        context = await self.rag_service.build_context(query, top_k=3)

        if context:
            enhanced_prompt = (
                f"{context}\n\nUser Question: {query}\n\n"
                "Please answer based on the context provided above."
            )
            print(f"\n--- RAG Context Retrieved ---\n{context}\n--- End Context ---\n")
        else:
            enhanced_prompt = query
            print("No relevant context found in database.\n")

        if self.llm_provider is None:
            print(f"Context:\n{context}")
            return

        print("Generating response...\n")
        messages = [ChatMessage(role=ChatRole.USER, content=enhanced_prompt)]

        async for chunk in self.llm_provider.generate(messages, self.model, stream=True):
            print(chunk, end="", flush=True)
        print("\n")

    async def run_lamp_personalization(
        self,
        sample_id: str,
        task_number: int | None = None,
        split: str = "validation",
        variant: str = "user_based",
        top_k: int = 5,
    ) -> None:
        """
        Run LaMP personalization task.

        Args:
            sample_id: LaMP sample ID to process.
            task_number: Optional task number filter.
            split: Data split (train/validation/test).
            variant: Variant (user_based/time_based).
            top_k: Number of profile docs to retrieve.
        """
        print(f"Loading LaMP sample: {sample_id}")

        result = await self.lamp_service.build_lamp_context(
            sample_id=sample_id,
            task_number=task_number,
            top_k=top_k,
            split=split,
            variant=variant,
        )

        if isinstance(result, LaMPContextError):
            print(f"Error: {result.error}")
            return

        lamp_context: LaMPContext = result
        sample = lamp_context.sample

        print(f"\n--- LaMP Task: {sample.task.name} ---")
        print(f"Sample ID: {sample.sample_id}")
        print(f"Split: {sample.split}, Variant: {sample.variant}")
        print(f"Profile items retrieved: {lamp_context.retrieved_profiles}")

        print(
            f"\n--- User Profile Context ---\n{lamp_context.profile_context}\n"
            "--- End Profile ---\n"
        )
        print(f"--- Input Task ---\n{lamp_context.input}\n--- End Input ---\n")

        if lamp_context.expected_output:
            print(
                f"--- Expected Output ---\n{lamp_context.expected_output}\n"
                "--- End Expected ---\n"
            )

        if self.llm_provider is None:
            return

        system_message = self.lamp_service.build_system_message(
            task_name=sample.task.name,
            profile_context=lamp_context.profile_context,
        )

        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content=system_message),
            ChatMessage(role=ChatRole.USER, content=lamp_context.input),
        ]

        print("Generating personalized response...\n")
        async for chunk in self.llm_provider.generate(messages, self.model, stream=True):
            print(chunk, end="", flush=True)
        print("\n")

    async def list_lamp_data(
        self,
        task_number: int | None = None,
        limit: int = 10,
    ) -> None:
        """
        List available LaMP tasks and samples.

        Args:
            task_number: Optional task number filter.
            limit: Maximum samples to display.
        """
        print("\n=== LaMP Tasks ===")
        tasks = await self.lamp_service.list_tasks()

        if not tasks:
            print("No LaMP tasks loaded. Run: python add_documents.py --lamp")
            return

        for task in tasks:
            print(f"  LaMP-{task.number}: {task.name} ({task.sample_count} samples)")

        print(f"\n=== Sample Data (limit: {limit}) ===")
        samples = await self.lamp_service.list_samples(
            task_number=task_number,
            limit=limit,
        )

        for sample in samples:
            print(f"\n  [{sample.task_name}] ID: {sample.sample_id}")
            print(f"    Split: {sample.split}, Variant: {sample.variant}")
            print(f"    Profile docs: {sample.profile_count}")
            print(f"    Input: {sample.input_preview}")
            if sample.output:
                print(f"    Output: {sample.output[:80]}...")

    async def get_random_sample(self) -> dict | None:
        """
        Get a random LaMP sample.

        Returns:
            Sample info dict or None if no samples exist.
        """
        sample_count = await self.lamp_service.count_samples()
        if sample_count == 0:
            return None

        random_offset = random.randint(0, sample_count - 1)
        samples = await self.lamp_service.list_samples(
            limit=1,
            offset=random_offset,
            include_profiles=False,
        )

        if not samples:
            return None

        sample = samples[0]
        return {
            "sample_id": sample.sample_id,
            "task_number": sample.task_number,
            "task_name": sample.task_name,
            "split": sample.split,
            "variant": sample.variant,
        }


async def create_application(
    no_llm: bool = False,
    model_path: str = "../mlc-llm/models/Qwen3-1.7B-q4f16_1-MLC",
) -> tuple[Application, PrismaConnection, MLCLLMProvider | None]:
    """
    Factory function to create and wire the application.

    This is the composition root where all dependencies are assembled.

    Args:
        no_llm: Skip LLM initialization if True.
        model_path: Path to the MLC-LLM model.

    Returns:
        Tuple of (Application, PrismaConnection, optional MLCLLMProvider).
    """
    db_connection = PrismaConnection()
    prisma = await db_connection.connect()

    embedder = LocalEmbedder()

    vector_store = PrismaVectorStore(prisma, embedder)
    document_repo = PrismaDocumentRepository(prisma)
    lamp_repo = PrismaLaMPRepository(prisma, embedder)

    rag_service = RAGService(vector_store, embedder)
    lamp_service = LaMPService(lamp_repo, embedder)
    document_service = DocumentService(document_repo, vector_store, embedder)

    llm_provider: MLCLLMProvider | None = None
    if not no_llm:
        from mlc_llm import MLCEngine

        engine = MLCEngine(model_path)
        llm_provider = MLCLLMProvider(engine)

    app = Application(
        rag_service=rag_service,
        lamp_service=lamp_service,
        document_service=document_service,
        llm_provider=llm_provider,
        model=model_path,
    )

    return app, db_connection, llm_provider


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RAG System with LaMP Personalization")
    parser.add_argument("--lamp", action="store_true", help="Run LaMP personalization mode")
    parser.add_argument("--sample-id", type=str, help="LaMP sample ID to process")
    parser.add_argument("--task", type=int, help="LaMP task number (1-7)")
    parser.add_argument(
        "--split", type=str, default="validation", help="Data split (train/validation/test)"
    )
    parser.add_argument(
        "--variant", type=str, default="user_based", help="Variant (user_based/time_based)"
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of profile docs to retrieve")
    parser.add_argument("--query", type=str, help="Custom query for generic RAG")
    parser.add_argument("--list", action="store_true", help="List available LaMP data")
    parser.add_argument("--random", action="store_true", help="Select random LaMP task and sample")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM inference (for testing)")

    args = parser.parse_args()

    print("Initializing RAG system...")
    app, db_connection, llm_provider = await create_application(no_llm=args.no_llm)

    try:
        if args.list:
            await app.list_lamp_data(task_number=args.task)
            return

        if args.random:
            random_sample = await app.get_random_sample()
            if random_sample is None:
                print("No LaMP samples found in database. Run: python add_documents.py --lamp")
                return

            args.lamp = True
            args.sample_id = random_sample["sample_id"]
            args.task = random_sample["task_number"]
            args.split = random_sample["split"]
            args.variant = random_sample["variant"]

            print(f"\n🎲 Randomly selected: {random_sample['task_name']} (Task {args.task})")
            print(
                f"   Sample ID: {args.sample_id}, Split: {args.split}, "
                f"Variant: {args.variant}\n"
            )

        if args.lamp and args.sample_id:
            await app.run_lamp_personalization(
                sample_id=args.sample_id,
                task_number=args.task,
                split=args.split,
                variant=args.variant,
                top_k=args.top_k,
            )
        elif args.query:
            await app.run_generic_rag(args.query)
        else:
            default_query = "What is the meaning of life?"
            await app.run_generic_rag(default_query)

    finally:
        if llm_provider:
            llm_provider.terminate()
        await db_connection.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
