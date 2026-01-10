"""
Document ingestion script for RAG and LaMP datasets.

This script provides utilities to add documents and LaMP benchmark data
to the database using the new SOA architecture.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from src.infrastructure.database import PrismaConnection
from src.infrastructure.document_repository import PrismaDocumentRepository
from src.infrastructure.embedder import LocalEmbedder
from src.infrastructure.vector_store import PrismaVectorStore
from src.services.document_service import DocumentService

async def add_lamp_dataset(prisma_client, limit: int | None = None) -> None:
    """
    Add LaMP dataset to the database.

    Args:
        prisma_client: Connected Prisma client.
        limit: Optional limit on number of samples to load per split/variant.
    """
    if limit:
        print(f"Loading LaMP dataset (limit: {limit} samples per split/variant)...")
    else:
        print("Loading LaMP dataset (no limit)...")

    lamp_dir = Path(__file__).parent.parent / "LaMP"
    if not lamp_dir.exists():
        print(f"Error: LaMP directory not found at {lamp_dir}")
        print("Please download LaMP dataset and place it in the LaMP/ directory")
        return

    embedder = LocalEmbedder()

    task_configs = [
        {"number": 1, "name": "Citation_Identification", "dir": "Citation_Identification"},
        {"number": 2, "name": "Movie_Tagging", "dir": "Movie_Tagging"},
        {"number": 3, "name": "Product_Rating", "dir": "Product_Rating"},
        {"number": 4, "name": "News_Headline_Generation", "dir": "News_Headline_Generation"},
        {"number": 5, "name": "Scholarly_Title_Generation", "dir": "Scholarly_Title_Generation"},
        {"number": 6, "name": "Tweet_Paraphrasing", "dir": "Tweet_Paraphrasing"},
        {"number": 7, "name": "Email_Subject_Generation", "dir": "Email_Subject_Generation"},
    ]

    for task_config in task_configs:
        task_dir = lamp_dir / task_config["dir"]
        if not task_dir.exists():
            print(f"  Skipping {task_config['name']} (directory not found)")
            continue

        print(f"\nProcessing LaMP-{task_config['number']}: {task_config['name']}...")

        task = await prisma_client.lamptask.upsert(
            where={"taskNumber": task_config["number"]},
            data={
                "create": {
                    "taskNumber": task_config["number"],
                    "name": task_config["name"],
                    "description": f"LaMP Task {task_config['number']}: {task_config['name']}",
                },
                "update": {},
            },
        )

        for variant in ["user_based", "time_based"]:
            for split in ["train", "validation", "test"]:
                split_capitalized = split.capitalize()
                questions_file = (
                    task_dir / split_capitalized / "Inputs" / f"{split}_questions_{variant}.json"
                )
                outputs_file = (
                    task_dir / split_capitalized / "Outputs" / f"{split}_outputs_{variant}.json"
                )

                if not questions_file.exists():
                    continue

                print(f"  Loading {split}/{variant}...")

                with open(questions_file, "r", encoding="utf-8") as f:
                    questions_data = json.load(f)

                outputs_data = {}
                if outputs_file.exists():
                    with open(outputs_file, "r", encoding="utf-8") as f:
                        outputs_json = json.load(f)
                        outputs_list = outputs_json.get("golds", [])
                        outputs_data = {item["id"]: item["output"] for item in outputs_list}

                if limit:
                    questions_data = questions_data[:limit]

                for item in questions_data:
                    sample_id = item["id"]
                    input_text = item["input"]
                    output_text = outputs_data.get(sample_id)
                    profile = item.get("profile", [])

                    sample = await prisma_client.lampsample.upsert(
                        where={
                            "sampleId_taskId_split_variant": {
                                "sampleId": sample_id,
                                "taskId": task.id,
                                "split": split,
                                "variant": variant,
                            }
                        },
                        data={
                            "create": {
                                "sampleId": sample_id,
                                "taskId": task.id,
                                "input": input_text,
                                "output": output_text,
                                "split": split,
                                "variant": variant,
                            },
                            "update": {
                                "input": input_text,
                                "output": output_text,
                            },
                        },
                    )

                    for profile_item in profile:
                        profile_id = profile_item.get("id", "")
                        title = profile_item.get("title")
                        content = profile_item.get("text", profile_item.get("abstract", ""))

                        if not content:
                            continue

                        embedding = embedder.encode(content)
                        embedding_str = json.dumps(embedding)

                        await prisma_client.profiledocument.upsert(
                            where={
                                "id": f"{sample.id}_{profile_id}",
                            },
                            data={
                                "create": {
                                    "id": f"{sample.id}_{profile_id}",
                                    "profileId": profile_id,
                                    "sampleId": sample.id,
                                    "title": title,
                                    "content": content,
                                    "embedding": embedding_str,
                                },
                                "update": {
                                    "title": title,
                                    "content": content,
                                    "embedding": embedding_str,
                                },
                            },
                        )

                print(f"    Loaded {len(questions_data)} samples")

    print("\nLaMP dataset loaded successfully!")


async def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Add documents to RAG database")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to load per split/variant (LaMP only)",
    )
    args = parser.parse_args()

    db_connection = PrismaConnection()
    prisma = await db_connection.connect()

    try:
        await add_lamp_dataset(prisma, limit=args.limit)
    finally:
        await db_connection.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
