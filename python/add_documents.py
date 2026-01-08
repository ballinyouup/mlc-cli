import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from db_init import init_prisma, close_prisma
from embeddings import EmbeddingModel
from rag import RAGSystem

# LaMP task mapping
LAMP_TASKS = {
    "Citation_Identification": {"number": 1, "description": "Personalized Citation Identification"},
    "Movie_Tagging": {"number": 2, "description": "Personalized Movie Tagging"},
    "Product_Rating": {"number": 3, "description": "Personalized Product Rating"},
    "News_Headline_Generation": {"number": 4, "description": "Personalized News Headline Generation"},
    "Scholarly_Title_Generation": {"number": 5, "description": "Personalized Scholarly Title Generation"},
    "Email_Subject_Generation": {"number": 6, "description": "Personalized Email Subject Generation"},
    "Tweet_Paraphrasing": {"number": 7, "description": "Personalized Tweet Paraphrasing"},
}

class LaMPDataLoader:
    def __init__(self, prisma, embedding_model: EmbeddingModel, lamp_dir: str = "LaMP"):
        self.prisma = prisma
        self.embedding_model = embedding_model
        self.lamp_dir = Path(lamp_dir)
        self.task_cache: Dict[str, str] = {}
    
    async def get_or_create_task(self, task_name: str) -> str:
        if task_name in self.task_cache:
            return self.task_cache[task_name]
        
        task_info = LAMP_TASKS.get(task_name)
        if not task_info:
            raise ValueError(f"Unknown LaMP task: {task_name}")
        
        existing = await self.prisma.lamptask.find_first(
            where={"taskNumber": task_info["number"]}
        )
        
        if existing:
            self.task_cache[task_name] = existing.id
            return existing.id
        
        task = await self.prisma.lamptask.create(
            data={
                "taskNumber": task_info["number"],
                "name": task_name,
                "description": task_info["description"]
            }
        )
        self.task_cache[task_name] = task.id
        return task.id
    
    def _load_json_file(self, filepath: Path) -> Optional[any]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def _get_profile_content(self, profile_item: dict) -> str:
        content_parts = []
        if profile_item.get("title"):
            content_parts.append(f"Title: {profile_item['title']}")
        if profile_item.get("abstract"):
            content_parts.append(profile_item["abstract"])
        elif profile_item.get("text"):
            content_parts.append(profile_item["text"])
        return "\n".join(content_parts)
    
    async def load_task_data(
        self, 
        task_name: str, 
        splits: List[str] = None,
        variants: List[str] = None,
        max_samples: int = None
    ) -> Dict[str, int]:
        splits = splits or ["train", "validation", "test"]
        variants = variants or ["user_based", "time_based"]
        
        task_dir = self.lamp_dir / task_name
        if not task_dir.exists():
            print(f"Task directory not found: {task_dir}")
            return {"samples": 0, "profiles": 0}
        
        task_id = await self.get_or_create_task(task_name)
        stats = {"samples": 0, "profiles": 0}
        
        for split in splits:
            split_dir = task_dir / split.capitalize()
            if not split_dir.exists():
                continue
            
            inputs_dir = split_dir / "Inputs"
            outputs_dir = split_dir / "Outputs"
            
            for variant in variants:
                if split == "test":
                    input_file = inputs_dir / f"test_questions_{variant}.json"
                else:
                    input_file = inputs_dir / f"{split}_questions_{variant}.json"
                
                if not input_file.exists():
                    continue
                
                print(f"Loading {task_name}/{split}/{variant}...")
                inputs_data = self._load_json_file(input_file)
                if not inputs_data:
                    continue
                
                outputs_map = {}
                if split == "test":
                    output_file = outputs_dir / f"test_outputs_{variant}.json"
                else:
                    output_file = outputs_dir / f"{split}_outputs_{variant}.json"
                
                if output_file.exists():
                    outputs_data = self._load_json_file(output_file)
                    if outputs_data and "golds" in outputs_data:
                        outputs_map = {g["id"]: g["output"] for g in outputs_data["golds"]}
                
                samples_to_process = inputs_data[:max_samples] if max_samples else inputs_data
                
                for sample_data in samples_to_process:
                    sample_id = sample_data["id"]
                    input_text = sample_data["input"]
                    output_text = outputs_map.get(sample_id)
                    profile = sample_data.get("profile", [])
                    
                    existing = await self.prisma.lampsample.find_first(
                        where={
                            "sampleId": sample_id,
                            "taskId": task_id,
                            "split": split,
                            "variant": variant
                        }
                    )
                    
                    if existing:
                        continue
                    
                    sample = await self.prisma.lampsample.create(
                        data={
                            "sampleId": sample_id,
                            "taskId": task_id,
                            "input": input_text,
                            "output": output_text,
                            "split": split,
                            "variant": variant
                        }
                    )
                    stats["samples"] += 1
                    
                    for profile_item in profile:
                        content = self._get_profile_content(profile_item)
                        if not content.strip():
                            continue
                        
                        embedding = self.embedding_model.encode(content)
                        embedding_str = self.embedding_model.serialize_embedding(embedding)
                        
                        await self.prisma.profiledocument.create(
                            data={
                                "profileId": profile_item.get("id", ""),
                                "sampleId": sample.id,
                                "title": profile_item.get("title"),
                                "content": content,
                                "embedding": embedding_str
                            }
                        )
                        stats["profiles"] += 1
                    
                    if stats["samples"] % 10 == 0:
                        print(f"  Processed {stats['samples']} samples, {stats['profiles']} profile docs...")
        
        return stats
    
    async def load_all_tasks(
        self, 
        splits: List[str] = None,
        variants: List[str] = None,
        max_samples_per_task: int = None
    ) -> Dict[str, Dict[str, int]]:
        results = {}
        
        for task_name in LAMP_TASKS.keys():
            task_dir = self.lamp_dir / task_name
            if task_dir.exists():
                print(f"\n{'='*50}")
                print(f"Loading task: {task_name}")
                print(f"{'='*50}")
                stats = await self.load_task_data(
                    task_name, 
                    splits=splits, 
                    variants=variants,
                    max_samples=max_samples_per_task
                )
                results[task_name] = stats
                print(f"Completed {task_name}: {stats['samples']} samples, {stats['profiles']} profiles")
        
        return results


async def add_lamp_documents(
    task_name: str = None,
    splits: List[str] = None,
    variants: List[str] = None,
    max_samples: int = None
):
    prisma = await init_prisma()
    embedding_model = EmbeddingModel()
    
    lamp_dir = Path(__file__).parent / "LaMP"
    loader = LaMPDataLoader(prisma, embedding_model, lamp_dir)
    
    if task_name:
        print(f"Loading LaMP task: {task_name}")
        stats = await loader.load_task_data(
            task_name, 
            splits=splits, 
            variants=variants,
            max_samples=max_samples
        )
        print(f"\nTotal: {stats['samples']} samples, {stats['profiles']} profile documents")
    else:
        print("Loading all LaMP tasks...")
        results = await loader.load_all_tasks(
            splits=splits, 
            variants=variants,
            max_samples_per_task=max_samples
        )
        
        total_samples = sum(r["samples"] for r in results.values())
        total_profiles = sum(r["profiles"] for r in results.values())
        print(f"\n{'='*50}")
        print(f"Grand Total: {total_samples} samples, {total_profiles} profile documents")
    
    await close_prisma(prisma)
    print("\nLaMP data loaded successfully!")



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add documents to RAG database")
    parser.add_argument("--lamp", action="store_true", help="Load LaMP dataset")
    parser.add_argument("--task", type=str, help="Specific LaMP task to load (e.g., Citation_Identification)")
    parser.add_argument("--splits", nargs="+", default=None, help="Splits to load (train, validation, test)")
    parser.add_argument("--variants", nargs="+", default=None, help="Variants to load (user_based, time_based)")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per task/split")
    parser.add_argument("--sample", action="store_true", help="Load sample documents (legacy)")
    
    args = parser.parse_args()
    
    if args.lamp:
        asyncio.run(add_lamp_documents(
            task_name=args.task,
            splits=args.splits,
            variants=args.variants,
            max_samples=args.max_samples
        ))
    elif args.sample:
        asyncio.run(add_sample_documents())
    else:
        print("Usage:")
        print("  Load LaMP data:    python add_documents.py --lamp [--task TASK] [--max-samples N]")
        print("  Load sample docs:  python add_documents.py --sample")
        print("\nAvailable LaMP tasks:")
        for task in LAMP_TASKS.keys():
            print(f"  - {task}")
