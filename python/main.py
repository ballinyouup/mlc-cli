import os
import asyncio
import argparse
from mlc_llm import MLCEngine
from db_init import init_prisma, close_prisma
from embeddings import EmbeddingModel
from rag import RAGSystem

# Set JIT policy for model compilation
os.environ["MLC_JIT_POLICY"] = "REDO"


async def run_generic_rag(rag_system: RAGSystem, engine: MLCEngine, model: str, query: str):
    """Run generic RAG query"""
    print("Retrieving relevant context...")
    context = await rag_system.build_context(query, top_k=3)
    
    if context:
        enhanced_prompt = f"{context}\n\nUser Question: {query}\n\nPlease answer based on the context provided above."
        print(f"\n--- RAG Context Retrieved ---\n{context}\n--- End Context ---\n")
    else:
        enhanced_prompt = query
        print("No relevant context found in database.\n")
    
    print("Generating response...\n")
    for response in engine.chat.completions.create(
            messages=[{"role": "user", "content": enhanced_prompt}],
            model=model,
            stream=True,
    ):
        for choice in response.choices:
            print(choice.delta.content, end="", flush=True)
    print("\n")


async def run_lamp_personalization(
    rag_system: RAGSystem, 
    engine: MLCEngine, 
    model: str,
    sample_id: str,
    task_number: int = None,
    split: str = "validation",
    variant: str = "user_based",
    top_k: int = 5
):
    """Run LaMP personalization task"""
    print(f"Loading LaMP sample: {sample_id}")
    
    lamp_context = await rag_system.build_lamp_context(
        sample_id=sample_id,
        task_number=task_number,
        top_k=top_k,
        split=split,
        variant=variant
    )
    
    if "error" in lamp_context:
        print(f"Error: {lamp_context['error']}")
        return
    
    sample = lamp_context["sample"]
    print(f"\n--- LaMP Task: {sample['task']['name']} ---")
    print(f"Sample ID: {sample['sampleId']}")
    print(f"Split: {sample['split']}, Variant: {sample['variant']}")
    print(f"Profile items retrieved: {lamp_context['retrieved_profiles']}")
    
    print(f"\n--- User Profile Context ---\n{lamp_context['profile_context']}\n--- End Profile ---\n")
    print(f"--- Input Task ---\n{lamp_context['input']}\n--- End Input ---\n")
    
    if lamp_context["expected_output"]:
        print(f"--- Expected Output ---\n{lamp_context['expected_output']}\n--- End Expected ---\n")
    
    # Build task-specific personalized prompt
    task_name = sample['task']['name']
    task_instructions = {
        "Citation_Identification": "You are analyzing which academic reference a researcher would cite in their paper. Based on the researcher's publication history and expertise shown below, determine which reference is most relevant to their research domain.",
        "Movie_Tagging": "Based on the user's movie watching history and preferences shown below, predict appropriate tags for a new movie.",
        "Product_Rating": "Based on the user's product review history and rating patterns shown below, predict how they would rate this product.",
        "News_Headline_Generation": "Based on the user's news article writing style and topics shown below, generate an appropriate headline.",
        "Scholarly_Title_Generation": "Based on the user's academic writing style and research focus shown below, generate an appropriate scholarly title.",
        "Email_Subject_Generation": "Based on the user's email writing style and subject line patterns shown below, generate an appropriate email subject.",
        "Tweet_Paraphrasing": "Based on the user's tweet writing style and language patterns shown below, paraphrase the given tweet in their style."
    }
    
    instruction = task_instructions.get(task_name, "Based on the user's historical data shown below, complete the following task.")
    
    enhanced_prompt = f"""{instruction}

User's Research/Activity History:
{lamp_context['profile_context']}

Task: {lamp_context['input']}

Important: Analyze the user's domain expertise and patterns from their history above. Provide ONLY the answer in the exact format requested, without explanation."""

    print("Generating personalized response...\n")
    for response in engine.chat.completions.create(
            messages=[{"role": "user", "content": enhanced_prompt}],
            model=model,
            stream=True,
    ):
        for choice in response.choices:
            print(choice.delta.content, end="", flush=True)
    print("\n")


async def list_lamp_data(rag_system: RAGSystem, task_number: int = None, limit: int = 10):
    """List available LaMP tasks and samples"""
    print("\n=== LaMP Tasks ===")
    tasks = await rag_system.list_lamp_tasks()
    
    if not tasks:
        print("No LaMP tasks loaded. Run: python add_documents.py --lamp")
        return
    
    for task in tasks:
        print(f"  LaMP-{task['number']}: {task['name']} ({task['sample_count']} samples)")
    
    print(f"\n=== Sample Data (limit: {limit}) ===")
    samples = await rag_system.list_lamp_samples(
        task_number=task_number,
        limit=limit
    )
    
    for sample in samples:
        print(f"\n  [{sample['task_name']}] ID: {sample['sampleId']}")
        print(f"    Split: {sample['split']}, Variant: {sample['variant']}")
        print(f"    Profile docs: {sample['profile_count']}")
        print(f"    Input: {sample['input_preview']}")
        if sample['output']:
            print(f"    Output: {sample['output'][:80]}...")


async def main():
    parser = argparse.ArgumentParser(description="RAG System with LaMP Personalization")
    parser.add_argument("--lamp", action="store_true", help="Run LaMP personalization mode")
    parser.add_argument("--sample-id", type=str, help="LaMP sample ID to process")
    parser.add_argument("--task", type=int, help="LaMP task number (1-7)")
    parser.add_argument("--split", type=str, default="validation", help="Data split (train/validation/test)")
    parser.add_argument("--variant", type=str, default="user_based", help="Variant (user_based/time_based)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of profile docs to retrieve")
    parser.add_argument("--query", type=str, help="Custom query for generic RAG")
    parser.add_argument("--list", action="store_true", help="List available LaMP data")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM inference (for testing)")
    
    args = parser.parse_args()
    
    # Initialize database and RAG system
    print("Initializing RAG system...")
    prisma = await init_prisma()
    embedding_model = EmbeddingModel()
    rag_system = RAGSystem(prisma, embedding_model)
    
    # List mode - no LLM needed
    if args.list:
        await list_lamp_data(rag_system, task_number=args.task)
        await close_prisma(prisma)
        return
    
    # Initialize LLM engine if needed
    engine = None
    # model = "../mlc-llm/models/Llama-3-8B-Instruct-q3f16_1-MLC"
    model = "../mlc-llm/models/Qwen3-4B-q4f16_1-MLC"
    
    if not args.no_llm:
        engine = MLCEngine(model, device="cuda")
    
    try:
        if args.lamp and args.sample_id:
            if args.no_llm:
                lamp_context = await rag_system.build_lamp_context(
                    sample_id=args.sample_id,
                    task_number=args.task,
                    top_k=args.top_k,
                    split=args.split,
                    variant=args.variant
                )
                if "error" not in lamp_context:
                    print(f"\n--- Profile Context ---\n{lamp_context['profile_context']}")
                    print(f"\n--- Input ---\n{lamp_context['input']}")
                    print(f"\n--- Expected ---\n{lamp_context['expected_output']}")
                else:
                    print(lamp_context["error"])
            else:
                await run_lamp_personalization(
                    rag_system, engine, model,
                    sample_id=args.sample_id,
                    task_number=args.task,
                    split=args.split,
                    variant=args.variant,
                    top_k=args.top_k
                )
        elif args.query:
            if args.no_llm:
                context = await rag_system.build_context(args.query, top_k=3)
                print(f"Context:\n{context}")
            else:
                await run_generic_rag(rag_system, engine, model, args.query)
        else:
            default_query = "What is the meaning of life?"
            if args.no_llm:
                context = await rag_system.build_context(default_query, top_k=3)
                print(f"Context:\n{context}")
            else:
                await run_generic_rag(rag_system, engine, model, default_query)
    finally:
        if engine:
            engine.terminate()
        await close_prisma(prisma)


if __name__ == "__main__":
    asyncio.run(main())