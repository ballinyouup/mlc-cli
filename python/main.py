import os
import asyncio
from mlc_llm import MLCEngine
from db_init import init_prisma, close_prisma
from embeddings import EmbeddingModel
from rag import RAGSystem

# Set JIT policy for model compilation
os.environ["MLC_JIT_POLICY"] = "REDO"

async def main():
    # Initialize database and RAG system
    print("Initializing RAG system...")
    prisma = await init_prisma()
    embedding_model = EmbeddingModel()
    rag_system = RAGSystem(prisma, embedding_model)
    
    # Create engine
    model = "../mlc-llm/models/Llama-3-8B-Instruct-q4f16_1-MLC"
    engine = MLCEngine(model, device="cuda")
    
    # User query
    user_query = "What is the meaning of life?"
    
    # Retrieve relevant context from RAG before generating response
    print("Retrieving relevant context...")
    context = await rag_system.build_context(user_query, top_k=3)
    
    # Build enhanced prompt with RAG context
    if context:
        enhanced_prompt = f"{context}\n\nUser Question: {user_query}\n\nPlease answer based on the context provided above."
        print(f"\n--- RAG Context Retrieved ---\n{context}\n--- End Context ---\n")
    else:
        enhanced_prompt = user_query
        print("No relevant context found in database.\n")
    
    # Run chat completion with RAG-enhanced prompt
    print("Generating response...\n")
    for response in engine.chat.completions.create(
            messages=[{"role": "user", "content": enhanced_prompt}],
            model=model,
            stream=True,
    ):
        for choice in response.choices:
            print(choice.delta.content, end="", flush=True)
    print("\n")
    
    # Cleanup
    engine.terminate()
    await close_prisma(prisma)

if __name__ == "__main__":
    asyncio.run(main())