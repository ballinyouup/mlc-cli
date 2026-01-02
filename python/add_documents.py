import asyncio
from db_init import init_prisma, close_prisma
from embeddings import EmbeddingModel
from rag import RAGSystem

async def add_sample_documents():
    """Add sample documents to the RAG database"""
    prisma = await init_prisma()
    embedding_model = EmbeddingModel()
    rag_system = RAGSystem(prisma, embedding_model)
    
    sample_docs = [
        {
            "content": """The meaning of life is a philosophical question concerning the significance of living or existence in general. 
            Many religions and philosophies have different perspectives on this question. Some believe it's about finding happiness and fulfillment, 
            others think it's about serving a higher purpose or deity, while some philosophical traditions suggest creating your own meaning.""",
            "metadata": {"topic": "philosophy", "source": "general_knowledge"}
        },
        {
            "content": """Artificial Intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. 
            AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.""",
            "metadata": {"topic": "technology", "source": "ai_basics"}
        },
        {
            "content": """Machine Learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. 
            It focuses on the development of computer programs that can access data and use it to learn for themselves.""",
            "metadata": {"topic": "machine_learning", "source": "ml_basics"}
        }
    ]
    
    print("Adding documents to RAG database...")
    for i, doc in enumerate(sample_docs, 1):
        doc_id = await rag_system.add_document(doc["content"], doc["metadata"])
        print(f"Added document {i}/{len(sample_docs)} - ID: {doc_id}")
    
    print("\nListing all documents:")
    documents = await rag_system.list_documents()
    for doc in documents:
        print(f"\nDocument ID: {doc['id']}")
        print(f"Preview: {doc['content_preview']}")
        print(f"Chunks: {doc['chunk_count']}")
        print(f"Metadata: {doc['metadata']}")
    
    await close_prisma(prisma)
    print("\nDocuments added successfully!")

if __name__ == "__main__":
    asyncio.run(add_sample_documents())
