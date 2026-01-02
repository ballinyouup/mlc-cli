import asyncio
from typing import List, Dict, Optional
from prisma import Prisma
from embeddings import EmbeddingModel
import json

class RAGSystem:
    def __init__(self, prisma: Prisma, embedding_model: EmbeddingModel):
        """Initialize RAG system with Prisma client and embedding model"""
        self.prisma = prisma
        self.embedding_model = embedding_model
    
    async def add_document(self, content: str, metadata: Optional[Dict] = None, chunk_size: int = 500) -> str:
        """Add a document to the database, split into chunks with embeddings"""
        metadata_str = json.dumps(metadata) if metadata else None
        
        document = await self.prisma.document.create(
            data={
                "content": content,
                "metadata": metadata_str
            }
        )
        
        chunks = self._split_text(content, chunk_size)
        
        for position, chunk_text in enumerate(chunks):
            embedding = self.embedding_model.encode(chunk_text)
            embedding_str = self.embedding_model.serialize_embedding(embedding)
            
            await self.prisma.chunk.create(
                data={
                    "documentId": document.id,
                    "content": chunk_text,
                    "embedding": embedding_str,
                    "position": position
                }
            )
        
        return document.id
    
    async def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve the most relevant chunks for a query"""
        query_embedding = self.embedding_model.encode(query)
        
        all_chunks = await self.prisma.chunk.find_many(
            include={"document": True}
        )
        
        chunk_scores = []
        for chunk in all_chunks:
            chunk_embedding = self.embedding_model.deserialize_embedding(chunk.embedding)
            similarity = self.embedding_model.cosine_similarity(query_embedding, chunk_embedding)
            
            chunk_scores.append({
                "chunk_id": chunk.id,
                "document_id": chunk.documentId,
                "content": chunk.content,
                "position": chunk.position,
                "similarity": similarity,
                "metadata": chunk.document.metadata if chunk.document else None
            })
        
        chunk_scores.sort(key=lambda x: x["similarity"], reverse=True)
        
        return chunk_scores[:top_k]
    
    async def build_context(self, query: str, top_k: int = 5) -> str:
        """Build context string from relevant chunks for RAG"""
        relevant_chunks = await self.retrieve_relevant_chunks(query, top_k)
        
        if not relevant_chunks:
            return ""
        
        context_parts = ["Retrieved context:"]
        for i, chunk in enumerate(relevant_chunks, 1):
            context_parts.append(f"\n[Context {i}] (Relevance: {chunk['similarity']:.3f})")
            context_parts.append(chunk['content'])
        
        return "\n".join(context_parts)
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks"""
        try:
            await self.prisma.document.delete(
                where={"id": document_id}
            )
            return True
        except Exception:
            return False
    
    async def list_documents(self) -> List[Dict]:
        """List all documents in the database"""
        documents = await self.prisma.document.find_many(
            include={"chunks": True}
        )
        
        return [
            {
                "id": doc.id,
                "content_preview": doc.content[:100] + "..." if len(doc.content) > 100 else doc.content,
                "metadata": json.loads(doc.metadata) if doc.metadata else None,
                "chunk_count": len(doc.chunks),
                "created_at": doc.createdAt
            }
            for doc in documents
        ]
    
    def _split_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into chunks of approximately chunk_size characters"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1
            if current_length + word_length > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
