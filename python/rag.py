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

    # ============== LaMP Personalization Methods ==============
    
    async def get_lamp_sample(self, sample_id: str, task_number: int = None, 
                              split: str = None, variant: str = None) -> Optional[Dict]:
        """Get a LaMP sample by its original ID with optional filters"""
        where_clause = {"sampleId": sample_id}
        if split:
            where_clause["split"] = split
        if variant:
            where_clause["variant"] = variant
        
        if task_number:
            task = await self.prisma.lamptask.find_first(
                where={"taskNumber": task_number}
            )
            if task:
                where_clause["taskId"] = task.id
        
        sample = await self.prisma.lampsample.find_first(
            where=where_clause,
            include={"task": True, "profileItems": True}
        )
        
        if not sample:
            return None
        
        return {
            "id": sample.id,
            "sampleId": sample.sampleId,
            "input": sample.input,
            "output": sample.output,
            "split": sample.split,
            "variant": sample.variant,
            "task": {
                "number": sample.task.taskNumber,
                "name": sample.task.name,
                "description": sample.task.description
            },
            "profile_count": len(sample.profileItems)
        }
    
    async def retrieve_profile_documents(
        self, 
        query: str, 
        sample_id: str = None,
        task_number: int = None,
        top_k: int = 5
    ) -> List[Dict]:
        """Retrieve most relevant profile documents for a query"""
        query_embedding = self.embedding_model.encode(query)
        
        where_clause = {}
        if sample_id:
            sample = await self.prisma.lampsample.find_first(
                where={"sampleId": sample_id}
            )
            if sample:
                where_clause["sampleId"] = sample.id
        
        profile_docs = await self.prisma.profiledocument.find_many(
            where=where_clause if where_clause else None,
            include={"sample": {"include": {"task": True}}}
        )
        
        scored_docs = []
        for doc in profile_docs:
            doc_embedding = self.embedding_model.deserialize_embedding(doc.embedding)
            similarity = self.embedding_model.cosine_similarity(query_embedding, doc_embedding)
            
            scored_docs.append({
                "profile_id": doc.profileId,
                "title": doc.title,
                "content": doc.content,
                "similarity": similarity,
                "sample_id": doc.sample.sampleId if doc.sample else None,
                "task_name": doc.sample.task.name if doc.sample and doc.sample.task else None
            })
        
        scored_docs.sort(key=lambda x: x["similarity"], reverse=True)
        return scored_docs[:top_k]
    
    async def build_lamp_context(
        self, 
        sample_id: str,
        task_number: int = None,
        top_k: int = 5,
        split: str = None,
        variant: str = None
    ) -> Dict:
        """Build personalized context for a LaMP sample"""
        sample = await self.get_lamp_sample(
            sample_id, 
            task_number=task_number,
            split=split,
            variant=variant
        )
        
        if not sample:
            return {"error": f"Sample {sample_id} not found"}
        
        relevant_profiles = await self.retrieve_profile_documents(
            query=sample["input"],
            sample_id=sample_id,
            top_k=top_k
        )
        
        context_parts = ["User's historical data:"]
        for i, profile in enumerate(relevant_profiles, 1):
            context_parts.append(f"\n[Profile {i}] (Relevance: {profile['similarity']:.3f})")
            if profile["title"]:
                context_parts.append(f"Title: {profile['title']}")
            context_parts.append(profile["content"])
        
        profile_context = "\n".join(context_parts) if relevant_profiles else ""
        
        return {
            "sample": sample,
            "input": sample["input"],
            "profile_context": profile_context,
            "expected_output": sample["output"],
            "retrieved_profiles": len(relevant_profiles)
        }
    
    async def list_lamp_tasks(self) -> List[Dict]:
        """List all LaMP tasks in the database"""
        tasks = await self.prisma.lamptask.find_many(
            include={"samples": True}
        )
        
        return [
            {
                "id": task.id,
                "number": task.taskNumber,
                "name": task.name,
                "description": task.description,
                "sample_count": len(task.samples)
            }
            for task in tasks
        ]
    
    async def list_lamp_samples(
        self, 
        task_number: int = None,
        split: str = None,
        variant: str = None,
        limit: int = 10
    ) -> List[Dict]:
        """List LaMP samples with optional filters"""
        where_clause = {}
        
        if task_number:
            task = await self.prisma.lamptask.find_first(
                where={"taskNumber": task_number}
            )
            if task:
                where_clause["taskId"] = task.id
        
        if split:
            where_clause["split"] = split
        if variant:
            where_clause["variant"] = variant
        
        samples = await self.prisma.lampsample.find_many(
            where=where_clause if where_clause else None,
            include={"task": True, "profileItems": True},
            take=limit
        )
        
        return [
            {
                "id": sample.id,
                "sampleId": sample.sampleId,
                "input_preview": sample.input[:100] + "..." if len(sample.input) > 100 else sample.input,
                "output": sample.output,
                "split": sample.split,
                "variant": sample.variant,
                "task_name": sample.task.name,
                "profile_count": len(sample.profileItems)
            }
            for sample in samples
        ]
