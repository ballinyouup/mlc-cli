import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
import json

class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding model"""
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def encode(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    @staticmethod
    def serialize_embedding(embedding: List[float]) -> str:
        """Serialize embedding to JSON string for storage"""
        return json.dumps(embedding)
    
    @staticmethod
    def deserialize_embedding(embedding_str: str) -> List[float]:
        """Deserialize embedding from JSON string"""
        return json.loads(embedding_str)
    
    @staticmethod
    def cosine_similarity(emb1: List[float], emb2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        emb1_np = np.array(emb1)
        emb2_np = np.array(emb2)
        
        dot_product = np.dot(emb1_np, emb2_np)
        norm1 = np.linalg.norm(emb1_np)
        norm2 = np.linalg.norm(emb2_np)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
