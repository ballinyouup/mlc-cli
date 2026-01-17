#there are 3 differet types of rag in here

#vector rag, graph rag ,and hybrid rag

import numpy as np
import json
from abc import ABC, abstractmethod
from collections import defaultdict
import logging
from datetime import datetime
import pickle

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")

from kg_db import KnowledgeGraph, KnowledgeGraphBuilder


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseRAG(ABC):
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def index(self, data, **kwargs):
        pass
    
    @abstractmethod
    def retrieve(self, query, k: int = 10, **kwargs):
        pass
    
    @abstractmethod
    def generate(self, query, context, **kwargs):
        pass
    
    def __call__(self, query, k: int = 10, **kwargs):
        context = self.retrieve(query, k=k, **kwargs)
        response = self.generate(query, context, **kwargs)
        return response, context


class VectorRAG(BaseRAG):
    def __init__(self, model_name = 'all-MiniLM-L6-v2'):
        super().__init__("VectorRAG")
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers required for VectorRAG")
        
        self.model = SentenceTransformer(model_name)
        self.vector_index = {
            'embeddings': [],
            'metadata': [],
            'items': []
        }
        self.logger.info(f"Initialized VectorRAG with model: {model_name}")
    
    def embed_text(self, text):
        if isinstance(text, str):
            text = [text]
        return self.model.encode(text, show_progress_bar=False)
    
    def prepare_text_for_embedding(self, item, task):
        if task == 'Movie_Tagging':
            description = item.get('description', '')
            tag = item.get('tag', '')
            return f"{tag}: {description}"
        
        elif task == 'Product_Rating':
            return item.get('text', item.get('description', ''))
        
        elif task == 'Citation_Identification':
            title = item.get('title', '')
            abstract = item.get('abstract', '')
            return f"{title}. {abstract}"
        
        elif task in ['Email_Subject_Generation', 'News_Headline_Generation', 'Scholarly_Title_Generation']:
            return item.get('text', item.get('content', ''))
        
        elif task == 'Tweet_Paraphrasing':
            return item.get('tweet', item.get('text', ''))
        
        else:
            text_parts = []
            for key, value in item.items():
                if isinstance(value, str) and len(value) > 0:
                    text_parts.append(value)
            return ' '.join(text_parts)
    
    def index(self, data, task = None, batch_size = 32):
        self.logger.info(f"Indexing {len(data)} items")
        
        self.vector_index = {
            'embeddings': [],
            'metadata': [],
            'items': []
        }
        
        texts = []
        for item in data:
            if 'profile' in item and item['profile']:
                for profile_item in item['profile']:
                    text = self.prepare_text_for_embedding(profile_item, task)
                    texts.append(text)
                    self.vector_index['metadata'].append({
                        'user_id': item.get('user_id'),
                        'item_id': profile_item.get('id'),
                        'timestamp': profile_item.get('date'),
                        'task': task,
                        'type': 'profile'
                    })
                    self.vector_index['items'].append(profile_item)
        

        self.logger.info(f"Generating embeddings for {len(texts)} texts")
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            embeddings = self.embed_text(batch)
            self.vector_index['embeddings'].extend(embeddings)
        
        self.vector_index['embeddings'] = np.array(self.vector_index['embeddings'])
        self.logger.info(f"Index built with {len(self.vector_index['embeddings'])} vectors")
    
    def retrieve(self, query, k=10, task=None, filter_by_user= False):
        query_text = query.get('input', '')
        if not query_text:
            return []
        
        if len(self.vector_index['embeddings']) == 0:
            self.logger.warning("Index is empty, returning empty results")
            return []
        
        query_embedding = self.embed_text(query_text)[0]
        
        # cos sim
        similarities = np.dot(self.vector_index['embeddings'], query_embedding) / (np.linalg.norm(self.vector_index['embeddings'], axis=1) * np.linalg.norm(query_embedding))
        
        valid_indices = np.arange(len(similarities))
        
        if task:
            valid_indices = [
                i for i in valid_indices 
                if self.vector_index['metadata'][i].get('task') == task
            ]
        
        if filter_by_user and 'user_id' in query:
            user_id = query['user_id']
            valid_indices = [
                i for i in valid_indices
                if self.vector_index['metadata'][i].get('user_id') == user_id
            ]
        
        if len(valid_indices) == 0:
            return []
        
        valid_similarities = similarities[valid_indices]
        top_k_local = np.argsort(valid_similarities)[-k:][::-1]
        top_k_global = [valid_indices[i] for i in top_k_local]
        
        results = []
        for idx in top_k_global:
            results.append({
                'item': self.vector_index['items'][idx],
                'metadata': self.vector_index['metadata'][idx],
                'score': float(similarities[idx])
            })
        
        return results
    
    def generate(self, query, context, task=None):
        if not context:
            return None
        
        if task == 'Movie_Tagging':
            tags = [item['item'].get('tag') for item in context if 'tag' in item['item']]
            if tags:
                from collections import Counter
                return Counter(tags).most_common(1)[0][0]
        
        elif task == 'Product_Rating':
            ratings = []
            for item in context:
                if 'rating' in item['item']:
                    try:
                        ratings.append(float(item['item']['rating']))
                    except:
                        pass
            return np.mean(ratings) if ratings else None
        
        else:
            return context[0]['item']
    
    def save_index(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.vector_index, f)
        self.logger.info(f"Index saved to {filepath}")
    
    def load_index(self, filepath):
        with open(filepath, 'rb') as f:
            self.vector_index = pickle.load(f)
        self.logger.info(f"Index loaded from {filepath}")


class GraphRAG(BaseRAG):
    def __init__(self, kg):
        super().__init__("GraphRAG")
        self.kg = kg
        self.logger.info("Initialized GraphRAG with knowledge graph")
    
    def index(self, data, task=None, **kwargs):
        
        if not task:
            raise ValueError("Task must be specified for GraphRAG indexing")
        
        builder = KnowledgeGraphBuilder(self.kg)
        builder.process_task_data(data, task)
        self.kg.commit()
        
        self.logger.info(f"Graph indexed for task: {task}")
    
    def retrieve(self, query, k=10, task=None, max_depth=2, use_temporal=True, use_collaborative=True):
        user_id = query.get('user_id')
        if not user_id:
            return []
        
        user_entity_id = f"user_{user_id}"
        
        # strat 1: get user's direct interaction history
        profile = self.kg.get_user_profile(
            user_entity_id,
            task=query.get('task'),
            limit=k,
            time_ordered=use_temporal
        )
        
        results = []
        for item in profile:
            item_entity = self.kg.get_entity(item['item_id'])
            if item_entity:
                results.append({
                    'item': item_entity['attributes'],
                    'metadata': item,
                    'score': item.get('weight', 1.0),
                    'retrieval_method': 'direct_profile'
                })
        
        # strat 2: collab filtering via similar users
        if use_collaborative and len(results) < k:
            similar_users = self.kg.find_similar_users(user_entity_id, k=5)
            
            for similar_user_id, similarity in similar_users:
                similar_profile = self.kg.get_user_profile(
                    similar_user_id,
                    task=query.get('task'),
                    limit=k
                )
                
                for item in similar_profile:
                    item_entity = self.kg.get_entity(item['item_id'])
                    if item_entity:
                        if not any(r['item'].get('item_id') == item_entity['attributes'].get('item_id') for r in results):
                            results.append({
                                'item': item_entity['attributes'],
                                'metadata': item,
                                'score': similarity * item.get('weight', 1.0),
                                'retrieval_method': 'collaborative'
                            })
                
                if len(results) >= k:
                    break
        
        # strat 3: graph traversal for exploration
        if len(results) < k and max_depth > 1:
            traversal_results = self.kg.graph_traversal(
                user_entity_id,
                max_depth=max_depth,
                relationship_types=['interacted_with', 'similar_to']
            )
            
            for traversal_item in traversal_results:
                if traversal_item['entity_id'].startswith('item_'):
                    entity = self.kg.get_entity(traversal_item['entity_id'])
                    if entity and not any(r['item'].get('item_id') == entity['attributes'].get('item_id') for r in results):
                        results.append({
                            'item': entity['attributes'],
                            'metadata': {'depth': traversal_item['depth']},
                            'score': 1.0 / (traversal_item['depth'] + 1),
                            'retrieval_method': 'graph_traversal'
                        })
                
                if len(results) >= k:
                    break
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]
    
    def generate(self, query, context, task=None):
        if not context:
            return None
        
        methods = [c.get('retrieval_method') for c in context]
    
        if task == 'Movie_Tagging':
            tags = []
            weights = []
            for item in context:
                if 'tag' in item['item']:
                    tags.append(item['item']['tag'])
                    weight = item['score']
                    if item.get('retrieval_method') == 'direct_profile':
                        weight *= 1.5  
                    weights.append(weight)
            
            if tags:
                tag_weights = defaultdict(float)
                for tag, weight in zip(tags, weights):
                    tag_weights[tag] += weight
                return max(tag_weights.items(), key=lambda x: x[1])[0]
        
        else:
            return context[0]['item']


class HybridRAG(BaseRAG):
    def __init__(self, vector_rag, graph_rag,vector_weight=0.5):
        super().__init__("HybridRAG")
        self.vector_rag = vector_rag
        self.graph_rag = graph_rag
        self.vector_weight = vector_weight
        self.graph_weight = 1.0 - vector_weight
        self.logger.info(f"Initialized HybridRAG (vector_weight={vector_weight})")
    
    def index(self, data, task: str = None, **kwargs):
        self.logger.info("Indexing in vector system")
        self.vector_rag.index(data, task=task, **kwargs)
        
        self.logger.info("Indexing in graph system")
        self.graph_rag.index(data, task=task, **kwargs)
        
        self.logger.info("Hybrid indexing complete")
    
    def retrieve(self, query, k = 10, task=None, vector_k=None, graph_k=None, rerank=True,diversity_weight=0.1):
        vector_k = vector_k or k * 2
        graph_k = graph_k or k * 2
        
        task = task or query.get('task')
        
        self.logger.debug(f"Retrieving {vector_k} from vector system")
        vector_results = self.vector_rag.retrieve(
            query, k=vector_k, task=task
        )
        
        self.logger.debug(f"Retrieving {graph_k} from graph system")
        graph_results = self.graph_rag.retrieve(
            query, k=graph_k, task=task
        )
        
        merged = self.merge_results(vector_results, graph_results)
        
        if rerank:
            merged = self.rerank(merged, query, diversity_weight)
        
        return merged[:k]
    
    def merge_results(self, vector_results, graph_results):
        item_scores = {}
        item_data = {}
        
        for rank, result in enumerate(vector_results):
            item_id = result['item'].get('id', str(result['item']))
            score = self.vector_weight * (1.0 / (rank + 1))
            
            if item_id not in item_scores:
                item_scores[item_id] = 0.0
                item_data[item_id] = result
            
            item_scores[item_id] += score
            item_data[item_id]['vector_score'] = result.get('score', 0.0)
            item_data[item_id]['vector_rank'] = rank
        

        for rank, result in enumerate(graph_results):
            item_id = result['item'].get('id', str(result['item']))
            score = self.graph_weight * (1.0 / (rank + 1))
            
            if item_id not in item_scores:
                item_scores[item_id] = 0.0
                item_data[item_id] = result
            
            item_scores[item_id] += score
            item_data[item_id]['graph_score'] = result.get('score', 0.0)
            item_data[item_id]['graph_rank'] = rank
            item_data[item_id]['retrieval_method'] = result.get('retrieval_method', 'unknown')
        
        merged = []
        for item_id, total_score in sorted(item_scores.items(), key=lambda x: x[1], reverse=True):
            result = item_data[item_id].copy()
            result['hybrid_score'] = total_score
            merged.append(result)
        
        return merged
    
    def rerank(self, results, query, diversity_weight):
        
        if len(results) <= 1:
            return results
        
        # extract features for diversity calculation
        # for now were gonna use simple attribute-based diversity
        
        reranked = [results[0]] 
        remaining = results[1:]
        
        while remaining and len(reranked) < len(results):
            mmr_scores = []
            
            for candidate in remaining:
                relevance = candidate.get('hybrid_score', 0.0)
                
                max_similarity = 0.0
                for selected in reranked:
                    similarity = self.compute_similarity(candidate['item'], selected['item'])
                    max_similarity = max(max_similarity, similarity)
                
                #mmr scroe
                mmr = (1 - diversity_weight) * relevance - diversity_weight * max_similarity
                mmr_scores.append(mmr)
            
            best_idx = np.argmax(mmr_scores)
            reranked.append(remaining.pop(best_idx))
        
        return reranked
    
    def compute_similarity(self, item1, item2):
        #jaccard similarity
        keys1 = set(str(v) for v in item1.values() if isinstance(v, (str, int, float)))
        keys2 = set(str(v) for v in item2.values() if isinstance(v, (str, int, float)))
        
        if not keys1 or not keys2:
            return 0.0
        
        intersection = len(keys1 & keys2)
        union = len(keys1 | keys2)
        
        return intersection / union if union > 0 else 0.0
    
    def generate(self, query, context,task=None, use_ensemble=True):
        if not context:
            return None
        
        if use_ensemble:
            vector_prediction = self.vector_rag.generate(query, context, task)
            graph_prediction = self.graph_rag.generate(query, context, task)
            
            if graph_prediction is not None:
                return graph_prediction
            else:
                return vector_prediction
        else:
            return context[0]['item']


def create_rag_systems(kg, model_name='all-MiniLM-L6-v2'):
    systems = {}
    
    # vector RAG
    try:
        vector_rag = VectorRAG(model_name=model_name)
        systems['vector'] = vector_rag
        logger.info("Created VectorRAG")
    except Exception as e:
        logger.warning(f"Could not create VectorRAG: {e}")
        vector_rag = None
    
    # graph RAG
    graph_rag = GraphRAG(kg)
    systems['graph'] = graph_rag
    logger.info("Created GraphRAG")
    
    # hybrid RAG
    if vector_rag:
        hybrid_rag = HybridRAG(vector_rag, graph_rag, vector_weight=0.6)
        systems['hybrid'] = hybrid_rag
        logger.info("Created HybridRAG")
    
    return systems
