import sqlite3
import json
import os
import logging
import networkx as nx
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import pickle


class KnowledgeGraph:
    def __init__(self, db_path = "../knowledge_graph.db", use_networkx = True):
        self.db_path = db_path
        self.connection = None
        self.cursor = None
        self.use_networkx = use_networkx
        
        if use_networkx:
            self.graph = nx.MultiDiGraph()  
        
        self.user_index = {}
        self.item_index = {}
        self.task_index = defaultdict(list)
        
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s")
        self.logger = logging.getLogger(__name__)
    
    def connect(self):
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()
        self.create_schema()
        self.logger.info(f"Connected to knowledge graph database: {self.db_path}")
    
    def create_schema(self):
        
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                task TEXT,
                attributes TEXT,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                relationship_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                task TEXT,
                timestamp TEXT,
                weight REAL DEFAULT 1.0,
                attributes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_id) REFERENCES entities(entity_id),
                FOREIGN KEY (target_id) REFERENCES entities(entity_id)
            )
        """)
        
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(entity_type)
        """)
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_entity_task ON entities(task)
        """)
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id)
        """)
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id)
        """)
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rel_type ON relationships(relationship_type)
        """)
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rel_timestamp ON relationships(timestamp)
        """)
        
        self.connection.commit()
        self.logger.info("Schema created successfully")
    
    def add_entity(self, entity_id, entity_type, task=None, attributes= None, embedding= None):
        attrs_json = json.dumps(attributes) if attributes else None
        emb_blob = pickle.dumps(embedding) if embedding is not None else None
        
        self.cursor.execute("""
            INSERT OR REPLACE INTO entities 
            (entity_id, entity_type, task, attributes, embedding)
            VALUES (?, ?, ?, ?, ?)
        """, (entity_id, entity_type, task, attrs_json, emb_blob))
        
        if self.use_networkx:
            self.graph.add_node(
                entity_id,
                entity_type=entity_type,
                task=task,
                attributes=attributes or {},
                embedding=embedding
            )
        
        if entity_type == 'user':
            self.user_index[entity_id] = attributes or {}
        elif entity_type == 'item':
            self.item_index[entity_id] = attributes or {}
        
        if task:
            self.task_index[task].append(entity_id)
    
    def add_relationship(self, source_id, target_id, relationship_type, task=None, timestamp=None, weight=1.0, attributes=None):
        attrs_json = json.dumps(attributes) if attributes else None
        
        self.cursor.execute("""
            INSERT INTO relationships 
            (source_id, target_id, relationship_type, task, timestamp, weight, attributes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (source_id, target_id, relationship_type, task, timestamp, weight, attrs_json))
        
        if self.use_networkx:
            self.graph.add_edge(
                source_id,
                target_id,
                relationship_type=relationship_type,
                task=task,
                timestamp=timestamp,
                weight=weight,
                attributes=attributes or {}
            )
    
    def get_entity(self, entity_id):
        self.cursor.execute("""
            SELECT entity_id, entity_type, task, attributes, embedding
            FROM entities WHERE entity_id = ?
        """, (entity_id,))
        
        row = self.cursor.fetchone()
        if row:
            return {
                'entity_id': row[0],
                'entity_type': row[1],
                'task': row[2],
                'attributes': json.loads(row[3]) if row[3] else {},
                'embedding': pickle.loads(row[4]) if row[4] else None
            }
        return None
    
    def get_neighbors(self, entity_id, relationship_type=None, direction = 'outgoing'):
        if direction == 'outgoing':
            query = """
                SELECT r.target_id, r.relationship_type, r.timestamp, r.weight, r.attributes
                FROM relationships r
                WHERE r.source_id = ?
            """
        else:  # incoming 
            query = """
                SELECT r.source_id, r.relationship_type, r.timestamp, r.weight, r.attributes
                FROM relationships r
                WHERE r.target_id = ?
            """
        
        if relationship_type:
            query += " AND r.relationship_type = ?"
            self.cursor.execute(query, (entity_id, relationship_type))
        else:
            self.cursor.execute(query, (entity_id,))
        
        neighbors = []
        for row in self.cursor.fetchall():
            neighbors.append({
                'neighbor_id': row[0],
                'relationship_type': row[1],
                'timestamp': row[2],
                'weight': row[3],
                'attributes': json.loads(row[4]) if row[4] else {}
            })
        
        return neighbors
    
    def get_user_profile(self, user_id, task=None, limit=None,time_ordered = True):
        query = """
            SELECT r.target_id, r.relationship_type, r.timestamp, r.weight, r.attributes, r.task
            FROM relationships r
            WHERE r.source_id = ? AND r.relationship_type = 'interacted_with'
        """
        params = [user_id]
        
        if task:
            query += " AND r.task = ?"
            params.append(task)
        
        if time_ordered:
            query += " ORDER BY r.timestamp DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        self.cursor.execute(query, params)
        
        profile = []
        for row in self.cursor.fetchall():
            profile.append({
                'item_id': row[0],
                'relationship_type': row[1],
                'timestamp': row[2],
                'weight': row[3],
                'attributes': json.loads(row[4]) if row[4] else {},
                'task': row[5]
            })
        
        return profile
    
    def find_similar_users(self, user_id, k=10, task=None):
        user_items = set(n['neighbor_id'] for n in self.get_neighbors(user_id))
        
        self.cursor.execute("""
            SELECT DISTINCT entity_id FROM entities WHERE entity_type = 'user'
        """)
        all_users = [row[0] for row in self.cursor.fetchall() if row[0] != user_id]
        
        similarities = []
        for other_user in all_users:
            other_items = set(n['neighbor_id'] for n in self.get_neighbors(other_user))
            
            if len(user_items) > 0 and len(other_items) > 0:
                intersection = len(user_items & other_items)
                union = len(user_items | other_items)
                similarity = intersection / union if union > 0 else 0
                
                if similarity > 0:
                    similarities.append((other_user, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def graph_traversal(self, start_id, max_depth=2, relationship_types=None):
        if not self.use_networkx:
            raise ValueError("NetworkX must be enabled for graph traversal")
        
        #finally used a data structure algorithm (bfs)        
        visited = {start_id: {'depth': 0, 'path': [start_id]}}
        queue = [(start_id, 0, [start_id])]
        
        while queue:
            current_id, depth, path = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            neighbors = self.graph.successors(current_id)
            
            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    edges = self.graph.get_edge_data(current_id, neighbor_id)
                    
                    if relationship_types:
                        valid_edge = any(
                            edge_data.get('relationship_type') in relationship_types
                            for edge_data in edges.values()
                        )
                        if not valid_edge:
                            continue
                    
                    new_path = path + [neighbor_id]
                    visited[neighbor_id] = {
                        'depth': depth + 1,
                        'path': new_path
                    }
                    queue.append((neighbor_id, depth + 1, new_path))
        
        return [
            {'entity_id': eid, **info}
            for eid, info in visited.items()
        ]
    
    def get_shortest_path(self, source_id, target_id):
        if not self.use_networkx:
            raise ValueError("NetworkX must be enabled for path finding")
        
        try:
            path = nx.shortest_path(self.graph, source_id, target_id)
            return path
        except nx.NetworkXNoPath:
            return None
    
    def compute_centrality(self, centrality_type = 'degree'):
        if not self.use_networkx:
            raise ValueError("NetworkX must be enabled for centrality computation")
        
        if centrality_type == 'degree':
            return nx.degree_centrality(self.graph)
        elif centrality_type == 'betweenness':
            return nx.betweenness_centrality(self.graph)
        elif centrality_type == 'closeness':
            return nx.closeness_centrality(self.graph)
        elif centrality_type == 'pagerank':
            return nx.pagerank(self.graph)
        else:
            raise ValueError(f"Unknown centrality type: {centrality_type}")
    
    def get_statistics(self) :
        stats = {}
        
        self.cursor.execute("SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type")
        stats['entity_counts'] = {row[0]: row[1] for row in self.cursor.fetchall()}
        
        self.cursor.execute("SELECT relationship_type, COUNT(*) FROM relationships GROUP BY relationship_type")
        stats['relationship_counts'] = {row[0]: row[1] for row in self.cursor.fetchall()}
        
        self.cursor.execute("SELECT task, COUNT(*) FROM entities WHERE task IS NOT NULL GROUP BY task")
        stats['task_distribution'] = {row[0]: row[1] for row in self.cursor.fetchall()}
        
        if self.use_networkx and len(self.graph) > 0:
            stats['graph_stats'] = {
                'num_nodes': self.graph.number_of_nodes(),
                'num_edges': self.graph.number_of_edges(),
                'density': nx.density(self.graph),
                'is_connected': nx.is_weakly_connected(self.graph)
            }
        
        return stats
    
    def commit(self):
        if self.connection:
            self.connection.commit()
    
    def close(self):
        if self.connection:
            self.connection.close()
            self.logger.info("Database connection closed")
    
    def save_graph(self, filepath):
        if self.use_networkx:
            nx.write_gpickle(self.graph, filepath)
            self.logger.info(f"Graph saved to {filepath}")
    
    def load_graph(self, filepath):
        if self.use_networkx:
            self.graph = nx.read_gpickle(filepath)
            self.logger.info(f"Graph loaded from {filepath}")


class KnowledgeGraphBuilder:
    def __init__(self, kg , data_root = "../data"):
        self.kg = kg
        self.data_root = Path(data_root)
        self.logger = logging.getLogger(__name__)
    
    def build_from_dataset(self, tasks, max_samples_per_task=None):
        self.logger.info("Starting knowledge graph construction")
        
        for task in tasks:
            self.logger.info(f"Processing task: {task}")
            
            train_path = self.data_root / task / 'Train' / 'Inputs' / 'train_questions_time_based.json'
            
            if not train_path.exists():
                self.logger.warning(f"Training data not found for {task}")
                continue
            
            try:
                with open(train_path, 'r') as f:
                    data = json.load(f)
                
                if max_samples_per_task:
                    data = data[:max_samples_per_task]
                
                self.process_task_data(data, task)
                
            except Exception as e:
                self.logger.error(f"Error processing {task}: {e}")
        
        self.kg.commit()
        self.logger.info("Knowledge graph construction complete")
        
        stats = self.kg.get_statistics()
        self.logger.info(f"Knowledge Graph Statistics: {json.dumps(stats, indent=2)}")
    
    def process_task_data(self, data, task):
        for item in data:
            user_id = str(item.get('user_id', 'unknown'))
            self.kg.add_entity(
                entity_id=f"user_{user_id}",
                entity_type='user',
                task=task,
                attributes={'user_id': user_id}
            )
            
            item_id = str(item.get('id', ''))
            if item_id:
                self.kg.add_entity(
                    entity_id=f"item_{item_id}",
                    entity_type='item',
                    task=task,
                    attributes={
                        'item_id': item_id,
                        'input': item.get('input', '')
                    }
                )
                
                self.kg.add_relationship(
                    source_id=f"user_{user_id}",
                    target_id=f"item_{item_id}",
                    relationship_type='queries',
                    task=task
                )
            
            if 'profile' in item and item['profile']:
                for profile_item in item['profile']:
                    profile_item_id = str(profile_item.get('id', profile_item.get('text', '')))
                    
                    if profile_item_id:
                        self.kg.add_entity(
                            entity_id=f"item_{profile_item_id}",
                            entity_type='item',
                            task=task,
                            attributes=profile_item
                        )

                        self.kg.add_relationship(
                            source_id=f"user_{user_id}",
                            target_id=f"item_{profile_item_id}",
                            relationship_type='interacted_with',
                            task=task,
                            timestamp=profile_item.get('date'),
                            attributes=profile_item
                        )


# class KBDatabase(KnowledgeGraph):
#     pass
