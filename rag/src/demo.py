import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from kg_db import KnowledgeGraph, KnowledgeGraphBuilder
from rag_systems import create_rag_systems
from evaluation import BenchmarkSuite, RAGEvaluator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format='[%(asctime)s] %(levelname)s - %(message)s')

def load_sample_data(data_root, task, max_samples=100):
    train_path = data_root / task / 'Train' / 'Inputs' / 'train_questions_time_based.json'
    
    if not train_path.exists():
        logger.warning(f"Data not found for {task}")
        return []
    
    try:
        with open(train_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} samples from {task}")
        return data[:max_samples]
    
    except Exception as e:
        logger.error(f"Error loading {task}: {e}")
        return []


def demo_knowledge_graph(data_root):
    logger.info("="*80)
    logger.info("Demo 1: Knowledge Graph Construction")
    logger.info("="*80)
    
    kg = KnowledgeGraph(db_path="../knowledge_graph.db", use_networkx=True)
    kg.connect()


    builder = KnowledgeGraphBuilder(kg, data_root=str(data_root))
    
    tasks = ['Movie_Tagging', 'Product_Rating']
    
    for task in tasks:
        data = load_sample_data(data_root, task, max_samples=50)
        if data:
            logger.info(f"\nProcessing {task}")
            builder.process_task_data(data, task)
    
    kg.commit()
    
    stats = kg.get_statistics()
    logger.info("\nKnowledge Graph Statistics:")
    logger.info(json.dumps(stats, indent=2))
    
    logger.info("\n" + "="*80)
    logger.info("Example: Get user profile")
    logger.info("="*80)
    
    kg.cursor.execute("SELECT entity_id FROM entities WHERE entity_type='user' LIMIT 1")
    result = kg.cursor.fetchone()
    
    if result:
        user_id = result[0]
        logger.info(f"User ID: {user_id}")
        
        profile = kg.get_user_profile(user_id, limit=5)
        logger.info(f"Profile length: {len(profile)} interactions")
        
        if profile:
            logger.info("First interaction:")
            logger.info(json.dumps(profile[0], indent=2, default=str))
    
    if result:
        logger.info("\n" + "-"*80)
        logger.info("Example: Find similar users")
        logger.info("-"*80)
        
        similar_users = kg.find_similar_users(user_id, k=3)
        logger.info(f"Found {len(similar_users)} similar users:")
        for sim_user, similarity in similar_users:
            logger.info(f"  {sim_user}: similarity={similarity:.3f}")
    
    return kg


def demo_rag_systems(kg, data_root):
    logger.info("\n" + "="*80)
    logger.info("DEMO 2: RAG Systems")
    logger.info("="*80)
    
    logger.info("\nCreating RAG systems")
    systems = create_rag_systems(kg)
    
    logger.info(f"Created {len(systems)} RAG systems:")
    for name in systems.keys():
        logger.info(f"  - {name}")
    
    task = 'Movie_Tagging'
    data = load_sample_data(data_root, task, max_samples=100)
    
    if not data:
        logger.warning("No data available for demo")
        return systems
    
    logger.info(f"\nIndexing data from {task}")
    
    for name, system in systems.items():
        logger.info(f"\nIndexing in {name}")
        try:
            system.index(data, task=task)
            logger.info(f"\t{name} indexed successfully")
        except Exception as e:
            logger.error(f"\tError indexing {name}: {e}")
    
    logger.info("\n" + "-"*80)
    logger.info("Example: Query RAG systems")
    logger.info("-"*80)
    
    if data:
        sample_query = {
            'id': 'demo_query',
            'input': 'A movie about time travel and paradoxes',
            'user_id': data[0].get('user_id'),
            'task': task
        }
        
        logger.info(f"\nQuery: {sample_query['input']}")
        logger.info("-"*40)
        
        for name, system in systems.items():
            logger.info(f"\n{name} Results:")
            try:
                results = system.retrieve(sample_query, k=5)
                logger.info(f"Retrieved {len(results)} items:")
                
                for i, result in enumerate(results[:3], 1):
                    item = result['item']
                    score = result.get('score', 0)
                    logger.info(f"\n  {i}. Score: {score:.4f}")
                    logger.info(f"     Tag: {item.get('tag', 'N/A')}")
                    logger.info(f"     Description: {item.get('description', 'N/A')[:100]}")
                
            except Exception as e:
                logger.error(f"  Error: {e}")
    
    return systems


def demo_evaluation(systems, data_root):
    logger.info("\n" + "="*80)
    logger.info("DEMO 3: Evaluation and Benchmarking")
    logger.info("="*80)
    
    task = 'Movie_Tagging'
    test_path = data_root / task / 'Test' / 'test_questions_time_based.json'
    
    if not test_path.exists():
        logger.warning(f"Test data not found for {task}")
        return
    
    logger.info(f"\nLoading test data from {task}")
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    
    test_data = test_data[:20] 
    logger.info(f"Loaded {len(test_data)} test samples")
    
    queries = []
    retrieval_gt = []
    generation_gt = []
    
    for item in test_data:
        queries.append(item)
        
        if 'profile' in item and item['profile']:
            retrieval_gt.append([str(p.get('id')) for p in item['profile']])
        else:
            retrieval_gt.append([])
        
        if 'profile' in item and item['profile']:
            generation_gt.append(item['profile'][0].get('tag', ''))
        else:
            generation_gt.append('')
    
    logger.info("\nRunning benchmark")
    benchmark = BenchmarkSuite(output_dir="../outputs/benchmarks")
    
    try:
        results = benchmark.compare_systems(
            systems=systems,
            queries=queries,
            retrieval_ground_truth=retrieval_gt,
            generation_ground_truth=generation_gt,
            task=task,
            total_items=1000,  # Estimate
            k=10
        )
        
        logger.info("\nBenchmark complete :)")
        logger.info(f"Results saved to: ../outputs/benchmarks/")
        
    except Exception as e:
        logger.error(f"Benchmark error: {e}")
        import traceback
        traceback.print_exc()


def main():
    logger.info("="*80)
    logger.info("Rag Systtem Demo")
    logger.info("="*80)
    
    data_root = Path(__file__).parent.parent / 'data'
    
    if not data_root.exists():
        logger.error(f"Data directory not found: {data_root}")
        return
    
    # d 1: kg
    kg = demo_knowledge_graph(data_root)
    
    # d 2: rag system
    systems = demo_rag_systems(kg, data_root)
    
    # d 3: eval
    if systems:
        demo_evaluation(systems, data_root)
    
    kg.close()


if __name__ == '__main__':
    main()
