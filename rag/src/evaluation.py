import numpy as np
import time
import json
from pathlib import Path
from collections import defaultdict
import logging

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge-score not installed. Install with: pip install rouge-score")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrievalMetrics:
    @staticmethod
    def precision_at_k(retrieved, relevant, k):
        if k == 0 or len(retrieved) == 0:
            return 0.0
        
        retrieved_k = set(retrieved[:k])
        relevant_set = set(relevant)
        
        return len(retrieved_k & relevant_set) / k
    
    @staticmethod
    def recall_at_k(retrieved, relevant, k) :
        if len(relevant) == 0:
            return 0.0
        
        retrieved_k = set(retrieved[:k])
        relevant_set = set(relevant)
        
        return len(retrieved_k & relevant_set) / len(relevant_set)
    
    @staticmethod
    def average_precision(retrieved, relevant):
        if len(relevant) == 0:
            return 0.0
        
        relevant_set = set(relevant)
        precisions = []
        num_relevant = 0
        
        for i, item in enumerate(retrieved):
            if item in relevant_set:
                num_relevant += 1
                precision = num_relevant / (i + 1)
                precisions.append(precision)
        
        return np.mean(precisions) if precisions else 0.0
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_lists, relevant_lists):
        reciprocal_ranks = []
        
        for retrieved, relevant in zip(retrieved_lists, relevant_lists):
            relevant_set = set(relevant)
            
            for i, item in enumerate(retrieved):
                if item in relevant_set:
                    reciprocal_ranks.append(1.0 / (i + 1))
                    break
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks)
    
    @staticmethod
    def ndcg_at_k(retrieved, relevant, k, relevance_scores=None) :
        if k == 0 or len(retrieved) == 0:
            return 0.0
        
        if relevance_scores is None:
            relevance_scores = {item: 1.0 for item in relevant}
        
        dcg = 0.0
        for i, item in enumerate(retrieved[:k]):
            relevance = relevance_scores.get(item, 0.0)
            dcg += relevance / np.log2(i + 2)
        
        sorted_relevant = sorted(relevant, key=lambda x: relevance_scores.get(x, 0.0), reverse=True)
        idcg = 0.0
        for i, item in enumerate(sorted_relevant[:k]):
            relevance = relevance_scores.get(item, 1.0)
            idcg += relevance / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0


class GenerationMetrics:
    def __init__(self):
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], 
                use_stemmer=True
            )
        else:
            self.rouge_scorer = None
    
    def rouge_scores(self, prediction, reference):
        if not self.rouge_scorer:
            raise ImportError("rouge-score required for ROUGE metrics")
        
        scores = self.rouge_scorer.score(reference, prediction)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    @staticmethod
    def exact_match(prediction, reference) :
        return 1.0 if prediction == reference else 0.0
    
    @staticmethod
    def mae(predictions, references) :
        return np.mean(np.abs(np.array(predictions) - np.array(references)))
    
    @staticmethod
    def rmse(predictions, references) :
        return np.sqrt(np.mean((np.array(predictions) - np.array(references)) ** 2))


class DiversityMetrics:
    @staticmethod
    def coverage(retrieved_lists, total_items) :
        all_retrieved = set()
        for retrieved in retrieved_lists:
            all_retrieved.update(retrieved)
        
        return len(all_retrieved) / total_items if total_items > 0 else 0.0
    
    @staticmethod
    def intra_list_diversity(retrieved, similarity_matrix=None) :
        if len(retrieved) <= 1:
            return 1.0
        
        if similarity_matrix is not None:
            indices = [int(item_id) for item_id in retrieved if item_id.isdigit()]
            if len(indices) <= 1:
                return 1.0
            
            total_dissimilarity = 0.0
            count = 0
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    total_dissimilarity += (1 - similarity_matrix[indices[i], indices[j]])
                    count += 1
            
            return total_dissimilarity / count if count > 0 else 1.0
        else:
            return len(set(retrieved)) / len(retrieved)
    
    @staticmethod
    def novelty(retrieved, popular_items):
        if len(retrieved) == 0:
            return 0.0
        
        novel_count = sum(1 for item in retrieved if item not in popular_items)
        return novel_count / len(retrieved)


class EfficiencyMetrics:
    @staticmethod
    def measure_latency(func, *args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        
        latency_ms = (end - start) * 1000
        return result, latency_ms
    
    @staticmethod
    def measure_throughput(func, queries, **kwargs) :
        start = time.time()
        
        for query in queries:
            func(query, **kwargs)
        
        end = time.time()
        duration = end - start
        
        return len(queries) / duration if duration > 0 else 0.0


class RAGEvaluator:
    def __init__(self, rag_system, name="RAG"):
        self.rag_system = rag_system
        self.name = name
        self.retrieval_metrics = RetrievalMetrics()
        self.generation_metrics = GenerationMetrics()
        self.diversity_metrics = DiversityMetrics()
        self.efficiency_metrics = EfficiencyMetrics()
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    def evaluate_retrieval(self, queries, ground_truth, k=10):
        self.logger.info(f"Evaluating retrieval on {len(queries)} queries")
        
        precisions = []
        recalls = []
        aps = []
        retrieved_lists = []
        latencies = []
        
        for query, relevant in zip(queries, ground_truth):
            retrieved, latency = self.efficiency_metrics.measure_latency(self.rag_system.retrieve, query, k=k)
            latencies.append(latency)
            
            retrieved_ids = [r['item'].get('id', str(r['item'])) for r in retrieved]
            retrieved_lists.append(retrieved_ids)
            precisions.append(self.retrieval_metrics.precision_at_k(retrieved_ids, relevant, k))
            recalls.append(
                self.retrieval_metrics.recall_at_k(retrieved_ids, relevant, k)
            )
            aps.append(
                self.retrieval_metrics.average_precision(retrieved_ids, relevant)
            )
        
        results = {
            f'precision@{k}': np.mean(precisions),
            f'recall@{k}': np.mean(recalls),
            'map': np.mean(aps),  #mean average pres
            'mrr': self.retrieval_metrics.mean_reciprocal_rank(
                retrieved_lists, ground_truth
            ),
            'avg_latency_ms': np.mean(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99)
        }
        
        return results
    
    def evaluate_generation(self, queries, ground_truth, task, k=10):
        self.logger.info(f"Evaluating generation on {len(queries)} queries")
        
        predictions = []
        references = []
        
        for query, reference in zip(queries, ground_truth):
            prediction, context = self.rag_system(query, k=k, task=task)
            
            predictions.append(prediction)
            references.append(reference)
        
        results = {}
        
        if task == 'Movie_Tagging':
            correct = sum(1 for p, r in zip(predictions, references) if p == r)
            results['accuracy'] = correct / len(predictions)
        
        elif task == 'Product_Rating':
            pred_floats = [float(p) if p is not None else 0.0 for p in predictions]
            ref_floats = [float(r) for r in references]
            results['mae'] = self.generation_metrics.mae(pred_floats, ref_floats)
            results['rmse'] = self.generation_metrics.rmse(pred_floats, ref_floats)
        
        elif task in ['Email_Subject_Generation', 'News_Headline_Generation','Scholarly_Title_Generation', 'Tweet_Paraphrasing']:
            if ROUGE_AVAILABLE:
                rouge_scores = []
                for pred, ref in zip(predictions, references):
                    if pred and ref:
                        scores = self.generation_metrics.rouge_scores(str(pred), str(ref))
                        rouge_scores.append(scores)
                
                if rouge_scores:
                    results['rouge1'] = np.mean([s['rouge1'] for s in rouge_scores])
                    results['rouge2'] = np.mean([s['rouge2'] for s in rouge_scores])
                    results['rougeL'] = np.mean([s['rougeL'] for s in rouge_scores])
        
        return results
    
    def evaluate_diversity(self, queries, total_items, popular_thresh=100, k=100) :
        self.logger.info(f"Evaluating diversity on {len(queries)} queries")
        
        retrieved_lists = []
        intra_diversities = []
        
        for query in queries:
            retrieved = self.rag_system.retrieve(query, k=k)
            retrieved_ids = [r['item'].get('id', str(r['item'])) for r in retrieved]
            retrieved_lists.append(retrieved_ids)
            
            diversity = self.diversity_metrics.intra_list_diversity(retrieved_ids)
            intra_diversities.append(diversity)
        
        results = {
            'coverage': self.diversity_metrics.coverage(retrieved_lists, total_items),
            'avg_intra_list_diversity': np.mean(intra_diversities)
        }
        
        return results
    
    def full_evaluation(self, queries,retrieval_ground_truth,generation_ground_truth,task, total_items, k=10):
        self.logger.info(f"Running full evaluation for {self.name}")
        
        results = {
            'system_name': self.name,
            'task': task,
            'num_queries': len(queries),
            'k': k
        }
        
        retrieval_results = self.evaluate_retrieval(queries, retrieval_ground_truth, k=k)
        results['retrieval'] = retrieval_results
        
        generation_results = self.evaluate_generation(queries, generation_ground_truth, task=task, k=k)
        results['generation'] = generation_results
        
        diversity_results = self.evaluate_diversity(queries, total_items, k=k)
        results['diversity'] = diversity_results
        
        self.logger.info(f"Evaluation complete for {self.name}")
        return results


class BenchmarkSuite:
    def __init__(self, output_dir: str = "../outputs/benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def compare_systems(self, systems,queries,retrieval_ground_truth,generation_ground_truth,task,total_items ,k=10):
        self.logger.info(f"Benchmarking {len(systems)} systems on {task}")
        
        results = {}
        
        for name, system in systems.items():
            evaluator = RAGEvaluator(system, name=name)
            
            try:
                system_results = evaluator.full_evaluation(queries=queries,retrieval_ground_truth=retrieval_ground_truth,generation_ground_truth=generation_ground_truth,task=task,total_items=total_items,k=k)
                results[name] = system_results
                
            except Exception as e:
                self.logger.error(f"Error evaluating {name}: {e}")
                results[name] = {'error': str(e)}

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"benchmark_{task}_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Benchmark results saved to {output_file}")
        
        self.print_comparison(results)
        
        return results
    
    def print_comparison(self, results):

        print("\n" + "="*80)
        print("Benchmarks results complete yay")
        print("="*80)


        print("\nRetrival Metrics:")
        print("-" * 80)
        print(f"{'System':<20} {'Precision@K':<12} {'Recall@K':<12} {'MAP':<12} {'MRR':<12}")
        print("-" * 80)
        
        for name, result in results.items():
            if 'error' in result:
                print(f"{name:<20} ERROR: {result['error']}")
                continue
            
            ret = result.get('retrieval', {})
            k = result.get('k', 10)
            print(f"{name:<20} "
                  f"{ret.get(f'precision@{k}', 0):<12.4f} "
                  f"{ret.get(f'recall@{k}', 0):<12.4f} "
                  f"{ret.get('map', 0):<12.4f} "
                  f"{ret.get('mrr', 0):<12.4f}")
        
        print("\n\nGeneration metrics")
        print("-" * 80)
        
        metric_keys = set()
        for result in results.values():
            if 'generation' in result:
                metric_keys.update(result['generation'].keys())
        
        if metric_keys:
            header = f"{'System':<20} " + " ".join(f"{k:<12}" for k in sorted(metric_keys))
            print(header)
            print("-" * 80)
            
            for name, result in results.items():
                if 'error' in result:
                    continue
                
                gen = result.get('generation', {})
                row = f"{name:<20} "
                for key in sorted(metric_keys):
                    row += f"{gen.get(key, 0):<12.4f} "
                print(row)
    
        print("\n\nEfficiency metrics:")
        print("-" * 80)
        print(f"{'System':<20} {'Avg Latency (ms)':<20} {'P95 Latency (ms)':<20}")
        print("-" * 80)
        
        for name, result in results.items():
            if 'error' in result:
                continue
            
            ret = result.get('retrieval', {})
            print(f"{name:<20} "
                  f"{ret.get('avg_latency_ms', 0):<20.2f} "
                  f"{ret.get('p95_latency_ms', 0):<20.2f}")
        
        print("\n" + "="*80)
