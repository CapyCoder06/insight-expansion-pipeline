"""
Retrieval Evaluation Module

Tests document retrieval quality BEFORE embedding.
Simulates retrieval using semantic similarity on raw document text.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
import json
from .validator import semantic_similarity
from .config import get_config
from .metadata_retrieval import QueryExpander, MetadataQueryMatcher, HybridScorer


class RetrievalEvaluator:
    """
    Evaluates retrieval quality of documents using semantic similarity.

    This module simulates vector retrieval by comparing query-document similarity
    on the raw text level. It helps catch poor document quality or missing content
    BEFORE the expensive embedding and vector DB upload step.

    Metrics:
    - Recall@k: fraction of relevant documents found in top-k results
    - Precision@k: fraction of retrieved documents that are relevant
    - MRR (Mean Reciprocal Rank): average of 1/rank of first relevant doc
    - Relevance Score: average semantic similarity of retrieved docs
    """

    def __init__(self, documents: List[Dict[str, Any]],
                 config: Optional[Any] = None):
        """
        Initialize evaluator.

        Args:
            documents: List of document dicts with 'text' and 'metadata'
            config: Configuration object (loads defaults if None)
        """
        self.documents = documents
        self.config = config if config is not None else get_config()
        self.top_k_default = self.config.get('evaluation.top_k', [1, 3, 5, 10])

        # Build document index: doc_id -> document
        self.doc_index = {i: doc for i, doc in enumerate(documents)}

        # Extract searchable text from each document (remove headers for matching? keep all)
        self.searchable_texts = []
        for doc in documents:
            text = doc.get('text', '')
            # Optional: strip markdown headers to focus on content
            # But keep for now to test full document retrieval
            self.searchable_texts.append(text)

    def evaluate_query(self,
                      query: str,
                      relevant_doc_ids: Set[int],
                      top_k: Optional[List[int]] = None,
                      return_details: bool = False) -> Dict[str, Any]:
        """
        Evaluate retrieval for a single query.

        Args:
            query: The search query string
            relevant_doc_ids: Set of document IDs that should be retrieved
            top_k: List of k values to compute metrics for (default: [1,3,5,10])
            return_details: If True, include retrieved list and scores

        Returns:
            Dictionary with metrics:
            - recall@k: values for each k
            - precision@k: values for each k
            - mrr: Mean Reciprocal Rank
            - relevance_scores: dict with avg_similarity@k
        """
        if top_k is None:
            top_k = self.top_k_default

        # Compute similarity scores for all documents
        scores = []
        for doc_id, text in enumerate(self.searchable_texts):
            sim = semantic_similarity(query, text)
            scores.append((doc_id, sim))

        # Sort by similarity descending
        scores.sort(key=lambda x: x[1], reverse=True)

        results = {
            'query': query,
            'num_relevant': len(relevant_doc_ids),
        }

        # Compute metrics for each k
        retrieved_ids = []
        retrieved_scores = {}
        recall = {}
        precision = {}
        relevant_retrieved = set()

        for k in top_k:
            # Get top-k retrieved IDs
            topk_ids = [doc_id for doc_id, _ in scores[:k]]
            retrieved_scores[k] = {
                'avg_similarity': self._avg_similarity_at_k(scores, k),
                'min_similarity': scores[k-1][1] if k <= len(scores) else 0.0
            }

            # Calculate recall@k
            retrieved_set = set(topk_ids)
            relevant_retrieved_k = retrieved_set & relevant_doc_ids
            recall_k = len(relevant_retrieved_k) / len(relevant_doc_ids) if relevant_doc_ids else 0.0
            recall[k] = round(recall_k, 3)
            precision[k] = round(len(relevant_retrieved_k) / k, 3) if k > 0 else 0.0

            # Track first relevant doc for MRR
            for rank, (doc_id, _) in enumerate(scores[:k], start=1):
                if doc_id in relevant_doc_ids and doc_id not in retrieved_ids:
                    retrieved_ids.append(doc_id)
                    # Store first relevant rank
                    results['first_relevant_rank'] = rank
                    break

        # MRR: Mean Reciprocal Rank (single query = 1/rank if found, else 0)
        mrr = 0.0
        if 'first_relevant_rank' in results:
            mrr = 1.0 / results['first_relevant_rank']
        results['mrr'] = round(mrr, 3)

        results['recall@k'] = recall
        results['precision@k'] = precision
        results['relevance_scores'] = {k: round(v['avg_similarity'], 3)
                                       for k, v in retrieved_scores.items()}

        if return_details:
            results['retrieved'] = [
                {
                    'doc_id': doc_id,
                    'score': round(score, 3),
                    'metadata': self.doc_index[doc_id].get('metadata', {}),
                    'text_preview': self.doc_index[doc_id].get('text', '')[:200]
                }
                for doc_id, score in scores[:max(top_k)]
            ]

        return results

    def evaluate_dataset(self,
                         test_queries: List[Dict[str, Any]],
                         top_k: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Evaluate multiple queries and compute aggregate metrics.

        Args:
            test_queries: List of dicts with keys:
                - query: str
                - relevant_doc_ids: Set[int] or List[int]
                - description: optional str
            top_k: List of k values for metrics

        Returns:
            Dictionary with per-query and aggregate metrics.
        """
        all_results = []
        for q_info in test_queries:
            query = q_info['query']
            relevant = set(q_info['relevant_doc_ids'])
            result = self.evaluate_query(query, relevant, top_k=top_k)
            result['description'] = q_info.get('description', '')
            all_results.append(result)

        # Compute aggregates
        aggregates = {}
        if all_results:
            for k in all_results[0]['recall@k'].keys():
                recalls = [r['recall@k'][k] for r in all_results]
                precisions = [r['precision@k'][k] for r in all_results]
                aggregates[f'avg_recall@{k}'] = round(sum(recalls) / len(recalls), 3)
                aggregates[f'avg_precision@{k}'] = round(sum(precisions) / len(precisions), 3)

            # Average MRR
            mrrs = [r['mrr'] for r in all_results]
            aggregates['avg_mrr'] = round(sum(mrrs) / len(mrrs), 3)

            # Success rates (recall@k > 0)
            for k in all_results[0]['recall@k'].keys():
                success_count = sum(1 for r in all_results if r['recall@k'][k] > 0)
                aggregates[f'queries_with_hits@{k}'] = f"{success_count}/{len(all_results)}"

        return {
            'per_query': all_results,
            'aggregates': aggregates,
            'total_queries': len(all_results)
        }

    def _avg_similarity_at_k(self, scores: List[Tuple[int, float]], k: int) -> float:
        """Compute average similarity of top-k results."""
        top_k_scores = [score for _, score in scores[:k]]
        return sum(top_k_scores) / len(top_k_scores) if top_k_scores else 0.0

    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a human-readable report from evaluation results.

        Args:
            results: Output from evaluate_dataset()

        Returns:
            Formatted string report
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("RETRIEVAL EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"\nTotal Queries: {results['total_queries']}")
        report_lines.append("\n--- Aggregate Metrics ---")
        aggs = results.get('aggregates', {})

        for key, value in sorted(aggs.items()):
            report_lines.append(f"  {key}: {value}")

        report_lines.append("\n--- Per-Query Results ---")
        for i, q_result in enumerate(results['per_query'], start=1):
            report_lines.append(f"\n{i}. Query: {q_result['query']}")
            report_lines.append(f"   Relevant docs: {q_result['num_relevant']}")
            report_lines.append(f"   MRR: {q_result['mrr']}")
            for k in sorted(q_result['recall@k'].keys()):
                report_lines.append(f"   Recall@{k}: {q_result['recall@k'][k]} | Precision@{k}: {q_result['precision@k'][k]}")
            if q_result.get('description'):
                report_lines.append(f"   Note: {q_result['description']}")

        report_lines.append("\n" + "=" * 60)
        return "\n".join(report_lines)


class MetadataAwareRetrievalEvaluator(RetrievalEvaluator):
    """
    Retrieval evaluator that uses metadata for hybrid scoring:
    - Query expansion via global synonyms
    - Metadata query matching
    - Hybrid scoring with configurable weights
    """
    def __init__(self, documents: List[Dict[str, Any]],
                 config: Optional[Any] = None,
                 use_metadata: bool = True):
        super().__init__(documents, config)
        self.use_metadata = use_metadata
        if self.use_metadata:
            self.global_synonyms = self._build_global_synonym_map()
            self.query_expander = QueryExpander(self.global_synonyms)
            self.meta_matcher = MetadataQueryMatcher()
            # Get weights and bonus config from config
            weights = self.config.get('retrieval.weights', {
                'text': 0.4,
                'metadata_query': 0.5,
                'expanded': 0.1
            })
            bonus_config = self.config.get('retrieval.metadata_bonus', {
                'enabled': False,
                'threshold': 0.7,
                'boost_factor': 1.5
            })
            self.hybrid_scorer = HybridScorer(weights, bonus_config)

    def _build_global_synonym_map(self) -> Dict[str, List[str]]:
        """Collect synonyms from all documents' metadata."""
        global_synonyms = {}
        for doc in self.documents:
            meta = doc.get('metadata', {})
            doc_synonyms = meta.get('synonyms', {})
            for term, syns in doc_synonyms.items():
                term_lower = term.lower()
                if term_lower not in global_synonyms:
                    global_synonyms[term_lower] = []
                for syn in syns:
                    syn_lower = syn.lower()
                    if syn_lower not in global_synonyms[term_lower]:
                        global_synonyms[term_lower].append(syn_lower)
        return global_synonyms

    def evaluate_query(self,
                       query: str,
                       relevant_doc_ids: Set[int],
                       top_k: Optional[List[int]] = None,
                       return_details: bool = False) -> Dict[str, Any]:
        """Override to use hybrid scoring when metadata is enabled."""
        if not self.use_metadata:
            return super().evaluate_query(query, relevant_doc_ids, top_k, return_details)

        if top_k is None:
            top_k = self.top_k_default

        # Expand query globally once
        expanded_variants = self.query_expander.expand(query)

        # Score all documents with hybrid scorer
        scores = []
        for doc_id, doc in self.doc_index.items():
            score = self.hybrid_scorer.score(query, doc, expanded_variants, self.meta_matcher)
            scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = {
            'query': query,
            'num_relevant': len(relevant_doc_ids),
        }

        # Compute metrics for each k (same as parent)
        retrieved_ids = []
        retrieved_scores = {}
        recall = {}
        precision = {}
        relevant_retrieved = set()

        for k in top_k:
            topk_ids = [doc_id for doc_id, _ in scores[:k]]
            retrieved_scores[k] = {
                'avg_similarity': self._avg_similarity_at_k(scores, k),
                'min_similarity': scores[k-1][1] if k <= len(scores) else 0.0
            }
            retrieved_set = set(topk_ids)
            relevant_retrieved_k = retrieved_set & relevant_doc_ids
            recall_k = len(relevant_retrieved_k) / len(relevant_doc_ids) if relevant_doc_ids else 0.0
            recall[k] = round(recall_k, 3)
            precision[k] = round(len(relevant_retrieved_k) / k, 3) if k > 0 else 0.0
            for rank, (doc_id, _) in enumerate(scores[:k], start=1):
                if doc_id in relevant_doc_ids and doc_id not in retrieved_ids:
                    retrieved_ids.append(doc_id)
                    results['first_relevant_rank'] = rank
                    break

        mrr = 0.0
        if 'first_relevant_rank' in results:
            mrr = 1.0 / results['first_relevant_rank']
        results['mrr'] = round(mrr, 3)

        results['recall@k'] = recall
        results['precision@k'] = precision
        results['relevance_scores'] = {k: round(v['avg_similarity'], 3)
                                       for k, v in retrieved_scores.items()}

        if return_details:
            results['retrieved'] = [
                {
                    'doc_id': doc_id,
                    'score': round(score, 3),
                    'metadata': self.doc_index[doc_id].get('metadata', {}),
                    'text_preview': self.doc_index[doc_id].get('text', '')[:200]
                }
                for doc_id, score in scores[:max(top_k)]
            ]

        return results


def evaluate_retrieval(query: str,
                       documents: List[Dict[str, Any]],
                       relevant_doc_ids: Set[int],
                       config: Optional[Any] = None) -> Dict[str, Any]:
    """
    Convenience function: evaluate retrieval for a single query.

    Args:
        query: Search query string
        documents: List of document dicts
        relevant_doc_ids: Set of document IDs known to be relevant
        config: Optional configuration object

    Returns:
        Dictionary of metrics (recall@k, precision@k, mrr, relevance_scores)
    """
    evaluator = RetrievalEvaluator(documents, config=config)
    return evaluator.evaluate_query(query, relevant_doc_ids)


def quick_benchmark(documents: List[Dict[str, Any]],
                    test_queries: List[Dict[str, Any]],
                    config: Optional[Any] = None) -> Dict[str, Any]:
    """
    Quick benchmark function for use in notebooks/scripts.

    Args:
        documents: List of document dicts
        test_queries: List of {'query': str, 'relevant_doc_ids': Set[int]}
        config: Optional configuration object

    Returns:
        Evaluation results
    """
    evaluator = RetrievalEvaluator(documents, config=config)
    return evaluator.evaluate_dataset(test_queries)


def load_test_queries_from_json(path: str) -> List[Dict[str, Any]]:
    """
    Load test queries from a JSON file.

    Expected format:
    [
      {
        "query": "Which category has highest profit?",
        "relevant_doc_ids": [0, 3],
        "description": "should retrieve fact about Technology category"
      },
      ...
    ]

    Args:
        path: Path to JSON file

    Returns:
        List of test query dicts
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def save_evaluation_results(results: Dict[str, Any], path: str):
    """
    Save evaluation results to JSON file.

    Args:
        results: Evaluation results dict
        path: Output file path
    """
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
