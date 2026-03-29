#!/usr/bin/env python
"""
Comparison Evaluation: Enrichment vs Baseline

Tests whether query-aware enrichment improves retrieval recall.
"""

import sys
from pathlib import Path
import json
from typing import List, Dict, Any

sys.path.insert(0, 'src')

from pipeline import DocumentPipeline, generate_from_insights
from pipeline.evaluator import RetrievalEvaluator, save_evaluation_results
from pipeline.config import load_config
from pipeline.enrichment import DocumentEnricher, select_top_variations


def load_test_queries(path: str) -> List[Dict[str, Any]]:
    """Load test queries from JSON."""
    with open(path, 'r') as f:
        return json.load(f)


def generate_documents(insights: List[Dict[str, Any]], enrich: bool, config=None) -> List[Dict[str, Any]]:
    """Generate documents with enrichment on or off."""
    pipeline = DocumentPipeline(enrich=enrich, config=config)
    # Use run with validation and chunking disabled to get documents with enrichment applied
    result = pipeline.run(insights, validate=False, chunk=False)
    return result['documents']


def _generate_enriched_with_mapping(insights: List[Dict[str, Any]], config=None):
    """
    Generate enriched documents and mapping from insight index to document indices.

    Returns:
        (enriched_docs, mapping) where mapping[i] = list of doc indices for insight i.
    """
    if config is None:
        config = load_config()

    enrichment_config = config.get('enrichment', {})
    enrich_variations = enrichment_config.get('variations_to_keep', 3)
    variations_to_generate = enrichment_config.get('variations_to_generate', 8)

    enricher = DocumentEnricher(num_variations=variations_to_generate, config=config)

    # Generate baseline documents first (one per insight)
    baseline_docs = generate_from_insights(insights)

    # Enrich each baseline doc and collect mapping
    enriched_docs = []
    mapping = []  # list of lists, index i -> list of enriched doc indices for insight i
    for i, doc in enumerate(baseline_docs):
        variations = enricher.expand_document(doc)
        top_vars = select_top_variations(variations, top_k=enrich_variations)
        start_idx = len(enriched_docs)
        enriched_docs.extend(top_vars)
        end_idx = len(enriched_docs)
        mapping.append(list(range(start_idx, end_idx)))

    return enriched_docs, mapping


def _transform_queries_for_enriched(test_queries: List[Dict[str, Any]], insight_to_enriched_indices: List[List[int]]) -> List[Dict[str, Any]]:
    """Transform baseline query relevance to enriched query relevance using mapping."""
    transformed = []
    for query in test_queries:
        new_query = query.copy()
        baseline_ids = query.get('relevant_doc_ids', [])
        # Map each baseline insight index to all its enriched variations
        enriched_relevant = set()
        for idx in baseline_ids:
            if 0 <= idx < len(insight_to_enriched_indices):
                enriched_relevant.update(insight_to_enriched_indices[idx])
        new_query['relevant_doc_ids'] = list(enriched_relevant)
        transformed.append(new_query)
    return transformed


def run_comparison_evaluation(insights_path: str, queries_path: str, output_dir: str = "output") -> Dict[str, Any]:
    """
    Run comparative evaluation: baseline vs enrichment.

    Returns:
        Dictionary with both result sets and comparison metrics
    """
    # Load data
    with open(insights_path) as f:
        insights = json.load(f)
    test_queries = load_test_queries(queries_path)

    results = {
        'baseline': None,
        'enriched': None,
        'comparison': {}
    }

    # --- Baseline (no enrichment) ---
    print("\n" + "="*60)
    print("BASELINE EVALUATION (No Enrichment)")
    print("="*60)
    baseline_docs = generate_documents(insights, enrich=False)
    print(f"Generated {len(baseline_docs)} documents")

    baseline_evaluator = RetrievalEvaluator(baseline_docs)
    results['baseline'] = baseline_evaluator.evaluate_dataset(test_queries)

    print("\nBaseline aggregates:")
    for k, v in results['baseline']['aggregates'].items():
        print(f"  {k}: {v}")

    # --- Enriched ---
    print("\n" + "="*60)
    print("ENRICHED EVALUATION (With Query-Aware Variations)")
    print("="*60)
    # Generate enriched with mapping
    enriched_docs, insight_to_enriched_indices = _generate_enriched_with_mapping(insights)
    print(f"Generated {len(enriched_docs)} documents")

    # Transform test queries to use enriched doc indices
    enriched_queries = _transform_queries_for_enriched(test_queries, insight_to_enriched_indices)

    enriched_evaluator = RetrievalEvaluator(enriched_docs)
    results['enriched'] = enriched_evaluator.evaluate_dataset(enriched_queries)

    print("\nEnriched aggregates:")
    for k, v in results['enriched']['aggregates'].items():
        print(f"  {k}: {v}")

    # --- Comparison ---
    print("\n" + "="*60)
    print("COMPARISON ANALYSIS")
    print("="*60)

    baseline_aggs = results['baseline']['aggregates']
    enriched_aggs = results['enriched']['aggregates']

    comparison = {}
    for metric in ['avg_recall@1', 'avg_recall@3', 'avg_recall@5', 'avg_recall@10',
                   'avg_precision@1', 'avg_precision@3', 'avg_precision@5', 'avg_precision@10',
                   'avg_mrr', 'queries_with_hits@1', 'queries_with_hits@3']:
        b_val = baseline_aggs.get(metric, 0)
        e_val = enriched_aggs.get(metric, 0)
        if isinstance(b_val, str) or isinstance(e_val, str):
            # Handle fraction strings like "3/5"
            continue
        diff = e_val - b_val
        rel_improvement = (diff / b_val * 100) if b_val > 0 else 0
        comparison[metric] = {
            'baseline': round(b_val, 3),
            'enriched': round(e_val, 3),
            'absolute_diff': round(diff, 3),
            'relative_improvement_%': round(rel_improvement, 1)
        }

    results['comparison'] = comparison

    print("\nMetric improvements:")
    for metric, vals in comparison.items():
        if metric.endswith('@1') or metric == 'avg_mrr':
            print(f"\n{metric}:")
            print(f"  Baseline:  {vals['baseline']}")
            print(f"  Enriched:  {vals['enriched']}")
            print(f"  Change:    {vals['absolute_diff']:+.3f} ({vals['relative_improvement_%']:+.1f}%)")

    # Save all results
    Path(output_dir).mkdir(exist_ok=True)
    baseline_path = Path(output_dir) / "comparison_baseline.json"
    enriched_path = Path(output_dir) / "comparison_enriched.json"
    comparison_path = Path(output_dir) / "comparison_results.json"

    save_evaluation_results(results['baseline'], str(baseline_path))
    save_evaluation_results(results['enriched'], str(enriched_path))
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\nSaved:")
    print(f"  Baseline results: {baseline_path}")
    print(f"  Enriched results: {enriched_path}")
    print(f"  Comparison:       {comparison_path}")

    return results


def generate_hit_rate_report(results: Dict[str, Any]) -> str:
    """
    Generate a human-readable hit rate comparison report.

    Hit rate = fraction of queries that retrieve at least one relevant doc in top-k.
    """
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("ENRICHMENT IMPACT: HIT RATE & RECALL COMPARISON")
    report_lines.append("="*70)

    baseline = results['baseline']
    enriched = results['enriched']
    comparison = results['comparison']

    total_queries = baseline['total_queries']

    report_lines.append(f"\nTotal test queries: {total_queries}")
    report_lines.append("\n--- Hit Rate (Queries with at least one relevant doc) ---")

    for k in [1, 3, 5, 10]:
        b_hits = baseline['aggregates'].get(f'queries_with_hits@{k}', '0/0')
        e_hits = enriched['aggregates'].get(f'queries_with_hits@{k}', '0/0')
        if isinstance(b_hits, str) and '/' in b_hits:
            b_num = int(b_hits.split('/')[0])
            e_num = int(e_hits.split('/')[0]) if isinstance(e_hits, str) and '/' in e_hits else int(e_hits)
            b_rate = b_num / total_queries
            e_rate = e_num / total_queries
            diff = e_rate - b_rate
            report_lines.append(f"\n@{k}:")
            report_lines.append(f"  Baseline:  {b_hits} ({b_rate:.1%})")
            report_lines.append(f"  Enriched:  {e_hits} ({e_rate:.1%})")
            report_lines.append(f"  Change:    {diff:+.1%}")

    report_lines.append("\n--- Recall@k (Average fraction of relevant docs retrieved) ---")
    for k in [1, 3, 5, 10]:
        metric = f'avg_recall@{k}'
        if metric in comparison:
            vals = comparison[metric]
            report_lines.append(f"\n{k}:")
            report_lines.append(f"  Baseline:  {vals['baseline']:.3f}")
            report_lines.append(f"  Enriched:  {vals['enriched']:.3f}")
            report_lines.append(f"  Change:    {vals['absolute_diff']:+.3f}")

    report_lines.append("\n--- MRR (Mean Reciprocal Rank) ---")
    if 'avg_mrr' in comparison:
        vals = comparison['avg_mrr']
        report_lines.append(f"\nMRR:")
        report_lines.append(f"  Baseline:  {vals['baseline']:.3f}")
        report_lines.append(f"  Enriched:  {vals['enriched']:.3f}")
        report_lines.append(f"  Change:    {vals['absolute_diff']:+.3f}")

    report_lines.append("\n" + "="*70)
    report_lines.append("CONCLUSION")
    report_lines.append("="*70)

    # Auto-conclusion based on metrics
    mrr_improvement = comparison.get('avg_mrr', {}).get('relative_improvement_%', 0)
    recall1_improvement = comparison.get('avg_recall@1', {}).get('relative_improvement_%', 0)

    if mrr_improvement > 5 or recall1_improvement > 10:
        conclusion = "Enrichment shows significant improvement in retrieval quality."
        recommendation = "Proceed with enrichment enabled for production deployment."
    elif mrr_improvement > 0 or recall1_improvement > 0:
        conclusion = "Enrichment shows modest improvement."
        recommendation = "Consider enrichment benefits vs. storage/performance costs."
    else:
        conclusion = "Enrichment does not show clear improvement."
        recommendation = "Review query patterns, enrichment strategies, or increase variations."

    report_lines.append(f"\n{conclusion}")
    report_lines.append(f"\nRecommendation: {recommendation}")
    report_lines.append("\n" + "="*70)

    return "\n".join(report_lines)


def main():
    """Run the comparison evaluation."""
    print("="*70)
    print("RETRIEVAL EVALUATION: ENRICHMENT COMPARISON")
    print("Testing if query-aware enrichment improves recall")
    print("="*70)

    # Load config (optional)
    try:
        config = load_config()
        print(f"\n[OK] Loaded config from: {config.config_path or 'defaults'}")
    except Exception as e:
        print(f"[WARN] Config warning: {e}")
        config = None

    # Paths
    insights_path = Path("insights_sample.json")
    queries_path = Path("evaluation/test_queries.json")

    if not insights_path.exists():
        print(f"[ERR] Insights file not found: {insights_path}")
        return
    if not queries_path.exists():
        print(f"[ERR] Test queries file not found: {queries_path}")
        return

    # Run comparison
    results = run_comparison_evaluation(str(insights_path), str(queries_path))

    # Generate and print hit rate report
    report = generate_hit_rate_report(results)
    print(report)

    # Save report
    output_dir = Path("output")
    report_path = output_dir / "enrichment_comparison_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n[OK] Report saved to: {report_path}")


if __name__ == "__main__":
    main()
