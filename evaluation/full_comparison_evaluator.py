#!/usr/bin/env python
"""
Full Comparison Evaluation: Baseline vs Enrichment vs Enrichment+Optimization

Tests the impact of:
1. Baseline: no enrichment, no optimization
2. Enriched: query-aware variations only
3. Optimized: enriched + retrieval metadata (queries & synonyms in metadata)

Metrics: recall@k, hit rate, MRR
"""

import sys
from pathlib import Path
import json
from typing import List, Dict, Any

sys.path.insert(0, 'src')

from pipeline import DocumentPipeline, generate_from_insights
from pipeline.evaluator import RetrievalEvaluator, MetadataAwareRetrievalEvaluator, save_evaluation_results
from pipeline.config import load_config
from pipeline.enrichment import DocumentEnricher, select_top_variations


def load_test_queries(path: str) -> List[Dict[str, Any]]:
    """Load test queries from JSON."""
    with open(path, 'r') as f:
        return json.load(f)


def generate_baseline_documents(insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate documents without enrichment or optimization."""
    pipeline = DocumentPipeline(enrich=False, optimize_retrieval=False)
    result = pipeline.run(insights, validate=False, chunk=False)
    return result['documents']


def generate_enriched_documents(insights: List[Dict[str, Any]], config=None) -> tuple:
    """
    Generate enriched documents with mapping.
    Returns: (enriched_docs, mapping) where mapping[i] = list of doc indices for insight i.
    """
    if config is None:
        config = load_config()

    enrichment_config = config.get('enrichment', {})
    enrich_variations = enrichment_config.get('variations_to_keep', 3)
    variations_to_generate = enrichment_config.get('variations_to_generate', 8)

    enricher = DocumentEnricher(num_variations=variations_to_generate, config=config)

    # Generate baseline documents first
    baseline_docs = generate_from_insights(insights)

    # Enrich each and collect mapping
    enriched_docs = []
    mapping = []
    for i, doc in enumerate(baseline_docs):
        variations = enricher.expand_document(doc)
        top_vars = select_top_variations(variations, top_k=enrich_variations)
        start_idx = len(enriched_docs)
        enriched_docs.extend(top_vars)
        end_idx = len(enriched_docs)
        mapping.append(list(range(start_idx, end_idx)))

    return enriched_docs, mapping


def generate_optimized_documents(insights: List[Dict[str, Any]], config=None) -> tuple:
    """
    Generate enriched + retrieval optimized documents.
    Returns: (optimized_docs, mapping) same format as enriched.
    """
    if config is None:
        config = load_config()

    # First generate enriched with mapping
    enriched_docs, mapping = generate_enriched_documents(insights, config)

    # Apply retrieval optimization metadata to each enriched document
    from pipeline.document_generator import extract_retrieval_metadata

    optimized_docs = []
    for doc in enriched_docs:
        retrieval_meta = extract_retrieval_metadata(doc["text"])
        new_metadata = doc["metadata"].copy()
        if retrieval_meta.get("queries"):
            new_metadata["queries"] = retrieval_meta["queries"]
        if retrieval_meta.get("synonyms"):
            new_metadata["synonyms"] = retrieval_meta["synonyms"]
        optimized_docs.append({
            "text": doc["text"],  # Keep text clean
            "metadata": new_metadata
        })

    return optimized_docs, mapping


def _transform_queries_for_enriched(test_queries: List[Dict[str, Any]], insight_to_doc_indices: List[List[int]]) -> List[Dict[str, Any]]:
    """Transform baseline query relevance to enriched/optimized doc indices using mapping."""
    transformed = []
    for query in test_queries:
        new_query = query.copy()
        baseline_ids = query.get('relevant_doc_ids', [])
        # Map each baseline insight index to all its variation indices
        expanded_ids = set()
        for idx in baseline_ids:
            if 0 <= idx < len(insight_to_doc_indices):
                expanded_ids.update(insight_to_doc_indices[idx])
        new_query['relevant_doc_ids'] = list(expanded_ids)
        transformed.append(new_query)
    return transformed


def run_full_comparison(insights_path: str, queries_path: str, output_dir: str = "output") -> Dict[str, Any]:
    """
    Run full comparison: baseline vs enriched vs optimized.

    Returns:
        Dictionary with all results and comparisons
    """
    # Load data
    with open(insights_path) as f:
        insights = json.load(f)
    test_queries = load_test_queries(queries_path)

    results = {
        'baseline': None,
        'enriched': None,
        'optimized': None,
        'comparison': {}
    }

    print("\n" + "="*70)
    print("RETRIEVAL EVALUATION: 3-WAY COMPARISON")
    print("="*70)

    # --- Baseline ---
    print("\n[1/3] BASELINE (no enrichment, no optimization)")
    print("-"*70)
    baseline_docs = generate_baseline_documents(insights)
    print(f"Generated {len(baseline_docs)} documents (1 per insight)")

    baseline_evaluator = RetrievalEvaluator(baseline_docs)
    results['baseline'] = baseline_evaluator.evaluate_dataset(test_queries)

    print("\nBaseline aggregates:")
    for k, v in sorted(results['baseline']['aggregates'].items()):
        if k.startswith('avg_') or k.startswith('queries_'):
            print(f"  {k}: {v}")

    # --- Enriched ---
    print("\n[2/3] ENRICHED (with query-aware variations)")
    print("-"*70)
    enriched_docs, mapping = generate_enriched_documents(insights)
    print(f"Generated {len(enriched_docs)} documents ({len(insights)} insights x {len(enriched_docs)//len(insights)} variations each)")

    # Transform queries to map to all variations
    enriched_queries = _transform_queries_for_enriched(test_queries, mapping)

    enriched_evaluator = RetrievalEvaluator(enriched_docs)
    results['enriched'] = enriched_evaluator.evaluate_dataset(enriched_queries)

    print("\nEnriched aggregates:")
    for k, v in sorted(results['enriched']['aggregates'].items()):
        if k.startswith('avg_') or k.startswith('queries_'):
            print(f"  {k}: {v}")

    # --- Optimized (Enriched + Retrieval Metadata) ---
    print("\n[3/3] OPTIMIZED (enriched + query/synonym metadata)")
    print("-"*70)
    optimized_docs, _ = generate_optimized_documents(insights)  # mapping same as enriched
    print(f"Generated {len(optimized_docs)} documents (same count as enriched)")

    # Queries are the same as enriched (text unchanged, so relevance mapping is identical)
    optimized_queries = enriched_queries

    # Use metadata-aware evaluator with metadata enabled to leverage queries & synonyms
    optimized_evaluator = MetadataAwareRetrievalEvaluator(optimized_docs, use_metadata=True)
    results['optimized'] = optimized_evaluator.evaluate_dataset(optimized_queries)

    print("\nOptimized aggregates:")
    for k, v in sorted(results['optimized']['aggregates'].items()):
        if k.startswith('avg_') or k.startswith('queries_'):
            print(f"  {k}: {v}")

    # --- Comparison ---
    print("\n" + "="*70)
    print("COMPARISON ANALYSIS")
    print("="*70)

    baseline_aggs = results['baseline']['aggregates']
    enriched_aggs = results['enriched']['aggregates']
    optimized_aggs = results['optimized']['aggregates']

    comparison = {
        'baseline_to_enriched': {},
        'baseline_to_optimized': {},
        'enriched_to_optimized': {}
    }

    metrics_of_interest = [
        'avg_recall@1', 'avg_recall@3', 'avg_recall@5', 'avg_recall@10',
        'avg_precision@1', 'avg_precision@3', 'avg_precision@5', 'avg_precision@10',
        'avg_mrr', 'queries_with_hits@1', 'queries_with_hits@3'
    ]

    def compute_diff(b_val, e_val):
        """Compute absolute and relative difference."""
        if isinstance(b_val, str) or isinstance(e_val, str):
            return None, None
        diff = e_val - b_val
        rel_imp = (diff / b_val * 100) if b_val > 0 else 0
        return round(diff, 3), round(rel_imp, 1)

    # Baseline vs Enriched
    print("\nBaseline -> Enriched:")
    for metric in metrics_of_interest:
        b_val = baseline_aggs.get(metric, 0)
        e_val = enriched_aggs.get(metric, 0)
        diff, rel = compute_diff(b_val, e_val)
        if diff is not None:
            comparison['baseline_to_enriched'][metric] = {
                'baseline': round(b_val, 3) if isinstance(b_val, (int, float)) else b_val,
                'enriched': round(e_val, 3) if isinstance(e_val, (int, float)) else e_val,
                'absolute_diff': diff,
                'relative_improvement_%': rel
            }
            if metric.endswith('@1') or metric == 'avg_mrr':
                print(f"  {metric}: {b_val:.3f} -> {e_val:.3f} ({diff:+.3f}, {rel:+.1f}%)")

    # Baseline vs Optimized
    print("\nBaseline -> Optimized:")
    for metric in metrics_of_interest:
        b_val = baseline_aggs.get(metric, 0)
        o_val = optimized_aggs.get(metric, 0)
        diff, rel = compute_diff(b_val, o_val)
        if diff is not None:
            comparison['baseline_to_optimized'][metric] = {
                'baseline': round(b_val, 3) if isinstance(b_val, (int, float)) else b_val,
                'optimized': round(o_val, 3) if isinstance(o_val, (int, float)) else o_val,
                'absolute_diff': diff,
                'relative_improvement_%': rel
            }
            if metric.endswith('@1') or metric == 'avg_mrr':
                print(f"  {metric}: {b_val:.3f} -> {o_val:.3f} ({diff:+.3f}, {rel:+.1f}%)")

    # Enriched vs Optimized
    print("\nEnriched -> Optimized:")
    for metric in metrics_of_interest:
        e_val = enriched_aggs.get(metric, 0)
        o_val = optimized_aggs.get(metric, 0)
        diff, rel = compute_diff(e_val, o_val)
        if diff is not None:
            comparison['enriched_to_optimized'][metric] = {
                'enriched': round(e_val, 3) if isinstance(e_val, (int, float)) else e_val,
                'optimized': round(o_val, 3) if isinstance(o_val, (int, float)) else o_val,
                'absolute_diff': diff,
                'relative_improvement_%': rel
            }
            if metric.endswith('@1') or metric == 'avg_mrr':
                print(f"  {metric}: {e_val:.3f} -> {o_val:.3f} ({diff:+.3f}, {rel:+.1f}%)")

    results['comparison'] = comparison

    # Save all results
    Path(output_dir).mkdir(exist_ok=True)
    baseline_path = Path(output_dir) / "full_comparison_baseline.json"
    enriched_path = Path(output_dir) / "full_comparison_enriched.json"
    optimized_path = Path(output_dir) / "full_comparison_optimized.json"
    comparison_path = Path(output_dir) / "full_comparison_results.json"

    save_evaluation_results(results['baseline'], str(baseline_path))
    save_evaluation_results(results['enriched'], str(enriched_path))
    save_evaluation_results(results['optimized'], str(optimized_path))
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\nSaved:")
    print(f"  Baseline:   {baseline_path}")
    print(f"  Enriched:   {enriched_path}")
    print(f"  Optimized:  {optimized_path}")
    print(f"  Comparison: {comparison_path}")

    return results


def generate_full_report(results: Dict[str, Any]) -> str:
    """Generate a comprehensive comparison report."""
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("FULL COMPARISON: BASELINE vs ENRICHMENT vs OPTIMIZED")
    report_lines.append("="*80)

    total_queries = results['baseline']['total_queries']
    report_lines.append(f"\nTotal test queries: {total_queries}")
    report_lines.append(f"Document counts:")
    report_lines.append(f"  Baseline:  {len(results['baseline'].get('per_query', []))} (simulated from {len(results['baseline']['per_query'])} docs)")
    report_lines.append(f"  Enriched:  N/A (var count varies)")
    report_lines.append(f"  Optimized: N/A (var count varies)")

    comparison = results['comparison']

    # Hit Rate
    report_lines.append("\n" + "="*80)
    report_lines.append("HIT RATE (fraction of queries with >=1 relevant doc in top-k)")
    report_lines.append("="*80)

    for scenario in ['baseline_to_enriched', 'baseline_to_optimized']:
        label = scenario.replace('_to_', ' -> ').title()
        report_lines.append(f"\n{label}:")
        b_hits = results['baseline']['aggregates'] if 'baseline' in scenario else results['baseline']['aggregates']
        t_hits = results['enriched']['aggregates'] if 'enriched' in scenario else results['optimized']['aggregates']

        for k in [1, 3, 5, 10]:
            b_hit = b_hits.get(f'queries_with_hits@{k}', '0/0')
            t_hit = t_hits.get(f'queries_with_hits@{k}', '0/0')
            if isinstance(b_hit, str) and '/' in b_hit:
                b_num = int(b_hit.split('/')[0])
                t_num = int(t_hit.split('/')[0]) if isinstance(t_hit, str) and '/' in t_hit else int(t_hit)
                b_rate = b_num / total_queries
                t_rate = t_num / total_queries
                diff = t_rate - b_rate
                report_lines.append(f"  @{k:2d}: {b_hit} ({b_rate:5.1%}) -> {t_hit} ({t_rate:5.1%})  {diff:+.1%}")

    # Recall@k
    report_lines.append("\n" + "="*80)
    report_lines.append("RECALL@k (average fraction of relevant docs retrieved)")
    report_lines.append("="*80)

    for k in [1, 3, 5, 10]:
        report_lines.append(f"\n@{k}:")
        for scenario, label in [('baseline_to_enriched', 'Baseline->Enriched'),
                               ('baseline_to_optimized', 'Baseline->Optimized')]:
            comp = comparison.get(scenario, {}).get(f'avg_recall@{k}')
            if comp:
                report_lines.append(f"  {label:25s}: {comp['baseline']:.3f} -> {comp['enriched' if 'enriched' in scenario else 'optimized']:.3f}  ({comp['absolute_diff']:+.3f}, {comp['relative_improvement_%']:+.1f}%)")

    # MRR
    report_lines.append("\n" + "="*80)
    report_lines.append("MRR (Mean Reciprocal Rank)")
    report_lines.append("="*80)

    for scenario, label in [('baseline_to_enriched', 'Baseline->Enriched'),
                           ('baseline_to_optimized', 'Baseline->Optimized')]:
        comp = comparison.get(scenario, {}).get('avg_mrr')
        if comp:
            report_lines.append(f"  {label:25s}: {comp['baseline']:.3f} -> {comp['enriched' if 'enriched' in scenario else 'optimized']:.3f}  ({comp['absolute_diff']:+.3f}, {comp['relative_improvement_%']:+.1f}%)")

    # Enriched vs Optimized (incremental value)
    report_lines.append("\n" + "="*80)
    report_lines.append("ENRICHED -> OPTIMIZED (added value of retrieval metadata)")
    report_lines.append("="*80)

    for k in [1, 3, 5, 10]:
        comp = comparison.get('enriched_to_optimized', {}).get(f'avg_recall@{k}')
        if comp:
            report_lines.append(f"\n@{k}:")
            report_lines.append(f"  Enriched:  {comp['enriched']:.3f}")
            report_lines.append(f"  Optimized: {comp['optimized']:.3f}  ({comp['absolute_diff']:+.3f}, {comp['relative_improvement_%']:+.1f}%)")

    comp_mrr = comparison.get('enriched_to_optimized', {}).get('avg_mrr')
    if comp_mrr:
        report_lines.append(f"\nMRR:")
        report_lines.append(f"  Enriched:  {comp_mrr['enriched']:.3f}")
        report_lines.append(f"  Optimized: {comp_mrr['optimized']:.3f}  ({comp_mrr['absolute_diff']:+.3f}, {comp_mrr['relative_improvement_%']:+.1f}%)")

    # Conclusion
    report_lines.append("\n" + "="*80)
    report_lines.append("CONCLUSION & RECOMMENDATIONS")
    report_lines.append("="*80)

    # Determine which layer gives biggest boost
    b_to_o_recall1 = comparison.get('baseline_to_optimized', {}).get('avg_recall@1', {}).get('absolute_diff', 0)
    b_to_e_recall1 = comparison.get('baseline_to_enriched', {}).get('avg_recall@1', {}).get('absolute_diff', 0)
    e_to_o_recall1 = comparison.get('enriched_to_optimized', {}).get('avg_recall@1', {}).get('absolute_diff', 0)

    b_to_o_mrr = comparison.get('baseline_to_optimized', {}).get('avg_mrr', {}).get('absolute_diff', 0)
    b_to_e_mrr = comparison.get('baseline_to_enriched', {}).get('avg_mrr', {}).get('absolute_diff', 0)
    e_to_o_mrr = comparison.get('enriched_to_optimized', {}).get('avg_mrr', {}).get('absolute_diff', 0)

    report_lines.append(f"\nImpact Analysis:")

    # Compare enrichment vs optimization gains
    if b_to_e_recall1 > b_to_o_recall1:
        report_lines.append(f"  * Enrichment provides larger boost than optimization alone")
    elif b_to_o_recall1 > b_to_e_recall1:
        report_lines.append(f"  * Optimization metadata provides larger boost than enrichment alone")
    else:
        report_lines.append(f"  * Enrichment and optimization provide similar gains")

    if e_to_o_recall1 > 0:
        report_lines.append(f"  * Optimization adds +{e_to_o_recall1:.3f} recall@1 on top of enrichment")
    else:
        report_lines.append(f"  * Optimization does not significantly improve over enrichment")

    # Overall recommendation
    report_lines.append(f"\nRecommendation:")

    if b_to_o_recall1 > 0.1 or b_to_o_mrr > 0.05:
        report_lines.append("  [OK] BOTH enrichment and optimization together provide substantial improvement.")
        report_lines.append("       Recommend: Enable both in production.")
    elif b_to_e_recall1 > 0.05:
        report_lines.append("  [OK] Enrichment alone provides meaningful improvement.")
        report_lines.append("       Recommend: Enable enrichment; optimization optional.")
    else:
        report_lines.append("  [!] Neither layer shows strong improvement. Review test queries and enrichment strategy.")

    report_lines.append("\n" + "="*80)
    return "\n".join(report_lines)


def main():
    """Run the full 3-way comparison evaluation."""
    print("="*80)
    print("RETRIEVAL EVALUATION: FULL COMPARISON")
    print("Comparing: Baseline vs Enrichment vs Optimization")
    print("="*80)

    # Load config
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
    results = run_full_comparison(str(insights_path), str(queries_path))

    # Generate and print full report
    report = generate_full_report(results)
    print(report)

    # Save report
    output_dir = Path("output")
    report_path = output_dir / "full_comparison_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n[OK] Report saved to: {report_path}")


if __name__ == "__main__":
    main()
