"""
Tests for enrichment comparison evaluation.
"""

import sys
from pathlib import Path
# Add project root to path to import evaluation and pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from evaluation.comparison_evaluator import (
    load_test_queries,
    generate_documents,
    run_comparison_evaluation,
    generate_hit_rate_report,
    _generate_enriched_with_mapping,
    _transform_queries_for_enriched
)
from pipeline.evaluator import RetrievalEvaluator
import json
import pytest


class TestComparisonEvaluator:
    """Tests for the comparison evaluation system."""

    def test_load_test_queries(self):
        """Test that test queries load correctly."""
        queries = load_test_queries('evaluation/test_queries.json')
        assert isinstance(queries, list)
        assert len(queries) >= 10
        for q in queries:
            assert 'query' in q
            assert 'relevant_doc_ids' in q
            assert isinstance(q['relevant_doc_ids'], list)
            assert len(q['relevant_doc_ids']) > 0

    def test_generate_documents_baseline(self):
        """Test baseline document generation."""
        insights = [
            {
                "text": "Technology has the highest profit of $664K",
                "dimensions": ["category"],
                "metrics": ["profit"],
                "type_hint": "fact"
            }
        ]
        docs = generate_documents(insights, enrich=False)
        assert len(docs) == 1  # 1 insight -> 1 doc (no enrichment)
        assert 'text' in docs[0]
        assert 'metadata' in docs[0]

    def test_generate_documents_enriched(self):
        """Test enriched document generation creates variations."""
        insights = [
            {
                "text": "Technology has the highest profit of $664K",
                "dimensions": ["category"],
                "metrics": ["profit"],
                "type_hint": "fact"
            }
        ]
        docs = generate_documents(insights, enrich=True)
        assert len(docs) >= 1  # At least 1 (original), typically 3
        assert len(docs) <= 10  # Upper bound
        # All docs have same metadata type
        metadata_types = [doc['metadata']['type'] for doc in docs]
        assert all(t == 'fact' for t in metadata_types)

    def test_enrichment_increases_document_count(self):
        """Test that enrichment increases total document count."""
        insights = [
            {
                "text": "Technology has the highest profit of $664K",
                "dimensions": ["category"],
                "metrics": ["profit"],
                "type_hint": "fact"
            },
            {
                "text": "Sales increased from $2.26M to $4.3M, CAGR ~24%/year",
                "dimensions": ["year"],
                "metrics": ["sales"],
                "type_hint": "trend"
            }
        ]
        baseline_docs = generate_documents(insights, enrich=False)
        enriched_docs = generate_documents(insights, enrich=True)
        assert len(enriched_docs) > len(baseline_docs)
        # Expect at least 2x more (2 insights * 3 variations = 6 vs 2)
        assert len(enriched_docs) >= len(baseline_docs) * 2

    def test_retrieval_evaluator_consistency(self):
        """Test that evaluator produces same results for same inputs."""
        docs = [
            {"text": "Technology profit is highest", "metadata": {"type": "fact"}},
            {"text": "Furniture margin is low", "metadata": {"type": "fact"}}
        ]
        evaluator = RetrievalEvaluator(docs)
        result1 = evaluator.evaluate_query("profit", {0}, top_k=[1])
        result2 = evaluator.evaluate_query("profit", {0}, top_k=[1])
        assert result1['recall@k'][1] == result2['recall@k'][1]
        assert result1['mrr'] == result2['mrr']

    def test_enriched_mapping_structure(self):
        """Test that enriched mapping returns correct structure."""
        insights = [
            {"text": "Tech highest profit", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"},
            {"text": "Sales increased", "dimensions": ["year"], "metrics": ["sales"], "type_hint": "trend"}
        ]
        enriched_docs, mapping = _generate_enriched_with_mapping(insights)
        assert len(enriched_docs) > 0
        assert len(mapping) == len(insights)
        for i, indices in enumerate(mapping):
            assert len(indices) >= 1  # each insight should have at least one doc
            # indices should be contiguous and increasing
            if i > 0:
                # Ensure no overlap between insights
                prev_indices = set(mapping[i-1])
                curr_indices = set(indices)
                assert prev_indices.isdisjoint(curr_indices)
        # Total docs count equals sum of lengths of mapping lists
        assert len(enriched_docs) == sum(len(indices) for indices in mapping)

    def test_transform_queries_for_enriched(self):
        """Test that query transformation correctly maps baseline IDs to enriched IDs."""
        mapping = [
            [0, 1, 2],   # insight 0 -> enriched indices 0,1,2
            [3, 4, 5],   # insight 1 -> enriched indices 3,4,5
            [6, 7, 8]    # insight 2 -> enriched indices 6,7,8
        ]
        test_queries = [
            {"query": "test1", "relevant_doc_ids": [0]},
            {"query": "test2", "relevant_doc_ids": [1]},
            {"query": "test3", "relevant_doc_ids": [0, 1]},
            {"query": "test4", "relevant_doc_ids": [2]}
        ]
        transformed = _transform_queries_for_enriched(test_queries, mapping)
        # trans[0] should map [0] -> [0,1,2]
        assert set(transformed[0]['relevant_doc_ids']) == {0, 1, 2}
        # trans[1] should map [1] -> [3,4,5]
        assert set(transformed[1]['relevant_doc_ids']) == {3, 4, 5}
        # trans[2] should map [0,1] -> [0,1,2,3,4,5]
        assert set(transformed[2]['relevant_doc_ids']) == {0, 1, 2, 3, 4, 5}
        # trans[3] should map [2] -> [6,7,8]
        assert set(transformed[3]['relevant_doc_ids']) == {6, 7, 8}

    def test_comparison_results_structure(self, tmp_path):
        """Test that run_comparison_evaluation returns expected structure."""
        # Use minimal insights
        insights_path = tmp_path / "insights.json"
        with open(insights_path, 'w') as f:
            json.dump([
                {"text": "Tech highest profit", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"}
            ], f)

        # Use minimal test queries file
        queries_path = tmp_path / "queries.json"
        with open(queries_path, 'w') as f:
            json.dump([
                {"query": "highest profit category", "relevant_doc_ids": [0]}
            ], f)

        results = run_comparison_evaluation(str(insights_path), str(queries_path), str(tmp_path))

        assert 'baseline' in results
        assert 'enriched' in results
        assert 'comparison' in results
        assert 'aggregates' in results['baseline']
        assert 'aggregates' in results['enriched']
        # Comparison should have at least one metric
        assert len(results['comparison']) > 0
        assert 'avg_recall@1' in results['comparison']

    def test_hit_rate_report_format(self):
        """Test that hit rate report is well-formatted."""
        results = {
            'baseline': {
                'total_queries': 10,
                'aggregates': {
                    'queries_with_hits@1': '3/10',
                    'queries_with_hits@3': '6/10'
                }
            },
            'enriched': {
                'total_queries': 10,
                'aggregates': {
                    'queries_with_hits@1': '5/10',
                    'queries_with_hits@3': '8/10'
                }
            },
            'comparison': {
                'avg_recall@1': {'baseline': 0.3, 'enriched': 0.5, 'absolute_diff': 0.2, 'relative_improvement_%': 66.7},
                'avg_mrr': {'baseline': 0.4, 'enriched': 0.6, 'absolute_diff': 0.2, 'relative_improvement_%': 50.0}
            }
        }
        report = generate_hit_rate_report(results)
        assert 'BASELINE' in report or 'Baseline' in report
        assert 'ENRICHED' in report or 'Enriched' in report
        assert 'COMPARISON' in report or 'Comparison' in report
        assert 'Hit Rate' in report or 'hit rate' in report


class TestEndToEndComparison:
    """End-to-end integration tests."""

    def test_full_comparison_pipeline(self, tmp_path):
        """Test the complete comparison evaluation on sample data."""
        # Prepare minimal insights based on real structure
        insights = [
            {
                "text": "Technology has the highest profit of $664K with margin 14%",
                "dimensions": ["category"],
                "metrics": ["profit"],
                "type_hint": "fact"
            },
            {
                "text": "Furniture has low margin of 7%, below average 11.6%",
                "dimensions": ["category"],
                "metrics": ["margin"],
                "type_hint": "fact"
            }
        ]
        insights_file = tmp_path / "insights.json"
        with open(insights_file, 'w') as f:
            json.dump(insights, f)

        # Create queries
        queries = [
            {"query": "highest profit", "relevant_doc_ids": [0]},
            {"query": "low margin category", "relevant_doc_ids": [1]}
        ]
        queries_file = tmp_path / "queries.json"
        with open(queries_file, 'w') as f:
            json.dump(queries, f)

        # Run comparison
        results = run_comparison_evaluation(str(insights_file), str(queries_file), str(tmp_path))

        # Validate structure
        assert results['baseline']['total_queries'] == 2
        assert results['enriched']['total_queries'] == 2
        assert isinstance(results['comparison'], dict)
        assert 'avg_recall@1' in results['comparison']

        # Enrichment should produce more documents
        baseline_docs = generate_documents(insights, enrich=False)
        enriched_docs = generate_documents(insights, enrich=True)
        assert len(enriched_docs) > len(baseline_docs)

    def test_run_comparison_creates_output_files(self, tmp_path):
        """Test that run_comparison_evaluation saves result files."""
        insights = [
            {"text": "Test insight", "dimensions": ["test"], "metrics": ["test"], "type_hint": "fact"}
        ]
        insights_file = tmp_path / "insights.json"
        with open(insights_file, 'w') as f:
            json.dump(insights, f)

        queries = [{"query": "test", "relevant_doc_ids": [0]}]
        queries_file = tmp_path / "queries.json"
        with open(queries_file, 'w') as f:
            json.dump(queries, f)

        results = run_comparison_evaluation(str(insights_file), str(queries_file), str(tmp_path))

        # Check output files exist
        assert (tmp_path / "comparison_baseline.json").exists()
        assert (tmp_path / "comparison_enriched.json").exists()
        assert (tmp_path / "comparison_results.json").exists()
