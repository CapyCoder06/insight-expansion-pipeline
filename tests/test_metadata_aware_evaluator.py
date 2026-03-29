import pytest
from src.pipeline.evaluator import MetadataAwareRetrievalEvaluator

def test_metadata_aware_evaluator_uses_metadata_when_enabled():
    docs = [
        {'text': 'doc1 content', 'metadata': {'queries': ['q1?'], 'synonyms': {}}},
        {'text': 'doc2 content', 'metadata': {}}
    ]
    evaluator = MetadataAwareRetrievalEvaluator(docs, use_metadata=True)
    # Should have built global synonym map (empty)
    assert evaluator.use_metadata is True
    assert hasattr(evaluator, 'query_expander')
    assert hasattr(evaluator, 'meta_matcher')
    assert hasattr(evaluator, 'hybrid_scorer')

def test_metadata_aware_evaluator_fallback_when_use_metadata_false():
    docs = [{'text': 'doc1', 'metadata': {}}]
    evaluator = MetadataAwareRetrievalEvaluator(docs, use_metadata=False)
    # Should still initialize but not use metadata
    assert evaluator.use_metadata is False

def test_metadata_aware_evaluator_global_synonym_map_built():
    docs = [
        {'text': 'doc1', 'metadata': {'synonyms': {'profit': ['earnings']}}},
        {'text': 'doc2', 'metadata': {'synonyms': {'sales': ['revenue']}}}
    ]
    evaluator = MetadataAwareRetrievalEvaluator(docs)
    assert 'profit' in evaluator.global_synonyms
    assert 'earnings' in evaluator.global_synonyms['profit']
    assert 'sales' in evaluator.global_synonyms

def test_metadata_aware_evaluator_fallback_to_baseline_when_use_metadata_false():
    """Ensure behavior matches base RetrievalEvaluator when use_metadata=False."""
    docs = [{'text': 'Technology profit', 'metadata': {}}]
    evaluator = MetadataAwareRetrievalEvaluator(docs, use_metadata=False)
    # Should be able to evaluate without errors
    result = evaluator.evaluate_query('Technology profit', {0})
    assert 'recall@k' in result
    assert result['recall@k'][1] == 1.0  # Exact match should retrieve at rank 1

def test_metadata_aware_evaluator_graceful_with_missing_metadata():
    docs = [{'text': 'some content', 'metadata': {}}]
    evaluator = MetadataAwareRetrievalEvaluator(docs, use_metadata=True)
    # Should not crash even though no metadata
    result = evaluator.evaluate_query('content?', {0})
    assert result is not None
    assert 'mrr' in result
