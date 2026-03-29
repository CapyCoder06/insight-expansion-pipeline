import pytest
from src.pipeline.evaluator import MetadataAwareRetrievalEvaluator

def test_hybrid_retrieval_with_metadata():
    docs = [
        {
            'text': 'Technology has high profit and margin.',
            'metadata': {
                'queries': ['Which category is most profitable?'],
                'synonyms': {'profit': ['earnings', 'income']}
            }
        },
        {
            'text': 'Furniture has low margin.',
            'metadata': {
                'queries': ['Which category has worst margin?'],
                'synonyms': {'margin': ['profitability']}
            }
        }
    ]
    evaluator = MetadataAwareRetrievalEvaluator(docs, use_metadata=True)
    # Query that should match doc 0 via synonyms and content
    result = evaluator.evaluate_query('Which category earns the most?', {0})
    # Should retrieve doc0, hopefully high score due to synonyms
    assert result['recall@k'][1] > 0  # At least something retrieved at rank 1

def test_fallback_to_text_only_when_use_metadata_false():
    docs = [{'text': 'Tech profit high', 'metadata': {}}]
    evaluator = MetadataAwareRetrievalEvaluator(docs, use_metadata=False)
    result = evaluator.evaluate_query('Tech profit', {0})
    assert result['recall@k'][1] == 1.0  # Should match exactly

def test_graceful_when_metadata_missing():
    docs = [{'text': 'some content', 'metadata': {}}]
    evaluator = MetadataAwareRetrievalEvaluator(docs, use_metadata=True)
    # Should not crash, even though no metadata
    result = evaluator.evaluate_query('content?', {0})
    assert result is not None
    assert 'mrr' in result

def test_metadata_improves_retrieval():
    """Compare metadata-aware vs text-only on same documents."""
    docs = [
        {
            'text': 'Technology has the highest profit of $664K with margin 14%.',
            'metadata': {
                'queries': ['Which category has the highest profit?', 'Top category by earnings?'],
                'synonyms': {'profit': ['earnings', 'income'], 'highest': ['maximum', 'top']}
            }
        },
        {
            'text': 'Furniture has low margin of 7%, below average 11.6%.',
            'metadata': {
                'queries': ['Which has the lowest margin?'],
                'synonyms': {'margin': ['profitability'], 'low': ['lowest', 'minimum']}
            }
        }
    ]
    query = "What category makes the most money?"

    # Text-only evaluator
    baseline = MetadataAwareRetrievalEvaluator(docs, use_metadata=False)
    baseline_result = baseline.evaluate_query(query, {0})

    # Metadata-aware evaluator
    meta = MetadataAwareRetrievalEvaluator(docs, use_metadata=True)
    meta_result = meta.evaluate_query(query, {0})

    # Metadata-aware should have better or equal recall@1
    # Because synonyms and metadata queries help match doc0
    assert meta_result['recall@k'][1] >= baseline_result['recall@k'][1]


def test_metadata_bonus_enabled_via_config():
    """Should apply metadata bonus when enabled in config."""
    from src.pipeline.config import Config

    # Create config with bonus enabled
    config = Config()
    config.config['retrieval'] = {
        'use_metadata': True,
        'weights': {'text': 0.4, 'metadata_query': 0.4, 'expanded': 0.2},
        'metadata_bonus': {
            'enabled': True,
            'threshold': 0.6,
            'boost_factor': 2.0
        }
    }

    docs = [
        {
            'text': 'Technology has high profit.',
            'metadata': {
                'queries': ['Which category leads in profit?']
            }
        }
    ]

    evaluator = MetadataAwareRetrievalEvaluator(docs, config=config, use_metadata=True)

    # The hybrid_scorer should have bonus config
    assert evaluator.hybrid_scorer.bonus_config['enabled'] is True
    assert evaluator.hybrid_scorer.bonus_config['threshold'] == 0.6
    assert evaluator.hybrid_scorer.bonus_config['boost_factor'] == 2.0


def test_metadata_bonus_disabled_by_default():
    """Metadata bonus should be disabled by default."""
    from src.pipeline.config import Config

    config = Config()
    # Ensure no custom config overrides defaults
    if 'retrieval' in config.config and 'metadata_bonus' in config.config['retrieval']:
        config.config['retrieval'].pop('metadata_bonus', None)

    docs = [{'text': 'doc', 'metadata': {'queries': []}}]
    evaluator = MetadataAwareRetrievalEvaluator(docs, config=config, use_metadata=True)

    # Should use default (disabled)
    assert evaluator.hybrid_scorer.bonus_config['enabled'] is False
