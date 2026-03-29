import pytest
from src.pipeline.metadata_retrieval import QueryExpander, MetadataQueryMatcher
from src.pipeline.validator import semantic_similarity

def test_query_expander_with_synonyms():
    synonyms = {
        'profit': ['earnings', 'income'],
        'sales': ['revenue']
    }
    expander = QueryExpander(synonyms)
    query = "Which category has the highest profit?"
    expanded = expander.expand(query)
    # Should include original and variants with synonyms
    assert query in expanded
    assert any('earnings' in q for q in expanded)
    assert any('income' in q for q in expanded)
    # Should have 1 + 2 = 3 variants (original + 2 replacements for 'profit')
    # Note: 'sales' not in query, so no expansion for that
    assert len(expanded) >= 2

def test_query_expander_without_synonyms():
    expander = QueryExpander({})
    query = "test query"
    expanded = expander.expand(query)
    assert expanded == [query]

def test_query_expander_deduplicates_variants():
    synonyms = {'test': ['test', 'exam']}
    expander = QueryExpander(synonyms)
    query = "test"
    expanded = expander.expand(query)
    # 'test' appears twice (original and synonym) but should deduplicate
    # 'exam' is a different synonym and should be present
    assert 'test' in expanded
    assert 'exam' in expanded
    assert len(expanded) == 2  # no duplicates, but both unique variants present


# MetadataQueryMatcher tests
def test_metadata_query_matcher_returns_max_similarity():
    matcher = MetadataQueryMatcher()
    query = "Which category has highest profit?"
    metadata_queries = [
        "What is the top category by earnings?",
        "Which product category leads in profit?",
        "Category with maximum profit"
    ]
    # Compute similarity manually to verify
    scores = [semantic_similarity(query, q) for q in metadata_queries]
    expected_max = max(scores)
    result = matcher.similarity(query, metadata_queries)
    assert result == pytest.approx(expected_max)
    assert 0 <= result <= 1

def test_metadata_query_matcher_empty_list():
    matcher = MetadataQueryMatcher()
    similarity = matcher.similarity("test query", [])
    assert similarity == 0.0


# HybridScorer tests
from src.pipeline.metadata_retrieval import HybridScorer

def test_hybrid_scorer_weights_normalized():
    weights = {'text': 0.6, 'metadata_query': 0.3, 'expanded': 0.1}
    scorer = HybridScorer(weights)
    total = sum(scorer.weights.values())
    assert abs(total - 1.0) < 1e-9

def test_hybrid_scorer_normalizes_non_unit_weights():
    weights = {'text': 6, 'metadata_query': 3, 'expanded': 1}
    scorer = HybridScorer(weights)
    total = sum(scorer.weights.values())
    assert abs(total - 1.0) < 1e-9
    # Check normalized values
    assert scorer.weights['text'] == pytest.approx(0.6)
    assert scorer.weights['metadata_query'] == pytest.approx(0.3)
    assert scorer.weights['expanded'] == pytest.approx(0.1)

def test_hybrid_scorer_compute_score():
    weights = {'text': 0.6, 'metadata_query': 0.3, 'expanded': 0.1}
    scorer = HybridScorer(weights)
    doc = {
        'text': 'Technology has high profit.',
        'metadata': {
            'queries': ['Which product category leads in profit?']
        }
    }
    expanded_variants = ['profit', 'earnings', 'income']
    meta_matcher = MetadataQueryMatcher()

    # Mock semantic_similarity at the point where it's used
    import src.pipeline.metadata_retrieval as mr
    original_sim = mr.semantic_similarity

    def mock_sim(a, b):
        if a == 'Which category has highest profit?' and b == 'Technology has high profit.':
            return 0.8
        if a in expanded_variants and b == 'Technology has high profit.':
            return 0.7
        if a == 'Which category has highest profit?' and b == 'Which product category leads in profit?':
            return 0.9
        return 0.0

    mr.semantic_similarity = mock_sim
    try:
        score = scorer.score('Which category has highest profit?', doc, expanded_variants, meta_matcher)
        expected = 0.6*0.8 + 0.3*0.9 + 0.1*0.7
        assert score == pytest.approx(expected)
    finally:
        mr.semantic_similarity = original_sim

def test_hybrid_scorer_empty_expanded_variants():
    weights = {'text': 0.6, 'metadata_query': 0.3, 'expanded': 0.1}
    scorer = HybridScorer(weights)
    doc = {'text': 'Some document text'}
    expanded_variants = []
    meta_matcher = MetadataQueryMatcher()
    import src.pipeline.metadata_retrieval as mr
    original_sim = mr.semantic_similarity

    def mock_sim(a, b):
        if a == 'query' and b == 'Some document text':
            return 0.5
        return 0.0

    mr.semantic_similarity = mock_sim
    try:
        score = scorer.score('query', doc, expanded_variants, meta_matcher)
        expected = 0.6*0.5 + 0.3*0.0 + 0.1*0.0
        assert score == pytest.approx(expected)
    finally:
        mr.semantic_similarity = original_sim

def test_hybrid_scorer_empty_document_text():
    weights = {'text': 0.6, 'metadata_query': 0.3, 'expanded': 0.1}
    scorer = HybridScorer(weights)
    doc = {'text': ''}
    expanded_variants = ['variant']
    meta_matcher = MetadataQueryMatcher()
    score = scorer.score('query', doc, expanded_variants, meta_matcher)
    assert score == 0.0


# Tests for metadata similarity bonus feature
def test_hybrid_scorer_metadata_bonus_above_threshold():
    """Should apply bonus boost when metadata similarity exceeds threshold."""
    weights = {'text': 0.4, 'metadata_query': 0.4, 'expanded': 0.2}
    bonus_config = {
        'enabled': True,
        'threshold': 0.7,
        'boost_factor': 1.5
    }
    scorer = HybridScorer(weights, bonus_config)
    doc = {
        'text': 'Technology has high profit.',
        'metadata': {
            'queries': ['Which product category leads in profit?']
        }
    }
    expanded_variants = ['profit']
    meta_matcher = MetadataQueryMatcher()

    import src.pipeline.metadata_retrieval as mr
    original_sim = mr.semantic_similarity

    def mock_sim(a, b):
        # High metadata query similarity (> threshold)
        if a == 'Which category has highest profit?' and b == 'Which product category leads in profit?':
            return 0.9  # > 0.7, should get bonus
        if a == 'Which category has highest profit?' and b == 'Technology has high profit.':
            return 0.8
        if a in expanded_variants and b == 'Technology has high profit.':
            return 0.6
        return 0.0

    mr.semantic_similarity = mock_sim
    try:
        score = scorer.score('Which category has highest profit?', doc, expanded_variants, meta_matcher)
        # Raw expected: 0.4*0.8 + (0.4*0.9)*1.5 + 0.2*0.6 = 0.32 + 0.54 + 0.12 = 0.98
        raw_expected = 0.4*0.8 + (0.4*0.9)*1.5 + 0.2*0.6
        # Normalized: _max_raw = 1 + w_meta*(boost-1) = 1 + 0.4*0.5 = 1.2
        normalized_expected = raw_expected / 1.2
        assert score == pytest.approx(normalized_expected)
        # Verify bonus actually increased the score compared to no-bonus normalized
        raw_no_bonus = 0.4*0.8 + 0.4*0.9 + 0.2*0.6
        normalized_no_bonus = raw_no_bonus / 1.2
        assert score > normalized_no_bonus
    finally:
        mr.semantic_similarity = original_sim


def test_hybrid_scorer_metadata_bonus_below_threshold():
    """Should NOT apply bonus when metadata similarity is below threshold."""
    weights = {'text': 0.4, 'metadata_query': 0.4, 'expanded': 0.2}
    bonus_config = {
        'enabled': True,
        'threshold': 0.7,
        'boost_factor': 1.5
    }
    scorer = HybridScorer(weights, bonus_config)
    doc = {
        'text': 'Technology has high profit.',
        'metadata': {
            'queries': ['Which product category leads in profit?']
        }
    }
    expanded_variants = ['profit']
    meta_matcher = MetadataQueryMatcher()

    import src.pipeline.metadata_retrieval as mr
    original_sim = mr.semantic_similarity

    def mock_sim(a, b):
        # Low metadata query similarity (< threshold)
        if a == 'Which category has highest profit?' and b == 'Which product category leads in profit?':
            return 0.5  # < 0.7, no bonus
        if a == 'Which category has highest profit?' and b == 'Technology has high profit.':
            return 0.8
        if a in expanded_variants and b == 'Technology has high profit.':
            return 0.6
        return 0.0

    mr.semantic_similarity = mock_sim
    try:
        score = scorer.score('Which category has highest profit?', doc, expanded_variants, meta_matcher)
        # Raw: 0.4*0.8 + 0.4*0.5 + 0.2*0.6 = 0.32 + 0.20 + 0.12 = 0.64
        raw_expected = 0.4*0.8 + 0.4*0.5 + 0.2*0.6
        # Normalized: _max_raw = 1 + w_meta*(boost-1) = 1 + 0.4*0.5 = 1.2 (even though this doc doesn't get boost, max_raw still based on config)
        normalized_expected = raw_expected / 1.2
        assert score == pytest.approx(normalized_expected)
    finally:
        mr.semantic_similarity = original_sim


def test_hybrid_scorer_metadata_bonus_disabled():
    """Should NOT apply bonus when feature is disabled."""
    weights = {'text': 0.4, 'metadata_query': 0.4, 'expanded': 0.2}
    bonus_config = {
        'enabled': False,
        'threshold': 0.7,
        'boost_factor': 1.5
    }
    scorer = HybridScorer(weights, bonus_config)
    doc = {
        'text': 'Technology has high profit.',
        'metadata': {
            'queries': ['Which product category leads in profit?']
        }
    }
    expanded_variants = ['profit']
    meta_matcher = MetadataQueryMatcher()

    import src.pipeline.metadata_retrieval as mr
    original_sim = mr.semantic_similarity

    def mock_sim(a, b):
        # High metadata query similarity but bonus disabled
        if a == 'Which category has highest profit?' and b == 'Which product category leads in profit?':
            return 0.9  # > 0.7 but disabled
        if a == 'Which category has highest profit?' and b == 'Technology has high profit.':
            return 0.8
        if a in expanded_variants and b == 'Technology has high profit.':
            return 0.6
        return 0.0

    mr.semantic_similarity = mock_sim
    try:
        score = scorer.score('Which category has highest profit?', doc, expanded_variants, meta_matcher)
        # No bonus: 0.4*0.8 + 0.4*0.9 + 0.2*0.6 = 0.32 + 0.36 + 0.12 = 0.80
        expected = 0.4*0.8 + 0.4*0.9 + 0.2*0.6
        assert score == pytest.approx(expected)
    finally:
        mr.semantic_similarity = original_sim


def test_hybrid_scorer_metadata_bonus_edge_case_exact_threshold():
    """Should apply bonus when similarity equals threshold (inclusive)."""
    weights = {'text': 0.4, 'metadata_query': 0.4, 'expanded': 0.2}
    bonus_config = {
        'enabled': True,
        'threshold': 0.7,
        'boost_factor': 1.5
    }
    scorer = HybridScorer(weights, bonus_config)
    doc = {
        'text': 'Technology has high profit.',
        'metadata': {
            'queries': ['Which product category leads in profit?']
        }
    }
    expanded_variants = ['profit']
    meta_matcher = MetadataQueryMatcher()

    import src.pipeline.metadata_retrieval as mr
    original_sim = mr.semantic_similarity

    def mock_sim(a, b):
        # Exact threshold value
        if a == 'Which category has highest profit?' and b == 'Which product category leads in profit?':
            return 0.7  # == 0.7, should get bonus
        if a == 'Which category has highest profit?' and b == 'Technology has high profit.':
            return 0.8
        if a in expanded_variants and b == 'Technology has high profit.':
            return 0.6
        return 0.0

    mr.semantic_similarity = mock_sim
    try:
        score = scorer.score('Which category has highest profit?', doc, expanded_variants, meta_matcher)
        # Raw: 0.4*0.8 + (0.4*0.7)*1.5 + 0.2*0.6 = 0.32 + 0.42 + 0.12 = 0.86
        raw_expected = 0.4*0.8 + (0.4*0.7)*1.5 + 0.2*0.6
        # Normalized: _max_raw = 1 + 0.4*0.5 = 1.2
        normalized_expected = raw_expected / 1.2
        assert score == pytest.approx(normalized_expected)
    finally:
        mr.semantic_similarity = original_sim


def test_hybrid_scorer_score_normalized_not_clamped():
    """Should normalize scores > 1.0 using monotonic function, not hard clipping."""
    weights = {'text': 0.4, 'metadata_query': 0.4, 'expanded': 0.2}
    bonus_config = {
        'enabled': True,
        'threshold': 0.5,  # Low threshold to easily trigger
        'boost_factor': 5.0  # High boost to cause overflow
    }
    scorer = HybridScorer(weights, bonus_config)
    doc = {
        'text': 'Technology has high profit.',
        'metadata': {
            'queries': ['Which product category leads in profit?']
        }
    }
    expanded_variants = ['profit']
    meta_matcher = MetadataQueryMatcher()

    import src.pipeline.metadata_retrieval as mr
    original_sim = mr.semantic_similarity

    def mock_sim(a, b):
        # All similarities are high
        if a == 'Which category has highest profit?':
            if b == 'Which product category leads in profit?':
                return 0.9  # meta similarity high
            if b == 'Technology has high profit.':
                return 0.95  # text similarity high
        if a in expanded_variants and b == 'Technology has high profit.':
            return 0.9  # expanded similarity high
        return 0.0

    mr.semantic_similarity = mock_sim
    try:
        score = scorer.score('Which category has highest profit?', doc, expanded_variants, meta_matcher)
        # Raw: 0.4*0.95 + (0.4*0.9)*5.0 + 0.2*0.9 = 0.38 + 1.8 + 0.18 = 2.36
        raw_expected = 0.4*0.95 + (0.4*0.9)*5.0 + 0.2*0.9
        # _max_raw = 1 + w_meta*(boost-1) = 1 + 0.4*(5-1) = 1 + 1.6 = 2.6
        normalized_expected = raw_expected / 2.6
        assert score < 1.0, f"Score {score} should be < 1.0 after normalization"
        assert score == pytest.approx(normalized_expected, abs=1e-3)
    finally:
        mr.semantic_similarity = original_sim


def test_hybrid_scorer_perfect_match_no_boost_equals_one():
    """Perfect match without bonus boost should score 1.0."""
    weights = {'text': 0.6, 'metadata_query': 0.3, 'expanded': 0.1}
    # Default bonus config (disabled)
    scorer = HybridScorer(weights)
    doc = {
        'text': 'Some content',
        'metadata': {
            'queries': ['query']
        }
    }
    expanded_variants = ['query']
    meta_matcher = MetadataQueryMatcher()

    import src.pipeline.metadata_retrieval as mr
    original_sim = mr.semantic_similarity

    def mock_sim(a, b):
        return 1.0  # Perfect similarity everywhere

    mr.semantic_similarity = mock_sim
    try:
        score = scorer.score('query', doc, expanded_variants, meta_matcher)
        # Without boost, max raw score = 1.0, normalization should yield 1.0
        assert score == pytest.approx(1.0)
    finally:
        mr.semantic_similarity = original_sim


def test_hybrid_scorer_ranking_preserved_after_normalization():
    """Monotonic normalization should preserve relative ranking of documents."""
    weights = {'text': 0.4, 'metadata_query': 0.4, 'expanded': 0.2}
    bonus_config = {
        'enabled': True,
        'threshold': 0.5,
        'boost_factor': 5.0
    }
    scorer = HybridScorer(weights, bonus_config)

    doc1 = {'text': 'Doc1', 'metadata': {'queries': ['q1']}}
    doc2 = {'text': 'Doc2', 'metadata': {'queries': ['q2']}}
    docs = [doc1, doc2]

    meta_matcher = MetadataQueryMatcher()

    import src.pipeline.metadata_retrieval as mr
    original_sim = mr.semantic_similarity

    # Doc1 gets higher raw score than doc2
    def mock_sim(a, b):
        if b == 'Doc1':
            return 0.95  # high similarity
        if b == 'Doc2':
            return 0.6   # lower similarity
        return 0.0

    mr.semantic_similarity = mock_sim
    try:
        score1 = scorer.score('query', doc1, [], meta_matcher)
        score2 = scorer.score('query', doc2, [], meta_matcher)
        # After normalization, ranking should be preserved
        assert score1 > score2, f"Ranking lost: {score1} should be > {score2}"
    finally:
        mr.semantic_similarity = original_sim
