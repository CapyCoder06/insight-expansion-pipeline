# Metadata-Aware Retrieval Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement metadata-aware retrieval evaluation that uses global synonym expansion, metadata query matching, and hybrid scoring.

**Architecture:** Create new `MetadataAwareRetrievalEvaluator` that extends `RetrievalEvaluator` with hybrid scoring. Add supporting classes: `QueryExpander`, `MetadataQueryMatcher`, `HybridScorer`. All in new `metadata_retrieval.py` module.

**Tech Stack:** Python, pytest, existing `semantic_similarity` function from validator.py

---

## File Structure

- **Create:** `src/pipeline/metadata_retrieval.py` - Contains QueryExpander, MetadataQueryMatcher, HybridScorer
- **Modify:** `src/pipeline/evaluator.py` - Add MetadataAwareRetrievalEvaluator class
- **Modify:** `src/pipeline/config.py` - Add retrieval weight defaults
- **Create:** `tests/test_metadata_retrieval.py` - Unit tests for new classes

---

### Task 1: Write unit tests for QueryExpander

**Files:**
- Create: `tests/test_metadata_retrieval.py`
- Create: `src/pipeline/metadata_retrieval.py` (skeleton)

- [ ] **Step 1: Create test file with QueryExpander tests**

Create `tests/test_metadata_retrieval.py` with pytest tests:

```python
import pytest
from src.pipeline.metadata_retrieval import QueryExpander

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
    assert expanded == ['test']  # or at least no duplicates
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metadata_retrieval.py::test_query_expander_with_synonyms -v`
Expected: FAIL - ModuleNotFoundError (module doesn't exist yet)

- [ ] **Step 3: Write minimal QueryExpander implementation**

Create `src/pipeline/metadata_retrieval.py` with skeleton:

```python
from typing import Dict, List

class QueryExpander:
    def __init__(self, global_synonyms: Dict[str, List[str]]):
        self.global_synonyms = global_synonyms

    def expand(self, query: str) -> List[str]:
        """Expand query by replacing terms with synonyms. Returns list of query variants."""
        variants = [query]
        words = query.split()
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?')
            if word_lower in self.global_synonyms:
                for synonym in self.global_synonyms[word_lower]:
                    new_words = words.copy()
                    new_words[i] = synonym
                    variants.append(' '.join(new_words))
        # Deduplicate while preserving order
        seen = set()
        unique_variants = []
        for v in variants:
            if v not in seen:
                seen.add(v)
                unique_variants.append(v)
        return unique_variants
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metadata_retrieval.py -v`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/metadata_retrieval.py tests/test_metadata_retrieval.py
git commit -m "feat: add QueryExpander for global query expansion"
```

---

### Task 2: Write unit tests for MetadataQueryMatcher

- [ ] **Step 1: Add tests to test_metadata_retrieval.py**

```python
from src.pipeline.metadata_retrieval import MetadataQueryMatcher

def test_metadata_query_matcher_returns_max_similarity():
    matcher = MetadataQueryMatcher()
    query = "Which category has highest profit?"
    metadata_queries = [
        "What is the top category by earnings?",
        "Which product category leads in profit?",
        "Category with maximum profit"
    ]
    # We'll mock semantic_similarity; for now assume it's imported
    from src.pipeline.validator import semantic_similarity
    # We'll test with real semantic_similarity
    similarity = matcher.similarity(query, metadata_queries)
    # Should be a float between 0 and 1
    assert 0 <= similarity <= 1
    # Should be the max similarity across all metadata queries
    # We can compute manually to verify
    max_manual = max(semantic_similarity(query, q) for q in metadata_queries)
    assert similarity == pytest.approx(max_manual)

def test_metadata_query_matcher_empty_list():
    matcher = MetadataQueryMatcher()
    similarity = matcher.similarity("test query", [])
    assert similarity == 0.0
```

- [ ] **Step 2: Run tests to verify they fail (ModuleNotFound) and then pass after implementation**

- [ ] **Step 3: Implement MetadataQueryMatcher**

In `src/pipeline/metadata_retrieval.py` add:

```python
from .validator import semantic_similarity

class MetadataQueryMatcher:
    def similarity(self, query: str, metadata_queries: List[str]) -> float:
        """Compute maximum semantic similarity between query and any metadata query."""
        if not metadata_queries:
            return 0.0
        scores = [semantic_similarity(query, meta_q) for meta_q in metadata_queries]
        return max(scores) if scores else 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/metadata_retrieval.py tests/test_metadata_retrieval.py
git commit -m "feat: add MetadataQueryMatcher with max similarity"
```

---

### Task 3: Write unit tests for HybridScorer

- [ ] **Step 1: Add tests for HybridScorer**

```python
from src.pipeline.metadata_retrieval import HybridScorer

def test_hybrid_scorer_weights_sum_to_one():
    weights = {'text': 0.6, 'metadata_query': 0.3, 'expanded': 0.1}
    scorer = HybridScorer(weights)
    assert abs(sum(scorer.weights.values()) - 1.0) < 1e-9

def test_hybrid_scorer_compute_score():
    weights = {'text': 0.6, 'metadata_query': 0.3, 'expanded': 0.1}
    scorer = HybridScorer(weights)
    # Mock document and expanded variants
    doc = {'text': 'Technology has high profit.'}
    expanded_variants = ['profit', 'earnings', 'income']
    meta_matcher = MetadataQueryMatcher()

    # We'll compute manually with known similarities
    # We can set semantic_similarity to return fixed values via monkeypatch
    from src.pipeline import validator
    original_sim = validator.semantic_similarity

    def mock_sim(a, b):
        if a == 'Which category has highest profit?' and b == 'Technology has high profit.':
            return 0.8
        if a in expanded_variants and b == 'Technology has high profit.':
            return 0.7
        if a == 'Which category has highest profit?' and b == 'Which product category leads in profit?':
            return 0.9
        return 0.0

    validator.semantic_similarity = mock_sim
    try:
        score = scorer.score('Which category has highest profit?', doc, expanded_variants, meta_matcher)
        expected = 0.6*0.8 + 0.3*0.9 + 0.1*0.7
        assert score == pytest.approx(expected)
    finally:
        validator.semantic_similarity = original_sim

def test_hybrid_scorer_with_default_weights():
    scorer = HybridScorer()  # Should use defaults if no weights provided?
    # But our design expects weights passed. Adjust test accordingly.
```

Note: We need to design HybridScorer to require weights. We'll adjust.

- [ ] **Step 2: Implement HybridScorer**

In `src/pipeline/metadata_retrieval.py`:

```python
class HybridScorer:
    def __init__(self, weights: Dict[str, float]):
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-9:
            # Normalize to 1.0
            self.weights = {k: v/total for k, v in weights.items()}
        else:
            self.weights = weights

    def score(self, query: str, document: Dict[str, Any],
              expanded_variants: List[str], meta_matcher: MetadataQueryMatcher) -> float:
        text = document.get('text', '')
        if not text:
            return 0.0
        # 1. text similarity
        text_sim = semantic_similarity(query, text)
        # 2. metadata query similarity
        metadata_queries = document.get('metadata', {}).get('queries', [])
        meta_sim = meta_matcher.similarity(query, metadata_queries)
        # 3. expanded query similarity (max over variants)
        if expanded_variants:
            expanded_scores = [semantic_similarity(variant, text) for variant in expanded_variants]
            expanded_sim = max(expanded_scores)
        else:
            expanded_sim = 0.0
        # Combine
        w = self.weights
        return (w['text'] * text_sim +
                w['metadata_query'] * meta_sim +
                w['expanded'] * expanded_sim)
```

- [ ] **Step 3: Run tests and fix any issues

- [ ] **Step 4: Commit**

```bash
git add src/pipeline/metadata_retrieval.py tests/test_metadata_retrieval.py
git commit -m "feat: add HybridScorer with weighted combination"
```

---

### Task 4: Implement MetadataAwareRetrievalEvaluator class

- [ ] **Step 1: Write tests for MetadataAwareRetrievalEvaluator**

Create `tests/test_metadata_aware_evaluator.py`:

```python
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
```

- [ ] **Step 2: Implement MetadataAwareRetrievalEvaluator in evaluator.py**

Add to `src/pipeline/evaluator.py`:

```python
from .metadata_retrieval import QueryExpander, MetadataQueryMatcher, HybridScorer

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
            # Get weights from config
            weights = self.config.get('retrieval.weights', {
                'text': 0.6,
                'metadata_query': 0.3,
                'expanded': 0.1
            })
            self.hybrid_scorer = HybridScorer(weights)

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
        # Rest is identical to parent - compute metrics...
        results = {
            'query': query,
            'num_relevant': len(relevant_doc_ids),
        }
        # [Same metric computation as parent]
        # ... (copy from RetrievalEvaluator.evaluate_query)
        # Copy lines 93-144 from parent method

        # For brevity in this plan, we'll copy the parent's metric computation code
        # [Will be implemented in detail]

        return results
```

We need to copy the metric computation logic from the parent. We'll do that in the actual implementation.

- [ ] **Step 3: Copy parent's metric computation code into the override**

Need to duplicate lines 93-144 from original `evaluate_query` but using our `scores`.

- [ ] **Step 4: Run tests to verify they pass**

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/evaluator.py tests/test_metadata_aware_evaluator.py
git commit -m "feat: add MetadataAwareRetrievalEvaluator with hybrid scoring"
```

---

### Task 5: Update config defaults

- [ ] **Step 1: Add retrieval weights to config defaults**

Modify `src/pipeline/config.py` in DEFAULTS:

```python
'retrieval': {
    'use_metadata': False,
    'weights': {
        'text': 0.6,
        'metadata_query': 0.3,
        'expanded': 0.1
    }
}
```

Add under 'evaluation' section:

```python
'evaluation': {
    'top_k': [1, 3, 5, 10]
},
'retrieval': {
    'use_metadata': False,
    'weights': {
        'text': 0.6,
        'metadata_query': 0.3,
        'expanded': 0.1
    }
}
```

- [ ] **Step 2: Write test to verify config loads retrieval values**

Add to existing config tests or create:

```python
def test_config_has_retrieval_defaults():
    from src.pipeline.config import Config
    config = Config()
    assert config.get('retrieval.use_metadata') is False
    weights = config.get('retrieval.weights')
    assert weights['text'] == 0.6
    assert weights['metadata_query'] == 0.3
    assert weights['expanded'] == 0.1
```

- [ ] **Step 3: Run tests and commit**

```bash
git add src/pipeline/config.py tests/
git commit -m "feat: add retrieval config defaults"
```

---

### Task 6: Integration test with sample documents

- [ ] **Step 1: Write integration test**

In `tests/test_metadata_integration.py`:

```python
import pytest
from src.pipeline.metadata_retrieval import MetadataAwareRetrievalEvaluator

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
    # Query that should match doc 0
    result = evaluator.evaluate_query('Which category earns the most?', {0})
    # Should retrieve doc0, hopefully high score due to synonyms
    assert result['recall@1'] > 0  # At least something retrieved

def test_fallback_to_text_only_when_use_metadata_false():
    docs = [{'text': 'Tech profit high', 'metadata': {}}]
    evaluator = MetadataAwareRetrievalEvaluator(docs, use_metadata=False)
    result = evaluator.evaluate_query('Tech profit', {0})
    assert result['recall@1'] == 1.0  # Should match exactly

def test_graceful_when_metadata_missing():
    docs = [{'text': 'some content', 'metadata': {}}]
    evaluator = MetadataAwareRetrievalEvaluator(docs, use_metadata=True)
    # Should not crash, even though no metadata
    result = evaluator.evaluate_query('content?', {0})
    assert result is not None
```

- [ ] **Step 2: Run integration tests, debug, fix

- [ ] **Step 3: Commit**

```bash
git add tests/test_metadata_integration.py
git commit -m "test: add integration tests for metadata-aware retrieval"
```

---

### Task 7: Update evaluation script to use new evaluator (optional but helpful)

Update `evaluation/comparison_evaluator.py` or `evaluation/full_comparison_evaluator.py` to add a mode that uses `MetadataAwareRetrievalEvaluator` for the optimized comparison.

But keep optional - user can invoke directly.

For now, skip - user can use the new evaluator class directly in their evaluation scripts.

---

### Task 8: Final validation tests

- [ ] **Step 1: Run all tests**

```bash
pytest tests/test_metadata_retrieval.py tests/test_metadata_aware_evaluator.py tests/test_metadata_integration.py -v
```

All tests must pass.

- [ ] **Step 2: Test with actual documents.json and test_queries.json**

Create a quick script or notebook cell:

```python
import json
from src.pipeline.evaluator import MetadataAwareRetrievalEvaluator

docs = json.load(open('output/documents.json'))
queries = json.load(open('evaluation/test_queries.json'))

# Test baseline-like behavior
evaluator_baseline = MetadataAwareRetrievalEvaluator(docs, use_metadata=False)
results_baseline = evaluator_baseline.evaluate_dataset(queries)
print("Baseline-like (use_metadata=False):", results_baseline['aggregates'])

# Test metadata-aware
evaluator_meta = MetadataAwareRetrievalEvaluator(docs, use_metadata=True)
results_meta = evaluator_meta.evaluate_dataset(queries)
print("Metadata-aware:", results_meta['aggregates'])
```

Verify that:
- use_metadata=False produces same results as old RetrievalEvaluator
- use_metadata=True produces different scores (usually higher)

- [ ] **Step 3: Commit**

```bash
git add .
git commit -m "test: final validation with real data"
```

---

## Validation Checklist

After implementation, verify:

1. ✅ Global Query Expansion: Synonym map built once from all documents
2. ✅ No OR string: Expanded terms as list, similarity aggregated via max
3. ✅ Metadata Query Matching: Uses max similarity against metadata.queries
4. ✅ Hybrid Scoring: 0.6 * text + 0.3 * meta_query + 0.1 * expanded
5. ✅ Fallback: use_metadata=False matches baseline; missing metadata handled
6. ✅ Metadata not appending to text: Only used in scoring
7. ✅ Deterministic: No randomness in scoring
8. ✅ Clean: Modular, simple, well-tested

---

**Implementation ready for evaluation after all tests pass.**