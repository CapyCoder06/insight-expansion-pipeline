# Metadata-Aware Retrieval Evaluation Design

**Date:** 2025-03-28
**Status:** Approved
**Related:** retrieval evaluation, hybrid search, metadata

---

## Overview

Enhance the `RetrievalEvaluator` to use document metadata for improved retrieval quality. The current evaluator only uses text similarity; we need to incorporate:

- Global query expansion using synonyms from all documents' metadata
- Matching against pre-generated metadata queries
- Hybrid scoring combining multiple similarity signals

**Goal:** Make retrieval optimization actually effective by simulating hybrid search.

---

## Key Design Decisions

### 1. Global Synonym Map

**Problem:** Synonyms should be applied globally, not per-document.

**Solution:** Build a unified synonym dictionary from ALL documents' `metadata["synonyms"]` before evaluation begins.

```python
# Collect synonyms from all documents
global_synonyms = {}
for doc in documents:
    doc_synonyms = doc.get("metadata", {}).get("synonyms", {})
    for term, syns in doc_synonyms.items():
        global_synonyms.setdefault(term, []).extend(syns)

# Deduplicate
for term in global_synonyms:
    global_synonyms[term] = list(set(global_synonyms[term]))
```

**Benefits:**
- Query expanded once upfront (efficient)
- Consistent expansion across all documents
- Simpler than per-document expansion

---

### 2. Query Expansion Without OR Logic

**Problem:** `semantic_similarity()` works on text, not boolean queries. Using "profit OR earnings" doesn't work.

**Solution:** Expand query into multiple concrete variants, compute similarity for each, then aggregate:

```python
def expand_query_globally(query: str, global_synonyms: Dict[str, List[str]]) -> List[str]:
    """Generate query variants by replacing terms with synonyms."""
    variants = [query]  # Always include original

    words = query.split()
    for i, word in enumerate(words):
        word_lower = word.lower().strip('.,!?')
        if word_lower in global_synonyms:
            for synonym in global_synonyms[word_lower]:
                variant_words = words.copy()
                variant_words[i] = synonym
                variants.append(' '.join(variant_words))

    return list(set(variards))  # Deduplicate
```

**Scoring aggregated expanded similarity:**

```python
expanded_scores = [semantic_similarity(expanded_query, doc_text) for expanded_query in expanded_variants]
expanded_sim = max(expanded_scores)  # Use max to capture best match
# Alternatively: average(expanded_scores)
```

**Rationale:** Using `max` ensures we get credit if ANY synonym variant matches well. This is more lenient and appropriate for retrieval.

---

### 3. Metadata Query Matching

For each document, compare user query against its `metadata["queries"]`:

```python
def compute_metadata_query_similarity(query: str, metadata_queries: List[str]) -> float:
    """Maximum similarity between query and any metadata query."""
    if not metadata_queries:
        return 0.0
    scores = [semantic_similarity(query, meta_q) for meta_q in metadata_queries]
    return max(scores)
```

**Why max?** If the user's query matches ANY alternative phrasing of what this document answers, that's a hit.

---

### 4. Hybrid Scoring Formula

```python
final_score = (
    0.6 * text_similarity(query, doc_text) +
    0.3 * metadata_query_similarity(query, doc.get("metadata", {}).get("queries", [])) +
    0.1 * expanded_query_similarity(query, doc_text, global_synonyms)
)
```

**Weights rationale:**
- **0.6 text similarity:** Primary signal - actual document content must match
- **0.3 metadata queries:** Secondary signal - captures alternate phrasings of user intent
- **0.1 expanded query:** Tertiary signal - vocabulary expansion, broader matching

Weights are configurable via config file.

---

## Architecture

### New Classes

**QueryExpander** (`src/pipeline/metadata_retrieval.py`)
```python
class QueryExpander:
    def __init__(self, global_synonyms: Dict[str, List[str]]):
        self.global_synonyms = global_synonyms

    def expand(self, query: str) -> List[str]:
        """Return list of query variants (original + synonym replacements)."""
        # Implementation as above
```

**MetadataQueryMatcher** (`src/pipeline/metadata_retrieval.py`)
```python
class MetadataQueryMatcher:
    def similarity(self, query: str, metadata_queries: List[str]) -> float:
        """Max similarity to any metadata query."""
        # Implementation as above
```

**HybridScorer** (`src/pipeline/metadata_retrieval.py`)
```python
class HybridScorer:
    def __init__(self, weights: Dict[str, float]):
        self.weights = weights  # {"text": 0.6, "metadata_query": 0.3, "expanded": 0.1}
        self._validate_weights()

    def score(self, query: str, document: Dict[str, Any],
              expanded_variants: List[str], meta_matcher: MetadataQueryMatcher) -> float:
        """Compute hybrid score for a document."""
        text_sim = semantic_similarity(query, document["text"])
        meta_sim = meta_matcher.similarity(query, document.get("metadata", {}).get("queries", []))
        expanded_sim = max(semantic_similarity(variant, document["text"]) for variant in expanded_variants)
        return (self.weights["text"] * text_sim +
                self.weights["metadata_query"] * meta_sim +
                self.weights["expanded"] * expanded_sim)
```

### Modified Component

**MetadataAwareRetrievalEvaluator** (`src/pipeline/evaluator.py`)
```python
class MetadataAwareRetrievalEvaluator(RetrievalEvaluator):
    def __init__(self, documents, config=None, use_metadata=True):
        super().__init__(documents, config)
        self.use_metadata = use_metadata
        self.global_synonyms = self._build_global_synonym_map()
        self.query_expander = QueryExpander(self.global_synonyms)
        self.meta_matcher = MetadataQueryMatcher()
        self.hybrid_scorer = HybridScorer(
            weights=config.get('retrieval.weights', {
                'text': 0.6,
                'metadata_query': 0.3,
                'expanded': 0.1
            })
        )

    def _build_global_synonym_map(self) -> Dict[str, List[str]]:
        """Collect synonyms from all documents."""
        # Implementation as described

    def evaluate_query(self, query, relevant_doc_ids, top_k=None, return_details=False):
        """Override to use hybrid scoring."""
        if not self.use_metadata:
            return super().evaluate_query(query, relevant_doc_ids, top_k, return_details)

        # Expand query globally once
        expanded_variants = self.query_expander.expand(query)

        # Compute scores with hybrid scoring
        scores = []
        for doc_id, doc in self.doc_index.items():
            score = self.hybrid_scorer.score(query, doc, expanded_variants, self.meta_matcher)
            scores.append((doc_id, score))

        # Sort and compute metrics (same as parent)
        scores.sort(key=lambda x: x[1], reverse=True)
        # ... rest same as parent implementation
```

---

## Edge Cases & Error Handling

| Scenario | Handling |
|----------|----------|
| Document missing `metadata["queries"]` | Treat as empty list → meta_query_sim = 0 |
| Document missing `metadata["synonyms"]` | Ignored during global map building |
| No synonyms found globally | expanded_variants = [original query] only |
| Empty expanded_variants list | Fallback to original query |
| `semantic_similarity` returns NaN/None | Treat as 0.0 for that component |
| Weights don't sum to 1.0 | Normalize automatically, log warning |

**Graceful degradation:** If metadata is incomplete, the system falls back to text similarity (weight 1.0 effectively for missing components).

---

## Configuration

Add to config (e.g., `config.yaml` or defaults):

```yaml
retrieval:
  use_metadata: true
  weights:
    text: 0.6
    metadata_query: 0.3
    expanded: 0.1
```

In code (`config.py`):
```python
DEFAULT_CONFIG = {
    'retrieval': {
        'use_metadata': False,  # Default off for backward compatibility
        'weights': {'text': 0.6, 'metadata_query': 0.3, 'expanded': 0.1}
    }
}
```

---

## Implementation Plan

1. **Create `src/pipeline/metadata_retrieval.py`**
   - `QueryExpander` class
   - `MetadataQueryMatcher` class
   - `HybridScorer` class

2. **Modify `src/pipeline/evaluator.py`**
   - Add `MetadataAwareRetrievalEvaluator` class
   - Override `evaluate_query()` method
   - Add `_build_global_synonym_map()` helper

3. **Add configuration**
   - Update `config.py` with retrieval weights defaults
   - Ensure config loading works

4. **Update evaluation scripts** (optional)
   - `full_comparison_evaluator.py` can be updated to use `MetadataAwareRetrievalEvaluator` for the "optimized" comparison
   - Or keep as-is (users can invoke the new evaluator explicitly)

5. **Testing**
   - Unit tests for each new class
   - Integration test with sample documents
   - Verify improved recall on test queries

---

## Success Criteria

- ✅ Hybrid scoring produces different rankings than text-only
- ✅ Documents with `metadata["queries"]` receive boost when user query matches
- ✅ Documents with rich `metadata["synonyms"]` match more query variants
- ✅ Evaluation runs successfully on full comparison (baseline → enriched → optimized)
- ✅ Improved recall@1 and MRR on test queries when using optimized documents with metadata

---

## Out of Scope

- Modifying the document generation/enrichment pipeline
- Changing metadata structure (we use existing `queries` and `synonyms` fields)
- Real vector database integration (this is pre-embedding evaluation)
- Feature flags beyond `use_metadata` toggle
