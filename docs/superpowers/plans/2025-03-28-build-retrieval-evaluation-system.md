# Retrieval Evaluation with Enrichment Comparison

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a retrieval evaluation system that compares query-aware enrichment against baseline, measuring recall@k and hit rate improvements.

**Architecture:** Create a standalone evaluation script that:
1. Loads insights
2. Generates documents in two modes (with enrichment / without enrichment)
3. Evaluates retrieval using pre-defined test queries
4. Compares metrics and generates a detailed report

**Tech Stack:** Python, existing pipeline modules (evaluator, enrichment, pipeline), semantic similarity

---

## File Structure

**Create:**
- `evaluation/comparison_evaluator.py` - Main script that compares enriched vs baseline retrieval
- `evaluation/test_queries.json` - Comprehensive test queries (general + specific) with ground truth
- `evaluation/README.md` - Documentation on how to run and interpret results

**Modify:**
- None required - use existing modules

**Test:**
- `tests/test_enrichment_comparison.py` - Validate the comparison evaluation logic

---

## Task 1: Create comprehensive test queries

**Files:**
- Create: `evaluation/test_queries.json`
- Test: `tests/test_enrichment_comparison.py::test_test_queries_load`

**Steps:**

- [ ] **Step 1: Design test query set**

Create a JSON file with 15-20 queries covering:
- General queries: broad, natural language questions
- Specific queries: detailed, keyword-rich searches
- Each query includes: `query` string, `relevant_doc_ids` (based on insights_sample.json), `description`

Ground truth mapping must be determined by manually reviewing the 10 insights in `insights_sample.json` and identifying which documents (after generation) should be relevant to each query.

Expected structure:
```json
[
  {
    "query": "Which category has the highest profit?",
    "relevant_doc_ids": [0],
    "description": "Should retrieve Technology fact"
  },
  {
    "query": "What causes profit loss when discount is high?",
    "relevant_doc_ids": [4],
    "description": "Should retrieve anomaly about discount losses"
  },
  ...
]
```

**Save to:** `evaluation/test_queries.json`

- [ ] **Step 2: Run test to verify JSON loads**

```bash
python -c "import json; f=open('evaluation/test_queries.json'); data=json.load(f); print(f'Loaded {len(data)} queries'); f.close()"
```

Expected: prints "Loaded X queries"

---

## Task 2: Create the comparison evaluation script

**Files:**
- Create: `evaluation/comparison_evaluator.py`
- Test: `tests/test_enrichment_comparison.py::test_load_test_queries`

**Steps:**

- [ ] **Step 3: Write the comparison evaluator module**

```python
#!/usr/bin/env python
"""
Comparison Evaluation: Enrichment vs Baseline

Tests whether query-aware enrichment improves retrieval recall.
"""

import sys
from pathlib import Path
import json
import numpy as np
from typing import List, Dict, Any

sys.path.insert(0, 'src')

from pipeline import DocumentPipeline
from pipeline.evaluator import RetrievalEvaluator, quick_benchmark, save_evaluation_results
from pipeline.config import load_config


def load_test_queries(path: str) -> List[Dict[str, Any]]:
    """Load test queries from JSON."""
    with open(path, 'r') as f:
        return json.load(f)


def generate_documents(insights: List[Dict[str, Any]], enrich: bool, config=None) -> List[Dict[str, Any]]:
    """Generate documents with enrichment on or off."""
    pipeline = DocumentPipeline(enrich=enrich, config=config)
    documents = pipeline.generate(insights)
    return documents


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
    enriched_docs = generate_documents(insights, enrich=True)
    print(f"Generated {len(enriched_docs)} documents")

    enriched_evaluator = RetrievalEvaluator(enriched_docs)
    results['enriched'] = enriched_evaluator.evaluate_dataset(test_queries)

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
```

- [ ] **Step 4: Test script loads correctly**

```bash
python evaluation/comparison_evaluator.py --help 2>&1 || echo "Script loads (no --help flag)"
```

Or manually check syntax:
```bash
python -m py_compile evaluation/comparison_evaluator.py
```

Expected: No syntax errors

---

## Task 3: Create additional test queries

**Files:**
- Modify: `evaluation/test_queries.json` (fill in all queries)
- Validate: `tests/test_enrichment_comparison.py::test_query_relevance_mapping`

**Steps:**

- [ ] **Step 5: Manually craft 15-20 test queries**

Based on the 10 insights in `insights_sample.json`, create queries:

**General queries** (broad, conversational):
- "What's the most profitable category?"
- "Are discounts hurting profits?"
- "How has sales changed over time?"
- "Which region has the worst margin?"
- "What is profit margin?"

**Specific queries** (detailed, keyword-rich):
- "Technology highest profit $664K margin 14%"
- "Furniture low margin 7% below average 11.6%"
- "Sales increased 2011 2014 CAGR 24% $2.26M $4.3M"
- "Discount >=30% causes loss -$813K 10701 orders"
- "Home Office segment highest margin 12%"
- "Consumer segment 51.5% revenue margin 11.5%"
- "Canada margin 26.6% exceptional"
- "Southeast Asia margin 2% $884K revenue"

For each query, determine which document IDs from the generated set (both baseline and enriched) should be considered relevant. Use a lenient threshold: a document is relevant if its content semantically matches the query intent (even if phrasing differs).

**Note:** Because enrichment creates variations, a query might match multiple variations of the same insight. Map accordingly (e.g., if insights_sample index 0 generates 3 enriched documents, all 3 may be relevant for a query about Technology profit).

Save the complete `evaluation/test_queries.json`.

- [ ] **Step 6: Verify all queries have non-empty relevant_doc_ids lists**

```bash
python -c "import json; q=json.load(open('evaluation/test_queries.json')); assert all(len(qq['relevant_doc_ids']) > 0 for qq in q), 'Some queries have empty relevant_doc_ids'; print(f'All {len(q)} queries have at least one relevant doc')"
```

Expected: "All X queries have at least one relevant doc"

---

## Task 4: Create comparison test

**Files:**
- Create: `tests/test_enrichment_comparison.py`
- Test: `pytest tests/test_enrichment_comparison.py -v`

**Steps:**

- [ ] **Step 7: Write unit tests for comparison logic**

```python
"""
Tests for enrichment comparison evaluation.
"""

import sys
sys.path.insert(0, 'src')

import pytest
import json
from pathlib import Path
from evaluation.comparison_evaluator import (
    load_test_queries,
    generate_documents,
    run_comparison_evaluation,
    generate_hit_rate_report
)
from pipeline.evaluator import RetrievalEvaluator


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
        assert len(docs) >= 1  # At least 1 (original), typically 3-4
        assert len(docs) <= 10  # Upper bound (3 variations kept from 8 generated)
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
        baseline_count = len(results['baseline']['per_query'])  # Actually per-query results, not doc count
        # Need to check document generation separately
        baseline_docs = generate_documents(insights, enrich=False)
        enriched_docs = generate_documents(insights, enrich=True)
        assert len(enriched_docs) > len(baseline_docs)
```

- [ ] **Step 8: Run tests and fix any failures**

```bash
cd tests && pytest test_enrichment_comparison.py -v
```

Expected: All tests pass. Fix any import path issues, missing dependencies, or logic errors.

---

## Task 5: Integrate with existing demo pipeline

**Files:**
- Create: `evaluation/enrichment_eval_demo.py` - User-friendly wrapper
- Test: `pytest evaluation/ -v` (no tests for demo, just verify runs)

**Steps:**

- [ ] **Step 9: Create convenient demo script**

```python
#!/usr/bin/env python
"""
Enrichment Evaluation Demo

Simple wrapper around comparison_evaluator for quick testing.
"""

from pathlib import Path
import sys

sys.path.insert(0, 'src')

from evaluation.comparison_evaluator import main as run_comparison

if __name__ == "__main__":
    print("Enrichment Impact Evaluation Demo")
    print("=" * 60)
    print("\nThis script compares retrieval performance:")
    print("  - Baseline: documents without enrichment")
    print("  - Enriched: documents with query-aware variations")
    print("\nMetrics: recall@k, hit rate, MRR")
    print("\nStarting evaluation...\n")
    run_comparison()
```

- [ ] **Step 10: Verify demo script runs successfully**

```bash
python evaluation/enrichment_eval_demo.py
```

Expected: Script completes, prints comparison report, saves results to `output/`.

---

## Task 6: Add documentation

**Files:**
- Create: `evaluation/README.md`

**Steps:**

- [ ] **Step 11: Write README**

```markdown
# Retrieval Evaluation: Enrichment Comparison

Evaluates whether query-aware enrichment improves retrieval recall and hit rate.

## Quick Start

```bash
# 1. Ensure test queries are defined
# (edit evaluation/test_queries.json to customize)

# 2. Run evaluation
python evaluation/enrichment_eval_demo.py

# 3. View results
# - output/comparison_results.json (metrics comparison)
# - output/enrichment_comparison_report.txt (human-readable report)
# - output/comparison_baseline.json (full baseline results)
# - output/comparison_enriched.json (full enriched results)
```

## How It Works

1. **Loads insights** from `insights_sample.json`
2. **Generates documents** in two modes:
   - Baseline: 1 document per insight (no enrichment)
   - Enriched: ~3 variations per insight (query-aware phrasing)
3. **Evaluates retrieval** using `RetrievalEvaluator` with semantic similarity
4. **Compares metrics** (recall@k, hit rate, MRR) between the two sets

## Test Queries

Edit `evaluation/test_queries.json` to define your test suite.

Each query requires:
- `query`: The search query string
- `relevant_doc_ids`: List of document indices that should be retrieved (0-based)
- `description`: Optional; notes about expected behavior

**Tips for creating good test queries:**
- Mix general and specific phrasing
- Include questions a real user would ask
- Cover all document types (fact, trend, anomaly, metric)
- Manually verify relevance mappings after document generation

## Metrics Explained

- **Recall@k**: Fraction of relevant documents found in top-k results. Higher is better.
- **Hit Rate**: Fraction of queries that retrieve ≥1 relevant document in top-k. Higher is better.
- **MRR** (Mean Reciprocal Rank): Average of 1/(first relevant rank). Closer to 1 is better.

## Interpretation

If enrichment improves metrics:
- Query-aware variations help retrieval
- Keep enrichment enabled in production

If enrichment degrades metrics:
- Variations may dilute content or introduce noise
- Review enrichment strategy (synonyms, query patterns)
- Consider increasing `enrichment.variations_to_keep` or tuning filters

If mixed results:
- Enrichment helps some queries but not others
- Analyze per-query results to identify patterns
- May need query-specific enrichment rules
```

- [ ] **Step 12: Verify README renders correctly (check for broken formatting)**

---

## Task 7: Final integration and verification

**Files:**
- No new files, just validation

**Steps:**

- [ ] **Step 13: Run full evaluation demo end-to-end**

```bash
python evaluation/enrichment_eval_demo.py
```

Verify:
- No errors
- Output files created in `output/`
- Report prints meaningful comparison

- [ ] **Step 14: Run all tests**

```bash
pytest tests/ -v
```

Expected: All tests pass (including existing tests)

- [ ] **Step 15: Add new test files to git tracking** (if using git)

```bash
git add evaluation/
git add tests/test_enrichment_comparison.py
git commit -m "feat(evaluation): add enrichment comparison system with comprehensive test queries"
```

---

## Self-Review

**Spec coverage check:**

Build retrieval evaluation system ✅
- Created `comparison_evaluator.py` that compares baseline vs enriched
- Generates reports with recall@k and hit rate

Test if query-aware enrichment improves recall ✅
- System explicitly tests enrichment on/off
- Computes recall@k for both and shows difference
- Reports improvement percentages

Steps:
1. Create test queries (general + specific) ✅
2. Retrieve top-k documents ✅ (RetrievalEvaluator does this)
3. Check if correct document appears ✅ (relevant_doc_ids matching)
Metrics: recall@k, hit rate ✅
Compare: with enrichment vs without ✅

All requirements covered.

**Placeholder scan:**
- No "TBD" or "TODO" in implementation code
- All code blocks are complete and executable
- Test queries need manual crafting (Step 5) - this is intentional and documented
- All file paths exact

**Type consistency:**
- `relevant_doc_ids` consistently `List[int]` in JSON, converted to `Set[int]` in evaluator
- `generate_documents` returns `List[Dict]` consistently
- Comparison results structure documented and tested

**Ready for execution.**