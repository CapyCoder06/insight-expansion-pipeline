#!/usr/bin/env python
"""
Demo script: Retrieval Evaluation BEFORE Embedding

This script demonstrates how to evaluate document retrieval quality
using semantic similarity before investing in embedding and vector DB operations.

Usage:
    python evaluation_demo.py
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, 'src')

from pipeline import DocumentPipeline, generate_from_insights
from pipeline.evaluator import RetrievalEvaluator, quick_benchmark, save_evaluation_results
from pipeline.config import load_config


def main():
    print("=" * 60)
    print("RETRIEVAL EVALUATION DEMO")
    print("(Pre-Embedding Quality Check)")
    print("=" * 60)

    # Load configuration
    try:
        config = load_config()
        print(f"[OK] Loaded config from: {config.config_path or 'defaults'}")
    except Exception as e:
        print(f"[WARN] Config warning: {e}")
        config = None

    # Load insights
    insights_path = Path("insights_sample.json")
    if not insights_path.exists():
        print(f"[ERR] Error: {insights_path} not found!")
        return

    with open(insights_path) as f:
        insights = json.load(f)

    print(f"[OK] Loaded {len(insights)} insights")

    # Generate documents (without enrichment for this demo)
    print("\n[1] Generating documents from insights...")
    pipeline = DocumentPipeline(enrich=False)  # Use config values
    documents = pipeline.generate(insights)
    print(f"[OK] Generated {len(documents)} documents")

    # Show sample document
    print("\nSample document:")
    print(f"  Type: {documents[0]['metadata']['type']}")
    print(f"  Text preview: {documents[0]['text'][:100]}...")

    # Define test queries with expected relevant document IDs
    # NOTE: You need to determine which doc IDs are relevant for each query
    # This is a manual step - you know which documents should answer each query
    print("\n[2] Setting up test queries...")
    test_queries = [
        {
            "query": "Which category has highest profit?",
            "relevant_doc_ids": [0],  # Document 0: Technology has highest profit
            "description": "Should retrieve the FACT about Technology's profit"
        },
        {
            "query": "Does discount affect profit?",
            "relevant_doc_ids": [4],  # Document 4: Discount >=30% causes losses
            "description": "Should retrieve anomaly about discount losses"
        },
        {
            "query": "What is profit margin?",
            "relevant_doc_ids": [9],  # Document 9: Metric definition
            "description": "Should retrieve METRIC definition"
        },
        {
            "query": "Which segment has highest margin?",
            "relevant_doc_ids": [8],  # Home Office segment highest margin
            "description": "Should retrieve fact about Home Office segment"
        },
        {
            "query": "Are there any regions with low profitability?",
            "relevant_doc_ids": [6],  # Southeast Asia low margin
            "description": "Should retrieve anomaly about Southeast Asia"
        }
    ]

    print(f"[OK] Prepared {len(test_queries)} test queries")

    # Run evaluation
    print("\n[3] Running retrieval evaluation...")
    evaluator = RetrievalEvaluator(documents, config=config)
    results = evaluator.evaluate_dataset(test_queries)

    # Print report
    print("\n" + evaluator.generate_report(results))

    # Save results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    results_path = output_dir / "retrieval_evaluation.json"
    save_evaluation_results(results, str(results_path))
    print(f"\n[OK] Results saved to: {results_path}")

    # Interpretation guide
    print("\n" + "=" * 60)
    print("INTERPRETATION GUIDE")
    print("=" * 60)
    print("""
How to read the metrics:

- Recall@k: Fraction of relevant documents found in top-k results.
  * 1.0 = all relevant docs retrieved
  * 0.0 = none found

- Precision@k: Fraction of retrieved docs that are relevant.
  * High = retrieval is accurate
  * Low = many irrelevant results

- MRR: Average of 1/(rank of first relevant doc).
  * Ranges 0..1
  * Closer to 1 = relevant docs appear at top of results

- Relevance Scores (avg_similarity@k):
  * Semantic similarity between query and retrieved docs
  * Higher = better conceptual match
  * Below 0.3 may indicate weak matching

DIAGNOSTICS:

If Recall@k is low:
  - Document text doesn't match expected query formulation
  - Missing key terms that users would search for
  - Documents may need enrichment (more variants)
  - Consider adding alternative phrasings

If Precision is low:
  - Documents are too generic / broad
  - May retrieve many docs that somewhat match but aren't relevant
  - Documents may need to be more specific

If Relevance scores are low (<0.3):
  - Documents don't contain the key semantic concepts
  - Consider improving insight definitions
  - Check if templates are stripping important context

NEXT STEPS:
- If metrics are poor, iterate on document generation (add more/better insights)
- Consider enabling enrichment to create more query variants
- Adjust template formatting to keep key retrieval terms
- Re-run evaluation after changes BEFORE embedding
""")

    print("=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
