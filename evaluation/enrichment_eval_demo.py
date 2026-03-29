#!/usr/bin/env python
"""
Enrichment Evaluation Demo

Simple wrapper around comparison_evaluator for quick testing.
"""

from pathlib import Path
import sys

# Add project root and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

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
