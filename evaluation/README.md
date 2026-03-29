# Retrieval Evaluation: Enrichment Comparison

Evaluates whether query-aware enrichment improves retrieval recall and hit rate.

## Quick Start

```bash
# 1. Ensure test queries are defined
# (edit evaluation/test_queries.json to customize)

# 2. Run evaluation
python evaluation/enrichment_eval_demo.py

# 3. View results
# - output/comparison_baseline.json (full baseline results)
# - output/comparison_enriched.json (full enriched results)
# - output/comparison_results.json (metrics comparison)
# - output/enrichment_comparison_report.txt (human-readable report)
```

## How It Works

1. **Loads insights** from `insights_sample.json`
2. **Generates documents** in two modes:
   - Baseline: 1 document per insight (no enrichment)
   - Enriched: ~3 query-aware variations per insight
3. **Evaluates retrieval** using `RetrievalEvaluator` with semantic similarity
4. **Compares metrics** (recall@k, hit rate, MRR) between the two sets

## Test Queries

Edit `evaluation/test_queries.json` to define your test suite.

Each query requires:
- `query`: The search query string
- `relevant_doc_ids`: List of **baseline** document indices (0-based) that should be retrieved
- `description`: Optional; notes about expected behavior

**Important:** The `relevant_doc_ids` refer to baseline document indices (0-9 for the 10 sample insights). The evaluation system automatically maps these to all enriched variations during the enriched evaluation.

### Crafting Good Test Queries

- **Mix general and specific phrasing**
  - General: "Which category has highest profit?"
  - Specific: "Technology highest profit $664K margin 14%"

- **Cover all document types** (fact, trend, anomaly, metric)

- **Manually verify** after first run that relevant documents truly match query intent

- Avoid overly ambiguous queries that might match multiple insights

## Metrics Explained

- **Recall@k**: Fraction of relevant documents found in top-k results. Range 0..1. Higher is better.
- **Hit Rate**: Fraction of queries that retrieve ≥1 relevant document in top-k. Range 0..1. Higher is better.
- **MRR** (Mean Reciprocal Rank): Average of 1/(rank of first relevant document). Closer to 1 is better.

## Interpretation

The output report provides:

- **Per-metric comparisons** (baseline vs enriched)
- **Absolute and relative improvements**
- **Automated conclusion** based on thresholds

### What the results mean

- **Significant improvement** (MRR +5% or Recall@1 +10%): Enrichment is working well.
- **Modest improvement** (positive but below thresholds): Consider trade-offs (storage, cost).
- **No improvement or degradation**: Review enrichment strategy, query patterns, or variation count.

### Common reasons enrichment may not help

1. **Insufficient variation quality** - synonyms/rephrasing may not match query distribution
2. **Too few variations** - increase `enrichment.variations_to_keep` in config.yaml
3. **Query set is too narrow** - test queries may already match baseline phrasing exactly
4. **Over-enrichment** - variations may dilute signal; try selecting top 2 instead of 3

## Configuration

Adjust enrichment parameters in `config.yaml`:

```yaml
enrichment:
  enabled: false          # pipeline-level switch (does not affect this eval)
  variations_to_generate: 8   # how many raw variations to create
  variations_to_keep: 3       # how many top variations to retain
```

The evaluation always runs with enrichment enabled for the "enriched" run and disabled for "baseline", regardless of config.

## Output Files

| File | Contents |
|------|----------|
| `output/comparison_baseline.json` | Full evaluation results for baseline (no enrichment) |
| `output/comparison_enriched.json` | Full evaluation results for enriched |
| `output/comparison_results.json` | Comparison metrics only (machine-readable) |
| `output/enrichment_comparison_report.txt` | Human-readable formatted report |

## Extending the Evaluation

- **Add more test queries** to `test_queries.json` for broader coverage
- **Customize metrics** by editing `run_comparison_evaluation()` in `comparison_evaluator.py`
- **Include additional k values** by modifying the `top_k` list in the config or hardcoded loops
- **Analyze per-query differences** by inspecting the `per_query` arrays in the JSON results

## Troubleshooting

**Import errors when running demo:**
- Ensure `evaluation/__init__.py` exists
- The script adds project root to `sys.path`; if moved, update paths

**Enrichment seems to worsen results:**
- Check whether the test queries are representative of real user queries
- Examine per-query results to see which queries failed
- Review the generated enriched documents in `output/comparison_enriched.json` to ensure they preserve meaning

**No output files generated:**
- Make sure `output/` directory exists (it should be created automatically)
- Check file write permissions
