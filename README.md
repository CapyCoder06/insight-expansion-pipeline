# RAG Document Pipeline: Insights Expansion

A semi-automated pipeline that transforms seed insights into a scalable, structured insight corpus.

## 🔥 Key Highlights

- **Pattern-based expansion**: Generates diverse insights from seed patterns
- **Signature deduplication**: Ensures uniqueness while preserving semantic diversity
- **Validation**: All insights validated against the dataset for accuracy
- **Target-driven**: Configurable corpus size (e.g., 10 → 100 insights)

## 📊 Results

- **10 seed insights** → **95 unique expanded insights**
- Near-duplicate elimination with 0.85 similarity threshold
- Expert-validated quality and dataset accuracy

## 🧠 Core Idea

Small sets of expert-validated insights are systematically expanded into larger, diverse corpora using pattern-based generation. This creates training data, evaluation sets, or insight corpora ready for downstream processing by teammates or external systems.

## Overview

This project implements a **focused pipeline** that expands structured insights using rule-based pattern generation and intelligent deduplication.

The expanded insights are ready for downstream processing by teammates or external systems.

---

## Insight Expansion

The insight expansion stage transforms a small set of seed insights (typically 10) into a larger, diverse corpus of structured insights using rule-based pattern generation and intelligent deduplication.

### Purpose

Scale expert-validated seed insights into a comprehensive insight corpus using:
- **Pattern-based generation**: Data-driven transformations (rankings, comparisons, aggregations, gap analysis, thresholds)
- **Signature deduplication**: Prevents redundant insights while preserving semantic diversity
- **Validation**: All insights validated against the dataset to ensure accuracy

### Pipeline Flow

```
insights_sample.json (10 seeds)
        ↓
expand_insights.py (pattern-based expansion)
        ↓
insights_expanded.json (95 insights) ← final output
```

The expanded insights are ready for downstream processing by teammates or external systems.

### How to Run

```bash
# Run expansion
python expand_insights.py --input insights_sample.json

# Or with custom target
python expand_insights.py --input insights_sample.json --target 100
```

### Configuration (config.yaml)

```yaml
expansion:
  target_count: 100           # Target number of unique insights
  dedup_threshold: 0.85       # Similarity threshold for near-duplicate removal
  output_file: output/insights_expanded.json
```

### Output Format

The output `insights_expanded.json` has the same structure as `insights_sample.json`:

```json
{
  "insights": [
    {
      "text": "Technology has the highest profit of $664K with margin 14%",
      "dimensions": ["category"],
      "metrics": ["profit", "margin"],
      "type_hint": "fact"
    },
    ...
  ]
}
```

All insights are validated against the dataset and deduplicated using signature-based deduplication to ensure uniqueness while preserving diversity.


## Data Format

### Insight Structure

The pipeline consumes and produces JSON with the following format:

```json
{
  "insights": [
    {
      "text": "Technology has the highest profit",
      "dimensions": ["category"],
      "metrics": ["profit"],
      "type_hint": "fact"      // optional
    }
  ]
}
```

Optional fields vary by insight type (e.g., `trend`, `issue`, `definition`, `question`, `answer`).

## Project Structure

```
.
├── src/
│   └── insight_expansion/ # Insight expansion modules
│       ├── __init__.py
│       ├── insight_generator.py  # Pattern-based expansion (rankings, comparisons, etc.)
│       └── insight_validator.py  # Validation & signature deduplication
│
├── tests/
│   └── insight_expansion/
│       ├── test_insight_generator.py
│       └── test_insight_validator.py
│
├── docs/                  # Design docs and planning notes
│   └── superpowers/
│       ├── plans/
│       └── specs/
│
├── eda/                   # (Note) Exploratory data analysis - source of structured insights
│   └── hypothesis_eda.ipynb
│
├── output/                # Pipeline outputs
│   └── insights_expanded.json  # Expanded insights (final output)
│
├── config.yaml            # Expansion configuration
├── expand_insights.py     # Main expansion script
├── insights_sample.json   # Sample seed insights (10 seeds)
├── requirements.txt
└── SYSTEM_DESIGN.md       # System design document (formerly CLAUDE.md)
```

## 📦 Sample Output

This repository includes a sample output from the pipeline:

- `output/insights_expanded.json`: expanded structured insights ready for downstream processing

The output contains 95 diverse insights generated from 10 seed examples.

Example structure:

```json
{
  "insights": [
    {
      "text": "Technology has the highest profit of $664K with margin 14%",
      "dimensions": ["category"],
      "metrics": ["profit", "margin"],
      "type_hint": "fact"
    },
    ...
  ]
}
```

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Demo

```bash
python expand_insights.py
```

This loads sample insights, expands them using pattern-based generation, validates the results, and saves to `output/insights_expanded.json`.

Run configuration is managed via `config.yaml`:

```yaml
expansion:
  target_count: 100
  dedup_threshold: 0.85
  output_file: output/insights_expanded.json
```


---

## Validation

All expanded insights are validated against the source dataset to ensure:

- **Semantic accuracy**: Insight text matches data patterns
- **No hallucinations**: All numbers and claims are grounded in data
- **Deduplication**: Near-duplicate insights are removed (similarity threshold: 0.85)

The validation step ensures the expanded corpus maintains quality and diversity.


---

## Design Principles

- **Pattern-based expansion**: Generates insights using data transformations (rankings, comparisons, aggregations, gap analysis, thresholds)
- **Signature deduplication**: Prevents redundant insights while preserving semantic diversity
- **Dataset validation**: All insights are verified against the source data
- **Configurability**: Target count and deduplication threshold adjustable via config.yaml
- **Deterministic core**: Same seed → consistent output (except for optional randomness in variations)

---

## Integration with Downstream Systems

The `insights_expanded.json` output is ready for downstream processing by teammates or external systems that handle:

- Document generation
- Chunking
- Embedding
- Vector database upsert (e.g., Pinecone)
- Retrieval

This pipeline focuses solely on creating a high-quality, diverse corpus of structured insights from seed data.

---

## Testing

```bash
pytest tests/insight_expansion/ -v
```

Tests cover:
- Pattern-based insight generation (rankings, comparisons, etc.)
- Signature deduplication and uniqueness validation
- Dataset accuracy verification

---

## EDA: Source of Insights

The `eda/hypothesis_eda.ipynb` notebook performs exploratory data analysis on the dataset to **generate structured insights**. These insights serve as the input to this pipeline.

This Jupyter notebook is the **manual/hypothesis-driven EDA** step that produces the JSON-formatted insights this pipeline consumes.

---


---

## License

[Specify license if applicable]
