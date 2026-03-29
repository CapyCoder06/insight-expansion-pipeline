# RAG Document Generation Pipeline

## Overview

Semi-automated pipeline for converting structured insights into standardized documents for Retrieval-Augmented Generation (RAG).

Based on the CLAUDE.md system design, this implementation covers **Steps 4-9**:
1. **Template Mapping** - Map insights to document types (FACT, TREND, ANOMALY, METRIC, QUESTION)
2. **Document Generation** - Apply templates to create standardized documents
3. **Enrichment** - Generate 5-10 semantic variations per document (optional)
4. **Variation Selection** - Filter top 3 highest quality variations
5. **Validation** - Ensure quality and prevent hallucinations
6. **Metadata Generation** - Add structured metadata for retrieval filtering
7. **Chunking** - Split documents into 200-500 token chunks for embedding

## Architecture

```
Insights (JSON)
    ↓
Template Mapping (rule-based classification)
    ↓
Document Generation (template rendering)
    ↓
[Optional] Enrichment (generate 5-10 variations)
    ↓
Selection (filter top 3 variations)
    ↓
Validation (quality checks)
    ↓
Chunking (semantic-aware splitting)
    ↓
Chunks (ready for embedding → Pinecone → RAG)
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Usage

#### Using the full pipeline

```python
from pipeline import DocumentPipeline

# Load your insights
insights = [
    {
        "text": "Technology has the highest profit",
        "dimensions": ["category"],
        "metrics": ["profit"]
    },
    # ... more insights
]

# Run pipeline
pipeline = DocumentPipeline(chunk_size=200, chunk_overlap=50)
result = pipeline.run(insights, validate=True, chunk=True)

# Get results
documents = result["documents"]
chunks = result["chunks"]
validation_summary = result["validation"]
```

#### Generate only (skip validation & chunking)

```python
from pipeline import generate_from_insights

documents = generate_from_insights(insights)
```

#### Individual components

```python
from pipeline import DocumentGenerator, enhance_for_retrieval

# Generate document
generator = DocumentGenerator()
doc = generator.generate_document(insight)

# Enhance for retrieval (adds query-style questions and synonyms)
enhanced_text = enhance_for_retrieval(doc["text"])
doc["text"] = enhanced_text

# Validate
from pipeline import DocumentValidator
validator = DocumentValidator()
result = validator.validate(doc)

# Chunk
from pipeline import DocumentChunker
chunker = DocumentChunker(chunk_size=200, overlap=50)
chunks = chunker.chunk_document(doc)
```

### Command line demo

```bash
python demo_pipeline.py
```

This will:
- Load sample insights from `insights_sample.json`
- Generate documents
- Validate them
- Chunk them
- Save outputs to `output/documents.json` and `output/chunks.json`

## Document Types

| Type | Use Case | Template |
|------|----------|----------|
| **FACT** | Static facts, comparisons, rankings | `[FACT] Title`, dimensions, metrics, grain |
| **TREND** | Time-series patterns, growth/decline | `[TREND] Title`, dimension, metrics, grain, trend direction |
| **ANOMALY** | Outliers, negative metrics, problems | `[ANOMALY] Title`, dimension, metric, issue, possible_cause |
| **METRIC** | Definitions and calculations | `[METRIC] metric_name`, definition, formula, interpretation |
| **QUESTION** | Q&A pairs | `[QUESTION] Q: ... A: ...` |

### Template Mapping Logic

The `TemplateMapper` automatically selects document types using:

1. **Explicit hint** - If insight includes `type_hint` field, it's used directly
2. **Keyword analysis** - Scores based on keywords in the insight text:
   - FACT: highest, lowest, best, worst, top, bottom, etc.
   - TREND: increase, decrease, growth, CAGR, YoY, etc.
   - ANOMALY: negative, loss, anomaly, outlier, problem, etc.
   - METRIC: margin, definition, formula, calculate
   - QUESTION: why, how, what, ?
3. **Special rules** - For profit metrics with negative values → ANOMALY

### Validation Rules

Each document must:
- Have non-empty text
- Include correct `type` in metadata
- Have proper type prefix in text (e.g., `[FACT]`)
- Include dimensions or metrics (for retrieval)
- Not contain hallucination markers ("I don't know", etc.)

### Chunking

### Chunking

Documents are split into chunks of 200-500 tokens (words) with overlap:
- Sentences boundaries respected when possible
- Metadata preserved and attached to each chunk
- Each chunk gets `chunk_index` and `chunk_count` fields

### Retrieval Optimization

The `enhance_for_retrieval()` function improves document searchability by adding:

- **Query-style questions** - Natural language questions users might ask
- **Synonym alternatives** - Key term synonyms to improve match recall
- **Alternative phrasings** - Different ways to express the same fact

Example:
```python
from pipeline import enhance_for_retrieval

doc_text = "# [FACT] Technology has highest profit\n\nTechnology leads with $664K profit."
enhanced = enhance_for_retrieval(doc_text)
```

Output:
```
# [FACT] Technology has highest profit

Technology leads with $664K profit.

# Retrieval Queries
- Which category has the highest profit?
- What is the top category by profit?
- Which category leads in profit?

# Alternative Terms
- profit: earnings, income, gains
- highest: maximum, peak, top
```

**Constraints:**
- Original meaning preserved (no new facts)
- Concise output (only essential additions)
- Works for all document types (FACT, TREND, ANOMALY, METRIC, QUESTION)

## Enrichment

The enrichment module expands each standardized document into multiple semantic variations to improve retrieval recall. This implements the Hybrid Enrichment strategy from CLAUDE.md.

### How It Works

1. **Input**: 1 standardized document
2. **Generation**: Produce 5-10 variations using:
   - Synonym substitution (controlled vocabulary)
   - Sentence restructuring (active/passive, reordering)
   - Framing variations (different reporting verbs: "Analysis shows", "Data indicates", etc.)
   - Type-specific templates (TREND, ANOMALY, QUESTION have custom patterns)
3. **Selection**: Filter top 3 highest quality variations

### Constraints

- **No hallucination** - All content derived from original; no new facts added
- **Meaning preservation** - Numeric values, entities, and relationships remain identical
- **Metadata consistency** - All variations share identical metadata structures

### Usage

```python
from pipeline import DocumentPipeline, expand_document, select_top_variations

# Option 1: Enable enrichment in pipeline
pipeline = DocumentPipeline(enrich=True, enrich_variations=3)
result = pipeline.run(insights)  # Returns 3 variants per insight

# Option 2: Use enrichment module directly
doc = generator.generate_document(insight)
variations = expand_document(doc)  # 5-10 variations
top3 = select_top_variations(variations, top_k=3)  # Keep best 3
```

### Quality Filtering

`select_top_variations` uses heuristics:
- Prioritizes original document (index 0)
- Rewards appropriate length (50-500 chars)
- Penalizes formatting errors (multiple spaces, empty text)
- Ensures diversity while maintaining quality

## Hybrid Retrieval Optimization

The system implements a sophisticated hybrid retrieval strategy that combines multiple scoring dimensions to improve document ranking accuracy. This approach addresses the vocabulary mismatch problem inherent in semantic search.

### How It Works

Hybrid retrieval computes a weighted score from three components:

1. **Text Similarity** (weight: 0.5)
   - Direct semantic similarity between query and document text
   - Provides foundation for content matching

2. **Metadata Query Matching** (weight: 0.4)
   - Matches query against metadata.queries (natural language questions generated for each document)
   - Captures intent alignment even when wording differs
   - Includes optional similarity bonus: when metadata similarity exceeds threshold (0.6), apply boost factor (1.1x)

3. **Synonym Expansion** (weight: 0.1)
   - Expands query terms using synonyms from metadata.synonyms
   - Improves recall for alternative phrasings

The final score is normalized to [0, 1] to ensure fair comparison across documents.

### Configuration

The retrieval configuration is defined in `config.yaml` under `retrieval_versions.retrieval_v1_stable`:

```yaml
retrieval_versions:
  retrieval_v1_stable:
    weights:
      text: 0.5
      metadata_query: 0.4
      expanded: 0.1
    metadata_bonus:
      enabled: true
      threshold: 0.6
      boost_factor: 1.1
```

### Evaluation Results

A comprehensive evaluation using 70 test queries compared baseline (text-only) vs optimized (hybrid) retrieval:

**Precision Improvements (Top-k results):**
- Precision@3: +52.2% (0.247 → 0.376)
- Precision@5: +65.6% (0.247 → 0.376)
- Precision@10: +101.0% (0.100 → 0.201)

**Trade-off:**
- Recall decreases slightly (documents become more selective in top positions)
- This trade-off is desirable: higher precision means users see more relevant results first

**Key Insight:**
The hybrid approach successfully surfaces more relevant documents in the top result positions, making it suitable for applications where the first few results matter most (e.g., question answering, chatbots).

### Metadata-Aware Evaluator

The `MetadataAwareRetrievalEvaluator` class implements the hybrid scoring during evaluation. It:
- Builds a global synonym map from all document metadata
- Expands queries with synonyms
- Matches query against document metadata.queries
- Applies the weighted hybrid scoring with normalization

This allows accurate simulation of production retrieval behavior before embedding.

## Data Format

### Input Insight Structure

```json
{
  "text": "Technology has the highest profit",
  "dimensions": ["category"],
  "metrics": ["profit"],
  "type_hint": "fact",      // optional
  "grain": "grouped by category",  // optional, auto-inferred
  "trend": "increasing",    // for TREND type
  "issue": "...",            // for ANOMALY type
  "possible_cause": "...",   // for ANOMALY type
  "definition": "...",       // for METRIC type
  "formula": "...",          // for METRIC type
  "interpretation": "...",   // for METRIC type
  "question": "...",         // for QUESTION type
  "answer": "..."            // for QUESTION type
}
```

### Output Document Structure

```json
{
  "text": "# [FACT] ...\n\n...\n- Dimensions: ...\n- Metrics: ...",
  "metadata": {
    "type": "fact",
    "dimensions": ["category"],
    "metrics": ["profit"],
    "title": "...",
    "grain": "grouped by category"
  }
}
```

### Output Chunk Structure

```json
{
  "text": "chunk content...",
  "metadata": {
    "type": "fact",
    "dimensions": ["category"],
    "metrics": ["profit"],
    "_doc_id": 0,
    "chunk_index": 0,
    "chunk_count": 2
  }
}
```

## File Structure

```
.
├── src/pipeline/
│   ├── __init__.py          # Pipeline orchestrator
│   ├── types.py             # DocumentType enum
│   ├── templates.py         # Template definitions
│   ├── template_mapper.py   # Insight → DocumentType mapping
│   ├── document_generator.py # Template rendering
│   ├── enrichment.py        # Variation generation & selection
│   ├── validator.py         # Quality validation
│   └── chunker.py           # Document chunking
├── tests/
│   ├── test_pipeline.py     # Pipeline unit tests (20)
│   └── test_enrichment.py   # Enrichment tests (12)
├── demo_pipeline.py         # Demo script
├── insights_sample.json     # Sample input
├── requirements.txt
└── CLAUDE.md                # System design
```

## Design Principles

- **No hallucination** - All content derived from input insight, no generation
- **Preserve meaning** - Templates maintain original insight intent
- **Optimized for retrieval** - Metadata includes dimensions, metrics, types
- **Human-in-the-loop** - Template mapping can be overridden via `type_hint`

## Testing

```bash
pytest tests/ -v
```

All **32 tests** should pass (20 pipeline + 12 enrichment).

## Constraints

- No autonomous knowledge generation - scales and structures human insights only
- Faithful to original insight - no interpretation beyond structure
- Templates are deterministic - same insight → same output
- Validation prevents missing required fields and common hallucination markers

## Integration

After chunking, the output should be:
1. Embedded (by teammate)
2. Upserted to Pinecone with metadata (by teammate)
3. Used for RAG retrieval (by teammate)

See CLAUDE.md for full pipeline context.

## Metadata Ranking Signals

The document generator now adds three ranking fields:

- **importance**: `"low"`, `"medium"`, `"high"` (auto-inferred from content keywords)
- **confidence**: `0.0–1.0` (initial score + updated by validation)
- **source**: `"EDA"` (default), or set via `insight["source"]`

Example metadata:
```json
{
  "type": "fact",
  "dimensions": ["category"],
  "metrics": ["profit"],
  "importance": "high",
  "confidence": 0.9,
  "source": "EDA"
}
```

Confidence is adjusted during validation:
- Valid: +0.2 (max 1.0)
- Invalid: -0.3
- Each issue: -0.1
- Each warning: -0.05

## Semantic Drift Prevention

To ensure generated documents preserve the original insight meaning, the validator provides:

```python
from pipeline.validator import semantic_similarity, validate_semantic_drift

original = "Technology has the highest profit."
generated = "# [FACT] Technology has highest profit\n\nTechnology leads with \$664K."

# Check similarity
sim = semantic_similarity(original, generated)
print(f"Similarity: {sim:.2f}")  # e.g., 0.824

# Validate against threshold
is_valid, sim = validate_semantic_drift(original, generated, threshold=0.65)
print(f"Acceptable: {is_valid}")  # True if similarity >= threshold
```

**Algorithm:**
- Weighted keyword/token overlap
- Synonym normalization (sales=revenue, increase=grow, highest=top)
- Number and entity matching (exact values preserved)
- Higher weight for numbers, entities, and domain keywords
- Lower weight for stop words

**Threshold:** Default 0.65 (adjustable based on tolerance for rewording)

Use this to screen generated documents for meaning preservation before embedding.
