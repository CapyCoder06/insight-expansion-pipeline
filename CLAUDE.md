# RAG Semi-Auto Document Pipeline (CLAUDE.md)

## 📌 Overview

This project implements a **semi-automated (human-in-the-loop) pipeline** for building a structured knowledge base for Retrieval-Augmented Generation (RAG).

The system combines:
- **Manual domain expertise** (hypothesis definition & validation intent)
- **Programmatic data analysis (EDA)**
- **Template-driven standardization**
- **LLM-based enrichment (controlled)**

The goal is to transform a small set of expert-defined hypotheses into a **scalable, high-quality document corpus** for semantic retrieval.

---

## 👥 Responsibility Split

- **Step 1–9 (Data → Chunking):** Implemented in this module
- **Step 10–12 (Embedding → Pinecone → RAG):** Implemented by teammates

---

## 🧠 Core Principle

> The system does not generate knowledge autonomously.
> 
> It **scales and structures human-defined insights** into machine-retrievable documents.

---

## 🧩 End-to-End Pipeline (Semi-Automated)

```
Dataset
  ↓
Hypothesis (Expert-defined, Manual)
  ↓
Validation (EDA, Programmatic)
  ↓
Insight (Structured)
  ↓
Template Mapping
  ↓
LLM Enrichment (Controlled Generation)
  ↓
Document Standardization
  ↓
Chunking
  ↓
(Embedding → Pinecone → RAG)
```

---

## 📊 Dataset

Source: https://www.kaggle.com/datasets/thuandao/superstore-sales-analytics

Used for:
- validating hypotheses
- grounding insights in real data
- avoiding hallucinated knowledge

---

## 📊 Hypothesis-Driven EDA Implementation

Hypothesis validation is implemented in a separate analysis module (Jupyter Notebook).

This module performs:
- data cleaning
- aggregation and statistical analysis
- hypothesis validation (supported / not supported)

The outputs of this step are structured insights, which are then fed into the document generation pipeline.

Reference:
- hypothesis_eda.ipynb

The analysis workflow is assisted by Claude Code for rapid iteration, while maintaining human control over hypothesis definition and interpretation.

---

Source: https://www.kaggle.com/datasets/thuandao/superstore-sales-analytics

Used for:
- validating hypotheses
- grounding insights in real data
- avoiding hallucinated knowledge

---

## 🧠 Step 1 — Hypothesis Definition (Manual)

Hypotheses are defined by domain understanding and exploratory analysis.

⚠️ This step is **not automated**.

### Example:
```
H1: Technology has the highest profit
H2: Higher discount leads to lower profit
H3: Some regions generate negative profit
```

---

## 🧪 Step 2 — Hypothesis Validation (EDA)

Each hypothesis is validated using data analysis:

- aggregation (groupby)
- correlation
- time-series analysis

### Output:
```json
{
  "hypothesis": "Technology has the highest profit",
  "result": "supported",
  "evidence": "Technology leads total profit across categories"
}
```

---

## 🧠 Step 3 — Insight Extraction

Validated hypotheses are converted into structured insights:

```json
{
  "text": "Technology has the highest profit",
  "dimensions": ["category"],
  "metrics": ["profit"],
  "type_hint": "fact"
}
```

---

## 🧱 Step 4 — Template System

Each insight is mapped to a document type:

- FACT
- TREND
- ANOMALY
- METRIC
- QUESTION

---

## 📄 Step 5 — Document Templates

### FACT
```markdown
# [FACT] {{title}}

{{description}}

- Dimensions: {{dimensions}}
- Metrics: {{metrics}}
- Grain: {{grain}}
```

### TREND
```markdown
# [TREND] {{title}}

{{description}}

- Dimension: {{dimension}}
- Metrics: {{metrics}}
- Grain: {{grain}}
- Trend: {{trend}}
```

### ANOMALY
```markdown
# [ANOMALY] {{title}}

{{description}}

- Dimension: {{dimension}}
- Metric: {{metric}}
- Issue: {{issue}}
- Possible Cause: {{possible_cause}}
```

### METRIC
```markdown
# [METRIC] {{metric}}

Definition:
{{definition}}

Formula:
{{formula}}

Interpretation:
{{interpretation}}
```

### QUESTION
```markdown
# [QUESTION]

Q: {{question}}

A: {{answer}}
```

---

## ⚙️ Step 6 — Enrichment (Hybrid Method - Recommended)

This system uses a **hybrid enrichment strategy** combining human control and LLM scalability.

### Process:

1. **Expert defines core knowledge (manual)**
2. **Templates define expansion dimensions (rule-based)**
3. **LLM generates multiple variants (controlled)**
4. **Expert reviews outputs (quality gate)**

---

### 🔥 Expansion–Filtering Strategy

- LLM expands: **10× documents**
- Human filters: **keep top ~3×**

Benefits:
- scalable
- controlled
- high-quality

---

## 🧪 Step 7 — Validation Rules

Each generated document must:

- remain faithful to the original insight
- avoid hallucination
- follow template format
- include key retrieval terms
- provide semantic richness

---

## 📦 Step 8 — Document Structure

Each document consists of:

### Text (for embedding)
### Metadata (for filtering)

```json
{
  "type": "fact",
  "dimensions": ["category"],
  "metrics": ["profit"]
}
```

---

## ✂️ Step 9 — Chunking (Implemented)

Documents are split into smaller chunks:

- 200–500 tokens
- semantic-aware splitting preferred

---

## 🧠 Step 10 — Embedding (Handled by teammate)

Convert text into vectors.

---

## 🌲 Step 11 — Pinecone Upsert (Handled by teammate)

Store embeddings with metadata.

---

## 🔍 Step 12 — RAG Retrieval (Handled by teammate)

Enable semantic search and QA.

---

## ⚠️ Common Pitfalls

| Issue | Impact |
|------|--------|
| Treating hypothesis as automated | incorrect system claim |
| Weak insights | poor retrieval |
| Hallucinated enrichment | wrong answers |
| No variation | low recall |

---

## 🚀 Final Output

From a small set of hypotheses → scalable document corpus:

- grounded in real data
- structured
- enriched
- chunked
- retrieval-ready

---

## 💡 Key Takeaway

> This is a **semi-automated knowledge generation system**, where human insight is amplified—not replaced—by AI.

---

## 🧠 One-line Summary

**Human Insight → Validated Knowledge → Structured Documents → RAG**

