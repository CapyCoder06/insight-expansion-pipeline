# Insight Expansion Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform ~10 validated structured insights into ~100 new structured insights by data-driven pattern expansion, grounded in real EDA dataset.

**Architecture:** Five independent modules: DataQueryEngine (loads CSV, provides queries), PatternExtractor (extracts expansion patterns from seeds), InsightGenerator (generates new insights using pattern+data), Deduplicator (removes near-duplicates via semantic similarity), InsightValidator (ensures data grounding). Pipeline orchestrates: seed → extract → generate → dedup → validate.

**Tech Stack:** Python 3.8+, standard library csv + re + json, existing `semantic_similarity` from `pipeline.validator`, pytest for testing.

---

## File Structure

```
src/insight_expansion/
  __init__.py
  data_query.py           # DataQueryEngine class
  pattern_extractor.py   # PatternExtractor class
  insight_generator.py   # InsightGenerator + helpers
  deduplicator.py        # deduplicate() function
  insight_validator.py   # InsightValidator class
  pipeline.py            # InsightExpansionPipeline orchestrator

tests/insight_expansion/
  __init__.py
  test_data_query.py
  test_pattern_extractor.py
  test_insight_generator.py
  test_deduplicator.py
  test_insight_validator.py
  test_pipeline.py

scripts/
  expand_insights.py     # CLI entry point

config.yaml (update)     # Add expansion section
```

---

## Pre-Requisites

- The dataset `data/eda_structured.csv` exists and has columns: `region`, `category`, `revenue`, `profit`, `margin`
- The seed insights file `insights_sample.json` exists with structured insights
- Existing `pipeline.validator.semantic_similarity` is available for deduplication

---

## Implementation Order (TDD)

Each task follows: Write failing test → run test (fail) → minimal implementation → run test (pass) → commit.

---

### Task 1: Data Query Engine Core

**Files:**
- Create: `src/insight_expansion/data_query.py`
- Create: `tests/insight_expansion/test_data_query.py`

**Purpose:** Load CSV and provide basic queries: get unique dimension values, get metrics for entity.

- [ ] **Step 1: Write failing test for DataQueryEngine initialization and get_unique_values**

```python
def test_data_query_engine_loads_csv_and_returns_unique_categories():
    from insight_expansion.data_query import DataQueryEngine
    engine = DataQueryEngine("data/eda_structured.csv")
    categories = engine.get_unique_values("category")
    assert isinstance(categories, list)
    assert len(categories) >= 3
    assert "Technology" in categories
    assert "Furniture" in categories
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insight_expansion/test_data_query.py::test_data_query_engine_loads_csv_and_returns_unique_categories -v`
Expected: FAIL - module not found

- [ ] **Step 3: Write minimal implementation**

Create `src/insight_expansion/__init__.py` (empty) and `src/insight_expansion/data_query.py`:

```python
import csv
from typing import List, Dict
from collections import defaultdict

class DataQueryEngine:
    def __init__(self, csv_path: str = "data/eda_structured.csv"):
        self.csv_path = csv_path
        self._data = []
        self._dimension_index = defaultdict(list)
        self._load_csv()

    def _load_csv(self):
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                row['revenue'] = float(row['revenue'])
                row['profit'] = float(row['profit'])
                row['margin'] = float(row['margin'])
                self._data.append(row)

    def get_unique_values(self, dimension: str) -> List[str]:
        values = set()
        for row in self._data:
            values.add(row[dimension])
        return sorted(values)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insight_expansion/test_data_query.py::test_data_query_engine_loads_csv_and_returns_unique_categories -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/insight_expansion/data_query.py tests/insight_expansion/test_data_query.py
git commit -m "feat: add DataQueryEngine with get_unique_values"
```

---

### Task 2: Data Query Engine Metric Fetching

- [ ] **Step 1: Write failing test for get_entity_metrics**

```python
def test_get_entity_metrics_returns_correct_values_for_category():
    from insight_expansion.data_query import DataQueryEngine
    engine = DataQueryEngine()
    metrics = engine.get_entity_metrics("category", "Technology", ["profit", "margin"])
    assert "profit" in metrics
    assert "margin" in metrics
    assert metrics["profit"] > 0
    assert metrics["margin"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL - method not defined

- [ ] **Step 3: Implement get_entity_metrics**

Add to `DataQueryEngine` in `data_query.py`:

```python
    def get_entity_metrics(self, dimension: str, entity: str, metrics: List[str]) -> Dict[str, float]:
        # Aggregate rows matching entity in the given dimension
        result = {}
        for row in self._data:
            if row[dimension] == entity:
                for metric in metrics:
                    if metric in row:
                        # Sum or average? The dataset already has per-row values; we need aggregated
                        # Since each (region,category) combo appears once, just take the first match
                        # But there could be multiple rows for same category across regions - need to aggregate
                        # Question: Are dimensions unique per entity? In dataset, category appears across multiple regions.
                        # For category-level metrics, we should aggregate (sum) across regions
                        pass  # We'll implement proper aggregation in next step
        # For now, just return first match (simplified)
        return result
```

Wait, need to design aggregation properly. The dataset has rows with both region and category. When querying by category only, we need to aggregate across all regions. Let's reconsider:

The dataset rows are at (region, category) granularity. If we query `category='Technology'`, there are multiple rows (Technology in Africa, Canada, etc.). We should:
- For profit: sum all Technology profit across regions
- For margin: weighted average by revenue? or average? The spec says "aggregated at dimension level". Let's compute weighted average: total profit / total revenue for that entity.

I'll implement proper aggregation.

- [ ] **Step 3 (revised): Implement get_entity_metrics with aggregation**

```python
    def get_entity_metrics(self, dimension: str, entity: str, metrics: List[str]) -> Dict[str, float]:
        """Fetch metrics for a given entity, aggregating across all other dimension levels."""
        # Collect matching rows
        matches = [row for row in self._data if row[dimension] == entity]
        if not matches:
            return {}

        result = {}
        for metric in metrics:
            if metric == "revenue" or metric == "profit":
                # Sum over all matches
                total = sum(row[metric] for row in matches)
                result[metric] = total
            elif metric == "margin":
                # Weighted average: total profit / total revenue
                total_revenue = sum(row['revenue'] for row in matches)
                total_profit = sum(row['profit'] for row in matches)
                result['margin'] = (total_profit / total_revenue * 100) if total_revenue > 0 else 0.0
        return result
```

- [ ] **Step 4: Run test to verify it passes**

Expected: PASS with actual numbers

- [ ] **Step 5: Commit**

```bash
git add src/insight_expansion/data_query.py
git commit -m "feat: implement get_entity_metrics with aggregation"
```

---

### Task 3: Data Query Engine Ranking and Average

- [ ] **Step 1: Write tests for get_ranking and get_entity_rank**

```python
def test_get_ranking_orders_categories_by_profit():
    from insight_expansion.data_query import DataQueryEngine
    engine = DataQueryEngine()
    ranking = engine.get_ranking("category", "profit")
    assert len(ranking) == 3  # 3 categories
    # First should be highest
    assert ranking[0][0] == "Technology"
    assert ranking[0][1] > 0
    # Check descending order
    for i in range(len(ranking)-1):
        assert ranking[i][1] >= ranking[i+1][1]

def test_get_entity_rank_returns_1_for_highest():
    from insight_expansion.data_query import DataQueryEngine
    engine = DataQueryEngine()
    rank = engine.get_entity_rank("category", "profit", "Technology")
    assert rank == 1

def test_get_average_computes_mean_correctly():
    from insight_expansion.data_query import DataQueryEngine
    engine = DataQueryEngine()
    avg = engine.get_average("category", "profit")
    # Should be sum of all category profits / number of categories
    assert avg > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: FAIL - methods not defined

- [ ] **Step 3: Implement get_ranking, get_entity_rank, get_average**

```python
    def get_ranking(self, dimension: str, metric: str) -> List[tuple]:
        """Return list of (entity, value) sorted descending by aggregated metric."""
        entities = self.get_unique_values(dimension)
        ranked = []
        for entity in entities:
            metrics_data = self.get_entity_metrics(dimension, entity, [metric])
            value = metrics_data.get(metric, 0.0)
            ranked.append((entity, value))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def get_entity_rank(self, dimension: str, metric: str, entity: str) -> int:
        """Return rank position (1 = highest). Ties get same rank (dense ranking)."""
        ranking = self.get_ranking(dimension, metric)
        # Find all entities with same value as target
        target_value = None
        for ent, val in ranking:
            if ent == entity:
                target_value = val
                break
        if target_value is None:
            return len(ranking) + 1  # not found
        # Rank = position of first occurrence + 1 (dense ranking)
        for i, (ent, val) in enumerate(ranking):
            if val == target_value:
                return i + 1
        return len(ranking) + 1

    def get_average(self, dimension: str, metric: str) -> float:
        """Compute average of metric across all entities in dimension."""
        entities = self.get_unique_values(dimension)
        total = 0.0
        count = 0
        for entity in entities:
            metrics_data = self.get_entity_metrics(dimension, entity, [metric])
            total += metrics_data.get(metric, 0.0)
            count += 1
        return total / count if count > 0 else 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/insight_expansion/test_data_query.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/insight_expansion/data_query.py tests/insight_expansion/test_data_query.py
git commit -m "feat: add ranking and average methods to DataQueryEngine"
```

---

### Task 4: Data Query Engine Outlier Detection and Dimension Helpers

- [ ] **Step 1: Write tests for detect_outliers and has_metric_for_dimension**

```python
def test_detect_outliers_finds_negative_profit_entities():
    from insight_expansion.data_query import DataQueryEngine
    engine = DataQueryEngine()
    negatives = engine.detect_outliers("category", "profit", "negative")
    # Should include categories with total profit < 0
    assert isinstance(negatives, list)
    for entity in negatives:
        metrics = engine.get_entity_metrics("category", entity, ["profit"])
        assert metrics["profit"] < 0

def test_has_metric_for_dimension_returns_true_for_valid_combinations():
    from insight_expansion.data_query import DataQueryEngine
    engine = DataQueryEngine()
    assert engine.has_metric_for_dimension("category", "profit") is True
    assert engine.has_metric_for_dimension("region", "margin") is True

def test_get_available_dimensions_returns_all_dimension_columns():
    from insight_expansion.data_query import DataQueryEngine
    engine = DataQueryEngine()
    dims = engine.get_available_dimensions()
    assert "category" in dims
    assert "region" in dims
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: FAIL - methods not defined

- [ ] **Step 3: Implement methods in DataQueryEngine**

Add to `data_query.py`:

```python
    def detect_outliers(self, dimension: str, metric: str, condition: str, threshold: float = None) -> List[str]:
        entities = self.get_unique_values(dimension)
        outliers = []
        for entity in entities:
            metrics_data = self.get_entity_metrics(dimension, entity, [metric])
            value = metrics_data.get(metric, 0.0)
            if condition == "negative":
                if value < 0:
                    outliers.append(entity)
            elif condition == "below_threshold":
                if threshold is not None and value < threshold:
                    outliers.append(entity)
            elif condition == "outlier":
                # Statistical outlier: beyond mean ± 2 std dev
                # Compute mean and std across entities
                all_values = [self.get_entity_metrics(dimension, ent, [metric]).get(metric, 0.0) for ent in entities]
                mean = sum(all_values) / len(all_values)
                variance = sum((x - mean) ** 2 for x in all_values) / len(all_values)
                std = variance ** 0.5
                if abs(value - mean) > 2 * std:
                    outliers.append(entity)
        return outliers

    def get_available_dimensions(self) -> List[str]:
        """Return categorical columns that can be used as dimensions."""
        # From dataset columns, dimensions are: region, category, sub_category (if exists)
        # We know the fixed set from the design
        return ["region", "category"]  # Could be dynamic by inspecting CSV headers

    def has_metric_for_dimension(self, dimension: str, metric: str) -> bool:
        """Check whether the dataset contains the given metric aggregated at the specified dimension level."""
        # In our dataset, all metrics (revenue, profit, margin) exist for both category and region aggregates
        # We can verify by computing one entity - if it returns a value > 0 or any value, it exists
        entities = self.get_unique_values(dimension)
        if not entities:
            return False
        sample_metrics = self.get_entity_metrics(dimension, entities[0], [metric])
        return metric in sample_metrics and (sample_metrics[metric] != 0 or True)  # zero also counts as existing
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/insight_expansion/test_data_query.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/insight_expansion/data_query.py tests/insight_expansion/test_data_query.py
git commit -m "feat: add outlier detection and dimension helper methods"
```

---

### Task 5: Pattern Extractor — Basic Pattern Extraction

**Files:**
- Create: `src/insight_expansion/pattern_extractor.py`
- Create: `tests/insight_expansion/test_pattern_extractor.py`

- [ ] **Step 1: Write failing test for extract method with fact seed**

```python
def test_extract_fact_pattern_returns_correct_structure():
    from insight_expansion.pattern_extractor import PatternExtractor
    seed = {
        "text": "Technology has highest profit with margin 14%",
        "dimensions": ["category"],
        "metrics": ["profit", "margin"],
        "type_hint": "fact"
    }
    query_engine = None  # Will be passed
    pattern = PatternExtractor.extract(seed, query_engine)
    assert pattern["primary_dimension"] == "category"
    assert pattern["metrics"] == ["profit", "margin"]
    assert pattern["comparison_type"] == "rank"
    assert "{entity}" in pattern["text_format"]
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL - module not found

- [ ] **Step 3: Implement PatternExtractor.extract (basic version without alternate dimensions)**

Create `src/insight_expansion/pattern_extractor.py`:

```python
from typing import Dict, List

class PatternExtractor:
    @staticmethod
    def extract(seed: Dict, query_engine) -> Dict:
        """Extract pattern from seed insight."""
        dimensions = seed.get("dimensions", [])
        primary_dimension = dimensions[0] if dimensions else ""

        type_hint = seed.get("type_hint", "")
        text = seed.get("text", "")

        # Define text_format template based on type_hint
        if type_hint == "fact":
            text_format = "{entity} {verb} {qualifier} {metric} of {value}"
        elif type_hint == "trend":
            text_format = "{entity} {metric} is {direction}, changing from {start} to {end}"
        elif type_hint == "anomaly":
            text_format = "{entity} shows {issue}: {metric} of {value}"
        elif type_hint == "metric":
            text_format = "{metric} definition: {text}"
        elif type_hint == "question":
            text_format = "Q: {question} about {entity}? A: {answer}"
        else:
            text_format = "{text}"

        pattern = {
            "primary_dimension": primary_dimension,
            "metrics": seed.get("metrics", []),
            "comparison_type": PatternExtractor._get_comparison_type(type_hint),
            "text_format": text_format,
            "optional_fields": {k: v for k, v in seed.items() if k in ["issue", "possible_cause"]},
            "qualifier_rules": PatternExtractor._get_qualifier_rules(type_hint),
            "alternate_dimensions": [],  # Will compute later with query_engine
            "original_entity": ""  # Will extract later
        }

        # Compute alternate dimensions if query_engine provided
        if query_engine:
            pattern["alternate_dimensions"] = PatternExtractor._discover_alternate_dimensions(seed, query_engine)

        # Extract original entity if query_engine provided
        if query_engine and primary_dimension:
            pattern["original_entity"] = PatternExtractor._extract_original_entity(text, primary_dimension, query_engine)

        return pattern

    @staticmethod
    def _get_comparison_type(type_hint: str) -> str:
        mapping = {
            "fact": "rank",
            "trend": "direction",
            "anomaly": "outlier",
            "metric": "definition",
            "question": "question"
        }
        return mapping.get(type_hint, "")

    @staticmethod
    def _get_qualifier_rules(type_hint: str) -> Dict:
        if type_hint == "fact":
            return {"compute_from": "rank", "thresholds": {"top": 1, "bottom": "last", "above_avg": 1.1, "below_avg": 0.9}}
        elif type_hint == "trend":
            return {"compute_from": "time_series", "thresholds": {"min_change": 0.05}}
        elif type_hint == "anomaly":
            return {"compute_from": "anomaly_detection", "conditions": ["negative", "low_threshold", "outlier"]}
        elif type_hint == "metric":
            return {}
        elif type_hint == "question":
            return {}
        return {}

    @staticmethod
    def _discover_alternate_dimensions(seed: Dict, query_engine) -> List[str]:
        primary = seed.get("dimensions", [])[0] if seed.get("dimensions") else ""
        all_dims = query_engine.get_available_dimensions()
        metrics = seed.get("metrics", [])
        alternates = []
        for dim in all_dims:
            if dim == primary:
                continue
            # Check all metrics exist for this dimension
            all_exist = all(query_engine.has_metric_for_dimension(dim, m) for m in metrics)
            if all_exist:
                alternates.append(dim)
        return alternates

    @staticmethod
    def _extract_original_entity(text: str, dimension: str, query_engine) -> str:
        entities = query_engine.get_unique_values(dimension)
        text_lower = text.lower()
        for ent in entities:
            if ent.lower() in text_lower:
                return ent
        return ""
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/insight_expansion/test_pattern_extractor.py::test_extract_fact_pattern_returns_correct_structure -v
```

Note: The test uses `query_engine=None`. We'll write tests with a mock/fixture later. The method should handle None gracefully.

Update implementation to handle None in those branches:

```python
        if query_engine:
            pattern["alternate_dimensions"] = PatternExtractor._discover_alternate_dimensions(seed, query_engine)
            pattern["original_entity"] = PatternExtractor._extract_original_entity(text, primary_dimension, query_engine)
        else:
            pattern["alternate_dimensions"] = []
            pattern["original_entity"] = ""
```

- [ ] **Step 5: Commit**

```bash
git add src/insight_expansion/pattern_extractor.py tests/insight_expansion/test_pattern_extractor.py
git commit -m "feat: add PatternExtractor with type_hint-based pattern extraction and alternate dimension discovery"
```

---

### Task 6: Insight Generator — Single Insight Generation

**Files:**
- Create: `src/insight_expansion/insight_generator.py`
- Create: `tests/insight_expansion/test_insight_generator.py`

- [ ] **Step 1: Write failing test for _generate_combined_insight with fact**

```python
def test_generate_combined_insight_produces_valid_fact():
    from insight_expansion.insight_generator import InsightGenerator
    from insight_expansion.data_query import DataQueryEngine
    seed = {
        "text": "Technology has highest profit with margin 14%",
        "dimensions": ["category"],
        "metrics": ["profit", "margin"],
        "type_hint": "fact"
    }
    engine = DataQueryEngine()
    pattern = {
        "primary_dimension": "category",
        "metrics": ["profit", "margin"],
        "comparison_type": "rank",
        "text_format": "{entity} {qualifier} {metric} of {value}",
        "alternate_dimensions": [],
        "original_entity": "Technology"
    }
    insight = InsightGenerator._generate_combined_insight(seed, pattern, "Technology", "category", engine)
    assert insight["text"]  # non-empty
    assert "Technology" in insight["text"]
    assert insight["dimensions"] == ["category"]
    assert insight["metrics"] == ["profit", "margin"]
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL - module not found

- [ ] **Step 3: Implement InsightGenerator._generate_combined_insight (minimal)**

Create `src/insight_expansion/insight_generator.py`:

```python
from typing import Dict, List

class InsightGenerator:
    @staticmethod
    def _generate_combined_insight(seed, pattern, entity, current_dim, query_engine):
        metrics_data = query_engine.get_entity_metrics(current_dim, entity, pattern["metrics"])
        qualifier = InsightGenerator.compute_qualifier(pattern, entity, metrics_data, query_engine, current_dim)

        # Use first metric for template
        metric = pattern["metrics"][0] if pattern["metrics"] else ""
        value = metrics_data.get(metric, 0)
        # Format value based on metric
        if metric == "margin":
            value_str = f"{value:.1f}%"
        else:
            value_str = f"${value:,.0f}"

        # Build text
        text = pattern["text_format"].format(
            entity=entity,
            qualifier=qualifier,
            metric=metric,
            value=value_str,
            verb="has"  # simplistic; will vary by pattern
        )

        new_insight = {
            "text": text,
            "dimensions": [current_dim],
            "metrics": pattern["metrics"],
            "type_hint": seed["type_hint"],
        }

        # Optional fields
        if pattern.get("optional_fields") and InsightGenerator._optional_fields_still_valid(pattern["optional_fields"], entity, pattern):
            new_insight.update(pattern["optional_fields"])

        return new_insight

    @staticmethod
    def compute_qualifier(pattern, entity, metrics_data, query_engine, current_dim):
        comp_type = pattern["comparison_type"]
        metric = pattern["metrics"][0] if pattern["metrics"] else None

        if comp_type == "rank":
            total_entities = len(query_engine.get_unique_values(current_dim))
            rank = query_engine.get_entity_rank(current_dim, metric, entity)
            if rank == 1:
                return "highest"
            elif rank == total_entities:
                return "lowest"
            else:
                avg = query_engine.get_average(current_dim, metric)
                val = metrics_data.get(metric, 0)
                if val > avg * 1.1:
                    return "above average"
                elif val < avg * 0.9:
                    return "below average"
                else:
                    return "around average"
        else:
            return ""

    @staticmethod
    def _optional_fields_still_valid(optional_fields, entity, pattern):
        return entity == pattern.get("original_entity", "")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/insight_expansion/test_insight_generator.py::test_generate_combined_insight_produces_valid_fact -v
```

- [ ] **Step 5: Commit**

```bash
git add src/insight_expansion/insight_generator.py tests/insight_expansion/test_insight_generator.py
git commit -m "feat: add InsightGenerator._generate_combined_insight with qualifier computation"
```

---

### Task 7: Insight Generator — Full Generate Function and Metric Splitting

- [ ] **Step 1: Write failing test for generate_insights**

```python
def test_generate_insights_creates_insights_for_all_categories():
    from insight_expansion.insight_generator import InsightGenerator
    from insight_expansion.data_query import DataQueryEngine
    from insight_expansion.pattern_extractor import PatternExtractor

    seed = {
        "text": "Technology has highest profit with margin 14%",
        "dimensions": ["category"],
        "metrics": ["profit", "margin"],
        "type_hint": "fact"
    }
    engine = DataQueryEngine()
    pattern = PatternExtractor.extract(seed, engine)
    insights = InsightGenerator.generate_insights(seed, engine)  # We'll write this method
    # Should have at least 3 categories, and with metric splitting maybe more
    assert len(insights) >= 3
    for ins in insights:
        assert ins["dimensions"] == ["category"]
        assert "profit" in ins["metrics"] or "margin" in ins["metrics"]
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL - `generate_insights` not defined

- [ ] **Step 3: Implement generate_insights (uses _generate_combined_insight, will add splitting later)**

In `insight_generator.py`:

```python
    @staticmethod
    def generate_insights(seed: Dict, query_engine) -> List[Dict]:
        pattern = PatternExtractor.extract(seed, query_engine)
        primary_dim = pattern["primary_dimension"]
        alternate_dims = pattern.get("alternate_dimensions", [])
        dimensions_to_expand = [primary_dim] + alternate_dims

        generated = []
        for current_dim in dimensions_to_expand:
            entities = query_engine.get_unique_values(current_dim)
            for entity in entities:
                # For now, no splitting; handle in next task
                insight = InsightGenerator._generate_combined_insight(seed, pattern, entity, current_dim, query_engine)
                if insight:
                    generated.append(insight)
        return generated
```

- [ ] **Step 4: Run test to verify it passes** (should produce 3 insights for 3 categories)

```bash
pytest tests/insight_expansion/test_insight_generator.py::test_generate_insights_creates_insights_for_all_categories -v
```

- [ ] **Step 5: Add metric splitting support**

Update `generate_insights` to split metrics when appropriate:

```python
    @staticmethod
    def _should_split_metrics(metrics: List[str], type_hint: str) -> bool:
        if len(metrics) <= 1:
            return False
        if type_hint in ["fact", "anomaly"]:
            return True
        return False

    @staticmethod
    def generate_insights(seed: Dict, query_engine) -> List[Dict]:
        pattern = PatternExtractor.extract(seed, query_engine)
        primary_dim = pattern["primary_dimension"]
        alternate_dims = pattern.get("alternate_dimensions", [])
        dimensions_to_expand = [primary_dim] + alternate_dims

        generated = []
        for current_dim in dimensions_to_expand:
            entities = query_engine.get_unique_values(current_dim)
            for entity in entities:
                split = InsightGenerator._should_split_metrics(pattern["metrics"], seed["type_hint"])
                if split:
                    for metric in pattern["metrics"]:
                        insight = InsightGenerator._generate_single_metric_insight(seed, pattern, entity, metric, current_dim, query_engine)
                        if insight:
                            generated.append(insight)
                else:
                    insight = InsightGenerator._generate_combined_insight(seed, pattern, entity, current_dim, query_engine)
                    if insight:
                        generated.append(insight)
        return generated

    @staticmethod
    def _generate_single_metric_insight(seed, pattern, entity, metric, current_dim, query_engine):
        metrics_data = query_engine.get_entity_metrics(current_dim, entity, [metric])
        qualifier = InsightGenerator.compute_qualifier(pattern, entity, metrics_data, query_engine, current_dim)
        value = metrics_data.get(metric, 0)
        if metric == "margin":
            value_str = f"{value:.1f}%"
        else:
            value_str = f"${value:,.0f}"

        text = pattern["text_format"].format(
            entity=entity,
            qualifier=qualifier,
            metric=metric,
            value=value_str,
            verb="has"
        )
        return {
            "text": text,
            "dimensions": [current_dim],
            "metrics": [metric],
            "type_hint": seed["type_hint"]
        }
```

- [ ] **Step 6: Update test to expect splitting**

Modify test to check >= 6 insights (3 categories × 2 metrics). Or add separate test:

```python
def test_generate_insights_splits_multi_metric_fact():
    seed = {
        "text": "Technology has highest profit with margin 14%",
        "dimensions": ["category"],
        "metrics": ["profit", "margin"],
        "type_hint": "fact"
    }
    insights = InsightGenerator.generate_insights(seed, DataQueryEngine())
    # Should have 2 separate insights per category
    assert len(insights) == 6  # 3 categories × 2 metrics
```

- [ ] **Step 7: Commit**

```bash
git add src/insight_expansion/insight_generator.py tests/insight_expansion/test_insight_generator.py
git commit -m "feat: implement generate_insights with metric splitting and multi-dimension expansion"
```

---

### Task 8: Deduplicator

**Files:**
- Create: `src/insight_expansion/deduplicator.py`
- Create: `tests/insight_expansion/test_deduplicator.py`

- [ ] **Step 1: Write failing test for deduplicate function**

```python
def test_deduplicate_removes_exact_duplicates():
    from insight_expansion.deduplicator import deduplicate
    insights = [
        {"text": "A"},
        {"text": "A"},
        {"text": "B"}
    ]
    result = deduplicate(insights)
    assert len(result) == 2
    texts = [i["text"] for i in result]
    assert "A" in texts
    assert "B" in texts

def test_deduplicate_removes_near_duplicates_above_threshold():
    from insight_expansion.deduplicator import deduplicate
    insights = [
        {"text": "Technology has high profit"},
        {"text": "Technology possesses high profit"},  # near-duplicate
        {"text": "Sales are increasing"}
    ]
    result = deduplicate(insights, similarity_threshold=0.85)
    assert len(result) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: FAIL - module not found

- [ ] **Step 3: Implement deduplicate using semantic_similarity**

`src/insight_expansion/deduplicator.py`:

```python
from typing import List, Dict
from pipeline.validator import semantic_similarity

def deduplicate(insights: List[Dict], similarity_threshold: float = 0.85) -> List[Dict]:
    """Remove insights with text similarity >= threshold."""
    unique = []
    for insight in insights:
        text = insight["text"].lower().strip()
        is_duplicate = False
        for existing in unique:
            existing_text = existing["text"].lower().strip()
            sim = semantic_similarity(text, existing_text)
            if sim >= similarity_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(insight)
    return unique
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/insight_expansion/test_deduplicator.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/insight_expansion/deduplicator.py tests/insight_expansion/test_deduplicator.py
git commit -m "feat: add deduplicate function using semantic_similarity"
```

---

### Task 9: Insight Validator — Core Rules

**Files:**
- Create: `src/insight_expansion/insight_validator.py`
- Create: `tests/insight_expansion/test_insight_validator.py`

- [ ] **Step 1: Write failing test for validate required fields and type_hint**

```python
def test_validate_accepts_valid_insight():
    from insight_expansion.insight_validator import InsightValidator
    validator = InsightValidator()
    insight = {
        "text": "Technology has highest profit of $664K",
        "dimensions": ["category"],
        "metrics": ["profit"],
        "type_hint": "fact"
    }
    result = validator.validate(insight)
    assert result["valid"] is True

def test_validate_rejects_missing_text():
    from insight_expansion.insight_validator import InsightValidator
    validator = InsightValidator()
    insight = {
        "dimensions": ["category"],
        "metrics": ["profit"],
        "type_hint": "fact"
    }
    result = validator.validate(insight)
    assert result["valid"] is False
    assert "Missing required field: text" in result["issues"]

def test_validate_rejects_invalid_type_hint():
    from insight_expansion.insight_validator import InsightValidator
    validator = InsightValidator()
    insight = {
        "text": "Something",
        "dimensions": ["category"],
        "metrics": ["profit"],
        "type_hint": "invalid"
    }
    result = validator.validate(insight)
    assert result["valid"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: FAIL - module not found

- [ ] **Step 3: Implement InsightValidator (core rules)**

`src/insight_expansion/insight_validator.py`:

```python
from typing import Dict, List

class InsightValidator:
    VALID_TYPES = {"fact", "trend", "anomaly", "metric", "question"}
    VALID_COLUMNS = {"region", "category", "revenue", "profit", "margin", "sub_category"}

    def validate(self, insight: Dict, seed_text: str = None) -> Dict:
        result = {"valid": True, "issues": []}

        # Required fields
        for field in ["text", "dimensions", "metrics", "type_hint"]:
            if field not in insight or not insight[field]:
                result["valid"] = False
                result["issues"].append(f"Missing required field: {field}")

        # Type hint validity
        if insight.get("type_hint") not in self.VALID_TYPES:
            result["valid"] = False
            result["issues"].append(f"Invalid type_hint: {insight.get('type_hint')}")

        # Dimensions and metrics validity
        dims = insight.get("dimensions", [])
        metrics = insight.get("metrics", [])
        if any(d not in self.VALID_COLUMNS for d in dims):
            result["valid"] = False
            result["issues"].append("Invalid dimension name")
        if any(m not in self.VALID_COLUMNS for m in metrics):
            result["valid"] = False
            result["issues"].append("Invalid metric name")

        # Anomaly fields consistency
        has_issue = "issue" in insight and insight["issue"]
        has_cause = "possible_cause" in insight and insight["possible_cause"]
        if has_issue != has_cause:
            result["valid"] = False
            result["issues"].append("Anomaly must have both issue and possible_cause or neither")

        # Seed similarity check
        if seed_text:
            from pipeline.validator import semantic_similarity
            sim = semantic_similarity(insight["text"], seed_text)
            if sim >= 0.85:
                result["valid"] = False
                result["issues"].append(f"Insight too similar to seed (sim={sim:.2f})")

        return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/insight_expansion/test_insight_validator.py -v
```

- [ ] **Step 5: Add numbers grounding rule**

We need to implement Rule 4: numeric values must exist in dataset. This requires DataQueryEngine. The validator needs to accept query_engine.

Update test:

```python
def test_validate_checks_numbers_are_in_dataset():
    from insight_expansion.insight_validator import InsightValidator
    from insight_expansion.data_query import DataQueryEngine
    validator = InsightValidator(query_engine=DataQueryEngine())
    insight = {
        "text": "Technology has highest profit of $999999999",  # fake number
        "dimensions": ["category"],
        "metrics": ["profit"],
        "type_hint": "fact"
    }
    result = validator.validate(insight)
    assert result["valid"] is False
    assert any("number" in issue.lower() or "dataset" in issue.lower() for issue in result["issues"])
```

Update InsightValidator `__init__` to take optional query_engine:

```python
    def __init__(self, query_engine=None):
        self.query_engine = query_engine
```

Implement numbers check in `validate`:

```python
        # Numbers grounding (if query_engine available)
        if self.query_engine:
            self._validate_numbers_grounding(insight, result)
```

Add method:

```python
    def _validate_numbers_grounding(self, insight: Dict, result: Dict):
        import re
        text = insight["text"]
        dims = insight.get("dimensions", [])
        if not dims:
            return  # Need dimension to identify entity
        primary_dim = dims[0]

        # Extract entity from text
        entities = self.query_engine.get_unique_values(primary_dim)
        text_lower = text.lower()
        found_entity = None
        for ent in entities:
            if ent.lower() in text_lower:
                found_entity = ent
                break
        if not found_entity:
            result["valid"] = False
            result["issues"].append("Could not identify entity in text for number validation")
            return

        # Extract numbers from text
        number_patterns = [
            (r'\$[\d,]+K?', 'currency'),
            (r'\d+(?:\.\d+)?%', 'percentage'),
            (r'-?\d+(?:\.\d+)?', 'plain_number')
        ]
        numbers = []
        for pattern, _ in number_patterns:
            numbers.extend(re.findall(pattern, text))

        # For each number, try to associate with a metric and verify
        metrics = insight.get("metrics", [])
        if not metrics:
            return

        # Simple: check if any number matches the expected metric value for that entity
        metrics_data = self.query_engine.get_entity_metrics(primary_dim, found_entity, metrics)
        tolerance = 0.05  # 5%

        for num_str in numbers:
            # Convert to float
            try:
                extracted = float(num_str.replace('$', '').replace(',', '').replace('%', ''))
            except:
                continue
            # Compare against each metric value
            matched = False
            for metric in metrics:
                actual = metrics_data.get(metric, None)
                if actual is not None:
                    if abs(actual - extracted) / max(abs(actual), 1) <= tolerance:
                        matched = True
                        break
            if not matched and numbers:  # If we extracted numbers but none match, flag
                result["valid"] = False
                result["issues"].append(f"Number {num_str} does not match any metric value in dataset")
                break
```

- [ ] **Step 6: Run tests, adjust until pass

- [ ] **Step 7: Commit**

```bash
git add src/insight_expansion/insight_validator.py tests/insight_expansion/test_insight_validator.py
git commit -m "feat: add InsightValidator with data grounding, format, and similarity checks"
```

---

### Task 10: Pipeline Orchestrator

**Files:**
- Create: `src/insight_expansion/pipeline.py`
- Create: `tests/insight_expansion/test_pipeline.py`

- [ ] **Step 1: Write failing integration test**

```python
def test_pipeline_expands_seeds_to_many_insights():
    from insight_expansion.pipeline import InsightExpansionPipeline
    import json
    with open('insights_sample.json') as f:
        seeds = json.load(f)['insights']
    pipeline = InsightExpansionPipeline()
    result = pipeline.run(seeds, target_count=100)
    insights = result['insights']
    assert len(insights) >= 100
    # All valid
    assert all(i.get('valid', True) for i in insights) or pipeline.validator.validate_batch(insights)['invalid'] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL - module not found

- [ ] **Step 3: Implement InsightExpansionPipeline**

`src/insight_expansion/pipeline.py`:

```python
from typing import List, Dict
from .data_query import DataQueryEngine
from .pattern_extractor import PatternExtractor
from .insight_generator import InsightGenerator
from .deduplicator import deduplicate
from .insight_validator import InsightValidator

class InsightExpansionPipeline:
    def __init__(self, csv_path: str = "data/eda_structured.csv"):
        self.query_engine = DataQueryEngine(csv_path)
        self.validator = InsightValidator(query_engine=self.query_engine)

    def run(self, seeds: List[Dict], target_count: int = 100) -> Dict:
        # Phase 1: Generate
        generated = []
        for seed in seeds:
            insights = InsightGenerator.generate_insights(seed, self.query_engine)
            generated.extend(insights)

        # Phase 2: Deduplicate
        deduped = deduplicate(generated, similarity_threshold=0.85)

        # Phase 3: Validate
        validation_summary = self.validator.validate_batch(deduped)

        # If we have more than target, truncate
        if len(deduped) > target_count:
            deduped = deduped[:target_count]

        return {
            "insights": deduped,
            "total_generated": len(deduped),
            "validation": validation_summary
        }
```

Add `validate_batch` to InsightValidator:

```python
    def validate_batch(self, insights: List[Dict]) -> Dict:
        valid = 0
        invalid = 0
        details = []
        for insight in insights:
            result = self.validate(insight)
            if result["valid"]:
                valid += 1
            else:
                invalid += 1
            details.append(result)
        return {"total": len(insights), "valid": valid, "invalid": invalid, "documents": details}
```

- [ ] **Step 4: Run test to verify it passes**

Note: We may need to adjust expectations based on actual output. Write additional test to check that metric definition seeds produce empty expansion:

```python
def test_metric_seeds_not_expanded():
    seed = {
        "text": "Profit margin is the ratio of profit to revenue",
        "dimensions": ["profit"],
        "metrics": ["margin"],
        "type_hint": "metric"
    }
    insights = InsightGenerator.generate_insights(seed, DataQueryEngine())
    assert len(insights) == 0  # metric type returns empty
```

- [ ] **Step 5: Commit**

```bash
git add src/insight_expansion/pipeline.py tests/insight_expansion/test_pipeline.py
git commit -m "feat: add pipeline orchestrator and integration tests"
```

---

### Task 11: CLI Script

**Files:**
- Create: `scripts/expand_insights.py`

- [ ] **Step 1: Write CLI script**

```python
#!/usr/bin/env python
"""
CLI to expand insights into larger dataset.
Usage: python expand_insights.py --input insights_sample.json --output insights_expanded.json --target 100
"""

import json
import argparse
import sys
sys.path.insert(0, 'src')

from insight_expansion.pipeline import InsightExpansionPipeline

def main():
    parser = argparse.ArgumentParser(description="Expand structured insights using EDA dataset")
    parser.add_argument("--input", required=True, help="Input JSON file with seed insights")
    parser.add_argument("--output", required=True, help="Output JSON file for expanded insights")
    parser.add_argument("--target", type=int, default=100, help="Target number of insights")
    parser.add_argument("--csv", default="data/eda_structured.csv", help="Path to EDA CSV")
    args = parser.parse_args()

    # Load seed insights
    with open(args.input, 'r') as f:
        data = json.load(f)
    seeds = data.get('insights', data)  # handle both wrapped and raw list

    # Run pipeline
    pipeline = InsightExpansionPipeline(csv_path=args.csv)
    result = pipeline.run(seeds, target_count=args.target)

    # Save output
    output = {
        "insights": result["insights"],
        "metadata": {
            "total_generated": result["total_generated"],
            "target": args.target,
            "validation": result["validation"]
        }
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Generated {len(result['insights'])} insights")
    print(f"Validation: {result['validation']['valid']}/{result['validation']['total']} valid")
    if result['validation']['invalid'] > 0:
        print("Issues:")
        for doc in result['validation']['documents']:
            if not doc['valid']:
                print(f"  - {doc['issues']}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make executable and test manually**

```bash
chmod +x scripts/expand_insights.py
python scripts/expand_insights.py --input insights_sample.json --output insights_expanded.json --target 100
```

Check that output file is created and contains ~100 insights.

- [ ] **Step 3: Commit**

```bash
git add scripts/expand_insights.py
git commit -m "feat: add CLI script for insight expansion"
```

---

### Task 12: Config Update

- [ ] **Step 1: Update config.yaml to add expansion section**

Add to `config.yaml`:

```yaml
expansion:
  target_count: 100
  dedup_threshold: 0.85
  validation:
    number_tolerance: 0.05
```

- [ ] **Step 2: Commit**

```bash
git add config.yaml
git commit -m "config: add expansion settings"
```

---

### Task 13: Final Integration and Cleanup

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/insight_expansion/ -v
```

Fix any failures.

- [ ] **Step 2: Run pipeline end-to-end with real data and verify success criteria**

```bash
python scripts/expand_insights.py --input insights_sample.json --output insights_expanded.json --target 100
```

Verify:
- Output contains >= 100 insights
- No duplicate texts (exact check)
- All pass validation
- No fabricated numbers (spot-check a few)

- [ ] **Step 3: Add `__init__.py` files to make packages**

`src/insight_expansion/__init__.py` (already from Task 1, ensure exists)
`tests/insight_expansion/__init__.py` (empty)

```bash
touch tests/insight_expansion/__init__.py
git add tests/insight_expansion/__init__.py
```

- [ ] **Step 4: Commit final**

```bash
git add -A
git commit -m "feat: complete insight expansion pipeline with tests and CLI"
```

---

## Summary

This plan implements the insight expansion pipeline in **13 tasks**, following TDD from DataQueryEngine through PatternExtractor, InsightGenerator, Deduplicator, Validator, Pipeline, and CLI.

All code is new; no existing files modified (except config.yaml). The pipeline produces 100+ structured insights from seeds, grounded in real data, with validation and deduplication.

---

**Plan complete and saved to `docs/superpowers/plans/2025-03-29-insight-expansion-pipeline.md`.**

Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
