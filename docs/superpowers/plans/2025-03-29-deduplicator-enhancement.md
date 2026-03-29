# Deduplicator Enhancement: Entity-Metric-Value Signature

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enhance the Deduplicator to deduplicate insights based on entity + metric + value signature in addition to text similarity, catching near-duplicates with different phrasing but same factual content.

**Architecture:** Extend Deduplicator to extract canonical signatures (entity, metric, numeric value) from each insight text, then use both text similarity and signature matching to identify duplicates. Build on existing similarity graph clustering with additional edges for signature matches.

**Tech Stack:** Python, regex, existing Deduplicator class

---

## File Structure Analysis

**Files to modify:**
- `src/insight_expansion/deduplicator.py` - Core deduplication logic; will add signature extraction and matching

**Test files:**
- `tests/insight_expansion/test_deduplicator.py` - Existing tests; need to add new test cases for signature-based deduplication

**Supporting files (reference only):**
- `src/insight_expansion/insight_generator.py` - To understand insight structure
- `expand_insights.py` - To understand data flow

---

## Task 1: Implement Signature Extraction Function

**Files:**
- Modify: `src/insight_expansion/deduplicator.py`
- Test: `tests/insight_expansion/test_deduplicator.py`

**Objective:** Create a function to extract `(entity, metric, value)` signature from insight text.

The signature extraction needs to handle patterns like:
- "Africa has below average margin of 11.3%" → (Africa, margin, 11.3)
- "Africa has moderate margin: 11.3%" → (Africa, margin, 11.3)
- "Africa margin is 11.3%" → (Africa, margin, 11.3)
- "Africa profit is $88,872" → (Africa, profit, 88872)
- "Technology: profit $663,779, margin 14.0%" → Two signatures: (Technology, profit, 663779), (Technology, margin, 14.0)

Assumptions:
- Entity is the first word/phrase before a verb like "has", "margin is", etc.
- For multi-metric insights (comma-separated), generate multiple signatures, one per metric-value pair
- Values are numeric; strip currency symbols ($) and commas; percentages as decimals

### Step 1: Write the failing test

Add to `tests/insight_expansion/test_deduplicator.py`:

```python
def test_extract_signature():
    """Test signature extraction from various text patterns."""
    from insight_expansion.deduplicator import Deduplicator

    # Single metric patterns
    assert Deduplicator._extract_signature("Africa has below average margin of 11.3%") == ("Africa", "margin", 11.3)
    assert Deduplicator._extract_signature("Africa has moderate margin: 11.3%") == ("Africa", "margin", 11.3)
    assert Deduplicator._extract_signature("Africa margin is 11.3%") == ("Africa", "margin", 11.3)
    assert Deduplicator._extract_signature("Africa profit is $88,872") == ("Africa", "profit", 88872)
    assert Deduplicator._extract_signature("Africa revenue of $783,776") == ("Africa", "revenue", 783776)

    # Different entities
    assert Deduplicator._extract_signature("Technology has highest margin of 14.0%") == ("Technology", "margin", 14.0)
    assert Deduplicator._extract_signature("Canada margin is 26.6%") == ("Canada", "margin", 26.6)

    # Multi-metric: should return list of signatures
    result = Deduplicator._extract_signature("Furniture: profit $286,782, margin 7.0%")
    assert ("Furniture", "profit", 286782) in result
    assert ("Furniture", "margin", 7.0) in result
    assert len(result) == 2

    # Edge case: no match returns None or empty list
    assert Deduplicator._extract_signature("Profit margin is calculated as profit divided by sales") is None
```

### Step 2: Run test to verify it fails

Run: `pytest tests/insight_expansion/test_deduplicator.py::test_extract_signature -v`
Expected: FAIL with `AttributeError: type object 'Deduplicator' has no attribute '_extract_signature'`

### Step 3: Write minimal implementation

In `src/insight_expansion/deduplicator.py`, add:

```python
import re
from typing import List, Optional, Tuple

class Deduplicator:
    # ... existing code remains ...

    @staticmethod
    def _extract_signature(text: str) -> Optional[List[Tuple[str, str, float]]]:
        """
        Extract (entity, metric, value) signature(s) from insight text.
        Returns a list of signatures (multiple for multi-metric insights).
        Returns None if no signature can be extracted.
        """
        # Normalize text: remove parenthetical notes, extra spaces
        clean_text = re.sub(r'\s*\([^)]*\)', '', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        signatures = []

        # Pattern 1: "Entity metric is VALUE" e.g., "Africa margin is 11.3%"
        pattern1 = r'^([A-Za-z\s]+?)\s+([a-z_]+)\s+is\s+([\d,]+(?:\.\d+)?%?)'
        match = re.search(pattern1, clean_text, re.IGNORECASE)
        if match:
            entity, metric, value_str = match.groups()
            entity = entity.strip()
            value = Deduplicator._parse_value(value_str)
            if value is not None:
                signatures.append((entity, metric, value))
                return signatures if len(signatures) > 1 else signatures[0] if signatures else None

        # Pattern 2: "Entity has/margin/revenue ... metric of VALUE" e.g., "Africa has below average margin of 11.3%"
        # Variations: "has", "has lowest", "has highest", "achieves", "generates"
        pattern2 = r'^([A-Za-z\s]+?)\s+(?:has|achieves|generates)\s+.*?([a-z_]+)\s+of\s+([\d,]+(?:\.\d+)?%?)'
        match = re.search(pattern2, clean_text, re.IGNORECASE)
        if match:
            entity, metric, value_str = match.groups()
            entity = entity.strip()
            value = Deduplicator._parse_value(value_str)
            if value is not None:
                signatures.append((entity, metric, value))
                return signatures if len(signatures) > 1 else signatures[0] if signatures else None

        # Pattern 3: "Entity: metric1 VALUE1, metric2 VALUE2" (multi-metric)
        pattern3 = r'^([A-Za-z\s]+?):\s+([a-z_]+)\s+([\d,]+(?:\.\d+)?%?)(?:,\s*([a-z_]+)\s+([\d,]+(?:\.\d+)?%?))?'
        match = re.search(pattern3, clean_text, re.IGNORECASE)
        if match:
            groups = match.groups()
            entity = groups[0].strip()
            # First metric-value pair
            metric1, val_str1 = groups[1], groups[2]
            value1 = Deduplicator._parse_value(val_str1)
            if value1 is not None:
                signatures.append((entity, metric1, value1))
            # Optional second metric-value pair
            if len(groups) >= 5 and groups[3] and groups[4]:
                metric2, val_str2 = groups[3], groups[4]
                value2 = Deduplicator._parse_value(val_str2)
                if value2 is not None:
                    signatures.append((entity, metric2, value2))
            if signatures:
                return signatures

        # No pattern matched
        return None

    @staticmethod
    def _parse_value(value_str: str) -> Optional[float]:
        """Parse numeric value from string, handling currency and percentages."""
        if not value_str:
            return None
        # Remove currency symbols, commas
        cleaned = value_str.replace('$', '').replace(',', '').strip()
        # Handle percentage: 11.3% -> 11.3
        if cleaned.endswith('%'):
            cleaned = cleaned[:-1]
        try:
            return float(cleaned)
        except ValueError:
            return None
```

### Step 4: Run test to verify it passes

Run: `pytest tests/insight_expansion/test_deduplicator.py::test_extract_signature -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/insight_expansion/deduplicator.py tests/insight_expansion/test_deduplicator.py
git commit -m "feat: add _extract_signature to deduplicate by entity+metric+value"
```

---

## Task 2: Integrate Signature Matching into Deduplication

**Files:**
- Modify: `src/insight_expansion/deduplicator.py` (deduplicate method)
- Test: `tests/insight_expansion/test_deduplicator.py`

**Objective:** Modify the `deduplicate` method to connect insights sharing the same signature in the similarity graph, ensuring they are clustered together.

### Step 1: Write the failing test

Add to `tests/insight_expansion/test_deduplicator.py`:

```python
def test_signature_based_deduplication():
    """Test that insights with same entity+metric+value are deduplicated even with different text."""
    from insight_expansion.deduplicator import Deduplicator

    insights = [
        {"text": "Africa has below average margin of 11.3%", "dimensions": ["region"], "metrics": ["margin"], "type_hint": "fact"},
        {"text": "Africa has moderate margin: 11.3%", "dimensions": ["region"], "metrics": ["margin"], "type_hint": "fact"},
        {"text": "Africa margin is 11.3%", "dimensions": ["region"], "metrics": ["margin"], "type_hint": "fact"},
        {"text": "Africa profit is $88,872", "dimensions": ["region"], "metrics": ["profit"], "type_hint": "fact"},
        {"text": "Canada has highest margin of 26.6%", "dimensions": ["region"], "metrics": ["margin"], "type_hint": "fact"},
    ]

    result = Deduplicator.deduplicate(insights)

    # Should keep only one Africa margin (11.3%), one Africa profit, one Canada margin
    result_texts = [r["text"] for r in result]
    assert len(result) == 3  # 3 unique signatures

    # Check that one Africa margin version is kept
    margin_africa = [t for t in result_texts if "Africa" in t and "margin" in t]
    assert len(margin_africa) == 1

    # Check Africa profit remains
    profit_africa = [t for t in result_texts if "Africa" in t and "profit" in t]
    assert len(profit_africa) == 1

    # Check Canada margin remains
    canada_margin = [t for t in result_texts if "Canada" in t and "margin" in t]
    assert len(canada_margin) == 1

    # Ensure no duplicate values
    assert "Africa has below average margin of 11.3%" not in result_texts or "Africa has moderate margin: 11.3%" in result_texts
```

### Step 2: Run test to verify it fails

Run: `pytest tests/insight_expansion/test_deduplicator.py::test_signature_based_deduplication -v`
Expected: FAIL (currently returns 5 insights because signatures not used)

### Step 3: Modify deduplicate method

Update `deduplicate` method in `src/insight_expansion/deduplicator.py`:

After building the text similarity graph (lines 30-38), add signature-based edges:

```python
        # Build similarity graph
        adj: List[List[int]] = [[] for _ in range(n)]
        texts = [item[0]["text"] for item in combined]

        # Text similarity edges
        for i in range(n):
            for j in range(i + 1, n):
                sim = SequenceMatcher(None, texts[i].lower(), texts[j].lower()).ratio()
                if sim > Deduplicator.SIMILARITY_THRESHOLD:
                    adj[i].append(j)
                    adj[j].append(i)

        # Signature-based deduplication: connect insights with same entity+metric+value
        signatures_map: Dict[Tuple[str, str, float], List[int]] = {}
        for idx, (insight, _, _) in enumerate(combined):
            sigs = Deduplicator._extract_signature(insight["text"])
            if sigs:
                # Normalize to list
                if not isinstance(sigs, list):
                    sigs = [sigs]
                for sig in sigs:
                    signatures_map.setdefault(sig, []).append(idx)

        # Connect all insights sharing the same signature
        for sig, indices in signatures_map.items():
            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        idx_i = indices[i]
                        idx_j = indices[j]
                        if idx_j not in adj[idx_i]:
                            adj[idx_i].append(idx_j)
                        if idx_i not in adj[idx_j]:
                            adj[idx_j].append(idx_i)
```

### Step 4: Run test to verify it passes

Run: `pytest tests/insight_expansion/test_deduplicator.py::test_signature_based_deduplication -v`
Expected: PASS

Also run all existing tests to ensure no regressions:
`pytest tests/insight_expansion/test_deduplicator.py -v`

### Step 5: Commit

```bash
git add src/insight_expansion/deduplicator.py tests/insight_expansion/test_deduplicator.py
git commit -m "feat: add signature-based deduplication to Deduplicator"
```

---

## Task 3: Verify with Real Data

**Files:**
- Script: `expand_insights.py` (CLI)
- Input: `insights_sample.json` (or provided seed file)
- Output: `output/insights_expanded.json`

**Objective:** Run the insight expansion with the enhanced Deduplicator and confirm the output has no (entity+metric+value) duplicates.

### Step 1: Check that existing tests still pass

Run: `pytest tests/insight_expansion/ -v`
Expected: All tests pass

### Step 2: Run the expansion script

```bash
python expand_insights.py --input insights_sample.json
```

Expected: Script completes successfully and writes `output/insights_expanded.json`.

### Step 3: Verify no signature duplicates in output

Create a quick verification script (or manually check):

```python
import json

with open('output/insights_expanded.json') as f:
    data = json.load(f)
insights = data['insights']

from insight_expansion.deduplicator import Deduplicator

seen_signatures = set()
duplicates = []
for ins in insights:
    sigs = Deduplicator._extract_signature(ins['text'])
    if sigs:
        if not isinstance(sigs, list):
            sigs = [sigs]
        for sig in sigs:
            if sig in seen_signatures:
                duplicates.append((ins['text'], sig))
            else:
                seen_signatures.add(sig)

if duplicates:
    print(f"Found {len(duplicates)} signature duplicates:")
    for text, sig in duplicates[:10]:
        print(f"  {sig}: {text}")
else:
    print("No signature duplicates found!")
```

Expected output: "No signature duplicates found!"

### Step 4: Check total count

The output should have fewer insights than before because duplicates are now removed. Print the count:
```bash
python -c "import json; data=json.load(open('output/insights_expanded.json')); print(f'Total insights: {len(data[\"insights\"])}')"
```

### Step 5: Commit (optional if counts changed)

If the output file changed (deduplication reduced count), commit:

```bash
git add output/insights_expanded.json
git commit -m "chore: regenerate insights with signature-based deduplication"
```

---

## Task 4: Edge Cases and Robustness

**Files:**
- Modify: `src/insight_expansion/deduplicator.py`
- Test: `tests/insight_expansion/test_deduplicator.py`

**Objective:** Handle edge cases: insights with no extractable signature, mixed units, rounding differences.

### Step 1: Write test for edge cases

```python
def test_signature_extraction_edge_cases():
    """Test edge cases for signature extraction."""
    from insight_expansion.deduplicator import Deduplicator

    # No extractable signature returns None
    assert Deduplicator._extract_signature("Profit margin is defined as profit/sales") is None

    # Different number formatting should parse to same value
    sig1 = Deduplicator._extract_signature("Africa margin is 11.30%")
    sig2 = Deduplicator._extract_signature("Africa margin is 11.3%")
    assert sig1 == sig2

    # Currency parsing with commas
    sig = Deduplicator._extract_signature("Africa revenue of $783,776")
    assert sig == ("Africa", "revenue", 783776)

    # Multi-metric in different order: (order doesn't matter if we use set semantics)
    sig_a = Deduplicator._extract_signature("Furniture: profit $286,782, margin 7.0%")
    sig_b = Deduplicator._extract_signature("Furniture: margin 7.0%, profit $286,782")
    # Both should produce same set of signatures
    assert set(sig_a) == set(sig_b)
```

### Step 2: Run test to check failures

Run: `pytest tests/insight_expansion/test_deduplicator.py::test_signature_extraction_edge_cases -v`
Fix any failures by adjusting `_parse_value` or regex patterns.

### Step 3: Run all deduplicator tests

`pytest tests/insight_expansion/test_deduplicator.py -v`
Expected: All pass

### Step 4: Commit

```bash
git add src/insight_expansion/deduplicator.py tests/insight_expansion/test_deduplicator.py
git commit -m "test: add edge case tests for signature extraction"
```

---

## Task 5: (Optional) Precision Tuning

**Files:**
- `src/insight_expansion/deduplicator.py` (SIMILARITY_THRESHOLD)

**Objective:** The existing SIMILARITY_THRESHOLD (0.85) plus signature matching may be too aggressive or too lenient.

**Note:** The user requirement is clear: same entity+metric+value should deduplicate. The score and completeness selection remain for clusters containing multiple insights with different signatures. No Further tuning needed unless tests fail.

---

## Self-Review Checklist

**Spec coverage:**
- [x] Deduplicate based on (entity + metric + value) combination
- [x] Text-only duplicates still caught by similarity
- [x] Run expand_insights.py and verify no signature duplicates in output

**Placeholder scan:**
- [x] All code provided in full
- [x] No "TBD" or "TODO" in implementation steps
- [x] All commands exact with expected results

**Type consistency:**
- `_extract_signature` returns `Optional[List[Tuple[str, str, float]]]`
- Used consistently in deduplicate method
- Test assertions match return types

---

## Plan Summary

- **Files touched:** `src/insight_expansion/deduplicator.py` (add methods, modify deduplicate), `tests/insight_expansion/test_deduplicator.py` (new tests)
- **New functions:** `_extract_signature(text)`, `_parse_value(value_str)`
- **Behavior change:** Insights with same (entity, metric, value) are now considered duplicates regardless of phrasing
- **Backward compatible:** Existing text similarity still works; signature is an additional dedup signal

---

**Plan complete and saved to `docs/superpowers/plans/2025-03-29-deduplicator-enhancement.md`.**

**Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
