"""Tests for Deduplicator."""

import sys
import os
test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(test_dir))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from insight_expansion.deduplicator import Deduplicator


def test_removes_exact_duplicate_texts():
    """Exact duplicate text entries should be removed, keeping first occurrence."""
    insights = [
        {"text": "Technology has highest profit", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"},
        {"text": "Technology has highest profit", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"},
        {"text": "Furniture has lowest profit", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"},
    ]
    result = Deduplicator.deduplicate(insights)
    assert len(result) == 2
    assert result[0]["text"] == "Technology has highest profit"
    assert result[1]["text"] == "Furniture has lowest profit"


def test_removes_near_duplicates_above_similarity_threshold():
    """Insights with text similarity > 0.85 should be considered duplicates."""
    insights = [
        {"text": "Technology has highest profit", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"},
        {"text": "Technology has highest profit!", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"},
        {"text": "Furniture has lowest profit", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"},
    ]
    result = Deduplicator.deduplicate(insights)
    assert len(result) == 2
    # Should keep first (more complete?) - both are identical in fields, so keep first
    assert result[0]["text"] == "Technology has highest profit"
    assert result[1]["text"] == "Furniture has lowest profit"


def test_keeps_distinct_texts_below_similarity_threshold():
    """Insights with similarity <= 0.85 should be kept."""
    insights = [
        {"text": "Technology has highest profit", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"},
        {"text": "Furniture has lowest profit", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"},
    ]
    result = Deduplicator.deduplicate(insights)
    assert len(result) == 2
    assert result[0]["text"] == "Technology has highest profit"
    assert result[1]["text"] == "Furniture has lowest profit"


def test_preserves_original_order():
    """Deduplication should preserve the relative order of kept insights."""
    insights = [
        {"text": "A", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"},
        {"text": "B", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"},
        {"text": "C", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"},
        {"text": "A", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"},
    ]
    result = Deduplicator.deduplicate(insights)
    assert [ins["text"] for ins in result] == ["A", "B", "C"]


def test_near_duplicate_keeps_more_complete_insight():
    """When near-duplicates differ, keep the one with more complete fields."""
    # Anomaly with issue/possible_cause is more complete than without
    insights = [
        {"text": "Technology has negative profit", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "anomaly"},
        {"text": "Technology has negative profit??", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "anomaly", "issue": "Negative profit", "possible_cause": "High discounts"},
    ]
    result = Deduplicator.deduplicate(insights)
    assert len(result) == 1
    assert result[0].get("issue") == "Negative profit"
    assert result[0].get("possible_cause") == "High discounts"


def test_near_duplicate_keeps_more_metrics():
    """Keep insight with more metrics when near-duplicate."""
    insights = [
        {"text": "Technology has metrics", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"},
        {"text": "Technology has metrics data", "dimensions": ["category"], "metrics": ["profit", "margin"], "type_hint": "fact"},
    ]
    result = Deduplicator.deduplicate(insights)
    assert len(result) == 1
    assert result[0]["metrics"] == ["profit", "margin"]


def test_near_duplicate_keeps_anomaly_with_cause():
    """Anomaly with both issue and cause beats one with only issue."""
    insights = [
        {"text": "Furniture negative profit", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "anomaly", "issue": "Negative profit"},
        {"text": "Furniture negative profit high", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "anomaly", "issue": "Negative profit", "possible_cause": "High discounts"},
    ]
    result = Deduplicator.deduplicate(insights)
    assert len(result) == 1
    kept = result[0]
    assert "possible_cause" in kept
    assert kept["possible_cause"] == "High discounts"


def test_seed_insights_not_removed():
    """Should preserve seed insights even if duplicates exist in expanded set."""
    insights = [
        {"text": "Seed insight", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"},
        {"text": "Seed insight", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"},
        {"text": "Another seed insight", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"},
        {"text": "Expanded insight", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"},
    ]
    seeds = [
        {"text": "Seed insight", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"},
        {"text": "Another seed insight", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"},
    ]
    result = Deduplicator.deduplicate(insights, seed_insights=seeds)
    # Seeds should be kept, and expanded insight kept. Duplicates among seeds? Seeds shouldn't be in insights list ideally but handle robustly.
    assert len(result) == 3
    texts = [ins["text"] for ins in result]
    assert "Seed insight" in texts
    assert "Another seed insight" in texts
    assert "Expanded insight" in texts


def test_seed_insights_exact_duplicates_among_themselves():
    """If seeds have duplicates among themselves, they're preserved as-is (or deduped?). Assume seeds are deduped separately."""
    # For simplicity: seeds list is assumed already unique; but if duplicates exist, we still should preserve all seeds? Likely deduplicate overall.
    # We'll assume seeds are unique pre-input, but robust code shouldn't drop seeds.
    insights = [
        {"text": "Expanded alpha", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"},
        {"text": "Expanded beta", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"},
    ]
    seeds = [
        {"text": "Seed X", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"},
        {"text": "Seed X", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"},  # duplicate within seeds
    ]
    result = Deduplicator.deduplicate(insights, seed_insights=seeds)
    # Both seeds present? Or one? Let's decide: seeds are sacred; don't deduplicate among them. Keep both.
    assert len(result) == 4
    assert sum(1 for ins in result if ins["text"] == "Seed X") == 2


def test_multiple_groups_of_near_duplicates():
    """Handle multiple overlapping duplicate clusters."""
    insights = [
        {"text": "Alpha region has high revenue", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"},
        {"text": "Alpha region has high revenue now", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"},
        {"text": "Beta category low margin", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"},
        {"text": "Beta category low margin trend", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"},
        {"text": "Gamma product profit increased", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"},
    ]
    result = Deduplicator.deduplicate(insights)
    assert len(result) == 3
    texts = {ins["text"] for ins in result}
    # The kept ones should be the first from each cluster (most complete same, so first)
    assert "Alpha region has high revenue" in texts
    assert "Beta category low margin" in texts
    assert "Gamma product profit increased" in texts


def test_near_duplicate_with_same_completeness_keeps_first():
    """If near-duplicates have equal completeness, keep the first occurrence."""
    insights = [
        {"text": "Tech high profit", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"},
        {"text": "Tech high profit!!", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"},
    ]
    result = Deduplicator.deduplicate(insights)
    assert len(result) == 1
    assert result[0]["text"] == "Tech high profit"


def test_keeps_insight_with_extra_optional_fields():
    """More optional fields = more complete."""
    insights = [
        {"text": "Report shows high profit", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"},
        {"text": "Report shows high profit updated", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact", "extra": "value"},
    ]
    result = Deduplicator.deduplicate(insights)
    assert len(result) == 1
    assert "extra" in result[0]


def test_deduplication_does_not_mutate_input():
    """Should not modify the original insights list."""
    insights = [
        {"text": "A", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"},
        {"text": "A", "dimensions": ["category"], "metrics": ["profit"], "type_hint": "fact"},
    ]
    original = insights.copy()
    Deduplicator.deduplicate(insights)
    assert insights == original


def test_extract_signature():
    """Test signature extraction from various text patterns."""
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


def test_signature_based_deduplication():
    """Test that insights with same entity+metric+value are deduplicated even with different text."""
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
