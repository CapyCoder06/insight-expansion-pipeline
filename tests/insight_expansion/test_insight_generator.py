"""Tests for InsightGenerator."""

import sys
import os
test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(test_dir))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from insight_expansion.insight_generator import InsightGenerator
from insight_expansion.data_query import DataQueryEngine


def test_generate_combined_insight_produces_valid_fact():
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


def test_generate_insights_creates_insights_for_all_categories():
    seed = {
        "text": "Technology has highest profit with margin 14%",
        "dimensions": ["category"],
        "metrics": ["profit", "margin"],
        "type_hint": "fact"
    }
    engine = DataQueryEngine()
    insights = InsightGenerator.generate_insights(seed, engine)
    # Should have insights for both primary (category) and alternate (region) dimensions
    assert len(insights) >= 3
    # Check dimensions are either category or region
    dims_set = set(tuple(ins["dimensions"]) for ins in insights)
    assert ("category",) in dims_set or "category" in [d[0] for d in dims_set if d]
    assert ("region",) in dims_set or "region" in [d[0] for d in dims_set if d]
    # Check metrics: with splitting should have single-metric insights
    for ins in insights:
        assert len(ins["metrics"]) == 1  # split
        assert ins["metrics"][0] in ["profit", "margin"]


def test_generate_insights_splits_multi_metric_fact():
    """With two metrics and fact type, should produce 2 insights per entity."""
    seed = {
        "text": "Technology has highest profit with margin 14%",
        "dimensions": ["category"],
        "metrics": ["profit", "margin"],
        "type_hint": "fact"
    }
    engine = DataQueryEngine()
    insights = InsightGenerator.generate_insights(seed, engine)
    # Count insights for category dimension only (exclude regions for this check)
    cat_insights = [ins for ins in insights if ins["dimensions"] == ["category"]]
    # 3 categories × 2 metrics = 6
    assert len(cat_insights) == 6
    # Ensure we have both metrics represented in separate insights
    metrics_found = set()
    for ins in cat_insights:
        metrics_found.update(ins["metrics"])
    assert metrics_found == {"profit", "margin"}


def test_generate_anomaly_only_includes_entities_with_negative_profit():
    engine = DataQueryEngine()
    seed = {
        "text": "Tables has negative profit",
        "dimensions": ["category"],
        "metrics": ["profit"],
        "type_hint": "anomaly",
        "issue": "Negative profit",
        "possible_cause": "High discounts"
    }
    # Determine if any category has negative profit
    categories = engine.get_unique_values("category")
    any_negative = any(
        engine.get_entity_metrics("category", cat, ["profit"])["profit"] < 0
        for cat in categories
    )

    insights = InsightGenerator.generate_insights(seed, engine)

    if any_negative:
        assert len(insights) > 0, "Expected at least one insight when negative profit entities exist"
        # Verify all insights correspond to negative profit entities
        # Get all possible entities for lookup (to parse entity from text)
        all_entities = engine.get_unique_values("category")
        # Sort by length descending to match multi-word entities first
        all_entities_sorted = sorted(all_entities, key=len, reverse=True)
        for insight in insights:
            text = insight["text"]
            # Extract entity from start of text
            entity = None
            for e in all_entities_sorted:
                if text.startswith(e):
                    entity = e
                    break
            assert entity is not None, f"Could not extract entity from insight text: {text}"
            dim = insight["dimensions"][0]
            metrics = engine.get_entity_metrics(dim, entity, ["profit"])
            assert metrics["profit"] < 0, f"Entity {entity} in dimension {dim} has non-negative profit {metrics['profit']}"
    else:
        assert len(insights) == 0, f"Expected 0 insights when no negative profit entities, got {len(insights)}"

