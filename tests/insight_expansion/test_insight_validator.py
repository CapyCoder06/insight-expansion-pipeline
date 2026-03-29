"""Tests for InsightValidator."""

import sys
import os
test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(test_dir))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from insight_expansion.insight_validator import InsightValidator
from insight_expansion.data_query import DataQueryEngine


def test_validates_non_empty_required_fields():
    """Test that empty required fields are rejected."""
    engine = DataQueryEngine()

    # Missing text
    insight = {
        "dimensions": ["category"],
        "metrics": ["profit"],
        "type_hint": "fact"
    }
    valid, errors = InsightValidator.validate(insight, engine)
    assert not valid
    assert "text" in errors[0].lower()

    # Missing dimensions
    insight = {
        "text": "Something something",
        "metrics": ["profit"],
        "type_hint": "fact"
    }
    valid, errors = InsightValidator.validate(insight, engine)
    assert not valid
    assert "dimensions" in errors[0].lower()

    # Missing metrics
    insight = {
        "text": "Something something",
        "dimensions": ["category"],
        "type_hint": "fact"
    }
    valid, errors = InsightValidator.validate(insight, engine)
    assert not valid
    assert "metrics" in errors[0].lower()

    # Missing type_hint
    insight = {
        "text": "Something something",
        "dimensions": ["category"],
        "metrics": ["profit"]
    }
    valid, errors = InsightValidator.validate(insight, engine)
    assert not valid
    assert "type_hint" in errors[0].lower()


def test_validates_type_hint_must_be_valid():
    """Test that type_hint is one of allowed values."""
    engine = DataQueryEngine()

    for invalid_type in ["invalid", "foo", "TREND", ""]:
        insight = {
            "text": "Something something",
            "dimensions": ["category"],
            "metrics": ["profit"],
            "type_hint": invalid_type
        }
        valid, errors = InsightValidator.validate(insight, engine)
        assert not valid, f"Should reject invalid type_hint: {invalid_type}"
        assert any("type_hint" in err.lower() and "valid" in err.lower() for err in errors)


def test_validates_dimensions_must_be_valid_column_names():
    """Test that all dimensions are valid column names in dataset."""
    engine = DataQueryEngine()

    insight = {
        "text": "Something something",
        "dimensions": ["invalid_column"],
        "metrics": ["profit"],
        "type_hint": "fact"
    }
    valid, errors = InsightValidator.validate(insight, engine)
    assert not valid
    assert any("dimension" in err.lower() and "valid" in err.lower() for err in errors)

    # Mixed valid and invalid
    insight = {
        "text": "Something something",
        "dimensions": ["category", "invalid_column"],
        "metrics": ["profit"],
        "type_hint": "fact"
    }
    valid, errors = InsightValidator.validate(insight, engine)
    assert not valid
    assert any("dimension" in err.lower() and "invalid" in err.lower() for err in errors)


def test_validates_metrics_must_be_valid_column_names():
    """Test that all metrics are valid column names in dataset."""
    engine = DataQueryEngine()

    insight = {
        "text": "Something something",
        "dimensions": ["category"],
        "metrics": ["invalid_metric"],
        "type_hint": "fact"
    }
    valid, errors = InsightValidator.validate(insight, engine)
    assert not valid
    assert any("metric" in err.lower() and "valid" in err.lower() for err in errors)

    # Mixed valid and invalid
    insight = {
        "text": "Something something",
        "dimensions": ["category"],
        "metrics": ["profit", "invalid_metric"],
        "type_hint": "fact"
    }
    valid, errors = InsightValidator.validate(insight, engine)
    assert not valid


def test_validates_numeric_values_exist_in_dataset():
    """Test that all numeric values in text exist in dataset within ±5% tolerance."""
    engine = DataQueryEngine()

    # Insight with a value that doesn't exist for that entity/dimension/metric combo
    insight = {
        "text": "Technology has profit of $999,999",  # unrealistic value
        "dimensions": ["category"],
        "metrics": ["profit"],
        "type_hint": "fact"
    }
    valid, errors = InsightValidator.validate(insight, engine)
    assert not valid
    assert any("value" in err.lower() and ("dataset" in err.lower() or "found" in err.lower()) for err in errors)

    # Insight with value that does exist (approximately)
    insight = {
        "text": "Technology has profit of $135,538",  # close to actual 135,538.42
        "dimensions": ["category"],
        "metrics": ["profit"],
        "type_hint": "fact"
    }
    valid, errors = InsightValidator.validate(insight, engine)
    assert valid, f"Should accept value within tolerance: {errors}"


def test_validates_anomaly_type_requires_possible_cause_if_issue_present():
    """Test that for anomaly type, if issue is present, possible_cause must also be present."""
    engine = DataQueryEngine()

    # Anomaly with issue but no possible_cause
    insight = {
        "text": "Furniture has negative profit",
        "dimensions": ["category"],
        "metrics": ["profit"],
        "type_hint": "anomaly",
        "issue": "Negative profit"
    }
    valid, errors = InsightValidator.validate(insight, engine)
    assert not valid
    assert any("possible_cause" in err.lower() for err in errors)

    # Anomaly with both issue and possible_cause
    insight = {
        "text": "Furniture has negative profit",
        "dimensions": ["category"],
        "metrics": ["profit"],
        "type_hint": "anomaly",
        "issue": "Negative profit",
        "possible_cause": "High discounts"
    }
    valid, errors = InsightValidator.validate(insight, engine)
    assert valid, f"Should accept anomaly with both fields: {errors}"

    # Anomaly without issue (possible_cause optional)
    insight = {
        "text": "Furniture has negative profit",
        "dimensions": ["category"],
        "metrics": ["profit"],
        "type_hint": "anomaly"
    }
    valid, errors = InsightValidator.validate(insight, engine)
    assert valid, f"Should accept anomaly without issue/cause: {errors}"


def test_validates_insight_text_not_identical_to_seed():
    """Test that insight text must not be identical or near-identical to seed text (similarity < 0.85)."""
    engine = DataQueryEngine()

    seed = {
        "text": "Technology has highest profit with margin 14%",
        "dimensions": ["category"],
        "metrics": ["profit", "margin"],
        "type_hint": "fact"
    }

    # Identical text should fail
    insight = {
        "text": "Technology has highest profit with margin 14%",
        "dimensions": ["category"],
        "metrics": ["profit"],
        "type_hint": "fact"
    }
    valid, errors = InsightValidator.validate(insight, engine, seed_text=seed["text"])
    assert not valid
    assert any("similar" in err.lower() or "identical" in err.lower() for err in errors)

    # Near-identical (small change) should fail if similarity >= 0.85
    insight = {
        "text": "Technology has highest profit with margin 15%",  # only 1 char diff
        "dimensions": ["category"],
        "metrics": ["profit"],
        "type_hint": "fact"
    }
    valid, errors = InsightValidator.validate(insight, engine, seed_text=seed["text"])
    assert not valid, "Near-identical text should fail similarity check"

    # Different enough should pass
    insight = {
        "text": "Furniture has lowest profit with margin 5%",
        "dimensions": ["category"],
        "metrics": ["profit"],
        "type_hint": "fact"
    }
    valid, errors = InsightValidator.validate(insight, engine, seed_text=seed["text"])
    assert valid, f"Different text should pass similarity check: {errors}"


def test_valid_insight_passes_all_checks():
    """Test a completely valid insight passes validation."""
    engine = DataQueryEngine()

    insight = {
        "text": "Furniture has profit of $54,550",
        "dimensions": ["category"],
        "metrics": ["profit"],
        "type_hint": "fact"
    }
    valid, errors = InsightValidator.validate(insight, engine)
    assert valid, f"Valid insight should pass: {errors}"


def test_validates_multiple_dimensions_all_must_be_valid():
    """Test that with multiple dimensions, all must be valid."""
    engine = DataQueryEngine()

    insight = {
        "text": "Something something",
        "dimensions": ["category", "region", "invalid_dim"],
        "metrics": ["profit"],
        "type_hint": "fact"
    }
    valid, errors = InsightValidator.validate(insight, engine)
    assert not valid
    assert any("dimension" in err.lower() for err in errors)


def test_validates_multiple_metrics_all_must_be_valid():
    """Test that with multiple metrics, all must be valid."""
    engine = DataQueryEngine()

    insight = {
        "text": "Something something",
        "dimensions": ["category"],
        "metrics": ["profit", "invalid_metric", "margin"],
        "type_hint": "fact"
    }
    valid, errors = InsightValidator.validate(insight, engine)
    assert not valid
    assert any("metric" in err.lower() for err in errors)


def test_numeric_value_extraction_from_margin():
    """Test that margin values (with %) are correctly validated."""
    engine = DataQueryEngine()

    # Valid margin value (Technology margin ~13%)
    insight = {
        "text": "Technology has margin of 13.0%",
        "dimensions": ["category"],
        "metrics": ["margin"],
        "type_hint": "fact"
    }
    valid, errors = InsightValidator.validate(insight, engine)
    assert valid, f"Valid margin should pass: {errors}"

    # Invalid margin value
    insight = {
        "text": "Technology has margin of 999%",
        "dimensions": ["category"],
        "metrics": ["margin"],
        "type_hint": "fact"
    }
    valid, errors = InsightValidator.validate(insight, engine)
    assert not valid
    assert any("value" in err.lower() for err in errors)


def test_numeric_value_multiple_metrics_in_text():
    """Test validation when text contains multiple numeric values."""
    engine = DataQueryEngine()

    # Both values should be validated
    insight = {
        "text": "Technology has profit of $135,538 and margin of 13.0%",
        "dimensions": ["category"],
        "metrics": ["profit", "margin"],
        "type_hint": "fact"
    }
    valid, errors = InsightValidator.validate(insight, engine)
    assert valid, f"Valid multi-metric insight should pass: {errors}"

    # One invalid value should fail
    insight = {
        "text": "Technology has profit of $999,999 and margin of 13.0%",
        "dimensions": ["category"],
        "metrics": ["profit", "margin"],
        "type_hint": "fact"
    }
    valid, errors = InsightValidator.validate(insight, engine)
    assert not valid
    assert any("profit" in err.lower() and "value" in err.lower() for err in errors)
