"""Validation for generated insights."""

import re
from typing import Dict, List, Tuple
from difflib import SequenceMatcher
from .data_query import DataQueryEngine


class InsightValidator:
    """Validates insights against dataset and rules."""

    ALLOWED_TYPES = {"fact", "trend", "anomaly", "metric", "question"}
    VALID_DIMENSIONS = {"region", "category"}
    VALID_METRICS = {"revenue", "profit", "margin"}
    VALUE_TOLERANCE = 0.05  # ±5%

    @staticmethod
    def validate(insight: Dict, engine: DataQueryEngine, seed_text: str = None) -> Tuple[bool, List[str]]:
        """Validate an insight. Returns (is_valid, error_messages)."""
        errors = []

        # 1. Required fields non-empty
        required_fields = ["text", "dimensions", "metrics", "type_hint"]
        for field in required_fields:
            if field not in insight or not insight[field]:
                errors.append(f"Missing or empty required field: {field}. Must provide a valid value.")

        if errors:
            return False, errors

        # 2. type_hint must be valid
        if insight["type_hint"] not in InsightValidator.ALLOWED_TYPES:
            errors.append(f"Invalid type_hint: {insight['type_hint']}. Valid options: {InsightValidator.ALLOWED_TYPES}")

        # 3. dimensions must be non-empty and all valid column names
        if not insight["dimensions"]:
            errors.append("Dimensions list cannot be empty")
        else:
            invalid_dims = [d for d in insight["dimensions"] if d not in InsightValidator.VALID_DIMENSIONS]
            if invalid_dims:
                errors.append(f"Invalid dimension(s): {invalid_dims}. Valid: {InsightValidator.VALID_DIMENSIONS}")

        # 4. metrics must be non-empty and all valid column names
        if not insight["metrics"]:
            errors.append("Metrics list cannot be empty")
        else:
            invalid_mets = [m for m in insight["metrics"] if m not in InsightValidator.VALID_METRICS]
            if invalid_mets:
                errors.append(f"Invalid metric(s): {invalid_mets}. Valid: {InsightValidator.VALID_METRICS}")

        # 5. For anomaly: if issue present, possible_cause must be present
        if insight["type_hint"] == "anomaly":
            if "issue" in insight and "possible_cause" not in insight:
                errors.append("Anomaly insight with 'issue' must also have 'possible_cause'")

        # 6. Numeric values must exist in dataset (±5%)
        text = insight["text"]
        # For each metric, extract numeric values from text and verify existence in raw data rows
        for metric in insight["metrics"]:
            if metric not in ["revenue", "profit", "margin"]:
                continue
            # Pattern: for profit/revenue: $ number (with commas, optional decimals); for margin: number followed by %
            if metric in ["profit", "revenue"]:
                pattern = r'\$(-?\d[\d,]*(?:\.\d+)?)'
            else:  # margin
                pattern = r'([\d,]+(?:\.\d+)?)%'

            matches = re.findall(pattern, text)
            if not matches:
                # No numeric value for this metric in text, skip validation
                continue

            # For each matched value (usually one), check it exists somewhere in dataset
            for value_str in matches:
                try:
                    text_value = float(value_str.replace(',', ''))
                except ValueError:
                    errors.append(f"Could not parse numeric value for {metric}: {value_str}")
                    continue

                # Check if ANY row in the raw data has this metric value within tolerance
                found = False
                for row in engine._data:
                    try:
                        row_value = float(row.get(metric, '')) if row.get(metric) is not None else None
                        if row_value is not None:
                            # Relative tolerance, avoid division by zero
                            denominator = max(abs(row_value), 1.0)
                            if abs(row_value - text_value) / denominator <= InsightValidator.VALUE_TOLERANCE:
                                found = True
                                break
                    except (ValueError, TypeError):
                        continue

                if not found:
                    errors.append(f"Numeric value {text_value} for {metric} not found in dataset within ±5% tolerance")

        # 7. Text similarity check if seed_text provided
        if seed_text:
            similarity = InsightValidator._simple_similarity(insight["text"], seed_text)
            if similarity >= 0.85:
                errors.append(f"Insight text too similar to seed (similarity: {similarity:.2f})")

        return len(errors) == 0, errors

    @staticmethod
    def _simple_similarity(text1: str, text2: str) -> float:
        """Compute simple character-level similarity (0-1) using difflib."""
        # Normalize
        t1 = ' '.join(text1.lower().split())
        t2 = ' '.join(text2.lower().split())

        if t1 == t2:
            return 1.0
        return SequenceMatcher(None, t1, t2).ratio()
