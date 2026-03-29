"""
Maps insights to document types based on content analysis.
"""

from typing import Dict, Any, Optional
from enum import Enum
from .types import DocumentType


class TemplateMapper:
    """Maps structured insights to appropriate document types."""

    def __init__(self):
        """Initialize the mapper with keyword patterns."""
        self.type_keywords = {
            DocumentType.FACT: [
                "highest", "lowest", "leads", "best", "worst",
                "most", "least", "maximum", "minimum", "top", "bottom"
            ],
            DocumentType.TREND: [
                "increase", "decrease", "growth", "decline", "rising",
                "falling", "upward", "downward", "cagr", "yoy", "year-over-year"
            ],
            DocumentType.ANOMALY: [
                "anomaly", "outlier", "unusual", "abnormal", "spike",
                "drop", "negative", "loss", "losses", "losing", "lose",
                "problem", "issue", "unprofitable", "debt", "underperforming"
            ],
            DocumentType.METRIC: [
                "margin", "profit margin", "definition", "formula", "calculate"
            ],
            DocumentType.QUESTION: [
                "why", "how", "what", "?", "question", "wonder"
            ]
        }

    def map_insight(self, insight: Dict[str, Any]) -> DocumentType:
        """
        Map an insight to a document type.

        Args:
            insight: Structured insight with keys: text, dimensions, metrics, type_hint

        Returns:
            The most appropriate document type
        """
        # Use explicit type_hint if provided
        if "type_hint" in insight:
            try:
                return DocumentType(insight["type_hint"])
            except (ValueError, KeyError):
                pass

        # Analyze text content
        text = insight.get("text", "").lower()

        # Check for anomaly indicators (profit < 0)
        if "profit" in insight.get("metrics", []):
            if "negative" in text or any(word in text for word in ["loss", "lose", "negative"]):
                return DocumentType.ANOMALY

        # Score each type based on keyword matches
        scores = {}
        for doc_type, keywords in self.type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            scores[doc_type] = score

        # Return highest scoring type
        if max(scores.values(), default=0) > 0:
            return max(scores, key=lambda k: scores[k])

        # Default to FACT for simple statements
        return DocumentType.FACT
