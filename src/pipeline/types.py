"""
Document types for the RAG pipeline.
"""

from enum import Enum


class DocumentType(str, Enum):
    """Supported document types."""
    FACT = "fact"
    TREND = "trend"
    ANOMALY = "anomaly"
    METRIC = "metric"
    QUESTION = "question"
