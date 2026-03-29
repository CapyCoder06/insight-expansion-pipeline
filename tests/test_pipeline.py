"""
Tests for the document pipeline.
"""

import pytest
import sys
sys.path.insert(0, 'src')

from pipeline.types import DocumentType
from pipeline.template_mapper import TemplateMapper
from pipeline.document_generator import DocumentGenerator, generate_from_insights
from pipeline.validator import DocumentValidator
from pipeline.chunker import DocumentChunker


class TestTemplateMapper:
    """Tests for TemplateMapper."""

    def setup_method(self):
        self.mapper = TemplateMapper()

    def test_map_fact_highest(self):
        insight = {
            "text": "Technology has the highest profit",
            "dimensions": ["category"],
            "metrics": ["profit"]
        }
        result = self.mapper.map_insight(insight)
        assert result == DocumentType.FACT

    def test_map_fact_best(self):
        insight = {
            "text": "Canada has the best margin",
            "dimensions": ["market"],
            "metrics": ["margin"]
        }
        result = self.mapper.map_insight(insight)
        assert result == DocumentType.FACT

    def test_map_trend_increase(self):
        insight = {
            "text": "Sales are increasing year over year with 24% CAGR",
            "dimensions": [],
            "metrics": ["sales"]
        }
        result = self.mapper.map_insight(insight)
        assert result == DocumentType.TREND

    def test_map_anomaly_negative_profit(self):
        insight = {
            "text": "Tables sub-category is losing money with negative profit",
            "dimensions": ["sub_category"],
            "metrics": ["profit"]
        }
        result = self.mapper.map_insight(insight)
        assert result == DocumentType.ANOMALY

    def test_map_with_type_hint(self):
        insight = {
            "text": "Some random statement",
            "type_hint": "metric",
            "dimensions": [],
            "metrics": ["margin"]
        }
        result = self.mapper.map_insight(insight)
        assert result == DocumentType.METRIC


class TestDocumentGenerator:
    """Tests for DocumentGenerator."""

    def setup_method(self):
        self.generator = DocumentGenerator()

    def test_generate_fact(self):
        insight = {
            "text": "Technology has the highest profit of $664K with margin 14%",
            "dimensions": ["category"],
            "metrics": ["profit"]
        }
        doc = self.generator.generate_document(insight)

        assert "text" in doc
        assert "metadata" in doc
        assert doc["metadata"]["type"] == "fact"
        assert "[FACT]" in doc["text"]
        assert "Technology" in doc["text"]
        assert "profit" in doc["metadata"]["metrics"]

    def test_generate_trend(self):
        insight = {
            "text": "Sales increased from $2.3M in 2011 to $4.3M in 2014",
            "dimensions": ["year"],
            "metrics": ["sales"],
            "trend": "increasing"
        }
        doc = self.generator.generate_document(insight)

        assert doc["metadata"]["type"] == "trend"
        assert "[TREND]" in doc["text"]
        assert doc["metadata"]["trend"] == "increasing"

    def test_generate_anomaly(self):
        insight = {
            "text": "Tables sub-category has negative profit of -$64K",
            "dimensions": ["sub_category"],
            "metrics": ["profit"],
            "issue": "Sustained losses over 4 years",
            "possible_cause": "High discount rates and low pricing"
        }
        doc = self.generator.generate_document(insight)

        assert doc["metadata"]["type"] == "anomaly"
        assert "[ANOMALY]" in doc["text"]
        assert "negative profit" in doc["text"].lower()

    def test_generate_metric(self):
        insight = {
            "text": "Profit margin measures profitability as profit divided by sales",
            "dimensions": [],
            "metrics": ["profit", "sales"],
            "definition": "The percentage of revenue that translates to profit",
            "formula": "profit / sales * 100%",
            "interpretation": "Higher margin indicates better efficiency"
        }
        doc = self.generator.generate_document(insight)

        assert doc["metadata"]["type"] == "metric"
        assert "[METRIC]" in doc["text"]
        assert "formula" in doc["text"].lower()

    def test_generate_question(self):
        insight = {
            "text": "Why is Southeast Asia margin only 2%?",
            "question": "What causes Southeast Asia's low profit margin?",
            "answer": "High discount rates and operational costs reduce profitability"
        }
        doc = self.generator.generate_document(insight)

        assert doc["metadata"]["type"] == "question"
        assert "[QUESTION]" in doc["text"]


class TestValidator:
    """Tests for DocumentValidator."""

    def setup_method(self):
        self.validator = DocumentValidator()

    def test_valid_document(self):
        doc = {
            "text": "# [FACT] Technology has highest profit\n\nTechnology leads with $664K profit.",
            "metadata": {
                "type": "fact",
                "dimensions": ["category"],
                "metrics": ["profit"]
            }
        }
        result = self.validator.validate(doc)
        assert result["valid"] is True
        assert len(result["issues"]) == 0

    def test_missing_text(self):
        doc = {
            "text": "",
            "metadata": {"type": "fact"}
        }
        result = self.validator.validate(doc)
        assert result["valid"] is False
        assert "empty" in result["issues"][0].lower()

    def test_missing_type(self):
        doc = {
            "text": "Some content",
            "metadata": {"dimensions": ["category"]}
        }
        result = self.validator.validate(doc)
        assert result["valid"] is False
        assert "type" in result["issues"][0].lower()

    def test_invalid_type(self):
        doc = {
            "text": "# [INVALID] Something",
            "metadata": {"type": "invalid_type"}
        }
        result = self.validator.validate(doc)
        assert result["valid"] is False
        assert "Invalid document type" in result["issues"][0]

    def test_missing_prefix(self):
        doc = {
            "text": "Technology has highest profit",
            "metadata": {"type": "fact", "dimensions": [], "metrics": []}
        }
        result = self.validator.validate(doc)
        assert result["valid"] is False
        assert "prefix" in result["issues"][0].lower()


class TestChunker:
    """Tests for DocumentChunker."""

    def setup_method(self):
        self.chunker = DocumentChunker(chunk_size=50, overlap=10)

    def test_small_document_no_chunk(self):
        text = "This is a short document with only a few words."
        doc = {"text": text, "metadata": {"type": "fact"}}
        chunks = self.chunker.chunk_document(doc)

        assert len(chunks) == 1
        assert chunks[0]["text"] == text
        assert chunks[0]["chunk_index"] == 0
        assert chunks[0]["chunk_count"] == 1

    def test_large_document_multiple_chunks(self):
        words = ["word"] * 100
        text = " ".join(words)
        doc = {"text": text, "metadata": {"type": "fact"}}
        chunks = self.chunker.chunk_document(doc)

        assert len(chunks) > 1
        # Check that metadata is preserved
        for chunk in chunks:
            assert chunk["metadata"]["type"] == "fact"
            assert "chunk_index" in chunk
            assert "chunk_count" in chunk

    def test_metadata_preserved(self):
        text = " ".join(["word"] * 80)
        doc = {"text": text, "metadata": {"type": "trend", "dimensions": ["year"]}}
        chunks = self.chunker.chunk_document(doc)

        for chunk in chunks:
            assert chunk["metadata"]["type"] == "trend"
            assert chunk["metadata"]["dimensions"] == ["year"]

    def test_batch_chunking(self):
        doc1 = {"text": " ".join(["word"] * 60), "metadata": {"type": "fact"}}
        doc2 = {"text": " ".join(["word"] * 60), "metadata": {"type": "trend"}}
        chunks = self.chunker.chunk_batch([doc1, doc2])

        assert len(chunks) > 1
        # Each chunk should have _doc_id
        for chunk in chunks:
            assert "_doc_id" in chunk["metadata"]
            assert chunk["metadata"]["_doc_id"] in [0, 1]


class TestIntegration:
    """Integration tests."""

    def test_full_pipeline(self):
        insights = [
            {
                "text": "Technology has the highest profit of $664K with margin 14%",
                "dimensions": ["category"],
                "metrics": ["profit"]
            },
            {
                "text": "Sales increased by 24% year over year from 2011 to 2014",
                "dimensions": ["year"],
                "metrics": ["sales"],
                "trend": "increasing"
            },
            {
                "text": "Tables sub-category has negative profit of -$64K",
                "dimensions": ["sub_category"],
                "metrics": ["profit"],
                "issue": "Sustained losses",
                "possible_cause": "High discount rates"
            }
        ]

        generator = DocumentGenerator()
        validator = DocumentValidator()
        chunker = DocumentChunker()

        # Generate
        docs = generate_from_insights(insights)
        assert len(docs) == 3

        # Validate - all should pass basic checks
        for doc in docs:
            result = validator.validate(doc)
            assert result["valid"], f"Invalid doc: {result['issues']}"

        # Chunk
        chunks = chunker.chunk_batch(docs)
        assert len(chunks) >= 3

        # Each chunk should have metadata
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk
            assert "_doc_id" in chunk["metadata"]
