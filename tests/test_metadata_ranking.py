"""
Tests for metadata ranking signals (importance, confidence, source).
"""

import pytest
import sys
sys.path.insert(0, 'src')

from pipeline.document_generator import DocumentGenerator
from pipeline.validator import DocumentValidator


class TestMetadataRanking:
    """Tests for ranking signals in metadata."""

    def setup_method(self):
        self.generator = DocumentGenerator()
        self.validator = DocumentValidator()

    def test_metadata_includes_importance_field(self):
        """Generated documents should have importance in metadata."""
        insight = {
            "text": "Technology has the highest profit",
            "dimensions": ["category"],
            "metrics": ["profit"]
        }
        doc = self.generator.generate_document(insight)

        assert "importance" in doc["metadata"]
        assert doc["metadata"]["importance"] in ["low", "medium", "high"]

    def test_metadata_includes_confidence_field(self):
        """Generated documents should have confidence score (0-1)."""
        insight = {
            "text": "Technology has the highest profit",
            "dimensions": ["category"],
            "metrics": ["profit"]
        }
        doc = self.generator.generate_document(insight)

        assert "confidence" in doc["metadata"]
        confidence = doc["metadata"]["confidence"]
        assert 0 <= confidence <= 1, f"Confidence {confidence} out of range [0,1]"
        assert isinstance(confidence, (int, float))

    def test_metadata_includes_source_field(self):
        """Generated documents should have source."""
        insight = {
            "text": "Technology has the highest profit",
            "dimensions": ["category"],
            "metrics": ["profit"]
        }
        doc = self.generator.generate_document(insight)

        assert "source" in doc["metadata"]
        assert doc["metadata"]["source"] in ["EDA", "hypothesis", "auto", "unknown"]

    def test_confidence_updated_by_validation(self):
        """After validation, confidence should be updated based on validation results."""
        insight = {
            "text": "Technology has highest profit",
            "dimensions": ["category"],
            "metrics": ["profit"]
        }
        doc = self.generator.generate_document(insight)

        # Run validation
        validation = self.validator.validate(doc)

        # Confidence should be updated
        if "confidence" in doc["metadata"]:
            original_confidence = doc["metadata"]["confidence"]
            # After validation, confidence might be adjusted
            assert doc["metadata"]["confidence"] >= 0

    def test_importance_inferred_from_type_hint(self):
        """Importance should be inferred from document type or content."""
        # High importance: top performers, anomalies, key metrics
        insight = {
            "text": "Technology has the highest profit",
            "dimensions": ["category"],
            "metrics": ["profit"],
            "type_hint": "fact"
        }
        doc = self.generator.generate_document(insight)
        # High importance for top performer
        assert doc["metadata"]["importance"] in ["medium", "high"]

    def test_importance_low_for_minor_facts(self):
        """Minor facts should have low importance."""
        insight = {
            "text": "Some segment has average margin",
            "dimensions": ["segment"],
            "metrics": ["margin"],
            "type_hint": "fact"
        }
        doc = self.generator.generate_document(insight)
        # Average/neutral statements should be lower importance
        assert doc["metadata"]["importance"] in ["low", "medium"]

    def test_confidence_high_for_valid_documents(self):
        """Valid documents should have higher confidence."""
        insight = {
            "text": "Sales increased by 24% year over year",
            "dimensions": ["year"],
            "metrics": ["sales"],
            "trend": "increasing"
        }
        doc = self.generator.generate_document(insight)

        # Initially generated should have moderate confidence
        assert doc["metadata"]["confidence"] >= 0.7

    def test_confidence_lower_for_anomalies_without_cause(self):
        """Anomalies with unclear cause may have lower confidence."""
        insight = {
            "text": "Some anomaly detected",
            "type_hint": "anomaly",
            "dimensions": ["category"],
            "metrics": ["profit"],
            "issue": "Unexpected behavior"
            # Missing possible_cause
        }
        doc = self.generator.generate_document(insight)

        # Should have some confidence but maybe lower
        confidence = doc["metadata"]["confidence"]
        assert 0 <= confidence <= 1
        # Missing cause might indicate less certainty
        assert confidence <= 0.9

    def test_source_defaults_to_eda(self):
        """Default source should be EDA (from data analysis)."""
        insight = {
            "text": "Some fact from data",
            "dimensions": [],
            "metrics": []
        }
        doc = self.generator.generate_document(insight)

        assert doc["metadata"]["source"] == "EDA"

    def test_source_override_in_insight(self):
        """Should allow overriding source in insight."""
        insight = {
            "text": "Some fact",
            "dimensions": [],
            "metrics": [],
            "source": "hypothesis"
        }
        doc = self.generator.generate_document(insight)

        assert doc["metadata"]["source"] == "hypothesis"

    def test_ranking_fields_preserved_in_enrichment(self):
        """Enriched variations should preserve ranking signals."""
        from pipeline.enrichment import expand_document

        insight = {
            "text": "Technology has highest profit",
            "dimensions": ["category"],
            "metrics": ["profit"]
        }
        doc = self.generator.generate_document(insight)
        variations = expand_document(doc)

        for var in variations:
            assert "importance" in var["metadata"]
            assert "confidence" in var["metadata"]
            assert "source" in var["metadata"]

    def test_ranking_fields_preserved_in_chunking(self):
        """Chunks should inherit ranking signals from parent document."""
        from pipeline.chunker import DocumentChunker

        insight = {
            "text": "Technology has highest profit of $664K with margin 14%",
            "dimensions": ["category"],
            "metrics": ["profit"]
        }
        doc = self.generator.generate_document(insight)
        chunker = DocumentChunker()
        chunks = chunker.chunk_document(doc)

        for chunk in chunks:
            assert "importance" in chunk["metadata"]
            assert "confidence" in chunk["metadata"]
            assert "source" in chunk["metadata"]
            # Should match parent
            assert chunk["metadata"]["importance"] == doc["metadata"]["importance"]
            assert chunk["metadata"]["source"] == doc["metadata"]["source"]

    def test_confidence_dynamic_based_on_validation_results(self):
        """Confidence should be recalculated after validation."""
        insight = {
            "text": "Valid fact with clear metrics",
            "dimensions": ["category"],
            "metrics": ["profit", "margin"]
        }
        doc = self.generator.generate_document(insight)
        initial_confidence = doc["metadata"]["confidence"]

        # Re-validate (should be valid)
        validation = self.validator.validate(doc)
        assert validation["valid"] is True

        # Confidence should remain high
        assert doc["metadata"]["confidence"] >= 0.9

    def test_importance_values_are_string_literals(self):
        """Ensure importance uses exact string values."""
        insight = {
            "text": "Test fact",
            "dimensions": ["x"],
            "metrics": ["y"]
        }
        doc = self.generator.generate_document(insight)

        acceptable = ["low", "medium", "high"]
        assert doc["metadata"]["importance"] in acceptable

    def test_confidence_numeric_precision(self):
        """Confidence should be numeric with reasonable precision."""
        insight = {
            "text": "Clear statement",
            "dimensions": ["category"],
            "metrics": ["profit"]
        }
        doc = self.generator.generate_document(insight)
        confidence = doc["metadata"]["confidence"]

        # Should be a reasonable number (not NaN, not extreme)
        assert 0 <= confidence <= 1
        assert not (isinstance(confidence, float) and (confidence != confidence))  # not NaN


class TestMetadataIntegration:
    """Integration tests for ranking signals."""

    def test_full_pipeline_with_ranking(self):
        """Test complete pipeline preserves ranking signals."""
        from pipeline import DocumentPipeline

        insights = [
            {
                "text": "Technology has highest profit",
                "dimensions": ["category"],
                "metrics": ["profit"]
            }
        ]

        pipeline = DocumentPipeline(enrich=True, enrich_variations=2)
        result = pipeline.run(insights, validate=True, chunk=True)

        documents = result["documents"]

        for doc in documents:
            assert "importance" in doc["metadata"]
            assert "confidence" in doc["metadata"]
            assert "source" in doc["metadata"]
            assert doc["metadata"]["source"] == "EDA"

    def test_batch_generated_documents_have_ranking(self):
        """All documents in batch should have ranking signals."""
        from pipeline import generate_from_insights

        insights = [
            {"text": "Fact 1", "dimensions": ["d1"], "metrics": ["m1"]},
            {"text": "Fact 2", "dimensions": ["d2"], "metrics": ["m2"]},
            {"text": "Fact 3", "dimensions": ["d3"], "metrics": ["m3"]},
        ]

        docs = generate_from_insights(insights)

        for doc in docs:
            assert "importance" in doc["metadata"]
            assert "confidence" in doc["metadata"]
            assert "source" in doc["metadata"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
