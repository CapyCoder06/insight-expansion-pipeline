"""
Integration tests for retrieval optimization in the pipeline.
"""

import pytest
import sys
sys.path.insert(0, 'src')

from pipeline import DocumentPipeline
from pipeline.document_generator import enhance_for_retrieval


class TestRetrievalOptimizationIntegration:
    """Tests for retrieval optimization integration."""

    def test_optimization_enabled_adds_metadata(self):
        """When optimize_retrieval=True, metadata should have queries and synonyms."""
        pipeline = DocumentPipeline(
            enrich=False,
            optimize_retrieval=True
        )

        insights = [
            {
                "text": "Technology has the highest profit",
                "dimensions": ["category"],
                "metrics": ["profit"]
            }
        ]

        result = pipeline.run(insights, validate=False, chunk=False)
        documents = result["documents"]

        assert len(documents) == 1
        doc = documents[0]
        doc_text = doc["text"]
        metadata = doc["metadata"]

        # Text should remain CLEAN (no appended sections)
        assert "Technology" in doc_text
        assert "highest profit" in doc_text.lower()
        assert "Retrieval Queries" not in doc_text
        assert "Alternative Terms" not in doc_text

        # Metadata should have retrieval fields
        assert "queries" in metadata
        assert isinstance(metadata["queries"], list)
        assert len(metadata["queries"]) > 0
        assert any("?" in q for q in metadata["queries"])  # Contains questions

        assert "synonyms" in metadata
        assert isinstance(metadata["synonyms"], dict)

    def test_optimization_disabled_preserves_original(self):
        """When optimize_retrieval=False, metadata should NOT have queries/synonyms fields."""
        pipeline = DocumentPipeline(
            enrich=False,
            optimize_retrieval=False
        )

        insights = [
            {
                "text": "Technology has the highest profit",
                "dimensions": ["category"],
                "metrics": ["profit"]
            }
        ]

        result = pipeline.run(insights, validate=False, chunk=False)
        documents = result["documents"]

        assert len(documents) == 1
        doc = documents[0]
        doc_text = doc["text"]
        metadata = doc["metadata"]

        # Should contain original content only
        assert "Technology" in doc_text
        assert "# [FACT]" in doc_text

        # Should NOT have retrieval metadata fields
        assert "queries" not in metadata
        assert "synonyms" not in metadata

    def test_optimization_works_with_enrichment(self):
        """Retrieval optimization should work alongside enrichment."""
        pipeline = DocumentPipeline(
            enrich=True,
            enrich_variations=2,
            optimize_retrieval=True
        )

        insights = [
            {
                "text": "Sales increased by 24% year over year",
                "dimensions": ["year"],
                "metrics": ["sales"],
                "trend": "increasing"
            }
        ]

        result = pipeline.run(insights, validate=False, chunk=False)
        documents = result["documents"]

        # Should have enrichment + optimization
        assert len(documents) >= 2  # At least 2 variations

        for doc in documents:
            doc_text = doc["text"]
            # Text should remain clean (no appended sections)
            assert "Retrieval Queries" not in doc_text
            assert "Alternative Terms" not in doc_text

            # Metadata should have retrieval fields
            assert "queries" in doc["metadata"]
            assert "synonyms" in doc["metadata"]

            # Metadata should be preserved
            assert "type" in doc["metadata"]
            assert doc["metadata"]["type"] in ["trend", "fact"]

    def test_optimization_preserves_metadata(self):
        """Metadata should be unchanged after optimization."""
        pipeline = DocumentPipeline(
            enrich=False,
            optimize_retrieval=True
        )

        insights = [
            {
                "text": "Canada achieves 26.6% margin",
                "dimensions": ["country"],
                "metrics": ["margin"]
            }
        ]

        result = pipeline.run(insights, validate=False, chunk=False)
        doc = result["documents"][0]

        # Check metadata keys preserved
        assert "type" in doc["metadata"]
        assert "dimensions" in doc["metadata"]
        assert "metrics" in doc["metadata"]
        assert doc["metadata"]["dimensions"] == ["country"]
        assert doc["metadata"]["metrics"] == ["margin"]

    def test_optimization_adds_synonyms_for_key_terms(self):
        """Metadata should contain synonyms for domain keywords."""
        pipeline = DocumentPipeline(
            enrich=False,
            optimize_retrieval=True
        )

        insights = [
            {
                "text": "Profit margin is highest for Technology",
                "dimensions": ["category"],
                "metrics": ["profit"]
            }
        ]

        result = pipeline.run(insights, validate=False, chunk=False)
        metadata = result["documents"][0]["metadata"]

        # Metadata should have synonyms
        assert "synonyms" in metadata
        synonyms = metadata["synonyms"]
        assert isinstance(synonyms, dict)

        # Should include at least one synonym for "profit" or "highest"
        all_synonyms = []
        for term_alts in synonyms.values():
            all_synonyms.extend(term_alts)
        has_synonym = any(syn in all_synonyms for syn in ["earnings", "income", "maximum", "peak", "top"])
        assert has_synonym, f"Expected synonyms in: {synonyms}"

    def test_optimization_preserves_numeric_values(self):
        """Exact numeric values must remain unchanged in text."""
        pipeline = DocumentPipeline(
            enrich=False,
            optimize_retrieval=True
        )

        insights = [
            {
                "text": "Sales grew by 24.5% to $664K",
                "dimensions": ["year"],
                "metrics": ["sales"]
            }
        ]

        result = pipeline.run(insights, validate=False, chunk=False)
        doc_text = result["documents"][0]["text"]

        # Text should contain original values exactly
        assert "24.5" in doc_text or "24.5%" in doc_text
        assert "$664K" in doc_text or "664" in doc_text

    def test_optimization_does_not_hallucinate(self):
        """Metadata should not introduce new entities in queries/synonyms."""
        pipeline = DocumentPipeline(
            enrich=False,
            optimize_retrieval=True
        )

        insights = [
            {
                "text": "Technology has highest profit",
                "dimensions": ["category"],
                "metrics": ["profit"]
            }
        ]

        result = pipeline.run(insights, validate=False, chunk=False)
        metadata = result["documents"][0]["metadata"]

        # All queries and synonyms should be generic (not adding new entities)
        queries = metadata.get("queries", [])
        for q in queries:
            assert "Furniture" not in q
            assert "Office" not in q

        synonyms = metadata.get("synonyms", {})
        for term, alts in synonyms.items():
            for alt in alts:
                assert "Furniture" not in alt
                assert "Office" not in alt

    def test_optimization_multiple_documents(self):
        """Should work correctly with multiple insights."""
        pipeline = DocumentPipeline(
            enrich=False,
            optimize_retrieval=True
        )

        insights = [
            {"text": "Technology has highest profit", "dimensions": ["category"], "metrics": ["profit"]},
            {"text": "Sales decreased in Q4", "dimensions": ["quarter"], "metrics": ["sales"], "trend": "decreasing"}
        ]

        result = pipeline.run(insights, validate=False, chunk=False)
        documents = result["documents"]

        assert len(documents) == 2
        for doc in documents:
            metadata = doc["metadata"]
            # Each doc should have retrieval metadata
            assert "queries" in metadata
            assert "synonyms" in metadata

    def test_optimization_callable_multiple_times(self):
        """Pipeline should be safe to call multiple times."""
        pipeline = DocumentPipeline(
            enrich=False,
            optimize_retrieval=True
        )

        insights = [
            {"text": "Test fact", "dimensions": ["x"], "metrics": ["y"]}
        ]

        result1 = pipeline.run(insights, validate=False, chunk=False)
        result2 = pipeline.run(insights, validate=False, chunk=False)

        assert len(result1["documents"]) == len(result2["documents"])
        # Both should have optimization metadata
        assert "queries" in result1["documents"][0]["metadata"]
        assert "queries" in result2["documents"][0]["metadata"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
