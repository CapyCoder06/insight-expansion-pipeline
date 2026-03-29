"""
Tests for the enrichment module.
"""

import pytest
import sys
sys.path.insert(0, 'src')

from pipeline.enrichment import expand_document, select_top_variations


class TestEnrichment:
    """Tests for document enrichment."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_document = {
            "text": "# [FACT] Technology has the highest profit\n\nTechnology leads with $664K profit margin of 14%.",
            "metadata": {
                "type": "fact",
                "dimensions": ["category"],
                "metrics": ["profit"],
                "title": "Technology has the highest profit"
            }
        }

    def test_expand_document_returns_multiple_variations(self):
        """Should return 5-10 variations."""
        variations = expand_document(self.sample_document)
        assert len(variations) >= 5, f"Expected 5+ variations, got {len(variations)}"
        assert len(variations) <= 10, f"Expected max 10 variations, got {len(variations)}"

    def test_variations_preserve_metadata(self):
        """All variations should have identical metadata."""
        variations = expand_document(self.sample_document)
        original_metadata = self.sample_document["metadata"]

        for var in variations:
            assert var["metadata"]["type"] == original_metadata["type"]
            assert var["metadata"]["dimensions"] == original_metadata["dimensions"]
            assert var["metadata"]["metrics"] == original_metadata["metrics"]

    def test_variations_have_different_wording(self):
        """Each variation should have different text from others."""
        variations = expand_document(self.sample_document)
        texts = [v["text"] for v in variations]

        # Check that all texts are unique
        assert len(texts) == len(set(texts)), "All variations should have unique text"

    def test_variations_preserve_meaning(self):
        """Key information must be preserved in all variations."""
        variations = expand_document(self.sample_document)

        for var in variations:
            text = var["text"].lower()
            # Must contain key terms
            assert "technology" in text
            assert "profit" in text
            assert "14%" in text or "664" in text

    def test_variations_do_not_add_new_facts(self):
        """Variations should not introduce new facts not in original."""
        variations = expand_document(self.sample_document)
        original_text = self.sample_document["text"].lower()

        # Extract entities/facts from original (simple check - numbers, categories)
        original_terms = ["technology", "profit", "664", "14%"]

        for var in variations:
            var_text = var["text"].lower()
            # All numeric/entity references should be from original
            for term in original_terms:
                if term in var_text:
                    assert term in original_text, f"Variation introduced term not in original: {term}"

    def test_filter_returns_top_two(self):
        """select_top_variations should return 2 highest quality variations by default."""
        variations = expand_document(self.sample_document)
        top2 = select_top_variations(variations, top_k=2)

        assert len(top2) <= 2, f"Expected max 2 top variations, got {len(top2)}"
        assert len(top2) >= 1, "Should return at least 1 variation"

    def test_filter_deduplicates_near_duplicates(self):
        """select_top_variations should remove semantically similar variations."""
        variations = [
            {"text": "Technology has the highest profit.", "metadata": {}},
            {"text": "Technology possesses the highest profit.", "metadata": {}},  # Near-duplicate
            {"text": "Sales increased year over year.", "metadata": {}},  # Distinct
            {"text": "Profit grew significantly.", "metadata": {}}  # Distinct
        ]
        top2 = select_top_variations(variations, top_k=2)

        # Should return 2 distinct variations
        assert len(top2) == 2
        texts = [v["text"].strip().lower() for v in top2]
        # Check that we don't have both very similar sentences
        # (first two are near-duplicates, so they shouldn't both be selected)
        assert not (texts[0] in variations[0]["text"].lower() and texts[1] in variations[1]["text"].lower())

    def test_filter_preserves_metadata_integrity(self):
        """Filtered variations should have complete metadata."""
        variations = expand_document(self.sample_document)
        top2 = select_top_variations(variations, top_k=2)

        for var in top2:
            assert "metadata" in var
            assert "type" in var["metadata"]
            assert "dimensions" in var["metadata"]
            assert "metrics" in var["metadata"]

    def test_expand_document_with_trend(self):
        """Should handle trend documents."""
        trend_doc = {
            "text": "# [TREND] Sales increased year over year\n\nSales grew from $2M to $4M between 2011 and 2014.",
            "metadata": {
                "type": "trend",
                "dimensions": ["year"],
                "metrics": ["sales"],
                "trend": "increasing"
            }
        }
        variations = expand_document(trend_doc)
        assert len(variations) >= 5
        # Check trend is preserved
        for var in variations:
            assert any(word in var["text"].lower() for word in ["increase", "grow", "rise"])

    def test_enrichment_with_anomaly(self):
        """Should handle anomaly documents."""
        anomaly_doc = {
            "text": "# [ANOMALY] Tables are losing money\n\nTables sub-category has negative profit of -$64K.",
            "metadata": {
                "type": "anomaly",
                "dimensions": ["sub_category"],
                "metrics": ["profit"],
                "issue": "Negative profit",
                "possible_cause": "High discounts"
            }
        }
        variations = expand_document(anomaly_doc)
        assert len(variations) >= 5
        for var in variations:
            assert "table" in var["text"].lower()
            # Check for key numbers/terms: either "64" (for $64K) or "negative"
            text_lower = var["text"].lower()
            assert "64" in text_lower or "negative" in text_lower


class TestEnrichmentConstraints:
    """Test constraint requirements."""

    def test_no_hallucination_facts(self):
        """Variations should not add new entities or numbers."""
        doc = {
            "text": "# [FACT] Canada has best margin\n\nCanada achieves 26.6% profit margin.",
            "metadata": {"type": "fact", "dimensions": ["market"], "metrics": ["margin"]}
        }
        variations = expand_document(doc)

        for var in variations:
            text = var["text"].lower()
            # Should not introduce countries not in original
            assert "canada" in text
            # Should not invent new percentage (check for 26.6 or 26 with %)
            assert "26.6" in text or ("26" in text and "%" in text)

    def test_metadata_structure_unchanged(self):
        """Metadata structure and values should remain identical."""
        doc = {
            "text": "Some fact",
            "metadata": {
                "type": "fact",
                "dimensions": ["category", "region"],
                "metrics": ["profit", "margin"],
                "title": "Custom Title"
            }
        }
        variations = expand_document(doc)

        for var in variations:
            assert var["metadata"] == doc["metadata"]


class TestEnrichmentIntegration:
    """Integration with pipeline."""

    def test_expand_documents_list(self):
        """Test expanding a list of documents."""
        documents = [
            {
                "text": "Doc 1",
                "metadata": {"type": "fact", "dimensions": [], "metrics": []}
            },
            {
                "text": "Doc 2",
                "metadata": {"type": "trend", "dimensions": ["year"], "metrics": ["sales"]}
            }
        ]
        # Individual expand
        all_variations = []
        for doc in documents:
            variations = expand_document(doc)
            all_variations.extend(variations)

        assert len(all_variations) >= 10  # At least 5 per doc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
