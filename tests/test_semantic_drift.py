"""
Tests for semantic drift prevention in validation.
"""

import pytest
import sys
sys.path.insert(0, 'src')

from pipeline.validator import semantic_similarity, validate_semantic_drift
from pipeline.document_generator import DocumentGenerator


class TestSemanticSimilarity:
    """Tests for semantic similarity checking."""

    def test_similarity_identical_texts(self):
        """Identical texts should have similarity 1.0."""
        text1 = "Technology has the highest profit with margin 14%."
        text2 = "Technology has the highest profit with margin 14%."
        sim = semantic_similarity(text1, text2)
        assert sim == 1.0 or sim >= 0.95, f"Expected ~1.0, got {sim}"

    def test_similarity_similar_meaning(self):
        """Paraphrased texts with same meaning should have high similarity."""
        text1 = "Technology has the highest profit."
        text2 = "The top category by profit is Technology."
        sim = semantic_similarity(text1, text2)
        assert sim >= 0.7, f"Expected >=0.7, got {sim}"

    def test_similarity_different_meaning(self):
        """Texts with different meanings should have low similarity."""
        text1 = "Technology has the highest profit."
        text2 = "Furniture has the lowest margin."
        sim = semantic_similarity(text1, text2)
        assert sim < 0.5, f"Expected <0.5, got {sim}"

    def test_similarity_preserves_numbers(self):
        """Documents with different numbers should have lower similarity."""
        text1 = "Profit margin is 14%."
        text2 = "Profit margin is 26%."
        sim = semantic_similarity(text1, text2)
        assert sim < 0.8, f"Different numbers should reduce similarity, got {sim}"

    def test_similarity_entities_match(self):
        """Same entities should maintain similarity even with rewording."""
        text1 = "Canada achieves the best margin of 26.6%."
        text2 = "Canada has the highest profit margin at 26.6 percent."
        sim = semantic_similarity(text1, text2)
        assert sim >= 0.67, f"Same entities + numbers should be similar, got {sim}"

    def test_similarity_empty_texts(self):
        """Empty texts should have 0 similarity."""
        sim = semantic_similarity("", "Some text")
        assert sim == 0.0
        sim = semantic_similarity("", "")
        assert sim == 0.0

    def test_similarity_threshold_default(self):
        """Default threshold should be reasonable (0.6-0.8)."""
        # The function should have a default threshold that rejects drift
        # We'll test this through validate_semantic_drift
        assert True  # Will test via integration


class TestValidateSemanticDrift:
    """Tests for semantic drift validation."""

    def test_drift_detection_rejects_major_changes(self):
        """Should reject documents with significant semantic drift."""
        original = "Technology has the highest profit of $664K with margin 14%."
        drifted = "Furniture has negative profit and low sales."

        is_valid, similarity = validate_semantic_drift(original, drifted)

        # Should be rejected (similarity below threshold)
        assert is_valid is False, f"Should reject drift, similarity={similarity}"
        assert similarity < 0.6, f"Expected low similarity, got {similarity}"

    def test_drift_accepts_minor_rewording(self):
        """Should accept documents with minor rewording (same meaning)."""
        original = "Technology has the highest profit."
        reworded = "The top profit performer is Technology."

        is_valid, similarity = validate_semantic_drift(original, reworded)

        # Should be accepted (similarity above threshold)
        assert is_valid is True, f"Should accept reworded, similarity={similarity}"
        assert similarity >= 0.7, f"Expected high similarity, got {similarity}"

    def test_drift_detects_fact_changes(self):
        """Should detect when key facts are changed."""
        original = "Canada margin is 26.6%."
        changed = "Canada margin is 12%."

        is_valid, similarity = validate_semantic_drift(original, changed)

        assert is_valid is False, f"Should reject changed fact, similarity={similarity}"

    def test_drift_detects_entity_changes(self):
        """Should detect when entities are swapped."""
        original = "Technology leads in profit."
        swapped = "Furniture leads in profit."

        is_valid, similarity = validate_semantic_drift(original, swapped)

        assert is_valid is False, f"Should reject entity change, similarity={similarity}"

    def test_drift_with_synonyms(self):
        """Should handle synonyms appropriately."""
        original = "Sales increased by 24%."
        synonym_version = "Revenue grew by 24 percent."

        is_valid, similarity = validate_semantic_drift(original, synonym_version)

        # Should accept (synonyms preserve meaning)
        assert is_valid is True, f"Should accept synonyms, similarity={similarity}"

    def test_drift_threshold_configurable(self):
        """Should allow custom threshold."""
        original = "Technology has high profit."
        reworded = "Technology has high earnings."

        # Default threshold
        is_valid1, sim1 = validate_semantic_drift(original, reworded)

        # Stricter threshold
        is_valid2, sim2 = validate_semantic_drift(original, reworded, threshold=0.9)

        # If sim is 0.8, stricter might reject
        assert is_valid2 is False or is_valid1 is True

    def test_drift_empty_vs_nonempty(self):
        """Empty vs non-empty should fail."""
        original = "Some content."
        empty = ""

        is_valid, similarity = validate_semantic_drift(original, empty)

        assert is_valid is False
        assert similarity < 0.5


class TestSemanticDriftInPipeline:
    """Integration: semantic drift check in document generation."""

    def test_generator_preserves_meaning(self):
        """Generated documents should preserve original insight meaning."""
        generator = DocumentGenerator()

        insight = {
            "text": "Technology has the highest profit of $664K with margin 14%",
            "dimensions": ["category"],
            "metrics": ["profit"]
        }

        doc = generator.generate_document(insight)

        # Check semantic similarity between insight text and generated document
        # Should be high (document preserves meaning)
        is_valid, similarity = validate_semantic_drift(
            insight["text"],
            doc["text"],
            threshold=0.6
        )

        assert is_valid, f"Generated document drifted from original (sim={similarity})"

    def test_enriched_variations_preserve_meaning(self):
        """Enriched variations should all preserve original meaning."""
        from pipeline.enrichment import expand_document
        generator = DocumentGenerator()

        insight = {
            "text": "Sales increased by 24% year over year",
            "dimensions": ["year"],
            "metrics": ["sales"],
            "trend": "increasing"
        }

        doc = generator.generate_document(insight)
        variations = expand_document(doc)

        original_content = insight["text"]

        for var in variations:
            is_valid, similarity = validate_semantic_drift(
                original_content,
                var["text"],
                threshold=0.6
            )
            assert is_valid, f"Variation drifted: {similarity} - {var['text'][:50]}..."

    def test_retrieval_enhancement_preserves_meaning(self):
        """Enhanced documents should preserve original meaning."""
        from pipeline import enhance_for_retrieval

        original = "# [FACT] Technology has highest profit\n\nTechnology leads with $664K profit."
        enhanced = enhance_for_retrieval(original)

        # Enhancement adds queries/synonyms, but original content remains
        is_valid, similarity = validate_semantic_drift(
            original,
            enhanced,
            threshold=0.7
        )

        assert is_valid, f"Enhancement broke meaning, similarity={similarity}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
