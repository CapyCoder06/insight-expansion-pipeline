"""
Tests for retrieval optimization enhancement.
"""

import pytest
import sys
sys.path.insert(0, 'src')

from pipeline.document_generator import enhance_for_retrieval


class TestRetrievalEnhancement:
    """Tests for document retrieval optimization."""

    def test_enhance_adds_query_phrasings(self):
        """Should add query-style questions for better retrieval."""
        text = "# [FACT] Technology has highest profit\n\nTechnology leads with $664K profit."
        enhanced = enhance_for_retrieval(text)

        # Should include original form and query variations
        assert "Which category has the highest profit?" in enhanced or \
               "What category has highest profit?" in enhanced
        assert "?" in enhanced  # Contains questions

    def test_enhance_adds_synonyms_section(self):
        """Should include alternative wordings for key terms."""
        text = "# [FACT] Technology has high profit margin\n\nProfit margin is 14% for Technology."
        enhanced = enhance_for_retrieval(text)

        # Should have alternative phrasing
        assert "earnings" in enhanced.lower() or "revenue" in enhanced.lower()
        assert "top" in enhanced.lower() or "best" in enhanced.lower()

    def test_enhance_preserves_original_meaning(self):
        """Must not alter original facts or add new information."""
        original_text = "# [FACT] Canada achieves 26.6% margin\n\nCanada has best profit margin of 26.6%."
        enhanced = enhance_for_retrieval(enhance_for_retrieval(original_text))

        # Original facts must remain
        assert "Canada" in enhanced
        assert "26.6" in enhanced or "26%" in enhanced
        # No new facts
        assert "USA" not in enhanced
        assert "30%" not in enhanced

    def test_enhance_concise_output(self):
        """Should not make document overly long."""
        text = "# [FACT] Short fact\n\nBrief content."
        enhanced = enhance_for_retrieval(text)

        # Should be reasonably concise (not 10x longer)
        assert len(enhanced) < len(text) * 5

    def test_enhance_handles_different_document_types(self):
        """Should work for FACT, TREND, ANOMALY, METRIC, QUESTION."""
        documents = [
            ("# [FACT] X\n\nX content.", "fact"),
            ("# [TREND] X\n\nX trend content.", "trend"),
            ("# [ANOMALY] X\n\nX anomaly content.", "anomaly"),
            ("# [METRIC] X\n\nX metric content.", "metric"),
            ("# [QUESTION] X\n\nQ: X\nA: Y", "question"),
        ]

        for doc_text, doc_type in documents:
            enhanced = enhance_for_retrieval(doc_text)
            assert doc_type in enhanced.lower() or doc_text.startswith("#")
            assert len(enhanced) >= len(doc_text)  # Has additional content

    def test_enhance_keeps_template_structure(self):
        """Should preserve document template format."""
        text = "# [FACT] Title\n\nDescription\n- Dimensions: x\n- Metrics: y"
        enhanced = enhance_for_retrieval(text)

        # Should keep the header and structure
        assert enhanced.startswith("# [FACT]")
        assert "- Dimensions:" in enhanced
        assert "- Metrics:" in enhanced

    def test_enhance_adds_synonym_variations(self):
        """Should provide synonyms for domain keywords."""
        text = "# [FACT] Sales increased\n\nSales grew by 10%."
        enhanced = enhance_for_retrieval(text)

        enhanced_lower = enhanced.lower()
        # Should include synonym for "sales" or "increased"
        synonyms_present = any(syn in enhanced_lower for syn in [
            "revenue", "turnover", "earnings",  # sales synonyms
            "rose", "climbed", "grew"  # increased synonyms
        ])
        assert synonyms_present, f"Expected synonyms in: {enhanced}"

    def test_enhance_with_numbers_preserves_values(self):
        """Must preserve exact numeric values."""
        text = "# [FACT] Profit margin is 14.5%\n\nMargin equals 14.5 percent."
        enhanced = enhance_for_retrieval(text)

        assert "14.5" in enhanced or "14.5%" in enhanced
        # Should not change to 15% or other values
        assert "15%" not in enhanced.replace("15.5", "")  # Allow if it's 14.5%

    def test_enhance_callable_multiple_times(self):
        """Should be safe to call enhancement multiple times."""
        text = "# [FACT] Base text\n\nContent."
        enhanced1 = enhance_for_retrieval(text)
        enhanced2 = enhance_for_retrieval(enhanced1)

        # Should not break structure
        assert "# [" in enhanced2
        assert len(enhanced2) >= len(enhanced1)  # Might add more, but not break

    def test_enhance_does_not_mark_as_ai_generated(self):
        """Should avoid AI/imagination markers."""
        text = "# [FACT] Standard fact\n\nNormal content."
        enhanced = enhance_for_retrieval(text)

        ai_indicators = ["as an AI", "I don't", "I cannot", "I'm sorry", "language model"]
        for indicator in ai_indicators:
            assert indicator.lower() not in enhanced.lower()


class TestRetrievalIntegration:
    """Integration with pipeline."""

    def test_enhanced_documents_valid_pipeline(self):
        """Enhanced documents should still pass validation."""
        from pipeline.document_generator import DocumentGenerator
        from pipeline.validator import DocumentValidator

        generator = DocumentGenerator()
        validator = DocumentValidator()

        insight = {
            "text": "Technology has highest profit",
            "dimensions": ["category"],
            "metrics": ["profit"]
        }

        doc = generator.generate_document(insight)
        enhanced = enhance_for_retrieval(doc["text"])

        # Check enhanced version has same metadata structure
        assert "type" in doc["metadata"]
        assert "fact" in enhanced.lower()  # Type prefix maintained

        # Should not break validation logic
        validation = validator.validate({
            "text": enhanced,
            "metadata": doc["metadata"]
        })
        # Validation should pass or have only minor warnings
        assert validation["valid"], f"Validation failed: {validation['issues']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
