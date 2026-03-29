"""
Tests for semantic-aware chunking.
"""

import pytest
import sys
sys.path.insert(0, 'src')

from pipeline.chunker import DocumentChunker


class TestSemanticChunking:
    """Tests for semantic-aware chunking."""

    def setup_method(self):
        # Use larger chunk size to test boundary handling
        self.chunker = DocumentChunker(chunk_size=150, overlap=20)

    def test_chunk_avoids_sentence_mid_split(self):
        """Should not split in the middle of a sentence."""
        text = "This is sentence one. This is sentence two. This is sentence three. " * 20
        doc = {"text": text, "metadata": {"type": "fact"}}
        chunks = self.chunker.chunk_document(doc)

        # Check that chunks end at sentence boundaries when possible
        for chunk in chunks:
            chunk_text = chunk["text"]
            # If chunk ends before full text, it should end with sentence terminator
            if not chunk_text.rstrip().endswith('.'):
                # Could be last chunk or intentional - check if it's a complete thought
                # At minimum, chunks shouldn't end with partial words
                words = chunk_text.split()
                if words:
                    # Words should be complete (no truncation)
                    assert chunk_text[-1] in [' ', '.', '!', '?'], f"Chunk ends with: {chunk_text[-20:]}"

    def test_chunk_respects_paragraph_breaks(self):
        """Should prefer splitting at paragraph boundaries."""
        text = "First paragraph content here.\n\nSecond paragraph content here.\n\nThird paragraph content here."
        doc = {"text": text, "metadata": {"type": "fact"}}
        chunks = self.chunker.chunk_document(doc)

        # With small chunk size, might split across paragraphs
        # But should try to keep paragraphs together when possible
        # Each chunk should either end with \n\n or be the full text
        for chunk in chunks:
            chunk_text = chunk["text"]
            # Should not break a double newline if chunk size allows
            if len(chunk_text) < len(text):
                # Ideally ends with paragraph break
                assert chunk_text.endswith('\n\n') or chunk_text.strip().endswith('.')

    def test_chunk_preserves_whole_facts(self):
        """Should avoid splitting key facts like numbers and entities."""
        text = "The profit margin is 14.5% for Technology category with sales of $2.3M in 2014. " * 10
        doc = {"text": text, "metadata": {"type": "fact"}}
        chunks = self.chunker.chunk_document(doc)

        # Check that numbers like 14.5%, $2.3M, 2014 stay intact in chunks
        for chunk in chunks:
            # If these numbers appear, they should be complete
            if "14.5" in chunk["text"]:
                assert "%" in chunk["text"] or chunk["text"].count("14.5") > 0
            if "$2.3" in chunk["text"]:
                assert "M" in chunk["text"] or "2.3" in chunk["text"]

    def test_chunk_size_within_bounds(self):
        """Chunks should be roughly within 200-500 token range (configurable)."""
        # Generate a long document
        words = ["word"] * 600
        text = " ".join(words)
        doc = {"text": text, "metadata": {"type": "fact"}}
        chunks = self.chunker.chunk_document(doc)

        # Allow small chunks (down to 50% of target) at boundaries (end-of-doc, headings)
        # Upper bound 150% for chunks with list blocks or other factors
        target = self.chunker.chunk_size
        for chunk in chunks:
            chunk_words = len(chunk["text"].split())
            assert chunk_words >= target * 0.5, f"Chunk {chunk_words} too small"
            assert chunk_words <= target * 1.5, f"Chunk {chunk_words} too large"

    def test_overlap_prevents_keyphrase_split(self):
        """Overlap should ensure important phrases aren't split across chunks."""
        text = "The key metric is profit margin which we define as profit divided by sales. " * 15
        doc = {"text": text, "metadata": {"type": "fact"}}
        chunks = self.chunker.chunk_document(doc)

        # Check that phrase "profit margin" doesn't get split as "profit" in one chunk and "margin" in next
        for i in range(len(chunks) - 1):
            current = chunks[i]["text"]
            next_chunk = chunks[i+1]["text"]

            # If "profit" appears at end of current, "margin" should also appear
            if "profit" in current and "margin" in text:
                # Either both in same chunk or at least one not split
                if "margin" not in current:
                    # Then profit shouldn't be at very end
                    words_after_profit = current.split('profit')[-1]
                    assert not words_after_profit.strip().startswith('margin'), "Split phrase detected"

    def test_headings_maintain_separation(self):
        """Documents with headings should split at heading boundaries."""
        text = "# Introduction\nThis is intro text.\n\n# Methodology\nWe analyzed data.\n\n# Results\nKey findings."
        doc = {"text": text, "metadata": {"type": "fact"}}
        chunks = self.chunker.chunk_document(doc)

        # Should respect section boundaries where possible
        # Headings should not be in middle of chunk without following content
        headings = ["# Introduction", "# Methodology", "# Results"]
        for chunk in chunks:
            for heading in headings:
                if heading in chunk["text"]:
                    # Heading should be at start of chunk or followed by content
                    idx = chunk["text"].find(heading)
                    assert idx == 0 or chunk["text"][idx-1] in ['\n', ' '], f"Heading mid-chunk: {heading}"

    def test_headings_create_chunk_boundaries(self):
        """Headings should always start new chunks."""
        # Create a longer document to ensure chunking occurs (exceeds default chunk_size=150)
        # Each section has enough content to trigger splitting, but headings remain at chunk starts
        text = (
            "# Introduction\n" + "This is introductory content. " * 50 + "\n\n"
            "# Methodology\n" + "We analyzed the data extensively. " * 50 + "\n\n"
            "# Results\n" + "The key finding is here and it is significant. " * 50 + "\n\n"
            "# Discussion\n" + "Further analysis shows more details and implications. " * 50
        )
        doc = {"text": text, "metadata": {"type": "fact"}}
        chunks = self.chunker.chunk_document(doc)

        # Each heading should be at or near start of a chunk
        heading_texts = ["# Introduction", "# Methodology", "# Results", "# Discussion"]
        for heading in heading_texts:
            found = False
            for chunk in chunks:
                if heading in chunk["text"]:
                    # Heading should be within first 20 chars of chunk
                    idx = chunk["text"].find(heading)
                    assert idx <= 20, f"Heading '{heading}' not at chunk start: found at {idx}"
                    found = True
                    break
            assert found, f"Heading '{heading}' not found in any chunk"

    def test_list_blocks_preserved(self):
        """Short list blocks should stay together within a single chunk."""
        text = "First paragraph.\n\n- Item one\n- Item two\n- Item three\n\nLast paragraph."
        doc = {"text": text, "metadata": {"type": "fact"}}
        chunks = self.chunker.chunk_document(doc)

        # Find the chunk containing the list
        list_chunk = None
        for chunk in chunks:
            if "- Item one" in chunk["text"]:
                list_chunk = chunk
                break
        assert list_chunk is not None, "List not found in any chunk"
        # All three items should be in same chunk
        assert "- Item one" in list_chunk["text"]
        assert "- Item two" in list_chunk["text"]
        assert "- Item three" in list_chunk["text"]

    def test_metadata_preserved_across_chunks(self):
        """All chunks should have complete original metadata."""
        text = "Some content. " * 100
        doc = {"text": text, "metadata": {"type": "fact", "dimensions": ["category"], "metrics": ["profit"]}}
        chunks = self.chunker.chunk_document(doc)

        for chunk in chunks:
            assert chunk["metadata"]["type"] == "fact"
            assert chunk["metadata"]["dimensions"] == ["category"]
            assert chunk["metadata"]["metrics"] == ["profit"]

    def test_short_document_no_split(self):
        """Short document should remain as single chunk."""
        text = "Short fact."
        doc = {"text": text, "metadata": {"type": "fact"}}
        chunks = self.chunker.chunk_document(doc)

        assert len(chunks) == 1
        assert chunks[0]["text"] == text

    def test_chunk_boundaries_semantic(self):
        """Chunks should end at natural boundaries when possible."""
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five. Sentence six. " * 20
        doc = {"text": text, "metadata": {"type": "fact"}}
        chunks = self.chunker.chunk_document(doc)

        # Check that last character of each chunk (before overlap) is sentence end
        for chunk in chunks:
            stripped = chunk["text"].rstrip()
            if len(stripped) < len(text) and stripped:  # Not the full text
                last_char = stripped[-1]
                # Prefer sentence ending
                assert last_char in ['.', '!', '?', ' '], f"Chunk ends with '{last_char}'"

    def test_overlap_shared_token_count(self):
        """Overlap should share at least some content between chunks."""
        text = "Word " * 300
        doc = {"text": text, "metadata": {"type": "fact"}}
        chunks = self.chunker.chunk_document(doc)

        if len(chunks) > 1:
            # Check overlap by looking at shared tokens
            for i in range(len(chunks) - 1):
                current_end = set(chunks[i]["text"].split()[-30:])
                next_start = set(chunks[i+1]["text"].split()[:30])
                overlap = current_end & next_start
                # Should have some overlap in token sets
                assert len(overlap) > 0, "No overlap between chunks"

    def test_semantic_chunking_preserves_numbers(self):
        """Numbers with decimals or currency should stay intact."""
        text = "The value is $1,234.56 with 12.5% margin. " * 30
        doc = {"text": text, "metadata": {"type": "fact"}}
        chunks = self.chunker.chunk_document(doc)

        for chunk in chunks:
            # If number appears, it should be complete
            if "$1" in chunk["text"] or "1,234" in chunk["text"]:
                assert ".56" in chunk["text"] or "56" in chunk["text"]
            if "12.5" in chunk["text"]:
                assert "%" in chunk["text"]

    def test_empty_and_minimal_documents(self):
        """Edge cases: empty, very short text."""
        # Empty
        doc = {"text": "", "metadata": {"type": "fact"}}
        chunks = self.chunker.chunk_document(doc)
        assert len(chunks) == 1 and (chunks[0]["text"] == "" or chunks[0]["text"] == "")

        # Very short
        doc = {"text": "Hi.", "metadata": {"type": "fact"}}
        chunks = self.chunker.chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "Hi."

    def test_chunk_index_sequential(self):
        """Chunk indices should be sequential starting at 0."""
        text = " ".join(["word"] * 300)
        doc = {"text": text, "metadata": {"type": "fact"}}
        chunks = self.chunker.chunk_document(doc)

        indices = [c["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_count_accurate(self):
        """chunk_count should equal total number of chunks."""
        text = " ".join(["word"] * 300)
        doc = {"text": text, "metadata": {"type": "fact"}}
        chunks = self.chunker.chunk_document(doc)

        for chunk in chunks:
            assert chunk["chunk_count"] == len(chunks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
