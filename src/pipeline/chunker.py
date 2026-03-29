"""
Chunking for document processing - semantic-aware version.
"""

from typing import List, Dict, Any
import re


class DocumentChunker:
    """Splits documents into chunks for embedding, preserving semantic boundaries."""

    def __init__(self, chunk_size: int = 200, overlap: int = 50, use_nltk: bool = True):
        """
        Initialize chunker.

        Args:
            chunk_size: Target chunk size in tokens (words)
            overlap: Number of words to overlap between chunks (set to 0 to disable)
            use_nltk: Try to use NLTK for sentence splitting if True, else use regex
        """
        self.chunk_size = chunk_size
        self.overlap = max(0, overlap)
        self.use_nltk = use_nltk

    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a document using semantic boundaries.

        Strategy:
        1. Split into sentences (maintains meaning units)
        2. Group sentences into chunks around target size
        3. Prefer splitting at sentence boundaries, paragraph breaks, headings
        4. Preserve numbers/entities by using sentence boundaries

        Args:
            document: Document with 'text' and 'metadata'

        Returns:
            List of chunk dictionaries with text, metadata, and chunk_index
        """
        text = document.get("text", "")
        metadata = document.get("metadata", {})

        # Quick return for short documents
        words = text.split()
        if len(words) <= self.chunk_size:
            return [{
                "text": text,
                "metadata": metadata,
                "chunk_index": 0,
                "chunk_count": 1
            }]

        # Split into sentences
        sentences = self._split_into_sentences(text)

        if len(sentences) <= 1:
            # Fallback to basic splitting if can't detect sentences
            return self._basic_chunk(text, metadata)

        # Group sentences into semantic chunks with structure awareness
        chunks = self._group_sentences_into_chunks(sentences, metadata)

        # Apply overlap after grouping (post-hoc overlap)
        if self.overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap_fixed(chunks, self.overlap)

        # Update chunk_count
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk["chunk_count"] = total_chunks

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using hybrid NLTK/regex approach."""
        # Try NLTK first if available or requested
        if self.use_nltk:
            try:
                import nltk
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt', quiet=True)
                sentences = nltk.tokenize.sent_tokenize(text)
                # Filter out empty strings
                return [s.strip() for s in sentences if s.strip()]
            except (ImportError, Exception):
                pass  # Fall to regex

        # Improved regex fallback
        return self._split_sentences_regex(text)

    def _split_sentences_regex(self, text: str) -> List[str]:
        """Split sentences using improved regex that handles abbreviations and decimals."""
        # Normalize whitespace first
        text = text.strip()
        if not text:
            return []

        # Split on paragraph breaks first, then process each paragraph
        paragraphs = text.split('\n\n')
        all_sentences = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Split on . ! ? followed by whitespace
            raw_sentences = re.split(r'(?<=[.!?])\s+', para)

            # Merge fragments that shouldn't have been split
            merged = []
            i = 0
            while i < len(raw_sentences):
                curr = raw_sentences[i]
                if i + 1 < len(raw_sentences):
                    next_sent = raw_sentences[i + 1]
                    # Merge if:
                    # - current ends with a decimal number and next starts with digit (e.g., "3.14" split as "3." and "14")
                    # - current ends with common abbreviation and next starts lowercase
                    # - current is very short (<5 words) and ends with period and next starts lowercase (continuation)
                    if (re.search(r'\d+\.\d*$', curr) and re.match(r'^\d', next_sent)):
                        merged.append(curr + ' ' + next_sent)
                        i += 2
                        continue
                    if (re.search(r'(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|etc)\.$', curr) and not re.match(r'^[A-Z]', next_sent)):
                        merged.append(curr + ' ' + next_sent)
                        i += 2
                        continue
                    if (len(curr.split()) < 5 and curr.endswith('.') and next_sent[0].islower()):
                        merged.append(curr + ' ' + next_sent)
                        i += 2
                        continue
                merged.append(curr)
                i += 1

            all_sentences.extend(merged)

        return all_sentences

    def _group_sentences_into_chunks(self, sentences: List[str], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Group sentences into chunks respecting structure and balancing size."""
        chunks = []
        current_sentences = []
        current_size = 0
        chunk_index = 0

        # Local helpers for structural boundaries
        def is_heading_start(s: str) -> bool:
            return s.lstrip().startswith('#')

        def is_list_item(s: str) -> bool:
            return re.match(r'^\s*([-*+]|\d+\.)\s+', s) is not None

        def list_block_key(s: str) -> tuple:
            if not is_list_item(s):
                return None
            m = re.match(r'^(\s*)([-*+]|\d+\.)\s+', s)
            indent = len(m.group(1))
            bullet = m.group(2)
            return (indent, bullet)

        for sentence in sentences:
            sentence_words = len(sentence.split())
            should_break = False

            # Hard break: heading always starts new chunk
            if is_heading_start(sentence):
                should_break = True
            # Hard break: exceed hard limit (1.5 × target)
            elif current_size > 0 and (current_size + sentence_words) > (self.chunk_size * 1.5):
                should_break = True
            # Soft break: exceed target and encounter list block boundary
            elif current_size > 0 and (current_size + sentence_words) > self.chunk_size:
                if is_list_item(sentence):
                    if current_sentences:
                        last = current_sentences[-1]
                        if is_list_item(last):
                            curr_key = list_block_key(last)
                            new_key = list_block_key(sentence)
                            if new_key != curr_key:
                                should_break = True
                        else:
                            should_break = True

            if should_break and current_sentences:
                chunk_text = " ".join(current_sentences)
                chunks.append({
                    "text": chunk_text,
                    "metadata": metadata.copy(),
                    "chunk_index": chunk_index,
                    "chunk_count": 0
                })
                chunk_index += 1
                # Reset without overlap (overlap applied later)
                current_sentences = []
                current_size = 0

            current_sentences.append(sentence)
            current_size += sentence_words

        # Add final chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append({
                "text": chunk_text,
                "metadata": metadata.copy(),
                "chunk_index": chunk_index,
                "chunk_count": 0
            })

        # Post-process: merge tiny chunks and split huge ones
        chunks = self._balance_chunks(chunks)

        return chunks

    def _balance_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Merge tiny chunks and split huge ones to improve size uniformity."""
        if not chunks:
            return chunks

        target = self.chunk_size
        balanced = []
        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            size = len(chunk["text"].split())
            # Very small chunk: try to merge with next
            if size < target * 0.4 and i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                # Do not merge across heading boundaries; heading must start new chunk
                if next_chunk["text"].lstrip().startswith('#'):
                    balanced.append(chunk)
                    i += 1
                    continue
                next_size = len(next_chunk["text"].split())
                combined_size = size + next_size
                if combined_size <= target * 1.2:
                    merged_text = chunk["text"] + " " + next_chunk["text"]
                    merged_metadata = chunk["metadata"].copy()
                    new_chunk = {
                        "text": merged_text,
                        "metadata": merged_metadata,
                        "chunk_index": 0,
                        "chunk_count": 0
                    }
                    balanced.append(new_chunk)
                    i += 2
                    continue
            # Very large chunk: split at sentence boundaries
            if size > target * 1.5:
                sub_chunks = self._resplit_large_chunk(chunk, target)
                balanced.extend(sub_chunks)
                i += 1
                continue
            balanced.append(chunk)
            i += 1

        # Re-index chunks
        for idx, ch in enumerate(balanced):
            ch["chunk_index"] = idx
            ch["chunk_count"] = len(balanced)

        return balanced

    def _resplit_large_chunk(self, chunk: Dict, target: int) -> List[Dict]:
        """Split a too-large chunk into smaller ones using sentence boundaries."""
        text = chunk["text"]
        sentences = self._split_into_sentences(text)
        if len(sentences) <= 1:
            return [chunk]

        sub_chunks = []
        current = []
        current_size = 0
        for sent in sentences:
            sent_size = len(sent.split())
            if current_size > 0 and (current_size + sent_size) > target:
                sub_chunks.append({
                    "text": " ".join(current),
                    "metadata": chunk["metadata"].copy(),
                    "chunk_index": 0,
                    "chunk_count": 0
                })
                current = []
                current_size = 0
            current.append(sent)
            current_size += sent_size
        if current:
            sub_chunks.append({
                "text": " ".join(current),
                "metadata": chunk["metadata"].copy(),
                "chunk_index": 0,
                "chunk_count": 0
            })
        return sub_chunks

    def _apply_overlap_fixed(self, chunks: List[Dict], overlap_words: int) -> List[Dict]:
        """Apply fixed word-based overlap between consecutive chunks, respecting heading boundaries."""
        if len(chunks) < 2 or overlap_words <= 0:
            return chunks

        result = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                result.append(chunk)
                continue

            # Skip overlap if chunk starts with a heading to preserve hard boundary
            if chunk["text"].lstrip().startswith('#'):
                result.append(chunk)
                continue

            prev = result[-1]
            prev_words = prev["text"].split()
            overlap_count = min(overlap_words, len(prev_words))
            overlap_text = " ".join(prev_words[-overlap_count:])

            current_text = chunk["text"]
            new_text = overlap_text + " " + current_text

            new_chunk = {
                "text": new_text.strip(),
                "metadata": chunk["metadata"].copy(),
                "chunk_index": chunk["chunk_index"],
                "chunk_count": chunk["chunk_count"]
            }
            result[-1] = prev
            result.append(new_chunk)

        return result

    def _basic_chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fallback basic word-based chunking when sentence splitting fails."""
        words = text.split()
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end])

            chunks.append({
                "text": chunk_text,
                "metadata": metadata.copy(),
                "chunk_index": chunk_index,
                "chunk_count": 0
            })

            chunk_index += 1
            # Stop if we've reached the end to avoid tiny trailing chunks
            if end >= len(words):
                break
            start = max(start + 1, end - self.overlap) if self.overlap > 0 else end

        # Update chunk_count
        for chunk in chunks:
            chunk["chunk_count"] = len(chunks)

        return chunks

    def chunk_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents.

        Args:
            documents: List of document dictionaries

        Returns:
            Flat list of all chunks with document_id added to metadata
        """
        all_chunks = []

        for doc_idx, doc in enumerate(documents):
            chunks = self.chunk_document(doc)
            for chunk in chunks:
                # Add document reference
                chunk["metadata"]["_doc_id"] = doc_idx
                all_chunks.append(chunk)

        return all_chunks
