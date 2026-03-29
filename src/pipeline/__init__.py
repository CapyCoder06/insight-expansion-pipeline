"""
Main pipeline orchestrator.
"""

from typing import List, Dict, Any, Optional
import json
from .document_generator import generate_from_insights, DocumentGenerator, enhance_for_retrieval
from .validator import DocumentValidator
from .chunker import DocumentChunker
from .enrichment import DocumentEnricher, expand_document, select_top_variations
from .config import Config, get_config


class DocumentPipeline:
    """
    End-to-end pipeline for converting insights to chunked documents.

    Steps:
    1. Generate documents from insights (template mapping + rendering)
    2. Validate documents (quality checks)
    3. Chunk documents (prepare for embedding)
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None,
                 enrich: bool = None, enrich_variations: int = None,
                 optimize_retrieval: bool = None, config: Config = None):
        """
        Initialize pipeline.

        Args:
            chunk_size: Target chunk size in words (overrides config if specified)
            chunk_overlap: Overlap between chunks in words (overrides config if specified)
            enrich: Whether to enable enrichment (overrides config if specified)
            enrich_variations: Number of top variations to keep per document (overrides config if specified)
            optimize_retrieval: Whether to enable retrieval optimization (overrides config if specified)
            config: Configuration object (loads from config.yaml if None)
        """
        self.generator = DocumentGenerator()

        # Load config if not provided
        if config is None:
            config = get_config()

        self.config = config
        self.validator = DocumentValidator(config=config)

        # Get values from config with optional parameter overrides
        chunking_config = config.get('chunking', {})
        self.chunk_size = chunk_size if chunk_size is not None else chunking_config.get('chunk_size', 200)
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else chunking_config.get('overlap', 50)
        use_nltk = chunking_config.get('use_nltk', True)

        enrichment_config = config.get('enrichment', {})
        self.enrich = enrich if enrich is not None else enrichment_config.get('enabled', False)
        self.enrich_variations = enrich_variations if enrich_variations is not None else enrichment_config.get('variations_to_keep', 3)

        retrieval_opt_config = config.get('retrieval_optimization', {})
        self.optimize_retrieval = optimize_retrieval if optimize_retrieval is not None else retrieval_opt_config.get('enabled', False)

        # Initialize components
        self.chunker = DocumentChunker(chunk_size=self.chunk_size, overlap=self.chunk_overlap, use_nltk=use_nltk)

        # Enricher generates a few extra variations for selection
        variations_to_generate = enrichment_config.get('variations_to_generate', 8)
        self.enricher = DocumentEnricher(num_variations=variations_to_generate, config=config)

    def run(self, insights: List[Dict[str, Any]],
            validate: bool = True,
            chunk: bool = True) -> Dict[str, Any]:
        """
        Run the full pipeline.

        Args:
            insights: List of insight dictionaries
            validate: Whether to run validation
            chunk: Whether to chunk documents

        Returns:
            Dictionary with:
                - documents: List of generated documents (enriched if enrich=True)
                - validation: Validation summary (if validate=True)
                - chunks: List of chunks (if chunk=True)
        """
        # Step 1: Generate documents
        documents = generate_from_insights(insights)

        # Step 1.5: Enrich (generate variations) if enabled
        if self.enrich:
            enriched_docs = []
            for doc in documents:
                variations = self.enricher.expand_document(doc)
                top_vars = select_top_variations(variations, top_k=self.enrich_variations)
                enriched_docs.extend(top_vars)
            documents = enriched_docs

        # Step 1.75: Apply retrieval optimization metadata (if enabled)
        if self.optimize_retrieval:
            from .document_generator import extract_retrieval_metadata
            optimized_docs = []
            for doc in documents:
                retrieval_meta = extract_retrieval_metadata(doc["text"])
                # Create new metadata with retrieval fields added
                new_metadata = doc["metadata"].copy()
                if retrieval_meta.get("queries"):
                    new_metadata["queries"] = retrieval_meta["queries"]
                if retrieval_meta.get("synonyms"):
                    new_metadata["synonyms"] = retrieval_meta["synonyms"]
                optimized_docs.append({
                    "text": doc["text"],  # Keep original text UNCHANGED
                    "metadata": new_metadata
                })
            documents = optimized_docs

        result = {"documents": documents}

        # Step 2 (or 3 if no enrichment): Validate
        if validate:
            validation = self.validator.validate_batch(documents)
            result["validation"] = validation

        # Step 3 (or 4): Chunk
        if chunk:
            chunks = self.chunker.chunk_batch(documents)
            result["chunks"] = chunks
            result["chunk_count"] = len(chunks)

        return result

    def generate(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate documents from insights (skips validation and chunking)."""
        return generate_from_insights(insights)

    def validate(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate documents."""
        return self.validator.validate_batch(documents)

    def chunk(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk documents."""
        return self.chunker.chunk_batch(documents)


def save_documents(documents: List[Dict[str, Any]], path: str):
    """Save documents to JSON file."""
    with open(path, 'w') as f:
        json.dump(documents, f, indent=2)


def save_chunks(chunks: List[Dict[str, Any]], path: str):
    """Save chunks to JSON file."""
    with open(path, 'w') as f:
        json.dump(chunks, f, indent=2)
