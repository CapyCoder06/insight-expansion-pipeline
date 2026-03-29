#!/usr/bin/env python
"""
Demo script for the Document Generation Pipeline.

Converts structured insights into standardized documents for RAG.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, 'src')

from pipeline import DocumentPipeline, save_documents, save_chunks
from pipeline.config import load_config


def main():
    """Run the demo pipeline."""
    print("=" * 60)
    print("Superstore Document Pipeline Demo")
    print("=" * 60)

    # Load configuration
    try:
        config = load_config()
        print(f"Loaded configuration from: {config.config_path or 'defaults'}")
        chunk_size = config.get('chunking.chunk_size')
        chunk_overlap = config.get('chunking.overlap')
        enrich = config.get('enrichment.enabled')
        enrich_variations = config.get('enrichment.variations_to_keep')
        print(f"  Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
        print(f"  Enrichment: {enrich}, Variations to keep: {enrich_variations}")
    except Exception as e:
        print(f"Warning: Could not load config: {e}. Using hardcoded demo values.")
        chunk_size = 100
        chunk_overlap = 20
        enrich = True
        enrich_variations = 3

    # Load sample insights
    insights_path = Path("insights_sample.json")
    if not insights_path.exists():
        print(f"Error: {insights_path} not found!")
        return

    with open(insights_path) as f:
        insights = json.load(f)

    print(f"Loaded {len(insights)} insights")

    # Initialize pipeline with config values (or overrides for demo)
    pipeline = DocumentPipeline(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        enrich=enrich,
        enrich_variations=enrich_variations
    )

    # Run pipeline
    print("\n[1/3] Generating documents...")
    result = pipeline.run(insights, validate=True, chunk=True)

    documents = result["documents"]
    chunks = result["chunks"]
    validation = result["validation"]

    print(f"Generated {len(documents)} documents")
    print(f"Created {len(chunks)} chunks")

    # Print validation summary
    print(f"\n[2/3] Validation Results:")
    print(f"  Valid: {validation['valid']}/{validation['total']}")
    print(f"  Invalid: {validation['invalid']}/{validation['total']}")

    if validation['invalid'] > 0:
        print("\n  Issues:")
        for i, doc_validation in enumerate(validation['documents']):
            if not doc_validation['valid']:
                print(f"    Doc {i}: {doc_validation['issues']}")

    # Show sample documents
    print(f"\n[3/3] Sample Documents:")
    for i, doc in enumerate(documents[:3]):
        print(f"\n--- Document {i} (type: {doc['metadata']['type']}) ---")
        print(doc['text'][:200])
        if len(doc['text']) > 200:
            print("...")
        print(f"Metadata: {doc['metadata']}")

    # Show chunk sample
    print(f"\n--- Sample Chunk ---")
    print(f"Chunk {chunks[0]['chunk_index'] + 1}/{chunks[0]['chunk_count']}")
    print(chunks[0]['text'][:150])
    if len(chunks[0]['text']) > 150:
        print("...")

    # Save outputs
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    save_documents(documents, output_dir / "documents.json")
    save_chunks(chunks, output_dir / "chunks.json")

    print(f"\nSaved:")
    print(f"  - Documents: {output_dir / 'documents.json'}")
    print(f"  - Chunks: {output_dir / 'chunks.json'}")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
