import logging
from pathlib import Path
from typing import Dict, Any, List
from ingestion.pdf_loader import PDFLoader
from preprocessing.cleaner import clean_pages
from preprocessing.chunker import SemanticChunker
from embeddings.encoder import EmbeddingEncoder
from retrieval.faiss_index import FaissIndex

logger = logging.getLogger(__name__)


def ingest_document(
    pdf_path: Path, 
    encoder: EmbeddingEncoder, 
    index: FaissIndex
) -> str:
    """
    Ingest a PDF document into the vector store.

    Returns the document ID (SHA-256 hash of the file).
    """
    loader = PDFLoader(pdf_path)
    data = loader.load()

    doc_id = data["doc_id"]
    pages = clean_pages(data["pages"])
    logger.info("Loaded document %s: %d pages", doc_id, len(pages))

    # Chunk using the encoder's underlying model for semantic splitting
    chunker = SemanticChunker(encoder_model=encoder.model)
    chunks = chunker.chunk(pages, doc_id)
    logger.info("Created %d chunks for document %s", len(chunks), doc_id)

    # Embed the chunks
    embeddings = encoder.embed_chunks(chunks)

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb

    # Add to index (embeddings are stripped inside index.add)
    index.add(chunks)
    index.save()

    logger.info("Document %s indexed successfully.", doc_id)
    return doc_id
