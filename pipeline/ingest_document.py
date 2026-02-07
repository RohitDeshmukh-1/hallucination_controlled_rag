from pathlib import Path
from ingestion.pdf_loader import PDFLoader
from preprocessing.cleaner import clean_pages
from preprocessing.chunker import SemanticChunker
from embeddings.encoder import EmbeddingEncoder
from retrieval.faiss_index import FaissIndex


def ingest_document(pdf_path: Path) -> str:
    loader = PDFLoader(pdf_path)
    data = loader.load()

    doc_id = data["doc_id"]
    pages = clean_pages(data["pages"])

    chunker = SemanticChunker()
    chunks = chunker.chunk(pages, doc_id)

    encoder = EmbeddingEncoder()
    embeddings = encoder.embed_chunks(chunks)

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb

    index = FaissIndex()
    index.load_or_create()
    index.add(chunks)
    index.save()

    return doc_id
