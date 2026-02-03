from ingestion.pdf_loader import PDFLoader
from preprocessing.chunker import SemanticChunker

# -------------------------
# RAW STRING PATHS (Windows)
# -------------------------
PROJECT_ROOT = r"C:\ML\hallucination_controlled_rag"
SAMPLE_PDF = r"C:\ML\hallucination_controlled_rag\data\raw\pdfs\sample.pdf"

def test_semantic_chunker():
    # Step 1: Load PDF
    loader = PDFLoader(SAMPLE_PDF)
    pdf_data = loader.load()

    print(f"Loaded PDF with {pdf_data['num_pages']} pages")

    # Step 2: Chunk semantically
    chunker = SemanticChunker()
    chunks = chunker.chunk(
        pages=pdf_data["pages"],
        doc_id=pdf_data["doc_id"]
    )

    print(f"\nGenerated {len(chunks)} chunks")

    # Step 3: Inspect chunks
    for i, chunk in enumerate(chunks[:3], start=1):
        print("\n" + "=" * 60)
        print(f"CHUNK {i}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Pages: {chunk['page_start']}–{chunk['page_end']}")
        print(f"Token count: {chunk['token_count']}")
        print("-" * 60)
        print(chunk["text"][:800])


if __name__ == "__main__":
    test_semantic_chunker()
