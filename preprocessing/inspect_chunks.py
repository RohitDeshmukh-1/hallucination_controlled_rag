import os
from ingestion.pdf_loader import PDFLoader
from preprocessing.chunker import SemanticChunker


# PROJECT-ROBUST PATH HANDLING (Windows-safe)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "pdfs")

# Discover PDFs dynamically
pdf_files = [
    f for f in os.listdir(PDF_DIR)
    if f.lower().endswith(".pdf")
]

if not pdf_files:
    raise RuntimeError(f"No PDFs found in {PDF_DIR}")

PDF_PATH = os.path.join(PDF_DIR, pdf_files[0])

print(f"Using PDF: {PDF_PATH}")

# Load + chunk

loader = PDFLoader(PDF_PATH)
pdf_data = loader.load()

chunker = SemanticChunker()
chunks = chunker.chunk(pdf_data["pages"], pdf_data["doc_id"])

print(f"\nTotal chunks: {len(chunks)}\n")

for i, c in enumerate(chunks[:5], 1):
    print("=" * 80)
    print(f"CHUNK {i}")
    print(f"Pages: {c['page_start']}–{c['page_end']}")
    print(f"Tokens: {c['token_count']}")
    print(c["text"][:800])
