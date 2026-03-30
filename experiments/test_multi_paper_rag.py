import os
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force some env vars for testing if needed
# os.environ["LLM_API_KEY"] = "your_key"

from embeddings.encoder import EmbeddingEncoder
from retrieval.faiss_index import FaissIndex
from retrieval.cross_encoder import CrossEncoderReranker
from generation.llm_client import LLMClient
from pipeline.ingest_document import ingest_document
from pipeline.query_pipeline import run_query_pipeline

def test_multi_paper():
    encoder = EmbeddingEncoder()
    index = FaissIndex()
    index.clear() # Start fresh for test
    
    reranker = CrossEncoderReranker()
    llm = LLMClient()
    
    print("\n--- Phase 1: Ingesting Paper A ---")
    doc_a_path = Path("paper_a.pdf")
    ingest_document(doc_a_path, encoder, index)
    
    print("\n--- Phase 2: Ingesting Paper B ---")
    doc_b_path = Path("paper_b.pdf")
    ingest_document(doc_b_path, encoder, index)
    
    print(f"\n--- Index Status: {index.chunk_count} chunks, {index.document_count} documents ---")
    
    # Question about Paper A
    print("\nQ1: What is the primary threat to polar bears?")
    res1 = run_query_pipeline("What is the primary threat to polar bears?", encoder, index, reranker, llm)
    print(f"A1: {res1['answer']}")
    
    # Question about Paper B
    print("\nQ2: How much has solar energy production increased?")
    res2 = run_query_pipeline("How much has solar energy production increased?", encoder, index, reranker, llm)
    print(f"A2: {res2['answer']}")
    
    # Question about BOTH or one while having both indexed
    print("\nQ3: Compare the threat to polar bears with renewable energy trends.")
    res3 = run_query_pipeline("Compare the threat to polar bears with renewable energy trends represented in the papers.", encoder, index, reranker, llm)
    print(f"A3: {res3['answer']}")

if __name__ == "__main__":
    if not Path("paper_a.pdf").exists():
        print("Error: paper_a.pdf not found. Run scripts/generate_test_papers.py first.")
    else:
        test_multi_paper()
