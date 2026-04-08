"""
Single-question experiment runner for quick manual verification.

Usage:
    python -m experiments.run_single_question
"""

import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embeddings.encoder import EmbeddingEncoder
from retrieval.faiss_index import FaissIndex
from retrieval.cross_encoder import CrossEncoderReranker
from generation.prompt_builder import PromptBuilder
from generation.llm_client import LLMClient
from generation.answer_verifier import AnswerVerifier
from generation.citation_extractor import CitationExtractor


QUESTION = "What is the main contribution of the Transformer architecture?"


def main():
    encoder = EmbeddingEncoder()
    index = FaissIndex()
    index.load_or_create()

    if index.chunk_count == 0:
        print("ERROR: No documents indexed. Upload a PDF first via the API or Streamlit UI.")
        return

    query_embedding = encoder.embed_query(QUESTION)
    retrieved = index.search(query_embedding, top_k=20)

    reranker = CrossEncoderReranker()
    evidence = reranker.rerank(QUESTION, retrieved, top_n=8)

    prompt = PromptBuilder().build(QUESTION, evidence)

    llm = LLMClient()
    answer = llm.generate(prompt)

    # Verification (pass the encoder model, not the encoder wrapper)
    verifier = AnswerVerifier(encoder_model=encoder.model)
    verification = verifier.verify(answer, evidence)

    # Citation extraction
    extractor = CitationExtractor()
    citation_result = extractor.extract_and_map(answer, evidence)

    print("\n" + "=" * 70)
    print("QUESTION:", QUESTION)
    print("=" * 70)
    print("\nANSWER:\n", answer)
    print("\nVERDICT:", verification["verdict"])
    print(f"  Confidence:    {verification.get('confidence', 0):.3f}")
    print(f"  Support Ratio: {verification.get('support_ratio', 0):.3f}")
    print(f"  Citation Coverage: {citation_result['citation_coverage']:.1%}")

    if verification.get("unsupported_sentences"):
        print("\nUNSUPPORTED SENTENCES:")
        for s in verification["unsupported_sentences"]:
            print(f"   {s}")

    if citation_result["inline_citations"]:
        print("\nCITATIONS:")
        for c in citation_result["inline_citations"]:
            print(f"  [{c['evidence_id']}] doc:{c['doc_id']} pages:{c['page_start']}-{c['page_end']}")


if __name__ == "__main__":
    main()
