from embeddings.encoder import EmbeddingEncoder
from retrieval.faiss_index import FaissIndex
from retrieval.cross_encoder import CrossEncoderReranker
from generation.prompt_builder import PromptBuilder
from generation.llm_client import LLMClient
from generation.answer_verifier import AnswerVerifier
from generation.citation_extractor import CitationExtractor


def run_query_pipeline(question: str):
    encoder = EmbeddingEncoder()
    index = FaissIndex()
    index.load()

    # ---------- HARD GUARDS ----------
    if not index.chunks:
        return {
            "answer": "No documents have been uploaded yet.",
            "verdict": "refused",
            "citations": [],
        }

    query_embedding = encoder.embed_query(question)
    retrieved = index.search(query_embedding, top_k=20)

    if not retrieved:
        return {
            "answer": "I cannot answer based on the provided documents.",
            "verdict": "refused",
            "citations": [],
        }

    reranker = CrossEncoderReranker()
    evidence = reranker.rerank(question, retrieved, top_n=8)

    if not evidence:
        return {
            "answer": "I cannot answer based on the provided documents.",
            "verdict": "refused",
            "citations": [],
        }

    prompt = PromptBuilder().build(question, evidence)

    llm = LLMClient()
    try:
        answer = llm.generate(prompt)
    except Exception:
        return {
            "answer": "The language model is temporarily unavailable.",
            "verdict": "refused",
            "citations": [],
        }

    # Extract and map citations from the answer
    citation_extractor = CitationExtractor()
    citation_result = citation_extractor.extract_and_map(answer, evidence)

    verifier = AnswerVerifier()
    verification = verifier.verify(answer, evidence)

    # Build detailed citations from extraction (only cited evidence)
    citations = [
        {
            "evidence_id": c["evidence_id"],
            "doc_id": c["doc_id"],
            "pages": f"{c['page_start']}-{c['page_end']}",
            "text_preview": c["text_preview"],
        }
        for c in citation_result["inline_citations"]
    ]

    if verification["verdict"] == "unsupported":
        return {
            "answer": "I cannot answer based on the provided documents.",
            "verdict": "refused",
            "unsupported_sentences": verification["unsupported_sentences"],
            "confidence": verification.get("confidence", 0.0),
            "citation_coverage": 0.0,
        }

    if verification["verdict"] == "partially_supported":
        caveat = (
            "\n\n*Note: Some aspects of this answer may have limited "
            "direct support in the source documents.*"
        )
        return {
            "answer": answer + caveat,
            "verdict": "partially_supported",
            "citations": citations,
            "confidence": verification.get("confidence", 0.0),
            "support_ratio": verification.get("support_ratio", 0.0),
            "citation_coverage": citation_result["citation_coverage"],
            "uncited_sentences": citation_result["uncited_sentences"],
        }

    return {
        "answer": answer,
        "verdict": "supported",
        "citations": citations,
        "confidence": verification.get("confidence", 1.0),
        "citation_coverage": citation_result["citation_coverage"],
    }
