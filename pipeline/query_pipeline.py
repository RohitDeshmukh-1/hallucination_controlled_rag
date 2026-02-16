import logging
from retrieval.faiss_index import FaissIndex
from retrieval.cross_encoder import CrossEncoderReranker
from embeddings.encoder import EmbeddingEncoder
from generation.llm_client import LLMClient
from generation.prompt_builder import PromptBuilder
from generation.answer_verifier import AnswerVerifier
from generation.citation_extractor import CitationExtractor

logger = logging.getLogger(__name__)


def run_query_pipeline(
    question: str,
    encoder: EmbeddingEncoder,
    index: FaissIndex,
    reranker: CrossEncoderReranker,
    llm: LLMClient
) -> dict:
    """
    Run the full query pipeline: retrieve → rerank → generate → verify → cite.
    """

    # Hard guards
    if not index.chunks:
        return {
            "answer": "No documents have been uploaded yet.",
            "verdict": "refused",
            "citations": [],
        }

    query_embedding = encoder.embed_query(question)
    retrieved = index.search(query_embedding, top_k=20)
    logger.info("Retrieved %d chunks for query: '%s'", len(retrieved), question[:80])

    if not retrieved:
        return {
            "answer": "I cannot answer based on the provided documents.",
            "verdict": "refused",
            "citations": [],
        }

    evidence = reranker.rerank(question, retrieved, top_n=8)
    logger.info(
        "Reranked to %d evidence chunks (top score: %.3f)",
        len(evidence),
        evidence[0].get("cross_score", 0) if evidence else 0,
    )

    if not evidence:
        return {
            "answer": "I cannot answer based on the provided documents.",
            "verdict": "refused",
            "citations": [],
        }

    prompt = PromptBuilder().build(question, evidence)

    try:
        answer = llm.generate(prompt)
        logger.info("LLM generated answer (%d chars)", len(answer))
    except Exception as e:
        logger.error("Pipeline LLM Error: %s", e)
        return {
            "answer": "The language model is temporarily unavailable.",
            "verdict": "refused",
            "citations": [],
        }

    # Extract and map citations from the answer
    citation_extractor = CitationExtractor()
    citation_result = citation_extractor.extract_and_map(answer, evidence)

    # Verify answer against evidence
    verifier = AnswerVerifier(encoder_model=encoder.model)
    verification = verifier.verify(answer, evidence)

    logger.info(
        "Verification verdict: %s (confidence=%.3f, support_ratio=%.3f)",
        verification["verdict"],
        verification.get("confidence", 0),
        verification.get("support_ratio", 0),
    )

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
