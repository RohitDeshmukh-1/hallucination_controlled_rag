import logging
from retrieval.faiss_index import FaissIndex
from retrieval.cross_encoder import CrossEncoderReranker
from embeddings.encoder import EmbeddingEncoder
from generation.llm_client import LLMClient
from generation.prompt_builder import PromptBuilder
from generation.answer_verifier import AnswerVerifier
from generation.citation_extractor import CitationExtractor

logger = logging.getLogger(__name__)

# Try to import the NLI verifier; graceful fallback if deps are missing
_nli_available = False
try:
    from evaluation.nli_verifier import NLIVerifier
    _nli_available = True
except ImportError:
    logger.info("NLI verifier not available — skipping NLI-based verification.")


def run_query_pipeline(
    question: str,
    encoder: EmbeddingEncoder,
    index: FaissIndex,
    reranker: CrossEncoderReranker,
    llm: LLMClient,
    enable_nli: bool = False,
) -> dict:
    """
    Run the full query pipeline: retrieve → rerank → generate → verify → cite.

    Parameters
    ----------
    enable_nli : bool
        If True and NLI model is available, runs dual-layer verification
        (cosine similarity + NLI entailment).
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

    # Verify answer against evidence (cosine similarity)
    verifier = AnswerVerifier(encoder_model=encoder.model)
    verification = verifier.verify(answer, evidence)

    logger.info(
        "Verification verdict: %s (confidence=%.3f, support_ratio=%.3f)",
        verification["verdict"],
        verification.get("confidence", 0),
        verification.get("support_ratio", 0),
    )

    # Optional NLI verification (dual-layer)
    nli_result = None
    if enable_nli and _nli_available:
        try:
            nli_verifier = NLIVerifier()
            nli_result = nli_verifier.verify(answer, evidence)
            logger.info(
                "NLI verdict: %s (entailment_ratio=%.3f)",
                nli_result["verdict"],
                nli_result.get("entailment_ratio", 0),
            )
        except Exception as e:
            logger.warning("NLI verification failed: %s", e)

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

    # Base response
    response = {
        "confidence": verification.get("confidence", 0.0),
        "support_ratio": verification.get("support_ratio", 0.0),
        "citation_coverage": citation_result["citation_coverage"],
        "citations": citations,
    }

    # Add NLI results if available
    if nli_result:
        response["nli_verdict"] = nli_result["verdict"]
        response["nli_entailment_ratio"] = nli_result["entailment_ratio"]

    if verification["verdict"] == "partially_supported":
        caveat = (
            "\n\n*Note: Some aspects of this answer may have limited "
            "direct support in the source documents.*"
        )
        response.update({
            "answer": answer + caveat,
            "verdict": "partially_supported",
            "uncited_sentences": citation_result["uncited_sentences"],
        })
        return response

    response.update({
        "answer": answer,
        "verdict": "supported",
    })
    return response
