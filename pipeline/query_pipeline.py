from embeddings.encoder import EmbeddingEncoder
from retrieval.faiss_index import FaissIndex
from retrieval.cross_encoder import CrossEncoderReranker
from generation.prompt_builder import PromptBuilder
from generation.llm_client import LLMClient
from generation.answer_verifier import AnswerVerifier


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

    verifier = AnswerVerifier()
    verification = verifier.verify(answer, evidence)

    if verification["verdict"] == "unsupported":
        return {
            "answer": "I cannot answer based on the provided documents.",
            "verdict": "refused",
            "unsupported_sentences": verification["unsupported_sentences"],
        }

    return {
        "answer": answer,
        "verdict": "supported",
        "citations": [
            {
                "doc_id": c["doc_id"],
                "pages": f"{c['page_start']}-{c['page_end']}",
            }
            for c in evidence
        ],
    }
