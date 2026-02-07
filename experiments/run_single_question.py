from embeddings.encoder import EmbeddingEncoder
from retrieval.faiss_index import FaissIndex
from retrieval.cross_encoder import CrossEncoderReranker
from generation.prompt_builder import PromptBuilder
from generation.llm_client import LLMClient
from generation.answer_verifier import AnswerVerifier


QUESTION = "What is the main contribution of the Transformer architecture?"


def main():
    encoder = EmbeddingEncoder()
    index = FaissIndex()        # already populated via uploads

    query_embedding = encoder.embed_query(QUESTION)

    retrieved = index.search(query_embedding, top_k=12)

    reranker = CrossEncoderReranker()
    evidence = reranker.rerank(QUESTION, retrieved)

    prompt = PromptBuilder().build(QUESTION, evidence)

    llm = LLMClient()
    answer = llm.generate(prompt)

    verifier = AnswerVerifier()
    verdict = verifier.verify(answer, evidence)

    print("\nANSWER:\n", answer)
    print("\nVERDICT:\n", verdict)


if __name__ == "__main__":
    main()
