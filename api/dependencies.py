import logging
from embeddings.encoder import EmbeddingEncoder
from retrieval.faiss_index import FaissIndex
from retrieval.cross_encoder import CrossEncoderReranker
from generation.llm_client import LLMClient
from configs.settings import settings

logger = logging.getLogger(__name__)

# Global singleton instances
_encoder: EmbeddingEncoder | None = None
_index: FaissIndex | None = None
_reranker: CrossEncoderReranker | None = None
_llm_client: LLMClient | None = None


def get_encoder() -> EmbeddingEncoder:
    global _encoder
    if _encoder is None:
        logger.info("Initializing EmbeddingEncoder...")
        _encoder = EmbeddingEncoder()
    return _encoder


def get_index() -> FaissIndex:
    global _index
    if _index is None:
        logger.info("Initializing FaissIndex...")
        _index = FaissIndex()
        _index.load_or_create()
    return _index


def get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        logger.info("Initializing CrossEncoderReranker...")
        _reranker = CrossEncoderReranker()
    return _reranker


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        logger.info("Initializing LLMClient...")
        _llm_client = LLMClient()
    return _llm_client


def clear_index():
    """Clear the FAISS index completely (removes all documents)."""
    global _index
    index = get_index()
    index.clear()
    logger.info("Index cleared via dependency manager.")


def reload_index():
    """Force reload of the index from disk (useful after ingestion)."""
    global _index
    if _index:
        _index.load()
        logger.info("Index reloaded from disk.")
