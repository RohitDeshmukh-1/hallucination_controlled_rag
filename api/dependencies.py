import logging
from pathlib import Path
from embeddings.encoder import EmbeddingEncoder
from retrieval.faiss_index import FaissIndex
from retrieval.cross_encoder import CrossEncoderReranker
from generation.llm_client import LLMClient
from configs.settings import settings

logger = logging.getLogger(__name__)

# Global singleton instances
_encoder: EmbeddingEncoder | None = None
_indexes: dict[str, FaissIndex] = {}
_reranker: CrossEncoderReranker | None = None
_llm_client: LLMClient | None = None


def get_encoder() -> EmbeddingEncoder:
    global _encoder
    if _encoder is None:
        logger.info("Initializing EmbeddingEncoder...")
        _encoder = EmbeddingEncoder()
    return _encoder


def _user_index_paths(user_id: str) -> tuple[Path, Path]:
    user_dir = settings.INDEX_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir / "faiss.index", user_dir / "chunks.pkl"


def get_index(user_id: str) -> FaissIndex:
    global _indexes
    if user_id not in _indexes:
        index_path, meta_path = _user_index_paths(user_id)
        logger.info("Initializing FaissIndex for user %s...", user_id)
        index = FaissIndex(index_path=index_path, meta_path=meta_path)
        index.load_or_create()
        _indexes[user_id] = index
    return _indexes[user_id]


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


def clear_index(user_id: str):
    """Clear a user's FAISS index completely."""
    index = get_index(user_id)
    index.clear()
    logger.info("Index cleared via dependency manager for user %s.", user_id)


def reload_index(user_id: str):
    """Force reload of a user's index from disk."""
    index = _indexes.get(user_id)
    if index:
        index.load()
        logger.info("Index reloaded from disk for user %s.", user_id)
