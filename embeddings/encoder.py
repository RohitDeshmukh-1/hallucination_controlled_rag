from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from configs.settings import settings


class EmbeddingEncoder:
    """
    High-performance embedding encoder with query caching.
    
    Ideally should be instantiated once as a singleton to avoid reloading the model.
    """

    def __init__(
        self,
        model_name: str = settings.EMBEDDING_MODEL_NAME,
        device: str = "cuda" if settings.USE_GPU else "cpu"
    ):
        self.model = SentenceTransformer(model_name, device=device)
        self._query_cache: Dict[str, np.ndarray] = {}

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query with in-memory caching.
        """
        if query not in self._query_cache:
            self._query_cache[query] = self.model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        return self._query_cache[query]

    def embed_chunks(self, chunks: List[Dict]) -> np.ndarray:
        """
        Embed document chunks (used only during ingestion).
        """
        texts = [c["text"] for c in chunks]
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
