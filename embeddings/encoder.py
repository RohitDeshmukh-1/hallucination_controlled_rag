from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingEncoder:
    """
    High-performance embedding encoder with query caching.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)
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
