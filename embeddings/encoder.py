import os
import json
import hashlib
from typing import List, Dict, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingEncoder:
    """
    Production-grade embedding encoder for semantic chunks.

    Responsibilities:
    - Embed chunk texts deterministically
    - Cache embeddings on disk
    - Return FAISS-ready vectors with metadata

    Design principles:
    - No recomputation if cache exists
    - Chunk-level granularity
    - Reproducible across runs
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: str = "data/processed/embeddings",
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.normalize = normalize

        self.model = SentenceTransformer(model_name)

        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def encode_chunks(
        self, chunks: List[Dict]
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Encode chunks into embeddings.

        Returns:
            embeddings: np.ndarray of shape (N, D)
            metadata: list of chunk metadata aligned with embeddings
        """
        cache_path = self._cache_path(chunks)

        if os.path.exists(cache_path):
            return self._load_cache(cache_path)

        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        if self.normalize:
            embeddings = self._normalize(embeddings)

        metadata = [
            {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "token_count": chunk["token_count"],
            }
            for chunk in chunks
        ]

        self._save_cache(cache_path, embeddings, metadata)

        return embeddings, metadata

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------
    def _cache_path(self, chunks: List[Dict]) -> str:
        """
        Generate deterministic cache path based on:
        - model name
        - chunk ids
        """
        hasher = hashlib.sha256()
        hasher.update(self.model_name.encode("utf-8"))

        for chunk in chunks:
            hasher.update(chunk["chunk_id"].encode("utf-8"))

        cache_key = hasher.hexdigest()[:32]

        return os.path.join(self.cache_dir, f"{cache_key}.npz")

    def _save_cache(
        self,
        path: str,
        embeddings: np.ndarray,
        metadata: List[Dict],
    ) -> None:
        np.savez_compressed(
            path,
            embeddings=embeddings,
            metadata=json.dumps(metadata),
        )

    def _load_cache(
        self, path: str
    ) -> Tuple[np.ndarray, List[Dict]]:
        data = np.load(path, allow_pickle=False)
        embeddings = data["embeddings"]
        metadata = json.loads(data["metadata"].item())
        return embeddings, metadata

    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.clip(norms, 1e-10, None)
