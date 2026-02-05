import os
import json
from typing import List, Dict, Tuple

import numpy as np
import faiss


class FaissRetriever:
    """
    Production-grade FAISS-based retriever for semantic chunks.

    Responsibilities:
    - Build FAISS index from embeddings
    - Persist and reload index
    - Perform top-k similarity search
    - Return aligned metadata for grounding

    Design principles:
    - Deterministic indexing
    - No hidden state
    - Disk persistence for reproducibility
    """

    def __init__(
        self,
        index_dir: str = "data/processed/faiss",
        index_name: str = "chunks.index",
        metadata_name: str = "chunks_metadata.json",
        use_cosine_similarity: bool = True,
    ):
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)

        self.index_path = os.path.join(index_dir, index_name)
        self.metadata_path = os.path.join(index_dir, metadata_name)

        self.use_cosine_similarity = use_cosine_similarity

        self.index = None
        self.metadata: List[Dict] = []

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def build(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict],
    ) -> None:
        """
        Build and persist FAISS index from embeddings.
        """
        dim = embeddings.shape[1]

        if self.use_cosine_similarity:
            # embeddings assumed to be L2-normalized
            index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexFlatL2(dim)

        index.add(embeddings)

        faiss.write_index(index, self.index_path)

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        self.index = index
        self.metadata = metadata

    def load(self) -> None:
        """
        Load FAISS index and metadata from disk.
        """
        self.index = faiss.read_index(self.index_path)

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Retrieve top-k most similar chunks for a query embedding.
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for rank, idx in enumerate(indices[0]):
            if idx < 0:
                continue

            chunk_meta = self.metadata[idx].copy()
            chunk_meta["score"] = float(scores[0][rank])
            results.append(chunk_meta)

        return results
