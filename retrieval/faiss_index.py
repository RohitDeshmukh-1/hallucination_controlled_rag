import faiss
import numpy as np
import pickle
import logging
from typing import List, Dict, Set
from pathlib import Path
from configs.settings import settings

logger = logging.getLogger(__name__)


class FaissIndex:
    """
    In-memory vector store wrapper around FAISS.
    Supports document-level tracking and index lifecycle management.
    """

    def __init__(self, dim: int = settings.EMBEDDING_DIM):
        self.index_path = settings.FAISS_INDEX_PATH
        self.meta_path = settings.FAISS_META_PATH
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[Dict] = []

        # Ensure directory exists
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def document_count(self) -> int:
        return len(self.get_doc_ids())

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def load_or_create(self):
        """Load index if it exists, otherwise start empty."""
        if self.index_path.exists() and self.meta_path.exists():
            self.load()
            logger.info(
                "Loaded existing index: %d chunks, %d documents",
                self.chunk_count,
                self.document_count,
            )
        else:
            logger.info("No existing index found, starting fresh.")

    def add(self, chunks: List[Dict]):
        """Add chunks with embeddings to the FAISS index."""
        if not chunks:
            return

        vectors = np.vstack([c["embedding"] for c in chunks]).astype("float32")
        self.index.add(vectors)

        # Store chunks WITHOUT the embedding to save memory/disk
        for c in chunks:
            meta = {k: v for k, v in c.items() if k != "embedding"}
            self.chunks.append(meta)

        logger.info("Added %d chunks to index (total: %d)", len(chunks), self.chunk_count)

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """Search the index for the top-k most similar chunks."""
        if self.index.ntotal == 0:
            return []

        scores, indices = self.index.search(
            query_embedding.reshape(1, -1).astype("float32"), top_k
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                result = dict(self.chunks[idx])
                result["similarity_score"] = float(score)
                results.append(result)

        return results

    def clear(self):
        """Reset the index to empty and delete persisted files."""
        self.index = faiss.IndexFlatIP(self.dim)
        self.chunks = []

        if self.index_path.exists():
            self.index_path.unlink()
        if self.meta_path.exists():
            self.meta_path.unlink()

        logger.info("Index cleared.")

    def get_doc_ids(self) -> Set[str]:
        """Return the set of unique document IDs in the index."""
        return {c.get("doc_id", "unknown") for c in self.chunks}

    def save(self):
        """Persist the FAISS index and chunk metadata to disk."""
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.chunks, f)
        logger.info("Index saved: %d chunks.", self.chunk_count)

    def load(self):
        """Load the FAISS index and chunk metadata from disk."""
        self.index = faiss.read_index(str(self.index_path))
        with open(self.meta_path, "rb") as f:
            self.chunks = pickle.load(f)
