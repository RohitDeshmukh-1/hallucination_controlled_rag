import faiss
import numpy as np
from typing import List, Dict
from pathlib import Path
import pickle


INDEX_DIR = Path("storage/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)


class FaissIndex:
    def __init__(self, dim: int = 384):
        self.index_path = INDEX_DIR / "faiss.index"
        self.meta_path = INDEX_DIR / "chunks.pkl"
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[Dict] = []

    def load_or_create(self):
        if self.index_path.exists():
            self.load()

    def add(self, chunks: List[Dict]):
        vectors = np.vstack([c["embedding"] for c in chunks])
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int):
        scores, indices = self.index.search(
            query_embedding.reshape(1, -1), top_k
        )

        results = []
        for idx in indices[0]:
            if idx >= 0:
                results.append(self.chunks[idx])

        return results

    def save(self):
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self):
        self.index = faiss.read_index(str(self.index_path))
        with open(self.meta_path, "rb") as f:
            self.chunks = pickle.load(f)
