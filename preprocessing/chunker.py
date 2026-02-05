import uuid
import re
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticChunker:
    """
    Section-agnostic, evidence-centric semantic chunker for academic documents.
    Designed for RAG with hallucination control.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_tokens: int = 450,
        min_tokens: int = 200,
        overlap_tokens: int = 50,
        similarity_threshold: float = 0.75,
        window_size: int = 3,
    ):
        self.model = SentenceTransformer(model_name)
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def chunk(self, pages: List[Dict], doc_id: str) -> List[Dict]:
        sentences, sentence_pages = self._split_into_sentences(pages)

        # Embed + normalize sentences
        embeddings = self.model.encode(sentences, convert_to_numpy=True)
        embeddings = self._normalize_embeddings(embeddings)

        chunks = []

        current_sentences = []
        current_embeddings = []
        current_tokens = 0

        for i, sentence in enumerate(sentences):
            sentence_tokens = self._estimate_tokens(sentence)
            sentence_embedding = embeddings[i]

            similarity = self._compute_similarity(
                sentence_embedding, current_embeddings
            )

            should_split = (
                current_tokens + sentence_tokens > self.max_tokens
                or (
                    similarity < self.similarity_threshold
                    and current_tokens >= self.min_tokens
                )
            )

            if should_split and current_sentences:
                chunks.append(
                    self._build_chunk(
                        current_sentences,
                        doc_id,
                        sentence_pages,
                        start_idx=i - len(current_sentences),
                        end_idx=i - 1,
                    )
                )

                # overlap
                current_sentences = self._get_overlap(current_sentences)
                current_embeddings = current_embeddings[-len(current_sentences):]
                current_tokens = sum(
                    self._estimate_tokens(s) for s in current_sentences
                )

            current_sentences.append(sentence)
            current_embeddings.append(sentence_embedding)
            current_tokens += sentence_tokens

        if current_sentences:
            chunks.append(
                self._build_chunk(
                    current_sentences,
                    doc_id,
                    sentence_pages,
                    start_idx=len(sentences) - len(current_sentences),
                    end_idx=len(sentences) - 1,
                )
            )

        return chunks

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------
    def _split_into_sentences(self, pages: List[Dict]):
        sentences = []
        sentence_pages = []

        for page in pages:
            page_num = page["page_num"]
            text = page["text"]

            # Conservative academic sentence splitting
            parts = re.split(
                r"(?<!et al)(?<!Fig)(?<!Eq)(?<!Dr)(?<!Mr)(?<!Ms)(?<=[.!?])\s+",
                text,
            )

            for s in parts:
                s = s.strip()
                if len(s) > 20:  # ignore junk fragments
                    sentences.append(s)
                    sentence_pages.append(page_num)

        return sentences, sentence_pages

    def _estimate_tokens(self, text: str) -> int:
        # lightweight, consistent approximation
        return max(1, int(len(text) / 4))

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.clip(norms, 1e-10, None)

    def _compute_similarity(
        self,
        sentence_embedding: np.ndarray,
        current_embeddings: List[np.ndarray],
    ) -> float:
        if not current_embeddings:
            return 1.0

        # use recent window to reduce centroid drift
        window = current_embeddings[-self.window_size :]
        centroid = np.mean(window, axis=0, keepdims=True)

        return cosine_similarity(
            sentence_embedding.reshape(1, -1), centroid
        )[0][0]

    def _get_overlap(self, sentences: List[str]) -> List[str]:
        tokens = 0
        overlap = []

        for s in reversed(sentences):
            tokens += self._estimate_tokens(s)
            overlap.insert(0, s)
            if tokens >= self.overlap_tokens:
                break

        return overlap

    def _build_chunk(
        self,
        sentences: List[str],
        doc_id: str,
        sentence_pages: List[int],
        start_idx: int,
        end_idx: int,
    ) -> Dict:
        pages = sentence_pages[start_idx : end_idx + 1]

        return {
            "chunk_id": str(uuid.uuid4()),
            "doc_id": doc_id,
            "text": " ".join(sentences),
            "page_start": min(pages),
            "page_end": max(pages),
            "token_count": sum(self._estimate_tokens(s) for s in sentences),
        }
