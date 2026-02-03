import uuid
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re


class SemanticChunker:
    """
    Section-agnostic semantic chunker for academic documents.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_tokens: int = 450,
        min_tokens: int = 200,
        overlap_tokens: int = 50,
        similarity_threshold: float = 0.75,
    ):
        self.model = SentenceTransformer(model_name)
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens
        self.similarity_threshold = similarity_threshold

    # -------------------------
    # Public API
    # -------------------------
    def chunk(self, pages: List[Dict], doc_id: str) -> List[Dict]:
        sentences, sentence_pages = self._split_into_sentences(pages)

        embeddings = self.model.encode(sentences, convert_to_numpy=True)

        chunks = []
        current_chunk = []
        current_embeddings = []
        current_token_count = 0

        for i, sentence in enumerate(sentences):
            sentence_tokens = self._estimate_tokens(sentence)

            if current_chunk:
                sim = cosine_similarity(
                    [embeddings[i]], [np.mean(current_embeddings, axis=0)]
                )[0][0]
            else:
                sim = 1.0

            if (
                current_token_count + sentence_tokens > self.max_tokens
                or (sim < self.similarity_threshold and current_token_count >= self.min_tokens)
            ):
                chunks.append(
                    self._build_chunk(
                        current_chunk,
                        doc_id,
                        sentence_pages,
                        start_idx=i - len(current_chunk),
                        end_idx=i - 1,
                    )
                )

                # overlap handling
                overlap = self._get_overlap(current_chunk)
                current_chunk = overlap
                current_embeddings = current_embeddings[-len(overlap):]
                current_token_count = sum(
                    self._estimate_tokens(s) for s in overlap
                )

            current_chunk.append(sentence)
            current_embeddings.append(embeddings[i])
            current_token_count += sentence_tokens

        if current_chunk:
            chunks.append(
                self._build_chunk(
                    current_chunk,
                    doc_id,
                    sentence_pages,
                    start_idx=len(sentences) - len(current_chunk),
                    end_idx=len(sentences) - 1,
                )
            )

        return chunks

    # -------------------------
    # Internal helpers
    # -------------------------
    def _split_into_sentences(self, pages: List[Dict]):
        sentences = []
        sentence_pages = []

        for page in pages:
            page_num = page["page_num"]
            text = page["text"]

            page_sentences = re.split(r"(?<=[.!?])\s+", text)
            for s in page_sentences:
                s = s.strip()
                if s:
                    sentences.append(s)
                    sentence_pages.append(page_num)

        return sentences, sentence_pages

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text.split()))

    def _get_overlap(self, chunk_sentences: List[str]) -> List[str]:
        tokens = 0
        overlap = []

        for sentence in reversed(chunk_sentences):
            tokens += self._estimate_tokens(sentence)
            overlap.insert(0, sentence)
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
