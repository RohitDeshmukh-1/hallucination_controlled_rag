from typing import List, Dict
import re

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class AnswerVerifier:
    """
    Verifies whether an LLM-generated answer is fully supported
    by the retrieved evidence chunks.

    The verifier operates at the sentence level and enforces
    conservative grounding constraints. If any sentence is not
    sufficiently supported by the evidence, verification fails.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.75,
    ):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold

    def verify(
        self,
        answer: str,
        evidence_chunks: List[Dict],
    ) -> Dict[str, object]:
        """
        Verify an answer against retrieved evidence.

        Parameters
        ----------
        answer : str
            Raw LLM-generated answer text.
        evidence_chunks : List[Dict]
            Retrieved chunks containing evidence text.

        Returns
        -------
        Dict[str, object]
            {
                "verdict": "supported" | "unsupported",
                "unsupported_sentences": List[str]
            }
        """

        answer_sentences = self._split_into_sentences(answer)
        evidence_texts = [chunk["text"] for chunk in evidence_chunks]

        if not answer_sentences or not evidence_texts:
            return self._unsupported(answer_sentences)

        sentence_embeddings = self.model.encode(
            answer_sentences, convert_to_numpy=True
        )
        evidence_embeddings = self.model.encode(
            evidence_texts, convert_to_numpy=True
        )

        unsupported = []

        for idx, sent_embedding in enumerate(sentence_embeddings):
            similarities = cosine_similarity(
                sent_embedding.reshape(1, -1),
                evidence_embeddings,
            )[0]

            if float(np.max(similarities)) < self.similarity_threshold:
                unsupported.append(answer_sentences[idx])

        if unsupported:
            return self._unsupported(unsupported)

        return {
            "verdict": "supported",
            "unsupported_sentences": [],
        }

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using conservative punctuation rules.
        """
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if p.strip()]

    def _unsupported(self, sentences: List[str]) -> Dict[str, object]:
        """
        Construct an unsupported verdict payload.
        """
        return {
            "verdict": "unsupported",
            "unsupported_sentences": sentences,
        }
