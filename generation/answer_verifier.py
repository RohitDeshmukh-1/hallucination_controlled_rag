from typing import List, Dict
import re

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class AnswerVerifier:
    """
    Verifies whether an LLM-generated answer is fully supported
    by the retrieved evidence chunks.

    The verifier operates at the sentence level with configurable
    tolerance. It allows partial support while maintaining
    hallucination awareness.
    """

    # Short filler phrases that don't require evidence grounding
    FILLER_PATTERNS = [
        r"^(yes|no|however|therefore|thus|in summary|to summarize|in conclusion)[,.]?$",
        r"^(this|that|it) (is|was|means|suggests|indicates)",
        r"^(based on|according to) (the|this)",
        r"^(let me|i will|i can)",
    ]

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.50,
        min_unsupported_ratio: float = 0.4,
        min_sentence_length: int = 20,
    ):
        """
        Parameters
        ----------
        similarity_threshold : float
            Minimum cosine similarity to consider a sentence supported.
        min_unsupported_ratio : float
            Maximum ratio of unsupported sentences before rejection.
            E.g., 0.4 means up to 40% unsupported is tolerated.
        min_sentence_length : int
            Sentences shorter than this are skipped (likely filler).
        """
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.min_unsupported_ratio = min_unsupported_ratio
        self.min_sentence_length = min_sentence_length

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

        # Filter out filler sentences that don't need grounding
        substantive_sentences = [
            s for s in answer_sentences if self._is_substantive(s)
        ]

        if not substantive_sentences:
            # All sentences are filler - accept as supported
            return {
                "verdict": "supported",
                "unsupported_sentences": [],
                "confidence": 1.0,
            }

        sentence_embeddings = self.model.encode(
            substantive_sentences, convert_to_numpy=True
        )
        evidence_embeddings = self.model.encode(
            evidence_texts, convert_to_numpy=True
        )

        unsupported = []
        sentence_scores = []

        for idx, sent_embedding in enumerate(sentence_embeddings):
            similarities = cosine_similarity(
                sent_embedding.reshape(1, -1),
                evidence_embeddings,
            )[0]

            max_sim = float(np.max(similarities))
            sentence_scores.append(max_sim)

            if max_sim < self.similarity_threshold:
                unsupported.append(substantive_sentences[idx])

        # Calculate support ratio and confidence
        unsupported_ratio = len(unsupported) / len(substantive_sentences)
        avg_confidence = float(np.mean(sentence_scores))

        if unsupported_ratio > self.min_unsupported_ratio:
            return {
                "verdict": "unsupported",
                "unsupported_sentences": unsupported,
                "confidence": avg_confidence,
                "support_ratio": 1 - unsupported_ratio,
            }

        if unsupported:
            return {
                "verdict": "partially_supported",
                "unsupported_sentences": unsupported,
                "confidence": avg_confidence,
                "support_ratio": 1 - unsupported_ratio,
            }

        return {
            "verdict": "supported",
            "unsupported_sentences": [],
            "confidence": avg_confidence,
            "support_ratio": 1.0,
        }

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using conservative punctuation rules.
        """
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if p.strip()]

    def _is_substantive(self, sentence: str) -> bool:
        """
        Check if a sentence is substantive (requires evidence grounding).
        Short sentences and common filler phrases are skipped.
        """
        if len(sentence) < self.min_sentence_length:
            return False

        sentence_lower = sentence.lower().strip()
        for pattern in self.FILLER_PATTERNS:
            if re.match(pattern, sentence_lower):
                return False

        return True

    def _unsupported(self, sentences: List[str]) -> Dict[str, object]:
        """
        Construct an unsupported verdict payload.
        """
        return {
            "verdict": "unsupported",
            "unsupported_sentences": sentences,
            "confidence": 0.0,
            "support_ratio": 0.0,
        }
