from typing import List, Dict, Any, Optional
import re
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from configs.settings import settings

logger = logging.getLogger(__name__)


class AnswerVerifier:
    """
    Verifies whether an LLM-generated answer is fully supported
    by the retrieved evidence chunks.
    """

    FILLER_PATTERNS = [
        r"^(yes|no|however|therefore|thus|in summary|to summarize|in conclusion|overall|additionally|furthermore|moreover)[,.]?$",
        r"^(this|that|it|these|those) (is|was|are|were|means|suggests|indicates|shows|demonstrates)",
        r"^(based on|according to) (the|this|these)",
        r"^(let me|i will|i can|note that|it is worth)",
        r"^(in other words|that is|for example|for instance|specifically)",
    ]

    def __init__(
        self,
        encoder_model: Any,
        similarity_threshold: float = settings.VERIFICATION_SIMILARITY_THRESHOLD,
        min_unsupported_ratio: float = settings.VERIFICATION_UNSUPPORTED_RATIO,
        min_sentence_length: int = 20,
    ):
        """
        Parameters
        ----------
        encoder_model : Any
            A sentence-transformer model or compatible object with .encode() method.
        similarity_threshold : float
            Minimum cosine similarity to consider a sentence supported.
        min_unsupported_ratio : float
            Maximum ratio of unsupported sentences before rejection.
        min_sentence_length : int
            Sentences shorter than this are skipped (likely filler).
        """
        self.model = encoder_model
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
                "verdict": "supported" | "partially_supported" | "unsupported",
                "unsupported_sentences": List[str],
                "confidence": float,
                "support_ratio": float,
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
            return {
                "verdict": "supported",
                "unsupported_sentences": [],
                "confidence": 1.0,
                "support_ratio": 1.0,
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

        unsupported_ratio = len(unsupported) / len(substantive_sentences)
        avg_confidence = float(np.mean(sentence_scores))

        logger.info(
            "Verification: %d/%d substantive sentences supported (threshold=%.2f, avg_sim=%.3f)",
            len(substantive_sentences) - len(unsupported),
            len(substantive_sentences),
            self.similarity_threshold,
            avg_confidence,
        )

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
        """Split text into sentences using conservative punctuation rules."""
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if p.strip()]

    def _is_substantive(self, sentence: str) -> bool:
        """
        Check if a sentence is substantive (requires evidence grounding).
        Short sentences, filler phrases, and citation-only references are skipped.
        """
        if len(sentence) < self.min_sentence_length:
            return False

        # Strip citation markers before checking
        clean = re.sub(r"\[E\d+\]", "", sentence).strip()
        if len(clean) < self.min_sentence_length:
            return False

        sentence_lower = clean.lower().strip()
        for pattern in self.FILLER_PATTERNS:
            if re.match(pattern, sentence_lower):
                return False

        return True

    def _unsupported(self, sentences: List[str]) -> Dict[str, object]:
        """Construct an unsupported verdict payload."""
        return {
            "verdict": "unsupported",
            "unsupported_sentences": sentences,
            "confidence": 0.0,
            "support_ratio": 0.0,
        }
