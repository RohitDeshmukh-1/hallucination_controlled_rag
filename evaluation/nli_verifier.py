"""
NLI-based Verification Module for Hallucination Detection.

Uses a lightweight NLI cross-encoder to classify each answer sentence
as ENTAILMENT / CONTRADICTION / NEUTRAL w.r.t. evidence passages.

This is the *gold-standard* approach in research for detecting
hallucinated claims and is complementary to embedding-based
cosine similarity verification.
"""

import re
import logging
from typing import List, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

# We lazy-import the model to avoid slow startup if this module is not used.
_nli_model = None


def _get_nli_model():
    """Lazy-load the NLI cross-encoder (downloaded once, cached by HF)."""
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder
        logger.info("Loading NLI model: cross-encoder/nli-deberta-v3-small ...")
        _nli_model = CrossEncoder(
            "cross-encoder/nli-deberta-v3-small",
            max_length=512,
        )
        logger.info("NLI model loaded.")
    return _nli_model


# Label mapping for the nli-deberta-v3 models:
#   0  contradiction,  1  entailment,  2  neutral
LABEL_MAP = {0: "contradiction", 1: "entailment", 2: "neutral"}


class NLIVerifier:
    """
    Natural Language Inference verifier for hallucination detection.

    For each substantive sentence in the generated answer, this module:
      1. Pairs the sentence with every evidence passage.
      2. Runs NLI classification.
      3. Takes the best (max entailment) score across all evidence.
      4. Sentences with max_entailment < threshold are flagged.

    The aggregate entailment ratio serves as a research-grade
    faithfulness metric.
    """

    FILLER_PATTERNS = [
        r"^(yes|no|however|therefore|thus|in summary|to summarize|in conclusion|overall)[,.]?$",
        r"^(this|that|it|these|those) (is|was|are|were|means|suggests|indicates)",
        r"^(based on|according to|as mentioned|as stated)",
        r"^(let me|i will|i can|note that|it is worth)",
        r"^(in other words|that is|for example|for instance|specifically)",
    ]

    def __init__(
        self,
        entailment_threshold: float = 0.65,
        max_unsupported_ratio: float = 0.5,
        min_sentence_length: int = 20,
    ):
        self.entailment_threshold = entailment_threshold
        self.max_unsupported_ratio = max_unsupported_ratio
        self.min_sentence_length = min_sentence_length

    def verify(
        self,
        answer: str,
        evidence_chunks: List[Dict],
    ) -> Dict[str, Any]:
        """
        Verify an answer against evidence using NLI.

        Returns
        -------
        Dict with keys:
            verdict : str - "entailed", "partially_entailed", "contradicted"
            entailment_ratio : float - fraction of sentences entailed
            avg_entailment_score : float
            per_sentence : List[Dict] - per-sentence breakdown
        """
        model = _get_nli_model()

        sentences = self._split_sentences(answer)
        evidence_texts = [c["text"] for c in evidence_chunks]

        if not evidence_texts:
            return self._empty_result(sentences)

        substantive = [s for s in sentences if self._is_substantive(s)]

        if not substantive:
            return {
                "verdict": "entailed",
                "entailment_ratio": 1.0,
                "avg_entailment_score": 1.0,
                "per_sentence": [],
            }

        per_sentence: List[Dict] = []
        entailed_count = 0

        for sentence in substantive:
            # Create all (evidence, sentence) pairs for NLI
            pairs = [(ev, sentence) for ev in evidence_texts]
            scores = model.predict(pairs)  # shape: (n_evidence, 3)

            if scores.ndim == 1:
                scores = scores.reshape(1, -1)

            # For each evidence passage, get the entailment probability
            entailment_scores = scores[:, 1]  # column 1 = entailment
            best_idx = int(np.argmax(entailment_scores))
            best_entailment = float(entailment_scores[best_idx])

            # Also check for contradictions
            contradiction_scores = scores[:, 0]
            max_contradiction = float(np.max(contradiction_scores))

            # Determine label
            best_label_idx = int(np.argmax(scores[best_idx]))
            best_label = LABEL_MAP.get(best_label_idx, "neutral")

            is_entailed = best_entailment >= self.entailment_threshold
            if is_entailed:
                entailed_count += 1

            per_sentence.append({
                "sentence": sentence,
                "best_entailment_score": round(best_entailment, 4),
                "max_contradiction_score": round(max_contradiction, 4),
                "best_label": best_label,
                "is_entailed": is_entailed,
                "best_evidence_idx": best_idx,
            })

        entailment_ratio = entailed_count / len(substantive)
        avg_score = float(np.mean([p["best_entailment_score"] for p in per_sentence]))

        # Determine verdict
        unsupported_ratio = 1.0 - entailment_ratio
        has_contradiction = any(
            p["best_label"] == "contradiction" and p["max_contradiction_score"] > 0.7
            for p in per_sentence
        )

        if has_contradiction or unsupported_ratio > self.max_unsupported_ratio:
            verdict = "contradicted"
        elif unsupported_ratio > 0:
            verdict = "partially_entailed"
        else:
            verdict = "entailed"

        logger.info(
            "NLI Verification: %d/%d entailed (ratio=%.3f, avg_score=%.3f, verdict=%s)",
            entailed_count, len(substantive),
            entailment_ratio, avg_score, verdict,
        )

        return {
            "verdict": verdict,
            "entailment_ratio": round(entailment_ratio, 4),
            "avg_entailment_score": round(avg_score, 4),
            "per_sentence": per_sentence,
        }

    def _split_sentences(self, text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if p.strip()]

    def _is_substantive(self, sentence: str) -> bool:
        if len(sentence) < self.min_sentence_length:
            return False
        clean = re.sub(r"\[E\d+\]", "", sentence).strip()
        if len(clean) < self.min_sentence_length:
            return False
        lower = clean.lower()
        return not any(re.match(p, lower) for p in self.FILLER_PATTERNS)

    def _empty_result(self, sentences: List[str]) -> Dict:
        return {
            "verdict": "contradicted",
            "entailment_ratio": 0.0,
            "avg_entailment_score": 0.0,
            "per_sentence": [
                {"sentence": s, "best_entailment_score": 0.0, "is_entailed": False}
                for s in sentences
            ],
        }
