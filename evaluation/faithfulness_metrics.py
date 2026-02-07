from typing import List, Dict


class FaithfulnessMetrics:
    """
    Computes faithfulness-oriented evaluation metrics for
    hallucination-controlled RAG systems.

    This module assumes that answer verification has already
    been performed and operates purely on verifier outputs.
    """

    def __init__(self):
        self.total_questions = 0
        self.refused_questions = 0
        self.total_sentences = 0
        self.unsupported_sentences = 0

    def update(self, verifier_result: Dict) -> None:
        """
        Update metric counters using a single verifier result.

        Parameters
        ----------
        verifier_result : Dict
            Output from AnswerVerifier.verify(...)
        """
        self.total_questions += 1

        if verifier_result["verdict"] == "unsupported":
            self.refused_questions += 1
            self.unsupported_sentences += len(
                verifier_result["unsupported_sentences"]
            )

        self.total_sentences += max(
            1, len(verifier_result.get("unsupported_sentences", []))
        )

    def compute(self) -> Dict[str, float]:
        """
        Compute aggregate faithfulness metrics.
        """
        if self.total_questions == 0:
            return {}

        sentence_support_rate = 1.0 - (
            self.unsupported_sentences / max(1, self.total_sentences)
        )

        refusal_rate = self.refused_questions / self.total_questions

        unsupported_claim_rate = (
            self.unsupported_sentences / max(1, self.total_sentences)
        )

        return {
            "sentence_support_rate": round(sentence_support_rate, 4),
            "refusal_rate": round(refusal_rate, 4),
            "unsupported_claim_rate": round(unsupported_claim_rate, 4),
        }