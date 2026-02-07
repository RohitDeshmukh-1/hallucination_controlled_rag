from typing import List, Dict


class PromptBuilder:
    """
    Constructs citation-aware, evidence-constrained prompts for academic RAG.

    This prompt enforces:
    - Evidence-only generation
    - Explicit per-sentence citations
    - Canonical citation identifiers ([E1], [E2], ...)
    - Principled abstention when evidence is insufficient

    The canonical citation format is intentionally style-agnostic and
    can be rendered into IEEE, APA, Nature, or other journal formats
    as a post-processing step.
    """

    def __init__(
        self,
        max_evidence_chunks: int = 5,
    ):
        self.max_evidence_chunks = max_evidence_chunks

    def build(
        self,
        question: str,
        retrieved_chunks: List[Dict],
    ) -> Dict[str, str]:
        """
        Build a citation-aware prompt.

        Parameters
        ----------
        question : str
            User question.
        retrieved_chunks : List[Dict]
            Evidence chunks after cross-encoder re-ranking.

        Returns
        -------
        Dict[str, str]
            Dictionary with 'system' and 'user' prompt fields.
        """

        system_prompt = self._system_instruction()
        evidence_block = self._format_evidence(retrieved_chunks)

        user_prompt = (
            "You are provided with evidence excerpts from academic documents.\n\n"
            "Evidence:\n"
            f"{evidence_block}\n\n"
            "Question:\n"
            f"{question}\n\n"
            "Instructions:\n"
            "- Answer using ONLY the evidence above.\n"
            "- Every factual sentence MUST end with one or more citations "
            "in the form [E1], [E2], etc.\n"
            "- Use only citation identifiers that appear in the evidence.\n"
            "- Do NOT combine multiple claims in one sentence unless all are cited.\n"
            "- If the evidence is insufficient, respond exactly with:\n"
            "  \"I cannot answer based on the provided documents.\""
        )

        return {
            "system": system_prompt,
            "user": user_prompt,
        }

    def _system_instruction(self) -> str:
        """
        System-level instruction enforcing strict grounding and citation behavior.
        """
        return (
            "You are an academic question-answering system.\n"
            "You must not use prior knowledge or make assumptions.\n"
            "All answers must be fully grounded in the provided evidence.\n"
            "If a claim cannot be supported with a citation, it must not be written."
        )

    def _format_evidence(self, chunks: List[Dict]) -> str:
        """
        Format evidence chunks into canonical citation blocks.
        """
        blocks = []

        for idx, chunk in enumerate(
            chunks[: self.max_evidence_chunks], start=1
        ):
            block = (
                f"[E{idx} | doc:{chunk['doc_id']} | pages:{chunk['page_start']}-{chunk['page_end']}]\n"
                f"{chunk['text']}"
            )
            blocks.append(block)

        return "\n\n".join(blocks)
