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
        max_evidence_chunks: int = 8,
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
            "- Answer using ONLY the evidence above. You may make reasonable "
            "inferences that follow directly from the evidence.\n"
            "- Factual claims MUST end with citations in the form [E1], [E2], etc.\n"
            "- Use only citation identifiers that appear in the evidence.\n"
            "- Connecting phrases and logical transitions do not require citations.\n"
            "- If the evidence provides partial information, answer what you can "
            "and note any limitations.\n"
            "- Only say you cannot answer if the evidence is completely irrelevant."
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
            "You are an academic question-answering system with strict citation requirements.\n"
            "CRITICAL RULES:\n"
            "1. You must cite sources using [E1], [E2], etc. format INLINE after each factual claim.\n"
            "2. Every sentence containing facts, numbers, names, or findings MUST end with a citation.\n"
            "3. You may only cite evidence IDs that appear in the provided evidence (E1 through E8 max).\n"
            "4. Do NOT use prior knowledge - only use information from the evidence.\n"
            "5. Place citations at the END of each sentence, before the period: '...the result is 0.5 [E1].'"
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
