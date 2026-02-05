from typing import List, Dict


class PromptBuilder:
    """
    Production-grade prompt constructor for hallucination-controlled RAG.

    Responsibilities:
    - Construct evidence-only prompts
    - Enforce grounding constraints
    - Explicitly allow refusal when evidence is insufficient

    Design principles:
    - Deterministic formatting
    - No implicit reasoning instructions
    - No reliance on LLM prior knowledge
    """

    def __init__(
        self,
        system_instruction: str = (
            "You are an academic assistant.\n"
            "You must answer the question using ONLY the evidence provided.\n"
            "Do not use prior knowledge.\n"
            "If the evidence does not contain the answer, say:\n"
            "\"I cannot answer based on the provided documents.\""
        ),
        max_evidence_chunks: int = 5,
    ):
        self.system_instruction = system_instruction
        self.max_evidence_chunks = max_evidence_chunks

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def build(
        self,
        question: str,
        retrieved_chunks: List[Dict],
    ) -> Dict[str, str]:
        """
        Build a grounded prompt from retrieved evidence.

        Returns:
            {
              "system": system prompt,
              "user": user prompt
            }
        """

        evidence_blocks = self._format_evidence(retrieved_chunks)

        user_prompt = (
            "Evidence:\n"
            f"{evidence_blocks}\n\n"
            "Question:\n"
            f"{question}"
        )

        return {
            "system": self.system_instruction,
            "user": user_prompt,
        }

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------
    def _format_evidence(self, chunks: List[Dict]) -> str:
        """
        Format retrieved chunks into strict evidence blocks.
        """
        blocks = []

        for i, chunk in enumerate(chunks[: self.max_evidence_chunks], start=1):
            block = (
                f"[E{i} | doc:{chunk['doc_id']} | pages:{chunk['page_start']}-{chunk['page_end']}]\n"
                f"{chunk.get('text', '')}"
            )
            blocks.append(block)

        return "\n\n".join(blocks)
