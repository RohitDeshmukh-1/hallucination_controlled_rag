"""
Citation Extractor Module.

Extracts canonical citations ([E1], [E2], ...) from LLM-generated answers
and maps them back to actual evidence chunks with page references.
"""

from typing import List, Dict, Tuple, Set
import re


class CitationExtractor:
    """
    Extracts and validates citations from generated answers.
    
    This module ensures that:
    - All citations in the answer are valid (exist in evidence)
    - Citations are properly mapped to source documents and pages
    - Uncited sentences are flagged for transparency
    """

    CITATION_PATTERN = re.compile(r"\[E(\d+)\]")

    def extract_and_map(
        self,
        answer: str,
        evidence_chunks: List[Dict],
    ) -> Dict:
        """
        Extract citations from answer and map to evidence metadata.

        Parameters
        ----------
        answer : str
            LLM-generated answer containing [E1], [E2], etc.
        evidence_chunks : List[Dict]
            Evidence chunks used for generation.

        Returns
        -------
        Dict
            {
                "answer_with_citations": str,
                "inline_citations": List[Dict],
                "citation_map": Dict[str, Dict],
                "uncited_sentences": List[str],
                "citation_coverage": float,
            }
        """
        # Build evidence map (1-indexed to match prompt)
        evidence_map = {}
        for idx, chunk in enumerate(evidence_chunks, start=1):
            evidence_map[f"E{idx}"] = {
                "evidence_id": f"E{idx}",
                "doc_id": chunk.get("doc_id", "unknown"),
                "page_start": chunk.get("page_start", 0),
                "page_end": chunk.get("page_end", 0),
                "text_preview": chunk.get("text", "")[:150] + "...",
            }

        # Extract all citations from answer
        found_citations = set(self.CITATION_PATTERN.findall(answer))
        
        # Validate citations exist in evidence
        valid_citations = {}
        invalid_citations = []
        
        for citation_num in found_citations:
            eid = f"E{citation_num}"
            if eid in evidence_map:
                valid_citations[eid] = evidence_map[eid]
            else:
                invalid_citations.append(eid)

        # Analyze sentence-level citation coverage
        sentences = self._split_sentences(answer)
        cited_sentences = []
        uncited_sentences = []

        for sentence in sentences:
            if self._is_substantive(sentence):
                if self.CITATION_PATTERN.search(sentence):
                    cited_sentences.append(sentence)
                else:
                    uncited_sentences.append(sentence)

        total_substantive = len(cited_sentences) + len(uncited_sentences)
        coverage = len(cited_sentences) / total_substantive if total_substantive > 0 else 1.0

        # Build inline citation list (ordered by appearance)
        inline_citations = []
        seen = set()
        for match in self.CITATION_PATTERN.finditer(answer):
            eid = f"E{match.group(1)}"
            if eid not in seen and eid in valid_citations:
                inline_citations.append(valid_citations[eid])
                seen.add(eid)

        return {
            "answer_with_citations": answer,
            "inline_citations": inline_citations,
            "citation_map": valid_citations,
            "invalid_citations": invalid_citations,
            "uncited_sentences": uncited_sentences,
            "citation_coverage": coverage,
        }

    def format_citation_footnotes(
        self,
        citation_map: Dict[str, Dict],
    ) -> str:
        """
        Format citations as footnotes for display.
        """
        if not citation_map:
            return ""

        lines = ["\n---\n**References:**\n"]
        for eid, meta in sorted(citation_map.items(), key=lambda x: int(x[0][1:])):
            pages = f"{meta['page_start']}-{meta['page_end']}"
            lines.append(f"- **[{eid}]** Document `{meta['doc_id']}`, Pages {pages}")

        return "\n".join(lines)

    def highlight_citations(self, answer: str) -> str:
        """
        Convert [E1] style citations to highlighted format for UI.
        """
        def replacer(match):
            eid = match.group(0)
            return f"**{eid}**"

        return self.CITATION_PATTERN.sub(
            lambda m: f"**[E{m.group(1)}]**",
            answer,
        )

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if p.strip()]

    def _is_substantive(self, sentence: str) -> bool:
        """Check if sentence requires citation (not filler)."""
        if len(sentence) < 20:
            return False
        filler_patterns = [
            r"^(yes|no|however|therefore|thus|in summary)[,.]?$",
            r"^(this|that|it) (is|was|means)",
            r"^(based on|according to)",
        ]
        sentence_lower = sentence.lower().strip()
        for pattern in filler_patterns:
            if re.match(pattern, sentence_lower):
                return False
        return True
