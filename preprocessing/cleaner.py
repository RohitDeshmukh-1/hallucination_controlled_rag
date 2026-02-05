import re
from typing import List, Dict


class TextCleaner:
    """
    Conservative, production-grade cleaner for academic documents.

    Design principles:
    - Remove metadata and boilerplate noise
    - Preserve scientific claims and numerical evidence
    - Section-agnostic
    - Deterministic and explainable
    """

    def __init__(self):
        self.email_pattern = re.compile(
            r"\b[\w\.-]+@[\w\.-]+\.\w+\b"
        )

        self.inline_citation_patterns = [
            re.compile(r"\[\d+(?:,\s*\d+)*\]"),                 # [1], [1, 2]
            re.compile(r"\([A-Z][a-z]+ et al\.,?\s*\d{4}\)"),   # (Smith et al., 2019)
            re.compile(r"\(\d{4}\)"),                           # (2017)
        ]

        self.footer_patterns = [
            re.compile(r"Proceedings of .*", re.IGNORECASE),
            re.compile(r"arXiv:\d+\.\d+v\d+", re.IGNORECASE),
            re.compile(r"\bNeurIPS\b|\bNIPS\b|\bICML\b|\bICLR\b", re.IGNORECASE),
        ]

        self.multispace_pattern = re.compile(r"\s+")

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def clean_pages(self, pages: List[Dict]) -> List[Dict]:
        """
        Clean page-wise text while preserving provenance.
        """
        return [
            {
                "page_num": page["page_num"],
                "text": self._clean_text(page["text"]),
            }
            for page in pages
        ]

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------
    def _clean_text(self, text: str) -> str:
        # Remove email addresses
        text = self.email_pattern.sub("", text)

        # Remove inline citation markers
        for pattern in self.inline_citation_patterns:
            text = pattern.sub("", text)

        # Remove conference / archive footers
        for pattern in self.footer_patterns:
            text = pattern.sub("", text)

        # Normalize repeated punctuation
        text = re.sub(r"\.{2,}", ".", text)

        # Normalize whitespace
        text = self.multispace_pattern.sub(" ", text)

        return text.strip()
