from typing import Dict
import re


class CitationRenderer:
    """
    Renders canonical evidence citations ([E1], [E2], ...)
    into journal-specific citation styles.

    This module is purely presentational and does not affect
    grounding or verification.
    """

    def __init__(self, evidence_map: Dict[str, Dict]):
        """
        Parameters
        ----------
        evidence_map : Dict
            Mapping from evidence ID (E1, E2, ...)
            to metadata (authors, year, pages, etc.).
        """
        self.evidence_map = evidence_map

    def render_ieee(self, text: str) -> str:
        """
        Render citations in IEEE numeric style.
        """
        return self._replace(text, prefix="[", suffix="]")

    def render_nature(self, text: str) -> str:
        """
        Render citations in Nature superscript style.
        """
        return self._replace(text, prefix="⁽", suffix="⁾")

    def render_apa(self, text: str) -> str:
        """
        Render citations in APA-style author-year format.
        """
        def repl(match):
            eid = match.group(1)
            meta = self.evidence_map.get(eid, {})
            return f"({meta.get('author', 'Unknown')}, {meta.get('year', 'n.d.')})"

        return re.sub(r"\[(E\d+)\]", repl, text)

    def _replace(self, text: str, prefix: str, suffix: str) -> str:
        """
        Generic numeric citation replacement.
        """
        return re.sub(
            r"\[(E\d+)\]",
            lambda m: f"{prefix}{m.group(1)[1:]}{suffix}",
            text,
        )
