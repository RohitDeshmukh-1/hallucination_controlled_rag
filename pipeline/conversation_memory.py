"""
Conversation Memory Manager for ResearchMind RAG.

Provides:
- Multi-turn conversation history tracking
- Pinnable key insights (memory pins)
- Context-aware query rewriting using history
- Session metadata management
"""
from __future__ import annotations

import uuid
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


class MemoryPin:
    """A user-pinned insight saved from a conversation turn."""

    def __init__(self, text: str, source_question: str, from_doc: Optional[str] = None):
        self.id = uuid.uuid4().hex[:8]
        self.text = text
        self.source_question = source_question
        self.from_doc = from_doc
        self.created_at = datetime.now()

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "text": self.text,
            "source_question": self.source_question,
            "from_doc": self.from_doc,
            "created_at": self.created_at.strftime("%H:%M"),
        }


class ConversationTurn:
    """A single turn: question + answer + metadata."""

    def __init__(
        self,
        question: str,
        answer: str,
        verdict: str,
        confidence: float,
        citations: List[Dict],
        evidence: List[Dict],
        support_ratio: float = 0.0,
        citation_coverage: float = 0.0,
    ):
        self.id = uuid.uuid4().hex[:8]
        self.question = question
        self.answer = answer
        self.verdict = verdict
        self.confidence = confidence
        self.citations = citations
        self.evidence = evidence
        self.support_ratio = support_ratio
        self.citation_coverage = citation_coverage
        self.timestamp = datetime.now()
        self.pinned = False

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "citations": self.citations,
            "support_ratio": self.support_ratio,
            "citation_coverage": self.citation_coverage,
            "timestamp": self.timestamp.strftime("%H:%M"),
            "pinned": self.pinned,
        }


class ConversationMemory:
    """
    Manages the full memory and context for a research session.

    Features:
    - Rolling history window for context-aware query rewriting
    - Memory pins for user-curated key facts
    - Document-scoped memory (per uploaded paper)
    - Session statistics
    """

    MAX_HISTORY_WINDOW = 6  # turns kept for context injection

    def __init__(self, session_name: str = "Research Session"):
        self.session_id = uuid.uuid4().hex[:12]
        self.session_name = session_name
        self.created_at = datetime.now()
        self.turns: List[ConversationTurn] = []
        self.pins: List[MemoryPin] = []
        self.active_docs: List[Dict] = []  # {doc_id, filename, chunk_count}

    # ── History ──────────────────────────────────────────────────────────────

    def add_turn(self, turn: ConversationTurn):
        """Append a new conversation turn."""
        self.turns.append(turn)
        logger.info("Memory: added turn %s (total=%d)", turn.id, len(self.turns))

    def get_recent_turns(self, n: Optional[int] = None) -> List[ConversationTurn]:
        """Return the N most recent turns (defaults to MAX_HISTORY_WINDOW)."""
        n = n or self.MAX_HISTORY_WINDOW
        return self.turns[-n:]

    def build_context_prompt(self) -> str:
        """
        Build a condensed context string from recent history for injection
        into the next query prompt, enabling multi-turn awareness.
        """
        recent = self.get_recent_turns()
        if not recent:
            return ""

        lines = ["Previous conversation context (for continuity only):"]
        for t in recent:
            lines.append(f"  Q: {t.question}")
            # Truncate long answers to keep context compact
            preview = t.answer[:200].replace("\n", " ")
            if len(t.answer) > 200:
                preview += "..."
            lines.append(f"  A: {preview}")

        return "\n".join(lines)

    def rewrite_query_with_context(self, question: str) -> str:
        """
        Expand a short/ambiguous query using recent conversation context.
        Simple heuristic: if question contains pronouns referring to previous
        turns, prepend context clues.
        """
        REFERENTIAL_TRIGGERS = [
            "it", "they", "this", "that", "those", "these",
            "his", "her", "their", "him", "them", "its", "she", "he",
            "the same", "above", "previous", "mentioned", "the paper",
            "the study", "the authors", "the method", "the approach",
        ]
        q_lower = question.lower()
        is_referential = any(
            q_lower.startswith(t) or f" {t} " in q_lower
            for t in REFERENTIAL_TRIGGERS
        )

        if not is_referential or not self.turns:
            return question

        # Inject last question topic as prefix context
        last_q = self.turns[-1].question
        rewritten = f"[Context: previously asked '{last_q}'] {question}"
        logger.info("Query rewritten with context: %s", rewritten[:100])
        return rewritten

    # ── Pins ─────────────────────────────────────────────────────────────────

    def add_pin(self, text: str, source_question: str, from_doc: Optional[str] = None) -> MemoryPin:
        """Pin a key insight to persistent memory."""
        pin = MemoryPin(text=text, source_question=source_question, from_doc=from_doc)
        self.pins.append(pin)
        logger.info("Memory pin added: %s...", text[:60])
        return pin

    def remove_pin(self, pin_id: str) -> bool:
        """Remove a pin by ID."""
        before = len(self.pins)
        self.pins = [p for p in self.pins if p.id != pin_id]
        return len(self.pins) < before

    def get_pins_context(self) -> str:
        """Return pinned insights as a prompt context block."""
        if not self.pins:
            return ""
        lines = ["Key facts pinned by user (treat as authoritative context):"]
        for pin in self.pins:
            lines.append(f"  - {pin.text}")
        return "\n".join(lines)

    # ── Documents ─────────────────────────────────────────────────────────────

    def register_document(self, doc_id: str, filename: str, chunk_count: int):
        """Register an uploaded document into the session."""
        # Avoid duplicates
        existing_ids = {d["doc_id"] for d in self.active_docs}
        if doc_id not in existing_ids:
            self.active_docs.append({
                "doc_id": doc_id,
                "filename": filename,
                "chunk_count": chunk_count,
                "indexed_at": datetime.now().strftime("%H:%M"),
            })

    # ── Session Stats ─────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return session-level aggregate statistics."""
        if not self.turns:
            return {
                "total_turns": 0,
                "supported": 0,
                "partial": 0,
                "refused": 0,
                "avg_confidence": 0.0,
                "avg_citation_coverage": 0.0,
            }

        verdicts = [t.verdict for t in self.turns]
        confidences = [t.confidence for t in self.turns if t.confidence > 0]
        coverages = [t.citation_coverage for t in self.turns if t.citation_coverage > 0]

        return {
            "total_turns": len(self.turns),
            "supported": verdicts.count("supported"),
            "partial": verdicts.count("partially_supported"),
            "refused": verdicts.count("refused"),
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "avg_citation_coverage": sum(coverages) / len(coverages) if coverages else 0.0,
        }

    def clear(self):
        """Reset the session completely."""
        self.turns.clear()
        self.pins.clear()
        self.active_docs.clear()
        logger.info("Memory cleared for session %s", self.session_id)
