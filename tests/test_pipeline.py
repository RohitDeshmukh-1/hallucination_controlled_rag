"""
Comprehensive test suite for the Hallucination-Controlled Academic RAG pipeline.

Covers:
  - Pipeline flow tests (success, empty index, LLM failure, no evidence)
  - AnswerVerifier unit tests
  - CitationExtractor unit tests
  - PromptBuilder unit tests
  - FaithfulnessMetrics unit tests
  - Edge case regression tests
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock

from pipeline.query_pipeline import run_query_pipeline
from retrieval.faiss_index import FaissIndex
from retrieval.cross_encoder import CrossEncoderReranker
from embeddings.encoder import EmbeddingEncoder
from generation.llm_client import LLMClient
from generation.answer_verifier import AnswerVerifier
from generation.citation_extractor import CitationExtractor
from generation.prompt_builder import PromptBuilder
from evaluation.faithfulness_metrics import FaithfulnessMetrics


# ──────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────────────────────────────────

SAMPLE_EVIDENCE = [
    {
        "chunk_id": "1",
        "text": "The Transformer architecture uses multi-head self-attention mechanisms to process sequences in parallel.",
        "doc_id": "doc_abc",
        "page_start": 3,
        "page_end": 4,
    },
    {
        "chunk_id": "2",
        "text": "Experiments on WMT 2014 English-to-German translation show a BLEU score of 28.4, outperforming all prior single models.",
        "doc_id": "doc_abc",
        "page_start": 7,
        "page_end": 8,
    },
]


@pytest.fixture
def mock_encoder():
    encoder = MagicMock(spec=EmbeddingEncoder)
    encoder.embed_query.return_value = np.random.rand(384).astype("float32")
    encoder.model = MagicMock()
    return encoder


@pytest.fixture
def mock_index():
    index = MagicMock(spec=FaissIndex)
    index.chunks = SAMPLE_EVIDENCE.copy()
    index.search.return_value = [
        {**chunk, "similarity_score": 0.9 - i * 0.1}
        for i, chunk in enumerate(SAMPLE_EVIDENCE)
    ]
    return index


@pytest.fixture
def mock_reranker():
    reranker = MagicMock(spec=CrossEncoderReranker)
    reranker.rerank.return_value = [
        {**chunk, "cross_score": 0.95 - i * 0.1}
        for i, chunk in enumerate(SAMPLE_EVIDENCE)
    ]
    return reranker


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLMClient)
    llm.generate.return_value = (
        "The Transformer uses multi-head self-attention [E1]. "
        "It achieved a BLEU score of 28.4 on WMT 2014 [E2]."
    )
    return llm


# ──────────────────────────────────────────────────────────────────────────
# Pipeline tests
# ──────────────────────────────────────────────────────────────────────────

class TestQueryPipeline:
    """Tests for the end-to-end query pipeline."""

    def test_success_supported(self, mock_encoder, mock_index, mock_reranker, mock_llm):
        """Pipeline returns 'supported' when all verifications pass."""
        with patch("pipeline.query_pipeline.AnswerVerifier") as MockVerifier, \
             patch("pipeline.query_pipeline.CitationExtractor") as MockExtractor:

            MockVerifier.return_value.verify.return_value = {
                "verdict": "supported",
                "confidence": 0.92,
                "support_ratio": 1.0,
                "unsupported_sentences": [],
            }
            MockExtractor.return_value.extract_and_map.return_value = {
                "inline_citations": [
                    {"evidence_id": "E1", "doc_id": "doc_abc", "page_start": 3, "page_end": 4, "text_preview": "..."},
                    {"evidence_id": "E2", "doc_id": "doc_abc", "page_start": 7, "page_end": 8, "text_preview": "..."},
                ],
                "citation_coverage": 1.0,
                "uncited_sentences": [],
            }

            result = run_query_pipeline("What is Transformer?", mock_encoder, mock_index, mock_reranker, mock_llm)

            assert result["verdict"] == "supported"
            assert len(result["citations"]) == 2
            assert result["confidence"] > 0.5

    def test_empty_index_refused(self, mock_encoder, mock_index, mock_reranker, mock_llm):
        """Pipeline refuses when no documents are indexed."""
        mock_index.chunks = []

        result = run_query_pipeline("Any question?", mock_encoder, mock_index, mock_reranker, mock_llm)

        assert result["verdict"] == "refused"
        assert "No documents" in result["answer"]
        mock_llm.generate.assert_not_called()

    def test_no_retrieval_results(self, mock_encoder, mock_index, mock_reranker, mock_llm):
        """Pipeline refuses when retrieval returns nothing."""
        mock_index.search.return_value = []

        result = run_query_pipeline("Obscure question?", mock_encoder, mock_index, mock_reranker, mock_llm)

        assert result["verdict"] == "refused"
        mock_llm.generate.assert_not_called()

    def test_no_reranked_evidence(self, mock_encoder, mock_index, mock_reranker, mock_llm):
        """Pipeline refuses when reranker returns empty results."""
        mock_reranker.rerank.return_value = []

        result = run_query_pipeline("Obscure question?", mock_encoder, mock_index, mock_reranker, mock_llm)

        assert result["verdict"] == "refused"
        mock_llm.generate.assert_not_called()

    def test_llm_failure_returns_refused(self, mock_encoder, mock_index, mock_reranker, mock_llm):
        """Pipeline returns refused when LLM raises an exception."""
        mock_llm.generate.side_effect = Exception("API timeout")

        with patch("pipeline.query_pipeline.CitationExtractor"), \
             patch("pipeline.query_pipeline.AnswerVerifier"):
            result = run_query_pipeline("What is X?", mock_encoder, mock_index, mock_reranker, mock_llm)

        assert result["verdict"] == "refused"
        assert "unavailable" in result["answer"].lower()

    def test_unsupported_verdict_triggers_refusal(self, mock_encoder, mock_index, mock_reranker, mock_llm):
        """Pipeline converts 'unsupported' verdict to 'refused'."""
        with patch("pipeline.query_pipeline.AnswerVerifier") as MockVerifier, \
             patch("pipeline.query_pipeline.CitationExtractor") as MockExtractor:

            MockVerifier.return_value.verify.return_value = {
                "verdict": "unsupported",
                "confidence": 0.2,
                "support_ratio": 0.1,
                "unsupported_sentences": ["Fabricated claim not in evidence."],
            }
            MockExtractor.return_value.extract_and_map.return_value = {
                "inline_citations": [],
                "citation_coverage": 0.0,
                "uncited_sentences": [],
            }

            result = run_query_pipeline("What is X?", mock_encoder, mock_index, mock_reranker, mock_llm)

        assert result["verdict"] == "refused"

    def test_partially_supported_adds_caveat(self, mock_encoder, mock_index, mock_reranker, mock_llm):
        """Pipeline adds a caveat note for partially supported answers."""
        with patch("pipeline.query_pipeline.AnswerVerifier") as MockVerifier, \
             patch("pipeline.query_pipeline.CitationExtractor") as MockExtractor:

            MockVerifier.return_value.verify.return_value = {
                "verdict": "partially_supported",
                "confidence": 0.6,
                "support_ratio": 0.7,
                "unsupported_sentences": ["Some claim."],
            }
            MockExtractor.return_value.extract_and_map.return_value = {
                "inline_citations": [
                    {"evidence_id": "E1", "doc_id": "doc_abc", "page_start": 3, "page_end": 4, "text_preview": "..."},
                ],
                "citation_coverage": 0.5,
                "uncited_sentences": ["Some claim."],
            }

            result = run_query_pipeline("What is X?", mock_encoder, mock_index, mock_reranker, mock_llm)

        assert result["verdict"] == "partially_supported"
        assert "limited" in result["answer"].lower()

    def test_empty_question_still_processed(self, mock_encoder, mock_index, mock_reranker, mock_llm):
        """Pipeline handles whitespace-only question strings (guard is in API layer)."""
        with patch("pipeline.query_pipeline.AnswerVerifier") as MockVerifier, \
             patch("pipeline.query_pipeline.CitationExtractor") as MockExtractor:
            MockVerifier.return_value.verify.return_value = {
                "verdict": "supported", "confidence": 1.0,
                "support_ratio": 1.0, "unsupported_sentences": [],
            }
            MockExtractor.return_value.extract_and_map.return_value = {
                "inline_citations": [], "citation_coverage": 1.0, "uncited_sentences": [],
            }

            result = run_query_pipeline("   ", mock_encoder, mock_index, mock_reranker, mock_llm)
            # Pipeline doesn't validate question content — that's the API's job
            assert result["verdict"] in ("supported", "refused")


# ──────────────────────────────────────────────────────────────────────────
# AnswerVerifier tests
# ──────────────────────────────────────────────────────────────────────────

class TestAnswerVerifier:
    """Tests for the sentence-level verification logic."""

    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        return model

    def test_all_supported(self, mock_model):
        # Model returns high similarity for everything
        mock_model.encode.return_value = np.array([[0.9, 0.1, 0.0]] * 2)

        verifier = AnswerVerifier(
            encoder_model=mock_model,
            similarity_threshold=0.3,
        )

        with patch("generation.answer_verifier.cosine_similarity") as mock_cos:
            mock_cos.return_value = np.array([[0.85]])

            result = verifier.verify(
                "The model achieved high accuracy on the benchmark dataset.",
                SAMPLE_EVIDENCE,
            )

        assert result["verdict"] == "supported"
        assert result["confidence"] > 0

    def test_empty_answer(self, mock_model):
        verifier = AnswerVerifier(encoder_model=mock_model)
        result = verifier.verify("", SAMPLE_EVIDENCE)
        assert result["verdict"] == "unsupported"

    def test_empty_evidence(self, mock_model):
        verifier = AnswerVerifier(encoder_model=mock_model)
        result = verifier.verify("Some answer text here.", [])
        assert result["verdict"] == "unsupported"

    def test_filler_only_answer(self, mock_model):
        verifier = AnswerVerifier(encoder_model=mock_model)
        result = verifier.verify("Yes. No. However.", SAMPLE_EVIDENCE)
        # All sentences are filler/short — should be treated as supported
        assert result["verdict"] == "supported"

    def test_sentence_splitting(self, mock_model):
        verifier = AnswerVerifier(encoder_model=mock_model)
        sentences = verifier._split_into_sentences(
            "First sentence. Second sentence! Third sentence?"
        )
        assert len(sentences) == 3

    def test_substantive_detection(self, mock_model):
        verifier = AnswerVerifier(encoder_model=mock_model)

        assert verifier._is_substantive(
            "The Transformer achieves state-of-the-art BLEU scores on WMT."
        )
        assert not verifier._is_substantive("Yes.")
        assert not verifier._is_substantive("[E1]")
        assert not verifier._is_substantive("In summary.")


# ──────────────────────────────────────────────────────────────────────────
# CitationExtractor tests
# ──────────────────────────────────────────────────────────────────────────

class TestCitationExtractor:
    """Tests for citation extraction and validation."""

    def test_basic_extraction(self):
        extractor = CitationExtractor()
        result = extractor.extract_and_map(
            "The model works well [E1]. The BLEU score is 28.4 [E2].",
            SAMPLE_EVIDENCE,
        )

        assert len(result["inline_citations"]) == 2
        assert result["citation_coverage"] == 1.0

    def test_invalid_citation_flagged(self):
        extractor = CitationExtractor()
        result = extractor.extract_and_map(
            "The model works well [E1]. Another claim [E99].",
            SAMPLE_EVIDENCE,
        )

        assert "E99" in result["invalid_citations"]
        assert len(result["inline_citations"]) == 1

    def test_no_citations(self):
        extractor = CitationExtractor()
        result = extractor.extract_and_map(
            "The model works well. The BLEU score is high.",
            SAMPLE_EVIDENCE,
        )

        assert len(result["inline_citations"]) == 0
        assert result["citation_coverage"] == 0.0

    def test_duplicate_citation_dedup(self):
        extractor = CitationExtractor()
        result = extractor.extract_and_map(
            "First claim [E1]. Second claim [E1]. Third claim [E1].",
            SAMPLE_EVIDENCE,
        )

        # Should only appear once in inline_citations
        assert len(result["inline_citations"]) == 1

    def test_empty_answer(self):
        extractor = CitationExtractor()
        result = extractor.extract_and_map("", SAMPLE_EVIDENCE)
        assert result["citation_coverage"] == 1.0  # no substantive sentences
        assert len(result["inline_citations"]) == 0

    def test_highlight_citations(self):
        extractor = CitationExtractor()
        highlighted = extractor.highlight_citations("Result is 0.5 [E1].")
        assert "**[E1]**" in highlighted


# ──────────────────────────────────────────────────────────────────────────
# PromptBuilder tests
# ──────────────────────────────────────────────────────────────────────────

class TestPromptBuilder:
    """Tests for prompt construction."""

    def test_basic_build(self):
        builder = PromptBuilder()
        prompt = builder.build("What is X?", SAMPLE_EVIDENCE)

        assert "system" in prompt
        assert "user" in prompt
        assert "[E1" in prompt["user"]
        assert "[E2" in prompt["user"]
        assert "What is X?" in prompt["user"]

    def test_evidence_limit(self):
        builder = PromptBuilder(max_evidence_chunks=1)
        prompt = builder.build("What is X?", SAMPLE_EVIDENCE)

        assert "[E1" in prompt["user"]
        assert "[E2" not in prompt["user"]

    def test_empty_evidence(self):
        builder = PromptBuilder()
        prompt = builder.build("What is X?", [])

        assert "system" in prompt
        assert "What is X?" in prompt["user"]

    def test_system_prompt_contains_rules(self):
        builder = PromptBuilder()
        prompt = builder.build("Q?", SAMPLE_EVIDENCE)

        system = prompt["system"].lower()
        assert "citation" in system
        assert "evidence" in system


# ──────────────────────────────────────────────────────────────────────────
# FaithfulnessMetrics tests
# ──────────────────────────────────────────────────────────────────────────

class TestFaithfulnessMetrics:
    """Tests for the faithfulness aggregation module."""

    def test_empty_compute(self):
        metrics = FaithfulnessMetrics()
        assert metrics.compute() == {}

    def test_all_supported(self):
        metrics = FaithfulnessMetrics()
        metrics.update({
            "verdict": "supported",
            "unsupported_sentences": [],
        })
        metrics.update({
            "verdict": "supported",
            "unsupported_sentences": [],
        })

        result = metrics.compute()
        assert result["refusal_rate"] == 0.0

    def test_mixed_verdicts(self):
        metrics = FaithfulnessMetrics()
        metrics.update({
            "verdict": "supported",
            "unsupported_sentences": [],
        })
        metrics.update({
            "verdict": "unsupported",
            "unsupported_sentences": ["Bad claim 1.", "Bad claim 2."],
        })

        result = metrics.compute()
        assert result["refusal_rate"] == 0.5
        assert result["unsupported_claim_rate"] > 0
