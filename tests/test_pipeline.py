import pytest
from unittest.mock import MagicMock, patch
from pipeline.query_pipeline import run_query_pipeline
from retrieval.faiss_index import FaissIndex
from retrieval.cross_encoder import CrossEncoderReranker
from embeddings.encoder import EmbeddingEncoder
from generation.llm_client import LLMClient

@pytest.fixture
def mock_encoder():
    encoder = MagicMock(spec=EmbeddingEncoder)
    encoder.embed_query.return_value = "mock_embedding"
    encoder.model = MagicMock() # For AnswerVerifier
    return encoder

@pytest.fixture
def mock_index():
    index = MagicMock(spec=FaissIndex)
    index.chunks = [{"chunk_id": "1", "text": "Evidence text", "doc_id": "doc1", "page_start": 1, "page_end": 1}]
    index.search.return_value = [{"chunk_id": "1", "text": "Evidence text", "doc_id": "doc1", "page_start": 1, "page_end": 1, "similarity_score": 0.95}]
    return index

@pytest.fixture
def mock_reranker():
    reranker = MagicMock(spec=CrossEncoderReranker)
    reranker.rerank.return_value = [
        {"chunk_id": "1", "text": "Evidence text", "doc_id": "doc1", "page_start": 1, "page_end": 1, "cross_score": 0.9}
    ]
    return reranker

@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLMClient)
    llm.generate.return_value = "This is a generated answer based on [E1]."
    return llm

def test_run_query_pipeline_success(mock_encoder, mock_index, mock_reranker, mock_llm):
    # Mock internal dependencies of the pipeline if needed
    # (e.g., AnswerVerifier, CitationExtractor, which are instantiated inside)
    
    with patch("pipeline.query_pipeline.AnswerVerifier") as MockVerifier, \
         patch("pipeline.query_pipeline.CitationExtractor") as MockExtractor:
        
        # Setup mock verifier
        verifier_instance = MockVerifier.return_value
        verifier_instance.verify.return_value = {
            "verdict": "supported",
            "confidence": 1.0,
            "support_ratio": 1.0
        }
        
        # Setup mock extractor
        extractor_instance = MockExtractor.return_value
        extractor_instance.extract_and_map.return_value = {
            "inline_citations": [
                {"evidence_id": "E1", "doc_id": "doc1", "page_start": 1, "page_end": 1, "text_preview": "Evidence text"}
            ],
            "citation_coverage": 1.0,
            "uncited_sentences": []
        }
        
        result = run_query_pipeline(
            question="What is this?",
            encoder=mock_encoder,
            index=mock_index,
            reranker=mock_reranker,
            llm=mock_llm
        )
        
        assert result["verdict"] == "supported"
        assert len(result["citations"]) == 1
        assert result["confidence"] == 1.0

def test_run_query_pipeline_hard_guard_no_docs(mock_encoder, mock_index, mock_reranker, mock_llm):
    mock_index.chunks = [] # Simulate empty index
    
    result = run_query_pipeline(
        question="What is this?",
        encoder=mock_encoder,
        index=mock_index,
        reranker=mock_reranker,
        llm=mock_llm
    )
    
    assert result["verdict"] == "refused"
    assert "No documents" in result["answer"]
