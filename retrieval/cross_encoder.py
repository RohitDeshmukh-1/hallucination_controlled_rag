from typing import List, Dict

import numpy as np
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Cross-encoder re-ranking module for high-precision evidence selection.

    This component refines the output of dense bi-encoder retrieval by
    jointly encoding (query, passage) pairs and scoring their relevance.
    It is designed to operate on a small candidate set (e.g., top-20)
    retrieved by FAISS.

    The reranker improves precision without sacrificing recall and is
    critical for reducing downstream hallucinations in generation.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_passages: int = 20,
    ):
        """
        Parameters
        ----------
        model_name : str
            HuggingFace identifier for the cross-encoder model.
        max_passages : int
            Maximum number of retrieved passages to re-rank.
        """
        self.model = CrossEncoder(model_name)
        self.max_passages = max_passages

    def rerank(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        top_n: int = 5,
    ) -> List[Dict]:
        """
        Re-rank retrieved chunks using cross-encoder relevance scores.

        Parameters
        ----------
        query : str
            User question.
        retrieved_chunks : List[Dict]
            Chunks retrieved by FAISS, each containing at least a 'text' field.
        top_n : int
            Number of top-ranked chunks to return.

        Returns
        -------
        List[Dict]
            Re-ranked chunks ordered by descending relevance score.
            Each chunk includes an added 'cross_score' field.
        """

        if not retrieved_chunks:
            return []

        candidates = retrieved_chunks[: self.max_passages]

        query_passage_pairs = [
            (query, chunk["text"]) for chunk in candidates
        ]

        scores = self.model.predict(query_passage_pairs)

        scored_chunks = []
        for chunk, score in zip(candidates, scores):
            enriched = dict(chunk)
            enriched["cross_score"] = float(score)
            scored_chunks.append(enriched)

        scored_chunks.sort(
            key=lambda x: x["cross_score"], reverse=True
        )

        return scored_chunks[:top_n]
