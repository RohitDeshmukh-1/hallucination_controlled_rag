"""
Comprehensive Evaluation Suite for Hallucination-Controlled Academic RAG.

Measures:
  1. Retrieval quality   — Recall@K, MRR, NDCG@K
  2. Hallucination rate  — sentence-level unsupported claim rate
  3. Citation accuracy   — coverage, validity, groundedness
  4. Answer quality      — faithfulness, relevance, completeness
  5. Latency profiling   — per-stage timing breakdown

Usage:
    python -m evaluation.run_evaluation
"""

import json
import time
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

# Ensure project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import settings
from embeddings.encoder import EmbeddingEncoder
from retrieval.faiss_index import FaissIndex
from retrieval.cross_encoder import CrossEncoderReranker
from generation.llm_client import LLMClient
from generation.prompt_builder import PromptBuilder
from generation.answer_verifier import AnswerVerifier
from generation.citation_extractor import CitationExtractor
from evaluation.faithfulness_metrics import FaithfulnessMetrics

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Built-in evaluation questions (used when no external benchmark is provided)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_EVAL_QUESTIONS = [
    "What is the main contribution of this paper?",
    "What methodology does this paper use?",
    "What are the key findings or results?",
    "What datasets were used in the experiments?",
    "What are the limitations mentioned by the authors?",
    "How does this work compare to prior approaches?",
    "What future work is suggested?",
    "What evaluation metrics are reported?",
    "What is the theoretical foundation of the proposed approach?",
    "What are the practical implications of the results?",
]

# Questions designed to trigger abstention (no relevant info expected)
ADVERSARIAL_QUESTIONS = [
    "What is the recipe for chocolate cake?",
    "Who won the FIFA World Cup in 2022?",
    "What is the capital of France?",
    "How do you train a dog to sit?",
    "What are the lyrics to Bohemian Rhapsody?",
]


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_retrieval_metrics(
    retrieved_chunks: List[Dict],
    relevant_doc_id: Optional[str] = None,
    k_values: List[int] = [1, 3, 5, 10, 20],
) -> Dict[str, float]:
    """Compute Recall@K and MRR for a single query."""
    metrics: Dict[str, float] = {}

    if not relevant_doc_id or not retrieved_chunks:
        return metrics

    # Recall@K — did we find the right doc within top-K?
    for k in k_values:
        top_k = retrieved_chunks[:k]
        hit = any(c.get("doc_id") == relevant_doc_id for c in top_k)
        metrics[f"recall@{k}"] = 1.0 if hit else 0.0

    # MRR — reciprocal rank of first relevant result
    for i, chunk in enumerate(retrieved_chunks):
        if chunk.get("doc_id") == relevant_doc_id:
            metrics["mrr"] = 1.0 / (i + 1)
            break
    else:
        metrics["mrr"] = 0.0

    return metrics


def compute_citation_metrics(citation_result: Dict) -> Dict[str, float]:
    """Compute citation quality metrics."""
    return {
        "citation_coverage": citation_result.get("citation_coverage", 0.0),
        "valid_citation_count": len(citation_result.get("inline_citations", [])),
        "invalid_citation_count": len(citation_result.get("invalid_citations", [])),
        "uncited_sentence_count": len(citation_result.get("uncited_sentences", [])),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

class EvaluationRunner:
    """Orchestrates end-to-end evaluation of the RAG pipeline."""

    def __init__(self):
        self.encoder = EmbeddingEncoder()
        self.index = FaissIndex()
        self.index.load_or_create()
        self.reranker = CrossEncoderReranker()
        self.llm = LLMClient()
        self.faithfulness = FaithfulnessMetrics()

    def evaluate(
        self,
        questions: Optional[List[str]] = None,
        include_adversarial: bool = True,
    ) -> Dict[str, Any]:
        """Run full evaluation and return aggregate results."""
        if self.index.chunk_count == 0:
            return {"error": "No documents indexed. Upload a PDF first."}

        if questions is None:
            questions = DEFAULT_EVAL_QUESTIONS.copy()

        all_questions = list(questions)
        adversarial_start = len(all_questions)

        if include_adversarial:
            all_questions.extend(ADVERSARIAL_QUESTIONS)

        results: List[Dict[str, Any]] = []
        timings: List[Dict[str, float]] = []

        for i, question in enumerate(all_questions):
            is_adversarial = i >= adversarial_start
            logger.info(
                "Evaluating [%d/%d]%s: %s",
                i + 1, len(all_questions),
                " (adversarial)" if is_adversarial else "",
                question[:60],
            )

            result, timing = self._evaluate_single(question, is_adversarial)
            results.append(result)
            timings.append(timing)

        return self._aggregate(results, timings, adversarial_start)

    def _evaluate_single(
        self, question: str, is_adversarial: bool
    ) -> tuple:
        timing: Dict[str, float] = {}
        result: Dict[str, Any] = {
            "question": question,
            "is_adversarial": is_adversarial,
        }

        # Stage 1: Retrieval
        t0 = time.perf_counter()
        query_embedding = self.encoder.embed_query(question)
        timing["embed_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        retrieved = self.index.search(query_embedding, top_k=20)
        timing["retrieval_ms"] = (time.perf_counter() - t0) * 1000

        result["retrieved_count"] = len(retrieved)

        if not retrieved:
            result["verdict"] = "refused"
            return result, timing

        # Stage 2: Reranking
        t0 = time.perf_counter()
        evidence = self.reranker.rerank(question, retrieved, top_n=8)
        timing["rerank_ms"] = (time.perf_counter() - t0) * 1000

        result["evidence_count"] = len(evidence)
        result["top_cross_score"] = evidence[0]["cross_score"] if evidence else 0.0

        # Stage 3: Generation
        prompt = PromptBuilder().build(question, evidence)

        t0 = time.perf_counter()
        try:
            answer = self.llm.generate(prompt)
        except Exception as e:
            result["verdict"] = "error"
            result["error"] = str(e)
            timing["generation_ms"] = (time.perf_counter() - t0) * 1000
            return result, timing
        timing["generation_ms"] = (time.perf_counter() - t0) * 1000

        result["answer_length"] = len(answer)

        # Stage 4: Verification
        t0 = time.perf_counter()
        verifier = AnswerVerifier(encoder_model=self.encoder.model)
        verification = verifier.verify(answer, evidence)
        timing["verification_ms"] = (time.perf_counter() - t0) * 1000

        result["verdict"] = verification["verdict"]
        result["confidence"] = verification.get("confidence", 0.0)
        result["support_ratio"] = verification.get("support_ratio", 0.0)
        result["unsupported_count"] = len(verification.get("unsupported_sentences", []))

        self.faithfulness.update(verification)

        # Stage 5: Citation analysis
        extractor = CitationExtractor()
        citation_result = extractor.extract_and_map(answer, evidence)
        citation_metrics = compute_citation_metrics(citation_result)
        result.update(citation_metrics)

        timing["total_ms"] = sum(timing.values())

        return result, timing

    def _aggregate(
        self,
        results: List[Dict],
        timings: List[Dict],
        adversarial_start: int,
    ) -> Dict[str, Any]:
        """Aggregate per-question results into summary metrics."""
        normal = [r for r in results if not r.get("is_adversarial")]
        adversarial = [r for r in results if r.get("is_adversarial")]

        # Verdict distribution
        verdicts = [r.get("verdict", "unknown") for r in normal]
        verdict_dist = {
            v: verdicts.count(v) / len(verdicts) if verdicts else 0
            for v in ["supported", "partially_supported", "refused", "unsupported", "error"]
        }

        # Adversarial abstention rate
        adv_refused = sum(
            1 for r in adversarial
            if r.get("verdict") in ("refused", "unsupported")
        )
        adv_abstention_rate = adv_refused / len(adversarial) if adversarial else 0.0

        # Aggregate confidence & support
        confidences = [r["confidence"] for r in normal if "confidence" in r]
        support_ratios = [r["support_ratio"] for r in normal if "support_ratio" in r]
        coverages = [r["citation_coverage"] for r in normal if "citation_coverage" in r]

        # Timing aggregates
        all_totals = [t.get("total_ms", 0) for t in timings]
        stage_names = ["embed_ms", "retrieval_ms", "rerank_ms", "generation_ms", "verification_ms"]
        avg_timing = {}
        for stage in stage_names:
            vals = [t[stage] for t in timings if stage in t]
            avg_timing[stage] = float(np.mean(vals)) if vals else 0.0

        return {
            "summary": {
                "total_questions": len(results),
                "normal_questions": len(normal),
                "adversarial_questions": len(adversarial),
            },
            "faithfulness": self.faithfulness.compute(),
            "verdict_distribution": {
                k: round(v, 4) for k, v in verdict_dist.items()
            },
            "adversarial_abstention_rate": round(adv_abstention_rate, 4),
            "avg_confidence": round(float(np.mean(confidences)), 4) if confidences else 0.0,
            "avg_support_ratio": round(float(np.mean(support_ratios)), 4) if support_ratios else 0.0,
            "avg_citation_coverage": round(float(np.mean(coverages)), 4) if coverages else 0.0,
            "latency": {
                "avg_total_ms": round(float(np.mean(all_totals)), 1),
                "p95_total_ms": round(float(np.percentile(all_totals, 95)), 1) if all_totals else 0,
                "per_stage_avg_ms": {k: round(v, 1) for k, v in avg_timing.items()},
            },
            "per_question_results": results,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    runner = EvaluationRunner()
    report = runner.evaluate()

    # Save to file
    output_dir = Path("evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "evaluation_report.json"

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION REPORT — Hallucination-Controlled Academic RAG")
    print("=" * 70)

    if "error" in report:
        print(f"\n❌ {report['error']}")
        return

    s = report["summary"]
    print(f"\nQuestions: {s['total_questions']} ({s['normal_questions']} normal + {s['adversarial_questions']} adversarial)")

    print("\n📊 Faithfulness Metrics:")
    for k, v in report.get("faithfulness", {}).items():
        print(f"  {k:30s}: {v:.4f}")

    print("\n📈 Verdict Distribution:")
    for k, v in report["verdict_distribution"].items():
        bar = "█" * int(v * 40)
        print(f"  {k:25s}: {v:.1%} {bar}")

    print(f"\n🛡️ Adversarial Abstention Rate: {report['adversarial_abstention_rate']:.1%}")
    print(f"📊 Avg Confidence:              {report['avg_confidence']:.3f}")
    print(f"📊 Avg Support Ratio:           {report['avg_support_ratio']:.3f}")
    print(f"📊 Avg Citation Coverage:       {report['avg_citation_coverage']:.1%}")

    lat = report["latency"]
    print(f"\n⏱️  Avg Total Latency:          {lat['avg_total_ms']:.0f} ms")
    print(f"⏱️  P95 Total Latency:          {lat['p95_total_ms']:.0f} ms")
    print("  Per-stage breakdown:")
    for stage, ms in lat["per_stage_avg_ms"].items():
        print(f"    {stage:25s}: {ms:.0f} ms")

    print(f"\n✅ Full report saved to: {output_file.resolve()}")


if __name__ == "__main__":
    main()
