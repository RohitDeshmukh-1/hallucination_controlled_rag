# Hallucination-Controlled Academic RAG

**A faithful, citation-rigorous, and abstention-aware Retrieval-Augmented Generation framework for academic / scientific question answering**

## 📄 Overview

Despite impressive generative capabilities, **Large Language Models (LLMs)** frequently hallucinate — especially on knowledge-intensive academic and scientific questions.

While **Retrieval-Augmented Generation (RAG)** greatly improves factual grounding by conditioning generation on retrieved documents, conventional RAG pipelines still allow:

- unsupported claims when retrieved context is weak/ambiguous
- implicit reasoning beyond provided evidence
- loose or post-hoc citation behavior
- overconfident answering even when evidence is clearly insufficient

**Hallucination-Controlled Academic RAG** introduces a stricter, more principled approach optimized for high-stakes academic & scientific domains.

### Core Design Principles

1. **Page-level mandatory citations**  
   Every factual statement must be explicitly tied to one or more exact page references from retrieved documents.

2. **Evidence-only generation**  
   The model is constrained to generate answers **exclusively** from retrieved & re-ranked evidence (no free-form world knowledge injection).

3. **Principled abstention**  
   When retrieved evidence is insufficient, ambiguous, or does not allow a high-confidence answer → the system **explicitly refuses to answer** instead of hallucinating or guessing.

The central goal shifts from **maximizing answer completeness** → **maximizing verifiable faithfulness**.

## ✨ Key Features

- Dense retrieval + cross-encoder re-ranking pipeline
- Strict page-level citation enforcement during generation
- Controlled decoding strategies that prevent unsupported claims
- Automatic abstention detection & generation of refusal responses
- Designed & evaluated primarily on academic/scientific QA datasets
- Modular architecture (easy to swap retriever, reranker, LLM backend)

## Project Status (early 2026)

- Working prototype implementation
- Preliminary experiments on scientific QA benchmarks
- Focus on faithfulness & citation quality over ROUGE/BLEU-style overlap

## Installation

```bash
# clone the repository
git clone https://github.com/yourusername/hallucination-controlled-academic-rag.git
cd hallucination-controlled-academic-rag

# recommended: use uv / pipx / conda
python -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
# or with uv (faster)
uv pip install -r requirements.txt