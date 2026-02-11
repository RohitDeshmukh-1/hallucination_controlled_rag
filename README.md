# Hallucination-Controlled Academic RAG

**A production-ready, citation-verified, hallucination-controlled RAG system for academic and scientific question answering.**

---

## 🏠 Overview

Large Language Models (LLMs) frequently hallucinate, especially on knowledge-intensive academic questions. While RAG improves grounding, conventional pipelines still allow:

- ❌ Unsupported claims when context is weak
- ❌ Implicit reasoning beyond evidence  
- ❌ Loose or missing citations
- ❌ Overconfident answers with insufficient evidence

**This system** takes a stricter approach optimized for high-stakes academic domains:

| Feature | Description |
|---------|-------------|
| 📍 **Inline Citations** | Every factual claim includes `[E1]`, `[E2]` references to exact pages |
| 🔒 **Evidence-Only Generation** | Answers use *only* retrieved content, no world knowledge |
| 🚫 **Principled Abstention** | System refuses to answer when evidence is insufficient |
| 📊 **Confidence Scoring** | Quantified support ratio and citation coverage metrics |
| ✅ **Post-Generation Verification** | Sentence-level semantic similarity checking |

---

## ✨ Key Features

- **Dense Retrieval + Cross-Encoder Reranking**: Two-stage retrieval for high precision
- **Citation Extraction & Mapping**: Automatic extraction of `[E1]`, `[E2]` citations with page references
- **Confidence Metrics**: Evidence support score, citation coverage, and verification verdicts
- **Three Verdict Levels**: `supported`, `partially_supported`, `refused`
- **Production-Ready UI**: Professional Streamlit interface with real-time status
- **Modular Architecture**: Easily swap LLM providers, embeddings, or retrievers

---

## 📐 Architecture

<h3 align="center">Hallucination-Controlled Academic RAG System Architecture</h3>

<p align="center">
  <img src=r"assets/rag_architecture.png" width="100%">
</p>

<p align="center">
  <em>
  Figure: Offline index construction and online inference pipeline with retrieval gating, citation-aware generation, faithfulness verification, confidence scoring, and abstention mechanism.
  </em>
</p>

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hallucination-controlled-academic-rag.git
cd hallucination-controlled-academic-rag

# Create virtual environment
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate
# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
