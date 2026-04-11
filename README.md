# Hallucination-Controlled RAG for Scientific Documents

> **Live Demo → [hallucination-controlled-rag.vercel.app](https://hallucination-controlled-rag.vercel.app/)**

A production-grade Retrieval-Augmented Generation system engineered for academic and scientific research — where a hallucinated citation isn't just wrong, it's harmful. This project goes beyond standard RAG by building a **multi-stage verification pipeline** that cross-validates every generated sentence against retrieved evidence before a single word reaches the user.

---

## The Problem with Standard RAG

Most RAG systems retrieve documents and pass them to an LLM with a vague instruction to "only use the context." The LLM still hallucinates — it interpolates, confabulates sources, and blends retrieved facts with parametric memory. In scientific workflows, this is unacceptable.

**This project treats hallucination as an engineering problem, not a prompting problem.**

---

## Architecture

The system is a five-stage pipeline where each stage acts as a filter:

```
PDF Upload
   │
   ▼
[Ingestion] ──── Text Extraction → Semantic Chunking → FAISS Dense Index
                                                              │
User Question                                                 │
   │                                                          ▼
   ▼                                                   [Retrieval]
Contextual Query Rewriting ──────────────────────► Dense Vector Search
(expands ambiguous queries using conversation history)        │
                                                              ▼
                                                     Cross-Encoder Reranking
                                                              │
                                                              ▼
                                                    [Generation]
                                        Prompt = Retrieved Chunks
                                                  + Memory Pins
                                                  + Conversation History
                                                              │
                                                              ▼
                                                       LLM Inference
                                                              │
                                                              ▼
                                                    [Verification]
                                          Sentence-Level NLI Entailment Check
                                          Citation Mapping → [E1], [E2] markers
                                          Faithfulness Score + Support Ratio
                                                              │
                                                              ▼
                                                 Verified Response Delivered
```

### Why Each Stage Matters

| Stage | What it solves |
|---|---|
| **Semantic Chunking** | Avoids splitting mid-argument; preserves reasoning units for better retrieval precision |
| **Contextual Query Rewriting** | Resolves pronoun references and follow-up questions before vector search — a common silent failure mode in multi-turn RAG |
| **Cross-Encoder Reranking** | Bi-encoder retrieval optimizes for similarity, not relevance. Cross-encoders compare query↔passage jointly for higher precision |
| **NLI Entailment Gate** | Each generated sentence is independently checked: does the retrieved context *entail* this claim? Sentences that fail are flagged or suppressed |
| **Citation Mapping** | Every claim is anchored to a specific document page — enabling external auditability, not just LLM confidence scores |

---

## Key Engineering Decisions

**Evidence-Only Grounding** — The prompt architecture is designed to suppress parametric memory. The LLM is constrained to reason only over retrieved context, not its pretraining knowledge. This is enforced structurally in prompt construction, not via instruction alone.

**Sentence-Level Verification, Not Response-Level** — Most hallucination-detection approaches score a response as a whole. This system checks every sentence independently against the evidence pool. A mostly-correct response with one hallucinated claim still gets that claim flagged.

**Memory Pins** — Research sessions are iterative. Users can pin verified insights from prior answers into persistent session context, preventing the model from contradicting established facts in follow-up queries.

**Confidence Metrics Exposed to User** — Faithfulness score, support ratio, and citation coverage are surfaced in the UI. This shifts the system from black-box inference to transparent, auditable reasoning — critical for scientific workflows.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | React 19 + Vite |
| **Backend** | FastAPI (Python 3.10+) |
| **Vector Store** | FAISS (dense retrieval) |
| **Reranking** | Cross-Encoder (sentence-transformers) |
| **NLI Verification** | Hugging Face NLI models |
| **LLM Inference** | Gemini / OpenAI (configurable) |
| **Observability** | LangSmith + StructLog |
| **CI/CD** | GitHub Actions + Docker |

---

## Evaluation

The pipeline includes a faithfulness evaluation suite that measures outputs against ground-truth evidence, not just LLM self-assessment:

```bash
# Run integration tests for memory and multi-turn context
pytest tests/test_memory_integration.py -v

# Full pipeline faithfulness evaluation
python -m evaluation.run_evaluation
```

Metrics tracked:
- **Faithfulness Score** — fraction of generated claims entailed by retrieved context
- **Support Ratio** — proportion of sentences with at least one valid citation anchor
- **Citation Coverage** — percentage of claims mapped to specific document pages

---

## Running Locally

**Prerequisites:** Python 3.10+, Node.js 20+

```bash
# Clone and install backend dependencies
pip install -r requirements.txt

# Install frontend
cd webapp && npm install

# Configure environment
echo "LLM_API_KEY=your_key_here" > .env

# Start backend
uvicorn api.app:app --reload --port 8000

# Start frontend (new terminal)
cd webapp && npm run dev
```

**Or with Docker:**
```bash
docker-compose up --build
```

---

## Deployment

The system is split across two hosting environments:

**Backend** — Deployed to Hugging Face Spaces as a Docker-based FastAPI service:
```bash
git remote add hf https://huggingface.co/spaces/<hf_username>/<space_name>
git push hf main
```

Required Space secrets: `LLM_API_KEY`. Optional: `LLM_API_BASE`, `LLM_MODEL`, `LANGCHAIN_API_KEY`.

**Frontend** — Deployed to Vercel. Set `VITE_API_URL=https://<space_name>.hf.space` in Vercel environment variables.

**CI/CD** — GitHub Actions runs Black formatting, Flake8 linting, and pytest on every push. Docker multi-stage builds validate container integrity before deployment.

---

## Why This Project

Standard RAG is a solved problem. Hallucination-controlled RAG at the sentence level, with NLI verification, citation anchoring, and multi-turn memory — in a production-deployed, fully evaluated system — is not.

This project was built to answer a specific question: *how do you make LLM outputs trustworthy enough for a researcher to stake their work on?*

The answer is: you don't rely on the LLM to self-correct. You build the verification layer externally, make it inspectable, and expose the confidence mechanics to the user.

---

**Live Demo → [hallucination-controlled-rag.vercel.app](https://hallucination-controlled-rag.vercel.app/)**