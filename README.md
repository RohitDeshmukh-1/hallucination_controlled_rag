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
| 🧠 **NLI-Based Verification** | Optional Natural Language Inference for research-grade hallucination detection |

---

## ✨ Key Features

- **Dense Retrieval + Cross-Encoder Reranking**: Two-stage retrieval for high precision
- **Citation Extraction & Mapping**: Automatic extraction of `[E1]`, `[E2]` citations with page references
- **Dual-Layer Verification**: Cosine-similarity + NLI-based entailment checking
- **Confidence Metrics**: Evidence support score, citation coverage, and verification verdicts
- **Three Verdict Levels**: `supported`, `partially_supported`, `refused`
- **Adversarial Abstention**: Tested against out-of-domain questions to ensure proper refusal
- **Production-Ready UI**: Professional Streamlit interface with real-time status
- **Modular Architecture**: Easily swap LLM providers, embeddings, or retrievers
- **Comprehensive Evaluation**: Research-grade evaluation suite with faithfulness metrics
- **Free-Tier Deployment**: Ready for Streamlit Cloud, Render.com, or Docker

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (optional)
- An API Key for an OpenAI-compatible LLM provider (e.g., Groq, OpenAI).

### 1. Installation (Local)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/hallucination-controlled-academic-rag.git
    cd hallucination-controlled-academic-rag
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Mac/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment:**
    Copy `.env.example` to `.env` and fill in your API keys.
    ```bash
    cp .env.example .env
    ```
    **Required Variables:**
    - `LLM_API_KEY`: Your API key (e.g., Groq or OpenAI key)
    - `LLM_API_BASE`: API Base URL (default: `https://api.groq.com/openai/v1`)
    - `LLM_MODEL`: Model name (default: `llama3-70b-8192`)

### 2. Running the Application

**Option A: Self-Contained Streamlit (Recommended)**
```bash
streamlit run frontend/streamlit_app.py
```
Access at `http://localhost:8501`. No separate backend needed.

**Option B: Separate Backend + Frontend**
```bash
# Terminal 1: Start the Backend API
uvicorn api.app:app --reload --port 8000

# Terminal 2: Start the Frontend UI
streamlit run frontend/streamlit_app.py
```

### 3. Running with Docker

```bash
docker-compose up --build
```
This starts both backend (port 8000) and frontend (port 8501).

---

## 🌐 Free-Tier Deployment

### Option 1: Streamlit Community Cloud (Recommended — Easiest)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set **Main file path** to `frontend/streamlit_app.py`
5. Add your secrets in the Streamlit dashboard:
   - Click **Advanced settings** → **Secrets**
   - Add:
     ```toml
     LLM_API_KEY = "your_groq_api_key_here"
     LLM_API_BASE = "https://api.groq.com/openai/v1"
     LLM_MODEL = "llama3-70b-8192"
     ```
6. Deploy!

### Option 2: Render.com (Free Tier)

1. Push to GitHub
2. Go to [render.com](https://render.com) and create a new **Web Service**
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` for configuration
5. Add `LLM_API_KEY` in the Render dashboard under **Environment Variables**
6. Deploy!

### Option 3: Docker (Any VPS / Railway / Fly.io)

```bash
docker build -t rag-app .
docker run -p 8000:8000 --env-file .env rag-app
```

---

## ☁️ Cloud Storage (Optional)

This system supports **Supabase Storage** for persisting PDFs in the cloud:
1. Create a [Supabase](https://supabase.com) project.
2. Go to **Storage**, create a new bucket named `papers`.
3. Set the bucket to **Public** if you want easy access, or private with proper policies.
4. Add the following to your environment variables or Streamlit secrets:
   - `SUPABASE_URL`: Your project URL
   - `SUPABASE_KEY`: Your project anon/service_role key
   - `SUPABASE_BUCKET`: `papers` (default)

---

## 🧪 Testing

Run the comprehensive test suite:

```bash
pytest tests/ -v
```

### Running Evaluation

After uploading at least one PDF, run the full evaluation:

```bash
python -m evaluation.run_evaluation
```

This generates a JSON report at `evaluation/results/evaluation_report.json` with:
- Faithfulness metrics (sentence support rate, refusal rate)
- Verdict distribution
- Adversarial abstention rate
- Latency profiling (per-stage breakdown)

---

## 🛠️ Configuration

Configuration is managed via `configs/settings.py` and environment variables. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Bi-encoder for embedding |
| `RERANKER_MODEL_NAME` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder for reranking |
| `LLM_TEMPERATURE` | `0.0` | Deterministic output |
| `VERIFICATION_SIMILARITY_THRESHOLD` | `0.55` | Min cosine similarity for support |
| `VERIFICATION_UNSUPPORTED_RATIO` | `0.5` | Max unsupported ratio before rejection |
| `UPLOAD_DIR` | `storage/uploads` | PDF storage location |
| `INDEX_DIR` | `storage/index` | FAISS index location |

---

## 📐 Architecture

### Hallucination-Controlled Academic RAG System Architecture

```
PDF Upload → Page Extraction → Text Cleaning → Semantic Chunking
                                                      ↓
                                              Embedding (MiniLM)
                                                      ↓
                                              FAISS Indexing
                                                      ↓
User Question → Query Embedding → Dense Retrieval (Top-20)
                                        ↓
                                  Cross-Encoder Reranking (Top-8)
                                        ↓
                                  Prompt Construction (citation-aware)
                                        ↓
                                  LLM Generation (Groq/OpenAI)
                                        ↓
                              ┌─────────┴──────────┐
                        Citation Extraction    Answer Verification
                        (inline [E1] mapping)  (cosine sim + optional NLI)
                              └─────────┬──────────┘
                                        ↓
                                  Verdict: supported | partially_supported | refused
```

1.  **Ingestion**: PDFs are chunked semantically based on sentence boundaries and similarity.
2.  **Retrieval**: Bi-encoder retrieval (FAISS) fetches top-20 candidates.
3.  **Reranking**: Cross-encoder reranks top candidates for precision.
4.  **Generation**: LLM generates answer using *only* the provided context.
5.  **Verification**: Answer is split into sentences and verified against evidence.
6.  **Citation**: Citations are mapped back to original PDF pages.

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Sentence Support Rate** | Fraction of substantive sentences supported by evidence |
| **Refusal Rate** | Fraction of questions where the system abstained |
| **Citation Coverage** | Fraction of substantive sentences with inline citations |
| **Adversarial Abstention** | Rate of correct refusal on out-of-domain questions |
| **Avg Confidence** | Mean cosine similarity between claims and evidence |
| **Latency Profile** | Per-stage timing (embed, retrieve, rerank, generate, verify) |

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

[MIT](https://choosealicense.com/licenses/mit/)
