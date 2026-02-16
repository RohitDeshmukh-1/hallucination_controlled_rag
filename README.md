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
- **Dockerized Deployment**: Includes `Dockerfile` and `docker-compose.yml` for easy setup.

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

**Start the Backend API:**
```bash
uvicorn api.app:app --reload --port 8000
```

**Start the Frontend UI:**
```bash
streamlit run frontend/streamlit_app.py
```

Access the UI at `http://localhost:8501`.

### 3. Running with Docker

Ensure Docker Desktop is running.

```bash
docker-compose up --build
```
This will start both the backend (port 8000) and frontend (port 8501).

---

## 🛠️ Configuration

Configuration is managed via `configs/settings.py` and environment variables. Key settings:

- `EMBEDDING_MODEL_NAME`: Default `sentence-transformers/all-MiniLM-L6-v2`
- `RERANKER_MODEL_NAME`: Default `cross-encoder/ms-marco-MiniLM-L-6-v2`
- `LLM_TEMPERATURE`: Default `0.0` (for deterministic output)
- `UPLOAD_DIR`: Where PDFs are stored (`storage/uploads`)
- `INDEX_DIR`: Where FAISS index is stored (`storage/index`)

---

## 🧪 Testing

Run unit tests to verify the pipeline logic:

```bash
pytest tests/
```

---

## 📐 Architecture

### Hallucination-Controlled Academic RAG System Architecture

![Hallucination-Controlled Academic RAG Architecture](./assets/rag_architecture.png)

1.  **Ingestion**: PDFs are chunked semantically based on sentence boundaries and similarity.
2.  **Retrieval**: Bi-encoder retrieval (FAISS) fetches top-20 candidates.
3.  **Reranking**: Cross-encoder reranks top candidates for precision.
4.  **Generation**: LLM generates answer using *only* the provided context.
5.  **Verification**: Answer is split into sentences and verified against evidence.
6.  **Citation**: Citations are mapped back to original PDF pages.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

[MIT](https://choosealicense.com/licenses/mit/)
