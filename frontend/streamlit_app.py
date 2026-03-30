"""
Hallucination-Controlled Academic RAG — Streamlit App (Self-contained)

All backend logic is embedded directly in this file so the app can be
deployed as a single Streamlit application without a separate API server.
"""

import sys
import os
import uuid
import logging
import tempfile
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Logging — must be initialized BEFORE any code that uses `logger`.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that pipeline / configs / etc.
# are importable regardless of how Streamlit is launched.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Load secrets from Streamlit (for Cloud deployment where .env is missing)
# This must happen BEFORE importing settings!
# ---------------------------------------------------------------------------
try:
    # Some Streamlit environments might not expose st.secrets as dict-like immediately
    # or raise exceptions on access if no secrets.toml exists.
    if hasattr(st, "secrets"):
        # We only copy top-level string keys to simulate env vars
        # This allows `LLM_API_KEY="xyz"` in secrets.toml to work as expected.
        for key, value in st.secrets.items():
            if isinstance(value, str) and key not in os.environ:
                os.environ[key] = value
                logger.info("Loaded secret %s into os.environ", key)
except Exception:
    # It's normal if secrets aren't configured locally or if specific keys are missing.
    pass

from configs.settings import settings
from embeddings.encoder import EmbeddingEncoder
from retrieval.faiss_index import FaissIndex
from retrieval.cross_encoder import CrossEncoderReranker
from generation.llm_client import LLMClient
from pipeline.ingest_document import ingest_document
from pipeline.query_pipeline import run_query_pipeline
from utils.storage import storage_client

# (Logging is initialized above, before the secrets block.)

# ---------------------------------------------------------------------------
# Singleton resources — cached across reruns via st.cache_resource
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model…")
def get_encoder() -> EmbeddingEncoder:
    logger.info("Initializing EmbeddingEncoder…")
    return EmbeddingEncoder()


@st.cache_resource(show_spinner="Loading FAISS index…")
def get_index() -> FaissIndex:
    logger.info("Initializing FaissIndex…")
    idx = FaissIndex()
    idx.load_or_create()
    return idx


@st.cache_resource(show_spinner="Loading reranker model…")
def get_reranker() -> CrossEncoderReranker:
    logger.info("Initializing CrossEncoderReranker…")
    return CrossEncoderReranker()


@st.cache_resource(show_spinner="Connecting to LLM…")
def get_llm_client() -> LLMClient:
    logger.info("Initializing LLMClient…")
    return LLMClient()


# ---------------------------------------------------------------------------
# Helper functions (previously FastAPI route handlers)
# ---------------------------------------------------------------------------

def _upload_and_index(uploaded_file) -> dict:
    """Save the uploaded PDF to a temp path, ingest, and return status info."""
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_id = f"{uuid.uuid4().hex}.pdf"
    file_path = settings.UPLOAD_DIR / file_id

    # Write the uploaded bytes to disk
    file_path.write_bytes(uploaded_file.getvalue())

    # Upload to cloud if configured
    storage_client.upload_file(file_path, file_id)

    encoder = get_encoder()
    index = get_index()

    # REMOVED: index.clear() call to allow multiple papers in one index
    logger.info("Ingesting new document (multiple support enabled): %s", uploaded_file.name)

    doc_id = ingest_document(file_path, encoder, index)

    return {
        "status": "indexed",
        "doc_id": doc_id,
        "filename": uploaded_file.name,
        "chunk_count": index.chunk_count,
        "document_count": index.document_count,
    }


def _ask_question(question: str) -> dict:
    """Run the full RAG query pipeline and return the result dict."""
    encoder = get_encoder()
    index = get_index()
    reranker = get_reranker()
    llm = get_llm_client()
    return run_query_pipeline(question, encoder, index, reranker, llm)


def _clear_index():
    """Clear the FAISS index completely."""
    index = get_index()
    index.clear()
    logger.info("Index cleared via Streamlit UI.")


def _index_status() -> dict:
    """Return current index metrics."""
    index = get_index()
    return {
        "chunk_count": index.chunk_count,
        "document_count": index.document_count,
        "doc_ids": list(index.get_doc_ids()),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Hallucination-Controlled Academic RAG",
    page_icon="📄",
    layout="centered",
)

st.title("📄 Hallucination-Controlled Academic Q&A")

st.markdown(
    """
Upload academic papers (PDF) and ask questions.
Answers are generated **only** from uploaded documents.
If evidence is insufficient, the system explicitly refuses to answer.
"""
)

# ── Sidebar — Index Management ────────────────────────────────────────────
with st.sidebar:
    st.header("Index Management")

    try:
        status = _index_status()
        st.metric("Indexed Chunks", status["chunk_count"])
        st.metric("Documents", status["document_count"])
        
        if status["document_count"] > 0:
            st.write("---")
            st.write("**Indexed Documents:**")
            for doc_id in status.get("doc_ids", []):
                st.caption(f"📄 `{doc_id[:12]}...`")
    except Exception:
        st.warning("Could not fetch index status.")

    if st.button("🗑️ Clear Index", type="secondary"):
        try:
            _clear_index()
            st.success("Index cleared!")
            st.session_state.doc_uploaded = False
            st.session_state.last_answer = None
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

# ── Session state ─────────────────────────────────────────────────────────
if "doc_uploaded" not in st.session_state:
    st.session_state.doc_uploaded = False

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

# ── 1. Upload ─────────────────────────────────────────────────────────────
st.header("1️⃣ Upload a PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
)

if uploaded_file:
    if st.button("📤 Upload & Index", type="primary"):
        with st.spinner("Indexing document (this may take a moment)…"):
            try:
                data = _upload_and_index(uploaded_file)
                st.success(
                    f"✅ **{uploaded_file.name}** indexed — "
                    f"{data.get('chunk_count', '?')} chunks created."
                )
                st.session_state.doc_uploaded = True
                st.session_state.last_answer = None
            except Exception as e:
                logger.error("Upload failed: %s", e, exc_info=True)
                st.error(f"Upload failed: {e}")

# ── 2. Ask a Question ────────────────────────────────────────────────────
st.header("2️⃣ Ask a Question")

question = st.text_input(
    "Enter your question",
    placeholder="What is the main contribution of this paper?",
)

if st.button("Ask", type="primary"):
    if not st.session_state.doc_uploaded:
        st.warning("Please upload a document first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer…"):
            try:
                st.session_state.last_answer = _ask_question(question)
            except Exception as e:
                logger.error("Query failed: %s", e, exc_info=True)
                st.error(f"Query failed: {e}")

# ── Display Answer ────────────────────────────────────────────────────────
if st.session_state.last_answer:
    result = st.session_state.last_answer

    st.subheader("Answer")

    verdict = result.get("verdict", "unknown")

    if verdict == "refused":
        st.error(result["answer"])
    else:
        # Verdict badge
        if verdict == "supported":
            st.success(f"✅ Verdict: **Supported** (confidence: {result.get('confidence', 0):.0%})")
        elif verdict == "partially_supported":
            st.warning(f"⚠️ Verdict: **Partially Supported** (confidence: {result.get('confidence', 0):.0%})")

        st.markdown(result["answer"])

        # Citations
        citations = result.get("citations", [])
        if citations:
            st.subheader("📚 Citations")
            for c in citations:
                with st.expander(f"[{c['evidence_id']}] Document `{c['doc_id']}` — Pages {c['pages']}"):
                    st.markdown(f"*{c.get('text_preview', 'No preview available.')}*")

        # Coverage metrics
        coverage = result.get("citation_coverage")
        if coverage is not None:
            st.caption(f"Citation coverage: {coverage:.0%}")

st.markdown("---")
st.caption(
    "Evidence-only generation • Explicit citation • Principled abstention"
)
