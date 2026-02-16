import streamlit as st
import requests
import os

BACKEND_URL = os.getenv("API_URL", "http://localhost:8000")

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

# Sidebar — Index Management
with st.sidebar:
    st.header("Index Management")

    # Show current index status
    try:
        status_resp = requests.get(f"{BACKEND_URL}/index/status", timeout=5)
        if status_resp.ok:
            status = status_resp.json()
            st.metric("Indexed Chunks", status["chunk_count"])
            st.metric("Documents", status["document_count"])
        else:
            st.warning("Could not fetch index status.")
    except Exception:
        st.warning("Backend not reachable.")

    if st.button("🗑️ Clear Index", type="secondary"):
        try:
            resp = requests.delete(f"{BACKEND_URL}/index", timeout=10)
            if resp.ok:
                st.success("Index cleared!")
                st.session_state.doc_uploaded = False
                st.session_state.last_answer = None
                st.rerun()
            else:
                st.error("Failed to clear index.")
        except Exception as e:
            st.error(f"Error: {e}")

# Session state
if "doc_uploaded" not in st.session_state:
    st.session_state.doc_uploaded = False

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

# Upload
st.header("1️⃣ Upload a PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
)

if uploaded_file:
    if st.button("📤 Upload & Index", type="primary"):
        with st.spinner("Indexing document (this may take a moment)..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/upload",
                    files={"file": (uploaded_file.name, uploaded_file, "application/pdf")},
                    timeout=300,
                )
                resp.raise_for_status()
                data = resp.json()

                st.success(
                    f"✅ **{uploaded_file.name}** indexed — "
                    f"{data.get('chunk_count', '?')} chunks created."
                )
                st.session_state.doc_uploaded = True
                st.session_state.last_answer = None

            except Exception as e:
                st.error(f"Upload failed: {e}")

# Question
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
        with st.spinner("Generating answer..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={"question": question},
                    timeout=300,
                )
                resp.raise_for_status()
                st.session_state.last_answer = resp.json()
            except Exception as e:
                st.error(f"Query failed: {e}")

# Display answer
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
