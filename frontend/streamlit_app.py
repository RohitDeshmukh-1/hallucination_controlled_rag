import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Hallucination-Controlled Academic RAG",
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

# -----------------------------
# Session state
# -----------------------------

if "doc_uploaded" not in st.session_state:
    st.session_state.doc_uploaded = False

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None


# -----------------------------
# Upload
# -----------------------------

st.header("1️⃣ Upload a PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
)

if uploaded_file and not st.session_state.doc_uploaded:
    with st.spinner("Indexing document..."):
        try:
            resp = requests.post(
                f"{BACKEND_URL}/upload",
                files={"file": (uploaded_file.name, uploaded_file, "application/pdf")},
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()

            st.success(f"Document indexed successfully (doc_id: {data['doc_id']})")
            st.session_state.doc_uploaded = True

        except Exception as e:
            st.error(f"Upload failed: {e}")


# -----------------------------
# Question
# -----------------------------

st.header("2️⃣ Ask a Question")

question = st.text_input(
    "Enter your question",
    placeholder="Who are the authors of this paper?",
)

if st.button("Ask"):
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


# -----------------------------
# Display answer
# -----------------------------

if st.session_state.last_answer:
    result = st.session_state.last_answer

    st.subheader("Answer")

    if result["verdict"] == "refused":
        st.error(result["answer"])
    else:
        st.write(result["answer"])

        if "citations" in result:
            st.subheader("Citations")
            for c in result["citations"]:
                st.markdown(
                    f"- **Document:** `{c['doc_id']}`, **Pages:** {c['pages']}"
                )

st.markdown("---")
st.caption(
    "Evidence-only generation • Explicit citation • Principled abstention"
)
