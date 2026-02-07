from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pathlib import Path
import shutil
import uuid

from pipeline.ingest_document import ingest_document
from pipeline.query_pipeline import run_query_pipeline

app = FastAPI(
    title="Hallucination-Controlled Academic RAG",
    version="1.0.0",
)

UPLOAD_DIR = Path("storage/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Request models
# -----------------------------

class QuestionRequest(BaseModel):
    question: str


# -----------------------------
# Routes
# -----------------------------

@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_id = f"{uuid.uuid4().hex}.pdf"
    file_path = UPLOAD_DIR / file_id

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        doc_id = ingest_document(file_path)
    except Exception as e:
        print("UPLOAD ERROR:", e)
        raise HTTPException(status_code=500, detail="Failed to ingest document")

    return {
        "status": "indexed",
        "doc_id": doc_id,
        "filename": file.filename,
    }


@app.post("/ask")
def ask_question(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        return run_query_pipeline(req.question)
    except Exception as e:
        # IMPORTANT: never crash, always fail closed
        print("ASK ERROR:", e)
        return {
            "answer": "I cannot answer due to an internal error.",
            "verdict": "refused",
            "citations": [],
        }


@app.get("/health")
def health_check():
    return {"status": "ok"}
