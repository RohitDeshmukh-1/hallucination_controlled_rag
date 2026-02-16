from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import shutil
import uuid
import logging

from pipeline.ingest_document import ingest_document
from pipeline.query_pipeline import run_query_pipeline
from api import dependencies
from configs.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load models at startup to avoid first-request latency."""
    logger.info("Application starting up — pre-loading models...")
    dependencies.get_encoder()
    dependencies.get_index()
    dependencies.get_reranker()
    dependencies.get_llm_client()
    logger.info("All models loaded.")
    yield
    logger.info("Application shutting down.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS — allow frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure upload directory exists
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# Request / Response models

class QuestionRequest(BaseModel):
    question: str


# Routes

@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    """Upload and index a PDF document. Clears previous index first."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_id = f"{uuid.uuid4().hex}.pdf"
    file_path = settings.UPLOAD_DIR / file_id

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        logger.error("File save error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to save file")

    try:
        encoder = dependencies.get_encoder()
        index = dependencies.get_index()

        # Clear stale index data so new uploads always start fresh
        index.clear()
        logger.info("Index cleared before ingesting new document.")

        doc_id = ingest_document(file_path, encoder, index)
    except Exception as e:
        logger.error("Ingestion error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to ingest document")

    return {
        "status": "indexed",
        "doc_id": doc_id,
        "filename": file.filename,
        "chunk_count": index.chunk_count,
        "document_count": index.document_count,
    }


@app.post("/ask")
def ask_question(req: QuestionRequest):
    """Ask a question against the indexed documents."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        encoder = dependencies.get_encoder()
        index = dependencies.get_index()
        reranker = dependencies.get_reranker()
        llm = dependencies.get_llm_client()

        return run_query_pipeline(req.question, encoder, index, reranker, llm)
    except Exception as e:
        logger.error("Query error: %s", e, exc_info=True)
        return {
            "answer": "I cannot answer due to an internal error.",
            "verdict": "refused",
            "citations": [],
        }


@app.delete("/index")
def clear_index():
    """Clear the entire vector index (removes all indexed documents)."""
    try:
        dependencies.clear_index()
        return {"status": "cleared", "message": "Index cleared successfully."}
    except Exception as e:
        logger.error("Clear index error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to clear index")


@app.get("/index/status")
def index_status():
    """Get the current status of the vector index."""
    index = dependencies.get_index()
    return {
        "chunk_count": index.chunk_count,
        "document_count": index.document_count,
        "doc_ids": list(index.get_doc_ids()),
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}
