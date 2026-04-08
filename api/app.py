from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import shutil
import uuid
import logging

from pipeline.ingest_document import ingest_document
from pipeline.query_pipeline import run_query_pipeline
from pipeline.conversation_memory import ConversationMemory, ConversationTurn
from api import dependencies
from configs.settings import settings
from utils.storage import storage_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# In-memory session store: session_id -> ConversationMemory
_sessions: dict[str, ConversationMemory] = {}


def _get_session(session_id: str) -> ConversationMemory:
    if session_id not in _sessions:
        _sessions[session_id] = ConversationMemory(session_name=f"Session {session_id[:6]}")
    return _sessions[session_id]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up - pre-loading models...")
    dependencies.get_encoder()
    dependencies.get_index()
    dependencies.get_reranker()
    dependencies.get_llm_client()
    logger.info("All models loaded.")
    yield
    logger.info("Application shutting down.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="2.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# -- CORS Configuration -------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# -- Request / Response Models ------------------------------------------------

class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"
    doc_id: Optional[str] = None
    enable_nli: bool = False
    use_memory_context: bool = True


class PinRequest(BaseModel):
    session_id: str
    text: str
    source_question: str
    from_doc: Optional[str] = None


class SessionRequest(BaseModel):
    session_name: Optional[str] = "New Session"


# -- Upload -------------------------------------------------------------------

@app.post("/upload")
def upload_pdf(file: UploadFile = File(...), session_id: str = "default"):
    """Upload and index a PDF document."""
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
        storage_client.upload_file(file_path, file_id)
        encoder = dependencies.get_encoder()
        index = dependencies.get_index()
        logger.info("Ingesting new document: %s", file.filename)
        doc_id = ingest_document(file_path, encoder, index)
    except Exception as e:
        logger.error("Ingestion error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to ingest document")

    # Register doc into session memory
    session = _get_session(session_id)
    session.register_document(doc_id, file.filename, index.chunk_count)

    return {
        "status": "indexed",
        "doc_id": doc_id,
        "filename": file.filename,
        "chunk_count": index.chunk_count,
        "document_count": index.document_count,
    }


# -- Ask ----------------------------------------------------------------------

@app.post("/ask")
def ask_question(req: QuestionRequest):
    """Ask a question against the indexed documents with optional memory context."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    session = _get_session(req.session_id)

    # Context-rewrite query using conversation history
    question = session.rewrite_query_with_context(req.question)

    # Build context strings for prompt injection
    conversation_context = session.build_context_prompt() if req.use_memory_context else ""
    pinned_context = session.get_pins_context() if req.use_memory_context else ""

    try:
        encoder = dependencies.get_encoder()
        index = dependencies.get_index()
        reranker = dependencies.get_reranker()
        llm = dependencies.get_llm_client()

        # Determine document scope: request-level doc_id overrides session-level docs
        if req.doc_id:
            doc_ids = [req.doc_id]
        else:
            doc_ids = [d["doc_id"] for d in session.active_docs] if session.active_docs else None

        result = run_query_pipeline(
            question=question,
            encoder=encoder,
            index=index,
            reranker=reranker,
            llm=llm,
            enable_nli=req.enable_nli,
            conversation_context=conversation_context,
            pinned_context=pinned_context,
            doc_ids=doc_ids,
        )
    except Exception as e:
        logger.error("Query error: %s", e, exc_info=True)
        return {
            "answer": "I cannot answer due to an internal error.",
            "verdict": "refused",
            "citations": [],
            "evidence": [],
        }

    # Store the turn in memory (strip raw evidence to save memory)
    if result.get("verdict") != "refused":
        turn = ConversationTurn(
            question=req.question,
            answer=result.get("answer", ""),
            verdict=result.get("verdict", "refused"),
            confidence=result.get("confidence", 0.0),
            citations=result.get("citations", []),
            evidence=[], # don't store full evidence in memory
            support_ratio=result.get("support_ratio", 0.0),
            citation_coverage=result.get("citation_coverage", 0.0),
        )
        session.add_turn(turn)

    # Include original question in response for UI tracking
    result["original_question"] = req.question
    result["rewritten_question"] = question if question != req.question else None
    return result


# -- Session & Memory Endpoints -----------------------------------------------

@app.post("/session/create")
def create_session(req: SessionRequest):
    """Create a new conversation session."""
    session_id = uuid.uuid4().hex[:16]
    _sessions[session_id] = ConversationMemory(session_name=req.session_name or "New Session")
    return {"session_id": session_id, "session_name": req.session_name}


@app.get("/session/{session_id}")
def get_session(session_id: str):
    """Get session metadata, history, and stats."""
    session = _get_session(session_id)
    return {
        "session_id": session_id,
        "session_name": session.session_name,
        "stats": session.get_stats(),
        "turns": [t.to_dict() for t in session.turns],
        "pins": [p.to_dict() for p in session.pins],
        "active_docs": session.active_docs,
    }


@app.get("/session/{session_id}/history")
def get_history(session_id: str):
    """Return conversation history for a session."""
    session = _get_session(session_id)
    return {"turns": [t.to_dict() for t in session.turns]}


@app.post("/session/{session_id}/pin")
def pin_insight(session_id: str, req: PinRequest):
    """Pin a key insight to session memory."""
    session = _get_session(session_id)
    pin = session.add_pin(
        text=req.text,
        source_question=req.source_question,
        from_doc=req.from_doc,
    )
    return {"status": "pinned", "pin": pin.to_dict()}


@app.delete("/session/{session_id}/pin/{pin_id}")
def remove_pin(session_id: str, pin_id: str):
    """Remove a pinned insight."""
    session = _get_session(session_id)
    removed = session.remove_pin(pin_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Pin not found")
    return {"status": "removed", "pin_id": pin_id}


@app.delete("/session/{session_id}/clear")
def clear_session(session_id: str):
    """Clear session history and pins (keeps documents)."""
    if session_id in _sessions:
        _sessions[session_id].clear()
    return {"status": "cleared"}


@app.get("/sessions")
def list_sessions():
    """List all active sessions with basic info."""
    return [
        {
            "session_id": sid,
            "session_name": mem.session_name,
            "turn_count": len(mem.turns),
            "pin_count": len(mem.pins),
            "doc_count": len(mem.active_docs),
            "created_at": mem.created_at.strftime("%Y-%m-%d %H:%M"),
        }
        for sid, mem in _sessions.items()
    ]


# -- Index Management ---------------------------------------------------------

@app.delete("/index")
def clear_index():
    try:
        dependencies.clear_index()
        return {"status": "cleared", "message": "Index cleared successfully."}
    except Exception as e:
        logger.error("Clear index error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to clear index")


@app.get("/index/status")
def index_status():
    index = dependencies.get_index()
    return {
        "chunk_count": index.chunk_count,
        "document_count": index.document_count,
        "doc_ids": list(index.get_doc_ids()),
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "2.0.0"}
