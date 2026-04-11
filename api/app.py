from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import shutil
import uuid
import logging

from pipeline.ingest_document import ingest_document
from pipeline.query_pipeline import run_query_pipeline
from pipeline.conversation_memory import ConversationMemory, ConversationTurn
from api import dependencies
from api.auth import (
    AuthError,
    AuthenticatedUser,
    create_access_token,
    get_current_user,
    user_store,
)
from configs.settings import settings

import gc
import os

# Force single thread for torch to save CPU/memory overhead on small instances
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
gc.collect()
logger = logging.getLogger(__name__)

# In-memory session store: user_id -> session_id -> ConversationMemory
_sessions: dict[str, dict[str, ConversationMemory]] = {}


def _get_user_sessions(user_id: str) -> dict[str, ConversationMemory]:
    return _sessions.setdefault(user_id, {})


def _create_session(user_id: str, session_name: str) -> tuple[str, ConversationMemory]:
    session_id = uuid.uuid4().hex[:16]
    session = ConversationMemory(session_name=session_name or "New Session")
    _get_user_sessions(user_id)[session_id] = session
    return session_id, session


def _resolve_session(
    user_id: str,
    session_id: Optional[str],
    *,
    create_if_missing: bool,
) -> tuple[str, ConversationMemory]:
    user_sessions = _get_user_sessions(user_id)
    if session_id and session_id in user_sessions:
        return session_id, user_sessions[session_id]

    if session_id and not create_if_missing:
        raise HTTPException(status_code=404, detail="Session not found")

    if session_id and create_if_missing:
        session = ConversationMemory(session_name=f"Session {session_id[:6]}")
        user_sessions[session_id] = session
        return session_id, session

    if user_sessions:
        latest_session_id = next(reversed(user_sessions))
        return latest_session_id, user_sessions[latest_session_id]

    return _create_session(user_id, "Research Session")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Models will load lazily on first request to avoid deployment timeouts
    logger.info(f"ResearchMind RAG system starting on {settings.PROJECT_NAME}")
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
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def root():
    return {
        "message": "Welcome to ResearchMind RAG API",
        "version": "2.0.0",
        "status": "online",
        "docs": "/docs"
    }


# -- Request / Response Models ------------------------------------------------

class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    doc_id: Optional[str] = None
    enable_nli: bool = False
    use_memory_context: bool = True


class PinRequest(BaseModel):
    session_id: Optional[str] = None
    text: str
    source_question: str
    from_doc: Optional[str] = None


class SessionRequest(BaseModel):
    session_name: Optional[str] = "New Session"


class AuthRequest(BaseModel):
    username: str
    password: str


# -- Upload -------------------------------------------------------------------

@app.post("/auth/register")
def register(req: AuthRequest):
    try:
        user = user_store.create_user(req.username, req.password)
    except AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    token = create_access_token(user)
    return {"token": token, "user": user.to_dict()}


@app.post("/auth/login")
def login(req: AuthRequest):
    try:
        user = user_store.authenticate(req.username, req.password)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    token = create_access_token(user)
    return {"token": token, "user": user.to_dict()}


@app.get("/auth/me")
def auth_me(current_user: AuthenticatedUser = Depends(get_current_user)):
    return {"user": current_user.to_dict()}


@app.post("/upload")
def upload_pdf(
    file: UploadFile = File(...),
    session_id: Optional[str] = None,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Upload and index a PDF document."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_id = f"{uuid.uuid4().hex}.pdf"
    upload_dir = settings.UPLOAD_DIR / current_user.user_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file_id

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        logger.error("File save error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to save file")

    try:
        encoder = dependencies.get_encoder()
        index = dependencies.get_index(current_user.user_id)
        logger.info("Ingesting new document: %s", file.filename)
        doc_id = ingest_document(file_path, encoder, index)
    except Exception as e:
        logger.error("Ingestion error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to ingest document")

    # Register doc into session memory
    _, session = _resolve_session(current_user.user_id, session_id, create_if_missing=True)
    session.register_document(doc_id, file.filename, index.chunk_count)
    
    # Explicitly clear memory after heavy operation
    gc.collect()

    return {
        "status": "indexed",
        "doc_id": doc_id,
        "filename": file.filename,
        "chunk_count": index.chunk_count,
        "document_count": index.document_count,
    }


# -- Ask ----------------------------------------------------------------------

@app.post("/ask")
def ask_question(
    req: QuestionRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Ask a question against the indexed documents with optional memory context."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    _, session = _resolve_session(current_user.user_id, req.session_id, create_if_missing=True)

    # Context-rewrite query using conversation history
    question = session.rewrite_query_with_context(req.question)

    # Build context strings for prompt injection
    conversation_context = session.build_context_prompt() if req.use_memory_context else ""
    pinned_context = session.get_pins_context() if req.use_memory_context else ""

    try:
        encoder = dependencies.get_encoder()
        index = dependencies.get_index(current_user.user_id)
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
def create_session(
    req: SessionRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Create a new conversation session."""
    session_name = req.session_name or "New Session"
    session_id, _ = _create_session(current_user.user_id, session_name)
    return {"session_id": session_id, "session_name": session_name}


@app.get("/session/{session_id}")
def get_session(
    session_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Get session metadata, history, and stats."""
    _, session = _resolve_session(current_user.user_id, session_id, create_if_missing=False)
    return {
        "session_id": session_id,
        "session_name": session.session_name,
        "stats": session.get_stats(),
        "turns": [t.to_dict() for t in session.turns],
        "pins": [p.to_dict() for p in session.pins],
        "active_docs": session.active_docs,
    }


@app.get("/session/{session_id}/history")
def get_history(
    session_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Return conversation history for a session."""
    _, session = _resolve_session(current_user.user_id, session_id, create_if_missing=False)
    return {"turns": [t.to_dict() for t in session.turns]}


@app.post("/session/{session_id}/pin")
def pin_insight(
    session_id: str,
    req: PinRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Pin a key insight to session memory."""
    _, session = _resolve_session(current_user.user_id, session_id, create_if_missing=False)
    pin = session.add_pin(
        text=req.text,
        source_question=req.source_question,
        from_doc=req.from_doc,
    )
    return {"status": "pinned", "pin": pin.to_dict()}


@app.delete("/session/{session_id}/pin/{pin_id}")
def remove_pin(
    session_id: str,
    pin_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Remove a pinned insight."""
    _, session = _resolve_session(current_user.user_id, session_id, create_if_missing=False)
    removed = session.remove_pin(pin_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Pin not found")
    return {"status": "removed", "pin_id": pin_id}


@app.delete("/session/{session_id}/clear")
def clear_session(
    session_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Clear session history and pins (keeps documents)."""
    _, session = _resolve_session(current_user.user_id, session_id, create_if_missing=False)
    session.clear()
    return {"status": "cleared"}


@app.get("/sessions")
def list_sessions(current_user: AuthenticatedUser = Depends(get_current_user)):
    """List all active sessions with basic info."""
    user_sessions = _get_user_sessions(current_user.user_id)
    return [
        {
            "session_id": sid,
            "session_name": mem.session_name,
            "turn_count": len(mem.turns),
            "pin_count": len(mem.pins),
            "doc_count": len(mem.active_docs),
            "created_at": mem.created_at.strftime("%Y-%m-%d %H:%M"),
        }
        for sid, mem in user_sessions.items()
    ]


# -- Index Management ---------------------------------------------------------

@app.delete("/index")
def clear_index(current_user: AuthenticatedUser = Depends(get_current_user)):
    try:
        dependencies.clear_index(current_user.user_id)
        for session in _get_user_sessions(current_user.user_id).values():
            session.active_docs.clear()
        return {"status": "cleared", "message": "Index cleared successfully."}
    except Exception as e:
        logger.error("Clear index error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to clear index")


@app.get("/index/status")
def index_status(current_user: AuthenticatedUser = Depends(get_current_user)):
    index = dependencies.get_index(current_user.user_id)
    return {
        "chunk_count": index.chunk_count,
        "document_count": index.document_count,
        "doc_ids": list(index.get_doc_ids()),
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "2.0.0"}
