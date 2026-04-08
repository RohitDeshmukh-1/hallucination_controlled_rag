from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings managed by Pydantic.
    Reads from environment variables and .env file.
    """
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Hallucination-Controlled Academic RAG"
    
    # LLM Configuration
    LLM_API_KEY: str
    LLM_API_BASE: str = "https://api.groq.com/openai/v1"
    LLM_MODEL: str = "llama3-70b-8192"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 1024

    # Monitoring Configuration
    LANGCHAIN_TRACING_V2: str = "false"
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: str = "researchmind-rag"

    # Embedding Configuration
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    
    # Reranking Configuration
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_MAX_PASSAGES: int = 20
    RERANKER_TOP_N: int = 5

    # Verification Thresholds
    # 0.55 is calibrated for all-MiniLM-L6-v2 cosine similarity;
    # 0.35 was too lenient and allowed semantically distant claims through.
    VERIFICATION_SIMILARITY_THRESHOLD: float = 0.55
    VERIFICATION_UNSUPPORTED_RATIO: float = 0.5

    # Storage Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    STORAGE_DIR: Path = BASE_DIR / "storage"
    UPLOAD_DIR: Path = STORAGE_DIR / "uploads"
    INDEX_DIR: Path = STORAGE_DIR / "index"
    
    # Faiss Index Paths
    FAISS_INDEX_PATH: Path = INDEX_DIR / "faiss.index"
    FAISS_META_PATH: Path = INDEX_DIR / "chunks.pkl"

    # Supabase Configuration (Optional for Cloud Storage)
    SUPABASE_URL: Optional[str] = None
    SUPABASE_KEY: Optional[str] = None
    SUPABASE_BUCKET: str = "papers"

    # Optimization
    USE_GPU: bool = False


settings = Settings()
