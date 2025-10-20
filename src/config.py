# src/config.py
import os

from dotenv import load_dotenv

load_dotenv()

# Dataset & Qdrant
DATASET_PATH_JSON = os.getenv(
    "DATASET_PATH_JSON", "../data/medical_qa_documents_with_id.json"
)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
HYBRID_COLLECTION = os.getenv(
    "QDRANT_HYBRID_COLLECTION", "medical-faq-sparse-and-dense"
)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "multi-qa-MiniLM-L6-cos-v1")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIMENSIONALITY", "384"))
QDRANT_BATCH_SIZE = int(os.getenv("QDRANT_BATCH_SIZE", "100"))

# PostgreSQL config
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", 5432))
PG_DB = os.getenv("PG_DB", "chat_db")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "admin")

# Reranker
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# LLM (Groq)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-20b")

# App defaults
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
DEFAULT_RERANK_K = int(os.getenv("DEFAULT_RERANK_K", "3"))


def require_llm_key():
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Put it into .env or set environment variable."
        )
