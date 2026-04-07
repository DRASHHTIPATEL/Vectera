"""Application paths and environment-driven settings."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
# Load from project root so OPENAI_API_KEY is found even when Streamlit's cwd differs.
load_dotenv(ROOT / ".env")

# Assessment: use Postgres (Supabase / RDS / Docker) when DATABASE_URL is set.
# If unset, the app falls back to SQLite + on-disk FAISS for local-only demos.
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))


def use_postgres() -> bool:
    return bool(DATABASE_URL)
DATA_DIR = ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "vectera.db"
FAISS_PATH = DATA_DIR / "faiss.index"
FAISS_META_PATH = DATA_DIR / "faiss_meta.json"

# --- LLM (OpenAI-compatible API: OpenAI, Groq, Together, local Ollama, etc.) ---
USE_OLLAMA = os.getenv("USE_OLLAMA", "").lower() in ("1", "true", "yes")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or None
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
_model_env = (os.getenv("OPENAI_MODEL") or "").strip()

if USE_OLLAMA:
    # Free local LLM: https://ollama.com — no cloud API key required.
    if not OPENAI_BASE_URL:
        OPENAI_BASE_URL = "http://127.0.0.1:11434/v1"
    if not OPENAI_API_KEY:
        OPENAI_API_KEY = "ollama"
    OPENAI_MODEL = _model_env or "llama3.2"
else:
    OPENAI_MODEL = _model_env or "gpt-4o-mini"

# Chunking: character-based with structure-first splits (not raw token windows)
CHUNK_TARGET_CHARS = 900
CHUNK_MIN_CHARS = 200
CHUNK_OVERLAP_CHARS = 180

# Retrieval
RETRIEVAL_CANDIDATES = 24
RETRIEVAL_FINAL_MIN = 5
RETRIEVAL_FINAL_MAX = 8
MAX_CHUNKS_PER_DOCUMENT_IN_BATCH = 3

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
