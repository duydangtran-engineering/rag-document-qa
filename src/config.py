"""Central configuration. Everything tunable lives here, loaded from .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = parent folder of src/
ROOT_DIR = Path(__file__).resolve().parent.parent

load_dotenv(ROOT_DIR / ".env")


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", "off", ""}


# --- Paths ---
DOCUMENTS_DIR = ROOT_DIR / "data" / "documents"
STORAGE_DIR = ROOT_DIR / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"
COLLECTION_NAME = "internal_docs"
# Optional extra abbreviations for query expansion, see src/query_expansion.py
ABBREVIATIONS_FILE = ROOT_DIR / "data" / "abbreviations.json"

# --- Models ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
# Tried in order when GEMINI_MODEL is out of quota (429), unavailable (404) or
# overloaded (503). Each model has its own free quota. Comma-separated.
GEMINI_FALLBACK_MODELS = [
    m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "").split(",") if m.strip()
]
# Reuse the stored answer when the same question retrieves the same passages.
ANSWER_CACHE_ENABLED = _flag("ANSWER_CACHE", "1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
# Chunks shorter than this carry no answer (a lone page number, a cover line) and
# are dropped at ingestion.
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", 60))
# Put the whole heading path ("PHẦN II > II. THỰC TRẠNG > 1. THAM GIA") in front of
# each chunk before embedding / reranking. 0 = only the deepest heading (old behaviour).
HEADING_PATH = _flag("HEADING_PATH", "1")

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", 5))

# Chunks whose cosine distance is above this are considered irrelevant.
# 0.0 = identical, 2.0 = opposite. Tune after you see real results.
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", 1.0))

# Chunking cuts text at a fixed size, so an enumerated list can be split in two.
# After retrieval we therefore also pull in the chunks either side of the best
# hits, which is what lets the model see a list whole instead of counting half
# of it. 0 disables the behaviour.
NEIGHBOR_WINDOW = int(os.getenv("NEIGHBOR_WINDOW", 1))
# Only the strongest hits (per intent of the question) are expanded, otherwise
# the context grows 3x for nothing.
NEIGHBOR_ANCHORS = int(os.getenv("NEIGHBOR_ANCHORS", 3))

# --- Reranking (cross-encoder, runs locally, see src/reranker.py) ---
RERANK_ENABLED = _flag("RERANK_ENABLED", "1")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
# How many vector-search candidates the reranker re-scores before keeping TOP_K.
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", 20))
# Chunks scoring below this are dropped (0.0 = keep everything). Pick a value
# from the threshold section of `python eval/eval_retrieval.py`.
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", 0.0))
RERANK_MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", 512))
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", 16))

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def ensure_dirs() -> None:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
