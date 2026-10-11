"""Turn text into vectors with BAAI/bge-m3, running locally on the GPU.

Why local instead of an embedding API: the documents are internal/confidential,
and embedding is the step that touches every single page of them.
bge-m3 is multilingual and handles Vietnamese well.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from sentence_transformers import SentenceTransformer

from . import config


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the model once per process. First run downloads ~2 GB from HuggingFace."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(config.EMBEDDING_MODEL, device=device)
    return model


def describe_device() -> str:
    if torch.cuda.is_available():
        return f"GPU ({torch.cuda.get_device_name(0)})"
    return "CPU (chậm hơn nhiều — kiểm tra lại bản cài torch CUDA)"


def embed_documents(texts: list[str], batch_size: int = 8) -> list[list[float]]:
    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    model = get_model()
    vector = model.encode(
        [text], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )
    return vector[0].tolist()
