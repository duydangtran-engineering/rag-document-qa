"""Cross-encoder reranking with BAAI/bge-reranker-v2-m3, running locally on the GPU.

Why a second model on top of the embedding search:
the embedding model (bi-encoder) turns the question and every chunk into vectors
SEPARATELY, then compares them. That is fast enough to scan the whole database,
but coarse — the eval showed the right chunk sitting at rank 6, 7, 11, 13 behind
chunks that merely share vocabulary ("nghèo, cận nghèo" -> income statistics).

A cross-encoder reads the question and ONE chunk TOGETHER and outputs a single
relevance score. Much more accurate, much too slow to run over 450 chunks — so we
use it only on the ~20 candidates the vector search already found, and keep the
best few. Typical cost on an RTX 3060: ~100-300 ms per question.

The score is a probability-like number in [0, 1] (sigmoid), which also separates
"in the documents" from "not in the documents" better than cosine similarity.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from sentence_transformers import CrossEncoder

from . import config


@lru_cache(maxsize=1)
def get_model() -> CrossEncoder:
    """Load once per process. First run downloads ~2.2 GB from HuggingFace."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = dict(device=device, max_length=config.RERANK_MAX_LENGTH)
    if device == "cuda":
        # Half precision: half the VRAM, roughly twice as fast, same ranking.
        try:
            return CrossEncoder(config.RERANK_MODEL,
                                model_kwargs={"torch_dtype": torch.float16}, **kwargs)
        except Exception:
            pass  # Fall back to the default precision below.
    return CrossEncoder(config.RERANK_MODEL, **kwargs)


def score(query: str, passages: list[str]) -> list[float]:
    """Relevance of each passage to the query, higher = more relevant."""
    if not passages:
        return []
    scores = get_model().predict(
        [(query, p) for p in passages],
        batch_size=config.RERANK_BATCH_SIZE,
        show_progress_bar=False,
    )
    return [float(s) for s in scores]
