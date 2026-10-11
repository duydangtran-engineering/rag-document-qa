"""Ingest ONE file already saved on disk, reporting progress as it goes.

`ingest.py` (the CLI) embeds an entire folder in a single batch call — fine for a
one-off bulk load, but useless for a progress bar. This module re-uses the same
load -> chunk -> embed -> store pipeline, just broken into small steps so each
one can be reported back to the caller (the web upload endpoint) as it happens.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, TypedDict

from . import config, embedder, vector_store
from .chunker import chunk_blocks
from .loader import load_document

# Smaller than ingest.py's implicit batch: the point here is frequent progress
# updates, not raw throughput. embed_documents() also accepts a batch_size, so
# we hand it exactly the slice we're already iterating.
EMBED_BATCH_SIZE = 8


class ProgressEvent(TypedDict, total=False):
    stage: str  # "loading" | "chunking" | "embedding" | "storing" | "done" | "error"
    message: str
    progress: float  # 0..100, only set during "embedding"
    chunks_created: int
    filename: str


def ingest_path(path: Path) -> Iterator[ProgressEvent]:
    """Load, chunk, embed and store one file, yielding a ProgressEvent per step.

    The caller (api.py) is responsible for having already saved `path` to disk
    (usually under config.DOCUMENTS_DIR) before calling this.
    """
    filename = path.name

    yield {"stage": "loading", "message": f"Đang đọc {filename}..."}
    try:
        blocks = load_document(path)
    except Exception as exc:
        yield {"stage": "error", "message": f"Không đọc được {filename}: {exc}"}
        return

    if not blocks:
        yield {"stage": "error", "message": f"{filename} không có nội dung đọc được."}
        return

    yield {
        "stage": "chunking",
        "message": f"Đã đọc {len(blocks)} trang, đang cắt chunk...",
    }
    chunks = chunk_blocks(blocks, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    if not chunks:
        yield {"stage": "error", "message": f"Không tạo được chunk nào từ {filename}."}
        return

    total = len(chunks)
    yield {
        "stage": "embedding",
        "message": f"Đang tạo embedding cho {total} chunk trên {embedder.describe_device()}...",
        "progress": 0,
    }

    texts = [c.embedding_text for c in chunks]
    vectors: list[list[float]] = []
    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = texts[start:start + EMBED_BATCH_SIZE]
        vectors.extend(embedder.embed_documents(batch, batch_size=len(batch)))
        done = start + len(batch)
        yield {
            "stage": "embedding",
            "message": f"Đã embedding {done}/{total} chunk",
            "progress": round(done / total * 100, 1),
        }

    yield {"stage": "storing", "message": "Đang lưu vào vector database..."}
    # Replace, don't append: a file uploaded twice (or edited and uploaded again)
    # must not leave its old chunks behind.
    vector_store.delete_source(filename)
    vector_store.add_chunks(chunks, vectors)

    yield {
        "stage": "done",
        "message": f"Hoàn tất: {filename} — {total} chunk đã sẵn sàng để tra cứu.",
        "chunks_created": total,
        "filename": filename,
    }
