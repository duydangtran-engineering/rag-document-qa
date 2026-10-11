"""Persistent vector database (ChromaDB), stored on disk under storage/chroma.

We pass embeddings in ourselves rather than letting Chroma compute them,
so the embedding model stays under our control.
"""

from __future__ import annotations

import chromadb
from chromadb.config import Settings

from . import config
from .chunker import Chunk


def get_collection():
    client = chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection() -> None:
    client = chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass  # Nothing to delete on a fresh install.


def add_chunks(chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    collection = get_collection()
    collection.upsert(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[c.metadata() for c in chunks],
        embeddings=embeddings,
    )


def delete_source(source: str) -> int:
    """Remove every chunk that came from one file. Returns how many were removed.

    Needed so that re-uploading (or re-ingesting) a file REPLACES its chunks. Chunk
    ids contain the chunk text, so without this an edited document would leave its
    old chunks in the database next to the new ones, and answers would quote both.
    """
    collection = get_collection()
    found = collection.get(where={"source": {"$eq": source}}, include=[])
    ids = found["ids"]
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def count() -> int:
    return get_collection().count()


def get_by_index(source: str, index: int) -> dict | None:
    """Fetch one chunk by its position in the document, not by similarity.

    Used to pull in a neighbouring chunk so text cut across a boundary can be
    read whole.
    """
    collection = get_collection()
    result = collection.get(
        where={"$and": [{"source": {"$eq": source}}, {"index": {"$eq": index}}]},
        include=["documents", "metadatas"],
        limit=1,
    )
    if not result["ids"]:
        return None
    return {
        "text": result["documents"][0],
        "metadata": result["metadatas"][0],
        "distance": None,  # Not a similarity hit; it came along for context.
    }


def search(query_embedding: list[float], top_k: int) -> list[dict]:
    """Return the closest chunks, each as {text, metadata, distance}."""
    collection = get_collection()
    if collection.count() == 0:
        return []

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits: list[dict] = []
    for text, metadata, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append({"text": text, "metadata": metadata, "distance": distance})
    return hits