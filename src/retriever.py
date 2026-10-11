"""Retrieval: question -> the most relevant chunks, numbered for citation.

Pipeline, for each intent of the question (see split_question):
  1. expand abbreviations          "LĐPCT" -> "LĐPCT (lao động phi chính thức)"
  2. vector search                 RERANK_CANDIDATES (20) nearest chunks, cheap
  3. cross-encoder rerank          re-score those 20 precisely, keep the best
Then the intents are merged, the chunks either side of each intent's strongest
hits are pulled in, and everything is put back in reading order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import config, embedder, reranker, vector_store
from .query_expansion import expand

# A question with two intents ("A là gì, và B bao nhiêu?") is squeezed into ONE
# vector, which lands between the two topics and near neither. Splitting it and
# embedding each part separately fixes that. Embedding runs locally on the GPU,
# so this costs zero extra API requests.
#
# Splitting is DELIBERATELY conservative. We never split on a bare "và", because
# "quy hoạch đô thị và nông thôn" is a single legal term. Only a comma (or
# semicolon / question mark) followed by a connective counts as a real boundary.
SPLIT_PATTERN = re.compile(r"(?:[,;]\s*(?:và|còn|đồng thời)\s+|\?\s+|;\s+)", re.IGNORECASE)

MIN_PART_LENGTH = 15  # A fragment shorter than this is not a standalone question.
MAX_PARTS = 3


def split_question(question: str) -> list[str]:
    """Break a multi-intent question apart, or return it unchanged."""
    parts = [p.strip(" ,;?.") for p in SPLIT_PATTERN.split(question)]
    parts = [p for p in parts if p]

    if len(parts) < 2 or len(parts) > MAX_PARTS:
        return [question.strip()]

    # If any fragment is too short to stand on its own, the split was wrong.
    if any(len(p) < MIN_PART_LENGTH for p in parts):
        return [question.strip()]

    return parts


@dataclass
class RetrievedChunk:
    number: int  # 1, 2, 3... this is the [n] the model must cite
    text: str
    source: str
    page: int
    heading: str
    citation: str
    distance: float | None  # None = pulled in as neighbouring context, not matched
    rerank_score: float | None = None  # None = reranker off, or neighbouring context

    @property
    def score(self) -> float | None:
        """Cosine similarity, easier to read than distance."""
        if self.distance is None:
            return None
        return round(1.0 - self.distance, 3)


def _key(hit: dict) -> tuple:
    return hit["metadata"].get("source"), hit["metadata"].get("index")


def _rerank_text(hit: dict) -> str:
    """Give the reranker the same heading context the embedding saw."""
    # The full heading path when the chunk has one (see Chunk.context_label).
    heading = (hit["metadata"].get("path") if config.HEADING_PATH else None) \
        or hit["metadata"].get("heading") or ""
    return f"{heading}\n{hit['text']}" if heading else hit["text"]


def rank(question: str, depth: int | None = None) -> list[dict]:
    """Ranked candidates for ONE intent: expand -> vector search -> rerank.

    Public so the evaluation script can measure ranking quality directly.
    Each returned hit is {text, metadata, distance[, rerank_score]}.
    """
    query = expand(question)
    use_rerank = config.RERANK_ENABLED
    pool = max(depth or 0, config.RERANK_CANDIDATES if use_rerank else config.TOP_K)

    hits = vector_store.search(embedder.embed_query(query), pool)
    hits = [h for h in hits if h["distance"] <= config.MAX_DISTANCE]

    if use_rerank and hits:
        scores = reranker.score(query, [_rerank_text(h) for h in hits])
        for hit, value in zip(hits, scores):
            hit["rerank_score"] = value
        hits.sort(key=lambda h: h["rerank_score"], reverse=True)
        hits = [h for h in hits if h["rerank_score"] >= config.RERANK_MIN_SCORE]

    return hits[:depth] if depth else hits


def retrieve(question: str, top_k: int | None = None) -> list[RetrievedChunk]:
    top_k = top_k or config.TOP_K
    parts = split_question(question)
    # One intent keeps top_k chunks. Several intents share a budget of 2 * top_k,
    # so a two-part question gets top_k chunks FOR EACH part.
    per_part = top_k if len(parts) == 1 else max(3, (2 * top_k) // len(parts))

    selected: list[dict] = []
    anchors: list[dict] = []
    seen: set[tuple] = set()
    for part in parts:
        part_hits = rank(part)[:per_part]
        # Neighbours are added around EACH intent's best hits. Taking the anchors
        # from the merged list instead let one intent grab all of them, and the
        # other intent's answer (cut across a chunk boundary) never got pulled in.
        anchors.extend(part_hits[: config.NEIGHBOR_ANCHORS])
        for hit in part_hits:
            if _key(hit) not in seen:  # A chunk serving two intents is kept once.
                seen.add(_key(hit))
                selected.append(hit)

    kept = _with_neighbours(selected, anchors)

    results: list[RetrievedChunk] = []
    for hit in kept:
        meta = hit["metadata"]
        results.append(
            RetrievedChunk(
                number=len(results) + 1,
                text=hit["text"],
                source=str(meta.get("source", "?")),
                page=int(meta.get("page", 0)),
                heading=str(meta.get("heading", "")),
                citation=str(meta.get("citation", "")),
                distance=float(hit["distance"]) if hit["distance"] is not None else None,
                rerank_score=hit.get("rerank_score"),
            )
        )
    return results


def _with_neighbours(hits: list[dict], anchors: list[dict]) -> list[dict]:
    """Add the chunks either side of the anchor hits, in reading order.

    Fixes a real failure: the list "Các loại quy hoạch ... gồm a) ... b) ..."
    is cut by the chunk boundary, so a model reading only the first half
    confidently answered "02 loại" when the law names five.
    """
    if config.NEIGHBOR_WINDOW <= 0 or not hits:
        return hits

    seen = {_key(h) for h in hits}
    extra: list[dict] = []

    for hit in anchors:
        source, index = _key(hit)
        if source is None or index is None:
            continue
        for step in range(1, config.NEIGHBOR_WINDOW + 1):
            for neighbour_index in (index - step, index + step):
                key = (source, neighbour_index)
                if neighbour_index < 0 or key in seen:
                    continue
                neighbour = vector_store.get_by_index(source, neighbour_index)
                if neighbour:
                    seen.add(key)
                    extra.append(neighbour)

    merged = hits + extra
    # Read order, not score order: chunks of the same document must follow each
    # other by position, otherwise a split sentence still reads backwards.
    best_rank: dict = {}
    for position, hit in enumerate(merged):
        best_rank.setdefault(hit["metadata"].get("source"), position)

    merged.sort(key=lambda h: (best_rank[h["metadata"].get("source")],
                               h["metadata"].get("index", 0)))
    return merged


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Format the chunks the way the model will see them."""
    blocks = []
    for chunk in chunks:
        blocks.append(
            f"[{chunk.number}] Nguồn: {chunk.citation}\n{chunk.text}"
        )
    return "\n\n---\n\n".join(blocks)
