"""FastAPI server for the document lookup system.

Run:
    uvicorn api:app --reload

Then open http://127.0.0.1:8000

Why a server is more than a wrapper around ask.py: the embedding model is loaded
ONCE at startup and stays in memory. The CLI reloads it on every run, which is
where most of the perceived latency came from.
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import config, embedder, generator, ingestor, reranker, retriever, vector_store

WEB_DIR = config.ROOT_DIR / "web"

# Guards against someone uploading a huge file and tying up the embedding
# worker for minutes. Bump this if your real documents are routinely bigger.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@asynccontextmanager
async def lifespan(_app: FastAPI):
    config.ensure_dirs()
    print("Loading embedding model, this takes a few seconds...")
    embedder.get_model()  # Warm up now so the first question is not slow.
    if config.RERANK_ENABLED:
        print(f"Loading reranker {config.RERANK_MODEL}...")
        reranker.get_model()
    if vector_store.count() > 0:
        # One throwaway search: the very first GPU call pays for CUDA kernel
        # setup (~1.5 s measured). Better here than on the user's first question.
        retriever.retrieve("khởi động hệ thống")
    print(f"Ready on {embedder.describe_device()} with {vector_store.count()} chunks"
          f" (rerank: {'on' if config.RERANK_ENABLED else 'off'}).")
    yield


app = FastAPI(title="Internal Document Lookup", version="1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class Source(BaseModel):
    number: int
    citation: str
    source: str
    page: int
    heading: str
    score: float | None  # None when the chunk was added as neighbouring context
    rerank_score: float | None = None  # 0..1 from the cross-encoder; None if off/neighbour
    preview: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    # Timed separately on purpose: retrieval runs locally on the GPU, generation
    # is a network call to Gemini. Knowing which dominates is what tells us
    # whether optimising the pipeline is worth anything at all.
    retrieval_ms: int
    generation_ms: int
    elapsed_ms: int
    # Only longer than one item when the question carried two separate intents.
    sub_questions: list[str]
    model: str | None = None  # Which Gemini model answered (fallbacks may kick in)
    cached: bool = False      # True = served from storage/answer_cache.json, no API call


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


@app.get("/api/status")
def read_status() -> dict:
    documents = sorted(
        p.name
        for p in config.DOCUMENTS_DIR.glob("*")
        if p.is_file() and p.suffix.lower() in config.SUPPORTED_EXTENSIONS
    )
    return {
        "chunks": vector_store.count(),
        "documents": documents,
        "model": config.GEMINI_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "device": embedder.describe_device(),
        "top_k": config.TOP_K,
        "reranker": config.RERANK_MODEL if config.RERANK_ENABLED else None,
        "fallback_models": config.GEMINI_FALLBACK_MODELS,
        "answer_cache": config.ANSWER_CACHE_ENABLED,
    }


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Save an uploaded file and ingest it, streaming progress as NDJSON.

    Why NDJSON over StreamingResponse instead of a normal JSON response: ingest
    of a real document (chunk + embed) takes anywhere from a few seconds to a
    minute, and a silent progress bar that doesn't move for that long reads as
    broken. Each line here is one `ingestor.ProgressEvent` as JSON, so the
    frontend can update the bar live. Browsers' EventSource only supports GET,
    which is why this uses a plain streamed fetch() body instead of real SSE.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Thiếu tên file.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in config.SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(config.SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng '{suffix}' không được hỗ trợ. Chỉ nhận: {allowed}",
        )

    # Read into memory HERE (inside the async endpoint) — UploadFile's underlying
    # temp file/connection is not guaranteed to stay usable once we hand off to
    # the generator below, which Starlette runs in a worker thread.
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"File vượt quá giới hạn {mb}MB.")

    config.ensure_dirs()
    dest = config.DOCUMENTS_DIR / file.filename

    def event_stream():
        try:
            dest.write_bytes(content)
            yield json.dumps(
                {"stage": "saved", "message": f"Đã lưu {file.filename}, bắt đầu xử lý..."},
                ensure_ascii=False,
            ) + "\n"

            for event in ingestor.ingest_path(dest):
                yield json.dumps(event, ensure_ascii=False) + "\n"

        except Exception as exc:
            # Same principle as the /api/ask error handling below: never let an
            # exception just kill the stream silently. Say what broke.
            yield json.dumps(
                {"stage": "error", "message": f"Lỗi khi xử lý {file.filename}: {exc}"},
                ensure_ascii=False,
            ) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


# Defined with `def`, not `async def`: FastAPI then runs it in a worker thread, so
# the blocking embedding / HTTP calls never freeze the event loop.
@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    if vector_store.count() == 0:
        raise HTTPException(
            status_code=409,
            detail="Chưa có tài liệu nào trong hệ thống. Chạy `python ingest.py` trước.",
        )

    started = time.perf_counter()
    sub_questions = retriever.split_question(request.question)
    chunks = retriever.retrieve(request.question, request.top_k)
    retrieved_at = time.perf_counter()

    info: dict = {}
    try:
        answer = generator.generate_answer(request.question, chunks, info=info)
    except Exception as exc:
        # Catch EVERYTHING. An SDK exception escaping the endpoint kills the
        # connection, and the browser can only report "server unreachable" —
        # which hides the real cause (quota, model name, Google outage).
        raise HTTPException(
            status_code=generator.http_status_for(exc),
            detail=str(generator.explain_api_error(exc)),
        ) from exc

    finished_at = time.perf_counter()

    return AskResponse(
        answer=answer,
        sources=[
            Source(
                number=c.number,
                citation=c.citation,
                source=c.source,
                page=c.page,
                heading=c.heading,
                score=c.score,
                rerank_score=_round(c.rerank_score),
                preview=c.text[:400],
            )
            for c in chunks
        ],
        retrieval_ms=int((retrieved_at - started) * 1000),
        generation_ms=int((finished_at - retrieved_at) * 1000),
        elapsed_ms=int((finished_at - started) * 1000),
        sub_questions=sub_questions,
        model=info.get("model"),
        cached=bool(info.get("cached")),
    )


@app.post("/api/ask/stream")
def ask_stream(request: AskRequest):
    """Same as /api/ask, but streams the answer as it's generated (NDJSON lines).

    Event shapes sent, in order:
      {"type": "meta", "sub_questions": [...], "sources": [...], "retrieval_ms": N}
      {"type": "answer_chunk", "text": "..."}   — zero or more, as Gemini streams
      {"type": "done", "generation_ms": N, "elapsed_ms": N}
    or, at any point instead of the above:
      {"type": "error", "status": <http status>, "message": "..."}

    Retrieval isn't streamed — it's 200-500ms, not worth it. Generation is where
    the 7-15s wait lives, so that's the part the frontend actually needs live.
    """
    if vector_store.count() == 0:
        raise HTTPException(
            status_code=409,
            detail="Chưa có tài liệu nào trong hệ thống. Chạy `python ingest.py` trước.",
        )

    def event_stream():
        started = time.perf_counter()
        sub_questions = retriever.split_question(request.question)
        chunks = retriever.retrieve(request.question, request.top_k)
        retrieved_at = time.perf_counter()

        yield json.dumps({
            "type": "meta",
            "sub_questions": sub_questions,
            "sources": [
                {
                    "number": c.number,
                    "citation": c.citation,
                    "source": c.source,
                    "page": c.page,
                    "heading": c.heading,
                    "score": c.score,
                    "rerank_score": _round(c.rerank_score),
                    "preview": c.text[:400],
                }
                for c in chunks
            ],
            "retrieval_ms": int((retrieved_at - started) * 1000),
        }, ensure_ascii=False) + "\n"

        info: dict = {}
        try:
            for piece in generator.generate_answer_stream(request.question, chunks, info=info):
                yield json.dumps({"type": "answer_chunk", "text": piece}, ensure_ascii=False) + "\n"
        except Exception as exc:
            yield json.dumps({
                "type": "error",
                "status": generator.http_status_for(exc),
                "message": str(generator.explain_api_error(exc)),
            }, ensure_ascii=False) + "\n"
            return

        finished_at = time.perf_counter()
        yield json.dumps({
            "type": "done",
            "generation_ms": int((finished_at - retrieved_at) * 1000),
            "elapsed_ms": int((finished_at - started) * 1000),
            "model": info.get("model"),
            "cached": bool(info.get("cached")),
        }, ensure_ascii=False) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
