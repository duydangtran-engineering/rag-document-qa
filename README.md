# RAG Document Lookup System (Vietnamese)

Retrieval-augmented question answering over internal Vietnamese documents (legal texts, reports) with **cited answers**: file name, page and article number. PBL5 Computer Systems project.

## How it works

```
PDF/DOCX -> loader -> structure-aware chunker (contextual headers)
         -> bge-m3 embeddings (local GPU) -> ChromaDB
question -> multi-part splitting -> vector search -> bge-reranker-v2-m3
         -> neighbour-chunk expansion -> Gemini API -> answer + citations
```

- **Embeddings & reranking** run locally (BAAI/bge-m3, BAAI/bge-reranker-v2-m3, tested on an RTX 3060); only the final generation calls the Gemini API.
- **FastAPI** server keeps the models in memory and serves a web UI with document upload and streaming answers.
- **Chunking fix for legal text:** chunks without a clause title dropped from 68% to 1.5%; each chunk carries a contextual header (document, chapter, article).
- Multi-part questions are split and retrieved separately; neighbour chunks are added for context. Gemini fallback models handle quota (429) and overload (503) errors.

## Evaluation

`eval/` contains a 50-question set (fact, paraphrase, casual speech, article lookup, lists, cross-chunk, multi-part, abbreviations, tables, out-of-scope) and scripts for retrieval metrics (Hit@k, MRR) and answer checks.

On this set the gold passage reached the LLM context for **42 of 43 answerable questions**.

```bash
python eval/eval_retrieval.py --show-fail          # no API calls
python eval/eval_retrieval.py --no-rerank          # compare without the reranker
python eval/eval_answers.py                        # uses the Gemini API
```

## Run

```bash
python -m venv .venv && .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

copy .env.example .env        # then put your own GEMINI_API_KEY in .env
# put PDF/DOCX files in data/documents/
python ingest.py --reset
uvicorn api:app --reload      # open http://127.0.0.1:8000
```

The source documents and the vector store are not published. Detailed setup notes (Vietnamese): [`docs/setup_vi.md`](docs/setup_vi.md).

## Stack

Python, FastAPI, ChromaDB, sentence-transformers (bge-m3), bge-reranker-v2-m3, Gemini API, PyMuPDF
