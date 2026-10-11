"""Remember answers so a repeated question costs no Gemini request.

During testing and demos the same questions get asked again and again, and the
free tier only allows a handful of requests per day per model.

The cache key is: normalised question + fingerprint of the exact passages that
were retrieved + the system prompt. So an answer is reused ONLY when the model
would have been given the identical input. Re-ingesting a document, changing
retrieval settings, or editing the prompt changes the key and the answer is
generated fresh — no stale answers after you change something.

Stored in storage/answer_cache.json. Delete that file to clear it, or set
ANSWER_CACHE=0 in .env to turn caching off.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import unicodedata

from . import config

PATH = config.STORAGE_DIR / "answer_cache.json"
MAX_ENTRIES = 1000

_lock = threading.Lock()
_data: dict | None = None


def _norm(question: str) -> str:
    text = unicodedata.normalize("NFC", question).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" \"'“”?.!\\")


def _key(question: str, chunks, salt: str) -> str:
    h = hashlib.sha1()
    h.update(_norm(question).encode("utf-8"))
    h.update(hashlib.sha1(salt.encode("utf-8")).digest())
    for c in chunks:
        h.update(f"|{c.number}|{c.source}|{c.page}|".encode("utf-8"))
        h.update(hashlib.sha1(c.text.encode("utf-8")).digest())
    return h.hexdigest()


def _load() -> dict:
    global _data
    if _data is None:
        try:
            _data = json.loads(PATH.read_text(encoding="utf-8"))
        except Exception:
            _data = {}
    return _data


def get(question: str, chunks, salt: str = "") -> dict | None:
    if not config.ANSWER_CACHE_ENABLED:
        return None
    with _lock:
        return _load().get(_key(question, chunks, salt))


def put(question: str, chunks, answer: str, model: str, salt: str = "") -> None:
    if not config.ANSWER_CACHE_ENABLED or not answer.strip():
        return
    with _lock:
        data = _load()
        data[_key(question, chunks, salt)] = {
            "question": question, "answer": answer, "model": model, "created": int(time.time()),
        }
        if len(data) > MAX_ENTRIES:  # Drop the oldest entries.
            for k in sorted(data, key=lambda k: data[k]["created"])[: len(data) - MAX_ENTRIES]:
                del data[k]
        try:
            PATH.parent.mkdir(parents=True, exist_ok=True)
            PATH.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:
            pass  # Caching is an optimisation; never fail an answer over it.
