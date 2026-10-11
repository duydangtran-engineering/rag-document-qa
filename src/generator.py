"""Answer generation with Gemini, tightly grounded on the retrieved chunks.

The whole value of a RAG system is that it refuses to invent things.
The system prompt below is the main defence against that, so treat it as
production code, not decoration.

Resilience, because the free tier is small and Google is sometimes overloaded:
  - MODEL FALLBACK: GEMINI_MODEL first, then each of GEMINI_FALLBACK_MODELS.
    Every model has its OWN free quota, so when one hits 429 the next one
    usually still works. A model that ran out is skipped until its quota resets.
  - RETRY: a 503/overload is retried with backoff (2 s, 5 s) before moving on
    to the next model — in the streaming path too.
  - ANSWER CACHE: the same question over the same retrieved passages returns the
    stored answer instantly and costs no request (see answer_cache.py).
"""

from __future__ import annotations

import json
import re
import threading
import time
from functools import lru_cache

from google import genai
from google.genai import types

from . import answer_cache, config
from .retriever import RetrievedChunk, build_context

SYSTEM_PROMPT = """Bạn là trợ lý tra cứu tài liệu nội bộ. Nhiệm vụ của bạn là trả lời câu hỏi CHỈ dựa trên các đoạn trích được cung cấp.

QUY TẮC BẮT BUỘC:
1. Chỉ dùng thông tin có trong phần "TÀI LIỆU THAM KHẢO". Tuyệt đối không dùng kiến thức bên ngoài.
2. Sau mỗi ý, ghi số nguồn trong ngoặc vuông, ví dụ: [1] hoặc [2][3]. Mỗi khẳng định phải có ít nhất một trích dẫn.
3. Nếu tài liệu không chứa đủ thông tin để trả lời, hãy nói rõ: "Tôi không tìm thấy thông tin này trong tài liệu được cung cấp." Sau đó có thể gợi ý người dùng đặt câu hỏi cụ thể hơn. KHÔNG được đoán.
4. Nếu các đoạn trích mâu thuẫn nhau, hãy nêu rõ sự mâu thuẫn và trích dẫn cả hai nguồn.
5. Trả lời HOÀN TOÀN bằng tiếng Việt (chỉ giữ nguyên tên riêng, thuật ngữ, tên biến và công thức đúng như trong tài liệu; không chen từ tiếng Anh khác), ngắn gọn, đúng trọng tâm. Chỉ trả lời đúng điều được hỏi, không thêm những ý người dùng không hỏi. Với nội dung pháp lý hoặc y dược, giữ nguyên số liệu, tên điều khoản và thuật ngữ đúng như trong tài liệu; khi nói một quy định có hiệu lực hay áp dụng thế nào, ghi đúng điều/khoản được nói tới, không nhầm với điều/khoản chứa câu quy định đó.
6. Không mở đầu bằng câu xã giao. Đi thẳng vào câu trả lời.
7. TUYỆT ĐỐI KHÔNG dùng cú pháp LaTeX khi viết công thức (không dùng $, \\frac, \\text, \\times, ^, _, {}...), kể cả khi tài liệu gốc trình bày công thức bằng LaTeX. Giao diện hiển thị chỉ hiểu văn bản thường và Markdown cơ bản (in đậm, danh sách), không hiểu LaTeX. Hãy viết công thức bằng ký hiệu thường: phân số viết dạng "a / b" hoặc "a chia cho b", nhân dùng "x" hoặc "*", luỹ thừa viết "a^2" thành "a mũ 2" hoặc "a bình phương". Ví dụ: viết "Precision = TP / (TP + FP)" thay vì "$\\text{Precision} = \\frac{TP}{TP+FP}$".
8. Nếu câu hỏi yêu cầu tóm tắt, liệt kê toàn bộ, hoặc tổng hợp cả tài liệu, hãy mở đầu bằng đúng một câu: "Tóm tắt dựa trên các đoạn liên quan nhất tìm được, có thể chưa bao quát toàn bộ tài liệu." rồi mới trả lời."""

USER_TEMPLATE = """TÀI LIỆU THAM KHẢO:
{context}

CÂU HỎI: {question}

Trả lời dựa hoàn toàn trên tài liệu trên, có trích dẫn [số nguồn]."""

# Shown when retrieval found nothing relevant enough (including questions that
# the reranker threshold rejected as out of scope). Written for the END USER —
# the "run python ingest.py" hint for an empty database lives in api.py.
NO_CONTEXT_ANSWER = (
    "Tôi không tìm thấy thông tin liên quan trong các tài liệu đã nạp. "
    "Câu hỏi có thể nằm ngoài phạm vi tài liệu; bạn thử diễn đạt lại cụ thể hơn."
)


class FriendlyError(RuntimeError):
    """An error whose message is already written for the end user.

    `status` is the HTTP status the web API should answer with."""

    def __init__(self, message: str, status: int = 500):
        super().__init__(message)
        self.status = status


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    if not config.GEMINI_API_KEY:
        raise RuntimeError(
            "Chưa có GEMINI_API_KEY. Hãy tạo file .env từ .env.example và điền key "
            "lấy tại https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=config.GEMINI_API_KEY)


# ---------------------------------------------------------------------------
# Model fallback bookkeeping
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_blocked_until: dict[str, float] = {}  # model -> time.time() when it may be tried again


def candidate_models() -> list[str]:
    """Models to try, in order, skipping those whose quota is known to be spent."""
    now = time.time()
    ordered = [config.GEMINI_MODEL] + [m for m in config.GEMINI_FALLBACK_MODELS
                                       if m != config.GEMINI_MODEL]
    with _lock:
        usable = [m for m in ordered if _blocked_until.get(m, 0) <= now]
    return usable


def _block(model: str, exc: Exception) -> None:
    """Remember not to call `model` again until its limit has probably reset."""
    message = str(exc)
    if "404" in message or "NOT_FOUND" in message:
        seconds = 24 * 3600          # Model retired / not available for this key.
    elif "PerDay" in message:
        seconds = 6 * 3600           # Daily quota: pointless to retry within minutes.
    else:
        delay = re.search(r"retry in ([\d.]+)s", message)
        seconds = float(delay.group(1)) + 1 if delay else 60  # Per-minute limit.
    with _lock:
        _blocked_until[model] = time.time() + seconds


def _is_quota(exc: Exception) -> bool:
    message = str(exc)
    return "429" in message or "RESOURCE_EXHAUSTED" in message


def _is_not_found(exc: Exception) -> bool:
    message = str(exc)
    return "404" in message or "NOT_FOUND" in message


def _is_invalid_argument(exc: Exception) -> bool:
    """True when Gemini rejected a field in the config rather than the request itself."""
    if _is_quota(exc) or _is_not_found(exc):
        return False  # A 429 message can contain "400" inside a number.
    message = str(exc)
    return "INVALID_ARGUMENT" in message or bool(re.match(r"^400\b", message))


# Server-side hiccups, not our fault and not our quota. Unlike 404 (wrong model)
# or 429 (out of quota), these clear up on their own — so we wait and retry.
TRANSIENT_STATUS = re.compile(r"^(500|502|503|504)\b")
TRANSIENT_NAMES = ("UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED", "overloaded")


def _is_transient(exc: Exception) -> bool:
    message = str(exc)
    return bool(TRANSIENT_STATUS.match(message)) or any(n in message for n in TRANSIENT_NAMES)


RETRY_DELAYS = (2.0, 5.0)  # waits before the 2nd and 3rd attempt, FIRST model only
# When Google is overloaded it usually is for every model at once. Trying all of
# them with retries made the user wait ~50 s for the same error, so stop after
# this many models have answered "overloaded".
MAX_OVERLOADED_MODELS = 3


# ---------------------------------------------------------------------------
# Per-model "which optional config fields does it accept" profile
# ---------------------------------------------------------------------------

PROFILE_PATH = config.STORAGE_DIR / "model_profile.json"
PROFILE_VERSION = 2  # Bump whenever the list returned by _attempts() changes.


def _read_profiles() -> dict:
    try:
        data = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"_version": PROFILE_VERSION}
    if data.get("_version") != PROFILE_VERSION:
        return {"_version": PROFILE_VERSION}  # Config list changed: probe again.
    return data


def _load_profile(model: str) -> int:
    """Index of the config that last worked for this model (0 if unknown)."""
    try:
        return max(0, int(_read_profiles().get(model, 0)))
    except Exception:
        return 0


def _save_profile(model: str, index: int) -> None:
    with _lock:
        profiles = _read_profiles()
        if profiles.get(model) == index:
            return
        profiles[model] = index
        try:
            PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            PROFILE_PATH.write_text(json.dumps(profiles, indent=2), encoding="utf-8")
        except Exception:
            pass  # A cache we cannot write is not worth failing the answer over.


def _attempts() -> list[dict]:
    """Richest config first; drop one optional field per retry.

    Which fields a model accepts changes over time, and a rejected field comes
    back as a bare "400 INVALID_ARGUMENT" that names nothing.
    """
    base = dict(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.1,  # Low: we want faithful extraction, not creativity.
        max_output_tokens=4096,
    )
    return [
        # No tools -> no automatic function calling; no hidden "thinking" tokens,
        # which are billed against max_output_tokens and can leave an empty answer.
        {**base,
         "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
         "thinking_config": types.ThinkingConfig(thinking_budget=0)},
        # Newer models refuse to switch thinking off, but accept "think as little
        # as possible". Without this they think at full depth: measured 15-25 s
        # for answers that otherwise take ~4 s.
        {**base,
         "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
         "thinking_config": types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL)},
        {**base,
         "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
         "thinking_config": types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)},
        # Keep only the AFC switch; the model thinks as it likes.
        {**base,
         "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True)},
        base,
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_answer(question: str, chunks: list[RetrievedChunk],
                    info: dict | None = None, use_cache: bool = True) -> str:
    """Full answer as one string. `info` (optional) receives {"model", "cached"}."""
    return "".join(generate_answer_stream(question, chunks, info=info, use_cache=use_cache))


def generate_answer_stream(question: str, chunks: list[RetrievedChunk],
                           info: dict | None = None, use_cache: bool = True):
    """Yield the answer text as Gemini produces it.

    Fallbacks (next config, retry, next model) happen only WHILE NOTHING HAS BEEN
    SENT yet. Once the first piece of text is out we are committed to that call.
    `info`, if given, is filled with the model that answered and whether it came
    from the cache — the web UI shows both.
    """
    info = info if info is not None else {}
    info.update(model=None, cached=False)

    if not chunks:
        yield NO_CONTEXT_ANSWER
        return

    if use_cache:
        hit = answer_cache.get(question, chunks, salt=SYSTEM_PROMPT)
        if hit:
            info.update(model=hit.get("model"), cached=True)
            yield hit["answer"]
            return

    prompt = USER_TEMPLATE.format(context=build_context(chunks), question=question)
    models = candidate_models()
    if not models:
        raise FriendlyError(
            "Tất cả model Gemini đã cấu hình đều đang hết hạn mức (429). "
            "Thêm model khác vào GEMINI_FALLBACK_MODELS trong .env, hoặc chờ hạn mức được đặt lại.",
            status=429,
        )

    last_error: Exception | None = None
    tried: list[str] = []
    overloaded = 0
    for position, model in enumerate(models):
        tried.append(model)
        pieces: list[str] = []
        delays = RETRY_DELAYS if position == 0 else ()
        try:
            for text in _stream_one_model(model, prompt, delays):
                pieces.append(text)
                yield text
        except _Committed as committed:
            raise committed.original  # Text already sent; cannot switch model now.
        except Exception as exc:
            last_error = exc
            if _is_quota(exc) or _is_not_found(exc):
                _block(model, exc)
                print(f"[generator] {model}: {'hết hạn mức' if _is_quota(exc) else 'không dùng được'}"
                      f" -> thử model kế tiếp")
                continue
            if _is_transient(exc):
                overloaded += 1
                print(f"[generator] {model}: quá tải ({str(exc)[:80]})")
                if overloaded >= MAX_OVERLOADED_MODELS:
                    break
                continue
            print(f"[generator] {model}: lỗi {str(exc)[:200]}")
            raise explain_api_error(exc, model, tried) from exc

        if pieces:
            info["model"] = model
            # A refusal is not cached: the next model, or the same one next time,
            # may well find the answer in the same passages.
            if use_cache and not is_refusal("".join(pieces)):
                answer_cache.put(question, chunks, "".join(pieces), model, salt=SYSTEM_PROMPT)
            return
        # Model accepted the request but produced no text: try the next one.

    if last_error is not None:
        raise explain_api_error(last_error, tried[-1], tried) from last_error
    yield NO_CONTEXT_ANSWER


def is_refusal(answer: str) -> bool:
    text = answer.lower()
    return "không tìm thấy thông tin" in text or "không tìm thấy thông tin này" in text


class _Committed(Exception):
    """Wraps an error raised after text was already streamed to the client."""

    def __init__(self, original: Exception):
        super().__init__(str(original))
        self.original = original


def _stream_one_model(model: str, prompt: str, delays: tuple = RETRY_DELAYS):
    """Stream from ONE model: try config variants, retry overloads with backoff."""
    attempts = _attempts()
    start = _load_profile(model)
    offset = 0
    tries = 0
    while start + offset < len(attempts):
        index = start + offset
        sent_any = False
        try:
            stream = get_client().models.generate_content_stream(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(**attempts[index]),
            )
            for chunk in stream:
                text = getattr(chunk, "text", None) or ""
                if not text:
                    continue
                if not sent_any:
                    _save_profile(model, index)
                    sent_any = True
                yield text
            return  # Finished (possibly with no text — the caller handles that).
        except Exception as exc:
            if sent_any:
                raise _Committed(exc) from exc
            if _is_invalid_argument(exc):
                offset += 1          # Drop one optional config field and retry.
                continue
            if _is_transient(exc) and tries < len(delays):
                time.sleep(delays[tries])
                tries += 1
                continue
            raise
    raise RuntimeError(f"400 INVALID_ARGUMENT: model {model} từ chối mọi cấu hình gửi lên.")


def http_status_for(exc: Exception) -> int:
    """Map an SDK failure onto the HTTP status the web API should return.

    Decided from the ORIGINAL exception, never from the translated message —
    the Vietnamese text for a 503 mentions the word "quota" and would otherwise
    be misread as a rate-limit error.
    """
    if isinstance(exc, FriendlyError):
        return exc.status
    if _is_quota(exc):
        return 429  # Too Many Requests
    if _is_transient(exc):
        return 503  # Service Unavailable
    if _is_not_found(exc):
        return 502  # Bad Gateway: upstream is misconfigured, not overloaded
    return 500


def explain_api_error(exc: Exception, model: str | None = None,
                      tried: list[str] | None = None) -> Exception:
    """Turn an API failure into an instruction the user can act on.

    Public because api.py needs it too: a web server must never let a raw SDK
    exception escape an endpoint.
    """
    message = str(exc)
    model = model or config.GEMINI_MODEL
    tried_list = tried or [model]
    tried_str = ", ".join(tried_list)

    if isinstance(exc, FriendlyError):
        return exc  # Already written for the user.

    if _is_not_found(exc):
        return FriendlyError(
            f"Model '{model}' không dùng được (Google thường khai tử model cũ).\n"
            f"Chạy `python list_models.py` để xem model nào còn khả dụng, "
            f"rồi sửa GEMINI_MODEL / GEMINI_FALLBACK_MODELS trong file .env.\n\nChi tiết: {message}",
            status=502,
        )

    if _is_quota(exc):
        limit = re.search(r"limit:\s*(\d+)", message)
        per_day = "PerDay" in message
        lines = [f"Đã hết hạn mức miễn phí của Gemini ở mọi model còn dùng được ({tried_str})."]
        if limit:
            scope = "mỗi NGÀY" if per_day else "mỗi phút"
            lines.append(f"Hạn mức của model cuối cùng: {limit.group(1)} request {scope}.")
        lines.append(
            "Thêm model khác vào GEMINI_FALLBACK_MODELS trong .env (xem `python list_models.py`) "
            "hoặc chờ hạn mức được đặt lại. Xem hạn mức thực tế tại https://ai.dev/rate-limit"
        )
        return FriendlyError("\n".join(lines), status=429)

    if _is_transient(exc):
        return FriendlyError(
            f"Máy chủ Gemini đang quá tải (503). Đã thử {len(tried_list)} model ({tried_str}); "
            f"model đầu tiên được thử lại {len(RETRY_DELAYS) + 1} lần, nhưng vẫn không được.\n"
            "Đây là sự cố phía Google, không phải lỗi code. Đợi vài phút rồi hỏi lại.",
            status=503,
        )

    return exc
