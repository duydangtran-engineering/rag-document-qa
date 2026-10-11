"""Đo chất lượng CÂU TRẢ LỜI (có gọi Gemini — mỗi câu tốn 1 request).

Vì quota miễn phí rất ít, script này:
  - mặc định chỉ chạy 10 câu mỗi lần (--limit), nghỉ giữa các lần gọi (--sleep);
  - LƯU KẾT QUẢ vào eval/results/answers_cache.json, lần sau tự bỏ qua câu đã chạy
    với cùng model  -> chạy rải nhiều ngày vẫn gộp được đủ bộ;
  - gặp lỗi 429 (hết quota) thì dừng êm và giữ lại những gì đã có.

Chạy:
    python eval/eval_answers.py                 # 10 câu chưa chạy tiếp theo
    python eval/eval_answers.py --limit 5 --types oos,article
    python eval/eval_answers.py --ids L07,L09,F03
    python eval/eval_answers.py --report        # chỉ in báo cáo từ cache, không gọi API

Chấm tự động (thô, cần người xem lại):
  - keywords : mọi nhóm answer_keywords đều có trong câu trả lời
  - cited    : có ít nhất một trích dẫn [n], và mọi [n] đều trỏ tới nguồn có thật
  - refusal  : câu ngoài tài liệu thì PHẢI từ chối, câu trong tài liệu thì KHÔNG được từ chối
Xuất thêm eval/results/answers_review.csv (mở bằng Excel) có cột trống để tự chấm
"đúng / sai / bịa" — đây mới là con số đưa vào báo cáo.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config, generator, retriever  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
RESULTS = EVAL_DIR / "results"
CACHE = RESULTS / "answers_cache.json"

REFUSAL_MARKERS = ("không tìm thấy thông tin", "không tìm thấy", "không có thông tin",
                   "tài liệu không đề cập", "không được đề cập")
CITATION = re.compile(r"\[(\d+)\]")


def norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def check(q: dict, answer: str, n_sources: int) -> dict:
    a = norm(answer)
    refused = any(m in a for m in REFUSAL_MARKERS)
    cites = [int(x) for x in CITATION.findall(answer)]
    result = {
        "refused": refused,
        "cited": bool(cites) and all(1 <= c <= n_sources for c in cites),
        "bad_citation": [c for c in cites if not 1 <= c <= n_sources],
    }
    if q.get("expect_refusal"):
        result["pass"] = refused
        return result
    groups = q.get("answer_keywords") or []
    missing = [g[0] for g in groups if not any(norm(alt) in a for alt in g)]
    result["missing_keywords"] = missing
    result["pass"] = (not refused) and not missing and result["cited"]
    return result


def load_cache() -> dict:
    try:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_cache(cache: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def report(cache: dict, questions: list[dict]) -> None:
    model = config.GEMINI_MODEL
    done = [(q, cache[f"{model}|{q['id']}"]) for q in questions if f"{model}|{q['id']}" in cache]
    if not done:
        print("Chưa có kết quả nào cho model", model)
        return

    print(f"\n=== KẾT QUẢ ({len(done)}/{len(questions)} câu, model {model}) ===")
    by_type: dict[str, list[bool]] = {}
    for q, r in done:
        by_type.setdefault(q["type"], []).append(r["check"]["pass"])
        mark = "OK " if r["check"]["pass"] else "XX "
        extra = ""
        if not r["check"]["pass"]:
            if r["check"].get("missing_keywords"):
                extra += f" thiếu={r['check']['missing_keywords']}"
            if not r["check"]["cited"] and not q.get("expect_refusal"):
                extra += " không/ sai trích dẫn"
            if r["check"]["refused"] != bool(q.get("expect_refusal")):
                extra += " từ chối SAI" if r["check"]["refused"] else " KHÔNG từ chối (nguy cơ bịa)"
        print(f"  {mark}[{q['id']}] {r['generation_ms']/1000:5.1f}s{extra}")

    print("\nTheo loại:")
    for t, xs in sorted(by_type.items()):
        print(f"  {t:<13} {sum(xs)}/{len(xs)}")
    total = [r["check"]["pass"] for _, r in done]
    times = [r["generation_ms"] for _, r in done]
    print(f"\nĐạt tự động: {sum(total)}/{len(total)} ({sum(total)/len(total):.0%})"
          f" | thời gian sinh TB {sum(times)/len(times)/1000:.1f}s, max {max(times)/1000:.1f}s")

    out = RESULTS / "answers_review.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig để Excel đọc đúng tiếng Việt
        w = csv.writer(f)
        w.writerow(["id", "type", "question", "answer", "auto_pass", "sources",
                    "CHAM_TAY (dung/thieu/sai/bia)", "ghi_chu"])
        for q, r in done:
            w.writerow([q["id"], q["type"], q["question"], r["answer"], r["check"]["pass"],
                        " | ".join(r["sources"]), "", ""])
    print(f"Đã xuất bảng chấm tay: {out.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá câu trả lời (tốn quota API)")
    parser.add_argument("--set", default=str(EVAL_DIR / "eval_set.json"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--sleep", type=float, default=6.0, help="giây nghỉ giữa 2 lần gọi")
    parser.add_argument("--ids", default="")
    parser.add_argument("--types", default="")
    parser.add_argument("--force", action="store_true", help="chạy lại cả câu đã có trong cache")
    parser.add_argument("--report", action="store_true", help="chỉ in báo cáo, không gọi API")
    args = parser.parse_args()

    data = json.loads(Path(args.set).read_text(encoding="utf-8"))
    questions = data["questions"] if isinstance(data, dict) else data
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",")}
        questions = [q for q in questions if q["id"] in wanted]
    if args.types:
        wanted = {x.strip() for x in args.types.split(",")}
        questions = [q for q in questions if q["type"] in wanted]

    cache = load_cache()
    if args.report:
        report(cache, questions)
        return

    model = config.GEMINI_MODEL
    pending = [q for q in questions if args.force or f"{model}|{q['id']}" not in cache]
    todo = pending[: args.limit]
    print(f"Model {model}: sẽ chạy {len(todo)} câu (sau lần này còn {len(pending) - len(todo)} câu chưa chạy)")

    for i, q in enumerate(todo, 1):
        chunks = retriever.retrieve(q["question"])
        t0 = time.perf_counter()
        try:
            info: dict = {}
            answer = generator.generate_answer(q["question"], chunks, info=info,
                                               use_cache=not args.force)
        except Exception as exc:
            msg = str(generator.explain_api_error(exc))
            print(f"  [{q['id']}] LỖI: {msg.splitlines()[0]}")
            if generator.http_status_for(exc) == 429:
                print("  Hết quota -> dừng, kết quả đã chạy được vẫn được lưu.")
                break
            continue
        ms = (time.perf_counter() - t0) * 1000
        result = check(q, answer, len(chunks))
        cache[f"{model}|{q['id']}"] = {
            "question": q["question"], "answer": answer, "check": result,
            "generation_ms": round(ms), "sources": [c.citation for c in chunks],
            "model": info.get("model"), "cached": bool(info.get("cached")),
        }
        save_cache(cache)
        via = " (bộ nhớ đệm)" if info.get("cached") else (f" [{info['model']}]" if info.get("model") else "")
        print(f"  ({i}/{len(todo)}) [{q['id']}] {'OK' if result['pass'] else 'XX'}  {ms/1000:.1f}s{via}")
        if i < len(todo):
            time.sleep(args.sleep)

    report(cache, questions)


if __name__ == "__main__":
    main()
