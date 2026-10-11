"""Đo chất lượng TRUY HỒI của hệ thống RAG — KHÔNG gọi Gemini, không tốn quota.

Chạy từ thư mục gốc project (đã activate .venv):
    python eval/eval_retrieval.py                # chạy toàn bộ bộ câu hỏi
    python eval/eval_retrieval.py --show-fail    # in chi tiết các câu truy hồi sai
    python eval/eval_retrieval.py --no-rerank    # tắt reranker để so sánh trước/sau
    python eval/eval_retrieval.py --k 8          # thử TOP_K khác mà không sửa .env

Đo 3 tầng:
  1. CHỈ VECTOR      : tìm kiếm embedding thuần trên câu hỏi gốc (mốc so sánh).
  2. PIPELINE XẾP HẠNG: mở rộng viết tắt + vector + rerank (nếu bật) — đây là thứ
                        tự thật sự hệ thống dùng để chọn TOP_K chunk.
  3. NGỮ CẢNH CUỐI   : retriever.retrieve (tách câu nhiều ý + chunk liền kề) —
                        đúng thứ Gemini nhìn thấy. Gold không có ở đây thì LLM
                        chắc chắn trả lời sai hoặc phải từ chối.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config, embedder, retriever, vector_store  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
RAW_DEPTH = 20  # số kết quả xét khi tính hạng


def norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def groups_rank(gold: list[list[str]], texts: list[str]) -> int | None:
    """Hạng (1-based) nhỏ nhất mà TẤT CẢ các nhóm gold đều đã xuất hiện. None = không đủ."""
    worst = 0
    for group in gold:
        alts = [norm(a) for a in group]
        rank = next((i + 1 for i, t in enumerate(texts) if any(a in t for a in alts)), None)
        if rank is None:
            return None
        worst = max(worst, rank)
    return worst


def load_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["questions"] if isinstance(data, dict) else data


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá truy hồi (không tốn quota API)")
    parser.add_argument("--set", default=str(EVAL_DIR / "eval_set.json"))
    parser.add_argument("--k", type=int, default=None, help="Ghi đè TOP_K")
    parser.add_argument("--no-rerank", action="store_true", help="Tắt reranker")
    parser.add_argument("--show-fail", action="store_true")
    args = parser.parse_args()

    if args.k:
        config.TOP_K = args.k
    if args.no_rerank:
        config.RERANK_ENABLED = False
    k = config.TOP_K
    rerank_on = config.RERANK_ENABLED

    questions = load_questions(Path(args.set))
    total_chunks = vector_store.count()
    print(f"DB: {total_chunks} chunk | TOP_K={k} | MAX_DISTANCE={config.MAX_DISTANCE} | "
          f"NEIGHBOR_WINDOW={config.NEIGHBOR_WINDOW} | thiết bị: {embedder.describe_device()}")
    print(f"Rerank: {config.RERANK_MODEL if rerank_on else 'TẮT'}")

    # Nạp model + chạy nháp 1 lần để thời gian đo không tính lúc khởi tạo GPU.
    embedder.get_model()
    retriever.retrieve("khởi động")

    rows: list[dict] = []
    for q in questions:
        question = q["question"]
        gold = q.get("gold") or []

        # Tầng 1: chỉ vector, câu hỏi gốc
        vec = vector_store.search(embedder.embed_query(question), RAW_DEPTH)
        vec_texts = [norm(h["text"]) for h in vec]

        # Tầng 2: pipeline xếp hạng (viết tắt + vector + rerank)
        ranked = retriever.rank(question, RAW_DEPTH)
        ranked_texts = [norm(h["text"]) for h in ranked]

        # Tầng 3: đúng pipeline production
        t0 = time.perf_counter()
        final = retriever.retrieve(question)
        ms = (time.perf_counter() - t0) * 1000
        final_texts = [norm(c.text) for c in final]

        row = {
            "id": q["id"], "type": q["type"], "question": question,
            "top1_sim": round(1 - vec[0]["distance"], 3) if vec else None,
            "top1_rerank": round(ranked[0]["rerank_score"], 3)
                           if ranked and "rerank_score" in ranked[0] else None,
            "retrieval_ms": round(ms),
            "n_context": len(final),
            "top_citations": [h["metadata"].get("citation") for h in ranked[:k]],
        }
        if gold:
            row["vec_rank"] = groups_rank(gold, vec_texts)
            row["rank"] = groups_rank(gold, ranked_texts)
            row["ctx_hit"] = groups_rank(gold, final_texts) is not None
            if q.get("source"):
                row["ctx_purity"] = round(
                    sum(c.source == q["source"] for c in final) / max(1, len(final)), 2)
        rows.append(row)

    scored = [r for r in rows if "rank" in r]
    in_scope = [r for r in rows if r["type"] != "oos"]
    oos = [r for r in rows if r["type"] == "oos"]
    n = len(scored)

    def hit(rs, depth, field="rank"):
        return sum(1 for r in rs if r[field] and r[field] <= depth)

    def mrr(rs, field="rank"):
        return sum(1 / r[field] for r in rs if r[field]) / len(rs) if rs else 0

    def pct(a, b):
        return f"{a}/{b} ({a / b:.0%})" if b else "-"

    label = "vector+rerank" if rerank_on else "vector+viết tắt"
    print("\n=== TỔNG HỢP ===")
    print(f"Câu có đáp án chuẩn : {n}")
    print(f"{'':20}{'chỉ vector':>16}{label:>18}")
    for depth in (1, 3, k, RAW_DEPTH):
        print(f"{'Hit@' + str(depth):20}{pct(hit(scored, depth, 'vec_rank'), n):>16}"
              f"{pct(hit(scored, depth), n):>18}")
    print(f"{'MRR@' + str(RAW_DEPTH):20}{mrr(scored, 'vec_rank'):>16.3f}{mrr(scored):>18.3f}")
    ctx = sum(r["ctx_hit"] for r in scored)
    print(f"\nGold nằm trong ngữ cảnh gửi LLM: {pct(ctx, n)}   <- CHỈ SỐ QUAN TRỌNG NHẤT")
    purities = [r["ctx_purity"] for r in scored if "ctx_purity" in r]
    if purities:
        print(f"Độ sạch nguồn (chunk đúng file) trung bình: {statistics.mean(purities):.0%}")
    print(f"Số chunk gửi LLM    : TB {statistics.mean(r['n_context'] for r in rows):.1f}")
    print(f"Thời gian truy hồi  : TB {statistics.mean(r['retrieval_ms'] for r in rows):.0f} ms, "
          f"max {max(r['retrieval_ms'] for r in rows)} ms")

    print("\n=== THEO LOẠI CÂU HỎI ===")
    for t in sorted({r["type"] for r in scored}):
        rs = [r for r in scored if r["type"] == t]
        print(f"  {t:<13} n={len(rs):<2}  Hit@1 {hit(rs, 1)}/{len(rs)}  Hit@{k} {hit(rs, k)}/{len(rs)}  "
              f"ngữ cảnh {sum(r['ctx_hit'] for r in rs)}/{len(rs)}")

    print("\n=== NGƯỠNG TỪ CHỐI (câu ngoài tài liệu) ===")
    field, name = ("top1_rerank", "điểm rerank") if rerank_on else ("top1_sim", "cosine similarity")
    ins = sorted(r[field] for r in in_scope if r[field] is not None)
    outs = sorted(r[field] for r in oos if r[field] is not None)
    if ins:
        print(f"Top-1 {name} câu TRONG tài liệu: min {ins[0]}  median {statistics.median(ins)}")
    if outs:
        print(f"Top-1 {name} câu NGOÀI tài liệu: max {outs[-1]}  median {statistics.median(outs)}")
        for r in sorted(oos, key=lambda r: -(r[field] or 0)):
            print(f"      {r[field]}  [{r['id']}] {r['question']}")
    if ins and outs:
        if outs[-1] < ins[0]:
            thr = round((outs[-1] + ins[0]) / 2, 3)
            env = f"RERANK_MIN_SCORE={thr}" if rerank_on else f"MAX_DISTANCE={1 - thr:.2f}"
            print(f"-> TÁCH ĐƯỢC. Có thể đặt {env} trong .env "
                  f"(khoảng trống {outs[-1]} → {ins[0]}; bộ câu hỏi còn nhỏ nên chọn thận trọng).")
        else:
            print("-> Hai vùng CHỒNG nhau: không chặn được bằng ngưỡng, vẫn phải dựa vào prompt từ chối.")

    fails = [r for r in scored if not r["ctx_hit"] or (r["rank"] or 99) > k]
    if fails:
        print(f"\n=== {len(fails)} CÂU CÓ VẤN ĐỀ ===")
        print("  (hạng = vị trí đoạn đúng sau pipeline xếp hạng; câu nhiều ý thì hạng không có ý nghĩa)")
        for r in fails:
            status = "MẤT KHỎI NGỮ CẢNH" if not r["ctx_hit"] else "hạng thấp nhưng vẫn vào ngữ cảnh"
            print(f"  [{r['id']}] ({r['type']}) hạng={r['rank']} (vector: {r['vec_rank']})  "
                  f"{status}  — {r['question']}")
            if args.show_fail:
                for c in r["top_citations"]:
                    print(f"        · {c}")

    out_dir = EVAL_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    tag = "rerank" if rerank_on else "norerank"
    out = out_dir / f"retrieval_{datetime.now():%Y%m%d_%H%M}_k{k}_{tag}.json"
    out.write_text(json.dumps({
        "config": {"TOP_K": k, "MAX_DISTANCE": config.MAX_DISTANCE,
                   "NEIGHBOR_WINDOW": config.NEIGHBOR_WINDOW, "CHUNK_SIZE": config.CHUNK_SIZE,
                   "RERANK": config.RERANK_MODEL if rerank_on else None, "chunks": total_chunks},
        "summary": {"n": n, "hit1": hit(scored, 1), "hit3": hit(scored, 3), f"hit{k}": hit(scored, k),
                    "mrr": round(mrr(scored), 3), "ctx_hit": ctx,
                    "vector_only": {"hit1": hit(scored, 1, "vec_rank"), f"hit{k}": hit(scored, k, "vec_rank")}},
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nĐã lưu chi tiết: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
