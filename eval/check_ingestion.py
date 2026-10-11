"""Kiểm tra KHÂU NẠP TÀI LIỆU — không cần GPU, không cần Gemini, không đụng vào database.

Chạy từ thư mục gốc project:
    python eval/check_ingestion.py              # báo cáo chất lượng nạp cho mọi file trong data/documents
    python eval/check_ingestion.py --show 5     # in thêm 5 chunk mẫu mỗi file
    python eval/check_ingestion.py --heading-tests   # chỉ chạy bộ test nhận diện tiêu đề

Báo cáo trả lời: loader có bỏ đúng header/footer, mục lục không? tiêu đề có bị nhận sai không?
còn chunk quá ngắn không? bảng có giữ được dòng tiêu đề cột không?
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.chunker import chunk_blocks  # noqa: E402
from src.loader import is_heading_line, load_pdf_with_report, load_document  # noqa: E402

# (dòng, có phải tiêu đề không). Mỗi ca là một lỗi thật đã gặp trong 450 chunk đầu tiên.
HEADING_CASES = [
    ("Điều 4. Điều khoản chuyển tiếp", True),
    ("Chương II", True),
    ("PHẦN II: CÁC KẾT QUẢ CHÍNH", True),
    ("PHỤ LỤC 3 PHƯƠNG PHÁP PHÂN TÍCH ĐỊNH LƯỢNG", True),
    ("II. THỰC TRẠNG THAM GIA BHXH Ở VIỆT NAM", True),
    ("1.2.3. Chế độ hưởng bảo hiểm xã hội tự nguyện", True),
    ("1.2. Vector đặc trưng — mô tả một thứ bằng vài con số", True),
    ("TÓM TẮT TOÀN VĂN", True),
    ("1. MỤC TIÊU CHUNG", True),
    # --- phải KHÔNG là tiêu đề ---
    ("Chương IV, Luật Bảo hiểm xã hội.", False),          # dòng thân bài bị ngắt dòng
    ("Điều 18 như sau:", False),
    ("Điều 5 của Luật này quy định", False),
    ("MANHATTAN", False),                                 # một từ in hoa
    ("BRIGHT_THRESHOLD", False),                          # tên biến trong code
    ("0.1964 mean_S", False),                             # số liệu, không phải mục 0.1964
    ("2.5 triệu đồng mỗi tháng", False),                  # số tiền ở đầu dòng bị ngắt
    ("BIÊN (DY/DX) HỆ SỐ CHẶN", False),                   # tiêu đề cột bảng
    ("PHẦN 9. BẢNG TRA CỨU NHANH THUẬT NGỮ ............ 26", False),  # dòng mục lục
]


def run_heading_tests() -> bool:
    bad = [(line, want) for line, want in HEADING_CASES if is_heading_line(line) != want]
    print(f"Test nhận diện tiêu đề: {len(HEADING_CASES) - len(bad)}/{len(HEADING_CASES)} đúng")
    for line, want in bad:
        print(f"   SAI: {line!r} -> mong đợi {'là' if want else 'KHÔNG là'} tiêu đề")
    return not bad


def report_file(path: Path, show: int) -> None:
    if path.suffix.lower() == ".pdf":
        blocks, rep = load_pdf_with_report(path)
    else:
        blocks, rep = load_document(path), {}
    chunks = chunk_blocks(blocks, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

    print(f"\n=== {path.name} ===")
    if rep:
        print(f"Trang: {rep['pages']} -> giữ {rep['kept_pages']}   "
              f"(bỏ trang mục lục/danh mục: {rep['index_pages'] or 'không có'})")
        print(f"Dòng header/footer lặp đã loại: {rep['running_lines'] or 'không có'}")
        print(f"Bảng đọc được dưới dạng bảng: {rep['tables']}")
    lengths = [len(c.text) for c in chunks]
    print(f"Chunk: {len(chunks)}   độ dài TB {statistics.mean(lengths):.0f}  "
          f"min {min(lengths)}  max {max(lengths)}")
    table_chunks = [c for c in chunks if c.text.lstrip().startswith("|") or "\n| ---" in c.text]
    print(f"Chunk chứa bảng: {len(table_chunks)}")

    # Tiêu đề: số lượng, và các tiêu đề đáng ngờ
    headings = Counter(c.heading for c in chunks)
    print(f"Tiêu đề khác nhau: {len(headings)}")
    suspicious = [h for h in headings
                  if re.search(r"\.{4,}|\d\.\d{3,}", h) or len(h.split()) < 2 and h.isupper()]
    print(f"Tiêu đề đáng ngờ: {suspicious or 'không có'}")

    # Rác còn sót
    leftovers = []
    for key in rep.get("running_lines", []):
        n = sum(1 for c in chunks if key.lower() in re.sub(r"\s+", " ", c.text).lower())
        if n:
            leftovers.append((key, n))
    print(f"Header còn sót trong chunk: {leftovers or 'không còn'}")
    dots = sum(1 for c in chunks if re.search(r"\.{5,}", c.text))
    print(f"Chunk còn dòng chấm mục lục: {dots}")
    print(f"Chunk ngắn < {config.MIN_CHUNK_CHARS} ký tự: {sum(1 for n in lengths if n < config.MIN_CHUNK_CHARS)}")

    if show:
        print(f"\n--- {show} chunk mẫu ---")
        step = max(1, len(chunks) // show)
        for c in chunks[::step][:show]:
            print(f"[trang {c.page} · #{c.index}] đường dẫn: {c.path or '(không có)'}")
            print("   " + c.text[:300].replace("\n", "\n   ") + ("…" if len(c.text) > 300 else ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", type=int, default=0)
    parser.add_argument("--heading-tests", action="store_true")
    args = parser.parse_args()

    ok = run_heading_tests()
    if args.heading_tests:
        sys.exit(0 if ok else 1)
    for path in sorted(config.DOCUMENTS_DIR.glob("*")):
        if path.suffix.lower() in config.SUPPORTED_EXTENSIONS:
            report_file(path, args.show)


if __name__ == "__main__":
    main()
