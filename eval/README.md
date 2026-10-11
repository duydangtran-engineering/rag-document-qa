# Bộ đánh giá RAG

Hai script, chạy từ thư mục gốc `RAG_PBL5` (đã activate `.venv`):

| Script | Gọi Gemini? | Đo gì |
|---|---|---|
| `python eval/eval_retrieval.py --show-fail` (thêm `--no-rerank` để so sánh khi tắt reranker) | **Không** (chạy thoải mái) | Hit@1/3/K, MRR, gold có nằm trong ngữ cảnh gửi LLM không, ngưỡng MAX_DISTANCE, thời gian truy hồi |
| `python eval/eval_answers.py` | Có, 1 request/câu | Câu trả lời có đủ ý, có trích dẫn hợp lệ, có từ chối đúng lúc không. Lưu cache nên chạy rải nhiều ngày được |

Quy trình khuyên dùng mỗi lần sửa code:

1. Sửa chunking / retrieval → `python ingest.py --reset` → `python eval/eval_retrieval.py`
2. So sánh file `eval/results/retrieval_*.json` trước/sau. Chỉ số chính: **"Gold nằm trong ngữ cảnh gửi LLM"**.
3. Khi retrieval đã ổn → `python eval/eval_answers.py` (10 câu/ngày) → mở `eval/results/answers_review.csv` bằng Excel, chấm tay cột `CHAM_TAY`. Đây là số liệu đưa vào báo cáo.

`eval_set.json` hiện có 50 câu (43 câu có đáp án chuẩn, 5 câu ngoài tài liệu, 2 câu tóm tắt) chia theo loại:
fact, paraphrase, casual (văn nói, không dấu câu), article (hỏi theo số Điều/khoản), list, cross_chunk,
multi (2 ý), abbreviation (UBND, LĐPCT...), table, oos (ngoài tài liệu), summary.
Nên bổ sung thêm câu hỏi thật do bạn bè / giảng viên đặt thử, và câu hỏi cho mỗi tài liệu mới nạp.
