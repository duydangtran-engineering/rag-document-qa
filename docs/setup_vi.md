# Hệ thống tra cứu tài liệu nội bộ (RAG) — Giai đoạn 1

Bản CLI chạy trong VSCode: nạp tài liệu → hỏi → trả lời **có trích dẫn** tên file, số trang, số Điều.

Giai đoạn 2 (web + voice) sẽ dựng lên trên đúng phần lõi này.

---

## 1. Chuẩn bị VSCode (làm 1 lần)

1. Cài **Python 3.11** (khuyến nghị) từ python.org. Khi cài nhớ tick **"Add Python to PATH"**.
2. Trong VSCode, cài 2 extension:
   - `Python` (Microsoft)
   - `Pylance` (Microsoft)
3. Mở thư mục `rag-tra-cuu` bằng VSCode: **File → Open Folder**.

## 2. Tạo môi trường ảo

Mở terminal trong VSCode (`Ctrl + ~`), gõ:

```bash
python -m venv .venv
```

Kích hoạt:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows CMD
.\.venv\Scripts\activate.bat
```

> Nếu PowerShell báo lỗi "running scripts is disabled", chạy một lần:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

Thấy `(.venv)` ở đầu dòng terminal là đúng.

Sau đó bấm `Ctrl+Shift+P` → `Python: Select Interpreter` → chọn cái nằm trong `.venv`.

## 3. Cài thư viện

**Bước 3a — cài PyTorch bản CUDA trước** (để dùng RTX 3060, nhanh hơn CPU khoảng 10–20 lần):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

> Nếu báo `No matching distribution found for torch`: bản Python của bạn quá mới so với
> index CUDA đang dùng. Kiểm tra bằng `python --version` rồi xem bảng ở mục "Lỗi thường gặp".

**Bước 3b — cài phần còn lại:**

```bash
pip install -r requirements.txt
```

Kiểm tra GPU đã nhận chưa:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Phải in ra `True`. Nếu `False` thì vẫn chạy được nhưng chậm — cài lại bước 3a.

## 4. Lấy API key Gemini

1. Vào https://aistudio.google.com/apikey → **Create API key** (miễn phí).
2. Copy file `.env.example` thành `.env`:

```bash
# Windows
copy .env.example .env
```

3. Mở `.env`, dán key vào dòng `GEMINI_API_KEY=`.

> `.env` đã nằm trong `.gitignore` nên sẽ không bị đẩy lên GitHub.

## 5. Nạp tài liệu

Copy các file PDF / DOCX / TXT vào thư mục `data/documents/`, rồi:

```bash
python ingest.py
```

Lần đầu sẽ tải model embedding `bge-m3` (~2 GB) từ HuggingFace — chỉ tải một lần.

Nạp lại từ đầu (khi đổi tài liệu hoặc đổi tham số chunk):

```bash
python ingest.py --reset
```

## 6. Hỏi đáp

```bash
python ask.py
```

Hoặc hỏi một câu rồi thoát luôn:

```bash
python ask.py "Mức bồi thường tối đa cho điều trị nội trú là bao nhiêu?"
```

Muốn xem hệ thống đã lấy đúng đoạn nào (rất hữu ích khi debug và khi viết báo cáo):

```bash
python ask.py --debug "câu hỏi của bạn"
```

## 7. Chạy giao diện web

```bash
pip install -r requirements.txt      # cài thêm fastapi + uvicorn
uvicorn api:app
```

Mở http://127.0.0.1:8000

**Đừng thêm `--reload` khi dùng bình thường.** Nó theo dõi cả thư mục `.venv`;
mỗi lần Windows hoặc phần mềm diệt virus chạm vào file trong `site-packages`
(hay xảy ra sau khi máy ngủ dậy), uvicorn tưởng code đổi và khởi động lại, cắt
đứt request đang chạy dở — trình duyệt sẽ quay vòng mãi.

Khi đang sửa code và muốn tự nạp lại, loại `.venv` ra:

```bash
uvicorn api:app --reload --reload-exclude ".venv/*"
```

Khác biệt so với `ask.py`: model embedding được nạp **một lần lúc khởi động** rồi giữ trong
bộ nhớ. CLI nạp lại mỗi lần chạy — đó là chỗ tốn thời gian nhất.

| Endpoint | Việc |
|---|---|
| `GET /` | Giao diện chat |
| `GET /api/status` | Số chunk, danh sách tài liệu, model, thiết bị |
| `POST /api/ask` | `{"question": "...", "top_k": 5}` → câu trả lời + nguồn trích dẫn |

Xem tài liệu API tự sinh tại http://127.0.0.1:8000/docs

---

## Cấu trúc code

```
rag-tra-cuu/
├── data/documents/     ← bạn bỏ tài liệu vào đây
├── storage/chroma/     ← vector database, tự sinh ra
├── src/
│   ├── config.py       ← mọi tham số điều chỉnh được
│   ├── loader.py       ← PDF/DOCX → text + số trang
│   ├── chunker.py      ← cắt đoạn có overlap, bắt "Điều N"
│   ├── embedder.py     ← bge-m3 chạy trên GPU
│   ├── vector_store.py ← ChromaDB
│   ├── retriever.py    ← tìm top-K đoạn liên quan
│   └── generator.py    ← Gemini + prompt chống bịa
├── ingest.py           ← script nạp tài liệu
└── ask.py              ← script hỏi đáp
```

Luồng chạy:

```
PDF/DOCX → loader → chunker → embedder → ChromaDB
                                              ↓
câu hỏi → embedder → retriever (top-K) → generator (Gemini) → câu trả lời + [1][2]
```

## Các tham số nên thử nghiệm (và ghi vào báo cáo)

Sửa trong `.env`, chạy lại `python ingest.py --reset`, so sánh kết quả:

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `CHUNK_SIZE` | 800 | Chunk nhỏ → trích dẫn chính xác hơn nhưng dễ mất ngữ cảnh |
| `CHUNK_OVERLAP` | 150 | Chống cắt đứt câu trả lời ở ranh giới chunk |
| `TOP_K` | 5 | Số đoạn đưa cho LLM. Tăng → ít bỏ sót, nhưng nhiễu hơn |
| `MAX_DISTANCE` | 1.0 | Ngưỡng loại đoạn không liên quan. Giảm → nghiêm hơn |
| `GEMINI_MODEL` | gemini-2.5-flash | Đổi sang `gemini-2.0-flash` nếu bị giới hạn quota |

## Lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `Chưa có GEMINI_API_KEY` | Chưa tạo file `.env` hoặc chưa dán key |
| `No matching distribution found for torch` | Index `cu121` chỉ có wheel tới Python 3.12. Đổi sang `cu126`, hoặc cài Python 3.12 rồi tạo lại venv |
| `torch.cuda.is_available()` là `False` | Cài lại torch theo bước 3a, cập nhật driver NVIDIA |
| Nạp PDF ra 0 trang | PDF là ảnh scan → cần OCR (sẽ làm ở giai đoạn sau) |
| `429 RESOURCE_EXHAUSTED` | Hết quota Gemini free trong phút đó, đợi 60 giây |
| Trả lời "không tìm thấy" dù tài liệu có | Tăng `TOP_K`, tăng `MAX_DISTANCE`, hoặc chạy `--debug` xem retriever lấy gì |

## Lộ trình tiếp theo

- [ ] **GĐ 2** — API FastAPI: `/upload`, `/ask`; giao diện web upload + chat
- [ ] **GĐ 3** — Voice: Web Speech API hoặc Whisper cho input, TTS cho output
- [ ] **GĐ 4** — Chụp ảnh đoạn trích: PyMuPDF render vùng chứa câu trả lời thành PNG
- [ ] **GĐ 5** — OCR cho PDF scan, hybrid search (BM25 + vector), reranker
- [ ] **GĐ 6** — Bộ test đánh giá: 20–30 câu hỏi có đáp án chuẩn, đo độ chính xác