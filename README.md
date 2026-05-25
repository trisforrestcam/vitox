# Vietnamese Profanity Filter

Bộ lọc từ tục tiểu tiếng Việt dùng trong môi trường production, kết hợp 2 lớp: lọc theo danh sách từ (wordlist) siêu nhanh và phân loại bằng mô hình học sâu PhoBERT.

---

## 🔥 Tại sao nên dùng?

- **2 lớp bảo vệ**: Lớp wordlist xử lý nhanh các từ tục phổ biến, lớp ML (PhoBERT) bắt được ngôn ngữ xúc phạm tinh tế, ẩn ý mà wordlist không đủ.
- **Tự động giải mã teencode**: Chuyển đổi các từ viết tắt teen-code (vd: `dm` → `đụ mẹ`) trước khi kiểm tra.
- **Dễ tích hợp**: Dùng trực tiếp trong Python hoặc chạy server API với FastAPI.

---

## 🏗 Kiến trúc

Pipeline xử lý:

```
Văn bản thô → Chuẩn hóa → Giải mã teencode → Kiểm tra Wordlist
                                          │
                              (có từ tục) └─→ Trả kết quả ngay
                                          │
                              (sạch)      └─→ Kiểm tra ML → Trả kết quả
```

| Lớp | Công nghệ | Mô tả |
|-----|-----------|-------|
| **Wordlist** | `better_profanity` + danh sách từ tiếng Việt | Quét nhanh, khớp chính xác từ tục và biến thể teencode |
| **ML** | `visolex/phobert-v2-hsd` (transformer) | Phân tích ngữ cảnh, phát hiện ngôn ngữ xúc phạm ẩn ý |

---

## 📦 Yêu cầu

- **Python** >= 3.10
- **uv** package manager ([https://docs.astral.sh/uv/](https://docs.astral.sh/uv/))

---

## ⚡ Cài đặt

```bash
# 1. Clone repo và di chuyển vào thư mục
git clone <repo-url>
cd vi-profanity-filter

# 2. Cài đặt môi trường và dependencies
uv sync

# Hoặc cài ở chế độ editable
uv pip install -e .
```

---

## 🚀 Cách dùng

### 1. Dùng trong Python

```python
from vi_profanity_filter.filter import ViProfanityFilter

# Khởi tạo (lần đầu sẽ tải model ML, có thể hơi lâu)
filter = ViProfanityFilter()

# Kiểm tra văn bản có chứa từ tục không
result = filter.check("Câu này có từ đụ")
print(result)
# {
#   "label": "OFFENSIVE",
#   "score": 1.0,
#   "is_profane": True,
#   "matched_words": ["đụ"],
#   "layer_used": "wordlist"
# }

# Che từ tục thành ***
censored = filter.censor("Câu này có từ đụ")
print(censored)  # "Câu này có từ ***"
```

#### Tùy chỉnh khi khởi tạo

```python
# Bỏ qua ML, chỉ dùng wordlist (nhẹ, nhanh, không cần GPU)
filter = ViProfanityFilter(skip_ml=True)

# Điều chỉnh ngưỡng tin cậy của ML (mặc định 0.7)
filter = ViProfanityFilter(ml_threshold=0.8)

# Dùng file wordlist tùy chỉnh
filter = ViProfanityFilter(wordlist_path="đường/dẫn/wordlist.txt")
```

#### Ý nghĩa các trường trong kết quả `check()`

| Trường | Kiểu | Ý nghĩa |
|--------|------|---------|
| `label` | str | Nhãn dự đoán: `"OFFENSIVE"` hoặc `"CLEAN"` |
| `score` | float | Độ tin cậy / mức độ nghiêm trọng (0.0 → 1.0) |
| `is_profane` | bool | `True` nếu văn bản bị đánh dấu là tục tiểu |
| `matched_words` | list | Danh sách từ tục tìm thấy (nếu có) |
| `layer_used` | str | Lớp phát hiện: `"wordlist"`, `"ml"` hoặc `"none"` |

---

### 2. Chạy server FastAPI

```bash
uv run uvicorn vi_profanity_filter.api.server:app \
  --host 0.0.0.0 --port 8000 --workers 1
```

#### Các endpoint

| Method | Đường dẫn | Chức năng |
|--------|-----------|-----------|
| GET | `/health` | Kiểm tra server có sống không |
| POST | `/check` | Kiểm tra văn bản có từ tục không |
| POST | `/censor` | Trả về văn bản đã che từ tục |

#### Ví dụ gọi API bằng `curl`

```bash
# Kiểm tra
curl -X POST http://localhost:8000/check \
  -H "Content-Type: application/json" \
  -d '{"text": "dm mày"}'

# Che từ tục
curl -X POST http://localhost:8000/censor \
  -H "Content-Type: application/json" \
  -d '{"text": "Câu này có từ đụ"}'
```

---

## 🧪 Chạy test

```bash
uv run pytest
```

---

## 🐳 Docker (tùy chọn)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install uv && uv pip install --system -e "."
CMD ["uvicorn", "vi_profanity_filter.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

---

## 📁 Cấu trúc thư mục

```
vi-profanity-filter/
├── pyproject.toml              # Cấu hình dự án & dependencies
├── README.md
├── plan.md
├── tests/
│   ├── __init__.py
│   └── test_filter.py          # Unit tests
└── vi_profanity_filter/
    ├── __init__.py
    ├── filter.py               # API chính (ViProfanityFilter)
    ├── data/
    │   └── vi_wordlist.txt     # Danh sách từ tục tiếng Việt
    ├── utils/
    │   ├── __init__.py
    │   ├── normalizer.py       # Chuẩn hóa văn bản
    │   └── teencode.py         # Giải mã teencode
    ├── layers/
    │   ├── __init__.py
    │   ├── wordlist_layer.py   # Lớp lọc wordlist
    │   └── ml_layer.py         # Lớp ML (PhoBERT)
    └── api/
        ├── __init__.py
        └── server.py           # Ứng dụng FastAPI
```

---

## 💡 Tips

- **Lần chạy đầu tiên** sẽ tự động tải model PhoBERT từ Hugging Face (~500MB), hãy đảm bảo có kết nối mạng.
- Nếu chỉ cần lọc đơn giản, không cần độ chính xác ngữ cảnh cao, hãy dùng `skip_ml=True` để tiết kiệm RAM và tăng tốc độ.
- Muốn bổ sung từ mới? Chỉ cần thêm vào file `vi_profanity_filter/data/vi_wordlist.txt` (mỗi từ một dòng).
