# Vietnamese Profanity Filter — Implementation Plan

## Tổng quan kiến trúc

Hệ thống gồm **2 lớp filter** chạy nối tiếp nhau để cân bằng tốc độ và độ chính xác:

```
Input text
    │
    ▼
┌─────────────────────────┐
│  Layer 1: Wordlist       │  ← nhanh, bắt từ rõ ràng + leetspeak
│  (better_profanity)      │     ~1ms
└────────────┬────────────┘
             │ CLEAN → tiếp tục Layer 2
             │ OFFENSIVE/HATE → trả về ngay
             ▼
┌─────────────────────────┐
│  Layer 2: ML Model       │  ← hiểu ngữ cảnh
│  (phobert-v2-hsd)        │     ~300ms CPU
└────────────┬────────────┘
             │
             ▼
        Kết quả cuối
   CLEAN / OFFENSIVE / HATE
```

---

## Cấu trúc thư mục

```
vi_profanity_filter/
├── __init__.py
├── filter.py               # Class chính — entry point
├── layers/
│   ├── __init__.py
│   ├── wordlist_layer.py   # Layer 1: better_profanity wrapper
│   └── ml_layer.py         # Layer 2: PhoBERT wrapper
├── utils/
│   ├── __init__.py
│   ├── normalizer.py       # Chuẩn hóa text trước khi filter
│   └── teencode.py         # Convert teencode → chuẩn (đ.m → đụ mẹ)
├── data/
│   └── vi_wordlist.txt     # Wordlist tiếng Việt tự build
├── api/
│   ├── __init__.py
│   └── server.py           # FastAPI server để self-host
├── requirements.txt
└── README.md
```

---

## Chi tiết từng file

### `requirements.txt`

```txt
better-profanity==0.7.0
transformers>=4.40.0
torch>=2.0.0
fastapi>=0.111.0
uvicorn>=0.30.0
pydantic>=2.0.0
# Tăng tốc inference CPU (tùy chọn)
optimum[onnxruntime]>=1.20.0
```

---

### `utils/normalizer.py`

**Mục đích:** Chuẩn hóa text trước khi đưa vào filter — tránh bị bypass bằng ký tự đặc biệt.

```python
import unicodedata
import re

class TextNormalizer:
    """
    Chuẩn hóa text tiếng Việt trước khi filter.
    Xử lý: unicode NFC, khoảng trắng thừa, ký tự lặp.
    """

    def normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)   # ă vs a + combining
        text = self._remove_excess_spaces(text)
        text = self._collapse_repeated_chars(text)  # "đụụụ" → "đụ"
        return text.strip()

    def _remove_excess_spaces(self, text: str) -> str:
        return re.sub(r"\s+", " ", text)

    def _collapse_repeated_chars(self, text: str) -> str:
        # Giữ tối đa 2 ký tự liên tiếp: "fuuuck" → "fuuck"
        return re.sub(r"(.)\1{2,}", r"\1\1", text)
```

---

### `utils/teencode.py`

**Mục đích:** Convert các dạng viết tắt/teencode phổ biến → dạng chuẩn để wordlist nhận ra.

```python
class TeencodeConverter:
    """
    Convert teencode tiếng Việt → dạng đầy đủ.
    Ví dụ: "đ.m" → "đụ mẹ", "cl" → "cặc lồn"
    Bổ sung thêm mapping tùy domain của bạn.
    """

    MAPPING = {
        r"\bđ\.m\b": "đụ mẹ",
        r"\bđmm\b": "đụ mẹ mày",
        r"\bvcl\b": "vãi cặc lồn",
        r"\bcl\b": "cặc lồn",
        r"\bvl\b": "vãi lồn",
        r"\bcc\b": "cặc",
        r"\bđcm\b": "đụ cái mẹ",
        r"\bkml\b": "không muốn lồn",
        r"\blon\b": "lồn",         # không dấu
        r"\bcac\b": "cặc",         # không dấu
        r"\bdit\b": "địt",         # không dấu
        r"\bdm\b": "đụ mẹ",
    }

    def convert(self, text: str) -> str:
        import re
        result = text.lower()
        for pattern, replacement in self.MAPPING.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result
```

> 💡 **Lưu ý:** File này quan trọng vì người Việt hay viết tắt để né filter.
> Bổ sung thêm mapping dựa vào data thực tế của bạn.

---

### `data/vi_wordlist.txt`

Wordlist cho Layer 1 — mỗi dòng 1 từ. Build file này theo domain thực tế.

```
# Cấu trúc: 1 từ/cụm từ per dòng, lowercase, có dấu
đụ
địt
cặc
lồn
đéo
vãi
chó má
súc vật
đồ ngu
thằng chó
con điếm
...
```

> 💡 Có thể lấy thêm từ dataset ViHSD (trích xuất các từ xuất hiện nhiều trong label HATE/OFFENSIVE).

---

### `layers/wordlist_layer.py`

**Pattern: Strategy** — có thể swap implementation khác mà không đổi interface.

```python
from abc import ABC, abstractmethod
from better_profanity import profanity as bp

class BaseFilterLayer(ABC):
    @abstractmethod
    def check(self, text: str) -> dict:
        """
        Returns:
            {
                "label": "CLEAN" | "OFFENSIVE" | "HATE",
                "score": float,
                "matched_words": list[str]  # từ bị phát hiện
            }
        """
        pass


class WordlistLayer(BaseFilterLayer):
    """
    Layer 1: Dùng better_profanity để bắt từ tục rõ ràng + leetspeak.
    Ưu điểm: ~1ms/câu, không cần GPU.
    Nhược điểm: không hiểu ngữ cảnh.
    """

    def __init__(self, wordlist_path: str):
        bp.load_censor_words_from_file(wordlist_path)
        self._profanity = bp

    def check(self, text: str) -> dict:
        is_profane = self._profanity.contains_profanity(text)
        censored = self._profanity.censor(text)

        # Tìm các từ bị censor để trả về
        matched = self._extract_matched_words(text, censored)

        return {
            "label": "OFFENSIVE" if is_profane else "CLEAN",
            "score": 1.0 if is_profane else 0.0,
            "matched_words": matched,
        }

    def _extract_matched_words(self, original: str, censored: str) -> list:
        """So sánh original vs censored để tìm từ bị filter."""
        orig_words = original.split()
        cens_words = censored.split()
        matched = []
        for o, c in zip(orig_words, cens_words):
            if o != c:
                matched.append(o)
        return matched
```

---

### `layers/ml_layer.py`

**Pattern: Adapter** — wrap HuggingFace pipeline vào interface chung `BaseFilterLayer`.

```python
from layers.wordlist_layer import BaseFilterLayer
from transformers import pipeline as hf_pipeline
import torch

LABEL_MAP = {
    "LABEL_0": "CLEAN",
    "LABEL_1": "OFFENSIVE",
    "LABEL_2": "HATE",
}

class MLLayer(BaseFilterLayer):
    """
    Layer 2: PhoBERT fine-tuned trên ViHSD — hiểu ngữ cảnh tiếng Việt.
    Model: visolex/phobert-v2-hsd
    Accuracy: 93.4%, Macro F1: 80.5%
    """

    def __init__(self, model_name: str = "visolex/phobert-v2-hsd"):
        device = 0 if torch.cuda.is_available() else -1
        self._classifier = hf_pipeline(
            "text-classification",
            model=model_name,
            device=device,
        )

    def check(self, text: str) -> dict:
        result = self._classifier(text, truncation=True, max_length=256)[0]
        label_raw = result["label"]

        # Map LABEL_0/1/2 → tên thật
        label = LABEL_MAP.get(label_raw, label_raw)

        return {
            "label": label,
            "score": round(result["score"], 3),
            "matched_words": [],   # ML không trả về từ cụ thể
        }
```

---

### `filter.py` — Class chính

**Pattern: Facade** — ẩn toàn bộ độ phức tạp, expose 1 interface đơn giản.

```python
from utils.normalizer import TextNormalizer
from utils.teencode import TeencodeConverter
from layers.wordlist_layer import WordlistLayer
from layers.ml_layer import MLLayer

class ViProfanityFilter:
    """
    Facade class — entry point duy nhất cho toàn bộ hệ thống.

    Pipeline:
        text → normalize → teencode convert
             → Layer 1 (wordlist, ~1ms)
             → nếu CLEAN → Layer 2 (ML, ~300ms)
             → kết quả

    Usage:
        filter = ViProfanityFilter()
        result = filter.check("mày là đồ chó")
        # {"label": "OFFENSIVE", "score": 0.88, "is_profane": True}
    """

    def __init__(
        self,
        wordlist_path: str = "data/vi_wordlist.txt",
        ml_model: str = "visolex/phobert-v2-hsd",
        skip_ml: bool = False,      # True nếu chỉ muốn dùng wordlist
        ml_threshold: float = 0.7,  # Score tối thiểu để tin ML
    ):
        self._normalizer = TextNormalizer()
        self._teencode = TeencodeConverter()
        self._wordlist_layer = WordlistLayer(wordlist_path)
        self._ml_layer = MLLayer(ml_model) if not skip_ml else None
        self._ml_threshold = ml_threshold

    def check(self, text: str) -> dict:
        """
        Kiểm tra text có chứa nội dung xấu không.

        Returns:
            {
                "label": "CLEAN" | "OFFENSIVE" | "HATE",
                "score": float,
                "is_profane": bool,
                "matched_words": list,   # từ bị bắt (nếu có)
                "layer_used": "wordlist" | "ml"
            }
        """
        # Tiền xử lý
        normalized = self._normalizer.normalize(text)
        converted = self._teencode.convert(normalized)

        # Layer 1: Wordlist (nhanh)
        wl_result = self._wordlist_layer.check(converted)
        if wl_result["label"] != "CLEAN":
            return {**wl_result, "is_profane": True, "layer_used": "wordlist"}

        # Layer 2: ML (chậm hơn nhưng hiểu ngữ cảnh)
        if self._ml_layer is None:
            return {**wl_result, "is_profane": False, "layer_used": "wordlist"}

        ml_result = self._ml_layer.check(normalized)  # dùng text normalized, không convert teencode
        is_profane = (
            ml_result["label"] != "CLEAN"
            and ml_result["score"] >= self._ml_threshold
        )
        return {**ml_result, "is_profane": is_profane, "layer_used": "ml"}

    def censor(self, text: str) -> str:
        """Trả về text đã censor thay vì dict."""
        result = self.check(text)
        if not result["is_profane"]:
            return text
        from better_profanity import profanity as bp
        return bp.censor(text)
```

---

### `api/server.py` — FastAPI self-host

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from filter import ViProfanityFilter

app = FastAPI(title="Vietnamese Profanity Filter API")

# Singleton — load model 1 lần khi start
_filter = ViProfanityFilter()


class CheckRequest(BaseModel):
    text: str

class CensorRequest(BaseModel):
    text: str


@app.post("/check")
def check(req: CheckRequest):
    """Kiểm tra text, trả về label + score."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text không được rỗng")
    return _filter.check(req.text)


@app.post("/censor")
def censor(req: CensorRequest):
    """Trả về text đã censor."""
    return {"censored_text": _filter.censor(req.text)}


@app.get("/health")
def health():
    return {"status": "ok"}
```

**Chạy server:**

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 1
```

> ⚠️ Dùng `--workers 1` vì ML model không thread-safe khi share giữa workers.
> Nếu cần scale, dùng nhiều process riêng biệt (Docker replicas).

---

## Thứ tự implement

```
Bước 1 — Setup
    pip install -r requirements.txt

Bước 2 — Build wordlist
    Tạo data/vi_wordlist.txt (bắt đầu từ ~50-100 từ cơ bản)
    Bổ sung dần từ data thực tế

Bước 3 — Utils
    Implement TextNormalizer (normalizer.py)
    Implement TeencodeConverter (teencode.py) — điều chỉnh MAPPING theo domain

Bước 4 — Layers
    Implement WordlistLayer (wordlist_layer.py)
    Test Layer 1 độc lập trước

    Implement MLLayer (ml_layer.py)
    Lần đầu chạy sẽ download model ~500MB về ~/.cache/huggingface

Bước 5 — Facade
    Implement ViProfanityFilter (filter.py)
    Test pipeline đầy đủ

Bước 6 — API
    Implement FastAPI server (api/server.py)
    Test bằng curl hoặc httpie
```

---

## Test cases cần cover

```python
# Cần pass tất cả các case này

# Case 1: Từ tục rõ ràng → Layer 1 bắt
assert filter.check("thằng địt mẹ")["label"] == "OFFENSIVE"

# Case 2: Từ bình thường dùng sai ngữ cảnh → Layer 2 bắt
assert filter.check("mày là con chó")["label"] == "OFFENSIVE"
assert filter.check("con chó nhà tôi rất ngoan")["label"] == "CLEAN"

# Case 3: Teencode → bị convert rồi bắt
assert filter.check("đcm mày")["label"] == "OFFENSIVE"

# Case 4: Leetspeak → better_profanity bắt
assert filter.check("c4c m4y")["label"] == "OFFENSIVE"

# Case 5: Text sạch
assert filter.check("hôm nay trời đẹp quá")["label"] == "CLEAN"

# Case 6: Text tiếng Anh섞 vào
assert filter.check("what the fuck mày")["label"] == "OFFENSIVE"
```

---

## Cân nhắc khi production

| Vấn đề | Giải pháp |
|---|---|
| Lần đầu start chậm (download model) | Pre-download model vào Docker image |
| RAM ~1.5GB khi load ML model | Dùng `skip_ml=True` nếu server yếu, chỉ dùng wordlist |
| Tốc độ Layer 2 ~300ms | Convert sang ONNX giảm còn ~80ms (xem thêm phần ONNX bên dưới) |
| False positive ("con chó" = CLEAN) | Điều chỉnh `ml_threshold` lên cao hơn (0.8+) |
| User né filter bằng ký tự lạ | Bổ sung thêm mapping trong `TeencodeConverter` |

### Tăng tốc với ONNX (tùy chọn)

```python
# Thêm vào ml_layer.py nếu muốn dùng ONNX thay PyTorch
from optimum.onnxruntime import ORTModelForSequenceClassification

class MLLayerONNX(BaseFilterLayer):
    def __init__(self, model_name: str = "visolex/phobert-v2-hsd"):
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = ORTModelForSequenceClassification.from_pretrained(
            model_name, export=True  # tự convert lần đầu
        )
    # ... giữ nguyên phần check()
```
