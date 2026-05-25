# Model & Pipeline

Tài liệu này giải thích model đang dùng, pipeline xử lý của hệ thống, cách tính score/label, các tham số có thể điều chỉnh, và hướng dẫn train hoặc nâng cấp model.

---

## 1. Model đang sử dụng

### `visolex/phobert-v2-hsd`

- **Hugging Face**: https://huggingface.co/visolex/phobert-v2-hsd
- **Base model**: [`vinai/phobert-base-v2`](https://huggingface.co/vinai/phobert-base-v2) (135M parameters)
- **GitHub (PhoBERT gốc)**: https://github.com/VinAIResearch/PhoBERT
- **Task**: Text classification – phân loại Hate Speech tiếng Việt thành 3 nhãn gốc:
  - `CLEAN` (0) – nội dung bình thường
  - `OFFENSIVE` (1) – nội dung phản cảm, tục tĩu
  - `HATE` (2) – ngôn từ thù địch, cực đoan

> **Output của hệ thống**: Hệ thống chỉ trả về **2 nhãn** — `CLEAN` và `OFFENSIVE`. Class `HATE` được remap thành `OFFENSIVE` (xem phần 2.2).

### ⚠️ Lưu ý quan trọng về model này

Repo `visolex/phobert-v2-hsd` dùng **custom model class** (`PhoBERTV2Model` trong `models.py`), **không tương thích** với `pipeline("text-classification")` hay `AutoModelForSequenceClassification` chuẩn của Transformers.

Vì vậy `MLLayer` không dùng `pipeline` mà load thủ công:
- `AutoTokenizer` để encode text
- `AutoModel` làm encoder (backbone)
- `nn.Linear` làm classifier head
- Load `state_dict` từ `model.safetensors` (hoặc `pytorch_model.bin`)

### Thông tin training

| Tham số | Giá trị |
|---|---|
| Dataset | ViHSD (~10.000 comments từ social media) |
| Framework | HuggingFace Transformers + PyTorch |
| Optimizer | AdamW |
| Learning rate | `2e-5` |
| Batch size | `32` |
| Max sequence length | `256` tokens |
| Epochs | `100` (early stopping patience = 5) |
| Weight decay | `0.01` |
| Warmup steps | `500` |
| LR scheduler | Cosine with warmup |

### Kết quả evaluation

| Metric | Score |
|---|---|
| Accuracy | `0.9341` |
| Macro-F1 | `0.8048` |
| Weighted-F1 | `0.9326` |

---

## 2. Pipeline xử lý của hệ thống

### 2.1. Architecture

Hệ thống được thiết kế theo **2-layer architecture** mặc định: `WordlistLayer` chạy trước, nếu sạch thì mới chạy `MLLayer`.

> **🚧 Hiện tại**: `WordlistLayer` đang bị **tạm tắt** trong `filter.py` để test độ ổn định của `MLLayer`. Muốn bật lại thì uncomment block có ghi chú `[TẠM TẮT]`.

```
Raw Input Text
      │
      ▼
┌─────────────────────┐
│  1. TextNormalizer  │
│     - Unicode NFC   │
│     - Remove zero-width chars
│     - Collapse repeated chars (max 2)
│     - Collapse whitespace
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  2. TeencodeConverter
│     - "dm" → "đụ mẹ"
│     - "vcl" → "vãi cặc lồn"
│     - ... (regex mapping)
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  3. WordlistLayer   │  ← [TẠM TẮT] Đang bypass để test ML
│     (better_profanity)
│     - Match word-by-word
│     - Trả về: label, score, matched_words
└─────────────────────┘
      │
      ├─► Nếu phát hiện profanity → RETURN ngay (layer_used="wordlist")
      │
      └─► Nếu sạch (hoặc đang bypass)
            │
            ▼
    ┌─────────────────────┐
    │  4. MLLayer         │  ← Chạy trên NORMALIZED text (không qua teencode)
    │     (PhoBERT HSD)   │
    │     - Custom _PhoBERTClassifier
    │     - encoder (AutoModel) + dropout + linear classifier
    │     - Trả về: label, confidence score
    └─────────────────────┘
            │
            ▼
    So sánh score với ml_threshold (default 0.9)
            │
            ▼
         RETURN (layer_used="ml")
```

### 2.2. Cách tính label và score

Model gốc output **3 logits** tương ứng 3 class: `CLEAN`, `OFFENSIVE`, `HATE`.

Hệ thống remap về **2 nhãn** theo công thức:

```python
probs = softmax(logits)  # [prob_CLEAN, prob_OFFENSIVE, prob_HATE]

toxic_score = max(prob_OFFENSIVE, prob_HATE)

if toxic_score >= ml_threshold:
    label  = "OFFENSIVE"
    score  = toxic_score
    is_profane = True
else:
    label  = "CLEAN"
    score  = prob_CLEAN
    is_profane = False
```

| Nhãn gốc | Nhãn trả về | Score |
|---|---|---|
| `CLEAN` | `CLEAN` | `prob_CLEAN` |
| `OFFENSIVE` | `OFFENSIVE` | `prob_OFFENSIVE` |
| `HATE` | `OFFENSIVE` | `prob_HATE` |

**Vì sao dùng `max` thay vì tổng?**
- Tổng `prob_OFFENSIVE + prob_HATE` dễ khiến score luôn ~0.99 kể cả model chỉ tin chắc ở mức 0.8.
- `max` giữ nguyên confidence của class mạnh nhất, giúp threshold 0.9 filter được false positive mà vẫn giữ true positive.

### 2.3. Tại sao wordlist chạy trước? (khi bật lại)

- **Nhanh**: regex + string matching, không cần GPU.
- **Chính xác với từ ngữ explicit**: phát hiện đúng từ cụ thể trong `vi_wordlist.txt`.
- **Tiết kiệm compute**: chỉ gọi ML model khi wordlist không bắt được.

### 2.4. Tại sao ML chạy trên normalized text (không qua teencode)?

- Model PhoBERT được train trên text tự nhiên, không phải teencode.
- Teencode conversion có thể làm sai lệch semantic nếu convert quá aggressive.
- ML layer đóng vai trò **bắt lướt** những câu toxic nhưng không chứa từ trong wordlist.

---

## 3. Các tham số có thể điều chỉnh

Khi khởi tạo `ViProfanityFilter`, bạn có thể truyền các tham số sau:

```python
from vi_profanity_filter import ViProfanityFilter

filter = ViProfanityFilter(
    wordlist_path="data/vi_wordlist.txt",   # Đường dẫn wordlist
    ml_model="visolex/phobert-v2-hsd",      # Model ID hoặc local path
    skip_ml=False,                           # True = tắt ML, chỉ dùng wordlist
    ml_threshold=0.9,                        # Ngưỡng để coi là toxic
)
```

### `wordlist_path`

- **Default**: `"data/vi_wordlist.txt"`
- **Ý nghĩa**: Đường dẫn đến file wordlist dạng plain-text (mỗi dòng 1 từ).
- **Lưu ý**: Đường dẫn relative sẽ được resolve từ package root (`vi_profanity_filter/`).
- **Khi nào đổi**: Nếu bạn có wordlist custom hoặc muốn dùng wordlist ở vị trí khác.

### `ml_model`

- **Default**: `"visolex/phobert-v2-hsd"`
- **Ý nghĩa**: Hugging Face model identifier hoặc đường dẫn local đến checkpoint.
- **Khi nào đổi**: Khi bạn đã fine-tune model mới hoặc muốn thử model HSD khác.
- **Yêu cầu**: Model phải có architecture tương thích (`AutoModel` encoder + `nn.Linear` classifier head).

### `skip_ml`

- **Default**: `False`
- **Ý nghĩa**: Nếu `True`, ML layer sẽ không được load.
- **Khi nào đổi**: Khi cần chạy nhanh, không cần độ chính xác cao, hoặc server không đủ RAM/CPU để load model 135M params.

### `ml_threshold`

- **Default**: `0.9`
- **Ý nghĩa**: Ngưỡng `toxic_score` tối thiểu để text được đánh dấu là `OFFENSIVE` / `is_profane=True`.
- **Cao hơn** (vd `0.95`): ít false positive, nhưng có thể miss toxic nhẹ (vd: "vl thật" với score ~0.16 sẽ bị miss).
- **Thấp hơn** (vd `0.7`): bắt được nhiều hơn, nhưng dễ flag nhầm (vd: "tôi nhìn thấy con chó" với score ~0.8 sẽ bị bắt).
- **Cách tốt nhất**: Chạy trên validation set và chọn threshold tối ưu theo precision/recall trade-off.

---

## 4. Cách nâng cấp / train lại model

### 4.1. Thu thập thêm dữ liệu (quan trọng nhất)

Model hiện tại train trên ~10k samples. Để cải thiện:

- Thu thập comments từ Facebook, YouTube, TikTok, Shopee...
- Gán nhãn thủ công theo 3 class gốc: `CLEAN`, `OFFENSIVE`, `HATE`.
- Khuyến nghị: **tối thiểu 5.000–10.000 samples mới** để fine-tune có ý nghĩa.

> **Lưu ý**: Dù bạn gán nhãn theo 3 class, hệ thống vẫn sẽ remap `HATE` → `OFFENSIVE` khi inference. Nên nếu không cần phân biệt HATE/OFFENSIVE, bạn có thể gán nhãn chỉ 2 class (`CLEAN` / `OFFENSIVE`) và fine-tune model 2-class từ đầu.

Format dữ liệu (CSV hoặc JSON):

```json
[
  {"text": "Câu này rất tệ", "label": "OFFENSIVE"},
  {"text": "Hôm nay trởi đẹp", "label": "CLEAN"},
  {"text": "Giết chết thằng đó", "label": "HATE"}
]
```

### 4.2. Fine-tune lại với HuggingFace

Cài đặt dependencies:

```bash
pip install transformers datasets evaluate accelerate
```

Script fine-tune cơ bản:

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import evaluate
import numpy as np

# 1. Load base model
model_name = "vinai/phobert-base-v2"  # hoặc "visolex/phobert-v2-hsd"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=3, id2label={0:"CLEAN", 1:"OFFENSIVE", 2:"HATE"}
)

# 2. Load dataset
dataset = load_dataset("csv", data_files={
    "train": "train.csv",
    "validation": "val.csv",
    "test": "test.csv",
})

# 3. Tokenize
def preprocess(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=256)

tokenized = dataset.map(preprocess, batched=True)

# 4. Metrics
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return f1.compute(predictions=predictions, references=labels, average="macro")

# 5. Training arguments
training_args = TrainingArguments(
    output_dir="./phobert-hsd-v2",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,
    warmup_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    logging_dir="./logs",
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 7. Train & save
trainer.train()
trainer.save_model("./my-phobert-hsd-model")
```

### 4.3. Tích hợp model mới vào hệ thống

```python
from vi_profanity_filter import ViProfanityFilter

# Dùng model local
filter = ViProfanityFilter(
    ml_model="./my-phobert-hsd-model",
    ml_threshold=0.9,
)

# Hoặc upload lên HuggingFace rồi dùng model ID
filter = ViProfanityFilter(ml_model="your-username/your-model-name")
```

> **Lưu ý**: Nếu model mới dùng custom class giống `visolex/phobert-v2-hsd`, hãy đảm bảo `ml_layer.py` có thể load được state_dict. Hiện tại `_PhoBERTClassifier` hỗ trợ architecture: `AutoModel` encoder + `nn.Linear` classifier head.

---

## 5. Tóm tắt các điểm cần nhớ

| Vấn đề | Giải pháp |
|---|---|
| Model bắt miss teencode mới | Thêm regex vào `TeencodeConverter.MAPPING` |
| Model bắt miss từ mới | Thêm từ vào `data/vi_wordlist.txt` (nếu bật wordlist) |
| Cần độ chính xác cao hơn | Fine-tune lại trên dataset lớn hơn, đa dạng hơn |
| Cần chạy nhanh hơn | `skip_ml=True` hoặc quantize model (ONNX/INT8) |
| Muốn thêm class mới (vd: SPAM) | Sửa `num_labels=4`, thêm nhãn vào dataset, fine-tune lại từ đầu |
| Model load weights ngẫu nhiên | Kiểm tra `model.safetensors` / `pytorch_model.bin` có trong repo không; đảm bảo `ml_layer.py` load đúng architecture |
| False positive với từ nhạy cảm (vd: "chó") | Điều chỉnh `ml_threshold` hoặc thêm whitelist layer |

---

## 6. Tài nguyên tham khảo

- [PhoBERT GitHub](https://github.com/VinAIResearch/PhoBERT)
- [PhoBERT-base-v2 HuggingFace](https://huggingface.co/vinai/phobert-base-v2)
- [visolex/phobert-v2-hsd HuggingFace](https://huggingface.co/visolex/phobert-v2-hsd)
- [HuggingFace Training Docs](https://huggingface.co/docs/transformers/training)
- [ViHSD Dataset Paper](https://arxiv.org/abs/2105.00408)
