"""Machine-learning profanity filter layer sử dụng Transformers model."""

import os
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .wordlist_layer import BaseFilterLayer


class _PhoBERTClassifier(nn.Module):
    """Wrapper model: PhoBERT encoder + dropout + linear classifier head."""

    def __init__(self, encoder_name: str = "vinai/phobert-base-v2", num_labels: int = 3) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, **kwargs):  # noqa: ARG002
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


class MLLayer(BaseFilterLayer):
    """Filter layer dùng transformer model (PhoBERT) để phát hiện profanity.

    Layer này load tokenizer và weights từ checkpoint HuggingFace / local path,
    sau đó chạy forward pass qua encoder + classifier head để lấy logits.
    """

    def __init__(self, model_name: str, threshold: float = 0.7) -> None:
        """Khởi tạo ML layer.

        Args:
            model_name: Hugging Face model identifier hoặc local path.
            threshold: Ngưỡng confidence tối thiểu để coi prediction là
                profane (được caller sử dụng).
        """
        self._threshold = threshold
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        # PhoBERT-v2 HSD dựa trên vinai/phobert-base-v2
        self._model = _PhoBERTClassifier(encoder_name="vinai/phobert-base-v2", num_labels=3)
        self._load_weights(model_name)
        self._model.eval()

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _resolve_weight_path(self, model_name: str) -> str | None:
        """Tìm file weights (``model.safetensors`` hoặc ``pytorch_model.bin``)."""
        # Local path
        if os.path.isdir(model_name):
            for fname in ("model.safetensors", "pytorch_model.bin"):
                p = os.path.join(model_name, fname)
                if os.path.isfile(p):
                    return p
            return None

        # HuggingFace Hub
        try:
            from huggingface_hub import hf_hub_download

            try:
                return hf_hub_download(repo_id=model_name, filename="model.safetensors")
            except Exception:
                return hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
        except Exception:
            return None

    def _load_weights(self, model_name: str) -> None:
        weight_path = self._resolve_weight_path(model_name)
        if weight_path is None:
            raise RuntimeError(f"Không tìm thấy weights cho model '{model_name}'")

        if weight_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(weight_path)
        else:
            state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)

        self._model.load_state_dict(state_dict, strict=False)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def check(self, text: str) -> dict[str, Any]:
        """Classify *text* bằng transformer model.

        Args:
            text: Input text cần phân tích.

        Returns:
            Dictionary với các keys:
                - ``label``: Chỉ có 2 giá trị: ``"CLEAN"`` hoặc ``"OFFENSIVE"``.
                    Class ``HATE`` từ model gốc được remap thành ``"OFFENSIVE"``.
                - ``score``: Nếu ``label="OFFENSIVE"`` thì score là
                    ``max(prob_OFFENSIVE, prob_HATE)``. Nếu ``label="CLEAN"`` thì
                    score là ``prob_CLEAN``.
                - ``matched_words``: List rỗng (ML layer không trả về
                  explicit word matches).
        """
        if not text or not isinstance(text, str):
            return {"label": "CLEAN", "score": 0.0, "matched_words": []}

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        with torch.no_grad():
            logits = self._model(**inputs)
            probs = torch.softmax(logits, dim=-1)[0]

        # Chỉ giữ 2 nhãn: CLEAN và OFFENSIVE.
        # HATE (class 2) được remap thành OFFENSIVE.
        # Score = max(prob_OFFENSIVE, prob_HATE) — lấy xác suất cao nhất
        # trong 2 class toxic để quyết định.
        prob_clean = float(probs[0].item())
        prob_offensive = float(probs[1].item())
        prob_hate = float(probs[2].item())

        toxic_score = max(prob_offensive, prob_hate)

        if toxic_score >= self._threshold:
            label = "OFFENSIVE"
            score = toxic_score
        else:
            label = "CLEAN"
            score = prob_clean

        return {
            "label": label,
            "score": score,
            "matched_words": [],
        }
