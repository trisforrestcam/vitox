"""Accent restoration for Vietnamese text using seq2seq model."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class AccentRestorer:
    """Restore Vietnamese diacritics (accents) for unmarked text.

    Uses ``nrl-ai/vn-diacritic-small`` (BARTpho-syllable, 115 M) to convert
    text without diacritics into fully marked Vietnamese.
    """

    DEFAULT_MODEL = "nrl-ai/vn-diacritic-small"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModelForSeq2SeqLM | None = None

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _lazy_load(self) -> None:
        """Load tokenizer + model on first use to avoid slow import."""
        if self._model is not None:
            return

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name)
        model = cast("torch.nn.Module", self._model)
        model.eval()

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def restore(self, text: str) -> str:
        """Restore diacritics for *text*.

        Args:
            text: Input text (preferably already normalized).

        Returns:
            Text with Vietnamese diacritics restored.
            If text is empty or loading fails, returns original text.
        """
        if not text or not isinstance(text, str):
            return text

        # Fast-path: nếu text đã có dấu tiếng Việt rõ ràng thì skip
        if self._has_vietnamese_diacritics(text):
            return text

        try:
            self._lazy_load()
        except Exception:
            # Graceful fallback nếu model chưa download hoặc lỗi
            return text

        import torch

        inputs = self._tokenizer( #type: ignore[reportCallIssue]
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_length=256)  # type: ignore[union-attr]

        restored: str = self._tokenizer.decode(outputs[0], skip_special_tokens=True)  # type: ignore[union-attr]
        return restored

    @staticmethod
    def _has_vietnamese_diacritics(text: str) -> bool:
        """Check if text already contains Vietnamese-specific diacritics.

        This avoids unnecessary inference on already-marked text.
        """
        # Các ký tự có dấu tiếng Việt đặc trưng: àáảãạ, èéẻẽẹ, ìíỉĩị, ...
        vietnamese_diacritics = set(
            "àáảãạèéẻẽẹìíỉĩịòóỏõọùúủũụ"
            "ầấẩẫậềếểễệồốổỗộừứửữự"
            "ăắằẳẵặđĩũơớờởỡợưứừửữự"
            "ÀÁẢÃẠÈÉẺẼẸÌÍỈĨỊÒÓỎÕỌÙÚỦŨỤ"
            "ẦẤẨẪẬỀẾỂỄỆỒỐỔỖỘỪỨỬỮỰ"
            "ĂẮẰẲẴẶĐĨŨƠỚỜỞỠỢƯỨỪỬỮỰ"
        )
        return any(ch in vietnamese_diacritics for ch in text)
