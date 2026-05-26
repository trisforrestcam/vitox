"""Facade module cho Vietnamese profanity filter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .layers.ml_layer import MLLayer
from .layers.wordlist_layer import BaseFilterLayer, WordlistLayer
from .utils.accent_restorer import AccentRestorer
from .utils.normalizer import TextNormalizer
from .utils.teencode import TeencodeConverter


class ViProfanityFilter:
    """Facade cấp cao, điều phối các bước: normalization, teencode conversion,
    wordlist matching và (tùy chọn) ML classification.
    """

    def __init__(
        self,
        wordlist_path: str = "data/vi_wordlist.txt",
        ml_model: str = "visolex/phobert-v2-hsd",
        skip_ml: bool = False,
        ml_threshold: float = 0.9,
        enable_accent_restore: bool = True,
    ) -> None:
        """Khởi tạo filter.

        Args:
            wordlist_path: Đường dẫn đến file wordlist dạng plain-text.
                Đường dẫn relative sẽ được resolve từ package root.
            ml_model: Hugging Face model identifier cho ML layer.
            skip_ml: Nếu ``True``, ML layer sẽ không được load,
                chỉ dùng wordlist layer.
            ml_threshold: Ngưỡng confidence score tối thiểu để ML layer
                đánh dấu text là profane.
            enable_accent_restore: Nếu ``True``, bật accent restoration
                trước khi đưa vào ML layer (giúp handle text không dấu).
        """
        self._normalizer = TextNormalizer()
        self._teencode = TeencodeConverter()
        self._accent_restorer = AccentRestorer() if enable_accent_restore else None
        self._ml_threshold = ml_threshold

        # Resolve wordlist path từ package root
        pkg_root = Path(__file__).resolve().parent
        resolved_wordlist = pkg_root / wordlist_path
        self._wordlist: WordlistLayer = WordlistLayer(str(resolved_wordlist))

        self._ml: BaseFilterLayer | None = None
        if not skip_ml:
            self._ml = MLLayer(ml_model, threshold=ml_threshold)

    def check(self, text: str) -> dict[str, Any]:
        """Phân tích *text* để phát hiện profanity.

        Pipeline:
        1. Normalize raw text.
        2. [TẠM TẮT] Wordlist layer – đang bypass để test ML layer.
        3. Nếu ML layer được bật, chạy nó trên text **đã normalize**
           và trả về kết quả.

        Args:
            text: Raw input text.

        Returns:
            Dictionary với các keys:
                - ``label``: Nhãn dự đoán. Chỉ có 2 giá trị: ``"CLEAN"`` hoặc ``"OFFENSIVE"``.
                    Class ``HATE`` từ model gốc được remap thành ``"OFFENSIVE"``.
                - ``score``: Confidence score. Nếu ``label="OFFENSIVE"`` thì score là
                    ``max(prob_OFFENSIVE, prob_HATE)``. Nếu ``label="CLEAN"`` thì score là
                    ``prob_CLEAN``.
                - ``is_profane``: ``True`` nếu text được đánh giá là profane.
                - ``matched_words``: List các từ profane tìm thấy.
                - ``layer_used``: ``"wordlist"`` hoặc ``"ml"``.
        """
        if not text or not isinstance(text, str):
            return {
                "label": "CLEAN",
                "score": 0.0,
                "is_profane": False,
                "matched_words": [],
                "layer_used": "none",
            }

        normalised = self._normalizer.normalize(text)
        # teencode_converted = self._teencode.convert(normalised)

        # Layer 1 – Wordlist (check cả có dấu và không dấu)
        wordlist_result = self._wordlist.check(normalised)
        if wordlist_result.get("label") != "CLEAN" or wordlist_result.get("matched_words"):
            return {
                "label": wordlist_result["label"],
                "score": wordlist_result["score"],
                "is_profane": True,
                "matched_words": wordlist_result.get("matched_words", []),
                "layer_used": "wordlist",
            }

        # Accent restoration: bổ sung dấu tiếng Việt cho text không dấu
        # trước khi đưa vào ML layer
        if self._accent_restorer is not None:
            restored = self._accent_restorer.restore(normalised)
            normalised = self._normalizer.strip_attached_punctuation(restored)
            print(f"[DEBUG] After accent restore: {normalised!r}")

        # Layer 2 – ML (chạy trên normalized + accent-restored text)
        if self._ml is not None:
            ml_result = self._ml.check(normalised)
            label = ml_result.get("label", "CLEAN")
            score = float(ml_result.get("score", 0.0))
            is_profane = label != "CLEAN" and score >= self._ml_threshold
            return {
                "label": label,
                "score": score,
                "is_profane": is_profane,
                "matched_words": ml_result.get("matched_words", []),
                "layer_used": "ml",
            }

        # Không có ML và wordlist sạch
        return {
            "label": "CLEAN",
            "score": 0.0,
            "is_profane": False,
            "matched_words": [],
            "layer_used": "wordlist",
        }

    def censor(self, text: str) -> str:
        """Trả về *text* với các từ profane được thay bằng dấu *.

        Hàm này delegate cho wordlist layer thông qua instance
        ``better_profanity`` bên trong.

        Args:
            text: Text gốc.

        Returns:
            Text đã được censor.
        """
        if not text or not isinstance(text, str):
            return text
        return self._wordlist.censor(text)
