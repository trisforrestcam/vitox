"""Wordlist-based profanity filter layer sử dụng better_profanity."""

import unicodedata
from abc import ABC, abstractmethod
from typing import Any


class BaseFilterLayer(ABC):
    """Abstract base class cho tất cả các filter layers."""

    @abstractmethod
    def check(self, text: str) -> dict[str, Any]:
        """Phân tích text đã cho và trả về result dict.

        Args:
            text: Input text cần phân tích.

        Returns:
            Dictionary chứa ít nhất các keys "label", "score" và "matched_words".
        """
        ...


class WordlistLayer(BaseFilterLayer):
    """Filter layer kiểm tra text dựa trên wordlist local thông qua better_profanity.

    Attributes:
        profanity: Một instance ``better_profanity.Profanity`` được load với
            wordlist do người dùng cung cấp.
    """

    def __init__(self, wordlist_path: str) -> None:
        """Khởi tạo layer với file wordlist.

        Args:
            wordlist_path: Đường dẫn đến file plain-text, mỗi dòng chứa
                một từ cần censor.
        """
        from better_profanity import Profanity

        self._profanity = Profanity()
        self._profanity.load_censor_words_from_file(wordlist_path)

        # Load wordlist không dấu để match text viết không dấu
        self._ascii_profanity = Profanity()
        ascii_words = self._build_ascii_wordlist(wordlist_path)
        self._ascii_profanity.load_censor_words(ascii_words)

    @staticmethod
    def _build_ascii_wordlist(wordlist_path: str) -> list[str]:
        """Đọc wordlist và strip dấu tiếng Việt + normalize đ -> d."""
        words: list[str] = []
        with open(wordlist_path, encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if w:
                    ascii_w = "".join(
                        ch
                        for ch in unicodedata.normalize("NFD", w)
                        if unicodedata.category(ch) != "Mn"
                    )
                    # 'đ' là ký tự riêng trong Unicode, không phải d + dấu
                    ascii_w = ascii_w.replace("đ", "d").replace("Đ", "D")
                    if ascii_w:
                        words.append(ascii_w)
        return words

    @staticmethod
    def _strip_diacritics(text: str) -> str:
        """Bỏ dấu tiếng Việt khỏi text + normalize đ -> d."""
        text = "".join(
            ch
            for ch in unicodedata.normalize("NFD", text)
            if unicodedata.category(ch) != "Mn"
        )
        return text.replace("đ", "d").replace("Đ", "D")

    def check(self, text: str) -> dict[str, Any]:
        """Kiểm tra *text* với wordlist đã load.

        Method này so sánh text gốc với bản censored word-by-word
        để trích xuất các từ profanity tìm thấy. Nếu output censored
        chia tách từ khác nhau (hiếm gặp), layer sẽ fallback về
        simple profanity bool check.

        Ngoài ra còn check text không dấu (ASCII fallback) để bắt
        các trick viết không dấu.

        Args:
            text: Input text cần phân tích.

        Returns:
            Dictionary với các keys:
                - ``label``: "OFFENSIVE" hoặc "CLEAN"
                - ``score``: 1.0 khi offensive, 0.0 khi clean
                - ``matched_words``: List các từ profane tìm thấy trong *text*
        """
        if not text or not isinstance(text, str):
            return {"label": "CLEAN", "score": 0.0, "matched_words": []}

        matched_words: list[str] = []

        # 1. Check text gốc (có dấu)
        censored = self._profanity.censor(text)
        try:
            original_words = text.split()
            censored_words = censored.split()

            if len(original_words) == len(censored_words):
                for orig, cens in zip(original_words, censored_words):
                    if orig != cens and "*" in cens:
                        matched_words.append(orig)
            else:
                if self._profanity.contains_profanity(text):
                    matched_words.append("[unknown]")
        except Exception:
            if self._profanity.contains_profanity(text):
                matched_words.append("[unknown]")

        # 2. Check text không dấu (ASCII fallback)
        if not matched_words:
            ascii_text = self._strip_diacritics(text)
            if self._ascii_profanity.contains_profanity(ascii_text):
                ascii_censored = self._ascii_profanity.censor(ascii_text)
                try:
                    ascii_words = ascii_text.split()
                    ascii_censored_words = ascii_censored.split()
                    if len(ascii_words) == len(ascii_censored_words):
                        for orig, cens in zip(ascii_words, ascii_censored_words):
                            if orig != cens and "*" in cens:
                                matched_words.append(orig)
                    else:
                        matched_words.append("[unknown]")
                except Exception:
                    matched_words.append("[unknown]")

        label = "OFFENSIVE" if matched_words else "CLEAN"
        score = 1.0 if matched_words else 0.0

        return {
            "label": label,
            "score": score,
            "matched_words": matched_words,
        }

    def censor(self, text: str) -> str:
        """Trả về *text* với các từ profane được thay bằng dấu *.

        Args:
            text: Text gốc.

        Returns:
            Text đã được censor.
        """
        if not text or not isinstance(text, str):
            return text
        return self._profanity.censor(text)
