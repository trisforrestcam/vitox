"""Text normalization utilities cho Vietnamese profanity filter."""

import re
import unicodedata


class TextNormalizer:
    """Normalize raw text về dạng clean, nhất quán."""

    # Các ký tự Unicode zero-width và invisible phổ biến
    ZERO_WIDTH_CHARS = {
        "\u200B",  # Zero Width Space
        "\u200C",  # Zero Width Non-Joiner
        "\u200D",  # Zero Width Joiner
        "\u2060",  # Word Joiner
        "\uFEFF",  # Zero Width No-Break Space (BOM)
        "\u00AD",  # Soft Hyphen
        "\u034F",  # Combining Grapheme Joiner
        "\u180E",  # Mongolian Vowel Separator
    }

    @classmethod
    def normalize(cls, text: str) -> str:
        """Áp dụng toàn bộ normalization pipeline.

        Các bước:
        1. Unicode NFC normalization.
        2. Loại bỏ zero-width / invisible characters.
        3. Strip leading / trailing whitespace.
        4. Collapse repeated characters (giữ tối đa 2 ký tự liên tiếp).
        5. Loại bỏ whitespace thừa.
        """
        text = cls.nfc(text)
        text = cls.remove_zero_width(text)
        text = cls.strip(text)
        text = cls.collapse_repeated_chars(text)
        text = cls.collapse_whitespace(text)
        return text

    @classmethod
    def nfc(cls, text: str) -> str:
        """Áp dụng Unicode NFC normalization."""
        return unicodedata.normalize("NFC", text)

    @classmethod
    def strip(cls, text: str) -> str:
        """Strip leading và trailing whitespace."""
        return text.strip()

    @classmethod
    def collapse_whitespace(cls, text: str) -> str:
        """Thay thế bất kỳ chuỗi whitespace nào bằng một dấu cách đơn."""
        return re.sub(r"\s+", " ", text)

    @classmethod
    def collapse_repeated_chars(cls, text: str, max_repeat: int = 2) -> str:
        """Thu gọn các chuỗi ký tự lặp lại về tối đa *max_repeat*."""
        # Dùng backreference để thu gọn bất kỳ ký tự nào lặp quá max_repeat lần
        return re.sub(r"(.)\1{%d,}" % max_repeat, r"\1" * max_repeat, text)

    @classmethod
    def remove_zero_width(cls, text: str) -> str:
        """Loại bỏ các ký tự Unicode zero-width và invisible phổ biến."""
        for ch in cls.ZERO_WIDTH_CHARS:
            text = text.replace(ch, "")
        return text

    @classmethod
    def strip_attached_punctuation(cls, text: str) -> str:
        """Loại bỏ dấu câu dính vào đầu / cuối mỗi từ.

        Giữ lại dấu câu đứng riêng hoặc ở giữa từ.
        """
        # Dấu câu phổ biến cần strip khỏi đầu/cuối word
        punct = r"\"'\`\*\.\,\;\:\!\?\(\)\[\]\{\}\<\>"
        return " ".join(
            re.sub(f"^[{punct}]+|[{punct}]+$", "", word) for word in text.split(" ")
        )
