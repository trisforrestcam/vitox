"""Teencode (teen code) converter cho Vietnamese profanity filter."""

import re


class TeencodeConverter:
    """Convert Vietnamese teen-code / abbreviated profanity thành từ đầy đủ."""

    MAPPING = {
        r"\bđ\.m\b": "đụ mẹ",
        r"\bđmm\b": "đụ mẹ",
        r"\bdm\b": "đụ mẹ",
        r"\bvcl\b": "vãi cặc lồn",
        r"\bcl\b": "cặc lồn",
        r"\bvl\b": "vãi lồn",
        r"\bcc\b": "cặc",
        r"\bđcm\b": "đụ cái mẹ",
        r"\blon\b": "lồn",
        r"\bcac\b": "cặc",
        r"\bdit\b": "địt",
    }

    @classmethod
    def convert(cls, text: str) -> str:
        """Áp dụng tất cả các teencode substitutions lên *text* (case-insensitive).

        Trả về text với các teen-code abbreviations được thay bằng
        từ profanity tiếng Việt đầy đủ tương ứng.
        """
        for pattern, replacement in cls.MAPPING.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text
