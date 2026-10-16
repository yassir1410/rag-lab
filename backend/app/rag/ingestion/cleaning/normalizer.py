import re
import unicodedata
from typing import Dict, Any, Tuple


class TextNormalizer:
    """
    Non-destructive text normalizer designed for educational inspection.
    Transforms raw extracted text into clean, consistent text suitable for downstream
    tokenization and chunking, while preserving code fences and tables.
    """

    def normalize(self, text: str) -> str:
        """
        Normalize text:
        1. Unicode normalization (NFKC)
        2. Normalize line endings (\r\n -> \n)
        3. Remove non-printable control characters (except \n, \t)
        4. Normalize spaces and excessive blank lines while respecting code blocks
        """
        if not text:
            return ""

        # 1. Unicode normalization (combines characters, converts compatibility characters)
        normalized = unicodedata.normalize("NFKC", text)

        # 2. Line break normalization
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

        # 3. Remove non-printable control characters except \n and \t
        normalized = "".join(
            ch for ch in normalized if ch in ("\n", "\t") or unicodedata.category(ch)[0] != "C"
        )

        # 4. Handle code blocks: don't collapse leading spaces inside markdown ``` code blocks
        lines = normalized.split("\n")
        processed_lines = []
        in_code_block = False

        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith("```"):
                in_code_block = not in_code_block
                processed_lines.append(line.rstrip())
                continue

            if in_code_block:
                # Inside code block: preserve leading indentation, strip trailing whitespace
                processed_lines.append(line.rstrip())
            else:
                # Outside code block: collapse multiple horizontal spaces/tabs
                cleaned = re.sub(r"[ \t]+", " ", line).strip()
                processed_lines.append(cleaned)

        # Recombine lines
        result = "\n".join(processed_lines)

        # 5. Collapse 3+ consecutive newlines to 2 newlines (paragraph boundary)
        result = re.sub(r"\n{3,}", "\n\n", result)

        return result.strip()

    def compare(self, raw_text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Returns normalized text and educational comparison statistics.
        """
        normalized = self.normalize(raw_text)

        raw_chars = len(raw_text)
        raw_words = len(raw_text.split()) if raw_text else 0
        norm_chars = len(normalized)
        norm_words = len(normalized.split()) if normalized else 0

        stats = {
            "raw_character_count": raw_chars,
            "raw_word_count": raw_words,
            "normalized_character_count": norm_chars,
            "normalized_word_count": norm_words,
            "character_reduction_pct": (
                round((raw_chars - norm_chars) / raw_chars * 100, 2)
                if raw_chars > 0
                else 0.0
            ),
        }

        return normalized, stats
