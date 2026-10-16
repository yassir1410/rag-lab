import re
from typing import List


class TokenizerService:
    """
    Tokenizer service for token counting and text tokenization.
    Tries tiktoken (cl100k_base) if installed, otherwise uses a deterministic
    regular-expression subword/word tokenizer modeled after GPT tokenization.
    """

    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding_name = encoding_name
        self._tiktoken_encoder = None

        try:
            import tiktoken
            self._tiktoken_encoder = tiktoken.get_encoding(encoding_name)
        except (ImportError, Exception):
            self._tiktoken_encoder = None

        # Standard fallback token pattern (words, contractions, punctuation, whitespace)
        self._regex_pattern = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d|[a-zA-ZÀ-ÿ]+|[0-9]+|[^\s\w]+|\s+""",
            re.UNICODE,
        )

    @property
    def is_tiktoken_available(self) -> bool:
        return self._tiktoken_encoder is not None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        if self._tiktoken_encoder:
            return len(self._tiktoken_encoder.encode(text, disallowed_special=()))
        return len(self._regex_pattern.findall(text))

    def tokenize_spans(self, text: str) -> List[tuple[int, int, str]]:
        """
        Tokenizes text returning a list of (start_char, end_char, token_text).
        Provides exact character offset mapping for token-based slicing.
        """
        if not text:
            return []

        spans = []
        for match in self._regex_pattern.finditer(text):
            spans.append((match.start(), match.end(), match.group()))
        return spans


# Default singleton instance
default_tokenizer = TokenizerService()
