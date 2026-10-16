import re
from typing import List, Dict, Any, Optional

# Fallback regex tokenizer for zero-dependency environments
DEFAULT_SUBWORD_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d|[a-zA-ZÀ-ÿ]+|[0-9]+|[^\s\w]+|\s+""",
    re.UNICODE,
)


class TokenizerInspector:
    """
    Inspects tokenization for embedding models, contrasting words, characters, and tokens.
    Exposes token IDs, truncation limits, and side-by-side model comparisons.
    """

    def __init__(self, model_id: str, max_seq_length: int = 256):
        self.model_id = model_id
        self.max_seq_length = max_seq_length
        self._tokenizer = None
        self._load_error = None
        self._is_loaded = False

    def _ensure_tokenizer(self):
        if self._is_loaded:
            return
        try:
            # 1. Check if model is already loaded in SentenceTransformer provider
            from app.rag.embeddings.registry import embedding_registry
            provider = embedding_registry.get_provider(self.model_id)
            if provider and hasattr(provider, "_model") and provider._model is not None:
                if hasattr(provider._model, "tokenizer") and provider._model.tokenizer is not None:
                    self._tokenizer = provider._model.tokenizer
                    self._is_loaded = True
                    return
        except Exception:
            pass

        try:
            from transformers import AutoTokenizer
            # 2. Fast local-only load to avoid network blocking
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, local_files_only=True
            )
        except Exception as e:
            self._load_error = str(e)
            self._tokenizer = None
        finally:
            self._is_loaded = True

    def inspect(self, text: str) -> Dict[str, Any]:
        """
        Inspects tokenization of given text.
        Returns tokens, token IDs, counts, warnings, and truncation status.
        """
        if not text:
            return {
                "model_id": self.model_id,
                "text": "",
                "char_count": 0,
                "word_count": 0,
                "token_count": 0,
                "max_seq_length": self.max_seq_length,
                "tokens": [],
                "token_ids": [],
                "is_truncated": False,
                "truncated_tokens_count": 0,
                "warnings": [],
            }

        char_count = len(text)
        word_count = len(text.split())
        self._ensure_tokenizer()

        warnings = []
        if self._tokenizer is not None:
            # Use real HuggingFace tokenizer
            encoded = self._tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False,
            )
            raw_token_ids = encoded["input_ids"]
            raw_tokens = self._tokenizer.convert_ids_to_tokens(raw_token_ids)
        else:
            # Fallback regex tokenization
            matches = list(DEFAULT_SUBWORD_PATTERN.finditer(text))
            raw_tokens = ["[CLS]"] + [m.group() for m in matches] + ["[SEP]"]
            raw_token_ids = [101] + [(hash(t) % 20000 + 1000) for t in raw_tokens[1:-1]] + [102]

        total_tokens = len(raw_tokens)
        is_truncated = total_tokens > self.max_seq_length
        truncated_count = max(0, total_tokens - self.max_seq_length)

        if is_truncated:
            warnings.append(
                f"Input length ({total_tokens} tokens) exceeds maximum sequence limit "
                f"({self.max_seq_length} tokens) for model '{self.model_id}'. "
                f"The final {truncated_count} tokens will be truncated during inference."
            )

        # Show first 150 tokens to avoid browser payload explosion
        display_tokens = raw_tokens[:min(total_tokens, 150)]
        display_ids = raw_token_ids[:min(total_tokens, 150)]

        return {
            "model_id": self.model_id,
            "text": text,
            "char_count": char_count,
            "word_count": word_count,
            "token_count": total_tokens,
            "max_seq_length": self.max_seq_length,
            "tokens": display_tokens,
            "token_ids": display_ids,
            "is_truncated": is_truncated,
            "truncated_tokens_count": truncated_count,
            "warnings": warnings,
            "ratio_char_to_token": round(char_count / max(total_tokens, 1), 2),
            "ratio_word_to_token": round(word_count / max(total_tokens, 1), 2),
        }
