import time
import uuid
from typing import List, Dict, Any, Optional
from app.rag.chunking.base import ChunkingStrategy, Chunk, ChunkMetadata, ChunkingResult
from app.rag.chunking.metrics import MetricsCalculator
from app.rag.chunking.tokenizer import default_tokenizer


class RecursiveChunking(ChunkingStrategy):
    """
    Hierarchical recursive character text splitter.
    Attempts to split on paragraph breaks (\\n\\n), then line breaks (\\n),
    then sentence boundaries (. ), then spaces ( ), and finally individual characters.
    """

    @property
    def strategy_name(self) -> str:
        return "recursive"

    @property
    def display_name(self) -> str:
        return "Recursive Character"

    @property
    def description(self) -> str:
        return (
            "Recursively splits text using a hierarchy of delimiters (paragraphs, lines, "
            "sentences, words). Keeps related paragraphs and sentences intact as much as possible "
            "within the configured size budget."
        )

    @property
    def default_config(self) -> Dict[str, Any]:
        return {
            "chunk_size": 500,
            "chunk_overlap": 50,
            "separators": ["\n\n", "\n", ". ", " ", ""],
            "measurement_unit": "character",  # "character" or "token"
        }

    def validate_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        chunk_size = int(config_dict.get("chunk_size", 500))
        chunk_overlap = int(config_dict.get("chunk_overlap", 50))
        separators = config_dict.get("separators", ["\n\n", "\n", ". ", " ", ""])
        unit = str(config_dict.get("measurement_unit", "character")).lower()

        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )
        if not isinstance(separators, list) or not separators:
            separators = ["\n\n", "\n", ". ", " ", ""]

        return {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "separators": separators,
            "measurement_unit": unit,
        }

    def _get_length(self, text: str, unit: str) -> int:
        if unit == "token":
            return default_tokenizer.count_tokens(text)
        return len(text)

    def _split_text(self, text: str, separators: List[str], chunk_size: int, unit: str) -> List[str]:
        """Recursively decomposes text using the priority list of separators."""
        final_pieces: List[str] = []

        # Find appropriate separator
        chosen_separator = ""
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                chosen_separator = ""
                break
            if sep in text:
                chosen_separator = sep
                new_separators = separators[i + 1 :]
                break

        if chosen_separator:
            splits = text.split(chosen_separator)
        else:
            splits = list(text)  # split into characters as last resort

        good_splits = []
        for s in splits:
            if chosen_separator and chosen_separator != "":
                # Re-attach separator when meaningful (like '. ' or '\n')
                s_with_sep = s + chosen_separator if s != splits[-1] else s
            else:
                s_with_sep = s

            if self._get_length(s_with_sep, unit) < chunk_size:
                good_splits.append(s_with_sep)
            else:
                if new_separators:
                    sub_splits = self._split_text(s_with_sep, new_separators, chunk_size, unit)
                    good_splits.extend(sub_splits)
                else:
                    good_splits.append(s_with_sep)

        return good_splits

    def _merge_splits(
        self, pieces: List[str], chunk_size: int, chunk_overlap: int, unit: str
    ) -> List[str]:
        """Merges small pieces into chunks up to chunk_size with overlap."""
        chunks: List[str] = []
        current_doc: List[str] = []
        total = 0

        for piece in pieces:
            p_len = self._get_length(piece, unit)
            if total + p_len > chunk_size and current_doc:
                merged = "".join(current_doc).strip()
                if merged:
                    chunks.append(merged)

                # Keep pieces that fit in chunk_overlap
                while current_doc and total > chunk_overlap:
                    removed = current_doc.pop(0)
                    total -= self._get_length(removed, unit)

            current_doc.append(piece)
            total += p_len

        if current_doc:
            merged = "".join(current_doc).strip()
            if merged:
                chunks.append(merged)

        return chunks

    def split(
        self,
        text: str,
        document_id: uuid.UUID,
        config_dict: Dict[str, Any],
        document_context: Optional[Dict[str, Any]] = None,
    ) -> ChunkingResult:
        start_time = time.perf_counter()
        config = self.validate_config(config_dict)
        chunk_size = config["chunk_size"]
        chunk_overlap = config["chunk_overlap"]
        separators = config["separators"]
        unit = config["measurement_unit"]

        if not text or not text.strip():
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            metrics = MetricsCalculator.calculate(
                chunks=[],
                source_text="",
                nominal_overlap=chunk_overlap,
                target_size=chunk_size,
                duration_ms=duration_ms,
            )
            return ChunkingResult(
                strategy=self.strategy_name,
                chunks=[],
                metrics=metrics,
                config=config,
                processing_duration_ms=duration_ms,
            )

        # 1. Split hierarchically
        pieces = self._split_text(text, separators, chunk_size, unit)

        # 2. Merge respecting size and overlap
        merged_texts = self._merge_splits(pieces, chunk_size, chunk_overlap, unit)

        # 3. Create Chunk objects with source offset mapping
        chunks: List[Chunk] = []
        search_start = 0

        for idx, chunk_text in enumerate(merged_texts):
            # Locate position in original text
            found_idx = text.find(chunk_text[:min(40, len(chunk_text))], search_start)
            start_char = None
            end_char = None

            if found_idx != -1:
                start_char = found_idx
                end_char = found_idx + len(chunk_text)
                search_start = max(0, start_char + 1)

            chunk = Chunk(
                id=uuid.uuid4(),
                document_id=document_id,
                sequence_index=idx,
                content=chunk_text,
                char_count=len(chunk_text),
                token_count=default_tokenizer.count_tokens(chunk_text),
                metadata=ChunkMetadata(
                    start_char=start_char,
                    end_char=end_char,
                    content_type="text",
                ),
            )
            chunks.append(chunk)

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        metrics = MetricsCalculator.calculate(
            chunks=chunks,
            source_text=text,
            nominal_overlap=chunk_overlap,
            target_size=chunk_size,
            duration_ms=duration_ms,
        )

        return ChunkingResult(
            strategy=self.strategy_name,
            chunks=chunks,
            metrics=metrics,
            config=config,
            processing_duration_ms=duration_ms,
        )
