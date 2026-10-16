import time
import uuid
from typing import List, Dict, Any, Optional
from app.rag.chunking.base import ChunkingStrategy, Chunk, ChunkMetadata, ChunkingResult
from app.rag.chunking.metrics import MetricsCalculator
from app.rag.chunking.tokenizer import default_tokenizer


class FixedSizeChunking(ChunkingStrategy):
    """
    Fixed-size sliding window chunking algorithm.
    Splits text strictly based on character or token lengths with configurable overlap.
    """

    @property
    def strategy_name(self) -> str:
        return "fixed_size"

    @property
    def display_name(self) -> str:
        return "Fixed-Size Window"

    @property
    def description(self) -> str:
        return (
            "Slices text into uniform chunks based on fixed character or token counts "
            "with a sliding window overlap. Fast and deterministic, but unaware of sentence "
            "or semantic boundaries."
        )

    @property
    def default_config(self) -> Dict[str, Any]:
        return {
            "chunk_size": 400,
            "chunk_overlap": 50,
            "measurement_unit": "character",  # "character" or "token"
        }

    def validate_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        chunk_size = int(config_dict.get("chunk_size", 400))
        chunk_overlap = int(config_dict.get("chunk_overlap", 50))
        unit = str(config_dict.get("measurement_unit", "character")).lower()

        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )
        if unit not in ("character", "token"):
            raise ValueError(f"measurement_unit must be 'character' or 'token', got '{unit}'")

        return {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "measurement_unit": unit,
        }

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
        unit = config["measurement_unit"]

        chunks: List[Chunk] = []

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

        step = chunk_size - chunk_overlap

        if unit == "character":
            start_idx = 0
            text_len = len(text)
            seq_idx = 0

            while start_idx < text_len:
                end_idx = min(start_idx + chunk_size, text_len)
                chunk_text = text[start_idx:end_idx]

                chunk = Chunk(
                    id=uuid.uuid4(),
                    document_id=document_id,
                    sequence_index=seq_idx,
                    content=chunk_text,
                    char_count=len(chunk_text),
                    token_count=default_tokenizer.count_tokens(chunk_text),
                    metadata=ChunkMetadata(
                        start_char=start_idx,
                        end_char=end_idx,
                        content_type="text",
                    ),
                )
                chunks.append(chunk)
                seq_idx += 1

                if end_idx >= text_len:
                    break
                start_idx += step

        else:  # token mode
            token_spans = default_tokenizer.tokenize_spans(text)
            total_tokens = len(token_spans)
            seq_idx = 0
            start_token_idx = 0

            while start_token_idx < total_tokens:
                end_token_idx = min(start_token_idx + chunk_size, total_tokens)
                start_char = token_spans[start_token_idx][0]
                end_char = token_spans[end_token_idx - 1][1]
                chunk_text = text[start_char:end_char]

                chunk = Chunk(
                    id=uuid.uuid4(),
                    document_id=document_id,
                    sequence_index=seq_idx,
                    content=chunk_text,
                    char_count=len(chunk_text),
                    token_count=end_token_idx - start_token_idx,
                    metadata=ChunkMetadata(
                        start_char=start_char,
                        end_char=end_char,
                        content_type="text",
                    ),
                )
                chunks.append(chunk)
                seq_idx += 1

                if end_token_idx >= total_tokens:
                    break
                start_token_idx += step

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
