import time
import uuid
from typing import List, Dict, Any, Optional
from app.rag.chunking.base import ChunkingStrategy, Chunk, ChunkMetadata, ChunkingResult
from app.rag.chunking.metrics import MetricsCalculator
from app.rag.chunking.strategies.recursive import RecursiveChunking
from app.rag.chunking.tokenizer import default_tokenizer


class ParentChildChunking(ChunkingStrategy):
    """
    Two-tier hierarchical Parent-Child chunking strategy.
    Divides text into large Parent passages (rich context for generation)
    and sub-segments each parent into smaller Child chunks (focused units for vector retrieval).
    """

    def __init__(self):
        self.recursive_splitter = RecursiveChunking()

    @property
    def strategy_name(self) -> str:
        return "parent_child"

    @property
    def display_name(self) -> str:
        return "Parent-Child (Hierarchical)"

    @property
    def description(self) -> str:
        return (
            "Creates a dual hierarchy: large Parent chunks provide complete surrounding context "
            "for the LLM, while smaller Child chunks serve as focused, highly specific retrieval units. "
            "Enables small-to-large context expansion."
        )

    @property
    def default_config(self) -> Dict[str, Any]:
        return {
            "parent_chunk_size": 1200,
            "parent_overlap": 100,
            "child_chunk_size": 300,
            "child_overlap": 30,
            "measurement_unit": "character",
        }

    def validate_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        parent_size = int(config_dict.get("parent_chunk_size", 1200))
        parent_overlap = int(config_dict.get("parent_overlap", 100))
        child_size = int(config_dict.get("child_chunk_size", 300))
        child_overlap = int(config_dict.get("child_overlap", 30))
        unit = str(config_dict.get("measurement_unit", "character")).lower()

        if parent_size <= 0:
            raise ValueError(f"parent_chunk_size must be positive, got {parent_size}")
        if child_size <= 0:
            raise ValueError(f"child_chunk_size must be positive, got {child_size}")
        if child_size >= parent_size:
            raise ValueError(
                f"child_chunk_size ({child_size}) must be strictly smaller than parent_chunk_size ({parent_size})"
            )
        if parent_overlap < 0 or parent_overlap >= parent_size:
            raise ValueError(f"Invalid parent_overlap: {parent_overlap}")
        if child_overlap < 0 or child_overlap >= child_size:
            raise ValueError(f"Invalid child_overlap: {child_overlap}")

        return {
            "parent_chunk_size": parent_size,
            "parent_overlap": parent_overlap,
            "child_chunk_size": child_size,
            "child_overlap": child_overlap,
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
        parent_size = config["parent_chunk_size"]
        parent_overlap = config["parent_overlap"]
        child_size = config["child_chunk_size"]
        child_overlap = config["child_overlap"]
        unit = config["measurement_unit"]

        if not text or not text.strip():
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            metrics = MetricsCalculator.calculate(
                chunks=[], source_text="", nominal_overlap=child_overlap, target_size=child_size, duration_ms=duration_ms
            )
            return ChunkingResult(
                strategy=self.strategy_name,
                chunks=[],
                metrics=metrics,
                config=config,
                processing_duration_ms=duration_ms,
            )

        # 1. Generate Parent chunks using recursive splitter
        parent_result = self.recursive_splitter.split(
            text=text,
            document_id=document_id,
            config_dict={
                "chunk_size": parent_size,
                "chunk_overlap": parent_overlap,
                "measurement_unit": unit,
            },
        )

        all_chunks: List[Chunk] = []
        seq_idx = 0

        for parent_idx, parent_chunk in enumerate(parent_result.chunks):
            parent_id = uuid.uuid4()
            parent_text = parent_chunk.content

            # 2. Split this parent into children
            child_result = self.recursive_splitter.split(
                text=parent_text,
                document_id=document_id,
                config_dict={
                    "chunk_size": child_size,
                    "chunk_overlap": child_overlap,
                    "measurement_unit": unit,
                },
            )

            child_ids: List[str] = []
            child_chunks_for_parent: List[Chunk] = []

            for c_idx, raw_child in enumerate(child_result.chunks):
                c_id = uuid.uuid4()
                child_ids.append(str(c_id))

                # Compute character offset in original document text
                child_start = None
                child_end = None
                if parent_chunk.metadata.start_char is not None and raw_child.metadata.start_char is not None:
                    child_start = parent_chunk.metadata.start_char + raw_child.metadata.start_char
                    child_end = child_start + len(raw_child.content)

                child_chunk = Chunk(
                    id=c_id,
                    document_id=document_id,
                    sequence_index=seq_idx,
                    content=raw_child.content,
                    char_count=len(raw_child.content),
                    token_count=default_tokenizer.count_tokens(raw_child.content),
                    metadata=ChunkMetadata(
                        start_char=child_start,
                        end_char=child_end,
                        content_type="text",
                        parent_chunk_id=str(parent_id),
                        custom={
                            "tier": "child",
                            "parent_index": parent_idx,
                            "child_index": c_idx,
                        },
                    ),
                )
                child_chunks_for_parent.append(child_chunk)
                seq_idx += 1

            # Build Parent chunk representation
            parent_entity = Chunk(
                id=parent_id,
                document_id=document_id,
                sequence_index=seq_idx,
                content=parent_text,
                char_count=len(parent_text),
                token_count=default_tokenizer.count_tokens(parent_text),
                metadata=ChunkMetadata(
                    start_char=parent_chunk.metadata.start_char,
                    end_char=parent_chunk.metadata.end_char,
                    content_type="text",
                    child_chunk_ids=child_ids,
                    custom={"tier": "parent", "parent_index": parent_idx, "children_count": len(child_ids)},
                ),
            )
            seq_idx += 1

            # Append parent followed by its children (or children followed by parent)
            all_chunks.append(parent_entity)
            all_chunks.extend(child_chunks_for_parent)

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        metrics = MetricsCalculator.calculate(
            chunks=all_chunks,
            source_text=text,
            nominal_overlap=child_overlap,
            target_size=child_size,
            duration_ms=duration_ms,
        )
        metrics["parent_chunks_count"] = len(parent_result.chunks)
        metrics["child_chunks_count"] = len(all_chunks) - len(parent_result.chunks)

        return ChunkingResult(
            strategy=self.strategy_name,
            chunks=all_chunks,
            metrics=metrics,
            config=config,
            processing_duration_ms=duration_ms,
        )
