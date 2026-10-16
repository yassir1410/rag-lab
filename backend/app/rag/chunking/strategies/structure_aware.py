import time
import uuid
from typing import List, Dict, Any, Optional
from app.rag.chunking.base import ChunkingStrategy, Chunk, ChunkMetadata, ChunkingResult
from app.rag.chunking.metrics import MetricsCalculator
from app.rag.chunking.strategies.recursive import RecursiveChunking
from app.rag.chunking.tokenizer import default_tokenizer


class StructureAwareChunking(ChunkingStrategy):
    """
    Structure-aware chunking strategy leveraging document sections and layout elements from Phase 02.
    Preserves heading hierarchies, tables, and fenced code blocks as atomic semantic units.
    Falls back to recursive splitting when document section hierarchy is absent.
    """

    def __init__(self):
        self.fallback_splitter = RecursiveChunking()

    @property
    def strategy_name(self) -> str:
        return "structure_aware"

    @property
    def display_name(self) -> str:
        return "Structure-Aware"

    @property
    def description(self) -> str:
        return (
            "Uses parsed document sections, heading hierarchy, and block types from Phase 02. "
            "Guarantees that tables and code blocks remain intact, and enriches chunks with "
            "section titles and page numbers for pinpoint citations."
        )

    @property
    def default_config(self) -> Dict[str, Any]:
        return {
            "chunk_size": 600,
            "chunk_overlap": 50,
            "preserve_tables": True,
            "preserve_code_blocks": True,
            "measurement_unit": "character",
        }

    def validate_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        chunk_size = int(config_dict.get("chunk_size", 600))
        chunk_overlap = int(config_dict.get("chunk_overlap", 50))
        preserve_tables = bool(config_dict.get("preserve_tables", True))
        preserve_code = bool(config_dict.get("preserve_code_blocks", True))
        unit = str(config_dict.get("measurement_unit", "character")).lower()

        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap must satisfy 0 <= overlap < chunk_size, got {chunk_overlap}")

        return {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "preserve_tables": preserve_tables,
            "preserve_code_blocks": preserve_code,
            "measurement_unit": unit,
        }

    def _get_length(self, text: str, unit: str) -> int:
        if unit == "token":
            return default_tokenizer.count_tokens(text)
        return len(text)

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

        sections = (document_context or {}).get("sections", [])

        # Fallback to recursive chunker if no section metadata is provided
        if not sections:
            result = self.fallback_splitter.split(text, document_id, config_dict, document_context)
            result.strategy = self.strategy_name
            return result

        chunks: List[Chunk] = []
        seq_idx = 0
        current_buffer: List[Dict[str, Any]] = []
        buffer_len = 0
        current_headings: List[str] = []

        for sec in sections:
            sec_content = sec.get("content", "").strip()
            sec_type = sec.get("content_type", "text")
            heading = sec.get("heading")
            page_no = sec.get("page_number")
            sec_id = str(sec.get("id", ""))

            if heading and heading not in current_headings:
                current_headings.append(heading)

            sec_len = self._get_length(sec_content, unit)

            # Atomic elements (tables, code blocks, or single large sections)
            is_atomic = (
                (sec_type == "table" and config["preserve_tables"])
                or (sec_type == "code" and config["preserve_code_blocks"])
            )

            if is_atomic:
                # Flush previous buffer before emitting atomic chunk
                if current_buffer:
                    merged_content = "\n\n".join(s["content"] for s in current_buffer)
                    pages = list({s["page"] for s in current_buffer if s["page"] is not None})
                    h_list = list({s["heading"] for s in current_buffer if s["heading"]})
                    s_ids = [s["id"] for s in current_buffer if s["id"]]

                    chunks.append(
                        Chunk(
                            id=uuid.uuid4(),
                            document_id=document_id,
                            sequence_index=seq_idx,
                            content=merged_content,
                            char_count=len(merged_content),
                            token_count=default_tokenizer.count_tokens(merged_content),
                            metadata=ChunkMetadata(
                                page_numbers=pages,
                                headings=h_list,
                                source_section_ids=s_ids,
                                content_type="text",
                            ),
                        )
                    )
                    seq_idx += 1
                    current_buffer = []
                    buffer_len = 0

                # Emit atomic block directly as a chunk
                chunks.append(
                    Chunk(
                        id=uuid.uuid4(),
                        document_id=document_id,
                        sequence_index=seq_idx,
                        content=sec_content,
                        char_count=len(sec_content),
                        token_count=default_tokenizer.count_tokens(sec_content),
                        metadata=ChunkMetadata(
                            page_numbers=[page_no] if page_no else [],
                            headings=[heading] if heading else [],
                            source_section_ids=[sec_id] if sec_id else [],
                            content_type=sec_type,
                            custom={"atomic_block": True},
                        ),
                    )
                )
                seq_idx += 1
                continue

            # Standard prose sections: group up to chunk_size
            if buffer_len + sec_len > chunk_size and current_buffer:
                merged_content = "\n\n".join(s["content"] for s in current_buffer)
                pages = list({s["page"] for s in current_buffer if s["page"] is not None})
                h_list = list({s["heading"] for s in current_buffer if s["heading"]})
                s_ids = [s["id"] for s in current_buffer if s["id"]]

                chunks.append(
                    Chunk(
                        id=uuid.uuid4(),
                        document_id=document_id,
                        sequence_index=seq_idx,
                        content=merged_content,
                        char_count=len(merged_content),
                        token_count=default_tokenizer.count_tokens(merged_content),
                        metadata=ChunkMetadata(
                            page_numbers=pages,
                            headings=h_list,
                            source_section_ids=s_ids,
                            content_type="text",
                        ),
                    )
                )
                seq_idx += 1
                current_buffer = []
                buffer_len = 0

            current_buffer.append({
                "content": sec_content,
                "page": page_no,
                "heading": heading,
                "id": sec_id,
                "type": sec_type,
            })
            buffer_len += sec_len

        # Flush remaining buffer
        if current_buffer:
            merged_content = "\n\n".join(s["content"] for s in current_buffer)
            pages = list({s["page"] for s in current_buffer if s["page"] is not None})
            h_list = list({s["heading"] for s in current_buffer if s["heading"]})
            s_ids = [s["id"] for s in current_buffer if s["id"]]

            chunks.append(
                Chunk(
                    id=uuid.uuid4(),
                    document_id=document_id,
                    sequence_index=seq_idx,
                    content=merged_content,
                    char_count=len(merged_content),
                    token_count=default_tokenizer.count_tokens(merged_content),
                    metadata=ChunkMetadata(
                        page_numbers=pages,
                        headings=h_list,
                        source_section_ids=s_ids,
                        content_type="text",
                    ),
                )
            )

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
