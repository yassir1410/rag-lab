import time
from dataclasses import dataclass
from typing import Dict, Any, Optional
from app.rag.ingestion.base import ParsedDocument
from app.rag.ingestion.cleaning.normalizer import TextNormalizer
from app.rag.ingestion.registry import ParserRegistry, default_parser_registry
from app.rag.ingestion.validation.file_validator import FileValidator


@dataclass
class IngestionResult:
    """Consolidated result produced by the IngestionPipeline."""
    original_filename: str
    sanitized_filename: str
    storage_filename: str
    sha256_hash: str
    mime_type: str
    file_size: int
    parsed_document: ParsedDocument
    normalized_full_text: str
    comparison_stats: Dict[str, Any]
    processing_duration_ms: float


class IngestionPipeline:
    """
    Coordinates document ingestion:
    1. Validation & sanitization (magic bytes, size, safe paths)
    2. Hashing (SHA-256 for duplicate detection)
    3. Parser resolution & execution (PDF, DOCX, TXT, Markdown)
    4. Text normalization & educational comparative metrics
    """

    def __init__(
        self,
        validator: Optional[FileValidator] = None,
        registry: Optional[ParserRegistry] = None,
        normalizer: Optional[TextNormalizer] = None,
    ):
        self.validator = validator or FileValidator()
        self.registry = registry or default_parser_registry
        self.normalizer = normalizer or TextNormalizer()

    def process(
        self, file_bytes: bytes, filename: str, reported_mime: str = ""
    ) -> IngestionResult:
        start_time = time.perf_counter()

        # 1. Validation and sanitization
        detected_mime, sanitized_filename = self.validator.validate(
            file_bytes=file_bytes, filename=filename, reported_mime=reported_mime
        )

        # 2. Cryptographic hashing & storage name generation
        sha256_hash = self.validator.compute_sha256(file_bytes)
        storage_filename = self.validator.generate_storage_filename(sanitized_filename)

        # 3. Locate parser and parse content
        parser = self.registry.get_parser(detected_mime, sanitized_filename)
        parsed_doc = parser.parse(file_bytes, sanitized_filename)

        # 4. Text normalization (educational comparison)
        normalized_full, comparison_stats = self.normalizer.compare(parsed_doc.raw_text)

        # 5. Normalize section contents individually
        for section in parsed_doc.sections:
            norm_sec, sec_stats = self.normalizer.compare(section.content)
            section.metadata["raw_char_count"] = sec_stats["raw_character_count"]
            section.metadata["raw_word_count"] = sec_stats["raw_word_count"]
            section.metadata["normalized_char_count"] = sec_stats["normalized_character_count"]
            section.metadata["normalized_word_count"] = sec_stats["normalized_word_count"]
            section.metadata["normalized_content"] = norm_sec

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return IngestionResult(
            original_filename=filename,
            sanitized_filename=sanitized_filename,
            storage_filename=storage_filename,
            sha256_hash=sha256_hash,
            mime_type=detected_mime,
            file_size=len(file_bytes),
            parsed_document=parsed_doc,
            normalized_full_text=normalized_full,
            comparison_stats=comparison_stats,
            processing_duration_ms=duration_ms,
        )
