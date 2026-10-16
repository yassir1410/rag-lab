from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ParsedSection:
    """Represents an extracted logical section, paragraph, or table from a document."""
    section_index: int
    content: str
    page_number: Optional[int] = None
    heading: Optional[str] = None
    content_type: str = "text"  # text, table, code, heading
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """Represents the structured result of parsing an entire document."""
    original_filename: str
    sections: List[ParsedSection]
    raw_text: str
    page_count: Optional[int] = None
    detected_language: Optional[str] = None
    extraction_method: str = "unknown"
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentParser(ABC):
    """Abstract interface for all format-specific document parsers."""

    @property
    @abstractmethod
    def supported_mime_types(self) -> List[str]:
        """List of MIME types supported by this parser."""
        pass

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """List of lowercase file extensions (with leading dot) supported."""
        pass

    @property
    @abstractmethod
    def extraction_method_name(self) -> str:
        """Name of the extraction method/library (e.g. 'pymupdf', 'python-docx')."""
        pass

    def can_handle(self, mime_type: str, filename: str) -> bool:
        """Check whether this parser can handle the given file."""
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        return (mime_type.lower() in [m.lower() for m in self.supported_mime_types]) or (
            ext in [e.lower() for e in self.supported_extensions]
        )

    @abstractmethod
    def parse(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        """Parse raw file bytes into a structured ParsedDocument."""
        pass
