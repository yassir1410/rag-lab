import re
from typing import List
from app.rag.ingestion.base import DocumentParser, ParsedDocument, ParsedSection
from app.rag.ingestion.exceptions import CorruptedFileError


class TxtParser(DocumentParser):
    """
    Parses plain text documents.
    Supports UTF-8 and UTF-8 BOM, handles decoding fallbacks gracefully,
    and segments text into paragraph-level sections based on blank lines.
    """

    @property
    def supported_mime_types(self) -> List[str]:
        return ["text/plain"]

    @property
    def supported_extensions(self) -> List[str]:
        return [".txt"]

    @property
    def extraction_method_name(self) -> str:
        return "txt_parser"

    def _decode_bytes(self, file_bytes: bytes, filename: str) -> str:
        """Attempt decoding with UTF-8 BOM, UTF-8, and fallback with informative errors."""
        # 1. UTF-8 with BOM support
        try:
            return file_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            pass

        # 2. Latin-1 / ISO-8859-1 fallback
        try:
            return file_bytes.decode("latin-1")
        except UnicodeDecodeError:
            pass

        # 3. Windows-1252 fallback
        try:
            return file_bytes.decode("cp1252")
        except UnicodeDecodeError as e:
            raise CorruptedFileError(
                f"Failed to decode text file '{filename}': Unsupported encoding or corrupted bytes. {str(e)}"
            )

    def parse(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        text = self._decode_bytes(file_bytes, filename)

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Split into paragraphs by 1 or more blank lines
        paragraphs = re.split(r"\n\s*\n+", text)
        sections: List[ParsedSection] = []
        section_idx = 0

        for para in paragraphs:
            cleaned = para.strip()
            if cleaned:
                sections.append(
                    ParsedSection(
                        section_index=section_idx,
                        content=cleaned,
                        page_number=None,
                        heading=None,
                        content_type="text",
                        metadata={"paragraph_index": section_idx},
                    )
                )
                section_idx += 1

        return ParsedDocument(
            original_filename=filename,
            sections=sections,
            raw_text=text.strip(),
            page_count=None,
            detected_language=None,
            extraction_method=self.extraction_method_name,
            warnings=[],
            metadata={
                "format": "TXT",
                "total_paragraphs": len(sections),
            },
        )
