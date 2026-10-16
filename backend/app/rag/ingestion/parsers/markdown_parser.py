import re
from typing import List, Optional
from app.rag.ingestion.base import DocumentParser, ParsedDocument, ParsedSection
from app.rag.ingestion.exceptions import CorruptedFileError


class MarkdownParser(DocumentParser):
    """
    Parses Markdown (.md) documents.
    Preserves heading hierarchies, code fences, lists, and tables without destroying syntax.
    Segments the document into logical sections grouped by headers or paragraphs.
    """

    @property
    def supported_mime_types(self) -> List[str]:
        return ["text/markdown", "text/x-markdown"]

    @property
    def supported_extensions(self) -> List[str]:
        return [".md", ".markdown"]

    @property
    def extraction_method_name(self) -> str:
        return "markdown_parser"

    def _decode_bytes(self, file_bytes: bytes, filename: str) -> str:
        try:
            return file_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                return file_bytes.decode("latin-1")
            except Exception as e:
                raise CorruptedFileError(
                    f"Failed to decode Markdown file '{filename}': {str(e)}"
                )

    def parse(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        text = self._decode_bytes(file_bytes, filename)
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        lines = text.split("\n")
        sections: List[ParsedSection] = []

        current_heading: Optional[str] = None
        current_heading_level: Optional[int] = None
        current_lines: List[str] = []
        section_idx = 0
        in_code_block = False

        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

        def flush_current_section():
            nonlocal section_idx, current_lines
            content = "\n".join(current_lines).strip()
            if content:
                # Determine primary content type
                c_type = "text"
                if content.startswith("```"):
                    c_type = "code"
                elif content.startswith("|") and "---" in content:
                    c_type = "table"

                sections.append(
                    ParsedSection(
                        section_index=section_idx,
                        content=content,
                        page_number=None,
                        heading=current_heading,
                        content_type=c_type,
                        metadata={
                            "heading_level": current_heading_level,
                            "has_code": "```" in content,
                        },
                    )
                )
                section_idx += 1
            current_lines = []

        for line in lines:
            stripped = line.strip()

            # Track fenced code block boundaries
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                current_lines.append(line)
                continue

            if in_code_block:
                current_lines.append(line)
                continue

            # Heading detected outside code block
            match = heading_pattern.match(line)
            if match:
                flush_current_section()
                level = len(match.group(1))
                h_text = match.group(2).strip()

                current_heading = h_text
                current_heading_level = level

                # Record the heading as its own section or begin new section
                sections.append(
                    ParsedSection(
                        section_index=section_idx,
                        content=line.strip(),
                        page_number=None,
                        heading=h_text,
                        content_type="heading",
                        metadata={"heading_level": level},
                    )
                )
                section_idx += 1
                continue

            # Paragraph break on empty line outside code block
            if not stripped and current_lines:
                flush_current_section()
                continue

            if stripped:
                current_lines.append(line)

        flush_current_section()

        return ParsedDocument(
            original_filename=filename,
            sections=sections,
            raw_text=text.strip(),
            page_count=None,
            detected_language=None,
            extraction_method=self.extraction_method_name,
            warnings=[],
            metadata={
                "format": "Markdown",
                "total_sections": len(sections),
            },
        )
