import io
from typing import List
import docx
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.rag.ingestion.base import DocumentParser, ParsedDocument, ParsedSection
from app.rag.ingestion.exceptions import CorruptedFileError


class DocxParser(DocumentParser):
    """
    Parses DOCX documents using python-docx.
    Traverses the document body in sequential order, preserving heading hierarchies
    and formatting tables into structured Markdown table representations.
    """

    @property
    def supported_mime_types(self) -> List[str]:
        return [
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/docx",
        ]

    @property
    def supported_extensions(self) -> List[str]:
        return [".docx"]

    @property
    def extraction_method_name(self) -> str:
        return "python-docx"

    def _format_table_as_markdown(self, table: Table) -> str:
        """Convert a docx Table into a clean Markdown table representation."""
        if not table.rows:
            return ""

        rows_data = []
        for row in table.rows:
            # Strip cell text and replace internal newlines with space to preserve table row integrity
            row_cells = [cell.text.replace("\n", " ").strip() for cell in row.cells]
            rows_data.append(row_cells)

        if not rows_data:
            return ""

        num_cols = max(len(r) for r in rows_data)
        # Pad shorter rows
        for r in rows_data:
            while len(r) < num_cols:
                r.append("")

        md_lines = []
        # Header row
        header = "| " + " | ".join(rows_data[0]) + " |"
        separator = "| " + " | ".join(["---"] * num_cols) + " |"
        md_lines.append(header)
        md_lines.append(separator)

        # Body rows
        for row in rows_data[1:]:
            md_lines.append("| " + " | ".join(row) + " |")

        return "\n".join(md_lines)

    def parse(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        try:
            doc = docx.Document(io.BytesIO(file_bytes))
        except Exception as e:
            raise CorruptedFileError(f"Failed to parse DOCX document '{filename}': {str(e)}")

        sections: List[ParsedSection] = []
        raw_text_parts: List[str] = []
        current_heading = None
        section_idx = 0

        # Traverse document body in true sequential element order
        for child in doc.element.body:
            if isinstance(child, CT_P):
                p = Paragraph(child, doc)
                text = p.text.strip()
                if not text:
                    continue

                style_name = p.style.name if p.style else ""
                is_heading = "Heading" in style_name or "Title" in style_name

                if is_heading:
                    current_heading = text

                section = ParsedSection(
                    section_index=section_idx,
                    content=text,
                    page_number=None,  # DOCX is reflowable, page numbers not fixed without word engine
                    heading=current_heading,
                    content_type="heading" if is_heading else "text",
                    metadata={
                        "style": style_name,
                        "is_heading": is_heading,
                    },
                )
                sections.append(section)
                raw_text_parts.append(text)
                section_idx += 1

            elif isinstance(child, CT_Tbl):
                tbl = Table(child, doc)
                table_md = self._format_table_as_markdown(tbl)
                if not table_md:
                    continue

                section = ParsedSection(
                    section_index=section_idx,
                    content=table_md,
                    page_number=None,
                    heading=current_heading,
                    content_type="table",
                    metadata={
                        "rows": len(tbl.rows),
                        "columns": len(tbl.columns) if tbl.rows else 0,
                    },
                )
                sections.append(section)
                raw_text_parts.append(table_md)
                section_idx += 1

        full_raw_text = "\n\n".join(raw_text_parts)

        return ParsedDocument(
            original_filename=filename,
            sections=sections,
            raw_text=full_raw_text,
            page_count=None,
            detected_language=None,
            extraction_method=self.extraction_method_name,
            warnings=[],
            metadata={
                "format": "DOCX",
                "total_sections": len(sections),
                "has_tables": any(s.content_type == "table" for s in sections),
            },
        )
