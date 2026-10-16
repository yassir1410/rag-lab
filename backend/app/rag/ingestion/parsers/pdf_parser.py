from typing import List
import fitz  # PyMuPDF
from app.rag.ingestion.base import DocumentParser, ParsedDocument, ParsedSection
from app.rag.ingestion.exceptions import CorruptedFileError, EncryptedFileError


class PdfParser(DocumentParser):
    """
    Extracts text page-by-page from PDF documents using PyMuPDF (fitz).
    Detects scanned/image-only pages and records OCR warnings without faking OCR.
    """

    @property
    def supported_mime_types(self) -> List[str]:
        return ["application/pdf"]

    @property
    def supported_extensions(self) -> List[str]:
        return [".pdf"]

    @property
    def extraction_method_name(self) -> str:
        return "pymupdf"

    def parse(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            raise CorruptedFileError(f"Failed to open PDF document '{filename}': {str(e)}")

        try:
            if doc.is_encrypted:
                raise EncryptedFileError(
                    f"PDF document '{filename}' is password-protected or encrypted"
                )

            page_count = len(doc)
            sections: List[ParsedSection] = []
            warnings: List[str] = []
            raw_text_parts: List[str] = []
            section_idx = 0
            scanned_pages_count = 0

            for page_index in range(page_count):
                page = doc[page_index]
                page_num = page_index + 1  # 1-indexed

                # Extract text preserving reading blocks
                page_text = page.get_text("text").strip()
                images = page.get_images()

                # Check for empty text or scanned page
                if not page_text:
                    if images:
                        scanned_pages_count += 1
                        warnings.append(
                            f"Page {page_num}: Scanned or image-only page detected ({len(images)} image(s)). "
                            "OCR is required to extract text (OCR not performed in Phase 02)."
                        )
                    else:
                        warnings.append(f"Page {page_num}: Blank page with no text or images.")
                    continue

                # Check for very low character count with images (likely a scanned page with accidental noise)
                if len(page_text) < 30 and images:
                    scanned_pages_count += 1
                    warnings.append(
                        f"Page {page_num}: Low text density ({len(page_text)} chars) with {len(images)} image(s). "
                        "Page appears largely scanned or graphic-based."
                    )

                raw_text_parts.append(page_text)

                # Segment page into paragraphs or blocks
                # PyMuPDF block extraction: (x0, y0, x1, y1, text, block_no, block_type)
                blocks = page.get_text("blocks")
                for block in blocks:
                    block_text = block[4].strip()
                    block_type = block[6]  # 0: text, 1: image
                    if block_type == 0 and block_text:
                        section = ParsedSection(
                            section_index=section_idx,
                            content=block_text,
                            page_number=page_num,
                            heading=None,
                            content_type="text",
                            metadata={
                                "bbox": [round(coord, 2) for coord in block[:4]],
                                "block_number": block[5],
                            },
                        )
                        sections.append(section)
                        section_idx += 1

            if scanned_pages_count > 0:
                warnings.insert(
                    0,
                    f"Document contains {scanned_pages_count} scanned/image-only page(s) out of {page_count}. "
                    "Text extraction is incomplete without an optical character recognition (OCR) engine.",
                )

            full_raw_text = "\n\n".join(raw_text_parts)

            return ParsedDocument(
                original_filename=filename,
                sections=sections,
                raw_text=full_raw_text,
                page_count=page_count,
                detected_language=None,
                extraction_method=self.extraction_method_name,
                warnings=warnings,
                metadata={
                    "format": "PDF",
                    "total_pages": page_count,
                    "scanned_pages_detected": scanned_pages_count,
                    "pdf_metadata": doc.metadata,
                },
            )
        finally:
            doc.close()
