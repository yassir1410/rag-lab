import io
from typing import Dict, Any, List
import pymupdf


class ScannedPDFProcessor:
    """Detects scanned vs native PDF pages and extracts high-resolution page bitmaps."""

    SCANNED_CHAR_THRESHOLD = 40  # Under 40 characters on a page typically indicates a scan or graphic

    @classmethod
    def analyze_page(cls, page: pymupdf.Page, page_number: int, dpi: int = 150) -> Dict[str, Any]:
        """Inspect a single PDF page, determine text content, and render pixmap."""
        rect = page.rect
        width = int(rect.width)
        height = int(rect.height)

        text = page.get_text("text").strip()
        char_count = len(text)
        is_scanned = char_count < cls.SCANNED_CHAR_THRESHOLD

        # Render page to PNG pixmap
        # dpi 150 provides crisp OCR readability without excessive memory overhead
        pixmap = page.get_pixmap(dpi=dpi)
        image_bytes = pixmap.tobytes("png")

        return {
            "page_number": page_number,
            "width": pixmap.width,
            "height": pixmap.height,
            "original_width": width,
            "original_height": height,
            "has_native_text": char_count >= cls.SCANNED_CHAR_THRESHOLD,
            "native_text_char_count": char_count,
            "native_text": text,
            "is_scanned": is_scanned,
            "image_bytes": image_bytes,
        }

    @classmethod
    def process_pdf(cls, file_path: str, dpi: int = 150, max_pages: int = 50) -> List[Dict[str, Any]]:
        """Process multi-page PDF up to a safety threshold of max_pages."""
        doc = pymupdf.open(file_path)
        pages_data = []
        try:
            total_pages = min(len(doc), max_pages)
            for i in range(total_pages):
                page = doc[i]
                page_info = cls.analyze_page(page, page_number=i + 1, dpi=dpi)
                pages_data.append(page_info)
        finally:
            doc.close()
        return pages_data
