import re
from typing import List, Dict, Any
import pymupdf


from app.rag.multimodal.ingestion.table_extractor import TableExtractor


class DocumentLayoutAnalyzer:
    """Analyzes document page geometry to classify structural elements and reconstruct natural reading order."""

    CAPTION_REGEX = re.compile(r"^(figure|fig\.|table|tableau|graphique|sch[eé]ma)\s*\d*[:.-]?", re.IGNORECASE)

    @classmethod
    def analyze_page_layout(
        cls,
        page: pymupdf.Page,
        page_number: int,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """Extract text blocks, images, and tables with classification and reading order."""
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height

        # 1. Extract raw blocks from PyMuPDF
        # block tuple: (x0, y0, x1, y1, text_or_image, block_no, block_type)
        # block_type == 0: text, block_type == 1: image
        raw_blocks = page.get_text("blocks")
        elements: List[Dict[str, Any]] = []

        # Find tables first to prevent table text from being split into isolated paragraphs
        table_bboxes = []
        extracted_tables = []
        try:
            extracted_tables = TableExtractor.extract_tables_from_page(page, page_number)
            for t in extracted_tables:
                b = t.get("bounding_box", {})
                tx0, ty0, tx1, ty1 = b.get("x0", 0), b.get("y0", 0), b.get("x1", 0), b.get("y1", 0)
                table_bboxes.append((tx0, ty0, tx1, ty1))
        except Exception:
            pass

        for b in raw_blocks:
            x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
            block_type = b[6] if len(b) > 6 else 0
            raw_content = b[4] if len(b) > 4 else ""

            # Check if block falls inside a table
            in_table = any(
                tx0 <= x0 and ty0 <= y0 and x1 <= tx1 and y1 <= ty1
                for tx0, ty0, tx1, ty1 in table_bboxes
            )
            if in_table:
                continue

            text_content = raw_content.strip() if isinstance(raw_content, str) else ""

            # Classify element type
            elem_type = "paragraph"
            if block_type == 1:
                elem_type = "figure"
            elif y0 < page_height * 0.07:
                elem_type = "header"
            elif y1 > page_height * 0.93:
                elem_type = "footer"
            elif cls.CAPTION_REGEX.match(text_content):
                elem_type = "caption"
            elif len(text_content) < 120 and (y0 < page_height * 0.25 or text_content.isupper()):
                # Likely section title or header
                if len(text_content.splitlines()) <= 2:
                    elem_type = "title"

            # Rescale bbox to pixel coordinates of rendered image
            bbox = {
                "x0": round(x0 * scale_x, 1),
                "y0": round(y0 * scale_y, 1),
                "x1": round(x1 * scale_x, 1),
                "y1": round(y1 * scale_y, 1),
                "width": round((x1 - x0) * scale_x, 1),
                "height": round((y1 - y0) * scale_y, 1),
            }

            elements.append({
                "page_number": page_number,
                "element_type": elem_type,
                "content_text": text_content,
                "bounding_box": bbox,
                "confidence": 0.95 if elem_type in ("title", "paragraph") else 0.85,
                "structured_data": {},
                # Store unscaled coords for reading order computation
                "_geom": (x0, y0, x1, y1),
            })

        # 2. Add table elements
        try:
            for t_idx, t in enumerate(extracted_tables):
                b = t.get("bounding_box", {})
                tx0 = b.get("x0", 0.0)
                ty0 = b.get("y0", 0.0)
                tx1 = b.get("x1", 0.0)
                ty1 = b.get("y1", 0.0)
                md_table = t.get("markdown_table", "")

                t_bbox = {
                    "x0": round(tx0 * scale_x, 1),
                    "y0": round(ty0 * scale_y, 1),
                    "x1": round(tx1 * scale_x, 1),
                    "y1": round(ty1 * scale_y, 1),
                    "width": round((tx1 - tx0) * scale_x, 1),
                    "height": round((ty1 - ty0) * scale_y, 1),
                }

                elements.append({
                    "page_number": page_number,
                    "element_type": "table",
                    "content_text": md_table,
                    "bounding_box": t_bbox,
                    "confidence": 0.95,
                    "structured_data": {
                        "table_id": t.get("table_id", f"tbl_{page_number}_{t_idx + 1}"),
                        "caption": t.get("caption", f"Table {t_idx + 1} (Page {page_number})"),
                        "num_rows": t.get("num_rows", 0),
                        "num_cols": t.get("num_cols", 0),
                        "headers": t.get("headers", []),
                        "rows": t.get("rows", []),
                        "markdown": md_table,
                        "raw_flattened_text": t.get("raw_flattened_text", ""),
                    },
                    "_geom": (tx0, ty0, tx1, ty1),
                })
        except Exception:
            pass

        # 3. Two-Column Layout Detection & Reading Order Sorting
        # Check if page exhibits multi-column distribution
        mid_x = page_width * 0.5
        left_blocks = [e for e in elements if e["_geom"][2] <= mid_x * 1.1]
        right_blocks = [e for e in elements if e["_geom"][0] >= mid_x * 0.9]
        is_two_column = len(left_blocks) >= 2 and len(right_blocks) >= 2

        if is_two_column:
            # Sort left column top-to-bottom, then right column top-to-bottom
            # Headers stay on top, footers stay at bottom
            headers = [e for e in elements if e["element_type"] == "header"]
            footers = [e for e in elements if e["element_type"] == "footer"]
            body = [e for e in elements if e["element_type"] not in ("header", "footer")]
            
            body_left = sorted([e for e in body if e["_geom"][0] < mid_x], key=lambda e: e["_geom"][1])
            body_right = sorted([e for e in body if e["_geom"][0] >= mid_x], key=lambda e: e["_geom"][1])
            sorted_elements = headers + body_left + body_right + footers
        else:
            # Standard top-to-bottom sorting
            sorted_elements = sorted(elements, key=lambda e: (e["_geom"][1], e["_geom"][0]))

        # Assign monotonic reading_order and remove temporary _geom
        for idx, elem in enumerate(sorted_elements):
            elem["reading_order"] = idx + 1
            elem.pop("_geom", None)

        return sorted_elements
