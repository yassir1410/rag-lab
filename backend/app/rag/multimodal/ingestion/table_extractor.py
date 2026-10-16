from typing import List, Dict, Any, Optional
import pymupdf


class TableExtractor:
    """Extracts tables with structural integrity, column preservation, and Markdown rendering."""

    @classmethod
    def extract_tables_from_page(
        cls, page: pymupdf.Page, page_number: int
    ) -> List[Dict[str, Any]]:
        """Extract all tables on a page into structured JSON and Markdown."""
        tables_data = []
        try:
            tabs = page.find_tables()
            for idx, tab in enumerate(tabs):
                raw_extracted = tab.extract()
                if not raw_extracted or len(raw_extracted) < 1:
                    continue

                headers = [str(c or "").strip() for c in raw_extracted[0]]
                raw_rows = raw_extracted[1:] if len(raw_extracted) > 1 else []
                rows = [[str(c or "").strip() for c in r] for r in raw_rows]

                # Format Markdown table
                md_lines = []
                if headers:
                    md_lines.append("| " + " | ".join(headers) + " |")
                    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for r in rows:
                    md_lines.append("| " + " | ".join(r) + " |")
                markdown_table = "\n".join(md_lines)

                # Raw flattened text (simulating naive OCR/parser collapse)
                flat_tokens = []
                for row_cells in raw_extracted:
                    flat_tokens.extend([str(c or "").strip() for c in row_cells if c])
                raw_flattened_text = " ".join(flat_tokens)

                tx0, ty0, tx1, ty1 = tab.bbox
                tables_data.append({
                    "table_id": f"tbl_p{page_number}_{idx + 1}",
                    "page_number": page_number,
                    "caption": f"Table {idx + 1} (Page {page_number})",
                    "num_rows": len(rows),
                    "num_cols": len(headers),
                    "headers": headers,
                    "rows": rows,
                    "markdown_table": markdown_table,
                    "raw_flattened_text": raw_flattened_text,
                    "bounding_box": {
                        "x0": round(tx0, 1),
                        "y0": round(ty0, 1),
                        "x1": round(tx1, 1),
                        "y1": round(ty1, 1),
                        "width": round(tx1 - tx0, 1),
                        "height": round(ty1 - ty0, 1),
                    },
                })
        except Exception:
            pass
        return tables_data

    @classmethod
    def extract_tables_from_doc(
        cls, file_path: str, target_page: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Extract tables from a whole document or a specific page."""
        doc = pymupdf.open(file_path)
        all_tables = []
        try:
            pages_to_check = (
                [target_page - 1]
                if target_page and 1 <= target_page <= len(doc)
                else range(len(doc))
            )
            for p_idx in pages_to_check:
                page = doc[p_idx]
                page_tables = cls.extract_tables_from_page(page, page_number=p_idx + 1)
                all_tables.extend(page_tables)
        finally:
            doc.close()
        return all_tables
