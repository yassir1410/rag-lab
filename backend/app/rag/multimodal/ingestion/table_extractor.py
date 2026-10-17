import logging
from typing import List, Dict, Any, Optional
import pymupdf

logger = logging.getLogger(__name__)


class TableExtractor:
    """Extracts tables with structural integrity, column preservation, and Markdown rendering.
    
    Supports:
    1. Fully bordered tables via native PyMuPDF line detection.
    2. Rule-divided / semi-bordered tables (e.g. LaTeX booktabs, NASA planetary compendiums).
    3. Borderless tables via whitespace and text-alignment column clustering.
    """

    @classmethod
    def _extract_tables_standard(
        cls, page: pymupdf.Page, page_number: int
    ) -> List[Dict[str, Any]]:
        """Strategy A: Standard PyMuPDF table finder for bordered tables."""
        tables_data = []
        tabs = page.find_tables()
        for idx, tab in enumerate(tabs):
            raw_extracted = tab.extract()
            if not raw_extracted or len(raw_extracted) < 1:
                continue

            headers = [str(c or "").strip() for c in raw_extracted[0]]
            raw_rows = raw_extracted[1:] if len(raw_extracted) > 1 else []
            rows = [[str(c or "").strip() for c in r] for r in raw_rows]

            if len(headers) < 2 or len(rows) < 1:
                continue

            md_lines = []
            if headers:
                md_lines.append("| " + " | ".join(headers) + " |")
                md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for r in rows:
                md_lines.append("| " + " | ".join(r) + " |")
            markdown_table = "\n".join(md_lines)

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
        return tables_data

    @classmethod
    def _extract_tables_rule_based(
        cls, page: pymupdf.Page, page_number: int
    ) -> List[Dict[str, Any]]:
        """Strategy B: Rule-divided tables with horizontal line dividers (e.g. under header & rows)."""
        drawings = page.get_drawings()
        h_lines = []
        for d in drawings:
            r = d.get("rect")
            if r.width >= 35.0 and r.height <= 3.0:
                h_lines.append(r)

        if not h_lines:
            return []

        # Group horizontal line segments sharing roughly the same Y coordinate
        y_lines: Dict[float, List[Any]] = {}
        for r in h_lines:
            matched_y = None
            for y in y_lines:
                if abs(r.y0 - y) < 2.0:
                    matched_y = y
                    break
            if matched_y is None:
                matched_y = round(r.y0, 1)
                y_lines[matched_y] = []
            y_lines[matched_y].append(r)

        sorted_ys = sorted(y_lines.keys())
        divider_ys = []
        for y in sorted_ys:
            rects = y_lines[y]
            span = max(r.x1 for r in rects) - min(r.x0 for r in rects)
            tot_w = sum(r.width for r in rects)
            # Table divider lines either split into column segments or span wide
            if len(rects) >= 2 and span >= 250.0:
                divider_ys.append(y)
            elif span >= 400.0 and tot_w >= 400.0:
                divider_ys.append(y)

        clusters: List[List[float]] = []
        curr: List[float] = []
        for y in divider_ys:
            if not curr:
                curr.append(y)
            else:
                dy = y - curr[-1]
                if 15.0 <= dy <= 85.0:
                    curr.append(y)
                else:
                    if len(curr) >= 2:
                        clusters.append(curr)
                    curr = [y]
        if len(curr) >= 2:
            clusters.append(curr)

        tables_data = []
        for c_idx, cl in enumerate(clusters):
            cluster_rects = [r for y in cl for r in y_lines[y]]
            x0 = min(r.x0 for r in cluster_rects)
            x1 = max(r.x1 for r in cluster_rects)

            y_top = max(0.0, cl[0] - 45.0)
            y_bottom = cl[-1]

            row_bounds = [(y_top, cl[0])] + [(cl[i], cl[i + 1]) for i in range(len(cl) - 1)]

            table_clip = pymupdf.Rect(x0 - 5.0, y_top, x1 + 5.0, y_bottom + 5.0)
            table_words = page.get_text("words", clip=table_clip)

            rows_words = [[] for _ in row_bounds]
            for w in table_words:
                wy_mid = (w[1] + w[3]) / 2.0
                for idx, (ry0, ry1) in enumerate(row_bounds):
                    if ry0 <= wy_mid <= ry1:
                        rows_words[idx].append(w)
                        break

            col_xs = sorted(set(round(r.x0, 1) for r in cluster_rects))
            col_xs.append(max(round(r.x1, 1) for r in cluster_rects))
            clean_col_xs = []
            for x in col_xs:
                if not clean_col_xs or (x - clean_col_xs[-1]) >= 15.0:
                    clean_col_xs.append(x)

            extracted_matrix = []
            for r_idx, rw in enumerate(rows_words):
                if not rw:
                    continue
                cells = [[] for _ in range(len(clean_col_xs) - 1)]
                for w in rw:
                    wx_mid = (w[0] + w[2]) / 2.0
                    for c_idx_col, (cx0, cx1) in enumerate(zip(clean_col_xs[:-1], clean_col_xs[1:])):
                        if cx0 <= wx_mid <= cx1 + 1.0:
                            cells[c_idx_col].append(w)
                            break
                cell_texts = [
                    " ".join(w[4] for w in sorted(c, key=lambda x: (x[1], x[0]))).strip()
                    for c in cells
                ]
                extracted_matrix.append(cell_texts)

            if not extracted_matrix or len(extracted_matrix) < 2:
                continue

            headers = extracted_matrix[0]
            data_rows = extracted_matrix[1:]

            md_lines = []
            md_lines.append("| " + " | ".join(headers) + " |")
            md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for r in data_rows:
                md_lines.append("| " + " | ".join(r) + " |")
            markdown_table = "\n".join(md_lines)

            flat_tokens = []
            for row_cells in extracted_matrix:
                flat_tokens.extend([c for c in row_cells if c])
            raw_flattened_text = " ".join(flat_tokens)

            tables_data.append({
                "table_id": f"tbl_p{page_number}_{c_idx + 1}",
                "page_number": page_number,
                "caption": f"Table {c_idx + 1} (Page {page_number})",
                "num_rows": len(data_rows),
                "num_cols": len(headers),
                "headers": headers,
                "rows": data_rows,
                "markdown_table": markdown_table,
                "raw_flattened_text": raw_flattened_text,
                "bounding_box": {
                    "x0": round(x0, 1),
                    "y0": round(y_top, 1),
                    "x1": round(x1, 1),
                    "y1": round(y_bottom, 1),
                    "width": round(x1 - x0, 1),
                    "height": round(y_bottom - y_top, 1),
                },
            })
        return tables_data

    @classmethod
    def _extract_tables_text_alignment(
        cls, page: pymupdf.Page, page_number: int
    ) -> List[Dict[str, Any]]:
        """Strategy C: Borderless text-alignment table detection using PyMuPDF text strategy."""
        tables_data = []
        try:
            tabs = page.find_tables(strategy="text")
            page_rect = page.rect
            for idx, tab in enumerate(tabs):
                tx0, ty0, tx1, ty1 = tab.bbox
                # Avoid full-page false positives
                if (tx1 - tx0) > 0.95 * page_rect.width and (ty1 - ty0) > 0.85 * page_rect.height:
                    continue

                raw_extracted = tab.extract()
                if not raw_extracted or len(raw_extracted) < 3:
                    continue

                headers = [str(c or "").strip() for c in raw_extracted[0]]
                raw_rows = raw_extracted[1:]
                rows = [[str(c or "").strip() for c in r] for r in raw_rows]

                # Filter empty rows
                rows = [r for r in rows if any(r)]
                if len(headers) < 2 or len(rows) < 2:
                    continue

                md_lines = []
                if headers:
                    md_lines.append("| " + " | ".join(headers) + " |")
                    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for r in rows:
                    md_lines.append("| " + " | ".join(r) + " |")
                markdown_table = "\n".join(md_lines)

                flat_tokens = []
                for row_cells in raw_extracted:
                    flat_tokens.extend([str(c or "").strip() for c in row_cells if c])
                raw_flattened_text = " ".join(flat_tokens)

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
        except Exception as e:
            logger.debug(f"Strategy C text alignment failed on page {page_number}: {e}")
        return tables_data

    @classmethod
    def extract_tables_from_page(
        cls, page: pymupdf.Page, page_number: int
    ) -> List[Dict[str, Any]]:
        """Extract all tables on a page using multi-strategy detection."""
        # 1. Try standard bordered detection
        try:
            std = cls._extract_tables_standard(page, page_number)
            if std:
                return std
        except Exception as e:
            logger.debug(f"Strategy A standard extraction failed on page {page_number}: {e}")

        # 2. Try horizontal rule-divided detection
        try:
            rule = cls._extract_tables_rule_based(page, page_number)
            if rule:
                return rule
        except Exception as e:
            logger.debug(f"Strategy B rule-based extraction failed on page {page_number}: {e}")

        # 3. Try borderless text-alignment detection
        try:
            text_align = cls._extract_tables_text_alignment(page, page_number)
            if text_align:
                return text_align
        except Exception as e:
            logger.debug(f"Strategy C text alignment failed on page {page_number}: {e}")

        return []

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
