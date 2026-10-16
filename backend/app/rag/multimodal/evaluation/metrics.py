from typing import List, Dict, Any
from app.rag.multimodal.ocr.tesseract_provider import EducationalOCRComparator


class MultimodalMetrics:
    """Computes quantitative metrics for OCR, tables, retrieval, and grounding."""

    @staticmethod
    def calculate_cer(reference: str, hypothesis: str) -> float:
        return EducationalOCRComparator.compute_cer(reference, hypothesis)

    @staticmethod
    def calculate_wer(reference: str, hypothesis: str) -> float:
        return EducationalOCRComparator.compute_wer(reference, hypothesis)

    @staticmethod
    def calculate_table_accuracy(
        expected_table: Dict[str, Any],
        actual_table: Dict[str, Any],
    ) -> float:
        """Compares row count, col count, and header alignment between ground truth and extracted table."""
        exp_headers = [h.lower().strip() for h in expected_table.get("headers", [])]
        act_headers = [h.lower().strip() for h in actual_table.get("headers", [])]

        if not exp_headers:
            return 1.0 if not act_headers else 0.5

        # Header IoU
        intersection = len(set(exp_headers).intersection(set(act_headers)))
        union = len(set(exp_headers).union(set(act_headers)))
        header_iou = intersection / union if union > 0 else 0.0

        # Row count diff penalty
        exp_rows = expected_table.get("num_rows", len(expected_table.get("rows", [])))
        act_rows = actual_table.get("num_rows", len(actual_table.get("rows", [])))
        row_diff = abs(exp_rows - act_rows)
        row_score = max(0.0, 1.0 - (row_diff / max(1, exp_rows)))

        return round(0.5 * header_iou + 0.5 * row_score, 3)

    @staticmethod
    def calculate_retrieval_recall(
        retrieved_ids: List[str],
        ground_truth_ids: List[str],
    ) -> float:
        if not ground_truth_ids:
            return 1.0
        hits = sum(1 for gid in ground_truth_ids if gid in retrieved_ids)
        return round(hits / len(ground_truth_ids), 3)
