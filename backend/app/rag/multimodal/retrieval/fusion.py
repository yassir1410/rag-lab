from typing import List, Dict, Any


class HybridMultimodalFusion:
    """Fuses OCR textual retrieval and visual embedding retrieval using Reciprocal Rank Fusion (RRF)."""

    DEFAULT_RRF_K = 60

    @classmethod
    def rrf_fuse(
        cls,
        text_results: List[Dict[str, Any]],
        visual_results: List[Dict[str, Any]],
        top_k: int = 5,
        text_weight: float = 0.5,
        visual_weight: float = 0.5,
        k: int = DEFAULT_RRF_K,
    ) -> List[Dict[str, Any]]:
        scores: Dict[str, float] = {}
        item_map: Dict[str, Dict[str, Any]] = {}

        # 1. Score text results
        for rank, item in enumerate(text_results):
            iid = item["id"]
            item_map[iid] = item
            rrf_contrib = text_weight * (1.0 / (k + rank + 1))
            scores[iid] = scores.get(iid, 0.0) + rrf_contrib

        # 2. Score visual results
        for rank, item in enumerate(visual_results):
            iid = item["id"]
            if iid not in item_map:
                item_map[iid] = item
            rrf_contrib = visual_weight * (1.0 / (k + rank + 1))
            scores[iid] = scores.get(iid, 0.0) + rrf_contrib

        # Sort items by merged RRF score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        max_score = max(scores.values()) if scores else 1.0

        fused_results = []
        for iid in sorted_ids[:top_k]:
            entry = dict(item_map[iid])
            # Normalize RRF score to 0..1 range for UI clarity
            entry["score"] = round(scores[iid] / max_score if max_score > 0 else scores[iid], 4)
            entry["strategy"] = "hybrid"
            fused_results.append(entry)

        return fused_results
