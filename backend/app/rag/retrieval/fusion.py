import uuid
from typing import List, Dict, Any, Optional, Tuple
from app.rag.retrieval.schemas import SearchResultItem


class CandidateRankingAudit:
    """Detailed trace of an individual chunk across retrieval and fusion stages."""

    def __init__(
        self,
        chunk_id: uuid.UUID,
        item: SearchResultItem,
        dense_rank: Optional[int] = None,
        dense_score: Optional[float] = None,
        lexical_rank: Optional[int] = None,
        lexical_score: Optional[float] = None,
        rrf_score: float = 0.0,
        linear_score: float = 0.0,
        rerank_score: Optional[float] = None,
        final_rank: int = 1,
    ):
        self.chunk_id = chunk_id
        self.item = item
        self.dense_rank = dense_rank
        self.dense_score = dense_score
        self.lexical_rank = lexical_rank
        self.lexical_score = lexical_score
        self.rrf_score = rrf_score
        self.linear_score = linear_score
        self.rerank_score = rerank_score
        self.final_rank = final_rank

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.item.document_id),
            "document_name": self.item.document_name,
            "sequence_index": self.item.sequence_index,
            "content_snippet": self.item.content[:160] + "..." if len(self.item.content) > 160 else self.item.content,
            "dense_rank": self.dense_rank,
            "dense_score": round(self.dense_score, 4) if self.dense_score is not None else None,
            "lexical_rank": self.lexical_rank,
            "lexical_score": round(self.lexical_score, 4) if self.lexical_score is not None else None,
            "rrf_score": round(self.rrf_score, 6),
            "linear_score": round(self.linear_score, 4),
            "rerank_score": round(self.rerank_score, 4) if self.rerank_score is not None else None,
            "final_rank": self.final_rank,
        }


def min_max_normalize(scores: List[float]) -> List[float]:
    """Normalizes an arbitrary list of numeric scores to the [0, 1] range."""
    if not scores:
        return []
    min_val = min(scores)
    max_val = max(scores)
    if max_val - min_val <= 1e-9:
        return [1.0 for _ in scores]
    return [(s - min_val) / (max_val - min_val) for s in scores]


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF) and Score-Based Blending Engine.
    Formula:
        RRF_score(d) = sum_{m in M} ( 1 / (k + rank_m(d)) )
    """

    def __init__(self, default_k: int = 60):
        self.default_k = default_k

    def fuse_rrf(
        self,
        dense_results: List[SearchResultItem],
        lexical_results: List[SearchResultItem],
        k: Optional[int] = None,
        top_k: int = 5,
    ) -> Tuple[List[SearchResultItem], List[CandidateRankingAudit]]:
        """
        Executes Reciprocal Rank Fusion over dense and lexical candidate lists.
        Returns:
            (fused_results, candidate_audits)
        """
        k_val = k if (k is not None and k > 0) else self.default_k

        # Map candidates by chunk_id
        items_by_id: Dict[uuid.UUID, SearchResultItem] = {}
        dense_ranks: Dict[uuid.UUID, int] = {}
        dense_scores: Dict[uuid.UUID, float] = {}
        lexical_ranks: Dict[uuid.UUID, int] = {}
        lexical_scores: Dict[uuid.UUID, float] = {}

        for idx, item in enumerate(dense_results, start=1):
            items_by_id[item.chunk_id] = item
            dense_ranks[item.chunk_id] = idx
            dense_scores[item.chunk_id] = item.similarity_score

        for idx, item in enumerate(lexical_results, start=1):
            if item.chunk_id not in items_by_id:
                items_by_id[item.chunk_id] = item
            lexical_ranks[item.chunk_id] = idx
            lexical_scores[item.chunk_id] = item.similarity_score

        # Calculate RRF score for every unique candidate
        all_candidate_ids = set(items_by_id.keys())
        rrf_scores: Dict[uuid.UUID, float] = {}

        for cid in all_candidate_ids:
            score = 0.0
            if cid in dense_ranks:
                score += 1.0 / (k_val + dense_ranks[cid])
            if cid in lexical_ranks:
                score += 1.0 / (k_val + lexical_ranks[cid])
            rrf_scores[cid] = score

        # Sort candidates descending by RRF score
        # Tie-breaker: dense rank if present, then lexical rank
        sorted_ids = sorted(
            all_candidate_ids,
            key=lambda cid: (
                rrf_scores[cid],
                -(dense_ranks.get(cid, 999999)),
                -(lexical_ranks.get(cid, 999999)),
            ),
            reverse=True,
        )

        fused_items: List[SearchResultItem] = []
        audits: List[CandidateRankingAudit] = []

        for final_rank, cid in enumerate(sorted_ids, start=1):
            orig_item = items_by_id[cid]
            r_score = rrf_scores[cid]

            # Construct copy with fused metadata
            fused_item = orig_item.model_copy(deep=True)
            fused_item.similarity_score = r_score
            fused_item.distance_metric = f"rrf_k{k_val}"
            fused_item.similarity_label = f"RRF: {r_score:.5f}"
            fused_item.distance_label = (
                f"D-Rank: {dense_ranks.get(cid, '-')} | L-Rank: {lexical_ranks.get(cid, '-')}"
            )

            audit = CandidateRankingAudit(
                chunk_id=cid,
                item=orig_item,
                dense_rank=dense_ranks.get(cid),
                dense_score=dense_scores.get(cid),
                lexical_rank=lexical_ranks.get(cid),
                lexical_score=lexical_scores.get(cid),
                rrf_score=r_score,
                final_rank=final_rank,
            )
            audits.append(audit)

            if len(fused_items) < top_k:
                fused_items.append(fused_item)

        return fused_items, audits

    def fuse_linear(
        self,
        dense_results: List[SearchResultItem],
        lexical_results: List[SearchResultItem],
        alpha: float = 0.5,
        top_k: int = 5,
    ) -> Tuple[List[SearchResultItem], List[CandidateRankingAudit]]:
        """
        Executes normalized linear score combination:
            Score = alpha * norm(dense_score) + (1 - alpha) * norm(lexical_score)
        """
        alpha_val = min(1.0, max(0.0, alpha))
        items_by_id: Dict[uuid.UUID, SearchResultItem] = {}
        dense_ranks: Dict[uuid.UUID, int] = {}
        dense_raw: Dict[uuid.UUID, float] = {}
        lexical_ranks: Dict[uuid.UUID, int] = {}
        lexical_raw: Dict[uuid.UUID, float] = {}

        for idx, item in enumerate(dense_results, start=1):
            items_by_id[item.chunk_id] = item
            dense_ranks[item.chunk_id] = idx
            dense_raw[item.chunk_id] = item.similarity_score

        for idx, item in enumerate(lexical_results, start=1):
            if item.chunk_id not in items_by_id:
                items_by_id[item.chunk_id] = item
            lexical_ranks[item.chunk_id] = idx
            lexical_raw[item.chunk_id] = item.similarity_score

        # Normalize dense and lexical scores independently
        dense_ids = list(dense_raw.keys())
        norm_dense = min_max_normalize([dense_raw[cid] for cid in dense_ids])
        dense_norm_map = {cid: norm for cid, norm in zip(dense_ids, norm_dense)}

        lexical_ids = list(lexical_raw.keys())
        norm_lexical = min_max_normalize([lexical_raw[cid] for cid in lexical_ids])
        lexical_norm_map = {cid: norm for cid, norm in zip(lexical_ids, norm_lexical)}

        all_candidate_ids = set(items_by_id.keys())
        comb_scores: Dict[uuid.UUID, float] = {}

        for cid in all_candidate_ids:
            d_norm = dense_norm_map.get(cid, 0.0)
            l_norm = lexical_norm_map.get(cid, 0.0)
            score = alpha_val * d_norm + (1.0 - alpha_val) * l_norm
            comb_scores[cid] = score

        sorted_ids = sorted(
            all_candidate_ids,
            key=lambda cid: comb_scores[cid],
            reverse=True,
        )

        blended_items: List[SearchResultItem] = []
        audits: List[CandidateRankingAudit] = []

        for final_rank, cid in enumerate(sorted_ids, start=1):
            orig_item = items_by_id[cid]
            c_score = comb_scores[cid]

            item_copy = orig_item.model_copy(deep=True)
            item_copy.similarity_score = c_score
            item_copy.distance_metric = f"linear_a{alpha_val:.2f}"
            item_copy.similarity_label = f"Blended: {c_score:.4f}"
            item_copy.distance_label = (
                f"D: {dense_raw.get(cid, 0.0):.3f} | L: {lexical_raw.get(cid, 0.0):.3f}"
            )

            audit = CandidateRankingAudit(
                chunk_id=cid,
                item=orig_item,
                dense_rank=dense_ranks.get(cid),
                dense_score=dense_raw.get(cid),
                lexical_rank=lexical_ranks.get(cid),
                lexical_score=lexical_raw.get(cid),
                linear_score=c_score,
                final_rank=final_rank,
            )
            audits.append(audit)

            if len(blended_items) < top_k:
                blended_items.append(item_copy)

        return blended_items, audits


rank_fusion = ReciprocalRankFusion()
