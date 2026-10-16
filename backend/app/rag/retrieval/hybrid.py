import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session

from app.rag.embeddings.registry import embedding_registry
from app.rag.retrieval.schemas import (
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    CandidateAuditItem,
    RetrievalStrategySpec,
)
from app.rag.retrieval.exact import ExactVectorSearcher
from app.rag.retrieval.hnsw import HNSWVectorSearcher
from app.rag.retrieval.ivfflat import IVFFlatVectorSearcher
from app.rag.retrieval.lexical import lexical_searcher
from app.rag.retrieval.fusion import rank_fusion, CandidateRankingAudit
from app.rag.retrieval.reranker import reranker_registry
from app.rag.retrieval.query_rewriter import query_processor
from app.rag.retrieval.context_optimizer import context_optimizer


RETRIEVAL_STRATEGIES: List[RetrievalStrategySpec] = [
    RetrievalStrategySpec(
        id="dense_only",
        name="Dense Vector Retrieval",
        description="Standard single-vector similarity search using pgvector (Exact KNN, HNSW, or IVFFlat).",
        uses_dense=True,
        uses_lexical=False,
        uses_fusion=False,
        uses_reranker=False,
        uses_query_expansion=False,
        uses_context_optimization=False,
    ),
    RetrievalStrategySpec(
        id="lexical_only",
        name="PostgreSQL Full-Text Search",
        description="Lexical keyword matching using native GIN-indexed tsvector and ts_rank_cd scoring.",
        uses_dense=False,
        uses_lexical=True,
        uses_fusion=False,
        uses_reranker=False,
        uses_query_expansion=False,
        uses_context_optimization=False,
    ),
    RetrievalStrategySpec(
        id="hybrid_linear",
        name="Hybrid Search (Linear Combination)",
        description="Parallel dense and lexical retrieval combined via min-max normalized weighted scores.",
        uses_dense=True,
        uses_lexical=True,
        uses_fusion=True,
        uses_reranker=False,
        uses_query_expansion=False,
        uses_context_optimization=False,
    ),
    RetrievalStrategySpec(
        id="hybrid_rrf",
        name="Hybrid Search (Reciprocal Rank Fusion)",
        description="Score-agnostic ordinal rank fusion combining dense and lexical lists with smoothing constant k.",
        uses_dense=True,
        uses_lexical=True,
        uses_fusion=True,
        uses_reranker=False,
        uses_query_expansion=False,
        uses_context_optimization=False,
    ),
    RetrievalStrategySpec(
        id="dense_rerank",
        name="Dense Retrieval + Cross-Encoder Rerank",
        description="Dense vector candidate retrieval followed by deep joint-attention Cross-Encoder rescoring.",
        uses_dense=True,
        uses_lexical=False,
        uses_fusion=False,
        uses_reranker=True,
        uses_query_expansion=False,
        uses_context_optimization=False,
    ),
    RetrievalStrategySpec(
        id="hybrid_rrf_rerank",
        name="Hybrid RRF + Cross-Encoder Rerank",
        description="Dense and lexical search fused with RRF, then refined by second-stage Cross-Encoder.",
        uses_dense=True,
        uses_lexical=True,
        uses_fusion=True,
        uses_reranker=True,
        uses_query_expansion=False,
        uses_context_optimization=False,
    ),
    RetrievalStrategySpec(
        id="multi_query_hybrid_rerank",
        name="Multi-Query Expansion + Hybrid RRF + Rerank",
        description="Decomposes user query into sub-queries, executes multi-perspective hybrid search, and reranks candidates.",
        uses_dense=True,
        uses_lexical=True,
        uses_fusion=True,
        uses_reranker=True,
        uses_query_expansion=True,
        uses_context_optimization=False,
    ),
    RetrievalStrategySpec(
        id="adaptive_context",
        name="Adaptive Context (Neighbor Expansion & Pruning)",
        description="Hybrid retrieval enriched with adjacent chunk expansion, redundancy pruning, and score dropoff cutoff.",
        uses_dense=True,
        uses_lexical=True,
        uses_fusion=True,
        uses_reranker=True,
        uses_query_expansion=False,
        uses_context_optimization=True,
    ),
]


class HybridRetrievalEngine:
    """
    Central orchestrator for Phase 08 Advanced Retrieval strategies.
    Coordinates query expansion, dense/lexical retrieval, fusion, reranking,
    context expansion, and candidate ranking provenance.
    """

    def __init__(self):
        self.registry = embedding_registry
        self.exact_searcher = ExactVectorSearcher()
        self.hnsw_searcher = HNSWVectorSearcher()
        self.ivfflat_searcher = IVFFlatVectorSearcher()
        self.lexical_searcher = lexical_searcher
        self.fusion = rank_fusion
        self.reranker_registry = reranker_registry
        self.query_processor = query_processor
        self.context_optimizer = context_optimizer

    def list_strategies(self) -> List[RetrievalStrategySpec]:
        return RETRIEVAL_STRATEGIES

    def _execute_dense_search(
        self,
        session: Session,
        query: str,
        request: SearchRequest,
        candidate_k: int,
    ) -> Tuple[List[SearchResultItem], float, float, Any, List[str]]:
        """Encodes query and executes dense vector search."""
        spec = self.registry.get_spec(request.model_id)
        provider = self.registry.get_provider(request.model_id)

        t0 = time.perf_counter()
        encode_res = provider.encode(
            [query],
            normalize_embeddings=True,
            is_query=True,
        )
        embed_lat = (time.perf_counter() - t0) * 1000.0

        if not encode_res.vectors:
            raise ValueError("Failed to generate embedding vector for query.")
        query_vector = encode_res.vectors[0]

        algo = request.search_algorithm.lower()
        warnings: List[str] = []

        if algo == "exact":
            results, plan, query_lat = self.exact_searcher.search(
                session=session,
                query_vector=query_vector,
                dimension=spec.dimension,
                model_id=request.model_id,
                metric=request.distance_metric,
                top_k=candidate_k,
                filters=request.filters,
                explain_plan=request.explain_plan,
            )
        elif algo == "hnsw":
            results, plan, query_lat = self.hnsw_searcher.search(
                session=session,
                query_vector=query_vector,
                dimension=spec.dimension,
                model_id=request.model_id,
                metric=request.distance_metric,
                top_k=candidate_k,
                filters=request.filters,
                explain_plan=request.explain_plan,
                ef_search=request.ef_search or 40,
            )
        elif algo == "ivfflat":
            results, plan, query_lat, ivf_warns = self.ivfflat_searcher.search(
                session=session,
                query_vector=query_vector,
                dimension=spec.dimension,
                model_id=request.model_id,
                metric=request.distance_metric,
                top_k=candidate_k,
                filters=request.filters,
                explain_plan=request.explain_plan,
                probes=request.probes or 1,
            )
            warnings.extend(ivf_warns)
        else:
            raise ValueError(f"Unsupported search algorithm '{algo}'.")

        return results, embed_lat, query_lat, plan, warnings

    def execute(self, session: Session, request: SearchRequest) -> SearchResponse:
        t_total_start = time.perf_counter()
        strategy = (request.strategy or "dense_only").lower().strip()
        timing: Dict[str, float] = {}
        warnings: List[str] = []
        expanded_queries: List[str] = [request.query]

        candidate_pool_size = max(request.top_k * 3, 15)

        # ----------------------------------------------------
        # 1. DENSE ONLY STRATEGY
        # ----------------------------------------------------
        if strategy == "dense_only":
            dense_res, embed_lat, query_lat, plan, warns = self._execute_dense_search(
                session, request.query, request, request.top_k
            )
            warnings.extend(warns)
            timing["embedding_ms"] = embed_lat
            timing["dense_search_ms"] = query_lat
            total_lat = (time.perf_counter() - t_total_start) * 1000.0

            audits = [
                CandidateAuditItem(
                    chunk_id=item.chunk_id,
                    document_id=item.document_id,
                    document_name=item.document_name,
                    sequence_index=item.sequence_index,
                    content_snippet=item.content[:160] + "...",
                    dense_rank=rank,
                    dense_score=item.similarity_score,
                    final_rank=rank,
                )
                for rank, item in enumerate(dense_res, start=1)
            ]

            return SearchResponse(
                query=request.query,
                strategy="dense_only",
                model_id=request.model_id,
                search_algorithm=request.search_algorithm,
                distance_metric=request.distance_metric,
                top_k=request.top_k,
                total_candidates=len(dense_res),
                results=dense_res,
                candidate_progression=audits,
                query_latency_ms=query_lat,
                embedding_latency_ms=embed_lat,
                rerank_latency_ms=0.0,
                total_latency_ms=total_lat,
                timing_breakdown=timing,
                expanded_queries=expanded_queries,
                execution_plan=plan,
                warnings=warnings,
            )

        # ----------------------------------------------------
        # 2. LEXICAL ONLY STRATEGY
        # ----------------------------------------------------
        if strategy == "lexical_only":
            lex_res, lex_lat = self.lexical_searcher.search(
                session=session,
                query=request.query,
                top_k=request.top_k,
                filters=request.filters,
                language=request.language,
            )
            timing["lexical_search_ms"] = lex_lat
            total_lat = (time.perf_counter() - t_total_start) * 1000.0

            audits = [
                CandidateAuditItem(
                    chunk_id=item.chunk_id,
                    document_id=item.document_id,
                    document_name=item.document_name,
                    sequence_index=item.sequence_index,
                    content_snippet=item.content[:160] + "...",
                    lexical_rank=rank,
                    lexical_score=item.similarity_score,
                    final_rank=rank,
                )
                for rank, item in enumerate(lex_res, start=1)
            ]

            return SearchResponse(
                query=request.query,
                strategy="lexical_only",
                model_id=request.model_id,
                search_algorithm=request.search_algorithm,
                distance_metric=request.distance_metric,
                top_k=request.top_k,
                total_candidates=len(lex_res),
                results=lex_res,
                candidate_progression=audits,
                query_latency_ms=lex_lat,
                embedding_latency_ms=0.0,
                rerank_latency_ms=0.0,
                total_latency_ms=total_lat,
                timing_breakdown=timing,
                expanded_queries=expanded_queries,
                execution_plan=None,
                warnings=warnings,
            )

        # ----------------------------------------------------
        # 3. QUERY EXPANSION (for multi_query_hybrid_rerank)
        # ----------------------------------------------------
        if strategy == "multi_query_hybrid_rerank" or request.enable_query_expansion:
            expanded_queries = self.query_processor.expand_multi_query(request.query, max_queries=3)

        # ----------------------------------------------------
        # 4. EXECUTE FIRST-STAGE RETRIEVAL (DENSE + LEXICAL)
        # ----------------------------------------------------
        all_dense_candidates: List[SearchResultItem] = []
        embed_lat_total = 0.0
        dense_lat_total = 0.0

        for q_text in expanded_queries:
            dense_res, e_lat, q_lat, plan, warns = self._execute_dense_search(
                session, q_text, request, candidate_pool_size
            )
            embed_lat_total += e_lat
            dense_lat_total += q_lat
            warnings.extend(warns)
            all_dense_candidates.extend(dense_res)

        # Deduplicate dense candidates while keeping highest similarity
        dense_map: Dict[uuid.UUID, SearchResultItem] = {}
        for item in all_dense_candidates:
            if item.chunk_id not in dense_map or item.similarity_score > dense_map[item.chunk_id].similarity_score:
                dense_map[item.chunk_id] = item
        dense_candidates = sorted(dense_map.values(), key=lambda x: x.similarity_score, reverse=True)

        all_lex_candidates: List[SearchResultItem] = []
        lex_lat_total = 0.0

        for q_text in expanded_queries:
            lex_res, l_lat = self.lexical_searcher.search(
                session=session,
                query=q_text,
                top_k=candidate_pool_size,
                filters=request.filters,
                language=request.language,
            )
            lex_lat_total += l_lat
            all_lex_candidates.extend(lex_res)

        lex_map: Dict[uuid.UUID, SearchResultItem] = {}
        for item in all_lex_candidates:
            if item.chunk_id not in lex_map or item.similarity_score > lex_map[item.chunk_id].similarity_score:
                lex_map[item.chunk_id] = item
        lexical_candidates = sorted(lex_map.values(), key=lambda x: x.similarity_score, reverse=True)

        timing["embedding_ms"] = embed_lat_total
        timing["dense_search_ms"] = dense_lat_total
        timing["lexical_search_ms"] = lex_lat_total

        # ----------------------------------------------------
        # 5. FUSION OR CANDIDATE SELECTION
        # ----------------------------------------------------
        fused_candidates: List[SearchResultItem] = []
        audits_raw: List[CandidateRankingAudit] = []

        if strategy in ("hybrid_rrf", "hybrid_rrf_rerank", "multi_query_hybrid_rerank", "adaptive_context"):
            t_fuse = time.perf_counter()
            fused_candidates, audits_raw = self.fusion.fuse_rrf(
                dense_results=dense_candidates,
                lexical_results=lexical_candidates,
                k=request.rrf_k,
                top_k=candidate_pool_size,
            )
            timing["fusion_ms"] = (time.perf_counter() - t_fuse) * 1000.0

        elif strategy == "hybrid_linear":
            t_fuse = time.perf_counter()
            fused_candidates, audits_raw = self.fusion.fuse_linear(
                dense_results=dense_candidates,
                lexical_results=lexical_candidates,
                alpha=request.hybrid_alpha,
                top_k=candidate_pool_size,
            )
            timing["fusion_ms"] = (time.perf_counter() - t_fuse) * 1000.0

        elif strategy == "dense_rerank":
            fused_candidates = dense_candidates[:candidate_pool_size]
            audits_raw = [
                CandidateRankingAudit(
                    chunk_id=item.chunk_id,
                    item=item,
                    dense_rank=rank,
                    dense_score=item.similarity_score,
                    final_rank=rank,
                )
                for rank, item in enumerate(fused_candidates, start=1)
            ]

        # ----------------------------------------------------
        # 6. RERANKING STAGE
        # ----------------------------------------------------
        rerank_lat = 0.0
        final_candidates = fused_candidates

        use_reranker = strategy in (
            "dense_rerank",
            "hybrid_rrf_rerank",
            "multi_query_hybrid_rerank",
            "adaptive_context",
        ) or request.reranker_model != "none"

        rerank_scores_map: Dict[uuid.UUID, float] = {}

        if use_reranker and fused_candidates:
            reranker_model_name = request.reranker_model if request.reranker_model != "none" else "tinybert"
            reranker = self.reranker_registry.get_reranker(reranker_model_name)
            t_rerank = time.perf_counter()
            final_candidates, rerank_lat, raw_rerank_scores = reranker.rerank(
                query=request.query,
                candidates=fused_candidates,
                top_k=candidate_pool_size,
            )
            timing["rerank_ms"] = rerank_lat
            for item, s_score in zip(final_candidates, raw_rerank_scores):
                rerank_scores_map[item.chunk_id] = s_score

        # ----------------------------------------------------
        # 7. CONTEXT OPTIMIZATION (for adaptive_context)
        # ----------------------------------------------------
        if strategy == "adaptive_context" or request.enable_neighbor_expansion:
            t_opt = time.perf_counter()
            # Neighbor expansion (+-1)
            expanded_items = self.context_optimizer.expand_neighboring_chunks(
                session=session,
                items=final_candidates[:request.top_k],
                max_expansion=1,
            )
            # Redundancy pruning
            pruned_items = self.context_optimizer.prune_redundancy(
                items=expanded_items,
                similarity_threshold=0.80,
            )
            # Adaptive cutoff
            final_candidates = self.context_optimizer.apply_adaptive_cutoff(
                items=pruned_items,
                relative_threshold=0.35,
                min_k=min(2, request.top_k),
            )
            timing["context_optimization_ms"] = (time.perf_counter() - t_opt) * 1000.0

        # Truncate to desired top_k
        final_results = final_candidates[:request.top_k]

        # ----------------------------------------------------
        # 8. ASSEMBLE PROGRESSION AUDIT
        # ----------------------------------------------------
        final_results_rank_map = {item.chunk_id: r for r, item in enumerate(final_results, start=1)}
        audit_items: List[CandidateAuditItem] = []

        for aud in audits_raw:
            c_dict = aud.to_dict()
            f_rank = final_results_rank_map.get(aud.chunk_id, 999)
            audit_items.append(
                CandidateAuditItem(
                    chunk_id=aud.chunk_id,
                    document_id=uuid.UUID(c_dict["document_id"]),
                    document_name=c_dict["document_name"],
                    sequence_index=c_dict["sequence_index"],
                    content_snippet=c_dict["content_snippet"],
                    dense_rank=c_dict["dense_rank"],
                    dense_score=c_dict["dense_score"],
                    lexical_rank=c_dict["lexical_rank"],
                    lexical_score=c_dict["lexical_score"],
                    rrf_score=c_dict["rrf_score"],
                    linear_score=c_dict["linear_score"],
                    rerank_score=rerank_scores_map.get(aud.chunk_id),
                    final_rank=f_rank,
                )
            )

        audit_items.sort(key=lambda a: a.final_rank)

        total_lat = (time.perf_counter() - t_total_start) * 1000.0

        return SearchResponse(
            query=request.query,
            strategy=strategy,
            model_id=request.model_id,
            search_algorithm=request.search_algorithm,
            distance_metric=request.distance_metric,
            top_k=request.top_k,
            total_candidates=len(fused_candidates),
            results=final_results,
            candidate_progression=audit_items,
            query_latency_ms=timing.get("dense_search_ms", 0.0) + timing.get("lexical_search_ms", 0.0),
            embedding_latency_ms=timing.get("embedding_ms", 0.0),
            rerank_latency_ms=rerank_lat,
            total_latency_ms=total_lat,
            timing_breakdown=timing,
            expanded_queries=expanded_queries,
            execution_plan=None,
            warnings=warnings,
        )


hybrid_engine = HybridRetrievalEngine()
