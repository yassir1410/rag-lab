import time
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.rag.retrieval.schemas import (
    SearchResultItem,
    ExecutionPlanSummary,
    RetrievalFilter,
)
from app.rag.retrieval.base import (
    BaseVectorSearcher,
    METRIC_TO_OPERATOR,
    compute_similarity_score,
    parse_execution_plan,
)
from app.rag.retrieval.exact import ExactVectorSearcher


class HNSWVectorSearcher(BaseVectorSearcher):
    """
    Approximate Nearest Neighbor (ANN) searcher utilizing PostgreSQL pgvector HNSW indexes.
    Traverses multi-layer proximity graphs with configurable query-time depth (hnsw.ef_search).
    """

    def __init__(self):
        self._exact_fallback = ExactVectorSearcher()

    def search(
        self,
        session: Session,
        query_vector: List[float],
        dimension: int,
        model_id: str,
        metric: str = "cosine",
        top_k: int = 5,
        filters: Optional[RetrievalFilter] = None,
        explain_plan: bool = True,
        ef_search: int = 40,
        force_index_scan: bool = False,
        **kwargs,
    ) -> Tuple[List[SearchResultItem], Optional[ExecutionPlanSummary], float]:
        is_postgres = session.bind.dialect.name == "postgresql" if session.bind else False
        if not is_postgres:
            # Fallback to exact searcher in SQLite unit tests
            results, plan, lat = self._exact_fallback.search(
                session=session,
                query_vector=query_vector,
                dimension=dimension,
                model_id=model_id,
                metric=metric,
                top_k=top_k,
                filters=filters,
                explain_plan=explain_plan,
            )
            if plan:
                plan.node_type = "HNSW (Emulated / SQLite)"
            return results, plan, lat

        operator = METRIC_TO_OPERATOR.get(metric, "<=>")
        vec_literal = "[" + ",".join(f"{x:.6f}" for x in query_vector) + "]"

        # Build dynamic filter WHERE clauses
        filter_clauses = []
        params: Dict[str, Any] = {
            "model_id": model_id,
            "dimension": dimension,
            "top_k": top_k,
            "q_vec": vec_literal,
        }

        if filters:
            if filters.document_id:
                filter_clauses.append("(d.id = :filter_doc_id OR ee.document_id = :filter_doc_id)")
                params["filter_doc_id"] = filters.document_id
            if filters.mime_type:
                filter_clauses.append("d.mime_type = :filter_mime")
                params["filter_mime"] = filters.mime_type
            if filters.chunking_experiment_id:
                filter_clauses.append("dc.experiment_id = :filter_chunking_exp")
                params["filter_chunking_exp"] = filters.chunking_experiment_id
            if filters.page_number is not None:
                filter_clauses.append(
                    "(dc.metadata->>'page_number' = :filter_page OR dc.metadata->>'page' = :filter_page)"
                )
                params["filter_page"] = str(filters.page_number)

        filter_sql = ""
        if filter_clauses:
            filter_sql = " AND " + " AND ".join(filter_clauses)

        start_time = time.perf_counter()

        # Ensure vector extension is loaded and disable JIT compilation penalty
        try:
            session.execute(text("LOAD 'vector'"))
        except Exception:
            pass
        session.execute(text("SET LOCAL jit = off"))

        # Set transaction-isolated HNSW search depth parameter
        clamped_ef = max(1, min(500, int(ef_search)))
        session.execute(text(f"SET LOCAL hnsw.ef_search = {clamped_ef}"))
        session.execute(text("SET LOCAL enable_indexscan = on"))

        if force_index_scan:
            # Used in benchmarks to evaluate index traversal even on small tables
            session.execute(text("SET LOCAL enable_seqscan = off"))

        base_query = f"""
            SELECT
                ce.id AS chunk_embedding_id,
                ce.chunk_id,
                ce.sequence_index,
                ce.token_count,
                ce.text_snippet,
                COALESCE(dc.document_id, ee.document_id) AS resolved_doc_id,
                COALESCE(d.original_filename, 'Synthetic / External') AS doc_name,
                COALESCE(d.mime_type, 'text/plain') AS doc_mime,
                COALESCE(dc.char_count, LENGTH(ce.text_snippet)) AS char_count,
                COALESCE(dc.content, ce.text_snippet) AS full_content,
                COALESCE(dc.metadata, '{{}}'::jsonb) AS chunk_meta,
                (CAST(ce.vector AS vector({dimension})) {operator} CAST(:q_vec AS vector({dimension}))) AS raw_dist
            FROM chunk_embeddings ce
            JOIN embedding_experiments ee ON ce.experiment_id = ee.id
            LEFT JOIN document_chunks dc ON ce.chunk_id = dc.id
            LEFT JOIN documents d ON (dc.document_id = d.id OR ee.document_id = d.id)
            WHERE ce.vector IS NOT NULL
              AND ee.model_id = :model_id
              AND ce.vector_dimension = :dimension
              {filter_sql}
            ORDER BY (CAST(ce.vector AS vector({dimension})) {operator} CAST(:q_vec AS vector({dimension}))) ASC
            LIMIT :top_k
        """

        plan_summary: Optional[ExecutionPlanSummary] = None
        if explain_plan:
            try:
                explain_query = f"EXPLAIN (ANALYZE, FORMAT JSON) {base_query}"
                res_explain = session.execute(text(explain_query), params).scalar()
                if isinstance(res_explain, str):
                    res_explain = json.loads(res_explain)
                plan_summary = parse_execution_plan(res_explain)
            except Exception as e:
                plan_summary = ExecutionPlanSummary(
                    node_type="HNSW Index / Plan Inspection Error",
                    planning_time_ms=0.0,
                    execution_time_ms=0.0,
                    actual_rows=0,
                    total_cost=0.0,
                    plan_tree_summary=f"Plan Error: {e}",
                    raw_plan={},
                )

        rows = session.execute(text(base_query), params).fetchall()
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        results: List[SearchResultItem] = []
        for row in rows:
            raw_dist = float(row.raw_dist)
            sim_score, dist_label, sim_label = compute_similarity_score(raw_dist, metric)
            meta = row.chunk_meta or {}
            page_num = meta.get("page_number") or meta.get("page")
            if page_num is not None:
                try:
                    page_num = int(page_num)
                except (ValueError, TypeError):
                    page_num = None

            results.append(
                SearchResultItem(
                    chunk_id=row.chunk_id or row.chunk_embedding_id,
                    document_id=row.resolved_doc_id or row.chunk_embedding_id,
                    document_name=row.doc_name,
                    mime_type=row.doc_mime,
                    sequence_index=row.sequence_index,
                    token_count=row.token_count,
                    char_count=row.char_count,
                    content=row.full_content,
                    metadata=meta,
                    page_number=page_num,
                    raw_distance=round(raw_dist, 6),
                    similarity_score=sim_score,
                    distance_metric=metric,
                    distance_label=dist_label,
                    similarity_label=sim_label,
                )
            )

        return results, plan_summary, latency_ms
