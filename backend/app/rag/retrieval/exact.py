import time
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models.embeddings import ChunkEmbedding, EmbeddingExperiment
from app.models.chunking import DocumentChunk
from app.models.document import Document
from app.rag.embeddings.similarity import cosine_similarity, dot_product, euclidean_distance
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


class ExactVectorSearcher(BaseVectorSearcher):
    """
    Brute-force Exact K-Nearest Neighbors (KNN) searcher using native pgvector operators.
    Guarantees 100% recall (ground truth baseline for ANN comparison).
    Forces sequential scan in PostgreSQL via SET LOCAL enable_indexscan = off.
    """

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
        **kwargs,
    ) -> Tuple[List[SearchResultItem], Optional[ExecutionPlanSummary], float]:
        operator = METRIC_TO_OPERATOR.get(metric, "<=>")
        is_postgres = session.bind.dialect.name == "postgresql" if session.bind else False
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
        plan_summary: Optional[ExecutionPlanSummary] = None

        if is_postgres:
            try:
                session.execute(text("LOAD 'vector'"))
            except Exception:
                pass
            session.execute(text("SET LOCAL jit = off"))
            # Force sequential scan so exact KNN is never diverted to an index
            session.execute(text("SET LOCAL enable_indexscan = off"))
            session.execute(text("SET LOCAL enable_bitmapscan = off"))

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

            if explain_plan:
                try:
                    explain_query = f"EXPLAIN (ANALYZE, FORMAT JSON) {base_query}"
                    res_explain = session.execute(text(explain_query), params).scalar()
                    if isinstance(res_explain, str):
                        res_explain = json.loads(res_explain)
                    plan_summary = parse_execution_plan(res_explain)
                except Exception as e:
                    plan_summary = ExecutionPlanSummary(
                        node_type="Seq Scan",
                        planning_time_ms=0.0,
                        execution_time_ms=0.0,
                        actual_rows=0,
                        total_cost=0.0,
                        plan_tree_summary=f"EXPLAIN plan inspection: {e}",
                        raw_plan={},
                    )

            rows = session.execute(text(base_query), params).fetchall()
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            results: List[SearchResultItem] = []
            for row in rows:
                raw_dist = float(row.raw_dist)
                sim_score, dist_label, sim_label = compute_similarity_score(raw_dist, metric)

                # Extract page number if present in metadata
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

        else:
            # Dialect is SQLite (or other fallback in fast unit tests):
            # Evaluate exact distance via Python over candidate rows
            query = (
                session.query(ChunkEmbedding, EmbeddingExperiment, DocumentChunk, Document)
                .join(EmbeddingExperiment, ChunkEmbedding.experiment_id == EmbeddingExperiment.id)
                .outerjoin(DocumentChunk, ChunkEmbedding.chunk_id == DocumentChunk.id)
                .outerjoin(
                    Document,
                    (DocumentChunk.document_id == Document.id)
                    | (EmbeddingExperiment.document_id == Document.id),
                )
                .filter(EmbeddingExperiment.model_id == model_id)
                .filter(ChunkEmbedding.vector != None)
            )

            if filters:
                if filters.document_id:
                    query = query.filter(
                        (Document.id == filters.document_id)
                        | (EmbeddingExperiment.document_id == filters.document_id)
                    )
                if filters.mime_type:
                    query = query.filter(Document.mime_type == filters.mime_type)
                if filters.chunking_experiment_id:
                    query = query.filter(DocumentChunk.experiment_id == filters.chunking_experiment_id)

            candidate_rows = query.all()
            scored_candidates = []

            for ce, ee, dc, doc in candidate_rows:
                # ce.vector can be a list or vector in Python
                vec = ce.vector
                if hasattr(vec, "tolist"):
                    vec = vec.tolist()
                elif isinstance(vec, str):
                    import ast
                    vec = ast.literal_eval(vec)

                if not vec or len(vec) != dimension:
                    continue

                if filters and filters.page_number is not None:
                    meta = dc.metadata_json if dc and hasattr(dc, "metadata_json") else {}
                    p = meta.get("page_number") or meta.get("page")
                    try:
                        if p is None or int(p) != int(filters.page_number):
                            continue
                    except (ValueError, TypeError):
                        continue

                if metric == "cosine":
                    cos_sim = cosine_similarity(query_vector, vec)
                    raw_dist = 1.0 - cos_sim
                elif metric == "l2":
                    raw_dist = euclidean_distance(query_vector, vec)
                elif metric == "inner_product":
                    dp = dot_product(query_vector, vec)
                    raw_dist = -dp
                else:
                    raw_dist = 1.0 - cosine_similarity(query_vector, vec)

                scored_candidates.append((raw_dist, ce, ee, dc, doc))

            # Sort ascending by distance (top-N heapsort)
            scored_candidates.sort(key=lambda x: x[0])
            top_candidates = scored_candidates[:top_k]

            latency_ms = (time.perf_counter() - start_time) * 1000.0

            results = []
            for raw_dist, ce, ee, dc, doc in top_candidates:
                sim_score, dist_label, sim_label = compute_similarity_score(raw_dist, metric)
                meta = dc.metadata_json if dc and hasattr(dc, "metadata_json") else {}
                page_num = meta.get("page_number") or meta.get("page")
                if page_num is not None:
                    try:
                        page_num = int(page_num)
                    except (ValueError, TypeError):
                        page_num = None

                results.append(
                    SearchResultItem(
                        chunk_id=ce.chunk_id or ce.id,
                        document_id=doc.id if doc else (ee.document_id or ce.id),
                        document_name=doc.original_filename if doc else "Mock / SQLite Document",
                        mime_type=doc.mime_type if doc else "text/plain",
                        sequence_index=ce.sequence_index,
                        token_count=ce.token_count,
                        char_count=dc.char_count if dc else len(ce.text_snippet),
                        content=dc.content if dc else ce.text_snippet,
                        metadata=meta,
                        page_number=page_num,
                        raw_distance=round(raw_dist, 6),
                        similarity_score=sim_score,
                        distance_metric=metric,
                        distance_label=dist_label,
                        similarity_label=sim_label,
                    )
                )

            plan_summary = ExecutionPlanSummary(
                node_type="Seq Scan (SQLite Fallback)",
                planning_time_ms=0.01,
                execution_time_ms=round(latency_ms, 3),
                actual_rows=len(results),
                total_cost=round(float(len(candidate_rows)), 2),
                plan_tree_summary="Seq Scan | In-memory exact search",
                raw_plan={},
            )

            return results, plan_summary, latency_ms
