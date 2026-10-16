import math
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from app.rag.retrieval.schemas import (
    SearchResultItem,
    ExecutionPlanSummary,
    RetrievalFilter,
)

METRIC_TO_OPERATOR = {
    "cosine": "<=>",
    "l2": "<->",
    "inner_product": "<#>",
}

METRIC_TO_OPCLASS = {
    "cosine": "vector_cosine_ops",
    "l2": "vector_l2_ops",
    "inner_product": "vector_ip_ops",
}


def compute_similarity_score(
    raw_distance: float,
    metric: str,
    is_normalized: bool = True,
) -> Tuple[float, str, str]:
    """
    Computes a mathematically sound similarity score from raw pgvector distance
    and returns appropriate human-readable labels.
    """
    if metric == "cosine":
        # Cosine distance in pgvector is 1 - cos(theta)
        # Cosine similarity is 1 - d
        sim = 1.0 - raw_distance
        # Clamp to [-1.0, 1.0] for precision safety
        sim = max(-1.0, min(1.0, sim))
        dist_label = "Cosine Distance (1 - cos θ)"
        sim_label = "Cosine Similarity (cos θ)"
        return round(sim, 4), dist_label, sim_label

    elif metric == "l2":
        # Euclidean distance in [0, inf)
        # Inverted harmonic similarity: 1 / (1 + d) mapped to (0, 1]
        sim = 1.0 / (1.0 + raw_distance) if raw_distance >= 0 else 0.0
        dist_label = "L2 Distance (||u - v||₂)"
        sim_label = "Harmonic L2 Similarity 1/(1+d)"
        return round(sim, 4), dist_label, sim_label

    elif metric == "inner_product":
        # pgvector <#> returns negative inner product -(u · v)
        # so that ORDER BY gives largest dot product first
        dot_prod = -raw_distance
        dist_label = "Negative Dot Product -(u · v)"
        sim_label = "Inner Product (u · v)"
        return round(dot_prod, 4), dist_label, sim_label

    else:
        return round(1.0 - raw_distance, 4), f"{metric} Distance", f"{metric} Similarity"


def extract_plan_node_type(plan_node: Dict[str, Any]) -> str:
    """Recursively checks for index scan vs seq scan in PostgreSQL execution plan."""
    node_type = plan_node.get("Node Type", "Unknown")
    if "Index Scan" in node_type:
        index_name = plan_node.get("Index Name", "")
        return f"Index Scan ({index_name})" if index_name else "Index Scan"
    if "Bitmap Index Scan" in node_type:
        index_name = plan_node.get("Index Name", "")
        return f"Bitmap Index Scan ({index_name})" if index_name else "Bitmap Index Scan"

    # Check child plans
    plans = plan_node.get("Plans", [])
    for sub in plans:
        sub_type = extract_plan_node_type(sub)
        if "Index Scan" in sub_type or "Bitmap" in sub_type:
            return sub_type

    return node_type


def parse_execution_plan(plan_result: Any) -> ExecutionPlanSummary:
    """Parses EXPLAIN (ANALYZE, FORMAT JSON) output from PostgreSQL."""
    if not plan_result or not isinstance(plan_result, list) or len(plan_result) == 0:
        return ExecutionPlanSummary(
            node_type="Unknown",
            planning_time_ms=0.0,
            execution_time_ms=0.0,
            actual_rows=0,
            total_cost=0.0,
            plan_tree_summary="No plan data available",
            raw_plan={},
        )

    root = plan_result[0]
    plan_node = root.get("Plan", {})
    planning_time = float(root.get("Planning Time", 0.0))
    execution_time = float(root.get("Execution Time", 0.0))
    actual_rows = int(plan_node.get("Actual Rows", 0))
    total_cost = float(plan_node.get("Total Cost", 0.0))

    primary_node = extract_plan_node_type(plan_node)
    summary_text = (
        f"{primary_node} | Cost: {total_cost:.2f} | Rows: {actual_rows} | "
        f"Exec: {execution_time:.3f}ms | Plan: {planning_time:.3f}ms"
    )

    return ExecutionPlanSummary(
        node_type=primary_node,
        planning_time_ms=round(planning_time, 3),
        execution_time_ms=round(execution_time, 3),
        actual_rows=actual_rows,
        total_cost=round(total_cost, 2),
        plan_tree_summary=summary_text,
        raw_plan=plan_node,
    )


class BaseVectorSearcher(ABC):
    """
    Abstract interface for vector retrieval strategies (Exact KNN, HNSW, IVFFlat).
    """

    @abstractmethod
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
        """
        Executes vector search and returns:
        (results, execution_plan_summary, query_latency_ms)
        """
        pass
