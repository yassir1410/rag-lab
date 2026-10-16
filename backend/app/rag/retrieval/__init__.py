"""
Retrieval Module (Phase 05: Vector Database & Semantic Search)
Implements:
- Exact KNN brute-force vector search via pgvector (<=>, <->, <#>)
- HNSW approximate nearest neighbor search
- IVFFlat approximate nearest neighbor search
- Query embedding with asymmetric prefix support (e.g. E5)
- Parameterized metadata filtering
- Dynamic pgvector index lifecycle management
- Comparative benchmarking (Recall@K, latency distributions, execution plan analysis)
"""

from app.rag.retrieval.schemas import (
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    ExecutionPlanSummary,
    IndexCreateRequest,
    IndexResponse,
    BenchmarkRequest,
    BenchmarkResponse,
    RetrievalExperimentCreate,
    RetrievalExperimentResponse,
)
from app.rag.retrieval.base import (
    BaseVectorSearcher,
    compute_similarity_score,
    parse_execution_plan,
)
from app.rag.retrieval.exact import ExactVectorSearcher
from app.rag.retrieval.hnsw import HNSWVectorSearcher
from app.rag.retrieval.ivfflat import IVFFlatVectorSearcher
from app.rag.retrieval.index_manager import index_manager, IndexManager
from app.rag.retrieval.benchmark import retrieval_benchmarker, RetrievalBenchmarker
from app.rag.retrieval.service import retrieval_service, RetrievalService

__all__ = [
    "SearchRequest",
    "SearchResponse",
    "SearchResultItem",
    "ExecutionPlanSummary",
    "IndexCreateRequest",
    "IndexResponse",
    "BenchmarkRequest",
    "BenchmarkResponse",
    "RetrievalExperimentCreate",
    "RetrievalExperimentResponse",
    "BaseVectorSearcher",
    "compute_similarity_score",
    "parse_execution_plan",
    "ExactVectorSearcher",
    "HNSWVectorSearcher",
    "IVFFlatVectorSearcher",
    "index_manager",
    "IndexManager",
    "retrieval_benchmarker",
    "RetrievalBenchmarker",
    "retrieval_service",
    "RetrievalService",
]
