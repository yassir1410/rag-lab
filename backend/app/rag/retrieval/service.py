import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy import desc, text
from sqlalchemy.orm import Session

from app.models.retrieval import RetrievalIndex, RetrievalExperiment
from app.rag.embeddings.registry import embedding_registry
from app.rag.retrieval.schemas import (
    SearchRequest,
    SearchResponse,
    IndexCreateRequest,
    IndexResponse,
    BenchmarkRequest,
    BenchmarkResponse,
    RetrievalExperimentCreate,
    RetrievalExperimentResponse,
)
from app.rag.retrieval.exact import ExactVectorSearcher
from app.rag.retrieval.hnsw import HNSWVectorSearcher
from app.rag.retrieval.ivfflat import IVFFlatVectorSearcher
from app.rag.retrieval.index_manager import index_manager
from app.rag.retrieval.benchmark import retrieval_benchmarker
from app.rag.retrieval.synthetic import generate_synthetic_benchmark_corpus
from app.rag.retrieval.hybrid import hybrid_engine


class RetrievalService:
    """
    Orchestration service for Vector Search Lab:
    - Query embedding with asymmetric prefix support (e.g. E5)
    - Vector search execution (Exact KNN, HNSW, IVFFlat)
    - Hybrid & Lexical search (Phase 08)
    - Dynamic index management (creation, size inspection, drop)
    - Comparative algorithm benchmarks
    - Experiment persistence
    """

    def __init__(self):
        self.registry = embedding_registry
        self.exact_searcher = ExactVectorSearcher()
        self.hnsw_searcher = HNSWVectorSearcher()
        self.ivfflat_searcher = IVFFlatVectorSearcher()
        self.index_manager = index_manager
        self.benchmarker = retrieval_benchmarker
        self.hybrid_engine = hybrid_engine

    def search(self, session: Session, request: SearchRequest) -> SearchResponse:
        return self.hybrid_engine.execute(session, request)

    def create_index(self, session: Session, request: IndexCreateRequest) -> IndexResponse:
        spec = self.registry.get_spec(request.model_id)
        return self.index_manager.create_index(session, request, spec.dimension)

    def list_indexes(self, session: Session) -> List[IndexResponse]:
        return self.index_manager.list_indexes(session)

    def drop_index(self, session: Session, index_id: uuid.UUID) -> bool:
        return self.index_manager.drop_index(session, index_id)

    def run_benchmark(self, session: Session, request: BenchmarkRequest) -> BenchmarkResponse:
        spec = self.registry.get_spec(request.model_id)
        provider = self.registry.get_provider(request.model_id)

        encode_res = provider.encode(
            [request.query],
            normalize_embeddings=True,
            is_query=True,
        )
        if not encode_res.vectors:
            raise ValueError("Failed to encode benchmark query.")

        return self.benchmarker.run_benchmark(
            session=session,
            query_vector=encode_res.vectors[0],
            dimension=spec.dimension,
            request=request,
            embedding_latency_ms=encode_res.latency_ms,
        )

    def generate_synthetic_dataset(
        self, session: Session, model_id: str, count: int = 300
    ) -> Dict[str, Any]:
        return generate_synthetic_benchmark_corpus(session, model_id, count)

    def list_experiments(self, session: Session) -> List[RetrievalExperimentResponse]:
        exps = session.query(RetrievalExperiment).order_by(desc(RetrievalExperiment.created_at)).all()
        return [
            RetrievalExperimentResponse(
                id=e.id,
                name=e.name,
                query_text=e.query_text,
                model_id=e.model_id,
                distance_metric=e.distance_metric,
                top_k=e.top_k,
                filters=e.filters or {},
                results=e.results or [],
                metrics=e.metrics or {},
                created_at=e.created_at.isoformat(),
            )
            for e in exps
        ]

    def save_experiment(
        self, session: Session, request: RetrievalExperimentCreate
    ) -> RetrievalExperimentResponse:
        exp = RetrievalExperiment(
            name=request.name,
            query_text=request.query_text,
            model_id=request.model_id,
            distance_metric=request.distance_metric,
            top_k=request.top_k,
            filters=request.filters or {},
            results=request.results or [],
            metrics=request.metrics or {},
        )
        session.add(exp)
        session.commit()
        session.refresh(exp)

        return RetrievalExperimentResponse(
            id=exp.id,
            name=exp.name,
            query_text=exp.query_text,
            model_id=exp.model_id,
            distance_metric=exp.distance_metric,
            top_k=exp.top_k,
            filters=exp.filters or {},
            results=exp.results or [],
            metrics=exp.metrics or {},
            created_at=exp.created_at.isoformat(),
        )

    def delete_experiment(self, session: Session, experiment_id: uuid.UUID) -> bool:
        exp = session.query(RetrievalExperiment).filter(RetrievalExperiment.id == experiment_id).first()
        if not exp:
            return False
        session.delete(exp)
        session.commit()
        return True


retrieval_service = RetrievalService()
