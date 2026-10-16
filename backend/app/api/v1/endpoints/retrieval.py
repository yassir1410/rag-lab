import uuid
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.rag.retrieval.service import retrieval_service
from app.rag.retrieval.schemas import (
    SearchRequest,
    SearchResponse,
    IndexCreateRequest,
    IndexResponse,
    BenchmarkRequest,
    BenchmarkResponse,
    RetrievalExperimentCreate,
    RetrievalExperimentResponse,
    RetrievalStrategySpec,
    LexicalSearchRequest,
    HybridSearchRequest,
    RerankRequest,
    RerankResponse,
)
from app.rag.retrieval.hybrid import hybrid_engine
from app.rag.retrieval.reranker import reranker_registry

router = APIRouter()


@router.get(
    "/strategies",
    response_model=List[RetrievalStrategySpec],
    summary="List supported retrieval strategies",
    description="Returns metadata catalog for all 8 supported Phase 08 retrieval strategies.",
)
def list_strategies():
    return hybrid_engine.list_strategies()


@router.post(
    "/lexical",
    response_model=SearchResponse,
    summary="Execute full-text lexical search",
    description="Searches stored chunks using PostgreSQL tsvector GIN index and ts_rank_cd scoring.",
)
def search_lexical(
    request: LexicalSearchRequest,
    db: Session = Depends(get_db),
):
    try:
        search_req = SearchRequest(
            query=request.query,
            strategy="lexical_only",
            top_k=request.top_k,
            filters=request.filters,
            language=request.language,
        )
        return hybrid_engine.execute(db, search_req)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lexical search failed: {str(e)}",
        )


@router.post(
    "/hybrid",
    response_model=SearchResponse,
    summary="Execute hybrid retrieval (Dense + Lexical + RRF/Linear)",
    description="Executes parallel dense and lexical retrieval, fusing candidates via RRF or linear combination, with optional reranking.",
)
def search_hybrid(
    request: HybridSearchRequest,
    db: Session = Depends(get_db),
):
    try:
        strat = "hybrid_rrf" if request.fusion_method == "rrf" else "hybrid_linear"
        if request.reranker_model != "none":
            strat = "hybrid_rrf_rerank"
        search_req = SearchRequest(
            query=request.query,
            strategy=strat,
            model_id=request.model_id,
            search_algorithm=request.search_algorithm,
            distance_metric=request.distance_metric,
            top_k=request.top_k,
            hybrid_alpha=request.hybrid_alpha,
            rrf_k=request.rrf_k,
            reranker_model=request.reranker_model,
            filters=request.filters,
            language=request.language,
        )
        return hybrid_engine.execute(db, search_req)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hybrid search failed: {str(e)}",
        )


@router.post(
    "/rerank",
    response_model=RerankResponse,
    summary="Rerank candidate chunks using Cross-Encoder",
    description="Jointly attends query against candidate passages and rescores them.",
)
def rerank_candidates(request: RerankRequest):
    try:
        reranker = reranker_registry.get_reranker(request.reranker_model)
        results, lat, _ = reranker.rerank(
            query=request.query,
            candidates=request.candidates,
            top_k=request.top_k,
        )
        return RerankResponse(
            query=request.query,
            reranker_model=request.reranker_model,
            results=results,
            latency_ms=lat,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reranking failed: {str(e)}",
        )


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Execute vector similarity search",
    description="Encodes query text, searches stored document chunk embeddings via pgvector, and returns relevant chunks with source metadata and execution plan.",
)
def search_vectors(
    request: SearchRequest,
    db: Session = Depends(get_db),
):
    try:
        return retrieval_service.search(db, request)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector search failed: {str(e)}",
        )


@router.get(
    "/indexes",
    response_model=List[IndexResponse],
    summary="List managed pgvector indexes",
    description="Lists active HNSW and IVFFlat indexes in PostgreSQL with parameter configs, build duration, and physical disk size.",
)
def list_indexes(
    db: Session = Depends(get_db),
):
    try:
        return retrieval_service.list_indexes(db)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list indexes: {str(e)}",
        )


@router.post(
    "/indexes",
    response_model=IndexResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a pgvector index (HNSW or IVFFlat)",
    description="Constructs a native PostgreSQL pgvector index matching the selected distance metric and vector dimension.",
)
def create_index(
    request: IndexCreateRequest,
    db: Session = Depends(get_db),
):
    try:
        return retrieval_service.create_index(db, request)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Index creation failed: {str(e)}",
        )


@router.delete(
    "/indexes/{index_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Drop a pgvector index",
    description="Drops the index in PostgreSQL and deletes its metadata record.",
)
def drop_index(
    index_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    try:
        success = retrieval_service.drop_index(db, index_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Index with ID '{index_id}' not found.",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to drop index: {str(e)}",
        )


@router.post(
    "/benchmark",
    response_model=BenchmarkResponse,
    summary="Run comparative algorithm benchmark",
    description="Compares Exact KNN, HNSW, and IVFFlat side-by-side, measuring Recall@K relative to exact search, latency distributions, and throughput.",
)
def run_benchmark(
    request: BenchmarkRequest,
    db: Session = Depends(get_db),
):
    try:
        return retrieval_service.run_benchmark(db, request)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Benchmark execution failed: {str(e)}",
        )


@router.post(
    "/synthetic-dataset",
    summary="Generate synthetic benchmark dataset",
    description="Generates a reproducible corpus of semantic vector chunks (N=300+) to enable representative HNSW and IVFFlat benchmarking.",
)
def create_synthetic_dataset(
    model_id: str = Query(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model to encode synthetic chunks",
    ),
    count: int = Query(default=300, ge=50, le=1000, description="Number of synthetic chunks"),
    db: Session = Depends(get_db),
):
    try:
        return retrieval_service.generate_synthetic_dataset(db, model_id, count)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Synthetic corpus generation failed: {str(e)}",
        )


@router.get(
    "/experiments",
    response_model=List[RetrievalExperimentResponse],
    summary="List saved retrieval experiments",
)
def list_experiments(
    db: Session = Depends(get_db),
):
    try:
        return retrieval_service.list_experiments(db)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list experiments: {str(e)}",
        )


@router.post(
    "/experiments",
    response_model=RetrievalExperimentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Save a retrieval experiment run",
)
def save_experiment(
    request: RetrievalExperimentCreate,
    db: Session = Depends(get_db),
):
    try:
        return retrieval_service.save_experiment(db, request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save experiment: {str(e)}",
        )


@router.delete(
    "/experiments/{experiment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a saved retrieval experiment",
)
def delete_experiment(
    experiment_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    try:
        success = retrieval_service.delete_experiment(db, experiment_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Retrieval experiment with ID '{experiment_id}' not found.",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete experiment: {str(e)}",
        )
