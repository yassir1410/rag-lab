import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status, Response
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.rag.chunking.registry import default_chunking_registry
from app.schemas.chunking import (
    StrategyInfoResponse,
    ChunkingPreviewRequest,
    ChunkingPreviewResponse,
    ChunkingExperimentCreate,
    ChunkingExperimentResponse,
    ChunkingExperimentListResponse,
    ChunkingCompareRequest,
    ChunkingCompareResponse,
    ChunkResponse,
)
from app.services.chunking_service import ChunkingService

router = APIRouter()
chunking_service = ChunkingService()


def _format_chunk_response(chunk) -> ChunkResponse:
    metadata_dict = chunk.metadata_json if hasattr(chunk, "metadata_json") else (
        chunk.metadata.__dict__ if hasattr(chunk.metadata, "__dict__") else {}
    )
    return ChunkResponse(
        id=chunk.id,
        document_id=chunk.document_id,
        sequence_index=chunk.sequence_index,
        content=chunk.content,
        char_count=chunk.char_count,
        token_count=chunk.token_count,
        parent_chunk_id=getattr(chunk, "parent_chunk_id", None) or (
            uuid.UUID(chunk.metadata.parent_chunk_id)
            if hasattr(chunk, "metadata") and getattr(chunk.metadata, "parent_chunk_id", None)
            else None
        ),
        metadata=metadata_dict,
        created_at=chunk.created_at,
    )


@router.get(
    "/strategies",
    response_model=List[StrategyInfoResponse],
    summary="List Chunking Strategies",
    description="Returns available chunking algorithms, descriptions, and default parameter schemas.",
)
def list_strategies() -> List[StrategyInfoResponse]:
    return [
        StrategyInfoResponse(**s) for s in default_chunking_registry.list_strategies()
    ]


@router.post(
    "/preview",
    response_model=ChunkingPreviewResponse,
    summary="Preview Document Chunking",
    description="Executes a chunking strategy on an ingested document in-memory without database persistence.",
)
def preview_chunking(
    request: ChunkingPreviewRequest,
    db: Session = Depends(get_db),
) -> ChunkingPreviewResponse:
    try:
        result = chunking_service.preview_chunking(
            db=db,
            document_id=request.document_id,
            strategy_name=request.strategy,
            config=request.config,
        )
        formatted_chunks = [_format_chunk_response(c) for c in result.chunks]
        return ChunkingPreviewResponse(
            strategy=result.strategy,
            chunks=formatted_chunks,
            metrics=result.metrics,
            config=result.config,
            processing_duration_ms=result.processing_duration_ms,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chunking preview failed: {str(e)}",
        )


@router.post(
    "/experiments",
    response_model=ChunkingExperimentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create & Persist Chunking Experiment",
    description="Executes a chunking algorithm and permanently persists the experiment and resulting chunks in PostgreSQL.",
)
def create_experiment(
    request: ChunkingExperimentCreate,
    db: Session = Depends(get_db),
) -> ChunkingExperimentResponse:
    try:
        experiment = chunking_service.create_and_persist_experiment(
            db=db,
            document_id=request.document_id,
            strategy_name=request.strategy,
            config=request.config,
        )
        formatted_chunks = [_format_chunk_response(c) for c in experiment.chunks]
        return ChunkingExperimentResponse(
            id=experiment.id,
            document_id=experiment.document_id,
            strategy=experiment.strategy,
            config=experiment.config,
            metrics=experiment.metrics,
            created_at=experiment.created_at,
            chunks_count=len(experiment.chunks),
            chunks=formatted_chunks,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to persist experiment: {str(e)}",
        )


@router.get(
    "/experiments",
    response_model=ChunkingExperimentListResponse,
    summary="List Saved Chunking Experiments",
)
def list_experiments(
    document_id: Optional[uuid.UUID] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> ChunkingExperimentListResponse:
    total, exps = chunking_service.list_experiments(
        db=db, document_id=document_id, limit=limit, offset=offset
    )
    items = [
        ChunkingExperimentResponse(
            id=e.id,
            document_id=e.document_id,
            strategy=e.strategy,
            config=e.config,
            metrics=e.metrics,
            created_at=e.created_at,
            chunks_count=len(e.chunks) if e.chunks else 0,
            chunks=None,  # omit chunks in list view for performance
        )
        for e in exps
    ]
    return ChunkingExperimentListResponse(total=total, items=items, limit=limit, offset=offset)


@router.get(
    "/experiments/{experiment_id}",
    response_model=ChunkingExperimentResponse,
    summary="Get Experiment Details",
)
def get_experiment(
    experiment_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> ChunkingExperimentResponse:
    exp = chunking_service.get_experiment(db=db, experiment_id=experiment_id)
    if not exp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment with id {experiment_id} not found",
        )
    formatted_chunks = [_format_chunk_response(c) for c in exp.chunks]
    return ChunkingExperimentResponse(
        id=exp.id,
        document_id=exp.document_id,
        strategy=exp.strategy,
        config=exp.config,
        metrics=exp.metrics,
        created_at=exp.created_at,
        chunks_count=len(exp.chunks),
        chunks=formatted_chunks,
    )


@router.delete(
    "/experiments/{experiment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Chunking Experiment",
)
def delete_experiment(
    experiment_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    deleted = chunking_service.delete_experiment(db=db, experiment_id=experiment_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment with id {experiment_id} not found",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/compare",
    response_model=ChunkingCompareResponse,
    summary="Compare Multiple Chunking Strategies",
    description="Applies multiple chunking strategy configurations to the same source document simultaneously.",
)
def compare_strategies(
    request: ChunkingCompareRequest,
    db: Session = Depends(get_db),
) -> ChunkingCompareResponse:
    try:
        raw_results = chunking_service.compare_strategies(
            db=db,
            document_id=request.document_id,
            strategies_config=[s.model_dump() for s in request.strategies],
        )
        formatted_results = [
            ChunkingPreviewResponse(
                strategy=r.strategy,
                chunks=[_format_chunk_response(c) for c in r.chunks],
                metrics=r.metrics,
                config=r.config,
                processing_duration_ms=r.processing_duration_ms,
            )
            for r in raw_results
        ]
        return ChunkingCompareResponse(
            document_id=request.document_id,
            results=formatted_results,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Strategy comparison failed: {str(e)}",
        )
