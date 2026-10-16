import uuid
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.graph import GraphEntity, GraphRelation, GraphExtractionRun
from app.rag.graph.schemas import (
    EntityResponse,
    RelationResponse,
    ExtractionRunRequest,
    ExtractionRunResponse,
    GraphTraversalRequest,
    GraphTraversalResponse,
    GraphPathsRequest,
    GraphPathsResponse,
    GraphQueryRequest,
    GraphQueryResponse,
    GraphCompareRequest,
    GraphCompareResponse,
    GraphStatsResponse,
    CommunitySummary,
    EntityMergeCandidate,
    EntityMergeRequest,
    GraphRetrievalStrategy,
)
from app.rag.graph.service import graph_rag_service
from app.rag.graph.repository import postgres_graph_repository
from app.rag.graph.construction.graph_builder import knowledge_graph_builder
from app.rag.graph.analytics.community_detection import community_detector
from app.rag.graph.resolution.entity_resolver import entity_resolver
from app.rag.graph.evaluation import graph_benchmark_runner, GraphBenchmarkReport

router = APIRouter()


@router.get("/stats", response_model=GraphStatsResponse)
def get_graph_stats(db: Session = Depends(get_db)):
    """
    Returns global graph analytics: node/edge counts, entity/relation types, density, and degree.
    """
    return graph_rag_service.get_stats(db)


@router.get("/entities", response_model=List[EntityResponse])
def list_graph_entities(
    entity_type: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    """
    Returns paginated knowledge graph entities with optional type and name search filters.
    """
    entities = postgres_graph_repository.list_entities(
        db, entity_type=entity_type, search=search, limit=limit, offset=offset
    )
    return [
        EntityResponse(
            id=e.id,
            canonical_name=e.canonical_name,
            entity_type=e.entity_type,
            aliases=e.aliases or [],
            properties=e.properties or {},
            extraction_method=e.extraction_method,
            created_at=e.created_at.isoformat() if e.created_at else "",
            updated_at=e.updated_at.isoformat() if e.updated_at else "",
            evidence_count=len(e.evidence_items or []),
            degree=len(e.outgoing_relations or []) + len(e.incoming_relations or []),
        )
        for e in entities
    ]


@router.get("/entities/{entity_id}", response_model=EntityResponse)
def get_graph_entity(
    entity_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    """
    Retrieves a single entity by its UUID.
    """
    entity = postgres_graph_repository.get_entity(db, entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found.")
    return EntityResponse(
        id=entity.id,
        canonical_name=entity.canonical_name,
        entity_type=entity.entity_type,
        aliases=entity.aliases or [],
        properties=entity.properties or {},
        extraction_method=entity.extraction_method,
        created_at=entity.created_at.isoformat() if entity.created_at else "",
        updated_at=entity.updated_at.isoformat() if entity.updated_at else "",
        evidence_count=len(entity.evidence_items or []),
        degree=len(entity.outgoing_relations or []) + len(entity.incoming_relations or []),
    )


@router.get("/relations", response_model=List[RelationResponse])
def list_graph_relations(
    entity_id: Optional[uuid.UUID] = None,
    relation_type: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    """
    Returns paginated graph relations with optional entity and type filters.
    """
    relations = postgres_graph_repository.list_relations(
        db, entity_id=entity_id, relation_type=relation_type, limit=limit, offset=offset
    )
    return [
        RelationResponse(
            id=r.id,
            source_entity_id=r.source_entity_id,
            target_entity_id=r.target_entity_id,
            relation_type=r.relation_type,
            properties=r.properties or {},
            extraction_method=r.extraction_method,
            confidence=r.confidence,
            source_name=r.source_entity.canonical_name if r.source_entity else None,
            target_name=r.target_entity.canonical_name if r.target_entity else None,
            created_at=r.created_at.isoformat() if r.created_at else "",
            evidence_count=len(r.evidence_items or []),
        )
        for r in relations
    ]


@router.post("/extract", response_model=ExtractionRunResponse)
def run_graph_extraction(
    request: ExtractionRunRequest,
    db: Session = Depends(get_db),
):
    """
    Triggers entity and relation extraction on an ingested document,
    resolves candidates, constructs the knowledge graph, and attaches provenance.
    """
    try:
        run = knowledge_graph_builder.build_from_document(
            session=db,
            document_id=request.document_id,
            strategy=request.strategy,
            provider_id=request.provider_id,
            model_id=request.model_id,
            extraction_config=request.extraction_config,
        )
        return ExtractionRunResponse(
            id=run.id,
            document_id=run.document_id,
            model_id=run.model_id,
            status=run.status,
            metrics=run.metrics or {},
            created_at=run.created_at.isoformat() if run.created_at else "",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Graph extraction failed: {str(exc)}")


@router.get("/extraction-runs", response_model=List[ExtractionRunResponse])
def list_extraction_runs(
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """
    Returns past knowledge graph extraction runs.
    """
    runs = db.query(GraphExtractionRun).order_by(GraphExtractionRun.created_at.desc()).limit(limit).all()
    return [
        ExtractionRunResponse(
            id=r.id,
            document_id=r.document_id,
            model_id=r.model_id,
            status=r.status,
            metrics=r.metrics or {},
            created_at=r.created_at.isoformat() if r.created_at else "",
        )
        for r in runs
    ]


@router.post("/traverse", response_model=GraphTraversalResponse)
def traverse_graph(
    request: GraphTraversalRequest,
    db: Session = Depends(get_db),
):
    """
    Performs bounded graph traversal starting from an entity up to N hops.
    """
    return graph_rag_service.traverse(db, request)


@router.post("/paths", response_model=GraphPathsResponse)
def find_graph_paths(
    request: GraphPathsRequest,
    db: Session = Depends(get_db),
):
    """
    Finds connecting paths between two entities within max_depth hops.
    """
    return graph_rag_service.find_paths(db, request)


@router.post("/query", response_model=GraphQueryResponse)
def execute_graph_query(
    request: GraphQueryRequest,
    db: Session = Depends(get_db),
):
    """
    Executes a Graph RAG query (entity, traversal, hybrid, or community-based)
    and generates a cited response grounded in documentary evidence.
    """
    try:
        return graph_rag_service.query(db, request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Graph query failed: {str(exc)}")


@router.post("/compare", response_model=GraphCompareResponse)
def compare_graph_strategies(
    request: GraphCompareRequest,
    db: Session = Depends(get_db),
):
    """
    Compares Entity Graph, Traversal, Hybrid Graph+Vector, and Community-based retrieval.
    """
    try:
        return graph_rag_service.compare_strategies(db, request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Strategy comparison failed: {str(exc)}")


@router.get("/communities", response_model=List[CommunitySummary])
def get_graph_communities(db: Session = Depends(get_db)):
    """
    Returns detected communities with member entities and thematic summaries.
    """
    return community_detector.detect_communities(db)


@router.get("/merge-candidates", response_model=List[EntityMergeCandidate])
def get_merge_candidates(db: Session = Depends(get_db)):
    """
    Identifies duplicate entity candidates and aliases for review.
    """
    return entity_resolver.find_merge_candidates(db)


@router.post("/merge", response_model=EntityResponse)
def merge_entities(
    request: EntityMergeRequest,
    db: Session = Depends(get_db),
):
    """
    Merges a duplicate entity into a surviving entity, rerouting relations and evidence.
    """
    try:
        survivor = postgres_graph_repository.merge_entities(
            db, surviving_id=request.surviving_entity_id, merged_id=request.merged_entity_id
        )
        db.commit()
        return EntityResponse(
            id=survivor.id,
            canonical_name=survivor.canonical_name,
            entity_type=survivor.entity_type,
            aliases=survivor.aliases or [],
            properties=survivor.properties or {},
            extraction_method=survivor.extraction_method,
            created_at=survivor.created_at.isoformat() if survivor.created_at else "",
            updated_at=survivor.updated_at.isoformat() if survivor.updated_at else "",
            evidence_count=len(survivor.evidence_items or []),
            degree=len(survivor.outgoing_relations or []) + len(survivor.incoming_relations or []),
        )
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Entity merge failed: {str(exc)}")


@router.post("/benchmark", response_model=GraphBenchmarkReport)
def run_graph_benchmark(
    strategy: GraphRetrievalStrategy = Query(default=GraphRetrievalStrategy.HYBRID_GRAPH_VECTOR),
    provider_id: str = Query(default="mock"),
    model_id: str = Query(default="mock-instruct-v1"),
    db: Session = Depends(get_db),
):
    """
    Executes the 15-scenario synthetic evaluation benchmark for Graph RAG.
    """
    try:
        return graph_benchmark_runner.run_benchmark(
            session=db,
            strategy=strategy,
            provider_id=provider_id,
            model_id=model_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Graph benchmark failed: {str(exc)}")
