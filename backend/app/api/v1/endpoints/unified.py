import json
import time
import uuid
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.rag.unified.schemas import (
    DocumentCapabilityReport,
    UnifiedRAGQueryRequest,
    UnifiedRAGQueryResponse,
    RouteComparisonResponse,
    RouteComparisonItem,
)
from app.rag.unified.capability_service import document_capability_service
from app.rag.unified.generation_service import unified_generation_service
from app.rag.unified.router import intelligent_router
from app.rag.multimodal.generation.vision_provider import VisionProviderError

router = APIRouter()

# In-memory LRU trace store
_unified_traces: Dict[str, UnifiedRAGQueryResponse] = {}


@router.get("/routes", summary="List available Unified RAG retrieval routes")
def list_unified_routes():
    """Returns all available retrieval engines and their index prerequisites."""
    return [
        {
            "route_id": "text_dense",
            "name": "Text Dense Search",
            "description": "High-dimensional pgvector cosine similarity search over chunked text passages.",
            "modality": "text",
            "required_indexes": ["text_chunks", "vector_embeddings"],
        },
        {
            "route_id": "text_hybrid",
            "name": "Text Hybrid Search (RRF)",
            "description": "Reciprocal Rank Fusion blending dense vector similarity with PostgreSQL full-text search and TinyBERT reranking.",
            "modality": "text",
            "required_indexes": ["text_chunks", "vector_embeddings", "tsvector"],
        },
        {
            "route_id": "multimodal_table",
            "name": "Multimodal Table Intelligence",
            "description": "Structured matrix retrieval preserving columns, rows, and Markdown representation.",
            "modality": "tabular",
            "required_indexes": ["table_elements"],
        },
        {
            "route_id": "multimodal_visual",
            "name": "Multimodal Visual & CLIP Search",
            "description": "Cross-modal text-to-image semantic matching and whole page visual asset retrieval.",
            "modality": "visual",
            "required_indexes": ["page_assets", "clip_embeddings"],
        },
        {
            "route_id": "graph",
            "name": "Knowledge Graph Traversal",
            "description": "Entity resolution, multi-hop relation traversal, and documentary graph evidence extraction.",
            "modality": "graph",
            "required_indexes": ["graph_entities", "graph_evidence"],
        },
        {
            "route_id": "agentic",
            "name": "Agentic Multi-Step Orchestration",
            "description": "Constrained Plan-Act-Observe loops with approved read-only retrieval tools.",
            "modality": "multi_step",
            "required_indexes": ["text_chunks", "vector_embeddings"],
        },
    ]


@router.get("/capabilities", response_model=DocumentCapabilityReport, summary="Inspect document indexing capabilities")
def get_capabilities(
    document_id: Optional[uuid.UUID] = Query(None, description="Optional document UUID filter"),
    db: Session = Depends(get_db),
):
    """
    Audits the current indexing status across Chunks, Embeddings, Tables, Visuals, and Knowledge Graph.
    Identifies which retrieval routes can be safely executed without data starvation.
    """
    return document_capability_service.get_document_capability(session=db, document_id=document_id)


@router.post("/query", response_model=UnifiedRAGQueryResponse, summary="Execute Unified RAG query synchronously")
def query_unified_rag(
    request: UnifiedRAGQueryRequest,
    db: Session = Depends(get_db),
):
    """
    Executes the complete Unified RAG pipeline:
    Query -> Intelligent Routing -> Multi-Route Retrieval -> Reciprocal Rank Fusion -> Context Assembly -> Text/Vision Generation -> Citation Validation.
    """
    try:
        response = unified_generation_service.execute_query(session=db, request=request)
        if len(_unified_traces) >= 50:
            _unified_traces.pop(next(iter(_unified_traces)))
        _unified_traces[response.trace_id] = response
        return response
    except VisionProviderError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except ValueError as e:
        if "timed out" in str(e).lower():
            raise HTTPException(status_code=504, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unified RAG query execution failed: {str(e)}")


@router.post("/query/stream", summary="Stream Unified RAG query with SSE tokens and routing updates")
async def query_unified_rag_stream(
    request: UnifiedRAGQueryRequest,
    db: Session = Depends(get_db),
):
    """
    Streams Unified RAG stages via Server-Sent Events (SSE):
    - 'status': Current pipeline phase
    - 'routing': Explainable routing determination
    - 'sources': Retrieved evidence items with [S#] tags
    - 'chunk': Progressive text tokens from the generator
    - 'done': Final trace metrics and validated citations
    """
    async def sse_stream():
        try:
            async for event in unified_generation_service.execute_query_stream(session=db, request=request):
                event_name = event.get("event", "message")
                raw_data = event.get("data", {})
                json_data = json.dumps(jsonable_encoder(raw_data), default=str)
                yield f"event: {event_name}\ndata: {json_data}\n\n"
        except Exception as e:
            err = json.dumps({"error": str(e)}, default=str)
            yield f"event: error\ndata: {err}\n\n"

    return StreamingResponse(
        sse_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/compare", response_model=RouteComparisonResponse, summary="Compare multiple retrieval routes side-by-side")
def compare_unified_routes(
    question: str = Query(..., min_length=2),
    document_id: Optional[uuid.UUID] = Query(None),
    db: Session = Depends(get_db),
):
    """Runs candidate routes side-by-side to evaluate retrieval quality and latency differences."""
    t0 = time.perf_counter()
    decision = intelligent_router.route_query(
        session=db, question=question, document_id=document_id, mode="comparative"
    )

    comparisons: List[RouteComparisonItem] = []
    for route in decision.selected_routes:
        t_route = time.perf_counter()
        evidence, _ = unified_generation_service.retriever.retrieve(
            session=db, query=question, routes=[route], document_id=document_id, top_k=3
        )
        elapsed = round((time.perf_counter() - t_route) * 1000.0, 2)
        top_score = evidence[0].calibrated_score if evidence else 0.0
        top_cit = evidence[0].origin_tag if evidence else "None"
        snippet = evidence[0].content[:150] + "..." if evidence else "No evidence found"

        comparisons.append(
            RouteComparisonItem(
                route=route,
                latency_ms=elapsed,
                evidence_count=len(evidence),
                top_score=top_score,
                top_citation=top_cit,
                snippet=snippet,
            )
        )

    total_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    rec = f"Recommended route: '{decision.selected_routes[0]}' based on detected intent: {', '.join(decision.intent_detected)}."

    return RouteComparisonResponse(
        question=question,
        document_id=document_id,
        comparisons=comparisons,
        routing_recommendation=rec,
        total_comparison_latency_ms=total_ms,
    )


@router.get("/traces/{trace_id}", response_model=UnifiedRAGQueryResponse, summary="Retrieve a Unified RAG execution trace")
def get_trace(trace_id: str):
    trace = _unified_traces.get(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace '{trace_id}' not found or expired.")
    return trace
