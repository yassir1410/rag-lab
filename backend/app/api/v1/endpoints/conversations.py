import uuid
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.rag.conversation.schemas import (
    ConversationCreateRequest,
    ConversationUpdateRequest,
    ConversationResponse,
    ConversationMessageResponse,
    TurnExecutionRequest,
    TurnExecutionResponse,
    MemorySelectionResult,
    QueryRewriteResult,
    StrategyComparisonResult,
    MultiTurnScenarioEvaluationResult,
)
from app.rag.conversation.service import conversational_service
from app.rag.conversation.orchestrator import conversational_orchestrator
from app.rag.conversation.memory import memory_registry
from app.rag.conversation.evaluation import conversational_benchmark_runner

router = APIRouter()


# ====================================================
# Global Endpoints (Defined before /{id} routes)
# ====================================================

@router.get("", response_model=Dict[str, Any])
def list_conversations(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    search: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """Lists persistent conversational sessions with search and pagination."""
    items, total = conversational_service.list_conversations(
        session=db, skip=skip, limit=limit, search=search, status=status
    )
    return {
        "items": items,
        "total": total,
        "skip": skip,
        "limit": limit,
    }


@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
def create_conversation(
    request: ConversationCreateRequest,
    db: Session = Depends(get_db),
):
    """Creates a new conversational RAG session with optional custom settings."""
    return conversational_service.create_conversation(
        session=db, title=request.title, settings=request.settings
    )


@router.get("/memory-strategies", response_model=List[Dict[str, Any]])
def list_memory_strategies():
    """Lists available conversational memory strategies and configuration parameters."""
    return memory_registry.list_strategies()


@router.post("/evaluation/benchmark", response_model=Dict[str, Any])
def run_conversational_benchmark(
    memory_strategy: str = Query("sliding_window"),
    retrieval_strategy: str = Query("hybrid_rrf"),
    llm_provider: str = Query("mock"),
    top_k: int = Query(5, ge=1, le=10),
    db: Session = Depends(get_db),
):
    """Executes the 12-scenario synthetic multi-turn conversational benchmark."""
    results = conversational_benchmark_runner.run_benchmark(
        session=db,
        memory_strategy=memory_strategy,
        retrieval_strategy=retrieval_strategy,
        llm_provider=llm_provider,
        top_k=top_k,
    )
    passed_count = sum(1 for r in results if r.passed)
    mean_overlap = sum(r.rewrite_keyword_overlap for r in results) / len(results) if results else 0.0
    mean_recall = sum(r.recall_at_5 for r in results) / len(results) if results else 0.0
    mean_mrr = sum(r.mrr for r in results) / len(results) if results else 0.0

    return {
        "memory_strategy": memory_strategy,
        "retrieval_strategy": retrieval_strategy,
        "total_scenarios": len(results),
        "passed_scenarios": passed_count,
        "pass_rate": round(passed_count / len(results), 4) if results else 0.0,
        "mean_rewrite_keyword_overlap": round(mean_overlap, 4),
        "mean_recall_at_5": round(mean_recall, 4),
        "mean_mrr": round(mean_mrr, 4),
        "results": results,
    }


# ====================================================
# Session-Specific Endpoints (/{id}/...)
# ====================================================

@router.get("/{conversation_id}", response_model=ConversationResponse)
def get_conversation(
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    """Fetches details and active settings for a conversational session."""
    conv = conversational_service.get_conversation(db, conversation_id)
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )
    return conv


@router.patch("/{conversation_id}", response_model=ConversationResponse)
def update_conversation(
    conversation_id: uuid.UUID,
    request: ConversationUpdateRequest,
    db: Session = Depends(get_db),
):
    """Updates session title, status, or settings."""
    conv = conversational_service.update_conversation(
        session=db,
        conversation_id=conversation_id,
        title=request.title,
        settings=request.settings,
        status=request.status,
    )
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )
    return conv


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    """Deletes a conversation and cascades cleanup of all messages and summaries."""
    success = conversational_service.delete_conversation(db, conversation_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )
    return None


@router.get("/{conversation_id}/messages", response_model=Dict[str, Any])
def list_messages(
    conversation_id: uuid.UUID,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Lists ordered dialogue messages for a conversation."""
    items, total = conversational_service.list_messages(
        session=db, conversation_id=conversation_id, skip=skip, limit=limit
    )
    return {
        "conversation_id": conversation_id,
        "items": items,
        "total": total,
        "skip": skip,
        "limit": limit,
    }


@router.post("/{conversation_id}/messages", response_model=TurnExecutionResponse)
def execute_turn(
    conversation_id: uuid.UUID,
    request: TurnExecutionRequest,
    db: Session = Depends(get_db),
):
    """Synchronously executes a conversational RAG turn and persists the exchange."""
    try:
        return conversational_orchestrator.execute_turn(
            session=db,
            conversation_id=conversation_id,
            request=request,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/{conversation_id}/stream")
def execute_turn_stream(
    conversation_id: uuid.UUID,
    request: TurnExecutionRequest,
    db: Session = Depends(get_db),
):
    """Executes a conversational RAG turn and streams real-time progression tokens via SSE."""
    # Ensure conversation exists before initiating stream
    conv = conversational_service.get_conversation(db, conversation_id)
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )

    event_generator = conversational_orchestrator.execute_turn_stream(
        session=db,
        conversation_id=conversation_id,
        request=request,
    )
    return StreamingResponse(
        event_generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{conversation_id}/memory", response_model=MemorySelectionResult)
def inspect_conversation_memory(
    conversation_id: uuid.UUID,
    strategy: Optional[str] = Query(None, description="Optional override to inspect different memory strategies"),
    db: Session = Depends(get_db),
):
    """Inspects how the active or selected memory strategy partitions conversation history."""
    try:
        return conversational_service.inspect_memory(
            session=db,
            conversation_id=conversation_id,
            strategy_override=strategy,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.post("/{conversation_id}/rewrite-preview", response_model=QueryRewriteResult)
def preview_query_rewrite(
    conversation_id: uuid.UUID,
    query: str = Query(..., min_length=1),
    db: Session = Depends(get_db),
):
    """Previews how a question would be rewritten into a standalone query without executing retrieval."""
    try:
        return conversational_service.preview_rewrite(
            session=db,
            conversation_id=conversation_id,
            query=query,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.post("/{conversation_id}/compare-strategies", response_model=List[StrategyComparisonResult])
def compare_memory_strategies(
    conversation_id: uuid.UUID,
    query: str = Query(..., min_length=1),
    db: Session = Depends(get_db),
):
    """Compares all 4 memory strategies on the same conversation history and query."""
    try:
        return conversational_service.compare_strategies(
            session=db,
            conversation_id=conversation_id,
            query=query,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
