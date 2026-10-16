import json
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.rag.generation.registry import llm_registry
from app.rag.generation.prompt_builder import prompt_builder
from app.rag.generation.orchestrator import rag_orchestrator
from app.rag.generation.schemas import (
    RAGQueryRequest,
    RAGQueryResponse,
    ProviderInfoResponse,
    ModelInfoResponse,
    PromptTemplateResponse,
)

router = APIRouter()


@router.get(
    "/providers",
    response_model=List[ProviderInfoResponse],
    summary="List registered LLM providers with availability status",
)
def list_providers():
    """Returns all registered LLM providers (Ollama, OpenAI, Gemini, Mock) with status indicators."""
    return llm_registry.list_providers()


@router.get(
    "/models",
    response_model=List[ModelInfoResponse],
    summary="List available LLM models across providers",
)
def list_models():
    """Returns available models across all registered LLM providers."""
    return llm_registry.list_all_models()


@router.get(
    "/prompts",
    response_model=List[PromptTemplateResponse],
    summary="List available RAG prompt templates",
)
def list_prompts():
    """Returns available prompt engineering templates: basic, grounded, concise, detailed."""
    return prompt_builder.list_templates()


@router.post(
    "/query",
    response_model=RAGQueryResponse,
    summary="Execute end-to-end RAG query pipeline synchronously",
)
def query_rag(
    request: RAGQueryRequest,
    db: Session = Depends(get_db),
):
    """
    Executes the complete End-to-End RAG pipeline:
    Query -> Embedding -> Vector Search -> Context Assembly -> Prompt Construction -> LLM Inference -> Citation Validation.
    """
    try:
        return rag_orchestrator.execute_query(session=db, request=request)
    except ConnectionError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except TimeoutError as e:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(e),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG pipeline execution failed: {str(e)}",
        )


@router.post(
    "/query/stream",
    summary="Execute RAG query with Server-Sent Events (SSE) token streaming",
)
async def query_rag_stream(
    request: RAGQueryRequest,
    db: Session = Depends(get_db),
):
    """
    Streams RAG response tokens and intermediate pipeline stages via Server-Sent Events (SSE).
    Events emitted:
    - 'status': Current pipeline phase
    - 'sources': Retrieved context passages with citation IDs
    - 'chunk': Progressive text tokens from the LLM
    - 'done': Final trace metrics, citation validation, and warnings
    - 'error': Error message on failure
    """
    async def sse_event_stream():
        try:
            async for event_dict in rag_orchestrator.execute_query_stream(session=db, request=request):
                event_name = event_dict.get("event", "message")
                raw_data = event_dict.get("data", {})
                json_safe_data = jsonable_encoder(raw_data)
                event_data = json.dumps(json_safe_data, default=str)
                yield f"event: {event_name}\ndata: {event_data}\n\n"
        except Exception as e:
            err_data = json.dumps({"error": str(e)}, default=str)
            yield f"event: error\ndata: {err_data}\n\n"

    return StreamingResponse(
        sse_event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/traces/{trace_id}",
    response_model=RAGQueryResponse,
    summary="Retrieve an in-memory RAG pipeline execution trace",
)
def get_trace(trace_id: str):
    """Fetches an in-memory trace containing complete pipeline metrics, prompts, and citations."""
    trace = rag_orchestrator.get_trace(trace_id)
    if not trace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trace '{trace_id}' not found or expired from cache.",
        )
    return trace
