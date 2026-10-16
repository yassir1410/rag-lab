import time
import uuid
from typing import Dict, Any, Optional, Type
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.rag.agents.tools.base import BaseAgentTool
from app.rag.agents.schemas import ToolResult
from app.rag.retrieval.hybrid import HybridRetrievalEngine
from app.rag.retrieval.schemas import SearchRequest, RetrievalFilter


class HybridSearchArgs(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000, description="Search query string")
    strategy: str = Field(
        default="hybrid_rrf",
        description="Hybrid fusion strategy: 'hybrid_rrf', 'hybrid_linear', 'hybrid_rrf_rerank', or 'dense_rerank'"
    )
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata filters, e.g. {'document_id': '...'}")
    top_k: int = Field(default=5, ge=1, le=15, description="Number of top fused chunks to retrieve")


class HybridSearchTool(BaseAgentTool):
    """
    Approved Tool: Hybrid Dense + Lexical Fusion Search.
    Fuses PostgreSQL full-text search (tsvector) with pgvector embeddings,
    optionally applying TinyBERT cross-encoder reranking.
    """

    def __init__(self, engine: Optional[HybridRetrievalEngine] = None):
        self.engine = engine or HybridRetrievalEngine()

    @property
    def name(self) -> str:
        return "hybrid_search"

    @property
    def description(self) -> str:
        return (
            "Performs hybrid search combining dense semantic similarity and PostgreSQL full-text keyword matching "
            "using Reciprocal Rank Fusion (RRF) or Linear combination. Recommended for technical queries with exact terms."
        )

    @property
    def args_schema(self) -> Type[BaseModel]:
        return HybridSearchArgs

    def execute(self, session: Session, **kwargs) -> ToolResult:
        t0 = time.perf_counter()
        call_id = kwargs.pop("call_id", str(uuid.uuid4())[:8])

        try:
            validated = self.args_schema(**kwargs)
            query = validated.query
            strategy = validated.strategy
            top_k = validated.top_k
            raw_filters = validated.filters or {}

            # Construct RetrievalFilter safely
            retrieval_filter = None
            if raw_filters:
                doc_id = raw_filters.get("document_id")
                retrieval_filter = RetrievalFilter(
                    document_id=uuid.UUID(doc_id) if doc_id else None
                )

            req = SearchRequest(
                query=query,
                strategy=strategy,
                top_k=top_k,
                filters=retrieval_filter,
                explain_plan=False,
            )

            resp = self.engine.execute(session, req)
            lat = (time.perf_counter() - t0) * 1000.0

            results_data = []
            for item in resp.results:
                results_data.append({
                    "chunk_id": str(item.chunk_id),
                    "document_id": str(item.document_id),
                    "document_name": item.document_name,
                    "sequence_index": item.sequence_index,
                    "score": round(item.similarity_score, 4),
                    "content": item.content,
                    "char_count": len(item.content),
                })

            preview = f"Hybrid search ({strategy}) retrieved {len(results_data)} chunk(s) for: '{query}'"
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                success=True,
                data=results_data,
                latency_ms=lat,
                items_count=len(results_data),
                raw_preview=preview,
            )

        except Exception as e:
            lat = (time.perf_counter() - t0) * 1000.0
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                success=False,
                error=str(e),
                latency_ms=lat,
                items_count=0,
                raw_preview=f"Error: {e}",
            )
