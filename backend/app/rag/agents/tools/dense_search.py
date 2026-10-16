import time
import uuid
from typing import Dict, Any, Optional, Type
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.rag.agents.tools.base import BaseAgentTool
from app.rag.agents.schemas import ToolResult
from app.rag.retrieval.hybrid import HybridRetrievalEngine
from app.rag.retrieval.schemas import SearchRequest, RetrievalFilter


class DenseSearchArgs(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000, description="Semantic search query string")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata filters, e.g. {'document_id': '...'}")
    top_k: int = Field(default=5, ge=1, le=15, description="Number of top chunks to retrieve")


class DenseSearchTool(BaseAgentTool):
    """
    Approved Tool: Dense Vector Similarity Search.
    Embeds the user query and searches pgvector chunks using high-dimensional cosine similarity.
    """

    def __init__(self, engine: Optional[HybridRetrievalEngine] = None):
        self.engine = engine or HybridRetrievalEngine()

    @property
    def name(self) -> str:
        return "dense_search"

    @property
    def description(self) -> str:
        return (
            "Performs semantic vector similarity search against document chunks using pgvector. "
            "Best for finding conceptually related text when exact keywords might differ."
        )

    @property
    def args_schema(self) -> Type[BaseModel]:
        return DenseSearchArgs

    def execute(self, session: Session, **kwargs) -> ToolResult:
        t0 = time.perf_counter()
        call_id = kwargs.pop("call_id", str(uuid.uuid4())[:8])

        try:
            validated = self.args_schema(**kwargs)
            query = validated.query
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
                strategy="dense_only",
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

            preview = f"Retrieved {len(results_data)} chunk(s) for query: '{query}'"
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
