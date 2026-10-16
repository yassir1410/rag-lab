import time
import uuid
from typing import Dict, List, Optional, AsyncIterator, Any
from sqlalchemy.orm import Session

from app.rag.retrieval.service import retrieval_service
from app.rag.retrieval.schemas import SearchRequest
from app.rag.generation.base import GenerationRequest
from app.rag.generation.context_builder import context_builder
from app.rag.generation.prompt_builder import prompt_builder
from app.rag.generation.citation_manager import citation_manager
from app.rag.generation.token_budget import token_budget_manager
from app.rag.generation.registry import llm_registry
from app.rag.generation.metrics import PipelineLatencyTracker
from app.rag.generation.schemas import (
    RAGQueryRequest,
    RAGQueryResponse,
    ContextSourceItem,
)


class RAGOrchestrator:
    """
    Coordinates the complete End-to-End Naive RAG pipeline:
    1. Input Validation
    2. Query Embedding (via Phase 05 RetrievalService)
    3. Vector Retrieval (Exact KNN, HNSW, or IVFFlat)
    4. Context Assembly & Citation Tagging ([S1], [S2]...)
    5. Prompt Construction (Configurable templates: basic, grounded, concise, detailed)
    6. LLM Generation (Local Ollama, OpenAI, Gemini, or Mock)
    7. Citation Extraction & Grounding Validation
    8. Trace & Latency Recording
    """

    def __init__(self, trace_cache_size: int = 50):
        self._traces: Dict[str, RAGQueryResponse] = {}
        self._trace_cache_size = trace_cache_size

    def get_trace(self, trace_id: str) -> Optional[RAGQueryResponse]:
        return self._traces.get(trace_id)

    def _record_trace(self, response: RAGQueryResponse):
        if len(self._traces) >= self._trace_cache_size:
            # Drop oldest trace
            first_key = next(iter(self._traces))
            del self._traces[first_key]
        self._traces[response.trace_id] = response

    def execute_query(
        self,
        session: Session,
        request: RAGQueryRequest,
    ) -> RAGQueryResponse:
        trace_id = str(uuid.uuid4())
        tracker = PipelineLatencyTracker()
        warnings: List[str] = []

        # 1. Input Validation & Bounds Clamping
        question_cleaned = request.question.strip()
        if not question_cleaned:
            raise ValueError("Query question cannot be empty.")

        # 2 & 3. Vector / Hybrid Retrieval (Phase 08 Enhanced)
        search_req = SearchRequest(
            query=question_cleaned,
            strategy=request.retrieval_strategy,
            model_id=request.embedding_model_id,
            search_algorithm=request.search_algorithm,
            distance_metric=request.distance_metric,
            top_k=request.top_k,
            filters=request.filters,
            hybrid_alpha=request.hybrid_alpha,
            rrf_k=request.rrf_k,
            reranker_model=request.reranker_model,
            enable_query_expansion=request.enable_query_expansion,
            enable_neighbor_expansion=request.enable_neighbor_expansion,
            ef_search=request.ef_search or 40,
            probes=request.probes or 1,
            explain_plan=False,
        )

        search_resp = retrieval_service.search(session, search_req)
        tracker.embedding_latency_ms = search_resp.embedding_latency_ms
        tracker.retrieval_latency_ms = search_resp.query_latency_ms
        warnings.extend(search_resp.warnings)

        # 4. Context Assembly & Stable Citation Tagging
        t_ctx_start = time.perf_counter()
        context_text, source_items, ctx_warns = context_builder.build_context(
            retrieved_items=search_resp.results,
            max_context_tokens=request.max_context_tokens,
        )
        tracker.context_assembly_ms = (time.perf_counter() - t_ctx_start) * 1000.0
        warnings.extend(ctx_warns)

        # 5. Prompt Construction
        t_prompt_start = time.perf_counter()
        built_prompt = prompt_builder.build_prompt(
            template_id=request.prompt_template,
            question=question_cleaned,
            context=context_text,
        )
        tracker.prompt_building_ms = (time.perf_counter() - t_prompt_start) * 1000.0

        # Calculate token metrics
        token_metrics = token_budget_manager.allocate(
            system_prompt=built_prompt.system_prompt,
            question=question_cleaned,
            context_text=context_text,
            max_output_tokens=request.max_output_tokens,
            model_context_window=4096,
            requested_max_context_tokens=request.max_context_tokens,
        )

        # 6. LLM Generation
        provider = llm_registry.get_provider(request.llm_provider)
        gen_req = GenerationRequest(
            prompt=built_prompt.user_prompt,
            system_prompt=built_prompt.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_output_tokens,
            metadata={"model": request.llm_model} if request.llm_model else {},
        )

        t_llm_start = time.perf_counter()
        gen_resp = provider.generate(gen_req)
        tracker.llm_inference_ms = (time.perf_counter() - t_llm_start) * 1000.0
        tracker.model_load_latency_ms = gen_resp.model_load_latency_ms

        # 7. Citation Extraction & Grounding Validation
        t_cite_start = time.perf_counter()
        citation_details, invalid_citations, uncited_sources = citation_manager.validate_citations(
            generated_text=gen_resp.content,
            sources=source_items,
        )
        tracker.citation_validation_ms = (time.perf_counter() - t_cite_start) * 1000.0

        if invalid_citations:
            warnings.append(
                f"Model generated {len(invalid_citations)} invalid citation(s) ({', '.join(invalid_citations)}) that were not in the retrieved context."
            )

        response = RAGQueryResponse(
            trace_id=trace_id,
            question=question_cleaned,
            answer=gen_resp.content,
            llm_provider=request.llm_provider,
            llm_model=gen_resp.model_id,
            prompt_template=request.prompt_template,
            sources=source_items,
            citations=citation_details,
            invalid_citations=invalid_citations,
            uncited_sources=uncited_sources,
            timing_metrics=tracker.to_schema(),
            token_metrics=token_metrics,
            assembled_prompt=built_prompt.full_prompt if request.include_assembled_prompt else None,
            warnings=warnings,
        )

        self._record_trace(response)
        return response

    async def execute_query_stream(
        self,
        session: Session,
        request: RAGQueryRequest,
    ) -> AsyncIterator[Dict[str, Any]]:
        trace_id = str(uuid.uuid4())
        tracker = PipelineLatencyTracker()
        warnings: List[str] = []

        # 1. Validation
        question_cleaned = request.question.strip()
        if not question_cleaned:
            yield {"event": "error", "data": {"error": "Question cannot be empty"}}
            return

        yield {"event": "status", "data": {"stage": "embedding", "message": "Encoding query vector..."}}

        # 2 & 3. Vector / Hybrid Retrieval (Phase 08 Enhanced)
        search_req = SearchRequest(
            query=question_cleaned,
            strategy=request.retrieval_strategy,
            model_id=request.embedding_model_id,
            search_algorithm=request.search_algorithm,
            distance_metric=request.distance_metric,
            top_k=request.top_k,
            filters=request.filters,
            hybrid_alpha=request.hybrid_alpha,
            rrf_k=request.rrf_k,
            reranker_model=request.reranker_model,
            enable_query_expansion=request.enable_query_expansion,
            enable_neighbor_expansion=request.enable_neighbor_expansion,
            ef_search=request.ef_search or 40,
            probes=request.probes or 1,
            explain_plan=False,
        )
        search_resp = retrieval_service.search(session, search_req)
        tracker.embedding_latency_ms = search_resp.embedding_latency_ms
        tracker.retrieval_latency_ms = search_resp.query_latency_ms
        warnings.extend(search_resp.warnings)

        yield {"event": "status", "data": {"stage": "retrieval", "message": f"Retrieved {len(search_resp.results)} candidate passages"}}

        # 4. Context Assembly
        t_ctx_start = time.perf_counter()
        context_text, source_items, ctx_warns = context_builder.build_context(
            retrieved_items=search_resp.results,
            max_context_tokens=request.max_context_tokens,
        )
        tracker.context_assembly_ms = (time.perf_counter() - t_ctx_start) * 1000.0
        warnings.extend(ctx_warns)

        # Emit sources early so UI can render retrieved evidence immediately
        yield {
            "event": "sources",
            "data": {
                "trace_id": trace_id,
                "sources": [s.model_dump(mode="json") for s in source_items],
            },
        }

        # 5. Prompt Construction
        t_prompt_start = time.perf_counter()
        built_prompt = prompt_builder.build_prompt(
            template_id=request.prompt_template,
            question=question_cleaned,
            context=context_text,
        )
        tracker.prompt_building_ms = (time.perf_counter() - t_prompt_start) * 1000.0

        token_metrics = token_budget_manager.allocate(
            system_prompt=built_prompt.system_prompt,
            question=question_cleaned,
            context_text=context_text,
            max_output_tokens=request.max_output_tokens,
            model_context_window=4096,
            requested_max_context_tokens=request.max_context_tokens,
        )

        yield {"event": "status", "data": {"stage": "generation", "message": f"Calling LLM provider '{request.llm_provider}'..."}}

        # 6. Stream LLM Generation
        provider = llm_registry.get_provider(request.llm_provider)
        gen_req = GenerationRequest(
            prompt=built_prompt.user_prompt,
            system_prompt=built_prompt.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_output_tokens,
            metadata={"model": request.llm_model} if request.llm_model else {},
        )

        t_llm_start = time.perf_counter()
        accumulated_text = []

        try:
            async for chunk in provider.generate_stream(gen_req):
                accumulated_text.append(chunk.content)
                yield {
                    "event": "chunk",
                    "data": {
                        "content": chunk.content,
                        "is_final": chunk.is_final,
                    },
                }
        except Exception as e:
            yield {"event": "error", "data": {"error": str(e)}}
            return

        tracker.llm_inference_ms = (time.perf_counter() - t_llm_start) * 1000.0
        full_answer = "".join(accumulated_text)

        # 7. Citation Extraction & Validation
        t_cite_start = time.perf_counter()
        citation_details, invalid_citations, uncited_sources = citation_manager.validate_citations(
            generated_text=full_answer,
            sources=source_items,
        )
        tracker.citation_validation_ms = (time.perf_counter() - t_cite_start) * 1000.0

        # Construct final trace
        final_response = RAGQueryResponse(
            trace_id=trace_id,
            question=question_cleaned,
            answer=full_answer,
            llm_provider=request.llm_provider,
            llm_model=request.llm_model or "default",
            prompt_template=request.prompt_template,
            sources=source_items,
            citations=citation_details,
            invalid_citations=invalid_citations,
            uncited_sources=uncited_sources,
            timing_metrics=tracker.to_schema(),
            token_metrics=token_metrics,
            assembled_prompt=built_prompt.full_prompt if request.include_assembled_prompt else None,
            warnings=warnings,
        )
        self._record_trace(final_response)

        # Yield complete metrics and citations to finalize streaming session
        yield {
            "event": "done",
            "data": {
                "trace_id": trace_id,
                "citations": [c.model_dump(mode="json") for c in citation_details],
                "invalid_citations": invalid_citations,
                "uncited_sources": uncited_sources,
                "timing_metrics": tracker.to_schema().model_dump(mode="json"),
                "token_metrics": token_metrics.model_dump(mode="json"),
                "assembled_prompt": built_prompt.full_prompt if request.include_assembled_prompt else None,
                "warnings": warnings,
            },
        }


rag_orchestrator = RAGOrchestrator()
