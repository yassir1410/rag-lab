import time
import json
import uuid
import asyncio
from typing import Dict, List, Optional, AsyncIterator, Any
from sqlalchemy.orm import Session

from app.models.conversation import Conversation, ConversationMessage, ConversationSummary
from app.rag.conversation.schemas import (
    TurnExecutionRequest,
    TurnExecutionResponse,
    ConversationSettings,
    MemoryConfig,
    ConversationMessageResponse,
    QueryRewriteResult,
    MemorySelectionResult,
)
from app.rag.conversation.repository import conversation_repo
from app.rag.conversation.memory import memory_registry
from app.rag.conversation.history_manager import history_manager
from app.rag.conversation.query_rewriter import query_rewriter
from app.rag.conversation.context_builder import conversational_context_builder
from app.rag.conversation.metrics import ConversationalPipelineLatencyTracker
from app.rag.retrieval.service import retrieval_service
from app.rag.retrieval.schemas import SearchRequest, RetrievalFilter
from app.rag.generation.context_builder import context_builder
from app.rag.generation.citation_manager import citation_manager
from app.rag.generation.registry import llm_registry
from app.rag.generation.base import GenerationRequest


class ConversationalRAGOrchestrator:
    """
    Central Coordinator for Phase 09 Conversational RAG & Memory Engineering:
    Executes the 14-step turn lifecycle across history loading, memory selection,
    coreference-aware query rewriting, Phase 08 hybrid retrieval, segregated prompt assembly,
    grounded LLM inference, citation extraction, and atomic turn persistence.
    """

    def __init__(self):
        self.repo = conversation_repo
        self.memory_registry = memory_registry
        self.history_manager = history_manager
        self.query_rewriter = query_rewriter
        self.context_builder = conversational_context_builder
        self.retrieval_service = retrieval_service
        self.doc_context_builder = context_builder
        self.citation_manager = citation_manager
        self.llm_registry = llm_registry

    def _resolve_settings(
        self,
        conversation: Conversation,
        overrides: Optional[Dict[str, Any]],
    ) -> ConversationSettings:
        merged = dict(conversation.settings or {})
        if overrides:
            merged.update(overrides)
        return ConversationSettings(**merged)

    def execute_turn(
        self,
        session: Session,
        conversation_id: uuid.UUID,
        request: TurnExecutionRequest,
    ) -> TurnExecutionResponse:
        tracker = ConversationalPipelineLatencyTracker()
        warnings: List[str] = []

        # 1. Conversation Validation
        conv = self.repo.get_conversation(session, conversation_id)
        if not conv:
            raise ValueError(f"Conversation '{conversation_id}' not found.")
        if conv.status == "archived":
            raise ValueError("Cannot append messages to an archived conversation.")

        settings = self._resolve_settings(conv, request.override_settings)

        # 2. History Loading & Turn Order Validation
        t_hist = time.perf_counter()
        raw_messages = self.repo.get_all_ordered_messages(session, conversation_id)
        latest_msg = raw_messages[-1] if raw_messages else None
        tracker.history_loading_ms = (time.perf_counter() - t_hist) * 1000.0

        clean_user_content = self.history_manager.sanitize_message_content(request.message)
        self.history_manager.validate_turn_order("user", latest_msg)

        # 3. Persist Incoming User Message
        t_persist = time.perf_counter()
        user_msg = self.repo.add_message(
            session=session,
            conversation_id=conversation_id,
            role="user",
            content=clean_user_content,
        )
        tracker.persistence_ms = (time.perf_counter() - t_persist) * 1000.0

        # 4. Memory Strategy Selection
        t_mem = time.perf_counter()
        memory_strategy = self.memory_registry.get_strategy(settings.memory_strategy)
        latest_summary = self.repo.get_latest_summary(session, conversation_id)
        mem_selection = memory_strategy.select_memory(
            messages=raw_messages,  # History before current user turn
            summary=latest_summary,
            config=settings.memory_config,
            total_context_budget=settings.max_context_tokens,
            prompt_and_retrieval_tokens=0,
        )
        tracker.memory_selection_ms = (time.perf_counter() - t_mem) * 1000.0

        # 5. History-Aware Query Rewriting
        t_rewrite = time.perf_counter()
        rewrite_res = self.query_rewriter.rewrite_query(
            query=clean_user_content,
            history=mem_selection.included_messages,
            summary=mem_selection.summary_used,
            llm_provider=settings.llm_provider,
            llm_model=settings.llm_model,
            enabled=settings.enable_query_rewriting,
        )
        tracker.query_rewriting_ms = (time.perf_counter() - t_rewrite) * 1000.0
        if rewrite_res.uncertainty_warning:
            warnings.append(rewrite_res.uncertainty_warning)

        effective_retrieval_query = rewrite_res.rewritten_query if rewrite_res.is_rewritten else clean_user_content

        # 6. Documentary Retrieval (Phase 08 Engine)
        filter_spec = RetrievalFilter(document_id=settings.document_filter) if settings.document_filter else None
        search_req = SearchRequest(
            query=effective_retrieval_query,
            strategy=settings.retrieval_strategy,
            model_id=settings.embedding_model_id,
            search_algorithm=settings.search_algorithm,
            distance_metric=settings.distance_metric,
            top_k=settings.top_k,
            filters=filter_spec,
            hybrid_alpha=settings.hybrid_alpha,
            rrf_k=settings.rrf_k,
            reranker_model=settings.reranker_model,
            enable_query_expansion=False,
            enable_neighbor_expansion=False,
            explain_plan=False,
        )

        search_resp = self.retrieval_service.search(session, search_req)
        tracker.embedding_ms = search_resp.embedding_latency_ms
        tracker.retrieval_ms = search_resp.query_latency_ms
        warnings.extend(search_resp.warnings)

        # 7. Document Context Assembly & Citation Tagging
        t_ctx = time.perf_counter()
        doc_context_text, source_items, ctx_warns = self.doc_context_builder.build_context(
            retrieved_items=search_resp.results,
            max_context_tokens=settings.max_context_tokens,
        )
        tracker.context_assembly_ms = (time.perf_counter() - t_ctx) * 1000.0
        warnings.extend(ctx_warns)

        # 8. Segregated Conversational Prompt Construction
        system_prompt, user_prompt, full_prompt = self.context_builder.build_conversational_prompt(
            question=clean_user_content,
            retrieved_context_text=doc_context_text,
            history_messages=mem_selection.included_messages,
            summary_text=mem_selection.summary_used,
        )

        # 9. Token Budget Accounting
        sys_tokens = self.history_manager.count_tokens(system_prompt)
        q_tokens = self.history_manager.count_tokens(clean_user_content)
        ctx_tokens = self.history_manager.count_tokens(doc_context_text)
        mem_tokens = mem_selection.allocated_memory_tokens
        token_breakdown = {
            "system_prompt_tokens": sys_tokens,
            "user_question_tokens": q_tokens,
            "conversation_memory_tokens": mem_tokens,
            "retrieved_context_tokens": ctx_tokens,
            "reserved_generation_tokens": settings.max_output_tokens,
            "total_prompt_tokens": sys_tokens + q_tokens + ctx_tokens + mem_tokens,
            "context_window": settings.max_context_tokens,
        }

        # 10. LLM Inference
        provider = self.llm_registry.get_provider(settings.llm_provider)
        gen_req = GenerationRequest(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=settings.temperature,
            max_tokens=settings.max_output_tokens,
            metadata={"model": settings.llm_model} if settings.llm_model else {},
        )

        t_gen = time.perf_counter()
        gen_resp = provider.generate(gen_req)
        tracker.llm_generation_ms = (time.perf_counter() - t_gen) * 1000.0

        # 11. Citation Extraction & Grounding Validation
        t_cite = time.perf_counter()
        citation_details, invalid_citations, uncited_sources = self.citation_manager.validate_citations(
            generated_text=gen_resp.content,
            sources=source_items,
        )
        tracker.citation_validation_ms = (time.perf_counter() - t_cite) * 1000.0

        if invalid_citations:
            warnings.append(
                f"Assistant cited {len(invalid_citations)} invalid identifier(s) ({', '.join(invalid_citations)}) not in retrieved passages."
            )

        # 12. Persist Assistant Turn
        t_p2 = time.perf_counter()
        trace_id = str(uuid.uuid4())
        assistant_meta = {
            "rewritten_query_info": rewrite_res.model_dump(),
            "sources": [s.model_dump() for s in source_items],
            "citations": [c.model_dump() for c in citation_details],
            "invalid_citations": invalid_citations,
            "uncited_sources": uncited_sources,
            "timing_breakdown": tracker.to_dict(),
            "token_metrics": token_breakdown,
            "memory_strategy": settings.memory_strategy,
            "retrieval_strategy": settings.retrieval_strategy,
            "warnings": warnings,
        }

        assistant_msg = self.repo.add_message(
            session=session,
            conversation_id=conversation_id,
            role="assistant",
            content=gen_resp.content,
            metadata_json=assistant_meta,
            trace_id=trace_id,
        )
        tracker.persistence_ms += (time.perf_counter() - t_p2) * 1000.0

        # 13. Auto-Summary Trigger (for summary_memory strategy)
        if settings.memory_strategy == "summary_memory":
            self._maybe_refresh_summary(session, conv, settings)

        user_schema = ConversationMessageResponse(
            id=user_msg.id,
            conversation_id=user_msg.conversation_id,
            role=user_msg.role,
            content=user_msg.content,
            sequence_number=user_msg.sequence_number,
            metadata_json=user_msg.metadata_json or {},
            trace_id=user_msg.trace_id,
            created_at=user_msg.created_at.isoformat(),
        )
        assistant_schema = ConversationMessageResponse(
            id=assistant_msg.id,
            conversation_id=assistant_msg.conversation_id,
            role=assistant_msg.role,
            content=assistant_msg.content,
            sequence_number=assistant_msg.sequence_number,
            metadata_json=assistant_msg.metadata_json or {},
            trace_id=assistant_msg.trace_id,
            created_at=assistant_msg.created_at.isoformat(),
        )

        return TurnExecutionResponse(
            user_message=user_schema,
            assistant_message=assistant_schema,
            rewritten_query_info=rewrite_res,
            memory_selection=mem_selection,
            sources=[s.model_dump() for s in source_items],
            citations=[c.model_dump() for c in citation_details],
            invalid_citations=invalid_citations,
            uncited_sources=uncited_sources,
            timing_breakdown=tracker.to_dict(),
            token_metrics=token_breakdown,
            warnings=warnings,
        )

    async def execute_turn_stream(
        self,
        session: Session,
        conversation_id: uuid.UUID,
        request: TurnExecutionRequest,
    ) -> AsyncIterator[str]:
        """
        SSE Streaming multi-turn orchestrator:
        Emits real-time progression events through each pipeline stage:
        - stage: init
        - stage: memory_selected
        - stage: query_rewritten
        - stage: retrieval_complete
        - stage: token (live token delta)
        - stage: citation_validation
        - stage: turn_complete
        """
        tracker = ConversationalPipelineLatencyTracker()
        warnings: List[str] = []

        def _sse_event(event_type: str, payload: Dict[str, Any]) -> str:
            return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"

        # 1. Validation
        conv = self.repo.get_conversation(session, conversation_id)
        if not conv or conv.status == "archived":
            yield _sse_event("error", {"error": "Conversation not found or archived."})
            return

        settings = self._resolve_settings(conv, request.override_settings)

        # 2. History & User Message
        raw_messages = self.repo.get_all_ordered_messages(session, conversation_id)
        clean_user_content = self.history_manager.sanitize_message_content(request.message)

        user_msg = self.repo.add_message(
            session=session,
            conversation_id=conversation_id,
            role="user",
            content=clean_user_content,
        )
        yield _sse_event("user_message_persisted", {
            "id": str(user_msg.id),
            "sequence_number": user_msg.sequence_number,
            "content": user_msg.content,
        })

        # 3. Memory Selection Event
        memory_strategy = self.memory_registry.get_strategy(settings.memory_strategy)
        latest_summary = self.repo.get_latest_summary(session, conversation_id)
        mem_selection = memory_strategy.select_memory(
            messages=raw_messages,
            summary=latest_summary,
            config=settings.memory_config,
            total_context_budget=settings.max_context_tokens,
        )
        yield _sse_event("stage_memory", {
            "strategy": settings.memory_strategy,
            "included_count": len(mem_selection.included_messages),
            "allocated_tokens": mem_selection.allocated_memory_tokens,
            "summary_used": bool(mem_selection.summary_used),
        })

        # 4. Query Rewriting Event
        rewrite_res = self.query_rewriter.rewrite_query(
            query=clean_user_content,
            history=mem_selection.included_messages,
            summary=mem_selection.summary_used,
            llm_provider=settings.llm_provider,
            llm_model=settings.llm_model,
            enabled=settings.enable_query_rewriting,
        )
        yield _sse_event("stage_rewriting", {
            "original_query": rewrite_res.original_query,
            "rewritten_query": rewrite_res.rewritten_query,
            "is_rewritten": rewrite_res.is_rewritten,
            "ambiguity_detected": rewrite_res.ambiguity_detected,
        })

        effective_query = rewrite_res.rewritten_query if rewrite_res.is_rewritten else clean_user_content

        # 5. Retrieval Event
        filter_spec = RetrievalFilter(document_id=settings.document_filter) if settings.document_filter else None
        search_req = SearchRequest(
            query=effective_query,
            strategy=settings.retrieval_strategy,
            model_id=settings.embedding_model_id,
            search_algorithm=settings.search_algorithm,
            distance_metric=settings.distance_metric,
            top_k=settings.top_k,
            filters=filter_spec,
            hybrid_alpha=settings.hybrid_alpha,
            rrf_k=settings.rrf_k,
            reranker_model=settings.reranker_model,
        )
        search_resp = self.retrieval_service.search(session, search_req)

        doc_context_text, source_items, ctx_warns = self.doc_context_builder.build_context(
            retrieved_items=search_resp.results,
            max_context_tokens=settings.max_context_tokens,
        )
        yield _sse_event("stage_retrieval", {
            "strategy": settings.retrieval_strategy,
            "retrieved_count": len(source_items),
            "sources": [s.model_dump() for s in source_items],
            "query_latency_ms": search_resp.query_latency_ms,
        })

        # 6. Prompt Construction
        system_prompt, user_prompt, _ = self.context_builder.build_conversational_prompt(
            question=clean_user_content,
            retrieved_context_text=doc_context_text,
            history_messages=mem_selection.included_messages,
            summary_text=mem_selection.summary_used,
        )

        # 7. LLM Streaming
        provider = self.llm_registry.get_provider(settings.llm_provider)
        gen_req = GenerationRequest(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=settings.temperature,
            max_tokens=settings.max_output_tokens,
            metadata={"model": settings.llm_model} if settings.llm_model else {},
        )

        full_answer = ""
        try:
            for token_delta in provider.generate_stream(gen_req):
                full_answer += token_delta
                yield _sse_event("token", {"delta": token_delta})
                # Small non-blocking yield for async loop
                await asyncio.sleep(0.001)
        except Exception as e:
            yield _sse_event("error", {"error": f"LLM streaming error: {str(e)}"})
            return

        # 8. Citation Extraction
        citation_details, invalid_citations, uncited_sources = self.citation_manager.validate_citations(
            generated_text=full_answer,
            sources=source_items,
        )

        # 9. Persist Assistant Turn
        trace_id = str(uuid.uuid4())
        assistant_meta = {
            "rewritten_query_info": rewrite_res.model_dump(),
            "sources": [s.model_dump() for s in source_items],
            "citations": [c.model_dump() for c in citation_details],
            "invalid_citations": invalid_citations,
            "uncited_sources": uncited_sources,
            "timing_breakdown": tracker.to_dict(),
            "memory_strategy": settings.memory_strategy,
            "retrieval_strategy": settings.retrieval_strategy,
            "warnings": warnings,
        }

        assistant_msg = self.repo.add_message(
            session=session,
            conversation_id=conversation_id,
            role="assistant",
            content=full_answer,
            metadata_json=assistant_meta,
            trace_id=trace_id,
        )

        yield _sse_event("turn_complete", {
            "assistant_message_id": str(assistant_msg.id),
            "sequence_number": assistant_msg.sequence_number,
            "full_answer": full_answer,
            "citations": [c.model_dump() for c in citation_details],
            "invalid_citations": invalid_citations,
            "timing": tracker.to_dict(),
        })

    def _maybe_refresh_summary(
        self,
        session: Session,
        conv: Conversation,
        settings: ConversationSettings,
    ) -> None:
        """Triggered when unsummarized turns exceed summary_trigger_turns."""
        messages = self.repo.get_all_ordered_messages(session, conv.id)
        latest_summary = self.repo.get_latest_summary(session, conv.id)
        last_covered_seq = latest_summary.covered_until_sequence if latest_summary else 0

        unsummarized = [m for m in messages if m.sequence_number > last_covered_seq]
        user_unsummarized = [m for m in unsummarized if m.role == "user"]

        if len(user_unsummarized) >= settings.memory_config.summary_trigger_turns:
            # We compress messages up to the turn right before keep_recent_turns
            keep_count = settings.memory_config.keep_recent_turns * 2
            to_compress = messages[:-keep_count] if len(messages) > keep_count else []
            if not to_compress:
                return

            last_seq = to_compress[-1].sequence_number
            # Build rolling summary
            prev_sum = latest_summary.summary_text if latest_summary else ""
            summary_content = "\n".join([f"{m.role}: {m.content}" for m in to_compress])

            sum_prompt = (
                f"You are a conversation summarizer. Distill the following dialogue into a concise summary.\n"
                f"Preserve all entities, technologies, numbers, user preferences, and constraints.\n"
                f"Do not hallucinate.\n\n"
                f"Existing Summary:\n{prev_sum}\n\n"
                f"New Dialogue to Incorporate:\n{summary_content}\n\n"
                f"Updated Concise Summary:"
            )

            try:
                provider = self.llm_registry.get_provider(settings.llm_provider)
                res = provider.generate(GenerationRequest(
                    prompt=sum_prompt,
                    system_prompt="You are a precise conversation summarizer.",
                    temperature=0.0,
                    max_tokens=256,
                ))
                new_summary_text = res.content.strip()
                if new_summary_text:
                    self.repo.save_summary(
                        session=session,
                        conversation_id=conv.id,
                        summary_text=new_summary_text,
                        covered_until_sequence=last_seq,
                        summarizer_model=settings.llm_model,
                    )
            except Exception:
                pass


conversational_orchestrator = ConversationalRAGOrchestrator()
