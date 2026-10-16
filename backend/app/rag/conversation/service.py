import uuid
import time
from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy.orm import Session

from app.models.conversation import Conversation, ConversationMessage, ConversationSummary
from app.rag.conversation.schemas import (
    ConversationResponse,
    ConversationMessageResponse,
    ConversationSummaryResponse,
    ConversationSettings,
    MemorySelectionResult,
    QueryRewriteResult,
    StrategyComparisonResult,
)
from app.rag.conversation.repository import conversation_repo
from app.rag.conversation.memory import memory_registry
from app.rag.conversation.query_rewriter import query_rewriter
from app.rag.retrieval.service import retrieval_service
from app.rag.retrieval.schemas import SearchRequest


class ConversationalService:
    """
    High-level business service for Conversational RAG:
    Manages session lifecycle, memory previews, standalone query reformulations,
    and multi-strategy comparison diagnostics.
    """

    def __init__(self):
        self.repo = conversation_repo
        self.memory_registry = memory_registry
        self.query_rewriter = query_rewriter
        self.retrieval_service = retrieval_service

    def _conv_to_response(self, conv: Conversation, msg_count: int = 0, latest_msg: Optional[str] = None) -> ConversationResponse:
        return ConversationResponse(
            id=conv.id,
            title=conv.title,
            settings=conv.settings or {},
            status=conv.status,
            created_at=conv.created_at.isoformat(),
            updated_at=conv.updated_at.isoformat(),
            message_count=msg_count,
            latest_message=latest_msg,
        )

    def create_conversation(
        self,
        session: Session,
        title: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> ConversationResponse:
        # Default settings if none supplied
        default_settings = ConversationSettings().model_dump()
        if settings:
            default_settings.update(settings)

        conv = self.repo.create_conversation(session, title=title, settings=default_settings)
        return self._conv_to_response(conv, msg_count=0)

    def get_conversation(
        self,
        session: Session,
        conversation_id: uuid.UUID,
    ) -> Optional[ConversationResponse]:
        conv = self.repo.get_conversation(session, conversation_id)
        if not conv:
            return None
        messages = self.repo.get_all_ordered_messages(session, conversation_id)
        latest = messages[-1].content if messages else None
        return self._conv_to_response(conv, msg_count=len(messages), latest_msg=latest)

    def list_conversations(
        self,
        session: Session,
        skip: int = 0,
        limit: int = 50,
        search: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Tuple[List[ConversationResponse], int]:
        convs, total = self.repo.list_conversations(
            session=session, skip=skip, limit=limit, search=search, status=status
        )
        results = []
        for c in convs:
            msgs = self.repo.get_all_ordered_messages(session, c.id)
            latest = msgs[-1].content if msgs else None
            results.append(self._conv_to_response(c, msg_count=len(msgs), latest_msg=latest))
        return results, total

    def update_conversation(
        self,
        session: Session,
        conversation_id: uuid.UUID,
        title: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None,
    ) -> Optional[ConversationResponse]:
        conv = self.repo.update_conversation(
            session=session,
            conversation_id=conversation_id,
            title=title,
            settings=settings,
            status=status,
        )
        if not conv:
            return None
        messages = self.repo.get_all_ordered_messages(session, conversation_id)
        latest = messages[-1].content if messages else None
        return self._conv_to_response(conv, msg_count=len(messages), latest_msg=latest)

    def delete_conversation(
        self,
        session: Session,
        conversation_id: uuid.UUID,
    ) -> bool:
        return self.repo.delete_conversation(session, conversation_id)

    def list_messages(
        self,
        session: Session,
        conversation_id: uuid.UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> Tuple[List[ConversationMessageResponse], int]:
        messages, total = self.repo.get_messages(
            session=session, conversation_id=conversation_id, skip=skip, limit=limit
        )
        schemas = [
            ConversationMessageResponse(
                id=m.id,
                conversation_id=m.conversation_id,
                role=m.role,
                content=m.content,
                sequence_number=m.sequence_number,
                metadata_json=m.metadata_json or {},
                trace_id=m.trace_id,
                created_at=m.created_at.isoformat(),
            )
            for m in messages
        ]
        return schemas, total

    def inspect_memory(
        self,
        session: Session,
        conversation_id: uuid.UUID,
        strategy_override: Optional[str] = None,
    ) -> MemorySelectionResult:
        conv = self.repo.get_conversation(session, conversation_id)
        if not conv:
            raise ValueError(f"Conversation '{conversation_id}' not found.")

        settings = ConversationSettings(**(conv.settings or {}))
        chosen_strategy = strategy_override or settings.memory_strategy

        strat = self.memory_registry.get_strategy(chosen_strategy)
        messages = self.repo.get_all_ordered_messages(session, conversation_id)
        summary = self.repo.get_latest_summary(session, conversation_id)

        return strat.select_memory(
            messages=messages,
            summary=summary,
            config=settings.memory_config,
            total_context_budget=settings.max_context_tokens,
        )

    def preview_rewrite(
        self,
        session: Session,
        conversation_id: uuid.UUID,
        query: str,
    ) -> QueryRewriteResult:
        conv = self.repo.get_conversation(session, conversation_id)
        if not conv:
            raise ValueError(f"Conversation '{conversation_id}' not found.")

        settings = ConversationSettings(**(conv.settings or {}))
        strat = self.memory_registry.get_strategy(settings.memory_strategy)
        messages = self.repo.get_all_ordered_messages(session, conversation_id)
        summary = self.repo.get_latest_summary(session, conversation_id)

        mem_result = strat.select_memory(
            messages=messages,
            summary=summary,
            config=settings.memory_config,
            total_context_budget=settings.max_context_tokens,
        )

        return self.query_rewriter.rewrite_query(
            query=query,
            history=mem_result.included_messages,
            summary=mem_result.summary_used,
            llm_provider=settings.llm_provider,
            llm_model=settings.llm_model,
            enabled=settings.enable_query_rewriting,
        )

    def compare_strategies(
        self,
        session: Session,
        conversation_id: uuid.UUID,
        query: str,
    ) -> List[StrategyComparisonResult]:
        """
        Runs controlled side-by-side execution comparing all 4 memory strategies
        against the exact same conversation history state.
        """
        conv = self.repo.get_conversation(session, conversation_id)
        if not conv:
            raise ValueError(f"Conversation '{conversation_id}' not found.")

        settings = ConversationSettings(**(conv.settings or {}))
        messages = self.repo.get_all_ordered_messages(session, conversation_id)
        summary = self.repo.get_latest_summary(session, conversation_id)

        strategies = ["no_memory", "sliding_window", "token_bounded", "summary_memory"]
        results: List[StrategyComparisonResult] = []

        for s_id in strategies:
            t0 = time.perf_counter()
            strat = self.memory_registry.get_strategy(s_id)
            mem_res = strat.select_memory(
                messages=messages,
                summary=summary,
                config=settings.memory_config,
                total_context_budget=settings.max_context_tokens,
            )

            rewrite_res = self.query_rewriter.rewrite_query(
                query=query,
                history=mem_res.included_messages,
                summary=mem_res.summary_used,
                llm_provider=settings.llm_provider,
                llm_model=settings.llm_model,
                enabled=settings.enable_query_rewriting,
            )

            effective_q = rewrite_res.rewritten_query if rewrite_res.is_rewritten else query
            search_res = self.retrieval_service.search(
                session=session,
                request=SearchRequest(
                    query=effective_q,
                    strategy=settings.retrieval_strategy,
                    model_id=settings.embedding_model_id,
                    top_k=settings.top_k,
                ),
            )

            lat = (time.perf_counter() - t0) * 1000.0
            ret_ids = [str(r.chunk_id) for r in search_res.results]
            ret_seqs = [r.sequence_index for r in search_res.results]

            preview = f"Rewritten to: '{effective_q}' | Retrieved {len(ret_ids)} passages."
            results.append(
                StrategyComparisonResult(
                    strategy=s_id,
                    rewritten_query=effective_q,
                    memory_tokens=mem_res.allocated_memory_tokens,
                    retrieved_chunk_ids=ret_ids,
                    retrieved_chunk_sequences=ret_seqs,
                    latency_ms=round(lat, 2),
                    answer_preview=preview,
                )
            )

        return results


conversational_service = ConversationalService()
