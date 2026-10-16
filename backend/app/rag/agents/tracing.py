import uuid
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from app.models.agent import AgentExecutionTrace
from app.rag.agents.schemas import AgentQueryResponse, AgentStepTrace, EvidenceItem


class AgentTraceManager:
    """
    Manages persistence and retrieval of Agent Execution Traces.
    Maintains an in-memory ring buffer of recent traces for low-latency inspection,
    and commits full structured traces to the PostgreSQL 'agent_traces' table.
    """

    def __init__(self, max_cached: int = 50):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_cached = max_cached

    def save_trace(
        self,
        session: Session,
        response: AgentQueryResponse,
        conversation_id: Optional[uuid.UUID] = None,
    ) -> str:
        trace_id = response.trace_id or str(uuid.uuid4())

        # Serialize step traces
        serialized_steps = [s.model_dump(mode="json") for s in response.steps]
        serialized_evidence = [e.model_dump(mode="json") for e in response.evidence]

        trace_data = {
            "trace_id": trace_id,
            "conversation_id": str(conversation_id) if conversation_id else None,
            "strategy": response.strategy.value if hasattr(response.strategy, "value") else str(response.strategy),
            "question": response.question,
            "final_answer": response.final_answer,
            "stop_reason": response.stop_reason.value if hasattr(response.stop_reason, "value") else str(response.stop_reason),
            "total_steps": response.total_steps,
            "tool_calls_count": response.tool_calls_count,
            "evidence_count": response.evidence_count,
            "latency_ms": response.latency_ms,
            "tokens_used": response.tokens_used,
            "steps": serialized_steps,
            "evidence": serialized_evidence,
            "citations": response.citations,
        }

        # Cache in memory
        self._cache[trace_id] = trace_data
        if len(self._cache) > self._max_cached:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        # Persist in DB
        try:
            db_trace = AgentExecutionTrace(
                id=uuid.UUID(trace_id),
                conversation_id=conversation_id,
                strategy=trace_data["strategy"],
                question=trace_data["question"],
                final_answer=trace_data["final_answer"],
                stop_reason=trace_data["stop_reason"],
                total_steps=trace_data["total_steps"],
                tool_calls_count=trace_data["tool_calls_count"],
                evidence_count=trace_data["evidence_count"],
                latency_ms=trace_data["latency_ms"],
                tokens_used=trace_data["tokens_used"],
                steps_data=serialized_steps,
                metadata_json={
                    "evidence": serialized_evidence,
                    "citations": response.citations,
                },
            )
            session.add(db_trace)
            session.commit()
        except Exception:
            session.rollback()
            # If DB insert fails (e.g. SQLite without UUID table in tests), keep in-memory trace

        return trace_id

    def get_trace(self, session: Session, trace_id: str) -> Optional[Dict[str, Any]]:
        # 1. Check in-memory cache
        if trace_id in self._cache:
            return self._cache[trace_id]

        # 2. Check DB
        try:
            t_uuid = uuid.UUID(trace_id)
            record = session.query(AgentExecutionTrace).filter(AgentExecutionTrace.id == t_uuid).first()
            if record:
                meta = record.metadata_json or {}
                data = {
                    "trace_id": str(record.id),
                    "conversation_id": str(record.conversation_id) if record.conversation_id else None,
                    "strategy": record.strategy,
                    "question": record.question,
                    "final_answer": record.final_answer,
                    "stop_reason": record.stop_reason,
                    "total_steps": record.total_steps,
                    "tool_calls_count": record.tool_calls_count,
                    "evidence_count": record.evidence_count,
                    "latency_ms": record.latency_ms,
                    "tokens_used": record.tokens_used,
                    "steps": record.steps_data or [],
                    "evidence": meta.get("evidence", []),
                    "citations": meta.get("citations", []),
                }
                self._cache[trace_id] = data
                return data
        except Exception:
            pass

        return None


# Singleton instance
agent_trace_manager = AgentTraceManager()
