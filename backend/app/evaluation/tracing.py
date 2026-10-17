import re
import time
import uuid
from contextlib import contextmanager
from typing import Dict, Any, List, Optional, Sequence
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from app.models.evaluation import EvaluationTrace


class RedactionPolicy:
    """
    Sanitizes trace payloads to ensure API keys, authorization tokens,
    and heavy binary/base64 image payloads never leak into trace records.
    """

    # Common API key header/field patterns
    SENSITIVE_KEYS = {
        "api_key",
        "authorization",
        "x-goog-api-key",
        "token",
        "secret",
        "bearer",
        "password",
    }

    # Matches data:image/png;base64,... strings
    BASE64_IMAGE_PATTERN = re.compile(r"data:image\/[a-zA-Z]+;base64,[A-Za-z0-9+/=]{50,}")

    @classmethod
    def redact_value(cls, key: str, value: Any) -> Any:
        key_lower = key.lower()
        if any(s in key_lower for s in cls.SENSITIVE_KEYS):
            return "[REDACTED_SECRET]"

        if isinstance(value, str):
            # Check for base64 images
            if "data:image" in value and ";base64," in value:
                return cls.BASE64_IMAGE_PATTERN.sub("[REDACTED_BASE64_IMAGE]", value)
            # Truncate extremely large string payloads in metadata
            if len(value) > 2000:
                return value[:1000] + f"... [TRUNCATED_{len(value)}_CHARS]"
            return value

        if isinstance(value, dict):
            return {k: cls.redact_value(k, v) for k, v in value.items()}

        if isinstance(value, list):
            return [cls.redact_value(f"{key}[{i}]", item) for i, item in enumerate(value)]

        return value

    @classmethod
    def sanitize_metadata(cls, metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {k: cls.redact_value(k, v) for k, v in metadata.items()}


class TraceStage:
    """
    Represents an individual timed step within a RAG pipeline execution.
    """

    def __init__(self, stage_name: str, start_ms: float, metadata: Optional[Dict[str, Any]] = None):
        self.stage_name = stage_name
        self.start_ms = start_ms
        self.duration_ms: float = 0.0
        self.metadata: Dict[str, Any] = metadata or {}
        self.error: Optional[str] = None

    def finish(self, end_ms: float, error: Optional[str] = None) -> None:
        self.duration_ms = max(0.0, end_ms - self.start_ms)
        if error:
            self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "start_ms": round(self.start_ms, 2),
            "duration_ms": round(self.duration_ms, 2),
            "metadata": RedactionPolicy.sanitize_metadata(self.metadata),
            "error": self.error,
        }


class EvaluationTracer:
    """
    Execution tracer for capturing end-to-end RAG lifecycle telemetry.
    Supports stage timing, correlation IDs, metadata sanitization,
    and PostgreSQL persistence.
    """

    def __init__(
        self,
        run_id: Optional[uuid.UUID] = None,
        item_result_id: Optional[uuid.UUID] = None,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        provider: str = "mock",
        model: str = "mock-instruct-v1",
        document_ids: Optional[Sequence[str]] = None,
    ):
        self.run_id = run_id
        self.item_result_id = item_result_id
        self.request_id = request_id
        self.trace_id = trace_id or str(uuid.uuid4())
        self.provider = provider
        self.model = model
        self.document_ids = list(document_ids or [])
        self.selected_routes: List[str] = []
        self.stages: List[TraceStage] = []
        self._start_perf = time.perf_counter()
        self.total_latency_ms: float = 0.0
        self.error_info: Optional[str] = None

    def _get_relative_ms(self) -> float:
        return (time.perf_counter() - self._start_perf) * 1000.0

    @contextmanager
    def stage(self, stage_name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for timing an execution stage.
        Usage:
            with tracer.stage("retrieval", {"top_k": 5}):
                ...
        """
        start_ms = self._get_relative_ms()
        stg = TraceStage(stage_name, start_ms, metadata=metadata)
        self.stages.append(stg)
        err = None
        try:
            yield stg
        except Exception as e:
            err = str(e)
            self.error_info = err
            raise
        finally:
            end_ms = self._get_relative_ms()
            stg.finish(end_ms, error=err)

    def record_stage(
        self,
        stage_name: str,
        duration_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Directly record an already-measured stage duration."""
        stg = TraceStage(stage_name, 0.0, metadata=metadata)
        stg.duration_ms = duration_ms
        if error:
            stg.error = error
        self.stages.append(stg)

    def set_routes(self, routes: Sequence[str]) -> None:
        self.selected_routes = list(routes)

    def set_model(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model

    def set_error(self, error: str) -> None:
        self.error_info = error

    def finish(self) -> "EvaluationTracer":
        self.total_latency_ms = (time.perf_counter() - self._start_perf) * 1000.0
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "run_id": str(self.run_id) if self.run_id else None,
            "item_result_id": str(self.item_result_id) if self.item_result_id else None,
            "document_ids": self.document_ids,
            "selected_routes": self.selected_routes,
            "provider": self.provider,
            "model": self.model,
            "stages": [s.to_dict() for s in self.stages],
            "total_latency_ms": round(self.total_latency_ms, 2),
            "error_info": self.error_info,
        }

    def persist(self, session: Session) -> EvaluationTrace:
        """Persists the sanitized trace record into the evaluation_traces table."""
        self.finish()
        trace_record = EvaluationTrace(
            run_id=self.run_id,
            item_result_id=self.item_result_id,
            trace_id=self.trace_id,
            request_id=self.request_id,
            document_ids=self.document_ids,
            selected_routes=self.selected_routes,
            provider=self.provider,
            model=self.model,
            stages=[s.to_dict() for s in self.stages],
            total_latency_ms=round(self.total_latency_ms, 2),
            error_info=self.error_info,
            created_at=datetime.now(timezone.utc),
        )
        session.add(trace_record)
        return trace_record
