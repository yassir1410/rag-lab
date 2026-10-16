import time
from dataclasses import dataclass, field
from typing import Optional
from app.rag.generation.schemas import PipelineTimingMetrics


@dataclass
class PipelineLatencyTracker:
    """Tracks latency for each phase of the RAG pipeline."""
    embedding_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    context_assembly_ms: float = 0.0
    prompt_building_ms: float = 0.0
    llm_inference_ms: float = 0.0
    citation_validation_ms: float = 0.0
    model_load_latency_ms: Optional[float] = None
    _pipeline_start: float = field(default_factory=time.perf_counter)

    def to_schema(self) -> PipelineTimingMetrics:
        total = (time.perf_counter() - self._pipeline_start) * 1000.0
        return PipelineTimingMetrics(
            embedding_latency_ms=round(self.embedding_latency_ms, 2),
            retrieval_latency_ms=round(self.retrieval_latency_ms, 2),
            context_assembly_ms=round(self.context_assembly_ms, 2),
            prompt_building_ms=round(self.prompt_building_ms, 2),
            llm_inference_ms=round(self.llm_inference_ms, 2),
            citation_validation_ms=round(self.citation_validation_ms, 2),
            model_load_latency_ms=round(self.model_load_latency_ms, 2) if self.model_load_latency_ms is not None else None,
            total_pipeline_latency_ms=round(total, 2),
        )
