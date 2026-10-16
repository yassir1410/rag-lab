import time
from typing import Dict, Any, List
from pydantic import BaseModel, Field


class ConversationalPipelineLatencyTracker:
    """Tracks latency for each discrete stage of the 14-step conversational RAG pipeline."""

    def __init__(self):
        self.t_start = time.perf_counter()
        self.history_loading_ms: float = 0.0
        self.memory_selection_ms: float = 0.0
        self.query_rewriting_ms: float = 0.0
        self.embedding_ms: float = 0.0
        self.retrieval_ms: float = 0.0
        self.context_assembly_ms: float = 0.0
        self.llm_generation_ms: float = 0.0
        self.citation_validation_ms: float = 0.0
        self.persistence_ms: float = 0.0
        self.summary_refresh_ms: float = 0.0

    @property
    def total_latency_ms(self) -> float:
        return (time.perf_counter() - self.t_start) * 1000.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "history_loading_ms": round(self.history_loading_ms, 2),
            "memory_selection_ms": round(self.memory_selection_ms, 2),
            "query_rewriting_ms": round(self.query_rewriting_ms, 2),
            "embedding_ms": round(self.embedding_ms, 2),
            "retrieval_ms": round(self.retrieval_ms, 2),
            "context_assembly_ms": round(self.context_assembly_ms, 2),
            "llm_generation_ms": round(self.llm_generation_ms, 2),
            "citation_validation_ms": round(self.citation_validation_ms, 2),
            "persistence_ms": round(self.persistence_ms, 2),
            "summary_refresh_ms": round(self.summary_refresh_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
        }


def compute_keyword_overlap(predicted_query: str, expected_query: str) -> float:
    """Computes Jaccard word-level overlap between predicted standalone query and expected query."""
    import re
    words_pred = set(re.findall(r"\b\w{3,}\b", predicted_query.lower()))
    words_exp = set(re.findall(r"\b\w{3,}\b", expected_query.lower()))
    if not words_exp:
        return 1.0 if not words_pred else 0.0
    intersection = words_pred.intersection(words_exp)
    union = words_pred.union(words_exp)
    return float(len(intersection)) / float(len(union)) if union else 1.0
