import time
import abc
import re
from typing import List, Tuple, Dict, Any, Optional
from app.rag.retrieval.schemas import SearchResultItem


class BaseReranker(abc.ABC):
    """Abstract interface for second-stage cross-encoder rerankers."""

    @abc.abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[SearchResultItem],
        top_k: int = 5,
    ) -> Tuple[List[SearchResultItem], float, List[float]]:
        """
        Reranks a list of candidate passages conditioned on the input query.
        Returns:
            (reranked_items, latency_ms, raw_reranker_scores)
        """
        pass


class NoOpReranker(BaseReranker):
    """Pass-through reranker preserving candidate order without modification."""

    def rerank(
        self,
        query: str,
        candidates: List[SearchResultItem],
        top_k: int = 5,
    ) -> Tuple[List[SearchResultItem], float, List[float]]:
        selected = candidates[:top_k]
        scores = [item.similarity_score for item in selected]
        return selected, 0.0, scores


class MockReranker(BaseReranker):
    """
    Deterministic cross-similarity scorer for unit tests and offline benchmarking.
    Computes lexical intersection, exact phrase matching, and length penalty.
    """

    def rerank(
        self,
        query: str,
        candidates: List[SearchResultItem],
        top_k: int = 5,
    ) -> Tuple[List[SearchResultItem], float, List[float]]:
        t0 = time.perf_counter()
        query_words = set(re.findall(r"\b\w{3,}\b", query.lower()))

        scored_candidates = []
        for item in candidates:
            content_lower = item.content.lower()
            overlap_count = sum(1 for w in query_words if w in content_lower)
            # Bonus for exact query phrase match
            phrase_bonus = 2.0 if query.lower() in content_lower else 0.0
            # Length normalization
            len_penalty = min(1.0, len(item.content) / 300.0)
            score = (overlap_count * 1.5 + phrase_bonus) / (0.5 + len_penalty)
            scored_candidates.append((score, item))

        # Sort descending
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_scored = scored_candidates[:top_k]

        reranked_items: List[SearchResultItem] = []
        scores: List[float] = []

        for rank, (score, orig_item) in enumerate(top_scored, start=1):
            item_copy = orig_item.model_copy(deep=True)
            item_copy.similarity_score = score
            item_copy.similarity_label = f"CrossEncoder: {score:.4f}"
            item_copy.distance_label = f"Rerank-Rank: {rank}"
            reranked_items.append(item_copy)
            scores.append(score)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return reranked_items, latency_ms, scores


class LocalCrossEncoderReranker(BaseReranker):
    """
    Lightweight, CPU-optimized Cross-Encoder using TinyBERT or MiniLM.
    Features lazy loading on first query to ensure instant backend startup.
    Falls back gracefully to MockReranker if offline or model weights are missing.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2"):
        self.model_name = model_name
        self._model = None
        self._fallback = MockReranker()

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name, max_length=512)
            except Exception:
                # Fallback to mock reranker if model download fails or offline
                self._model = "fallback"

    def rerank(
        self,
        query: str,
        candidates: List[SearchResultItem],
        top_k: int = 5,
    ) -> Tuple[List[SearchResultItem], float, List[float]]:
        if not candidates:
            return [], 0.0, []

        self._load_model()
        if self._model == "fallback" or self._model is None:
            return self._fallback.rerank(query, candidates, top_k)

        t0 = time.perf_counter()
        pairs = [[query, item.content] for item in candidates]

        try:
            raw_scores = self._model.predict(pairs)
            scored = list(zip(raw_scores, candidates))
            scored.sort(key=lambda x: float(x[0]), reverse=True)
            top_scored = scored[:top_k]

            reranked: List[SearchResultItem] = []
            final_scores: List[float] = []

            for rank, (score, orig) in enumerate(top_scored, start=1):
                item_copy = orig.model_copy(deep=True)
                s_float = float(score)
                item_copy.similarity_score = s_float
                item_copy.similarity_label = f"TinyBERT: {s_float:.4f}"
                item_copy.distance_label = f"Rerank-Rank: {rank}"
                reranked.append(item_copy)
                final_scores.append(s_float)

            latency_ms = (time.perf_counter() - t0) * 1000.0
            return reranked, latency_ms, final_scores
        except Exception:
            # On unexpected failure, fall back cleanly
            return self._fallback.rerank(query, candidates, top_k)


class RerankerRegistry:
    """Registry providing singleton access to configured rerankers."""

    def __init__(self):
        tinybert = LocalCrossEncoderReranker("cross-encoder/ms-marco-TinyBERT-L-2-v2")
        self._rerankers: Dict[str, BaseReranker] = {
            "none": NoOpReranker(),
            "noop": NoOpReranker(),
            "mock": MockReranker(),
            "tinybert": tinybert,
            "cross-encoder/ms-marco-tinybert-l-2-v2": tinybert,
            "ms-marco-tinybert": tinybert,
            "ms-marco-tinybert-l-2-v2": tinybert,
        }

    def get_reranker(self, name: Optional[str] = "none") -> BaseReranker:
        key = (name or "none").lower().strip()
        if key in self._rerankers:
            return self._rerankers[key]
        if "tinybert" in key:
            return self._rerankers["tinybert"]
        if "mock" in key:
            return self._rerankers["mock"]
        return self._rerankers["none"]

    def list_rerankers(self) -> List[Dict[str, Any]]:
        return [
            {"id": "none", "name": "No-Op (Pass-through)", "description": "Preserves first-stage retrieval ordering without reranking overhead."},
            {"id": "tinybert", "name": "TinyBERT Cross-Encoder", "description": "Lightweight 2-layer CPU Cross-Encoder performing full joint self-attention."},
            {"id": "mock", "name": "Deterministic Mock Reranker", "description": "Fast keyword and phrase intersection scorer for unit testing and CI."},
        ]


reranker_registry = RerankerRegistry()
