import os
import time
from typing import List, Dict, Any, Optional
from app.rag.embeddings.base import BaseEmbeddingProvider, EmbeddingResult


class LocalSentenceTransformerProvider(BaseEmbeddingProvider):
    """
    Local CPU neural embedding provider using SentenceTransformers.
    Optimized for CPU inference with thread capping, lazy-loading, and prefix support.
    """

    def __init__(
        self,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int = 384,
        max_seq_length: int = 256,
        query_prefix: str = "",
        passage_prefix: str = "",
        cache_dir: Optional[str] = None,
    ):
        self._model_id = model_id
        self._dimension = dimension
        self._max_seq_length = max_seq_length
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._cache_dir = cache_dir or os.getenv("HF_HOME", "/tmp/huggingface_cache")

        self._model = None
        self._load_error: Optional[str] = None
        self._is_loaded = False

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @property
    def is_available(self) -> bool:
        return self._is_loaded and self._model is not None

    def _ensure_model(self) -> None:
        """Lazy loader for SentenceTransformer model."""
        if self._is_loaded:
            if self._load_error:
                raise RuntimeError(
                    f"Failed to load embedding model '{self._model_id}': {self._load_error}"
                )
            return

        try:
            import torch
            # Prevent PyTorch from consuming all CPU cores on laptops
            cpu_threads = min(4, max(1, os.cpu_count() or 2))
            torch.set_num_threads(cpu_threads)

            from sentence_transformers import SentenceTransformer

            try:
                # Fast path: load from local cache without network lookup
                self._model = SentenceTransformer(
                    self._model_id,
                    device="cpu",
                    cache_folder=self._cache_dir,
                    local_files_only=True,
                )
            except Exception:
                # Fallback to download if online
                self._model = SentenceTransformer(
                    self._model_id,
                    device="cpu",
                    cache_folder=self._cache_dir,
                    local_files_only=False,
                )
            # Override max_seq_length if model specifies
            if hasattr(self._model, "max_seq_length"):
                self._max_seq_length = self._model.max_seq_length
            self._is_loaded = True
        except Exception as e:
            self._load_error = str(e)
            self._is_loaded = True
            raise RuntimeError(
                f"Failed to load embedding model '{self._model_id}': {e}"
            )

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        is_query: bool = False,
    ) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(
                vectors=[],
                model_id=self._model_id,
                dimension=self._dimension,
                is_normalized=normalize_embeddings,
                total_tokens=0,
                latency_ms=0.0,
                metadata={"batch_size": batch_size, "is_query": is_query},
            )

        self._ensure_model()
        start_time = time.perf_counter()

        # Apply query/passage prefix if configured (e.g. for e5 models)
        prefix = self._query_prefix if is_query else self._passage_prefix
        prefixed_texts = [f"{prefix}{t}" if prefix else t for t in texts]

        # SentenceTransformers encode
        raw_embeddings = self._model.encode(
            prefixed_texts,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        vectors: List[List[float]] = [
            [float(val) for val in row] for row in raw_embeddings
        ]

        # Calculate approximate token count
        total_tokens = sum(len(t.split()) for t in texts)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return EmbeddingResult(
            vectors=vectors,
            model_id=self._model_id,
            dimension=len(vectors[0]) if vectors else self._dimension,
            is_normalized=normalize_embeddings,
            total_tokens=total_tokens,
            latency_ms=round(latency_ms, 3),
            metadata={
                "batch_size": batch_size,
                "is_query": is_query,
                "prefix_used": prefix,
                "device": "cpu",
            },
        )
