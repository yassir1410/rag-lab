import hashlib
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class EmbeddingResult:
    """Represents the output of an embedding generation operation."""
    vectors: List[List[float]]
    model_id: str
    dimension: int
    is_normalized: bool
    total_tokens: int
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseEmbeddingProvider(ABC):
    """
    Abstract Base Class for all embedding providers (local, API, mock).
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Unique model identifier (e.g. 'sentence-transformers/all-MiniLM-L6-v2')."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Output embedding vector dimension."""
        pass

    @property
    @abstractmethod
    def max_seq_length(self) -> int:
        """Maximum context length in tokens."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether the model and its dependencies are loaded and ready."""
        pass

    @abstractmethod
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        is_query: bool = False,
    ) -> EmbeddingResult:
        """
        Encodes a list of strings into dense embedding vectors.
        """
        pass


class DeterministicMockEmbeddingProvider(BaseEmbeddingProvider):
    """
    Deterministic embedding provider using MD5 word n-gram hashing and sinusoidal projection.
    Provides fast, deterministic unit test vectors without network downloads or GPU/CPU strain.
    """

    def __init__(self, model_id: str = "mock-deterministic-v1", dimension: int = 384, max_seq_length: int = 256):
        self._model_id = model_id
        self._dimension = dimension
        self._max_seq_length = max_seq_length

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
        return True

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        is_query: bool = False,
    ) -> EmbeddingResult:
        start_time = time.perf_counter()
        vectors: List[List[float]] = []
        total_tokens = 0

        for text in texts:
            words = text.lower().split()
            total_tokens += len(words)
            vec = [0.0] * self._dimension

            if not text.strip():
                # Zero vector for blank input
                vectors.append(vec)
                continue

            for w_idx, word in enumerate(words):
                h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
                for d in range(self._dimension):
                    # Combine hash with harmonic sinusoidal wave for continuous smooth semantics
                    phase = (h % 997) * 0.01 + (d * 0.1) + (w_idx * 0.05)
                    vec[d] += math.sin(phase)

            if normalize_embeddings:
                norm = math.sqrt(sum(x * x for x in vec))
                if norm > 1e-12:
                    vec = [x / norm for x in vec]

            vectors.append(vec)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return EmbeddingResult(
            vectors=vectors,
            model_id=self._model_id,
            dimension=self._dimension,
            is_normalized=normalize_embeddings,
            total_tokens=total_tokens,
            latency_ms=round(latency_ms, 3),
            metadata={"batch_size": batch_size, "is_query": is_query, "provider": "mock"},
        )
