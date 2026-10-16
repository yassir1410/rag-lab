import time
from typing import List, Dict, Any, Optional
import httpx
from app.rag.embeddings.base import BaseEmbeddingProvider, EmbeddingResult


class ApiEmbeddingProvider(BaseEmbeddingProvider):
    """
    HTTP API embedding provider compatible with OpenAI / Ollama / LocalAI endpoints.
    Does not require paid credentials by default (can connect to local Ollama on http://localhost:11434).
    """

    def __init__(
        self,
        model_id: str = "text-embedding-3-small",
        dimension: int = 1536,
        max_seq_length: int = 8192,
        base_url: str = "http://localhost:11434/v1",
        api_key: Optional[str] = None,
    ):
        self._model_id = model_id
        self._dimension = dimension
        self._max_seq_length = max_seq_length
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or "mock-key"

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
        if not texts:
            return EmbeddingResult(
                vectors=[],
                model_id=self._model_id,
                dimension=self._dimension,
                is_normalized=normalize_embeddings,
                total_tokens=0,
                latency_ms=0.0,
                metadata={"provider": "api"},
            )

        start_time = time.perf_counter()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        payload = {
            "input": texts,
            "model": self._model_id,
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                res = client.post(f"{self._base_url}/embeddings", json=payload, headers=headers)
                res.raise_for_status()
                data = res.json()
                data_items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
                vectors = [item["embedding"] for item in data_items]
        except Exception as e:
            raise RuntimeError(
                f"API embedding request failed against '{self._base_url}': {e}"
            )

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        total_tokens = sum(len(t.split()) for t in texts)

        return EmbeddingResult(
            vectors=vectors,
            model_id=self._model_id,
            dimension=len(vectors[0]) if vectors else self._dimension,
            is_normalized=normalize_embeddings,
            total_tokens=total_tokens,
            latency_ms=round(latency_ms, 3),
            metadata={"provider": "api", "base_url": self._base_url},
        )
