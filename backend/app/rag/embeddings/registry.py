import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from app.rag.embeddings.base import (
    BaseEmbeddingProvider,
    DeterministicMockEmbeddingProvider,
)
from app.rag.embeddings.providers.sentence_transformers import (
    LocalSentenceTransformerProvider,
)
from app.rag.embeddings.providers.api_provider import ApiEmbeddingProvider


@dataclass
class ModelSpecification:
    model_id: str
    display_name: str
    provider: str  # "sentence_transformers" | "mock" | "api"
    dimension: int
    max_seq_length: int
    language: str
    description: str
    query_prefix: str = ""
    passage_prefix: str = ""
    approx_size_mb: float = 0.0
    normalized_by_default: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmbeddingModelRegistry:
    """
    Registry for embedding models and providers.
    Provides lazy instantiation and prevents duplicate in-memory models.
    """

    def __init__(self):
        self._specs: Dict[str, ModelSpecification] = {}
        self._providers: Dict[str, BaseEmbeddingProvider] = {}
        self._register_default_models()

    def _register_default_models(self) -> None:
        # 1. Deterministic Mock Baseline (Always available, 0MB, for tests & instant dry-runs)
        self.register(
            ModelSpecification(
                model_id="mock-deterministic-v1",
                display_name="Deterministic Mock Embeddings",
                provider="mock",
                dimension=384,
                max_seq_length=256,
                language="Multilingual / Synthetic",
                description="Deterministic sinusoidal n-gram hashing baseline for instant offline testing and unit tests without downloading weights.",
                approx_size_mb=0.0,
                normalized_by_default=True,
            )
        )

        # 2. sentence-transformers/all-MiniLM-L6-v2 (Lightweight English baseline)
        self.register(
            ModelSpecification(
                model_id="sentence-transformers/all-MiniLM-L6-v2",
                display_name="All-MiniLM-L6-v2 (English)",
                provider="sentence_transformers",
                dimension=384,
                max_seq_length=256,
                language="English",
                description="Fast and robust English baseline. 384 dimensions, 5-layer transformer, ideal for rapid CPU prototyping.",
                approx_size_mb=80.0,
                normalized_by_default=True,
            )
        )

        # 3. sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Multilingual FR/EN baseline)
        self.register(
            ModelSpecification(
                model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                display_name="Paraphrase Multilingual MiniLM-L12-v2 (FR/EN)",
                provider="sentence_transformers",
                dimension=384,
                max_seq_length=128,
                language="Multilingual (French, English, 50+ languages)",
                description="Multilingual transformer trained on parallel translation corpora. Ideal for French & English cross-lingual retrieval.",
                approx_size_mb=470.0,
                normalized_by_default=True,
            )
        )

        # 4. intfloat/multilingual-e5-small (High-performance modern multilingual model with prefixes)
        self.register(
            ModelSpecification(
                model_id="intfloat/multilingual-e5-small",
                display_name="Multilingual-E5-Small (Asymmetric Prefixes)",
                provider="sentence_transformers",
                dimension=384,
                max_seq_length=512,
                language="Multilingual (100+ languages)",
                description="High-performance multilingual retrieval model. Requires 'query: ' for search questions and 'passage: ' for document chunks.",
                query_prefix="query: ",
                passage_prefix="passage: ",
                approx_size_mb=130.0,
                normalized_by_default=True,
            )
        )

    def register(self, spec: ModelSpecification) -> None:
        self._specs[spec.model_id] = spec

    def list_models(self) -> List[Dict]:
        models = []
        for model_id, spec in self._specs.items():
            models.append({
                "model_id": spec.model_id,
                "display_name": spec.display_name,
                "provider": spec.provider,
                "dimension": spec.dimension,
                "max_seq_length": spec.max_seq_length,
                "language": spec.language,
                "description": spec.description,
                "query_prefix": spec.query_prefix,
                "passage_prefix": spec.passage_prefix,
                "approx_size_mb": spec.approx_size_mb,
                "normalized_by_default": spec.normalized_by_default,
                "is_loaded": model_id in self._providers and self._providers[model_id].is_available,
            })
        return models

    def get_spec(self, model_id: str) -> ModelSpecification:
        if model_id not in self._specs:
            raise ValueError(
                f"Model '{model_id}' is not registered. Available: {list(self._specs.keys())}"
            )
        return self._specs[model_id]

    def get_provider(self, model_id: str) -> BaseEmbeddingProvider:
        if model_id in self._providers:
            return self._providers[model_id]

        spec = self.get_spec(model_id)

        if spec.provider == "mock":
            provider = DeterministicMockEmbeddingProvider(
                model_id=spec.model_id,
                dimension=spec.dimension,
                max_seq_length=spec.max_seq_length,
            )
        elif spec.provider == "sentence_transformers":
            provider = LocalSentenceTransformerProvider(
                model_id=spec.model_id,
                dimension=spec.dimension,
                max_seq_length=spec.max_seq_length,
                query_prefix=spec.query_prefix,
                passage_prefix=spec.passage_prefix,
            )
        elif spec.provider == "api":
            provider = ApiEmbeddingProvider(
                model_id=spec.model_id,
                dimension=spec.dimension,
                max_seq_length=spec.max_seq_length,
            )
        else:
            raise ValueError(f"Unsupported provider type '{spec.provider}'")

        self._providers[model_id] = provider
        return provider


# Singleton instance
embedding_registry = EmbeddingModelRegistry()
