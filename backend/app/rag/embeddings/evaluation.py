import time
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from app.rag.embeddings.base import BaseEmbeddingProvider
from app.rag.embeddings.similarity import cosine_similarity, dot_product, euclidean_distance


@dataclass
class BenchmarkPair:
    category: str
    text_a: str
    text_b: str
    is_positive: bool
    language: str
    notes: str


BENCHMARK_PAIRS: List[BenchmarkPair] = [
    # 1. Paraphrase / Semantic Equivalence (English)
    BenchmarkPair(
        category="Paraphrase (EN)",
        text_a="FastAPI is a modern, high-performance web framework for building APIs with Python.",
        text_b="A fast, asynchronous Python web framework designed for creating REST APIs is FastAPI.",
        is_positive=True,
        language="en",
        notes="Identical meaning expressed with different syntax and vocabulary.",
    ),
    # 2. Unrelated / Negative Distractor (English)
    BenchmarkPair(
        category="Unrelated Distractor (EN)",
        text_a="FastAPI is a modern, high-performance web framework for building APIs with Python.",
        text_b="To bake a chocolate cake, preheat the oven to 180 degrees and mix two cups of flour.",
        is_positive=False,
        language="en",
        notes="Completely unrelated topic with zero semantic overlap.",
    ),
    # 3. Cross-Lingual Semantic Equivalence (French <-> English)
    BenchmarkPair(
        category="Cross-Lingual (FR/EN)",
        text_a="L'architecture de microservices améliore la scalabilité et l'isolation des pannes.",
        text_b="Microservice architecture improves system scalability and fault isolation.",
        is_positive=True,
        language="fr-en",
        notes="Direct cross-lingual semantic pair; tests multilingual embedding space alignment.",
    ),
    # 4. Cross-Lingual Negative Distractor (French <-> English)
    BenchmarkPair(
        category="Cross-Lingual Distractor (FR/EN)",
        text_a="L'architecture de microservices améliore la scalabilité et l'isolation des pannes.",
        text_b="The national soccer team won the tournament after a thrilling penalty shootout.",
        is_positive=False,
        language="fr-en",
        notes="Cross-lingual negative pair testing false alignment across languages.",
    ),
    # 5. Technical Jargon & Dense Vocabulary
    BenchmarkPair(
        category="Technical Jargon",
        text_a="High-dimensional dense vectors projected through UMAP preserve local neighborhood topology.",
        text_b="Dimensionality reduction using non-linear manifold approximation retains cluster structure.",
        is_positive=True,
        language="en",
        notes="High-density NLP and ML vocabulary.",
    ),
    # 6. Subtle Negation / Counter-Factual
    BenchmarkPair(
        category="Subtle Semantic Contrast",
        text_a="The service guarantees low latency and ninety-nine percent uptime.",
        text_b="The service suffers from high latency and frequent downtime.",
        is_positive=False,
        language="en",
        notes="Lexically similar words with antonymous meaning; tests contextual sensitivity.",
    ),
]


class EmbeddingBenchmarkSuite:
    """
    Evaluates embedding models across semantic categories, latency, throughput, and storage footprint.
    """

    def __init__(self, provider: BaseEmbeddingProvider):
        self.provider = provider

    def run_benchmark(self) -> Dict[str, Any]:
        """
        Executes the benchmark evaluation and returns objective metrics.
        """
        all_texts = []
        for pair in BENCHMARK_PAIRS:
            all_texts.extend([pair.text_a, pair.text_b])

        # Latency & throughput test
        start_time = time.perf_counter()
        encode_result = self.provider.encode(
            all_texts,
            batch_size=16,
            normalize_embeddings=True,
        )
        total_duration_ms = (time.perf_counter() - start_time) * 1000.0

        total_tokens = encode_result.total_tokens
        tokens_per_sec = (total_tokens / max(total_duration_ms, 0.001)) * 1000.0
        texts_per_sec = (len(all_texts) / max(total_duration_ms, 0.001)) * 1000.0

        # Evaluate pairs
        pair_results = []
        pos_similarities = []
        neg_similarities = []

        for idx, pair in enumerate(BENCHMARK_PAIRS):
            vec_a = encode_result.vectors[idx * 2]
            vec_b = encode_result.vectors[idx * 2 + 1]

            cos_sim = cosine_similarity(vec_a, vec_b)
            eucl_dist = euclidean_distance(vec_a, vec_b)
            dot_prod = dot_product(vec_a, vec_b)

            if pair.is_positive:
                pos_similarities.append(cos_sim)
            else:
                neg_similarities.append(cos_sim)

            pair_results.append({
                "category": pair.category,
                "text_a": pair.text_a,
                "text_b": pair.text_b,
                "is_positive": pair.is_positive,
                "language": pair.language,
                "notes": pair.notes,
                "cosine_similarity": round(cos_sim, 4),
                "euclidean_distance": round(eucl_dist, 4),
                "dot_product": round(dot_prod, 4),
            })

        avg_pos = sum(pos_similarities) / max(len(pos_similarities), 1)
        avg_neg = sum(neg_similarities) / max(len(neg_similarities), 1)
        margin = avg_pos - avg_neg

        # Storage footprint calculations (float32 = 4 bytes per dimension)
        dim = self.provider.dimension
        bytes_per_vector = dim * 4
        storage_10k_mb = (bytes_per_vector * 10_000) / (1024 * 1024)
        storage_100k_mb = (bytes_per_vector * 100_000) / (1024 * 1024)
        storage_1m_mb = (bytes_per_vector * 1_000_000) / (1024 * 1024)

        return {
            "model_id": self.provider.model_id,
            "dimension": dim,
            "total_benchmark_pairs": len(BENCHMARK_PAIRS),
            "quality_metrics": {
                "mean_positive_similarity": round(avg_pos, 4),
                "mean_negative_similarity": round(avg_neg, 4),
                "discriminative_margin": round(margin, 4),
                "separation_ratio": round(avg_pos / max(avg_neg, 0.001), 2),
            },
            "performance_metrics": {
                "total_duration_ms": round(total_duration_ms, 2),
                "avg_latency_per_text_ms": round(total_duration_ms / max(len(all_texts), 1), 2),
                "throughput_tokens_per_sec": round(tokens_per_sec, 1),
                "throughput_texts_per_sec": round(texts_per_sec, 1),
            },
            "storage_footprint": {
                "dimension": dim,
                "bytes_per_vector": bytes_per_vector,
                "storage_10k_vectors_mb": round(storage_10k_mb, 2),
                "storage_100k_vectors_mb": round(storage_100k_mb, 2),
                "storage_1m_vectors_mb": round(storage_1m_mb, 2),
            },
            "pairs": pair_results,
        }
