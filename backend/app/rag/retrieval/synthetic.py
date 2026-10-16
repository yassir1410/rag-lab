import uuid
import time
from typing import List, Dict, Any
from sqlalchemy.orm import Session

from app.models.embeddings import EmbeddingExperiment, ChunkEmbedding
from app.rag.embeddings.registry import embedding_registry

SAMPLE_BENCHMARK_TOPICS = [
    (
        "Distributed Databases & Consensus",
        [
            "Raft consensus protocol maintains a replicated log across distributed state machines using leader election and log replication.",
            "Paxos consensus guarantees safety in asynchronous networks under arbitrary node crashes but requires two round-trips for quorum commits.",
            "Spanner uses TrueTime API with GPS and atomic clocks to provide externally consistent linearizable transactions across global datacenters.",
            "Multi-Raft divides partition ranges into independent consensus groups to scale throughput horizontally across cluster nodes.",
            "Vector databases utilize inverted file clustering to divide high-dimensional vector spaces into coarse quantizer Voronoi cells.",
            "Write-ahead logging (WAL) guarantees durability by appending sequential mutation records before committing modifications to table pages.",
        ],
    ),
    (
        "Vector Indexes & High-Dimensional Geometry",
        [
            "Hierarchical Navigable Small World (HNSW) graphs organize vectors into hierarchical layers with exponentially decreasing edge density.",
            "The curse of dimensionality causes distances between high-dimensional vectors to become equidistant, degrading partition efficiency.",
            "Cosine similarity measures the angular alignment between normalized embedding vectors irrespective of absolute vector magnitude.",
            "IVFFlat reduces search candidate sets by assigning queries to nearest Voronoi centroid clusters through inverted file lists.",
            "In inner product metric, pgvector returns negative dot products so ascending sorting returns top maximum similarity items.",
            "HNSW ef_search parameter balances greedy routing query latency with recall precision across proximity graph layers.",
        ],
    ),
    (
        "Deep Learning & Transformer NLP",
        [
            "Self-attention mechanism computes pairwise token compatibility matrices using scaled dot-product query-key projection weights.",
            "BERT uses masked language modeling and next sentence prediction objectives to learn bidirectional contextual token representations.",
            "SentenceTransformers apply mean pooling over output hidden states followed by contrastive learning to map sentences into dense semantic vector spaces.",
            "Asymmetric retrieval models like E5 require distinct query and passage prefixes to specialize latent vector representations.",
            "Subword tokenizers like Byte-Pair Encoding (BPE) split rare words into frequent character n-grams to eliminate out-of-vocabulary tokens.",
            "Cross-encoders compute all-to-all attention between query and candidate documents, yielding high ranking accuracy at higher computational cost.",
        ],
    ),
    (
        "Cloud Systems & Modern Architecture",
        [
            "Kubernetes orchestrates containerized workloads across compute clusters using declarative control loops and controller managers.",
            "Event-driven architecture decouples services through asynchronous pub-sub message brokers like Apache Kafka or RabbitMQ.",
            "Command Query Responsibility Segregation (CQRS) separates read models from write command pipelines to optimize query scaling.",
            "Circuit breakers prevent cascading failures in distributed microservices by failing fast when downstream services become unresponsive.",
            "Domain-Driven Design establishes bounded contexts and ubiquitous language to align business domains with software components.",
            "Observability platforms collect distributed traces, structured metrics, and logs to diagnose distributed request latency bottlenecks.",
        ],
    ),
]


def generate_synthetic_benchmark_corpus(
    session: Session,
    model_id: str,
    target_count: int = 300,
) -> Dict[str, Any]:
    """
    Generates a reproducible collection of synthetic semantic chunks and embeddings
    to enable meaningful HNSW and IVFFlat benchmarking (N >= 300).
    """
    spec = embedding_registry.get_spec(model_id)
    provider = embedding_registry.get_provider(model_id)

    # Generate target_count text passages by permuting and augmenting topic templates
    passages: List[str] = []
    topic_labels: List[str] = []

    idx = 0
    while len(passages) < target_count:
        for topic_name, sentences in SAMPLE_BENCHMARK_TOPICS:
            for s in sentences:
                variant = f"[{topic_name} - Part {idx // len(sentences) + 1}] {s}"
                passages.append(variant)
                topic_labels.append(topic_name)
                if len(passages) >= target_count:
                    break
            if len(passages) >= target_count:
                break
        idx += 1

    start_time = time.perf_counter()
    encode_result = provider.encode(passages, batch_size=64, normalize_embeddings=True, is_query=False)
    embedding_latency = (time.perf_counter() - start_time) * 1000.0

    # Create synthetic embedding experiment
    exp = EmbeddingExperiment(
        name=f"Synthetic Benchmark Corpus (N={len(passages)})",
        document_id=None,
        chunking_experiment_id=None,
        model_id=model_id,
        provider=spec.provider,
        vector_dimension=spec.dimension,
        is_normalized=True,
        parameters={"synthetic": True, "target_count": target_count},
        metrics={
            "vector_count": len(passages),
            "generation_latency_ms": round(embedding_latency, 2),
        },
    )
    session.add(exp)
    session.commit()
    session.refresh(exp)

    # Insert chunk_embeddings
    chunk_rows = []
    for i, (text_content, vec, topic) in enumerate(zip(passages, encode_result.vectors, topic_labels)):
        row = ChunkEmbedding(
            experiment_id=exp.id,
            chunk_id=None,
            sequence_index=i,
            text_snippet=text_content,
            token_count=len(text_content.split()),
            vector=vec,
            vector_dimension=spec.dimension,
        )
        chunk_rows.append(row)

    session.bulk_save_objects(chunk_rows)
    session.commit()

    return {
        "experiment_id": str(exp.id),
        "name": exp.name,
        "model_id": model_id,
        "vector_dimension": spec.dimension,
        "count": len(passages),
        "embedding_latency_ms": round(embedding_latency, 2),
    }
