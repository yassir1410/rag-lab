import React from 'react';
import { ComingSoon } from '../components/ui/ComingSoon';

export function RagChatPage() {
  return (
    <ComingSoon
      title="RAG Chat & Generation Studio"
      phaseNumber="10"
      phaseName="Context Synthesis, Citations & Streaming Responses"
      description="Interactive chat interface grounded in retrieved documents. Features real-time token streaming, exact source attribution citations, and prompt engineering templating."
      plannedFeatures={[
        "SSE Token Streaming",
        "Source document citations & page links",
        "Local LLM runtime (Ollama/vLLM) & API models",
        "Context window token budgeting",
        "Lost-in-the-middle context placement strategy",
      ]}
    />
  );
}

export function EmbeddingsExplorerPage() {
  return (
    <ComingSoon
      title="Embeddings Explorer"
      phaseNumber="04"
      phaseName="Vector Spaces, Dimensions & Dense Representations"
      description="Visual inspection of high-dimensional vector spaces. Generates dense embeddings using open-source SentenceTransformers and projects them into 2D/3D via UMAP/PCA."
      plannedFeatures={[
        "HuggingFace SentenceTransformers integration",
        "Embedding dimension & latency benchmarking",
        "Cosine similarity vs Euclidean distance metrics",
        "2D/3D semantic vector map projections",
      ]}
    />
  );
}

export function RetrievalLabPage() {
  return (
    <ComingSoon
      title="Retrieval Laboratory"
      phaseNumber="05 - 08"
      phaseName="Dense, Sparse, Hybrid & Reranking Engines"
      description="Multi-stage retrieval testing suite comparing pgvector HNSW/IVFFlat indexes, BM25 lexical keyword matching, Reciprocal Rank Fusion (RRF), and cross-encoder reranking."
      plannedFeatures={[
        "pgvector HNSW vs IVFFlat index benchmarks",
        "PostgreSQL Full-Text Search & BM25",
        "Hybrid Search with Reciprocal Rank Fusion",
        "Cross-Encoder Reranker integration",
      ]}
    />
  );
}

export function EvaluationPage() {
  return (
    <ComingSoon
      title="RAG Evaluation & Metrics"
      phaseNumber="11"
      phaseName="Quantitative Assessment & Benchmarks"
      description="RAG automated evaluation engine computing industry-standard metrics: Faithfulness, Answer Relevance, Context Precision, and Context Recall."
      plannedFeatures={[
        "Context Precision & Recall tracking",
        "Faithfulness / Hallucination detection",
        "Synthetic evaluation testset generation",
        "Automated regression testing",
      ]}
    />
  );
}

export function ExperimentsPage() {
  return (
    <ComingSoon
      title="Laboratory Experiments"
      phaseNumber="All"
      phaseName="Hands-on Architectural & Quality Studies"
      description="Interactive reproducible experiments designed to observe and measure how each RAG architectural choice impacts end-to-end performance."
      plannedFeatures={[
        "Experiment runner and comparison cards",
        "Reproducible benchmarks with synthetic data",
        "Parameter sensitivity analysis",
      ]}
    />
  );
}

export function SettingsPage() {
  return (
    <ComingSoon
      title="Laboratory Settings"
      phaseNumber="Cfg"
      phaseName="Environment & Infrastructure Configuration"
      description="Global configuration for vector stores, database connection pools, parser thresholds, and local model endpoints."
      plannedFeatures={[
        "Database pool parameters",
        "Upload directory management",
        "Model provider keys and endpoints",
      ]}
    />
  );
}
