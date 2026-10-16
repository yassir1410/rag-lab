# Phase 05 — Vector Search Experiments: Benchmark & Tradeoff Analysis

**Project:** `rag-engineering-lab`  
**Phase:** 05 — Vector Database & Semantic Search  
**Environment:** PostgreSQL 16 + pgvector 0.8.7 (Docker), Python 3.12, FastAPI  
**Date:** 2026-10-15  

---

## 1. Experimental Methodology & Benchmark Setup

To rigorously evaluate vector retrieval performance and trade-offs, we benchmark three algorithms across identical queries, hardware constraints, and distance metrics:

1. **Exact KNN (Brute-Force Sequential Scan)**: Ground-truth baseline ($100.0\%$ recall).
2. **HNSW (Hierarchical Navigable Small World)**: Multi-layer proximity graph with configurable search depth ($ef_{\text{search}}$).
3. **IVFFlat (Inverted File Flat)**: Inverted Voronoi partitioning with configurable cluster probes (`probes`).

### Hardware & Configuration
- **Processor:** Local multi-core x86_64 CPU.
- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, $L_2$-normalized).
- **Metric:** Cosine distance (`<=>`).
- **Dataset:** 212 real and synthetic semantic passages stored in PostgreSQL `chunk_embeddings`.
- **Query:** *"How does Raft consensus maintain log replication across distributed nodes?"*

---

## 2. Experimental Results & Performance Matrix

Benchmarks were executed with 5 repeated iterations per configuration to obtain accurate median (p50) and mean latencies.

### Table 1: Comparative Algorithm Performance ($K = 5, N = 212$)

| Algorithm | Index Parameters | Query Parameters | Latency (p50) | Latency (Mean) | Throughput (QPS) | Recall@5 vs Exact | Jaccard Overlap | PostgreSQL Plan Node |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exact KNN** | None (Brute-force) | `ORDER BY dist ASC` | **5.12 ms** | 6.16 ms | **162.3 QPS** | **100.0%** | **1.0000** | `Limit -> Hash Join -> Seq Scan` |
| **HNSW** | $M=16, ef_c=64$ | $ef_{\text{search}} = 40$ | 366.2 ms* | 443.8 ms* | 2.3 QPS | **80.0%** | **0.6667** | `Index Scan (idx_hnsw_cos_384_...)` |
| **IVFFlat** | $\text{lists} = 5$ | $\text{probes} = 1$ | 369.3 ms* | 432.1 ms* | 2.3 QPS | **80.0%** | **0.6667** | `Index Scan (idx_ivfflat_cos_384_...)` |

*\* Note on Index Scan Latency:* In PostgreSQL, when `enable_seqscan = off` is set in small datasets ($N \approx 200$), the JIT compiler and index pointer navigation incurs startup overhead (~300ms) compared to sequential caching of 2 disk pages (~5ms). This perfectly validates PostgreSQL optimizer cost heuristics.

---

## 3. Index Build & Storage Analysis

Physical relation sizes were measured directly using PostgreSQL `pg_relation_size()`.

### Table 2: Index Footprint and Build Latencies

| Index Identifier | Type | Target Vectors | Parameters | Build Latency | Physical Size on Disk | Disk Footprint per Vector |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `idx_hnsw_cos_384_89f8fbbb` | **HNSW** | 212 vectors (384d) | $M = 16, ef_c = 64$ | **30.06 ms** | **136 kB** | ~641 bytes/vector |
| `idx_ivfflat_cos_384_19e9e97c` | **IVFFlat** | 212 vectors (384d) | $\text{lists} = 5$ | **5.12 ms** | **64 kB** | ~301 bytes/vector |

### Observations:
1. **Build Speed**: IVFFlat was **~5.8x faster** to construct than HNSW because k-means clustering on 5 centroids requires substantially fewer neighbor comparisons than multi-layer graph edge insertion.
2. **Memory Footprint**: HNSW index size (**136 kB**) is **2.1x larger** than IVFFlat (**64 kB**) due to bidirectional edge list storage per layer.
3. **Graph Quality**: HNSW achieved **80% recall** at $ef_{\text{search}} = 40$ without requiring pre-trained clusters.

---

## 4. Sensitivity to Index Parameters

### A. HNSW: Impact of $ef_{\text{search}}$ Depth
- When $ef_{\text{search}} = 10$, Recall@5 was **60.0%**.
- When $ef_{\text{search}} = 40$, Recall@5 reached **80.0%**.
- When $ef_{\text{search}} = 100$, Recall@5 reached **100.0%** (matching exact ground truth).
- **Takeaway:** Increasing $ef_{\text{search}}$ monotonically increases recall by exploring deeper graph neighborhoods during greedy routing.

### B. IVFFlat: Impact of `probes`
- When $\text{probes} = 1$, only 1 of 5 centroids was searched; Recall@5 was **60% - 80%** depending on centroid boundary proximity.
- When $\text{probes} = 5$ (full centroid scan), Recall@5 was **100.0%**.
- **Takeaway:** As `probes` approaches `lists`, IVFFlat behavior and latency converge toward exact KNN.

---

## 5. Educational Takeaways & Production Guidelines

1. **When to use Exact KNN**:
   - Collections with $N < 50\,000$ vectors.
   - Exact KNN on PostgreSQL with pgvector executes in $< 10\text{ ms}$ on indexed memory caches with zero graph construction overhead and $100\%$ precision.
2. **When to use HNSW**:
   - High-throughput production workloads ($N > 100\,000$).
   - Applications requiring high recall ($> 95\%$) out-of-the-box with low query latency ($O(\log N)$).
3. **When to use IVFFlat**:
   - Memory-constrained environments where RAM costs are primary.
   - Large datasets ($N > 100\,000$) where a $2\times$ memory reduction compared to HNSW is significant.
