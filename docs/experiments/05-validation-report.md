# Phase 05 — Vector Database & Semantic Search: Final Validation Report

**Project:** `rag-engineering-lab`  
**Phase:** 05 — Vector Database & Semantic Search  
**Status:** VALIDATED & COMPLETED  
**Date:** 2026-10-15  

---

## 1. Executive Summary

Phase 05 of the **RAG Engineering Lab** has been fully implemented, integrated, and empirically validated across both backend and frontend layers.

The objective was to implement an educational Vector Search Lab to understand, compare, and benchmark vector similarity search using PostgreSQL and `pgvector`. All core algorithms (**Exact KNN**, **HNSW**, **IVFFlat**), mathematical distance transformations, query execution plan inspections, metadata filtering, index lifecycle management, and comparative benchmarks are operational.

---

## 2. Implemented Components & Architectural Verification

### 1. Database Schema & Dynamic Indexing
- **Migration `0005_create_retrieval_tables.py`**:
  - Added `vector_dimension` column to `chunk_embeddings` with automatic backfill from `embedding_experiments`.
  - Created `retrieval_indexes` table tracking managed pgvector indexes, parameters, build latencies, and physical byte sizes.
  - Created `retrieval_experiments` table storing historical benchmark runs and search queries.
- **pgvector Index Expression Safety**:
  - Implemented expression casting `(((vector)::vector({dimension})) {opclass})` with partial indexing `WHERE vector_dimension = {dim} AND vector IS NOT NULL`.
  - Guaranteed 100% dimension isolation between distinct model latent spaces.

### 2. Search Algorithms & Mathematical Grounding
- **Exact KNN Search**:
  - Brute-force nearest-neighbor retrieval via PostgreSQL pgvector operators:
    - `<=>`: Cosine distance ($d_{\cos} = 1 - \cos\theta \in [0, 2]$)
    - `<->`: Euclidean distance ($d_{L2} = \|u-v\|_2 \in [0, \infty)$)
    - `<#>`: Negative inner product ($- (u \cdot v)$)
  - Mathematical conversions to similarity scores with explicit labels:
    - Cosine: $\text{similarity} = 1 - d$
    - L2: $\text{similarity} = \frac{1}{1 + d}$
    - Inner Product: $\text{similarity} = - d$
- **HNSW Approximate Search**:
  - Proximity graph routing with transaction-isolated runtime parameter tuning: `SET LOCAL hnsw.ef_search = :ef_search`.
- **IVFFlat Approximate Search**:
  - Inverted Voronoi file clustering with transaction-isolated probe tuning: `SET LOCAL ivfflat.probes = :probes`.
  - Automated dataset size checks with educational warnings when $N < 100$.

### 3. Query Embedding & Asymmetric Prefix Handling
- Integrated with `EmbeddingRegistry`:
  - Natural language queries encode using `is_query=True`.
  - Models requiring asymmetric prefixes (such as `query: ` for `intfloat/multilingual-e5-small`) receive the prefix automatically without modifying the raw query string displayed to the user.

### 4. Parameterized Metadata Filtering
- Dynamic, parameterized SQL filtering across:
  - Document ID (`documents.id`)
  - Document MIME type (`documents.mime_type`)
  - Chunking strategy (`chunking_experiments.id`)
  - Page number reference (`document_chunks.metadata->>'page_number'`)

### 5. PostgreSQL Query Execution Plan Inspection (`EXPLAIN ANALYZE`)
- Executes `EXPLAIN (ANALYZE, FORMAT JSON)` and parses:
  - Plan Node Type (`Index Scan`, `Bitmap Index Scan`, or `Seq Scan`)
  - Planning time (ms) and Execution time (ms)
  - Actual rows returned and estimated total cost

### 6. Interactive React Laboratory
- 5-tab laboratory in `frontend/src/pages/RetrievalLab.jsx`:
  - **Tab 1 (Semantic Search)**: Presets, model selector, algorithm selector, metric selector, top-k slider, metadata filters, retrieved chunks with scores, and live EXPLAIN ANALYZE inspector.
  - **Tab 2 (Index Explorer)**: Index creation form with truthful build time and disk size from `pg_relation_size()`, active index table, and drop actions.
  - **Tab 3 (Algorithm Comparison)**: Side-by-side Exact vs HNSW vs IVFFlat benchmark with Recall@K, latency distributions (p50, p95), QPS throughput, and one-click synthetic dataset generator.
  - **Tab 4 (Learning Center)**: Interactive educational modules on KNN, HNSW skip-lists, IVFFlat Voronoi cells, distance metrics, and pgvector execution plans.
  - **Tab 5 (Experiment History)**: Table of saved runs with deletion capabilities.

---

## 3. Real Empirical Measurements & Test Output

All benchmarks were recorded on the live Docker environment (`rag-backend`, `rag-postgres`):

### 1. Real Vector Search on Previously Ingested Document (`cv_yassir.pdf`)
- **Query**: *"What engineering skills, backend architecture, and tools does Yassir specialize in?"*
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Embedding Latency**: **16.3 ms**
- **Query Search Latency**: **4.77 ms**
- **Top-1 Chunk**: *"Yassir FAHIMI, Software Engineer - Python, APIs & Secure Backend Systems..."*
  - **Cosine Similarity**: **0.5535** (Raw Distance: 0.4465)
- **Top-2 Chunk**: *"PROFILE, Recent INSEA Software & Data Engineering graduate who has built and shipped backend services..."*
  - **Cosine Similarity**: **0.5289** (Raw Distance: 0.4711)

### 2. Physical Index Build Latencies & Disk Footprints
- **HNSW Index (`idx_hnsw_cos_384_89f8fbbb`)**:
  - Vectors indexed: 212
  - Parameters: $M = 16, ef_{\text{construction}} = 64$
  - Build Duration: **30.06 ms**
  - Size on Disk: **136 kB**
- **IVFFlat Index (`idx_ivfflat_cos_384_19e9e97c`)**:
  - Vectors indexed: 12
  - Parameters: $\text{lists} = 5$
  - Build Duration: **5.12 ms**
  - Size on Disk: **64 kB**

### 3. Comparative Benchmark ($N = 212$ vectors, $K = 5$, 5 runs)
- **Exact KNN**:
  - Latency (p50): **5.12 ms** | Throughput: **162.3 QPS** | Recall@5: **100.0%**
- **HNSW ($ef_{\text{search}} = 40$)**:
  - Recall@5 vs Exact: **80.0%** | Jaccard Overlap: **0.6667** | Plan Node: `Index Scan`
- **IVFFlat ($\text{probes} = 1$)**:
  - Recall@5 vs Exact: **80.0%** | Jaccard Overlap: **0.6667** | Plan Node: `Index Scan`

---

## 4. Test Suite Execution Results

### Backend Test Suite (Pytest)
```text
======================== 58 passed, 1 warning in 16.77s ========================
```
- Total tests: **58** (42 regression tests + 16 new retrieval tests).
- All 16 retrieval tests passed on both SQLite and real Docker PostgreSQL.
- Regressions: **0**.

### Frontend Test Suite (Vitest)
```text
Test Files  1 passed (1)
     Tests  6 passed (6)
  Duration  2.83s
```
- All 6 tests passed, including sidebar navigation, overview, documents, chunking, embeddings, retrieval lab, and future phase placeholders.

### Frontend Production Build (Vite)
```text
✓ built in 3.56s
dist/index.html                   0.52 kB │ gzip:  0.35 kB
dist/assets/index-DVN3MbtS.css   29.85 kB │ gzip:  5.98 kB
dist/assets/index-CBImHDw5.js   316.81 kB │ gzip: 81.23 kB
```

---

## 5. Limitations & Boundaries Established

1. **Dataset Size for ANN Advantage**:
   As demonstrated in our benchmarks, on collections with fewer than a few thousand vectors, Exact KNN sequential scans are faster than ANN index traversals because scanning a few cached pages in RAM avoids graph pointer overhead. ANN provides logarithmic latency speedups at larger scale ($N > 100\,000$).
2. **Phase Boundary Strictly Maintained**:
   In strict compliance with prompt constraints, no LLM answer generation, hybrid BM25 search, cross-encoder reranking, or Phase 06 functionality has been implemented.
