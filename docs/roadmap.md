# RAG Engineering Curriculum Roadmap (16 Phases)

This roadmap outlines the complete, pedagogical progression of the **RAG Engineering Lab**, from raw foundation to production-grade agentic and multimodal systems.

---

### Phase 01: Project Foundation & Architecture
- **Status:** Completed
- **Milestones:**
  - Modular monorepo layout (backend, frontend, docs, sample_documents).
  - FastAPI application with lightweight `/api/v1/health` and deep `/api/v1/health/ready`.
  - PostgreSQL 16 with native `pgvector` extension.
  - SQLAlchemy 2 session management and Alembic migrations.
  - Docker Compose multi-service containerization with health checks.
  - Plain React JavaScript frontend with Vite and Tailwind CSS.

### Phase 02: Document Ingestion & Parsing
- **Status:** Completed
- **Milestones:**
  - Multi-format parsers: PDF (PyMuPDF), DOCX (python-docx), TXT (UTF-8/BOM), Markdown (.md).
  - File validator with magic-byte verification, size limits, and path traversal protection.
  - SHA-256 duplicate detection and enforcement.
  - Non-destructive `TextNormalizer` with educational comparison metrics.
  - REST endpoints under `/api/v1/documents`.
  - Frontend drag-and-drop upload and **Educational Learning View**.

### Phase 03: Text Chunking Strategies
- **Status:** Completed
- **Milestones:**
  - 6 actual chunking algorithms: Fixed-Size Window, Recursive Character, Sentence-Based (Bilingual FR/EN), Semantic Similarity with breakpoint detection, Parent-Child (Hierarchical), and Structure-Aware (preserving tables and code).
  - BPE token-level counting and character-offset mapping.
  - Objective metrics engine: token distribution, observed overlap, duplicated text ratio, and adjacent semantic similarities.
  - PostgreSQL models and Alembic migration `0003_create_chunking_tables.py` for reproducible experiments.
  - REST endpoints under `/api/v1/chunking` (preview, experiments, compare).
  - Interactive React Chunking Lab with sliders, source boundary mapping, token histogram charts, and side-by-side strategy comparison.

### Phase 04: Embeddings & Vector Space Lab
- **Status:** Completed
- **Milestones:**
  - Dense embedding generation using open-source SentenceTransformers on CPU (`all-MiniLM-L6-v2`, `paraphrase-multilingual-MiniLM-L12-v2`, `multilingual-e5-small`).
  - Model registry with lazy loading, device capping, and query/passage asymmetric prefixes.
  - Tokenization inspector analyzing word vs. character vs. subword tokens, token IDs, and context length truncation limits.
  - Pure mathematical vector utilities: Cosine Similarity, Dot Product, Euclidean Distance, and L2 Normalization.
  - Dimensionality reduction engine: PCA (variance ratio), t-SNE (perplexity adjustment), and UMAP 2D semantic maps with small-dataset safeguards.
  - Evaluation benchmark suite evaluating semantic pairs, cross-lingual FR/EN alignment, discriminative margins ($\Delta$), throughput, and vector storage footprint.
  - PostgreSQL persistence: `embedding_experiments` and `chunk_embeddings` tables with Alembic migration `0004_create_embeddings_tables.py`.
  - Interactive Embeddings Explorer in plain React JavaScript with Learning Mode and Expert Mode.

### Phase 05: Vector Database & Semantic Search
- **Status:** Completed
- **Milestones:**
  - High-dimensional vector similarity search in PostgreSQL using pgvector (`<=>`, `<->`, `<#>`).
  - Indexing strategies comparison: Exact KNN (sequential scan), IVFFlat (inverted Voronoi cells), and HNSW (Hierarchical Navigable Small World).
  - Benchmark Correctness Audit: isolated session benchmarking, elimination of LLVM JIT overhead (cutting ANN latency by 98%), plan node type verification.
  - Interactive Retrieval Lab in React with live EXPLAIN ANALYZE execution plan inspector.

### Phase 06: Naive RAG & Grounded LLM Generation
- **Status:** Completed
- **Milestones:**
  - End-to-end RAG pipeline: Query -> Vector Retrieval -> Context Assembly -> Prompt Construction -> LLM Inference -> Citation Validation.
  - Local-first LLM provider architecture with Ollama (`qwen2.5:3b`) default, optional cloud OpenAI/Gemini, and test Mock provider.
  - ContextBuilder with deduplication, stable `[S1]`, `[S2]` citation tags, and token budget allocation.
  - 4 configurable prompt templates (basic, grounded, concise, detailed) with untrusted context defense notice.
  - CitationManager parsing bracketed references, validating against context, and detecting hallucinations.
  - Real-time Server-Sent Events (SSE) token streaming via FastAPI.
  - Interactive React RAG Chat studio with Learning Mode, Sources Panel, and Pipeline Latency Inspector.

### Phase 07: RAG Evaluation & Quality Benchmarking
- **Status:** Completed
- **Milestones:**
  - Golden Dataset management system with human-in-the-loop review workflows in PostgreSQL.
  - Pure Python retrieval metrics suite: Precision@K, Recall@K, Hit Rate@K, MRR, MAP@K, and nDCG@K (binary & graded).
  - Claim-level answer faithfulness evaluation distinguishing supported, unsupported, and contradicted claims.
  - Answer relevance evaluation assessing question answering completeness and correct refusal handling.
  - Citation quality evaluation measuring ID validity, citation support precision, and coverage.
  - Multi-stage EvaluationRunner orchestrating retrieval, generation, and LLM-as-a-judge pipelines.
  - High-resolution latency percentile tracking (mean, p50, p95, p99) and model token pricing registry (USD).
  - Interactive React Evaluation Studio with KPIs scorecard, dataset manager, question inspector, side-by-side run comparison, and interactive educational learning mode sandboxes.
  - 12 reproducible evaluation experiments and comprehensive validation report.

### Phase 08: Advanced Retrieval, Hybrid Search & RAG Optimization
- **Status:** Completed
- **Milestones:**
  - PostgreSQL native full-text search with generated `tsv_content` column, `ts_rank_cd`, and GIN index.
  - Hybrid retrieval combining dense pgvector and sparse lexical full-text search across 8 configurable architectures.
  - Reciprocal Rank Fusion (RRF) with configurable rank constant $k$ (default 60) and normalized linear combination.
  - Cross-Encoder Reranking interface with CPU-optimized TinyBERT (`cross-encoder/ms-marco-TinyBERT-L-2-v2`), mock, and no-op options.
  - Advanced Query Processing with entity/keyword extraction and multi-query expansion.
  - Context Optimization with contiguous neighbor expansion, hash/Jaccard redundancy reduction, and adaptive score cutoff.
  - Reliable evaluation upgrades: graded relevance labels (0/1/2), revision history tracking, atomic claim extractor with boilerplate stripping, and separate unanswerable metrics.
  - Interactive React updates: 8-strategy selector, dynamic tuning controls, 5-stage pipeline visualizer, candidate progression audit table, and interactive RRF simulator in Retrieval Lab; Evidence Auditor modal and unanswerable KPI cards in Evaluation Studio.
  - 12 reproducible experiments on golden CV benchmark showing Recall@5 boost from 56.2% to 60.4% with Adaptive Context Selection.
  - Comprehensive French pedagogical documentation and validation reports (`docs/tutorials/08-advanced-retrieval.md`, `docs/experiments/08-retrieval-optimization.md`, `docs/experiments/08-validation-report.md`).

### Phase 09: Context Construction & Prompt Engineering
- **Status:** Planned
- **Milestones:**
  - Dynamic prompt templates with token budgeting and overflow trimming.
  - Mitigation of "Lost in the Middle" phenomenon (placing crucial context at the start/end).
  - Document metadata injection (titles, page numbers, timestamps).

### Phase 10: Generation & Response Synthesis
- **Status:** Planned
- **Milestones:**
  - Local LLM inference via Ollama / vLLM and API-based models.
  - Server-Sent Events (SSE) real-time token streaming.
  - Exact source attribution and interactive citation links to document pages.

### Phase 11: RAG Evaluation Framework
- **Status:** Planned
- **Milestones:**
  - Quantitative metrics suite: Context Precision, Context Recall, Faithfulness, and Answer Relevance.
  - Automated synthetic test dataset generation using LLM self-questioning.
  - Continuous integration regression tests on retrieval accuracy.

### Phase 12: Query Transformation & Expansion
- **Status:** Planned
- **Milestones:**
  - Hypothetical Document Embeddings (HyDE).
  - Multi-query expansion and query rewriting.
  - Step-back prompting for abstract conceptual queries.

### Phase 13: Advanced Routing & Self-Reflective / Corrective RAG
- **Status:** Planned
- **Milestones:**
  - Adaptive intent classification and router to specialized vector indexes.
  - Corrective RAG (CRAG): evaluating retrieved document confidence and triggering web search fallback.
  - Self-RAG: reflection tokens evaluating relevance before generation.

### Phase 14: Agentic RAG
- **Status:** Planned
- **Milestones:**
  - Multi-step tool use, reasoning, and iterative query formulation (ReAct).
  - Graph-based agent orchestration (LangGraph / custom state machine).
  - Dynamic human-in-the-loop verification for low-confidence queries.

### Phase 15: Graph RAG & Knowledge Graphs
- **Status:** Planned
- **Milestones:**
  - Entity and relation extraction from ingested sections.
  - Graph database traversal (Neo4j / NetworkX).
  - Hybrid vector + graph multi-hop relational retrieval.

### Phase 16: Multimodal RAG & Production Hardening
- **Status:** Planned
- **Milestones:**
  - Multimodal document ingestion (images, diagrams, charts via vision models and CLIP embeddings).
  - Semantic caching (Redis / pgvector) to reduce duplicate LLM generation costs.
  - Distributed rate limiting, observability with OpenTelemetry, and security hardening.
