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

### Phase 09: Conversational RAG & Memory Engineering
- **Status:** Completed
- **Milestones:**
  - PostgreSQL persistence for sessions: `conversations`, `conversation_messages`, and `conversation_summaries` tables with strict monotonic sequence numbers and cascading cleanup.
  - 3 Distinct Memory Strategies: Sliding Window ($K$ turns), Token-Bounded Memory (dynamic context envelope budgeting), and Summary Memory (rolling LLM-generated summaries with sequence tracking).
  - History-Aware Query Rewriter: Coreference and pronoun resolution, standalone question detection, and ambiguity alerts.
  - Conversational Context Builder: Strict segregation between conversation memory and authoritative documentary context to prevent memory contamination.
  - Multi-Turn Conversational Orchestrator: Complete 14-step turn lifecycle supporting both synchronous and SSE token streaming.
  - Versioned REST API under `/api/v1/conversations` (sessions, turns, streaming, memory inspection, rewrite preview, strategy comparison).
  - React Conversational Lab interface with session manager, streaming chat, 6-tab diagnostic inspector, strategy comparison mode, and interactive learning center.
  - 12-Scenario synthetic multi-turn benchmark with 91.7% pass rate.
  - Comprehensive French tutorial and experimental reports (`docs/tutorials/09-conversational-rag.md`, `docs/experiments/09-conversational-experiments.md`, `docs/experiments/09-validation-report.md`).

### Phase 10: Agentic RAG & Tool-Orchestrated Retrieval
- **Status:** Completed
- **Milestones:**
  - 5 strictly allowlisted read-only tools: `dense_search`, `hybrid_search`, `read_document_section`, `inspect_document_metadata`, `inspect_evidence` with typed Pydantic JSON schemas.
  - 4 agent strategies: Single-Step (Tool Baseline), Iterative Retrieval (Plan-Act-Observe Loop), Plan-and-Execute (Decomposition), and Adaptive Routing (Dynamic Fast-Path).
  - Source-aware `EvidenceCollector`: automated chunk deduplication, stable `[S1]`, `[S2]` tags, document provenance tracking, and citation validation.
  - Deterministic stopping conditions and resource bounding: `MAX_ITERATIONS`, `MAX_TOOL_CALLS`, `TIME_BUDGET_EXCEEDED`, `NO_NEW_EVIDENCE`, `SUFFICIENT_EVIDENCE`.
  - Database trace persistence: `agent_traces` table with Alembic migration `0009_agent_traces`.
  - Versioned REST API under `/api/v1/agents` with real-time SSE streaming (`meta`, `step`, `token`, `citations`, `done`).
  - React Agentic RAG Lab UI with 6 tabs: Agent Playground, Agent Timeline, Tool Explorer, Strategy Comparison, Trace Inspector, and Learning Center.
  - 15-scenario synthetic multi-step benchmark suite with 93.3% pass rate.
  - French educational tutorial, experiment report, and validation report (`docs/tutorials/10-agentic-rag.md`, `docs/experiments/10-agentic-experiments.md`, `docs/experiments/10-validation-report.md`).

### Phase 11: Graph RAG & Knowledge Graph Engineering
- **Status:** Completed
- **Milestones:**
  - Relational Property Graph in PostgreSQL: `graph_entities`, `graph_relations`, `graph_evidence`, and `graph_extraction_runs` with Alembic migration `0010_graph_rag_tables`.
  - Extensible `GraphRepository` abstraction decoupled from proprietary graph databases.
  - Multi-engine entity and relation extraction: Rule-Based (regex + ontology dictionaries), NLP patterns, and Structured LLM with strict JSON schema validation.
  - Strict ontology and domain/range constraint enforcement (`GraphSchemaValidator`).
  - Entity Resolution: canonical slug normalizer, Levenshtein fuzzy similarity matching ($\ge 0.88$), alias mapping, and `human_review` safety state for ambiguous merges.
  - Bounded graph traversal using parameterized recursive SQL CTEs in PostgreSQL 16 with cycle detection (`NOT (target_id = ANY(path))`) and depth limit (`max_hops`).
  - Graph analytics: degree centrality, graph density, connected components, and label propagation community clustering.
  - 5 Retrieval Paradigms: Vector-Only Baseline, Entity-Based Graph, Multi-Hop Traversal, Hybrid Graph + Vector (Reciprocal Rank Fusion, $k=60$), and Community-Based Thematic Summarization.
  - Verifiable citation integrity: all graph entities and relations linked to exact source chunks in `graph_evidence` with stable `[S1]`, `[S2]` citation tags.
  - Dedicated React Graph RAG Lab with 7 interactive tabs: Builder, SVG Network Explorer, Entity Resolution, Query Lab, Communities, Strategy Compare, and Learning Center.
  - 15-scenario synthetic evaluation benchmark with 93.3% multi-hop resolution and 100% hybrid pass rate.
  - Comprehensive French tutorial, experiment report, and validation report (`docs/tutorials/11-graph-rag.md`, `docs/experiments/11-graph-rag-experiments.md`, `docs/experiments/11-validation-report.md`).

### Phase 12: Multimodal RAG & Document Intelligence
- **Status:** Completed
- **Milestones:**
  - Relational Property Graph & Multimodal tables in PostgreSQL: `document_page_assets`, `document_elements`, and `multimodal_embeddings` with Alembic migration `0011_multimodal_rag_tables`.
  - Scanned PDF detection heuristic ($< 40$ chars threshold) with high-fidelity $150\text{ DPI}$ pixmap rendering via PyMuPDF.
  - Spatial document layout analysis: title, header, footer, table, figure, caption classification with bounding boxes and 2-column reading order reconstruction.
  - Structural table extraction preserving row/column matrix coordinates and formatting clean Markdown tables vs. raw flattened text.
  - Tesseract OCR provider with language selection (`eng`, `fra`), word confidence tracking, and educational CER/WER Levenshtein metrics comparator.
  - Aligned cross-modal visual embeddings using `sentence-transformers/clip-ViT-B-32` in a shared 512-dimensional vector space with unit L2 normalization.
  - 5 Multimodal retrieval paradigms: OCR Text Retrieval, Image-to-Image, Text-to-Image (CLIP), Page-Level Retrieval, and Hybrid Multimodal RRF Fusion ($k=60$).
  - Grounded vision-language generation with verifiable citations (`[IMG_P1]`, `[PAGE_1]`, `[TBL_P1]`) and hallucination rejection.
  - React Multimodal Lab interface with 8 interactive tabs: Document Analyzer, OCR Lab, Table Intelligence, Image Embeddings, Multimodal Search, Vision Q&A, Strategy Compare, and Learning Center.
  - 15-scenario synthetic multimodal evaluation benchmark suite with 100% pass rate.
  - Comprehensive French tutorial, experiment report, and validation report (`docs/tutorials/12-multimodal-rag.md`, `docs/experiments/12-multimodal-experiments.md`, `docs/experiments/12-validation-report.md`).

### Phase 13: Production Hardening & Observability
- **Status:** Planned
- **Milestones:**
  - Semantic caching (Redis / pgvector) to reduce duplicate LLM generation costs.
  - Distributed rate limiting and concurrent request throttling.
  - OpenTelemetry distributed tracing and production observability.
  - Adversarial prompt injection defense and safety auditing.

