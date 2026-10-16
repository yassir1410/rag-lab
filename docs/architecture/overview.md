# Architecture Overview — RAG Engineering Lab

## 1. Vision & Core Philosophy

The **RAG Engineering Lab** is an educational, production-ready environment designed to systematically explore and benchmark every tier of Retrieval-Augmented Generation (RAG).

Unlike standard tutorials that treat RAG as a "black-box" call to an all-in-one framework, this laboratory is built from first principles with a modular monorepo structure. Each stage—ingestion, chunking, embedding, indexing, retrieval, reranking, synthesis, and evaluation—is implemented as an isolated, inspectable subsystem with strict interfaces.

```
                          ┌────────────────────────┐
                          │     React Frontend     │
                          │   (JS / Vite / TW)     │
                          └───────────┬────────────┘
                                      │ HTTP / JSON
                                      ▼
                          ┌────────────────────────┐
                          │   FastAPI App (v1)     │
                          │ (Lifespan, Middleware) │
                          └───────────┬────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
         ▼                            ▼                            ▼
┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
│ Ingestion Engine│          │ Database Layer  │          │ Storage Engine  │
│ - FileValidator │          │ - SQLAlchemy 2  │          │ - Disk storage  │
│ - Multi-Parsers │          │ - PostgreSQL 16 │          │ - UUID safe     │
│ - TextNormalizer│          │ - pgvector      │          │   filenames     │
└─────────────────┘          └─────────────────┘          └─────────────────┘
```

---

## 2. Directory Layout & Layer Responsibilities

```
rag-engineering-lab/
├── backend/
│   ├── app/
│   │   ├── api/             # REST endpoints (health, documents, etc.)
│   │   ├── core/            # Centralized settings (pydantic-settings) & logging
│   │   ├── db/              # SQLAlchemy session, engine, DeclarativeBase
│   │   ├── models/          # ORM database entities (Document, DocumentSection)
│   │   ├── schemas/         # Pydantic v2 validation and response models
│   │   ├── services/        # Business logic & transactional operations
│   │   └── rag/             # Educational RAG subsystems
│   │       ├── ingestion/   # Parsers (PDF, DOCX, TXT, MD), normalizer, validator
│   │       ├── chunking/    # Phase 03 - Splitting strategies
│   │       ├── embeddings/  # Phase 04 - Dense vector generation
│   │       ├── retrieval/   # Phases 05-08 - Dense, sparse, hybrid & rerankers
│   │       ├── generation/  # Phases 09-10 - LLM context synthesis & streaming
│   │       └── evaluation/  # Phase 11 - RAG metrics & benchmarks
│   ├── alembic/             # Database migrations (pgvector, tables)
│   └── tests/               # Pytest suite with synthetic fixtures
├── frontend/
│   ├── src/
│   │   ├── components/      # UI primitives & layout (Sidebar, Header, Card, Badge)
│   │   ├── pages/           # Views (Overview, Documents, ComingSoon pages)
│   │   ├── services/        # Centralized fetch API client
│   │   └── hooks/           # State management hooks (useHealth, useDocuments)
│   └── tests/               # Vitest component test suite
├── docs/                    # Architecture, tutorials, decisions & experiments
├── sample_documents/        # Educational documents for testing
├── docker-compose.yml       # PostgreSQL + pgvector, backend, and frontend
└── README.md
```

---

## 3. Key Design Decisions

### 3.1. Non-Destructive Ingestion & Dual Storage
When documents are ingested, the original file is preserved on disk under a unique generated UUID, and extracted sections retain their raw text. Normalization produces an educational comparison dataset without altering source integrity.

### 3.2. Sequential Section Order & Page-Level Attribution
Downstream RAG synthesis requires citations and attribution. PyMuPDF extracts 1-indexed page numbers, while DOCX and Markdown extract heading hierarchies.

### 3.3. Duplicate Policy
Files are hashed using SHA-256. If a file with an identical hash already exists in the system, it is rejected with an HTTP 409 Conflict. This prevents redundant processing and duplicate vector indexing.

### 3.4. Pure JavaScript React Dashboard
The frontend uses standard JavaScript (`.jsx`) with Tailwind CSS and modular component structure, providing rapid iteration and universal readability without build overhead.
