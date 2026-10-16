# ADR 0001: Monorepo Architecture, PostgreSQL with pgvector, and FastAPI Backend

## Status
Accepted

## Context
Retrieval-Augmented Generation (RAG) involves multiple distinct disciplines: document processing, database indexing, vector mathematics, REST APIs, and interactive UI visualization.
To make this laboratory truly educational, we required:
1. Complete transparency over data transformations.
2. Fast local iteration without cloud dependencies.
3. A robust relational foundation supporting native vector operations.
4. Clean separation between API, database, and client layers.

## Decision
1. **Modular Monorepo**: We organize the code as a single repository containing `backend/`, `frontend/`, `docs/`, and `sample_documents/`.
2. **PostgreSQL 16 with pgvector**: Instead of an isolated single-purpose vector DB (e.g. Chroma, Pinecone), PostgreSQL allows storing relational document metadata alongside high-dimensional vector embeddings in the exact same transactional engine (ACID compliant).
3. **FastAPI & SQLAlchemy 2**: FastAPI provides asynchronous request handling, automatic OpenAPI generation, and Pydantic v2 validation. SQLAlchemy 2 provides type-safe ORM models and flexible session lifecycle management.
4. **React (JavaScript) + Vite + Tailwind CSS**: Standard JavaScript `.jsx` with Vite provides sub-second hot reload and readable UI components without TypeScript compilation layers.
5. **Alembic Migrations**: All schema modifications (including extension activation and table creation) are tracked via deterministic revision scripts.

## Alternatives Considered
- **Microservices across separate repos**: Rejected due to high cognitive overhead and synchronisation issues for students.
- **Standalone Vector DB (Chroma / Qdrant)**: Good for dedicated vector workloads, but requires running multiple datastores and makes relational joins (e.g. document -> sections -> permissions -> evaluations) complex.
- **TypeScript for Frontend**: Expressly avoided to match the educational requirement for plain JavaScript.

## Consequences
- Single `docker compose up` brings up the entire environment.
- Students can inspect raw SQL queries and vector distance metrics directly in Postgres.
- Upgrading to future phases (chunking, embeddings, hybrid search) simply extends existing modular packages.
