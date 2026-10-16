# RAG Engineering Lab

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-blue.svg)](https://react.dev/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue.svg)](https://www.postgresql.org/)
[![pgvector](https://img.shields.io/badge/pgvector-0.7+-orange.svg)](https://github.com/pgvector/pgvector)
[![Phase](https://img.shields.io/badge/Phase-08%20Advanced%20Retrieval%20Complete-emerald.svg)](docs/roadmap.md)

A comprehensive, hands-on, educational laboratory designed to master every layer of **Retrieval-Augmented Generation (RAG)** from first principles.

---

## Current Status: Phase 08 — Advanced Retrieval, Hybrid Search & RAG Optimization

- **Phase 01 (Foundation)**: Modular monorepo, FastAPI backend, PostgreSQL 16 + pgvector, Alembic migrations, Docker Compose with health checks, and a responsive React JavaScript dashboard.
- **Phase 02 (Document Ingestion)**: Multi-format document parsers (PDF, DOCX, TXT, Markdown), magic-byte validation, SHA-256 duplicate detection, non-destructive text normalization, section extraction, and interactive educational comparison view.
- **Phase 03 (Chunking Strategies)**: 6 real chunking algorithms (Fixed-size, Recursive, Sentence with bilingual FR/EN rules, Semantic with cosine breakpoints, Parent-Child hierarchical, Structure-Aware), token/character metrics engine, experiments persistence, and interactive React Chunking Lab.
- **Phase 04 (Embeddings Engineering)**: Open-source dense SentenceTransformers on CPU, model registry with lazy loading and asymmetric prefixes, subword tokenization inspector with truncation warnings, manual vector mathematics (Cosine, Dot Product, Euclidean Distance, L2 Normalization), 2D semantic projections (PCA, t-SNE, UMAP), multi-pair evaluation benchmark suite, PostgreSQL vector persistence, and interactive Embeddings Explorer dashboard.
- **Phase 05 (Vector Database & Semantic Search)**: Exact KNN, HNSW, IVFFlat indexing in PostgreSQL with pgvector, JIT benchmark audit, query plan explain analyzer, and isolated benchmark tables.
- **Phase 06 (Naive RAG & Grounded Generation)**: End-to-end RAG orchestrator, multi-provider LLM support (Ollama, OpenAI, Gemini, Mock), prompt engineering, context budgeting, citation manager with [S#] extraction, and SSE real-time streaming.
- **Phase 07 (RAG Evaluation & Quality Benchmarking)**: Golden dataset manager with human review workflows, pure Python retrieval metrics (P@K, R@K, HitRate, MRR, nDCG, MAP), claim-level faithfulness evaluation, citation support precision, model pricing registry, and interactive React Evaluation Studio with learning mode sandboxes.
- **Phase 08 (Advanced Retrieval & Hybrid Search)**: PostgreSQL native GIN full-text search (`tsvector`), Reciprocal Rank Fusion (RRF, $k=60$) & normalized linear blending, Cross-Encoder reranking (TinyBERT CPU), multi-query expansion, contiguous neighbor context optimization, graded relevance auditing (0/1/2), and interactive 8-strategy visualizer in Retrieval Lab and Evaluation Studio.

---

## Prerequisites

- **Docker & Docker Compose** (Recommended): Docker 24+ and Compose v2+.
- **Or for Local Development**:
  - Python 3.12+
  - Node.js 20+ and npm 10+
  - PostgreSQL 16 with `pgvector` extension

---

## Quick Start (Docker Compose)

The easiest way to run the entire stack with zero manual dependency configuration:

```bash
# 1. Clone or enter project directory
cd rag-engineering-lab

# 2. Copy environment file
cp .env.example .env

# 3. Start PostgreSQL, Backend, and Frontend containers
docker compose up --build -d

# 4. Check services status
docker compose ps
```

Services will be available at:
- **Frontend Dashboard**: [http://localhost:5173](http://localhost:5173)
- **Backend API Docs (Swagger UI)**: [http://localhost:8000/api/v1/docs](http://localhost:8000/api/v1/docs)
- **Backend Health Probe**: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)
- **Readiness & Database Probe**: [http://localhost:8000/api/v1/health/ready](http://localhost:8000/api/v1/health/ready)

---

## Local Non-Docker Development Workflow

If you prefer running the backend and frontend directly on your host machine:

### 1. Database Setup
Start PostgreSQL with pgvector using Docker:
```bash
docker run -d --name rag-postgres -p 5432:5432 \
  -e POSTGRES_USER=rag_user \
  -e POSTGRES_PASSWORD=rag_password \
  -e POSTGRES_DB=rag_db \
  pgvector/pgvector:pg16
```

### 2. Backend Setup
```bash
cd backend

# Create and activate Python virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run database migrations
alembic upgrade head

# Start FastAPI development server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start Vite development server
npm run dev
```

---

## Testing & Quality Assurance

### Run Backend Tests (Pytest)
```bash
cd backend
pytest tests -v
```

### Run Frontend Tests (Vitest)
```bash
cd frontend
npm test
```

---

## Environment Variables Reference

| Variable | Default Value | Description |
|---|---|---|
| `PROJECT_NAME` | `"RAG Engineering Lab"` | Application name displayed in headers & logs |
| `ENVIRONMENT` | `development` | Runtime environment (`development`, `production`) |
| `PORT` | `8000` | Backend API server listening port |
| `POSTGRES_SERVER` | `localhost` (`postgres` in Docker) | Database host address |
| `POSTGRES_PORT` | `5432` | Database port |
| `POSTGRES_USER` | `rag_user` | PostgreSQL username |
| `POSTGRES_PASSWORD` | `rag_password` | PostgreSQL password |
| `POSTGRES_DB` | `rag_db` | PostgreSQL database name |
| `STORAGE_DIR` | `data/uploads` | Path for safe local file persistence |
| `MAX_UPLOAD_SIZE_MB` | `15` | Maximum upload size threshold |
| `VITE_API_URL` | `http://localhost:8000` | Backend URL used by React frontend |

---

## Curriculum Documentation

- [Architecture Overview](docs/architecture/overview.md)
- [Architecture Decision Record (ADR 0001)](docs/decisions/0001-architecture.md)
- [Tutoriel 01: Les Fondations du RAG](docs/tutorials/01-foundations.md) (en français)
- [Tutoriel 02: Ingestion & Parsing Documentaire](docs/tutorials/02-document-ingestion.md) (en français)
- [Laboratory Experiments Guide](docs/experiments/02-ingestion-experiments.md)
- [16-Phase Curriculum Roadmap](docs/roadmap.md)

---

## Troubleshooting

- **Database Connection Error (`connection refused`)**: Ensure PostgreSQL container is running via `docker compose ps` and port `5432` is not occupied by an existing local PostgreSQL instance.
- **pgvector extension missing**: Run `alembic upgrade head` to execute migration `0001_init_pgvector.py`, or verify that your PostgreSQL image is `pgvector/pgvector:pg16`.
- **Duplicate Document (HTTP 409)**: Files with identical cryptographic SHA-256 hashes are automatically rejected to preserve vector index integrity. Rename or alter the file content to index a distinct document.
- **Scanned PDF Warning**: PyMuPDF automatically detects image-only pages without text and issues an `OCR_REQUIRED` warning. OCR engines will be introduced in advanced multimodal phases.
# rag-lab
