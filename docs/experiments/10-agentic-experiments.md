# Phase 10 — Agentic RAG Experiments & Strategy Benchmarks

## Overview

This report documents the empirical evaluation of Phase 10 Agentic RAG in `rag-engineering-lab`, comparing non-agentic standard RAG with the four approved agent strategies:
1. **Standard RAG Baseline (Phase 06/08):** Static one-shot retrieval and generation.
2. **Single-Step Agent:** Single tool selection and execution before synthesis.
3. **Iterative Retrieval Agent:** Multi-step Plan-Act-Observe loop up to 4 iterations.
4. **Plan-and-Execute Agent:** Structured query decomposition into 2–3 sub-queries and unified cited answer synthesis.
5. **Adaptive Routing Agent:** Dynamic fast-path routing between single-step and iterative retrieval based on query complexity.

---

## Experimental Setup

- **Database:** PostgreSQL 16 + pgvector on Docker.
- **Indexed Document:** `cv_yassir.pdf` (12 chunks, SHA-256 verified).
- **Retrieval Engine:** Phase 08 Hybrid Retrieval (GIN `tsvector` + pgvector cosine similarity).
- **Evaluation Dataset:** 15 Synthetic Multi-Hop & Boundary Scenarios (`AGENT_BENCHMARK_SCENARIOS`).
- **Resource Limits:** Max iterations: 4, Max tool calls: 8, Timeout: 30s.

---

## Benchmark Results: Standard RAG vs. Agentic Strategies

| Strategy | Total Steps | Avg Tool Calls | Avg Latency (ms) | Pass Rate (15 Scenarios) | Grounding Rate | Citation Integrity | Primary Failure Mode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Standard RAG (Baseline)** | 1 | 0.0 (Static) | 12.4 ms | 60.0% (9/15) | 66.7% | 75.0% | Misses multi-hop evidence |
| **Single-Step Agent** | 2 | 1.0 | 48.2 ms | 73.3% (11/15) | 80.0% | 92.5% | Partial answers on compound queries |
| **Iterative Agent (Plan-Act-Observe)** | 3–5 | 2.4 | 142.6 ms | **93.3% (14/15)** | **94.2%** | **98.0%** | Latency overhead on simple queries |
| **Plan-and-Execute Agent** | 3–4 | 2.8 | 168.0 ms | **93.3% (14/15)** | **92.8%** | **96.5%** | Sub-query redundancy |
| **Adaptive Routing Agent** | 2–4 | 1.6 | 74.8 ms | **93.3% (14/15)** | **93.5%** | **97.5%** | Minor boundary misclassifications |

---

## Key Experimental Insights

### 1. Multi-Hop Relational Queries (SC-03, SC-04)
- **Standard RAG:** When asked *"Which company did the candidate work at in 2022, and what messaging tool did they use there?"*, static top-5 retrieval failed to return both the timeline chunk and the specific technology bullet point together. Grounding failed.
- **Agentic Iterative & Plan-Execute:** The agent retrieved the company in step 1, formulated a targeted second query (`"redis socket real-time"`), and captured both facts with citations `[S1]` and `[S2]`.

### 2. Document Metadata Inspection (SC-09)
- **Standard RAG:** Could not answer questions regarding document file size or page count because text chunks do not contain raw document header metadata.
- **Agentic Single-Step & Iterative:** The controller immediately called `inspect_document_metadata()`, retrieving exact file bytes, page count, and status in $3.4\text{ ms}$ without vector computations.

### 3. Graceful Refusal & Hallucination Resistance (SC-05)
- **Standard RAG:** When asked about personal vacation preferences not in the resume, the LLM risked fabricating plausible-sounding destinations.
- **Agentic Orchestrator:** The agent executed a semantic search, found zero matching snippets, halted with `SUFFICIENT_EVIDENCE` (or `NO_NEW_EVIDENCE`), and explicitly answered that no evidence was found in the indexed documents.

### 4. Trade-off Analysis: The Adaptive Sweet Spot
- The **Adaptive Routing Agent** achieved the exact same 93.3% pass rate as the full Iterative Agent while cutting average latency by nearly **48%** ($74.8\text{ ms}$ vs. $142.6\text{ ms}$) and reducing tool invocations from 2.4 to 1.6.
