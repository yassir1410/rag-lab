import React from 'react';
import { render, screen, fireEvent, within } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import App from '../src/App';

// Mock API calls
vi.mock('../src/services/api', () => ({
  api: {
    getHealth: vi.fn().mockResolvedValue({ status: 'ok', version: '0.3.0', environment: 'development' }),
    getReadiness: vi.fn().mockResolvedValue({
      status: 'ready',
      database_connected: true,
      pgvector_available: true,
      storage_writable: true,
    }),
    listDocuments: vi.fn().mockResolvedValue({ total: 0, items: [] }),
    getDocument: vi.fn(),
    getDocumentSections: vi.fn(),
    getDocumentContent: vi.fn(),
    uploadDocument: vi.fn(),
    deleteDocument: vi.fn(),
    getChunkingStrategies: vi.fn().mockResolvedValue([
      {
        strategy_name: 'fixed_size',
        display_name: 'Fixed-Size Window',
        description: 'Fixed-size character or token sliding window.',
        default_config: { chunk_size: 400, chunk_overlap: 50 },
      },
    ]),
    previewChunking: vi.fn().mockResolvedValue({
      strategy: 'fixed_size',
      chunks: [],
      metrics: { total_chunks: 0 },
      config: {},
      processing_duration_ms: 1.2,
    }),
    createChunkingExperiment: vi.fn(),
    listChunkingExperiments: vi.fn().mockResolvedValue({ total: 0, items: [] }),
    getChunkingExperiment: vi.fn(),
    deleteChunkingExperiment: vi.fn(),
    compareChunkingStrategies: vi.fn(),
    // Retrieval mocks
    listModels: vi.fn().mockResolvedValue([
      { model_id: 'sentence-transformers/all-MiniLM-L6-v2', display_name: 'All-MiniLM-L6-v2', dimension: 384 },
    ]),
    getDocuments: vi.fn().mockResolvedValue({ total: 0, items: [] }),
    getChunkingExperiments: vi.fn().mockResolvedValue({ total: 0, items: [] }),
    listIndexes: vi.fn().mockResolvedValue([]),
    listRetrievalExperiments: vi.fn().mockResolvedValue([]),
    searchVectors: vi.fn().mockResolvedValue({
      query: 'test query',
      model_id: 'sentence-transformers/all-MiniLM-L6-v2',
      search_algorithm: 'exact',
      distance_metric: 'cosine',
      top_k: 5,
      total_candidates: 0,
      results: [],
      query_latency_ms: 1.0,
      embedding_latency_ms: 2.0,
      total_latency_ms: 3.0,
      warnings: [],
    }),
    // RAG mocks (Phase 06)
    getRAGProviders: vi.fn().mockResolvedValue([
      { provider_id: 'mock', display_name: 'Mock LLM', is_available: true },
    ]),
    getRAGModels: vi.fn().mockResolvedValue([
      {
        model_id: 'mock-instruct-v1',
        display_name: 'Mock Instruct',
        provider_id: 'mock',
        context_window: 4096,
      },
    ]),
    getRAGPromptTemplates: vi.fn().mockResolvedValue([
      { id: 'grounded', name: 'Grounded RAG', description: 'Strict citations' },
    ]),
    getEmbeddingModels: vi.fn().mockResolvedValue([
      { model_id: 'sentence-transformers/all-MiniLM-L6-v2', display_name: 'All-MiniLM' },
    ]),
    queryRAG: vi.fn().mockResolvedValue({
      trace_id: 'test-trace-123',
      question: 'test question',
      answer: 'Test answer [S1]',
      sources: [],
      citations: [],
      timing_metrics: {},
      token_metrics: {},
    }),
    // Evaluation mocks (Phase 07)
    listEvaluationDatasets: vi.fn().mockResolvedValue([
      { id: 'ds-1', name: 'Test Dataset', question_count: 5 },
    ]),
    listEvaluationRuns: vi.fn().mockResolvedValue([]),
    getEvaluationMetricsCatalog: vi.fn().mockResolvedValue([
      { id: 'precision_at_k', name: 'Precision@K', category: 'retrieval' },
    ]),
    listEvaluationQuestions: vi.fn().mockResolvedValue([]),
  },
}));

describe('RAG Engineering Lab Frontend', () => {
  it('renders application with sidebar and overview by default', async () => {
    render(<App />);

    expect(screen.getByText('RAG Lab')).toBeInTheDocument();
    expect(screen.getByText('RAG Engineering Laboratory')).toBeInTheDocument();

    const aside = screen.getByRole('complementary');
    expect(within(aside).getByText('Overview')).toBeInTheDocument();
    expect(within(aside).getByText('Documents')).toBeInTheDocument();
    expect(within(aside).getByText('Chunking Lab')).toBeInTheDocument();
    expect(within(aside).getByText('Embeddings Explorer')).toBeInTheDocument();
    expect(within(aside).getByText('Retrieval Lab')).toBeInTheDocument();
  });

  it('switches to Documents page on sidebar click', async () => {
    render(<App />);

    const aside = screen.getByRole('complementary');
    const documentsNavButton = within(aside).getByText('Documents');
    fireEvent.click(documentsNavButton);

    expect(screen.getByText('Document Ingestion & Parsing')).toBeInTheDocument();
    expect(screen.getByText(/Supported formats:/i)).toBeInTheDocument();
  });

  it('switches to Chunking Lab on sidebar click', async () => {
    render(<App />);

    const aside = screen.getByRole('complementary');
    const chunkingNavButton = within(aside).getByText('Chunking Lab');
    fireEvent.click(chunkingNavButton);

    expect(screen.getByText('Text Chunking Laboratory')).toBeInTheDocument();
    expect(screen.getByText(/Algorithm Parameters:/i)).toBeInTheDocument();
  });

  it('switches to Embeddings Lab on sidebar click', async () => {
    render(<App />);

    const aside = screen.getByRole('complementary');
    const embeddingsNavButton = within(aside).getByText('Embeddings Explorer');
    fireEvent.click(embeddingsNavButton);

    expect(screen.getByText('Embeddings Engineering Lab')).toBeInTheDocument();
    expect(
      screen.getByText('Registered Embedding Models (Local CPU Inference)')
    ).toBeInTheDocument();
  });

  it('switches to Retrieval Lab on sidebar click', async () => {
    render(<App />);

    const aside = screen.getByRole('complementary');
    const retrievalNavButton = within(aside).getByText('Retrieval Lab');
    fireEvent.click(retrievalNavButton);

    expect(screen.getByText(/Vector Database & Semantic Search Lab/i)).toBeInTheDocument();
    expect(screen.getByText(/Exact KNN/i)).toBeInTheDocument();
  });

  it('switches to RAG Chat on sidebar click', async () => {
    render(<App />);

    const aside = screen.getByRole('complementary');
    const chatNavButton = within(aside).getByText('RAG Chat');
    fireEvent.click(chatNavButton);

    expect(
      screen.getByText(/RAG Chat & Grounded Generation Studio/i)
    ).toBeInTheDocument();
    expect(screen.getByText(/Execute RAG/i)).toBeInTheDocument();
  });

  it('switches to Evaluation Lab on sidebar click', async () => {
    render(<App />);

    const aside = screen.getByRole('complementary');
    const evalNavButton = within(aside).getByText('Evaluation Lab');
    fireEvent.click(evalNavButton);

    expect(screen.getByText(/Phase 07 — RAG Evaluation & Quality Benchmarking/i)).toBeInTheDocument();
    expect(screen.getByText(/Overview & KPIs/i)).toBeInTheDocument();
  });

  it('displays Coming Soon screen when clicking future phases (e.g. Experiments)', async () => {
    render(<App />);

    const aside = screen.getByRole('complementary');
    const expNavButton = within(aside).getByText('Experiments');
    fireEvent.click(expNavButton);

    expect(screen.getByText(/Coming Soon/i)).toBeInTheDocument();
  });
});
