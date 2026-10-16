const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class ApiClient {
  constructor(baseUrl = API_BASE_URL) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
  }

  async _request(endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    const headers = {
      ...options.headers,
    };

    if (!(options.body instanceof FormData) && !headers['Content-Type']) {
      headers['Content-Type'] = 'application/json';
    }

    try {
      const response = await fetch(url, { ...options, headers });

      if (response.status === 204) {
        return null;
      }

      const data = await response.json().catch(() => null);

      if (!response.ok) {
        const error = new Error(
          (data && (data.detail?.message || data.detail || data.message)) ||
          `HTTP Error ${response.status}: ${response.statusText}`
        );
        error.status = response.status;
        error.data = data;
        throw error;
      }

      return data;
    } catch (err) {
      if (err.name === 'TypeError' && err.message.includes('fetch')) {
        const networkError = new Error(
          `Unable to connect to backend server at ${this.baseUrl}. Please verify the service is running.`
        );
        networkError.status = 0;
        throw networkError;
      }
      throw err;
    }
  }

  // Health & Readiness
  async getHealth() {
    return this._request('/api/v1/health');
  }

  async getReadiness() {
    return this._request('/api/v1/health/ready');
  }

  // Documents API
  async uploadDocument(file) {
    const formData = new FormData();
    formData.append('file', file);

    return this._request('/api/v1/documents', {
      method: 'POST',
      body: formData,
    });
  }

  async listDocuments(limit = 50, offset = 0) {
    return this._request(`/api/v1/documents?limit=${limit}&offset=${offset}`);
  }

  async getDocuments(limit = 50, offset = 0) {
    return this.listDocuments(limit, offset);
  }

  async getDocument(documentId) {
    return this._request(`/api/v1/documents/${documentId}`);
  }

  async getDocumentSections(documentId) {
    return this._request(`/api/v1/documents/${documentId}/sections`);
  }

  async getDocumentContent(documentId) {
    return this._request(`/api/v1/documents/${documentId}/content`);
  }

  async deleteDocument(documentId) {
    return this._request(`/api/v1/documents/${documentId}`, {
      method: 'DELETE',
    });
  }

  // Chunking API
  async getChunkingStrategies() {
    return this._request('/api/v1/chunking/strategies');
  }

  async previewChunking(documentId, strategy, config) {
    return this._request('/api/v1/chunking/preview', {
      method: 'POST',
      body: JSON.stringify({
        document_id: documentId,
        strategy,
        config,
      }),
    });
  }

  async createChunkingExperiment(documentId, strategy, config) {
    return this._request('/api/v1/chunking/experiments', {
      method: 'POST',
      body: JSON.stringify({
        document_id: documentId,
        strategy,
        config,
      }),
    });
  }

  async listChunkingExperiments(documentId = null) {
    const url = documentId
      ? `/api/v1/chunking/experiments?document_id=${documentId}`
      : '/api/v1/chunking/experiments';
    return this._request(url);
  }

  async getChunkingExperiments(documentId = null) {
    return this.listChunkingExperiments(documentId);
  }

  async getChunkingExperiment(experimentId) {
    return this._request(`/api/v1/chunking/experiments/${experimentId}`);
  }

  async deleteChunkingExperiment(experimentId) {
    return this._request(`/api/v1/chunking/experiments/${experimentId}`, {
      method: 'DELETE',
    });
  }

  async compareChunkingStrategies(documentId, strategies) {
    return this._request('/api/v1/chunking/compare', {
      method: 'POST',
      body: JSON.stringify({
        document_id: documentId,
        strategies,
      }),
    });
  }

  // Embeddings Lab API
  async getEmbeddingModels() {
    return this._request('/api/v1/embeddings/models');
  }

  async tokenizeText(text, modelId) {
    return this._request('/api/v1/embeddings/tokenize', {
      method: 'POST',
      body: JSON.stringify({ text, model_id: modelId }),
    });
  }

  async compareTokenizers(text, modelIds) {
    return this._request('/api/v1/embeddings/tokenize/compare', {
      method: 'POST',
      body: JSON.stringify({ text, model_ids: modelIds }),
    });
  }

  async generateEmbeddings(payload) {
    return this._request('/api/v1/embeddings/generate', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  async calculateSimilarity(textA, textB, modelId, normalize = true) {
    return this._request('/api/v1/embeddings/similarity', {
      method: 'POST',
      body: JSON.stringify({
        text_a: textA,
        text_b: textB,
        model_id: modelId,
        normalize,
      }),
    });
  }

  async calculateSimilarityMatrix(texts, modelId, metric = 'cosine', normalize = true) {
    return this._request('/api/v1/embeddings/matrix', {
      method: 'POST',
      body: JSON.stringify({
        texts,
        model_id: modelId,
        metric,
        normalize,
      }),
    });
  }

  async projectEmbeddings(payload) {
    return this._request('/api/v1/embeddings/project', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  async runEmbeddingBenchmark(modelId) {
    return this._request('/api/v1/embeddings/benchmark', {
      method: 'POST',
      body: JSON.stringify({ model_id: modelId }),
    });
  }

  async createEmbeddingExperiment(payload) {
    return this._request('/api/v1/embeddings/experiments', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  async getEmbeddingExperiments(documentId = null) {
    const url = documentId
      ? `/api/v1/embeddings/experiments?document_id=${documentId}`
      : '/api/v1/embeddings/experiments';
    return this._request(url);
  }

  async getEmbeddingExperiment(experimentId) {
    return this._request(`/api/v1/embeddings/experiments/${experimentId}`);
  }

  async deleteEmbeddingExperiment(experimentId) {
    return this._request(`/api/v1/embeddings/experiments/${experimentId}`, {
      method: 'DELETE',
    });
  }

  // Retrieval & Vector Search API (Phase 05 & Phase 08)
  async listRetrievalStrategies() {
    return this._request('/api/v1/retrieval/strategies');
  }

  async searchLexical(payload) {
    return this._request('/api/v1/retrieval/lexical', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  async searchHybrid(payload) {
    return this._request('/api/v1/retrieval/hybrid', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  async rerankCandidates(payload) {
    return this._request('/api/v1/retrieval/rerank', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  async searchVectors(payload) {
    return this._request('/api/v1/retrieval/search', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  async listIndexes() {
    return this._request('/api/v1/retrieval/indexes');
  }

  async createIndex(payload) {
    return this._request('/api/v1/retrieval/indexes', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  async dropIndex(indexId) {
    return this._request(`/api/v1/retrieval/indexes/${indexId}`, {
      method: 'DELETE',
    });
  }

  async runRetrievalBenchmark(payload) {
    return this._request('/api/v1/retrieval/benchmark', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  async generateSyntheticDataset(modelId, count = 300) {
    return this._request(
      `/api/v1/retrieval/synthetic-dataset?model_id=${encodeURIComponent(modelId)}&count=${count}`,
      { method: 'POST' }
    );
  }

  async listRetrievalExperiments() {
    return this._request('/api/v1/retrieval/experiments');
  }

  async saveRetrievalExperiment(payload) {
    return this._request('/api/v1/retrieval/experiments', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  async deleteRetrievalExperiment(experimentId) {
    return this._request(`/api/v1/retrieval/experiments/${experimentId}`, {
      method: 'DELETE',
    });
  }

  // RAG Generation & Chat API (Phase 06)
  async getRAGProviders() {
    return this._request('/api/v1/rag/providers');
  }

  async getRAGModels() {
    return this._request('/api/v1/rag/models');
  }

  async getRAGPromptTemplates() {
    return this._request('/api/v1/rag/prompts');
  }

  async queryRAG(payload) {
    return this._request('/api/v1/rag/query', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  async queryRAGStream(payload, onEvent, onError) {
    const url = `${API_BASE_URL}/api/v1/rag/query/stream`;
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: Failed to stream RAG query`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop(); // keep last incomplete chunk

        for (const block of lines) {
          if (!block.trim()) continue;
          let eventType = 'message';
          let eventData = null;

          const blockLines = block.split('\n');
          for (const line of blockLines) {
            if (line.startsWith('event: ')) {
              eventType = line.slice(7).trim();
            } else if (line.startsWith('data: ')) {
              try {
                eventData = JSON.parse(line.slice(6));
              } catch {
                eventData = line.slice(6);
              }
            }
          }

          if (eventData !== null) {
            onEvent(eventType, eventData);
          }
        }
      }
    } catch (err) {
      if (onError) onError(err);
      else throw err;
    }
  }

  async getRAGTrace(traceId) {
    return this._request(`/api/v1/rag/traces/${traceId}`);
  }

  // ==========================================
  // Phase 07 Evaluation & Benchmarking API
  // ==========================================

  async listEvaluationDatasets() {
    return this._request('/api/v1/evaluation/datasets');
  }

  async createEvaluationDataset(data) {
    return this._request('/api/v1/evaluation/datasets', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getEvaluationDataset(datasetId) {
    return this._request(`/api/v1/evaluation/datasets/${datasetId}`);
  }

  async updateEvaluationDataset(datasetId, data) {
    return this._request(`/api/v1/evaluation/datasets/${datasetId}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async deleteEvaluationDataset(datasetId) {
    return this._request(`/api/v1/evaluation/datasets/${datasetId}`, {
      method: 'DELETE',
    });
  }

  async listEvaluationQuestions(datasetId, params = {}) {
    const qs = new URLSearchParams(params).toString();
    const queryStr = qs ? `?${qs}` : '';
    return this._request(`/api/v1/evaluation/datasets/${datasetId}/questions${queryStr}`);
  }

  async addEvaluationQuestion(datasetId, data) {
    return this._request(`/api/v1/evaluation/datasets/${datasetId}/questions`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updateEvaluationQuestion(questionId, data) {
    return this._request(`/api/v1/evaluation/questions/${questionId}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async deleteEvaluationQuestion(questionId) {
    return this._request(`/api/v1/evaluation/questions/${questionId}`, {
      method: 'DELETE',
    });
  }

  async reviewEvaluationQuestion(questionId, data) {
    return this._request(`/api/v1/evaluation/questions/${questionId}/review`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async labelEvaluationQuestion(questionId, data) {
    return this._request(`/api/v1/evaluation/questions/${questionId}/labels`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getQuestionAudit(questionId) {
    return this._request(`/api/v1/evaluation/questions/${questionId}/audit`);
  }

  async importEvaluationQuestions(datasetId, data) {
    return this._request(`/api/v1/evaluation/datasets/${datasetId}/import`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async exportEvaluationDataset(datasetId) {
    return this._request(`/api/v1/evaluation/datasets/${datasetId}/export`);
  }

  async createEvaluationRun(data) {
    return this._request('/api/v1/evaluation/runs', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async listEvaluationRuns(params = {}) {
    const qs = new URLSearchParams(params).toString();
    const queryStr = qs ? `?${qs}` : '';
    return this._request(`/api/v1/evaluation/runs${queryStr}`);
  }

  async getEvaluationRun(runId) {
    return this._request(`/api/v1/evaluation/runs/${runId}`);
  }

  async getEvaluationRunQuestions(runId) {
    return this._request(`/api/v1/evaluation/runs/${runId}/questions`);
  }

  async cancelEvaluationRun(runId) {
    return this._request(`/api/v1/evaluation/runs/${runId}/cancel`, {
      method: 'POST',
    });
  }

  async retryEvaluationRun(runId) {
    return this._request(`/api/v1/evaluation/runs/${runId}/retry`, {
      method: 'POST',
    });
  }

  async compareEvaluationRuns(runA, runB) {
    return this._request(`/api/v1/evaluation/compare?run_a=${runA}&run_b=${runB}`);
  }

  async getEvaluationMetricsCatalog() {
    return this._request('/api/v1/evaluation/metrics');
  }

  async runTradeoffExperiment(params = {}) {
    const qs = new URLSearchParams(params).toString();
    const queryStr = qs ? `?${qs}` : '';
    return this._request(`/api/v1/evaluation/experiments/tradeoff${queryStr}`, {
      method: 'POST',
    });
  }
}

export const api = new ApiClient();
export default api;

