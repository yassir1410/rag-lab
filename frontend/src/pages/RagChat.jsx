import React, { useState, useEffect, useRef } from 'react';
import {
  MessageSquare,
  Send,
  Cpu,
  Database,
  Layers,
  Sparkles,
  AlertTriangle,
  CheckCircle,
  Clock,
  Sliders,
  BookOpen,
  FileText,
  HelpCircle,
  Copy,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  Zap,
  Terminal,
  ShieldAlert,
} from 'lucide-react';
import api from '../services/api';

const PRESET_QUESTIONS = [
  "What are the candidate's core backend engineering competencies?",
  "Does the profile mention experience with Docker, FastAPI, or PostgreSQL?",
  "Explain vector similarity search and high-dimensional indexes.",
  "What are the key differences between HNSW and IVFFlat algorithms?",
  "What consensus algorithms are discussed in the ingested papers?",
];

export function RagChat() {
  // Config state
  const [providers, setProviders] = useState([]);
  const [models, setModels] = useState([]);
  const [promptTemplates, setPromptTemplates] = useState([]);
  const [embeddingModels, setEmbeddingModels] = useState([]);
  
  const [selectedProvider, setSelectedProvider] = useState('ollama');
  const [selectedModel, setSelectedModel] = useState('qwen2.5:3b');
  const [selectedEmbeddingModel, setSelectedEmbeddingModel] = useState('sentence-transformers/all-MiniLM-L6-v2');
  const [searchAlgo, setSearchAlgo] = useState('exact');
  const [distanceMetric, setDistanceMetric] = useState('cosine');
  const [topK, setTopK] = useState(5);
  const [temperature, setTemperature] = useState(0.0);
  const [maxContextTokens, setMaxContextTokens] = useState(2048);
  const [maxOutputTokens, setMaxOutputTokens] = useState(512);
  const [selectedTemplate, setSelectedTemplate] = useState('grounded');

  // Mode toggles
  const [isStreaming, setIsStreaming] = useState(true);
  const [isLearningMode, setIsLearningMode] = useState(true);
  const [showPromptInspector, setShowPromptInspector] = useState(false);
  const [highlightedSourceId, setHighlightedSourceId] = useState(null);

  // Chat & Execution state
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [streamingStage, setStreamingStage] = useState('');
  const [currentResponse, setCurrentResponse] = useState(null);
  const [streamingAnswer, setStreamingAnswer] = useState('');
  const [error, setError] = useState(null);

  const sourcesRef = useRef({});

  // Initial data loading
  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      const [provList, modelList, promptList, embList] = await Promise.all([
        api.getRAGProviders().catch(() => []),
        api.getRAGModels().catch(() => []),
        api.getRAGPromptTemplates().catch(() => []),
        api.getEmbeddingModels().catch(() => []),
      ]);

      setProviders(provList);
      setModels(modelList);
      setPromptTemplates(promptList);
      setEmbeddingModels(embList);

      // Auto-select ready provider if default Ollama is offline
      const ollama = provList.find((p) => p.provider_id === 'ollama');
      if (ollama && ollama.is_available) {
        setSelectedProvider('ollama');
        setSelectedModel(ollama.default_model || 'qwen2.5:3b');
      } else {
        const firstReady = provList.find((p) => p.is_available);
        if (firstReady) {
          setSelectedProvider(firstReady.provider_id);
          setSelectedModel(firstReady.default_model || '');
        }
      }
    } catch (err) {
      console.error('Failed to load RAG initialization data:', err);
    }
  };

  const handleProviderChange = (newProviderId) => {
    setSelectedProvider(newProviderId);
    const prov = providers.find((p) => p.provider_id === newProviderId);
    if (prov) {
      setSelectedModel(prov.default_model || (prov.supported_models && prov.supported_models[0]) || '');
    }
  };

  const handleCitationClick = (citationTag) => {
    const cleanId = citationTag.replace(/[\[\]]/g, '');
    setHighlightedSourceId(cleanId);
    if (sourcesRef.current[cleanId]) {
      sourcesRef.current[cleanId].scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  };

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    if (!question.trim() || loading) return;

    setLoading(true);
    setError(null);
    setCurrentResponse(null);
    setStreamingAnswer('');
    setStreamingStage('Initiating RAG pipeline...');

    const payload = {
      question: question.trim(),
      embedding_model_id: selectedEmbeddingModel,
      search_algorithm: searchAlgo,
      distance_metric: distanceMetric,
      top_k: Number(topK),
      llm_provider: selectedProvider,
      llm_model: selectedModel,
      prompt_template: selectedTemplate,
      temperature: Number(temperature),
      max_output_tokens: Number(maxOutputTokens),
      max_context_tokens: Number(maxContextTokens),
      include_assembled_prompt: true,
    };

    if (isStreaming) {
      let accumulatedText = '';
      let initialSources = [];

      await api.queryRAGStream(
        payload,
        (eventType, eventData) => {
          if (eventType === 'status') {
            setStreamingStage(eventData.message || eventData.stage);
          } else if (eventType === 'sources') {
            initialSources = eventData.sources || [];
            setCurrentResponse((prev) => ({
              ...(prev || {}),
              trace_id: eventData.trace_id,
              sources: initialSources,
            }));
          } else if (eventType === 'chunk') {
            accumulatedText += eventData.content || '';
            setStreamingAnswer(accumulatedText);
          } else if (eventType === 'done') {
            setCurrentResponse({
              trace_id: eventData.trace_id,
              question: payload.question,
              answer: accumulatedText,
              llm_provider: payload.llm_provider,
              llm_model: payload.llm_model || 'default',
              prompt_template: payload.prompt_template,
              sources: initialSources,
              citations: eventData.citations || [],
              invalid_citations: eventData.invalid_citations || [],
              uncited_sources: eventData.uncited_sources || [],
              timing_metrics: eventData.timing_metrics || {},
              token_metrics: eventData.token_metrics || {},
              assembled_prompt: eventData.assembled_prompt,
              warnings: eventData.warnings || [],
            });
            setLoading(false);
            setStreamingStage('');
          } else if (eventType === 'error') {
            setError(eventData.error || 'Streaming error encountered');
            setLoading(false);
          }
        },
        (err) => {
          setError(err.message || 'Failed to stream response from backend');
          setLoading(false);
        }
      );
    } else {
      try {
        const res = await api.queryRAG(payload);
        setCurrentResponse(res);
        setStreamingAnswer(res.answer);
      } catch (err) {
        setError(err.message || 'RAG query failed');
      } finally {
        setLoading(false);
        setStreamingStage('');
      }
    }
  };

  const renderFormattedAnswer = (text) => {
    if (!text) return null;
    // Replace [S#] or [S1, S2, S3] with clickable badges
    const parts = text.split(/(\[[^\]]*S\d+[^\]]*\])/g);
    return (
      <div className="text-slate-100 text-sm leading-relaxed whitespace-pre-wrap font-sans">
        {parts.map((part, i) => {
          if (/^\[[^\]]*S\d+[^\]]*\]$/.test(part)) {
            const tokens = part.match(/S\d+/gi) || [];
            if (tokens.length > 0) {
              return (
                <span key={i} className="inline-flex items-center gap-1 mx-0.5 align-middle">
                  <span className="text-slate-400 font-mono text-xs">[</span>
                  {tokens.map((tok, tIdx) => {
                    const normTok = tok.toUpperCase();
                    const isHighlighted = highlightedSourceId === normTok;
                    return (
                      <React.Fragment key={tIdx}>
                        <button
                          type="button"
                          onClick={() => handleCitationClick(`[${normTok}]`)}
                          className={`inline-flex items-center px-1.5 py-0.5 rounded text-xs font-mono font-bold transition-all cursor-pointer ${
                            isHighlighted
                              ? 'bg-amber-400 text-slate-950 ring-2 ring-amber-300 scale-105'
                              : 'bg-emerald-500/20 text-emerald-300 hover:bg-emerald-500/40 border border-emerald-500/40'
                          }`}
                          title={`Click to view supporting passage for [${normTok}]`}
                        >
                          {normTok}
                        </button>
                        {tIdx < tokens.length - 1 && <span className="text-slate-500 text-xs">,</span>}
                      </React.Fragment>
                    );
                  })}
                  <span className="text-slate-400 font-mono text-xs">]</span>
                </span>
              );
            }
          }
          return <span key={i}>{part}</span>;
        })}
      </div>
    );
  };

  const currentProviderObj = providers.find((p) => p.provider_id === selectedProvider);

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-16 font-sans">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
              <MessageSquare className="w-6 h-6 text-emerald-400" />
              RAG Chat & Grounded Generation Studio
            </h1>
            <span className="px-2.5 py-0.5 text-xs font-mono rounded-full bg-emerald-950 text-emerald-400 border border-emerald-800">
              Phase 06 • Naive RAG
            </span>
          </div>
          <p className="text-sm text-slate-400 mt-1">
            Connect semantic vector retrieval with local LLM synthesis. Inspect prompts, context token budgets, and verified citations.
          </p>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => setIsStreaming(!isStreaming)}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono border transition-colors flex items-center gap-1.5 ${
              isStreaming
                ? 'bg-blue-500/10 border-blue-500/40 text-blue-300'
                : 'bg-slate-800 border-slate-700 text-slate-400'
            }`}
          >
            <Zap className="w-3.5 h-3.5" />
            {isStreaming ? 'SSE Streaming: ON' : 'SSE Streaming: OFF'}
          </button>
          <button
            type="button"
            onClick={() => setIsLearningMode(!isLearningMode)}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono border transition-colors flex items-center gap-1.5 ${
              isLearningMode
                ? 'bg-purple-500/10 border-purple-500/40 text-purple-300'
                : 'bg-slate-800 border-slate-700 text-slate-400'
            }`}
          >
            <BookOpen className="w-3.5 h-3.5" />
            {isLearningMode ? 'Learning Mode: ON' : 'Expert Mode'}
          </button>
        </div>
      </div>

      {/* Ollama Offline Warning Banner */}
      {selectedProvider === 'ollama' && currentProviderObj && !currentProviderObj.is_available && (
        <div className="bg-amber-500/10 border border-amber-500/30 text-amber-200 p-4 rounded-xl flex items-start gap-3 text-xs leading-relaxed">
          <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
          <div className="space-y-1.5">
            <div className="font-semibold text-amber-300">Local Ollama Service Not Detected</div>
            <p className="text-slate-300">
              Ollama is not running on your host at <code className="bg-slate-900 px-1 py-0.5 rounded text-amber-400 font-mono">http://localhost:11434</code>.
              To use local CPU inference, run the following in a host terminal:
            </p>
            <div className="bg-slate-950 p-2.5 rounded font-mono text-slate-200 border border-slate-800 flex items-center justify-between">
              <code>curl -fsSL https://ollama.com/install.sh | sh && ollama pull qwen2.5:3b && ollama serve</code>
            </div>
            <p className="text-[11px] text-slate-400">
              Tip: You can switch to the <span className="text-purple-300 font-mono">Mock LLM</span> provider in the left panel to test the complete pipeline deterministically without downloading models.
            </p>
          </div>
        </div>
      )}

      {/* Main 3-Column Studio Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Configuration & Budget (3 cols) */}
        <div className="lg:col-span-3 space-y-4">
          <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 space-y-4 text-xs">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <span className="font-semibold text-slate-200 flex items-center gap-1.5">
                <Sliders className="w-4 h-4 text-emerald-400" />
                Pipeline Parameters
              </span>
              <button
                type="button"
                onClick={loadInitialData}
                className="text-slate-500 hover:text-slate-300 transition-colors"
                title="Refresh Provider Status"
              >
                <RefreshCw className="w-3.5 h-3.5" />
              </button>
            </div>

            {/* LLM Provider */}
            <div>
              <label className="block text-slate-400 mb-1 font-medium">LLM Provider</label>
              <select
                value={selectedProvider}
                onChange={(e) => handleProviderChange(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-slate-200"
              >
                {providers.map((p) => (
                  <option key={p.provider_id} value={p.provider_id}>
                    {p.display_name} {p.is_available ? '✓' : '(Offline)'}
                  </option>
                ))}
              </select>
            </div>

            {/* Model ID */}
            <div>
              <div className="flex justify-between items-center mb-1">
                <label className="text-slate-400 font-medium">Model ID</label>
                {currentProviderObj?.default_model && (
                  <span className="text-[10px] text-slate-500 font-mono">
                    Default: {currentProviderObj.default_model}
                  </span>
                )}
              </div>
              {currentProviderObj?.supported_models && currentProviderObj.supported_models.length > 0 ? (
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-slate-200 font-mono"
                >
                  {currentProviderObj.supported_models.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type="text"
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  placeholder="e.g. qwen2.5:3b"
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-slate-200 font-mono"
                />
              )}
            </div>

            {/* Prompt Template */}
            <div>
              <label className="block text-slate-400 mb-1 font-medium">Prompt Engineering Template</label>
              <select
                value={selectedTemplate}
                onChange={(e) => setSelectedTemplate(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-slate-200"
              >
                {promptTemplates.map((t) => (
                  <option key={t.id} value={t.id}>
                    {t.name}
                  </option>
                ))}
              </select>
              {isLearningMode && (
                <p className="text-[11px] text-slate-500 mt-1">
                  {promptTemplates.find((t) => t.id === selectedTemplate)?.description}
                </p>
              )}
            </div>

            {/* Retrieval Algorithm */}
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-slate-400 mb-1 font-medium">Algorithm</label>
                <select
                  value={searchAlgo}
                  onChange={(e) => setSearchAlgo(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-slate-200"
                >
                  <option value="exact">Exact KNN</option>
                  <option value="hnsw">HNSW</option>
                  <option value="ivfflat">IVFFlat</option>
                </select>
              </div>
              <div>
                <label className="block text-slate-400 mb-1 font-medium">Metric</label>
                <select
                  value={distanceMetric}
                  onChange={(e) => setDistanceMetric(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-slate-200"
                >
                  <option value="cosine">Cosine</option>
                  <option value="l2">L2</option>
                  <option value="inner_product">Dot Product</option>
                </select>
              </div>
            </div>

            {/* Top-K */}
            <div>
              <div className="flex justify-between items-center mb-1">
                <label className="text-slate-400 font-medium">Top-K Chunks</label>
                <span className="font-mono text-emerald-400 font-semibold">{topK}</span>
              </div>
              <input
                type="range"
                min="1"
                max="10"
                value={topK}
                onChange={(e) => setTopK(e.target.value)}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
              />
            </div>

            {/* Temperature */}
            <div>
              <div className="flex justify-between items-center mb-1">
                <label className="text-slate-400 font-medium">Temperature</label>
                <span className="font-mono text-blue-400 font-semibold">{temperature}</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="1.0"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(e.target.value)}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
              />
              {isLearningMode && (
                <span className="text-[10px] text-slate-500 block mt-0.5">
                  0.0 = Deterministic grounding; higher = more creative variation.
                </span>
              )}
            </div>

            {/* Context Budget */}
            <div>
              <div className="flex justify-between items-center mb-1">
                <label className="text-slate-400 font-medium">Context Budget</label>
                <span className="font-mono text-purple-400 font-semibold">{maxContextTokens} tok</span>
              </div>
              <input
                type="range"
                min="256"
                max="4096"
                step="256"
                value={maxContextTokens}
                onChange={(e) => setMaxContextTokens(e.target.value)}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-purple-500"
              />
            </div>
          </div>

          {/* Token Allocation Inspector */}
          {currentResponse?.token_metrics && (
            <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 space-y-2 text-xs">
              <div className="font-semibold text-slate-300 flex items-center gap-1.5 pb-2 border-b border-slate-800">
                <Cpu className="w-3.5 h-3.5 text-purple-400" />
                Token Budget Inspector
              </div>
              <div className="space-y-1.5 font-mono text-[11px]">
                <div className="flex justify-between text-slate-400">
                  <span>System Prompt:</span>
                  <span className="text-slate-200">{currentResponse.token_metrics.system_prompt_tokens} tok</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>User Question:</span>
                  <span className="text-slate-200">{currentResponse.token_metrics.user_question_tokens} tok</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Retrieved Context:</span>
                  <span className="text-emerald-400 font-bold">{currentResponse.token_metrics.context_tokens} tok</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Reserved Output:</span>
                  <span className="text-blue-400">{currentResponse.token_metrics.reserved_output_tokens} tok</span>
                </div>
                <div className="pt-1.5 border-t border-slate-800 flex justify-between font-bold">
                  <span className="text-slate-300">Total Prompt:</span>
                  <span className="text-purple-300">{currentResponse.token_metrics.total_prompt_tokens} / {currentResponse.token_metrics.model_context_window}</span>
                </div>
              </div>
              <p className="text-[10px] text-slate-500 italic pt-1">
                Tokenizer: {currentResponse.token_metrics.tokenizer_name}
              </p>
            </div>
          )}
        </div>

        {/* Center Column: Chat & Stream Panel (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          {/* Query Form */}
          <form onSubmit={handleSubmit} className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between text-xs text-slate-400 font-medium">
              <span>Ask Document Question</span>
              <span className="text-[11px] text-slate-500">Press Submit or Enter</span>
            </div>

            <textarea
              rows={3}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="e.g. What are the candidate's core backend engineering competencies?"
              className="w-full bg-slate-950 border border-slate-800 rounded-lg p-3 text-sm text-slate-100 placeholder-slate-600 focus:outline-none focus:ring-1 focus:ring-emerald-500"
            />

            {/* Presets */}
            <div className="flex flex-wrap gap-1.5 pt-1">
              {PRESET_QUESTIONS.slice(0, 3).map((pq, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => setQuestion(pq)}
                  className="text-[10px] text-slate-400 hover:text-slate-200 bg-slate-950 px-2 py-1 rounded border border-slate-800/80 transition-colors"
                >
                  "{pq.slice(0, 42)}..."
                </button>
              ))}
            </div>

            <div className="flex items-center justify-between pt-2">
              <div className="text-xs text-slate-500 font-mono">
                {streamingStage && (
                  <span className="text-emerald-400 flex items-center gap-1.5">
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    {streamingStage}
                  </span>
                )}
              </div>
              <button
                type="submit"
                disabled={loading || !question.trim()}
                className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors"
              >
                {loading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Send className="w-3.5 h-3.5" />}
                {loading ? 'Synthesizing...' : 'Execute RAG'}
              </button>
            </div>
          </form>

          {/* Error Banner */}
          {error && (
            <div className="bg-red-500/10 border border-red-500/20 text-red-300 p-4 rounded-xl text-xs flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 shrink-0 text-red-400" />
              <span>{error}</span>
            </div>
          )}

          {/* Response Display */}
          {(streamingAnswer || currentResponse) && (
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <div className="flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-emerald-400" />
                  <span className="text-xs font-bold text-white uppercase tracking-wide">
                    Synthesized Grounded Answer
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-mono bg-slate-950 px-2 py-0.5 rounded text-slate-400 border border-slate-800">
                    {selectedProvider} / {selectedModel}
                  </span>
                </div>
              </div>

              {/* Answer Content */}
              <div className="bg-slate-950 p-4 rounded-lg border border-slate-800/80">
                {renderFormattedAnswer(streamingAnswer || currentResponse?.answer)}
                {loading && (
                  <span className="inline-block w-2 h-4 bg-emerald-400 animate-pulse ml-1 align-middle" />
                )}
              </div>

              {/* Citation Verification Summary */}
              {currentResponse?.citations && currentResponse.citations.length > 0 && (
                <div className="pt-2 border-t border-slate-800/80 space-y-2 text-xs">
                  <div className="flex items-center justify-between text-slate-400 font-medium">
                    <span className="flex items-center gap-1.5">
                      <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
                      Cited Sources Validation
                    </span>
                    <span className="text-[11px] text-slate-500 font-mono">
                      {currentResponse.citations.filter((c) => c.is_valid).length} valid / {currentResponse.citations.length} cited
                    </span>
                  </div>

                  <div className="flex flex-wrap gap-2">
                    {currentResponse.citations.map((c, idx) => (
                      <div
                        key={idx}
                        className={`px-2.5 py-1 rounded text-[11px] font-mono flex items-center gap-1.5 border ${
                          c.is_valid
                            ? 'bg-emerald-950/40 border-emerald-800/50 text-emerald-300'
                            : 'bg-red-950/40 border-red-800/50 text-red-300'
                        }`}
                      >
                        <span className="font-bold">{c.citation_id}</span>
                        <span>{c.document_name || 'Unknown Document'}</span>
                        {c.page_number && <span className="text-slate-500">(p.{c.page_number})</span>}
                        {c.is_valid ? (
                          <CheckCircle className="w-3 h-3 text-emerald-400" />
                        ) : (
                          <AlertTriangle className="w-3 h-3 text-red-400" />
                        )}
                      </div>
                    ))}
                  </div>

                  <p className="text-[10px] text-slate-500 italic">
                    Note: Citation ID matching proves the model referenced that passage; it does not constitute mathematical formal proof of factual entailment.
                  </p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right Column: Retrieved Sources & Passages (4 cols) */}
        <div className="lg:col-span-4 space-y-4">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <span className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
                <Database className="w-4 h-4 text-blue-400" />
                Retrieved Context Sources ({currentResponse?.sources?.length || 0})
              </span>
              <span className="text-[10px] text-slate-500 font-mono">Top-K = {topK}</span>
            </div>

            {(!currentResponse || !currentResponse.sources || currentResponse.sources.length === 0) ? (
              <div className="p-8 text-center text-xs text-slate-500 space-y-2">
                <FileText className="w-8 h-8 mx-auto text-slate-700" />
                <p>No query executed yet.</p>
                <p className="text-[11px] text-slate-600">
                  Retrieved passages with attribution tags [S1], [S2] will appear here.
                </p>
              </div>
            ) : (
              <div className="space-y-3 max-h-[600px] overflow-y-auto pr-1">
                {currentResponse.sources.map((src) => {
                  const cleanId = src.citation_id.replace(/[\[\]]/g, '');
                  const isHighlighted = highlightedSourceId === cleanId;
                  return (
                    <div
                      key={src.citation_id}
                      ref={(el) => (sourcesRef.current[cleanId] = el)}
                      className={`p-3 rounded-lg border transition-all text-xs space-y-2 ${
                        isHighlighted
                          ? 'bg-amber-950/20 border-amber-500/60 ring-1 ring-amber-400'
                          : src.was_cited
                          ? 'bg-slate-950 border-emerald-800/40'
                          : 'bg-slate-950/60 border-slate-800'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span
                            className={`px-1.5 py-0.5 rounded text-[11px] font-mono font-bold ${
                              src.was_cited
                                ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                                : 'bg-slate-800 text-slate-400 border border-slate-700'
                            }`}
                          >
                            {src.citation_id}
                          </span>
                          <span className="text-slate-300 font-medium truncate max-w-[150px]">
                            {src.document_name}
                          </span>
                        </div>
                        <div className="flex items-center gap-1.5 font-mono text-[10px]">
                          {src.page_number && <span className="text-slate-500">p.{src.page_number}</span>}
                          <span className="text-blue-400 font-bold">{src.similarity_score.toFixed(3)}</span>
                        </div>
                      </div>

                      <p className="text-slate-300 text-[11px] leading-relaxed font-mono bg-slate-900/60 p-2 rounded border border-slate-800/60">
                        {src.content}
                      </p>

                      <div className="flex items-center justify-between text-[10px] text-slate-500">
                        <span>Tokens: ~{src.token_count}</span>
                        {src.was_cited ? (
                          <span className="text-emerald-400 flex items-center gap-1 font-medium">
                            <CheckCircle className="w-3 h-3" /> Cited in answer
                          </span>
                        ) : (
                          <span className="text-slate-500">Uncited</span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Bottom Row: Pipeline Stage Timings & Prompt Inspector */}
      {currentResponse && (
        <div className="space-y-4 pt-4 border-t border-slate-800">
          {/* Pipeline Latency Timeline */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-slate-200 flex items-center gap-2">
                <Clock className="w-4 h-4 text-emerald-400" />
                Pipeline Stage Execution Latencies
              </span>
              <span className="font-mono text-emerald-400 font-bold">
                Total: {currentResponse.timing_metrics?.total_pipeline_latency_ms} ms
              </span>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-6 gap-3 text-center text-xs font-mono">
              <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                <div className="text-[10px] text-slate-500 font-sans">1. Embedding</div>
                <div className="text-purple-300 font-bold text-sm mt-0.5">
                  {currentResponse.timing_metrics?.embedding_latency_ms} ms
                </div>
              </div>
              <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                <div className="text-[10px] text-slate-500 font-sans">2. Retrieval ({searchAlgo})</div>
                <div className="text-blue-300 font-bold text-sm mt-0.5">
                  {currentResponse.timing_metrics?.retrieval_latency_ms} ms
                </div>
              </div>
              <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                <div className="text-[10px] text-slate-500 font-sans">3. Context Build</div>
                <div className="text-cyan-300 font-bold text-sm mt-0.5">
                  {currentResponse.timing_metrics?.context_assembly_ms} ms
                </div>
              </div>
              <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                <div className="text-[10px] text-slate-500 font-sans">4. Prompt Build</div>
                <div className="text-indigo-300 font-bold text-sm mt-0.5">
                  {currentResponse.timing_metrics?.prompt_building_ms} ms
                </div>
              </div>
              <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                <div className="text-[10px] text-slate-500 font-sans">5. LLM Inference</div>
                <div className="text-emerald-300 font-bold text-sm mt-0.5">
                  {currentResponse.timing_metrics?.llm_inference_ms} ms
                </div>
              </div>
              <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                <div className="text-[10px] text-slate-500 font-sans">6. Citation Check</div>
                <div className="text-amber-300 font-bold text-sm mt-0.5">
                  {currentResponse.timing_metrics?.citation_validation_ms} ms
                </div>
              </div>
            </div>
          </div>

          {/* Prompt Inspector Toggle */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
            <button
              type="button"
              onClick={() => setShowPromptInspector(!showPromptInspector)}
              className="w-full px-5 py-3 text-xs font-semibold text-slate-300 flex items-center justify-between hover:bg-slate-800/50 transition-colors"
            >
              <span className="flex items-center gap-2">
                <Terminal className="w-4 h-4 text-purple-400" />
                Prompt Inspector (Full Assembled Prompt sent to LLM)
              </span>
              {showPromptInspector ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>

            {showPromptInspector && (
              <div className="p-5 border-t border-slate-800 space-y-3 text-xs">
                <div className="p-2.5 rounded bg-amber-500/10 border border-amber-500/20 text-amber-300 text-[11px] flex items-center gap-2">
                  <ShieldAlert className="w-4 h-4 shrink-0" />
                  <span>
                    Security Notice: Retrieved content is treated as untrusted text. Authorization tokens and API keys are strictly excluded from prompts.
                  </span>
                </div>
                <pre className="bg-slate-950 p-4 rounded-lg border border-slate-800/80 font-mono text-[11px] text-slate-300 whitespace-pre-wrap max-h-96 overflow-y-auto">
                  {currentResponse.assembled_prompt}
                </pre>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
export default RagChat;
