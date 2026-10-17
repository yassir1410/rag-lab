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
  Compass,
  Table,
  Image as ImageIcon,
  Network,
  Eye,
  Columns,
  Info,
} from 'lucide-react';
import api from '../services/api';

const PRESET_QUESTIONS = [
  "According to the planetary data table on page 1, what is Jupiter's surface gravity? Cite the source.",
  "What are the candidate's core backend engineering competencies?",
  "What spectral resolution does the NIRCam instrument on JWST provide?",
  "Does the profile mention experience with Docker, FastAPI, or PostgreSQL?",
  "Explain vector similarity search and high-dimensional indexes.",
];

export function RagChat() {
  // Config state
  const [providers, setProviders] = useState([]);
  const [models, setModels] = useState([]);
  const [promptTemplates, setPromptTemplates] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [selectedDocumentId, setSelectedDocumentId] = useState('');
  const [capabilities, setCapabilities] = useState(null);

  // Routing Mode state: auto, manual, comparative
  const [routingMode, setRoutingMode] = useState('auto');
  const [manualRoutes, setManualRoutes] = useState(['text_hybrid']);
  const [comparativeResults, setComparativeResults] = useState(null);

  // Model & Execution parameters
  const [selectedProvider, setSelectedProvider] = useState('gemini');
  const [selectedModel, setSelectedModel] = useState('');
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
  const [selectedEvidenceModal, setSelectedEvidenceModal] = useState(null);

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

  // Capability reload when document changes
  useEffect(() => {
    loadCapabilities(selectedDocumentId || null);
  }, [selectedDocumentId]);

  const loadInitialData = async () => {
    try {
      const [provList, modelList, promptList, docList] = await Promise.all([
        api.getRAGProviders().catch(() => []),
        api.getRAGModels().catch(() => []),
        api.getRAGPromptTemplates().catch(() => []),
        api.listDocuments(50, 0).catch(() => ({ items: [] })),
      ]);

      setProviders(provList);
      setModels(modelList);
      setPromptTemplates(promptList);
      setDocuments(docList.items || []);

      // Auto-select Gemini provider if available, else first available
      const gemini = provList.find((p) => p.provider_id === 'gemini');
      if (gemini && gemini.is_available) {
        setSelectedProvider('gemini');
        setSelectedModel(gemini.default_model || 'gemini-3.8-flash');
      } else {
        const firstReady = provList.find((p) => p.is_available);
        if (firstReady) {
          setSelectedProvider(firstReady.provider_id);
          setSelectedModel(firstReady.default_model || '');
        }
      }
    } catch (err) {
      console.error('Failed to load Unified RAG initialization data:', err);
    }
  };

  const loadCapabilities = async (docId) => {
    try {
      const caps = await api.getUnifiedCapabilities(docId);
      setCapabilities(caps);
    } catch (err) {
      console.error('Failed to load document capabilities:', err);
    }
  };

  const handleProviderChange = (providerId) => {
    setSelectedProvider(providerId);
    const pObj = providers.find((p) => p.provider_id === providerId);
    if (pObj && pObj.default_model) {
      setSelectedModel(pObj.default_model);
    }
  };

  const toggleManualRoute = (routeId) => {
    if (manualRoutes.includes(routeId)) {
      if (manualRoutes.length > 1) {
        setManualRoutes(manualRoutes.filter((r) => r !== routeId));
      }
    } else {
      setManualRoutes([...manualRoutes, routeId]);
    }
  };

  const currentProviderObj = providers.find((p) => p.provider_id === selectedProvider);

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    if (!question.trim() || loading) return;

    setLoading(true);
    setError(null);
    setCurrentResponse(null);
    setStreamingAnswer('');
    setComparativeResults(null);
    setStreamingStage('Initiating Intelligent Query Router...');

    const payload = {
      question: question.trim(),
      document_id: selectedDocumentId || null,
      mode: routingMode,
      manual_routes: routingMode === 'manual' ? manualRoutes : null,
      top_k: Number(topK),
      llm_provider: selectedProvider,
      llm_model: selectedModel || null,
      prompt_template: selectedTemplate,
      temperature: Number(temperature),
      max_output_tokens: Number(maxOutputTokens),
      max_context_tokens: Number(maxContextTokens),
      include_assembled_prompt: true,
    };

    // If comparative mode, also run side-by-side comparison
    if (routingMode === 'comparative') {
      try {
        const compRes = await api.compareUnifiedRoutes(payload.question, payload.document_id);
        setComparativeResults(compRes);
      } catch (err) {
        console.error('Comparative evaluation error:', err);
      }
    }

    if (isStreaming) {
      let accumulatedText = '';
      let initialSources = [];
      let currentRouting = null;

      await api.queryUnifiedRAGStream(
        payload,
        (eventType, eventData) => {
          if (eventType === 'status') {
            setStreamingStage(eventData.message || eventData.stage);
          } else if (eventType === 'routing') {
            currentRouting = eventData;
            setCurrentResponse((prev) => ({
              ...(prev || {}),
              routing: eventData,
            }));
          } else if (eventType === 'sources') {
            initialSources = eventData.sources || [];
            setCurrentResponse((prev) => ({
              ...(prev || {}),
              trace_id: eventData.trace_id,
              evidence_items: initialSources,
            }));
          } else if (eventType === 'chunk') {
            accumulatedText += eventData.content || '';
            setStreamingAnswer(accumulatedText);
          } else if (eventType === 'done') {
            setCurrentResponse((prev) => ({
              ...(prev || {}),
              trace_id: eventData.trace_id,
              question: payload.question,
              answer: accumulatedText,
              routing: currentRouting || prev?.routing,
              evidence_items: initialSources,
              citations: eventData.citations || [],
              invalid_citations: eventData.invalid_citations || [],
              uncited_sources: eventData.uncited_sources || [],
              model_used: eventData.model_used || selectedModel,
              is_vision_enabled: eventData.is_vision_enabled || false,
              warnings: eventData.warnings || [],
            }));
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
        const res = await api.queryUnifiedRAG(payload);
        setCurrentResponse(res);
        setStreamingAnswer(res.answer);
      } catch (err) {
        setError(err.message || 'Unified RAG query failed');
      } finally {
        setLoading(false);
        setStreamingStage('');
      }
    }
  };

  const renderFormattedAnswer = (text) => {
    if (!text) return null;
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
                    const cleanId = tok.toUpperCase();
                    const isHigh = highlightedSourceId === cleanId;
                    return (
                      <button
                        key={tIdx}
                        type="button"
                        onClick={() => {
                          setHighlightedSourceId(isHigh ? null : cleanId);
                          if (sourcesRef.current[cleanId]) {
                            sourcesRef.current[cleanId].scrollIntoView({
                              behavior: 'smooth',
                              block: 'center',
                            });
                          }
                        }}
                        className={`font-mono text-xs font-bold px-1.5 py-0.5 rounded transition-all cursor-pointer ${
                          isHigh
                            ? 'bg-amber-400 text-slate-950 shadow-md ring-2 ring-amber-300 scale-105'
                            : 'bg-cyan-500/20 text-cyan-300 hover:bg-cyan-500/30 border border-cyan-500/40'
                        }`}
                        title={`Jump to Source [${cleanId}]`}
                      >
                        {cleanId}
                      </button>
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

  return (
    <div className="space-y-6 pb-12">
      {/* Studio Header */}
      <div className="border-b border-slate-800 pb-5">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 text-xs font-semibold uppercase tracking-wider rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                Phase 12.5 • Unified RAG
              </span>
              <span className="text-xs text-slate-500 font-mono">Auto-Routing & Grounded Multi-Modal Chat</span>
            </div>
            <h1 className="text-2xl font-bold text-slate-100 tracking-tight mt-1 flex items-center gap-2">
              <Compass className="w-6 h-6 text-cyan-400" />
              Unified RAG & Intelligent Retrieval Routing
            </h1>
            <p className="text-xs text-slate-400 mt-1 max-w-3xl leading-relaxed">
              Query dense text passages, structured tables, visual figures, and knowledge graphs seamlessly. The Intelligent Router audits index capabilities and directs questions to optimal retrieval engines.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400">Routing Mode:</span>
            <div className="inline-flex bg-slate-900 border border-slate-800 rounded-lg p-0.5 text-xs font-medium">
              <button
                type="button"
                onClick={() => setRoutingMode('auto')}
                className={`px-3 py-1 rounded-md transition-all ${
                  routingMode === 'auto'
                    ? 'bg-cyan-600 text-white shadow-sm font-semibold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Auto (Router)
              </button>
              <button
                type="button"
                onClick={() => setRoutingMode('manual')}
                className={`px-3 py-1 rounded-md transition-all ${
                  routingMode === 'manual'
                    ? 'bg-cyan-600 text-white shadow-sm font-semibold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Manual
              </button>
              <button
                type="button"
                onClick={() => setRoutingMode('comparative')}
                className={`px-3 py-1 rounded-md transition-all ${
                  routingMode === 'comparative'
                    ? 'bg-cyan-600 text-white shadow-sm font-semibold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Comparative
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Document Selector & Capability Readiness Bar */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 space-y-3">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-3">
          <div className="flex items-center gap-2 w-full md:w-auto">
            <label className="text-xs font-semibold text-slate-300 flex items-center gap-1.5 shrink-0">
              <FileText className="w-4 h-4 text-cyan-400" />
              Target Document:
            </label>
            <select
              value={selectedDocumentId}
              onChange={(e) => setSelectedDocumentId(e.target.value)}
              className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 w-full md:w-80"
            >
              <option value="">All Documents (Global Corpus)</option>
              {documents.map((doc) => (
                <option key={doc.id} value={doc.id}>
                  {doc.original_filename} ({doc.status})
                </option>
              ))}
            </select>
          </div>

          {/* Capabilities Badges */}
          {capabilities && (
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <span className="text-[11px] text-slate-500 font-medium">Index Readiness:</span>
              <span
                className={`px-2 py-0.5 rounded text-[11px] font-mono border flex items-center gap-1 ${
                  capabilities.has_text_chunks
                    ? 'bg-emerald-950/40 text-emerald-300 border-emerald-800/60'
                    : 'bg-slate-950 text-slate-500 border-slate-800'
                }`}
              >
                <Layers className="w-3 h-3" /> Chunks ({capabilities.chunk_count})
              </span>
              <span
                className={`px-2 py-0.5 rounded text-[11px] font-mono border flex items-center gap-1 ${
                  capabilities.has_tables
                    ? 'bg-amber-950/40 text-amber-300 border-amber-800/60'
                    : 'bg-slate-950 text-slate-500 border-slate-800'
                }`}
              >
                <Table className="w-3 h-3" /> Tables ({capabilities.table_count})
              </span>
              <span
                className={`px-2 py-0.5 rounded text-[11px] font-mono border flex items-center gap-1 ${
                  capabilities.has_page_images
                    ? 'bg-purple-950/40 text-purple-300 border-purple-800/60'
                    : 'bg-slate-950 text-slate-500 border-slate-800'
                }`}
              >
                <ImageIcon className="w-3 h-3" /> Images ({capabilities.page_count})
              </span>
              <span
                className={`px-2 py-0.5 rounded text-[11px] font-mono border flex items-center gap-1 ${
                  capabilities.has_knowledge_graph
                    ? 'bg-blue-950/40 text-blue-300 border-blue-800/60'
                    : 'bg-slate-950 text-slate-500 border-slate-800'
                }`}
              >
                <Network className="w-3 h-3" /> Graph ({capabilities.graph_evidence_count})
              </span>
            </div>
          )}
        </div>

        {/* Missing Index Guidance Banner */}
        {capabilities && selectedDocumentId && capabilities.recommended_actions?.length > 0 && (
          <div className="bg-amber-500/10 border border-amber-500/20 text-amber-200 px-3 py-2 rounded-lg text-xs flex items-center gap-2">
            <Info className="w-4 h-4 text-amber-400 shrink-0" />
            <span>
              {capabilities.recommended_actions[0]} (The router will automatically route queries to existing representations).
            </span>
          </div>
        )}
      </div>

      {/* Main Studio Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Configuration & Pipeline Parameters (3 cols) */}
        <div className="lg:col-span-3 space-y-4">
          <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 space-y-4 text-xs">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <span className="font-semibold text-slate-200 flex items-center gap-1.5">
                <Sliders className="w-4 h-4 text-cyan-400" />
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

            {/* Manual Route Selector (Visible when mode === 'manual') */}
            {routingMode === 'manual' && (
              <div className="space-y-2 pb-2 border-b border-slate-800">
                <label className="block text-slate-400 font-medium">Select Active Routes</label>
                <div className="space-y-1.5">
                  {[
                    { id: 'text_hybrid', label: 'Text Hybrid (RRF)' },
                    { id: 'text_dense', label: 'Text Dense KNN' },
                    { id: 'multimodal_table', label: 'Multimodal Table' },
                    { id: 'multimodal_visual', label: 'Visual & CLIP' },
                    { id: 'graph', label: 'Knowledge Graph' },
                    { id: 'agentic', label: 'Agentic Tools' },
                  ].map((rt) => (
                    <label key={rt.id} className="flex items-center gap-2 cursor-pointer text-slate-300">
                      <input
                        type="checkbox"
                        checked={manualRoutes.includes(rt.id)}
                        onChange={() => toggleManualRoute(rt.id)}
                        className="rounded bg-slate-950 border-slate-800 text-cyan-500 focus:ring-0"
                      />
                      <span>{rt.label}</span>
                    </label>
                  ))}
                </div>
              </div>
            )}

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
              <input
                type="text"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                placeholder="e.g. gemini-3.8-flash"
                className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-slate-200 font-mono"
              />
            </div>

            {/* Prompt Template */}
            <div>
              <label className="block text-slate-400 mb-1 font-medium">Prompt Template</label>
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
            </div>

            {/* Top-K */}
            <div>
              <div className="flex justify-between items-center mb-1">
                <label className="text-slate-400 font-medium">Top-K Evidence Items</label>
                <span className="font-mono text-cyan-400 font-semibold">{topK}</span>
              </div>
              <input
                type="range"
                min="1"
                max="10"
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="w-full accent-cyan-500"
              />
            </div>

            {/* Temperature */}
            <div>
              <div className="flex justify-between items-center mb-1">
                <label className="text-slate-400 font-medium">Temperature</label>
                <span className="font-mono text-cyan-400 font-semibold">{temperature}</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="1.0"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                className="w-full accent-cyan-500"
              />
            </div>
          </div>
        </div>

        {/* Center Column: Question, Answer, Router Rationale (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          {/* Question Input Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-3">
            <form onSubmit={handleSubmit} className="space-y-3">
              <div className="relative">
                <textarea
                  rows={3}
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Ask a question about regular text, tables, figures, or entity relations..."
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-1 focus:ring-cyan-500 resize-none font-sans"
                />
              </div>

              <div className="flex items-center justify-between pt-1">
                <div className="flex items-center gap-2">
                  <label className="text-xs text-slate-400 flex items-center gap-1.5 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={isStreaming}
                      onChange={(e) => setIsStreaming(e.target.checked)}
                      className="rounded bg-slate-950 border-slate-800 text-cyan-500 focus:ring-0"
                    />
                    SSE Streaming
                  </label>
                </div>

                <button
                  type="submit"
                  disabled={loading || !question.trim()}
                  className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-800 disabled:text-slate-600 text-white font-medium text-xs rounded-lg transition-colors flex items-center gap-2 shadow-sm cursor-pointer"
                >
                  {loading ? (
                    <>
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      Routing & Retrieving...
                    </>
                  ) : (
                    <>
                      <Send className="w-3.5 h-3.5" />
                      Run Unified RAG
                    </>
                  )}
                </button>
              </div>
            </form>

            {/* Presets */}
            <div className="pt-2 border-t border-slate-800/80">
              <div className="text-[11px] font-medium text-slate-500 mb-1.5 flex items-center gap-1">
                <Sparkles className="w-3 h-3 text-cyan-400" />
                Benchmark Example Queries:
              </div>
              <div className="flex flex-wrap gap-1.5">
                {PRESET_QUESTIONS.map((pq, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => {
                      setQuestion(pq);
                      if (pq.includes('Jupiter') || pq.includes('NIRCam')) {
                        const solarDoc = documents.find((d) => d.original_filename?.includes('solar_system'));
                        if (solarDoc) setSelectedDocumentId(solarDoc.id);
                      } else {
                        const cvDoc = documents.find((d) => d.original_filename?.includes('cv_yassir'));
                        if (cvDoc) setSelectedDocumentId(cvDoc.id);
                      }
                    }}
                    className="text-[11px] bg-slate-950 hover:bg-slate-800 text-slate-300 px-2 py-1 rounded border border-slate-800 text-left truncate max-w-full"
                  >
                    {pq}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Error Banner */}
          {error && (
            <div className="bg-red-500/10 border border-red-500/30 text-red-300 p-4 rounded-xl text-xs flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
              <div>
                <span className="font-semibold block mb-1">Execution Error</span>
                <p className="text-slate-300">{error}</p>
              </div>
            </div>
          )}

          {/* Streaming Live Stage Banner */}
          {loading && streamingStage && (
            <div className="bg-slate-900/80 border border-cyan-500/30 p-3 rounded-xl flex items-center gap-3 text-xs text-cyan-300">
              <RefreshCw className="w-4 h-4 animate-spin text-cyan-400 shrink-0" />
              <span>{streamingStage}</span>
            </div>
          )}

          {/* Router Explanation Card */}
          {currentResponse?.routing && (
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-3.5 space-y-2 text-xs">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <span className="font-semibold text-slate-200 flex items-center gap-1.5">
                  <Compass className="w-4 h-4 text-cyan-400" />
                  Router Decision Trail
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800 uppercase">
                  {currentResponse.routing.mode} Mode
                </span>
              </div>

              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-slate-400 text-[11px]">Selected Routes:</span>
                {currentResponse.routing.selected_routes.map((rt) => (
                  <span
                    key={rt}
                    className="px-2 py-0.5 rounded text-[11px] font-mono bg-cyan-500/15 text-cyan-300 border border-cyan-500/30"
                  >
                    {rt}
                  </span>
                ))}
                {currentResponse.routing.effective_page_number && (
                  <span className="px-2 py-0.5 rounded text-[11px] font-mono bg-indigo-500/20 text-indigo-300 border border-indigo-500/40">
                    Page {currentResponse.routing.effective_page_number} Filter
                  </span>
                )}
              </div>

              <div className="text-[11px] text-slate-300 space-y-1">
                {currentResponse.routing.routing_reasons.map((r, i) => (
                  <div key={i} className="flex items-start gap-1.5">
                    <CheckCircle className="w-3 h-3 text-cyan-400 shrink-0 mt-0.5" />
                    <span>{r}</span>
                  </div>
                ))}
              </div>

              {currentResponse.routing.warnings?.length > 0 && (
                <div className="pt-1 text-[11px] text-amber-300 space-y-1">
                  {currentResponse.routing.warnings.map((w, i) => (
                    <div key={i} className="flex items-start gap-1.5">
                      <AlertTriangle className="w-3 h-3 text-amber-400 shrink-0 mt-0.5" />
                      <span>{w}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Comparative Results Preview (when active) */}
          {comparativeResults && (
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-3 text-xs">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <span className="font-semibold text-slate-200 flex items-center gap-1.5">
                  <Columns className="w-4 h-4 text-purple-400" />
                  Comparative Route Evaluation
                </span>
                <span className="text-[10px] font-mono text-slate-400">
                  Total: {comparativeResults.total_comparison_latency_ms} ms
                </span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                {comparativeResults.comparisons.map((c) => (
                  <div key={c.route} className="bg-slate-950 p-2.5 rounded-lg border border-slate-800 space-y-1">
                    <div className="flex items-center justify-between font-mono">
                      <span className="text-cyan-300 font-bold">{c.route}</span>
                      <span className="text-slate-400">{c.latency_ms} ms</span>
                    </div>
                    <div className="flex justify-between text-[11px] text-slate-400">
                      <span>Found: {c.evidence_count} items</span>
                      <span className="text-emerald-400 font-mono">Top: {c.top_score}</span>
                    </div>
                    <div className="text-[10px] text-slate-500 truncate">{c.snippet}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Generated Grounded Answer Card */}
          {(currentResponse || streamingAnswer) && (
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-3">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <div className="flex items-center gap-2">
                  <MessageSquare className="w-4 h-4 text-cyan-400" />
                  <span className="text-xs font-semibold text-slate-200">Grounded Synthesis</span>
                </div>
                <div className="flex items-center gap-2">
                  {currentResponse?.is_vision_enabled && (
                    <span className="text-[10px] font-mono bg-purple-950 text-purple-300 px-2 py-0.5 rounded border border-purple-800 flex items-center gap-1">
                      <Eye className="w-3 h-3" /> Vision Enabled
                    </span>
                  )}
                  <span className="text-[10px] font-mono bg-slate-950 px-2 py-0.5 rounded text-slate-400 border border-slate-800">
                    {currentResponse?.model_used || selectedModel}
                  </span>
                </div>
              </div>

              {/* Answer Content */}
              <div className="bg-slate-950 p-4 rounded-lg border border-slate-800/80">
                {renderFormattedAnswer(streamingAnswer || currentResponse?.answer)}
                {loading && (
                  <span className="inline-block w-2 h-4 bg-cyan-400 animate-pulse ml-1 align-middle" />
                )}
              </div>

              {/* Citation Verification Summary */}
              {currentResponse?.citations && currentResponse.citations.length > 0 && (
                <div className="pt-2 border-t border-slate-800/80 space-y-2 text-xs">
                  <div className="flex items-center justify-between text-slate-400 font-medium">
                    <span className="flex items-center gap-1.5">
                      <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
                      Verified Source Citations
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
                        <span>{c.origin_tag || c.document_name}</span>
                        {c.page_number && <span className="text-slate-500">(p.{c.page_number})</span>}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right Column: Retrieved Evidence Grouped by Modality (4 cols) */}
        <div className="lg:col-span-4 space-y-4">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <span className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
                <Database className="w-4 h-4 text-cyan-400" />
                Unified Evidence Pool ({currentResponse?.evidence_items?.length || 0})
              </span>
              <span className="text-[10px] text-slate-500 font-mono">Fused Top-K = {topK}</span>
            </div>

            {(!currentResponse || !currentResponse.evidence_items || currentResponse.evidence_items.length === 0) ? (
              <div className="p-8 text-center text-xs text-slate-500 space-y-2">
                <FileText className="w-8 h-8 mx-auto text-slate-700" />
                <p>No query executed yet.</p>
                <p className="text-[11px] text-slate-600">
                  Retrieved tables, figures, graph edges, and text passages with [S1], [S2] tags will appear here.
                </p>
              </div>
            ) : (
              <div className="space-y-3 max-h-[640px] overflow-y-auto pr-1">
                {currentResponse.evidence_items.map((item) => {
                  const cleanId = item.citation_id.replace(/[\[\]]/g, '');
                  const isHighlighted = highlightedSourceId === cleanId;
                  const isTable = item.evidence_type === 'table';
                  const isVisual = item.evidence_type === 'image' || item.evidence_type === 'figure';
                  const isGraph = item.evidence_type === 'graph';

                  return (
                    <div
                      key={item.evidence_id || item.citation_id}
                      ref={(el) => (sourcesRef.current[cleanId] = el)}
                      className={`p-3 rounded-lg border transition-all text-xs space-y-2 ${
                        isHighlighted
                          ? 'bg-amber-950/20 border-amber-500/60 ring-1 ring-amber-400'
                          : item.was_cited
                          ? 'bg-slate-950 border-emerald-800/40'
                          : 'bg-slate-950/60 border-slate-800'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span
                            className={`px-1.5 py-0.5 rounded text-[11px] font-mono font-bold ${
                              item.was_cited
                                ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                                : 'bg-slate-800 text-slate-400 border border-slate-700'
                            }`}
                          >
                            {item.citation_id}
                          </span>
                          <span className="px-1.5 py-0.5 rounded text-[10px] font-mono uppercase bg-slate-900 border border-slate-800 text-cyan-400">
                            {item.evidence_type}
                          </span>
                          <span className="text-slate-400 text-[10px] font-mono">{item.origin_tag}</span>
                        </div>
                        <div className="flex items-center gap-1.5 font-mono text-[10px]">
                          {item.page_number && <span className="text-slate-500">p.{item.page_number}</span>}
                          <span className="text-cyan-400 font-bold">{item.calibrated_score.toFixed(3)}</span>
                        </div>
                      </div>

                      {/* Content Preview based on Modality */}
                      {isTable ? (
                        <div className="bg-slate-900/90 p-2 rounded border border-slate-800 text-[11px] font-mono overflow-x-auto text-amber-200">
                          <div className="text-[10px] text-slate-500 mb-1 flex items-center justify-between">
                            <span>Structured Table Matrix</span>
                            {item.structured_data?.num_rows && (
                              <span>{item.structured_data.num_rows}r × {item.structured_data.num_cols}c</span>
                            )}
                          </div>
                          <pre className="whitespace-pre">{item.content}</pre>
                        </div>
                      ) : isVisual && item.image_url ? (
                        <div className="space-y-1.5">
                          <div className="h-32 bg-slate-900 rounded overflow-hidden border border-slate-800 flex items-center justify-center">
                            <img src={item.image_url} alt="Visual asset" className="max-h-full object-contain" />
                          </div>
                          <p className="text-[11px] text-slate-400">{item.content}</p>
                        </div>
                      ) : isGraph ? (
                        <div className="bg-blue-950/20 border border-blue-800/40 p-2 rounded text-[11px] font-mono text-blue-200 space-y-1">
                          <div className="text-[10px] text-blue-400 font-bold">Knowledge Graph Evidence</div>
                          <p>{item.content}</p>
                        </div>
                      ) : (
                        <p className="text-slate-300 text-[11px] leading-relaxed font-mono bg-slate-900/60 p-2 rounded border border-slate-800/60">
                          {item.content}
                        </p>
                      )}

                      <div className="flex items-center justify-between text-[10px] text-slate-500">
                        <span>Engine: {item.engine}</span>
                        {item.was_cited ? (
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

      {/* Pipeline Stage Timings Breakdown */}
      {currentResponse?.timing_metrics && (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-3">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold text-slate-200 flex items-center gap-2">
              <Clock className="w-4 h-4 text-cyan-400" />
              Pipeline Execution Latency Breakdown
            </span>
            <span className="font-mono text-cyan-400 font-bold">
              Total: {currentResponse.timing_metrics.total_latency_ms} ms
            </span>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-center text-xs font-mono">
            <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
              <div className="text-[10px] text-slate-500 font-sans">1. Routing</div>
              <div className="text-purple-300 font-bold text-sm mt-0.5">
                {currentResponse.timing_metrics.routing_latency_ms} ms
              </div>
            </div>
            <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
              <div className="text-[10px] text-slate-500 font-sans">2. Retrieval</div>
              <div className="text-blue-300 font-bold text-sm mt-0.5">
                {currentResponse.timing_metrics.retrieval_latency_ms} ms
              </div>
            </div>
            <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
              <div className="text-[10px] text-slate-500 font-sans">3. Fusion & Context</div>
              <div className="text-cyan-300 font-bold text-sm mt-0.5">
                {currentResponse.timing_metrics.fusion_latency_ms} ms
              </div>
            </div>
            <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
              <div className="text-[10px] text-slate-500 font-sans">4. Generation</div>
              <div className="text-emerald-300 font-bold text-sm mt-0.5">
                {currentResponse.timing_metrics.generation_latency_ms} ms
              </div>
            </div>
            <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
              <div className="text-[10px] text-slate-500 font-sans">5. Citation Validation</div>
              <div className="text-amber-300 font-bold text-sm mt-0.5">
                {currentResponse.timing_metrics.citation_validation_ms} ms
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default RagChat;
