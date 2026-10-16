import React, { useState, useEffect } from 'react';
import {
  Search,
  Layers,
  Cpu,
  Database,
  BookOpen,
  Clock,
  Sliders,
  BarChart3,
  CheckCircle,
  AlertCircle,
  AlertTriangle,
  Trash2,
  Sparkles,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  Play,
  Save,
  FileText,
  Tag,
  Zap,
  Info,
  Network,
  GitMerge,
  ArrowRight,
  Check,
  SlidersHorizontal,
  ListFilter,
  ShieldCheck,
} from 'lucide-react';
import api from '../services/api';

const PRESET_QUERIES = [
  "What are the candidate's core backend engineering competencies?",
  "How does microservice architecture improve system scalability?",
  "Explain vector similarity search and high-dimensional indexes.",
  "What are the differences between Raft and Paxos consensus?",
  "How does PostgreSQL pgvector perform nearest-neighbor retrieval?",
];

const RETRIEVAL_STRATEGIES = [
  { id: 'dense_only', label: '1. Dense Vector Only', desc: 'Exact KNN or HNSW cosine similarity search' },
  { id: 'lexical_only', label: '2. Lexical Only (BM25 / GIN)', desc: 'PostgreSQL full-text tsvector search with ts_rank_cd' },
  { id: 'hybrid_linear', label: '3. Hybrid Linear Combination', desc: 'Normalized weighted blend: α · Dense + (1-α) · Lexical' },
  { id: 'hybrid_rrf', label: '4. Hybrid RRF', desc: 'Reciprocal Rank Fusion (k=60), rank-based calibration-free' },
  { id: 'dense_rerank', label: '5. Dense + Reranker', desc: 'Dense vector retrieval followed by Cross-Encoder TinyBERT' },
  { id: 'hybrid_rrf_rerank', label: '6. Hybrid RRF + Reranker', desc: 'Dual-channel RRF candidate pool re-scored by Cross-Encoder' },
  { id: 'multi_query_hybrid_rerank', label: '7. Multi-Query + Hybrid + Reranker', desc: 'Query expansion + multi-channel fusion + Cross-Encoder' },
  { id: 'adaptive_context', label: '8. Adaptive Context Selection', desc: 'Contiguous neighbor stitching, deduplication & score filtering' },
];

export function RetrievalLab() {
  const [activeTab, setActiveTab] = useState('semantic_search');

  // Shared state
  const [models, setModels] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [chunkExperiments, setChunkExperiments] = useState([]);
  const [selectedModel, setSelectedModel] = useState('sentence-transformers/all-MiniLM-L6-v2');

  // Search tab state
  const [searchQuery, setSearchQuery] = useState(PRESET_QUERIES[0]);
  const [searchAlgo, setSearchAlgo] = useState('exact'); // exact, hnsw, ivfflat
  const [distanceMetric, setDistanceMetric] = useState('cosine'); // cosine, l2, inner_product
  const [topK, setTopK] = useState(5);
  const [efSearch, setEfSearch] = useState(40);
  const [probes, setProbes] = useState(1);
  const [showFilters, setShowFilters] = useState(false);
  const [filterDocId, setFilterDocId] = useState('');
  const [filterChunkExpId, setFilterChunkExpId] = useState('');
  const [filterPage, setFilterPage] = useState('');
  const [showPlan, setShowPlan] = useState(true);

  // Phase 08 Advanced Retrieval State
  const [retrievalStrategy, setRetrievalStrategy] = useState('dense_only');
  const [hybridAlpha, setHybridAlpha] = useState(0.5);
  const [rrfK, setRrfK] = useState(60);
  const [rerankerModel, setRerankerModel] = useState('cross-encoder/ms-marco-TinyBERT-L-2-v2');
  const [rerankTopN, setRerankTopN] = useState(5);
  const [searchLanguage, setSearchLanguage] = useState('english');
  const [enableQueryExpansion, setEnableQueryExpansion] = useState(false);
  const [enableNeighborExpansion, setEnableNeighborExpansion] = useState(false);
  const [showProgressionTable, setShowProgressionTable] = useState(true);

  // RRF Simulator State in Learning Center
  const [simRankDense, setSimRankDense] = useState(1);
  const [simRankLexical, setSimRankLexical] = useState(15);
  const [simRankDenseB, setSimRankDenseB] = useState(8);
  const [simRankLexicalB, setSimRankLexicalB] = useState(9);
  const [simK, setSimK] = useState(60);

  // Search Results state
  const [searching, setSearching] = useState(false);
  const [searchResponse, setSearchResponse] = useState(null);
  const [searchError, setSearchError] = useState(null);

  // Index Explorer state
  const [indexes, setIndexes] = useState([]);
  const [loadingIndexes, setLoadingIndexes] = useState(false);
  const [creatingIndex, setCreatingIndex] = useState(false);
  const [newIndexType, setNewIndexType] = useState('hnsw');
  const [newIndexMetric, setNewIndexMetric] = useState('cosine');
  const [hnswM, setHnswM] = useState(16);
  const [hnswEfC, setHnswEfC] = useState(64);
  const [ivfLists, setIvfLists] = useState(10);
  const [indexActionMsg, setIndexActionMsg] = useState(null);

  // Benchmark Lab state
  const [benchmarking, setBenchmarking] = useState(false);
  const [benchmarkResponse, setBenchmarkResponse] = useState(null);
  const [benchmarkRuns, setBenchmarkRuns] = useState(5);
  const [generatingSynthetic, setGeneratingSynthetic] = useState(false);
  const [syntheticStatus, setSyntheticStatus] = useState(null);

  // Experiment History state
  const [experiments, setExperiments] = useState([]);
  const [loadingExperiments, setLoadingExperiments] = useState(false);
  const [savingExperiment, setSavingExperiment] = useState(false);
  const [expSaveSuccess, setExpSaveSuccess] = useState(null);

  // Initial load
  useEffect(() => {
    loadModels();
    loadDocuments();
    loadIndexes();
    loadExperiments();
  }, []);

  useEffect(() => {
    if (filterDocId) {
      loadChunkExperiments(filterDocId);
    } else {
      setChunkExperiments([]);
      setFilterChunkExpId('');
    }
  }, [filterDocId]);

  const loadModels = async () => {
    try {
      const res = await api.listModels();
      const items = res?.items || (Array.isArray(res) ? res : []);
      setModels(items);
      if (items.length > 0 && !selectedModel) {
        setSelectedModel(items[0].model_id);
      }
    } catch (err) {
      console.error('Failed to load models:', err);
    }
  };

  const loadDocuments = async () => {
    try {
      const res = await api.getDocuments();
      const items = res?.items || (Array.isArray(res) ? res : []);
      setDocuments(items);
    } catch (err) {
      console.error('Failed to load documents:', err);
    }
  };

  const loadChunkExperiments = async (docId) => {
    try {
      const res = await api.getChunkingExperiments(docId);
      const items = res?.items || (Array.isArray(res) ? res : []);
      setChunkExperiments(items);
    } catch (err) {
      console.error('Failed to load chunking experiments:', err);
    }
  };

  const loadIndexes = async () => {
    setLoadingIndexes(true);
    try {
      const res = await api.listIndexes();
      setIndexes(Array.isArray(res) ? res : []);
    } catch (err) {
      console.error('Failed to load indexes:', err);
    } finally {
      setLoadingIndexes(false);
    }
  };

  const loadExperiments = async () => {
    setLoadingExperiments(true);
    try {
      const res = await api.listRetrievalExperiments();
      setExperiments(Array.isArray(res) ? res : []);
    } catch (err) {
      console.error('Failed to load experiments:', err);
    } finally {
      setLoadingExperiments(false);
    }
  };

  // Perform vector search
  const handleSearch = async (overrideQuery = null) => {
    const q = overrideQuery || searchQuery;
    if (!q || !q.trim()) return;

    setSearching(true);
    setSearchError(null);

    const payload = {
      query: q,
      model_id: selectedModel,
      retrieval_strategy: retrievalStrategy,
      search_algorithm: searchAlgo,
      distance_metric: distanceMetric,
      top_k: Number(topK),
      ef_search: Number(efSearch),
      probes: Number(probes),
      explain_plan: true,
      filters: {},
      hybrid_alpha: Number(hybridAlpha),
      rrf_k: Number(rrfK),
      search_language: searchLanguage,
      reranker_model: rerankerModel,
      rerank_top_n: Number(rerankTopN),
      enable_query_expansion: Boolean(enableQueryExpansion),
      enable_neighbor_expansion: Boolean(enableNeighborExpansion),
    };

    if (filterDocId) payload.filters.document_id = filterDocId;
    if (filterChunkExpId) payload.filters.chunking_experiment_id = filterChunkExpId;
    if (filterPage && !isNaN(Number(filterPage))) payload.filters.page_number = Number(filterPage);

    try {
      const res = await api.searchVectors(payload);
      setSearchResponse(res);
    } catch (err) {
      setSearchError(err.message || 'Search execution failed');
      setSearchResponse(null);
    } finally {
      setSearching(false);
    }
  };

  // Create pgvector Index
  const handleCreateIndex = async () => {
    setCreatingIndex(true);
    setIndexActionMsg(null);
    try {
      const params = {};
      if (newIndexType === 'hnsw') {
        params.m = Number(hnswM);
        params.ef_construction = Number(hnswEfC);
      } else {
        params.lists = Number(ivfLists);
      }

      const res = await api.createIndex({
        index_type: newIndexType,
        distance_metric: newIndexMetric,
        model_id: selectedModel,
        parameters: params,
      });

      setIndexActionMsg({
        type: 'success',
        text: `Index '${res.index_name}' successfully built in ${res.build_duration_ms}ms (${res.index_size_pretty})`,
      });
      loadIndexes();
    } catch (err) {
      setIndexActionMsg({
        type: 'error',
        text: err.message || 'Index creation failed',
      });
    } finally {
      setCreatingIndex(false);
    }
  };

  // Drop pgvector Index
  const handleDropIndex = async (indexId) => {
    if (!window.confirm('Are you sure you want to drop this PostgreSQL index?')) return;
    try {
      await api.dropIndex(indexId);
      loadIndexes();
    } catch (err) {
      alert(`Failed to drop index: ${err.message}`);
    }
  };

  // Run Benchmark
  const handleRunBenchmark = async () => {
    setBenchmarking(true);
    try {
      const res = await api.runRetrievalBenchmark({
        query: searchQuery,
        model_id: selectedModel,
        distance_metric: distanceMetric,
        top_k: Number(topK),
        hnsw_ef_search: Number(efSearch),
        ivfflat_probes: Number(probes),
        runs: Number(benchmarkRuns),
      });
      setBenchmarkResponse(res);
    } catch (err) {
      alert(`Benchmark failed: ${err.message}`);
    } finally {
      setBenchmarking(false);
    }
  };

  // Generate Synthetic Dataset
  const handleGenerateSynthetic = async () => {
    setGeneratingSynthetic(true);
    setSyntheticStatus(null);
    try {
      const res = await api.generateSyntheticDataset(selectedModel, 300);
      setSyntheticStatus({
        type: 'success',
        text: `Generated 300 synthetic benchmark vectors for model '${selectedModel}' in ${res.embedding_latency_ms}ms.`,
      });
    } catch (err) {
      setSyntheticStatus({
        type: 'error',
        text: err.message || 'Synthetic generation failed',
      });
    } finally {
      setGeneratingSynthetic(false);
    }
  };

  // Save Search Run as Experiment
  const handleSaveExperiment = async () => {
    if (!searchResponse) return;
    setSavingExperiment(true);
    setExpSaveSuccess(null);
    try {
      const name = `Search: "${searchResponse.query.slice(0, 30)}..." [${searchResponse.search_algorithm}]`;
      await api.saveRetrievalExperiment({
        name,
        query_text: searchResponse.query,
        model_id: searchResponse.model_id,
        distance_metric: searchResponse.distance_metric,
        top_k: searchResponse.top_k,
        filters: searchResponse.filters || {},
        results: (searchResponse.results || []).map((r) => ({
          chunk_id: r.chunk_id,
          doc: r.document_name,
          score: r.similarity_score,
          dist: r.raw_distance,
        })),
        metrics: {
          search_algorithm: searchResponse.search_algorithm,
          query_latency_ms: searchResponse.query_latency_ms,
          total_latency_ms: searchResponse.total_latency_ms,
          plan_node_type: searchResponse.execution_plan?.node_type || 'Unknown',
        },
      });
      setExpSaveSuccess('Experiment saved successfully!');
      loadExperiments();
      setTimeout(() => setExpSaveSuccess(null), 3000);
    } catch (err) {
      alert(`Failed to save experiment: ${err.message}`);
    } finally {
      setSavingExperiment(false);
    }
  };

  const handleDeleteExperiment = async (expId) => {
    if (!window.confirm('Delete this historical retrieval experiment?')) return;
    try {
      await api.deleteRetrievalExperiment(expId);
      loadExperiments();
    } catch (err) {
      alert(`Failed to delete experiment: ${err.message}`);
    }
  };

  return (
    <div className="flex-1 overflow-y-auto bg-slate-950 p-6 space-y-6 text-slate-100">
      {/* Header */}
      <div className="flex items-center justify-between pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
              <Search className="w-6 h-6 text-blue-500" />
              Vector Database & Semantic Search Lab
            </h1>
            <span className="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20">
              Phase 05
            </span>
          </div>
          <p className="text-sm text-slate-400 mt-1">
            Master exact vs approximate nearest neighbor search (HNSW, IVFFlat), pgvector operators, execution plans, and retrieval recall.
          </p>
        </div>

        {/* Global Model Selector */}
        <div className="flex items-center gap-2 bg-slate-900 border border-slate-800 rounded-lg p-2">
          <Cpu className="w-4 h-4 text-blue-400" />
          <span className="text-xs text-slate-400">Embedding Model:</span>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="bg-slate-950 border border-slate-700 text-xs rounded px-2 py-1 text-slate-200 focus:outline-none focus:border-blue-500"
          >
            {models.map((m) => (
              <option key={m.model_id} value={m.model_id}>
                {m.display_name} ({m.dimension}d)
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Laboratory Navigation Tabs */}
      <div className="flex border-b border-slate-800 space-x-1">
        {[
          { id: 'semantic_search', label: 'Semantic Search', icon: Search },
          { id: 'index_explorer', label: 'Index Explorer (HNSW / IVFFlat)', icon: Database },
          { id: 'benchmark_lab', label: 'Algorithm Comparison & Recall', icon: BarChart3 },
          { id: 'learning_center', label: 'Learning Center', icon: BookOpen },
          { id: 'experiment_history', label: 'Experiment History', icon: Clock },
        ].map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2.5 text-xs font-medium border-b-2 transition-colors ${
                isActive
                  ? 'border-blue-500 text-blue-400 bg-blue-500/5'
                  : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-700'
              }`}
            >
              <Icon className="w-4 h-4" />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* TAB 1: SEMANTIC SEARCH */}
      {activeTab === 'semantic_search' && (
        <div className="space-y-6">
          {/* Strategy Selector Header Bar */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <GitMerge className="w-4 h-4 text-blue-400" />
                <span className="text-xs font-semibold text-slate-200 uppercase tracking-wide">
                  Retrieval Strategy Architecture (Phase 08)
                </span>
              </div>
              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-blue-950 text-blue-300 border border-blue-800">
                Active: {retrievalStrategy}
              </span>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {RETRIEVAL_STRATEGIES.map((strat) => {
                const isSelected = retrievalStrategy === strat.id;
                return (
                  <button
                    key={strat.id}
                    onClick={() => setRetrievalStrategy(strat.id)}
                    className={`text-left p-2.5 rounded-lg border transition-all ${
                      isSelected
                        ? 'bg-blue-600/15 border-blue-500/50 text-blue-200 shadow-md shadow-blue-500/5'
                        : 'bg-slate-950/60 border-slate-800/80 hover:bg-slate-900 hover:border-slate-700 text-slate-400'
                    }`}
                  >
                    <div className="text-xs font-semibold flex items-center justify-between">
                      <span>{strat.label}</span>
                      {isSelected && <Check className="w-3.5 h-3.5 text-blue-400" />}
                    </div>
                    <div className="text-[10px] text-slate-500 mt-1 line-clamp-1">
                      {strat.desc}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Query Input Card */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
            <div className="space-y-2">
              <label className="text-xs font-medium text-slate-300 flex items-center justify-between">
                <span>Natural Language Query</span>
                <span className="text-[11px] text-slate-500">
                  {retrievalStrategy.includes('lexical') || retrievalStrategy.includes('hybrid')
                    ? 'Evaluated simultaneously across pgvector and PostgreSQL GIN tsvector'
                    : 'Evaluated against dense vector embeddings'}
                </span>
              </label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                  placeholder="Enter a search question..."
                  className="flex-1 bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-100 placeholder:text-slate-600 focus:outline-none focus:border-blue-500"
                />
                <button
                  onClick={() => handleSearch()}
                  disabled={searching}
                  className="px-5 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded-lg text-xs font-medium flex items-center gap-2 shadow-lg shadow-blue-500/10 transition-colors"
                >
                  {searching ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
                  Execute Search
                </button>
              </div>

              {/* Preset Query Chips */}
              <div className="flex flex-wrap items-center gap-1.5 pt-1">
                <span className="text-[11px] text-slate-500 mr-1 flex items-center gap-1">
                  <Sparkles className="w-3 h-3 text-amber-400" /> Presets:
                </span>
                {PRESET_QUERIES.map((preset, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      setSearchQuery(preset);
                      handleSearch(preset);
                    }}
                    className="text-[11px] px-2.5 py-1 rounded bg-slate-800/60 hover:bg-slate-800 text-slate-300 border border-slate-700/50 transition-colors"
                  >
                    {preset.slice(0, 42)}...
                  </button>
                ))}
              </div>
            </div>

            {/* Retrieval Controls Grid */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 pt-3 border-t border-slate-800">
              {/* Dense Algorithm (relevant for dense, hybrid, rerank) */}
              {retrievalStrategy !== 'lexical_only' && (
                <div>
                  <label className="text-xs font-medium text-slate-300 block mb-1.5">Dense Index Scan</label>
                  <select
                    value={searchAlgo}
                    onChange={(e) => setSearchAlgo(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 text-xs rounded-lg p-2 text-slate-200 focus:outline-none focus:border-blue-500"
                  >
                    <option value="exact">Exact KNN (Seq Scan)</option>
                    <option value="hnsw">HNSW (Graph Scan)</option>
                    <option value="ivfflat">IVFFlat (Inverted Cells)</option>
                  </select>
                </div>
              )}

              {/* Distance Metric */}
              {retrievalStrategy !== 'lexical_only' && (
                <div>
                  <label className="text-xs font-medium text-slate-300 block mb-1.5">Distance Operator</label>
                  <select
                    value={distanceMetric}
                    onChange={(e) => setDistanceMetric(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 text-xs rounded-lg p-2 text-slate-200 focus:outline-none focus:border-blue-500"
                  >
                    <option value="cosine">Cosine Distance (&lt;=&gt;)</option>
                    <option value="l2">Euclidean L2 Distance (&lt;-&gt;)</option>
                    <option value="inner_product">Inner Product (&lt;#&gt;)</option>
                  </select>
                </div>
              )}

              {/* Lexical Search Language */}
              {(retrievalStrategy.includes('lexical') || retrievalStrategy.includes('hybrid') || retrievalStrategy === 'adaptive_context') && (
                <div>
                  <label className="text-xs font-medium text-slate-300 block mb-1.5">Lexical GIN Language</label>
                  <select
                    value={searchLanguage}
                    onChange={(e) => setSearchLanguage(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 text-xs rounded-lg p-2 text-slate-200 focus:outline-none focus:border-blue-500"
                  >
                    <option value="english">English (tsvector)</option>
                    <option value="french">French (tsvector)</option>
                    <option value="simple">Simple (No Stemming)</option>
                  </select>
                </div>
              )}

              {/* Top-K */}
              <div>
                <div className="flex justify-between items-center mb-1.5">
                  <label className="text-xs font-medium text-slate-300">Top-K Result Pool</label>
                  <span className="text-xs font-mono text-blue-400 font-semibold">{topK}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="30"
                  value={topK}
                  onChange={(e) => setTopK(e.target.value)}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
              </div>

              {/* Hybrid Linear Alpha Slider */}
              {retrievalStrategy === 'hybrid_linear' && (
                <div>
                  <div className="flex justify-between items-center mb-1.5">
                    <label className="text-xs font-medium text-slate-300">Alpha Weight (α)</label>
                    <span className="text-xs font-mono text-purple-400 font-semibold">
                      {(hybridAlpha * 100).toFixed(0)}% Dense / {((1 - hybridAlpha) * 100).toFixed(0)}% Lex
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.0"
                    max="1.0"
                    step="0.05"
                    value={hybridAlpha}
                    onChange={(e) => setHybridAlpha(parseFloat(e.target.value))}
                    className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-purple-500"
                  />
                </div>
              )}

              {/* RRF K Parameter */}
              {(retrievalStrategy.includes('rrf') || retrievalStrategy === 'multi_query_hybrid_rerank') && (
                <div>
                  <div className="flex justify-between items-center mb-1.5">
                    <label className="text-xs font-medium text-slate-300">RRF Constant (k)</label>
                    <span className="text-xs font-mono text-emerald-400 font-semibold">{rrfK}</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="100"
                    value={rrfK}
                    onChange={(e) => setRrfK(parseInt(e.target.value))}
                    className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                  />
                </div>
              )}

              {/* Cross-Encoder Model */}
              {(retrievalStrategy.includes('rerank') || retrievalStrategy === 'adaptive_context') && (
                <div>
                  <label className="text-xs font-medium text-slate-300 block mb-1.5">Reranker Model</label>
                  <select
                    value={rerankerModel}
                    onChange={(e) => setRerankerModel(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 text-xs rounded-lg p-2 text-slate-200 focus:outline-none focus:border-blue-500"
                  >
                    <option value="cross-encoder/ms-marco-TinyBERT-L-2-v2">TinyBERT-L-2-v2 (CPU Fast)</option>
                    <option value="mock">Mock Reranker (Deterministic)</option>
                    <option value="none">No-Op / Bypass</option>
                  </select>
                </div>
              )}
            </div>

            {/* Advanced Phase 08 Toggles */}
            <div className="flex flex-wrap items-center gap-6 pt-3 border-t border-slate-800/60 text-xs text-slate-300">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={enableQueryExpansion}
                  onChange={(e) => setEnableQueryExpansion(e.target.checked)}
                  className="rounded bg-slate-950 border-slate-800 text-blue-600 focus:ring-0"
                />
                <span>Multi-Query Expansion (Rewrite acronyms & entities)</span>
              </label>

              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={enableNeighborExpansion}
                  onChange={(e) => setEnableNeighborExpansion(e.target.checked)}
                  className="rounded bg-slate-950 border-slate-800 text-blue-600 focus:ring-0"
                />
                <span>Contiguous Neighbor Expansion (Stitch adjacent chunks)</span>
              </label>
            </div>

            {/* Filter Toggle Header */}
            <div className="pt-2">
              <button
                onClick={() => setShowFilters(!showFilters)}
                className="text-xs text-slate-400 hover:text-slate-200 flex items-center gap-1.5"
              >
                <Sliders className="w-3.5 h-3.5 text-blue-400" />
                {showFilters ? 'Hide Metadata Filters' : 'Show Metadata Filters (Document, Page, Experiment)'}
                {showFilters ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
              </button>

              {showFilters && (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-3 mt-2 border-t border-slate-800/60">
                  <div>
                    <label className="text-[11px] text-slate-400 block mb-1">Filter by Document</label>
                    <select
                      value={filterDocId}
                      onChange={(e) => setFilterDocId(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 text-xs rounded p-2 text-slate-200"
                    >
                      <option value="">All Ingested Documents</option>
                      {documents.map((d) => (
                        <option key={d.id} value={d.id}>
                          {d.original_filename}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="text-[11px] text-slate-400 block mb-1">Filter by Chunking Strategy</label>
                    <select
                      value={filterChunkExpId}
                      onChange={(e) => setFilterChunkExpId(e.target.value)}
                      disabled={!filterDocId || chunkExperiments.length === 0}
                      className="w-full bg-slate-950 border border-slate-800 text-xs rounded p-2 text-slate-200 disabled:opacity-50"
                    >
                      <option value="">All Strategies</option>
                      {chunkExperiments.map((c) => (
                        <option key={c.id} value={c.id}>
                          {c.strategy} ({c.metrics?.chunk_count || 0} chunks)
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="text-[11px] text-slate-400 block mb-1">Filter by Page Reference</label>
                    <input
                      type="number"
                      min="1"
                      placeholder="e.g. 1"
                      value={filterPage}
                      onChange={(e) => setFilterPage(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 text-xs rounded p-2 text-slate-200"
                    />
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Search Error Alert */}
          {searchError && (
            <div className="bg-red-500/10 border border-red-500/20 text-red-300 p-4 rounded-xl flex items-center gap-3">
              <AlertCircle className="w-5 h-5 text-red-400 shrink-0" />
              <div className="text-xs">{searchError}</div>
            </div>
          )}

          {/* Warnings Banner */}
          {searchResponse?.warnings?.length > 0 && (
            <div className="bg-amber-500/10 border border-amber-500/20 text-amber-300 p-3.5 rounded-xl space-y-1">
              <div className="flex items-center gap-2 font-medium text-xs text-amber-400">
                <AlertTriangle className="w-4 h-4 shrink-0" /> Educational Notice:
              </div>
              {searchResponse.warnings.map((w, idx) => (
                <div key={idx} className="text-xs text-amber-200/90 pl-6">
                  {w}
                </div>
              ))}
            </div>
          )}

          {/* Results Summary & Save */}
          {searchResponse && (
            <div className="space-y-4">
              <div className="flex items-center justify-between bg-slate-900 border border-slate-800 rounded-xl px-4 py-3">
                <div className="flex items-center gap-3">
                  <span className="text-xs text-slate-400">
                    Retrieved <strong className="text-white">{searchResponse.results.length}</strong> chunks in
                  </span>
                  <span className="px-2 py-0.5 rounded text-xs font-mono bg-blue-500/20 text-blue-300 border border-blue-500/30">
                    Query: {searchResponse.query_latency_ms}ms
                  </span>
                  <span className="px-2 py-0.5 rounded text-xs font-mono bg-slate-800 text-slate-300 border border-slate-700">
                    Encode: {searchResponse.embedding_latency_ms}ms
                  </span>
                  <span className="px-2 py-0.5 rounded text-xs font-mono bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                    Total: {searchResponse.total_latency_ms}ms
                  </span>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setShowPlan(!showPlan)}
                    className="text-xs px-2.5 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 flex items-center gap-1.5"
                  >
                    <Cpu className="w-3.5 h-3.5 text-blue-400" />
                    {showPlan ? 'Hide EXPLAIN Plan' : 'Inspect EXPLAIN Plan'}
                  </button>
                  <button
                    onClick={handleSaveExperiment}
                    disabled={savingExperiment}
                    className="text-xs px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 flex items-center gap-1.5"
                  >
                    <Save className="w-3.5 h-3.5 text-emerald-400" />
                    {savingExperiment ? 'Saving...' : 'Save Run'}
                  </button>
                </div>
              </div>

              {expSaveSuccess && (
                <div className="bg-emerald-500/10 border border-emerald-500/20 text-emerald-300 px-4 py-2 rounded-lg text-xs flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-emerald-400" />
                  {expSaveSuccess}
                </div>
              )}

              {/* Execution Plan Card */}
              {showPlan && searchResponse.execution_plan && (
                <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
                      <Cpu className="w-4 h-4 text-emerald-400" />
                      PostgreSQL Query Execution Plan (EXPLAIN ANALYZE)
                    </span>
                    <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                      Node: <strong className="text-blue-400">{searchResponse.execution_plan.node_type}</strong>
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 font-mono">
                    {searchResponse.execution_plan.plan_tree_summary}
                  </p>
                  <div className="grid grid-cols-4 gap-2 pt-2 text-[11px] text-slate-400 font-mono">
                    <div>Planning: {searchResponse.execution_plan.planning_time_ms}ms</div>
                    <div>Execution: {searchResponse.execution_plan.execution_time_ms}ms</div>
                    <div>Rows: {searchResponse.execution_plan.actual_rows}</div>
                    <div>Est. Cost: {searchResponse.execution_plan.total_cost}</div>
                  </div>
                </div>
              )}

              {/* Phase 08 Pipeline Visualizer */}
              <div className="bg-slate-950/90 border border-slate-800 rounded-xl p-4 space-y-3">
                <div className="flex items-center justify-between pb-2 border-b border-slate-800/80">
                  <div className="flex items-center gap-2">
                    <Layers className="w-4 h-4 text-indigo-400" />
                    <span className="text-xs font-semibold text-slate-200 uppercase tracking-wider">
                      Multi-Stage Retrieval Pipeline Execution
                    </span>
                  </div>
                  <span className="text-[10px] font-mono text-slate-400">
                    Strategy: <strong className="text-indigo-300">{searchResponse.retrieval_strategy}</strong>
                  </span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-5 gap-2 text-xs">
                  {/* Stage 1: Query Processing */}
                  <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800 space-y-1">
                    <div className="text-[10px] font-mono uppercase text-slate-400 flex items-center justify-between">
                      <span>1. Query Rewrite</span>
                      <span className="text-blue-400 font-semibold">
                        {searchResponse.timing_breakdown?.query_expansion_ms || 0}ms
                      </span>
                    </div>
                    <div className="text-[11px] text-slate-300 font-medium truncate">
                      {searchResponse.expanded_queries?.length > 1
                        ? `${searchResponse.expanded_queries.length} queries generated`
                        : 'Single query pass'}
                    </div>
                    {searchResponse.expanded_queries?.length > 1 && (
                      <div className="text-[10px] text-slate-500 truncate">
                        {searchResponse.expanded_queries.slice(1).join(' | ')}
                      </div>
                    )}
                  </div>

                  {/* Stage 2: Dual Channel Retrieval */}
                  <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800 space-y-1">
                    <div className="text-[10px] font-mono uppercase text-slate-400 flex items-center justify-between">
                      <span>2. Dual Retrieval</span>
                      <span className="text-emerald-400 font-semibold">
                        {((searchResponse.timing_breakdown?.dense_retrieval_ms || 0) + (searchResponse.timing_breakdown?.lexical_retrieval_ms || 0)).toFixed(1)}ms
                      </span>
                    </div>
                    <div className="flex items-center gap-1.5 text-[10px]">
                      <span className={`px-1.5 py-0.5 rounded font-mono ${
                        searchResponse.retrieval_strategy !== 'lexical_only'
                          ? 'bg-blue-950 text-blue-400 border border-blue-800'
                          : 'bg-slate-800 text-slate-500'
                      }`}>
                        Dense ({searchResponse.timing_breakdown?.dense_retrieval_ms || 0}ms)
                      </span>
                      <span className={`px-1.5 py-0.5 rounded font-mono ${
                        searchResponse.retrieval_strategy.includes('lexical') || searchResponse.retrieval_strategy.includes('hybrid') || searchResponse.retrieval_strategy === 'adaptive_context'
                          ? 'bg-purple-950 text-purple-400 border border-purple-800'
                          : 'bg-slate-800 text-slate-500'
                      }`}>
                        GIN ({searchResponse.timing_breakdown?.lexical_retrieval_ms || 0}ms)
                      </span>
                    </div>
                  </div>

                  {/* Stage 3: Fusion */}
                  <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800 space-y-1">
                    <div className="text-[10px] font-mono uppercase text-slate-400 flex items-center justify-between">
                      <span>3. Fusion Engine</span>
                      <span className="text-amber-400 font-semibold">
                        {searchResponse.timing_breakdown?.fusion_ms || 0}ms
                      </span>
                    </div>
                    <div className="text-[11px] text-slate-300 font-mono">
                      {searchResponse.retrieval_strategy.includes('rrf')
                        ? `RRF (k=${rrfK})`
                        : searchResponse.retrieval_strategy === 'hybrid_linear'
                        ? `Linear (α=${hybridAlpha})`
                        : 'Single channel pass'}
                    </div>
                  </div>

                  {/* Stage 4: Reranking */}
                  <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800 space-y-1">
                    <div className="text-[10px] font-mono uppercase text-slate-400 flex items-center justify-between">
                      <span>4. Cross-Encoder</span>
                      <span className="text-rose-400 font-semibold">
                        {searchResponse.timing_breakdown?.rerank_ms || 0}ms
                      </span>
                    </div>
                    <div className="text-[11px] text-slate-300 font-mono truncate">
                      {searchResponse.retrieval_strategy.includes('rerank') || searchResponse.retrieval_strategy === 'adaptive_context'
                        ? rerankerModel.split('/').pop()
                        : 'Bypassed'}
                    </div>
                  </div>

                  {/* Stage 5: Context Optimization */}
                  <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800 space-y-1">
                    <div className="text-[10px] font-mono uppercase text-slate-400 flex items-center justify-between">
                      <span>5. Context Stitched</span>
                      <span className="text-teal-400 font-semibold">
                        {searchResponse.timing_breakdown?.context_optimization_ms || 0}ms
                      </span>
                    </div>
                    <div className="text-[11px] text-slate-300 font-mono">
                      {enableNeighborExpansion ? 'Neighbor stitched' : 'Standard chunks'}
                    </div>
                  </div>
                </div>
              </div>

              {/* Candidate Progression Audit Table */}
              {searchResponse.candidate_progression?.length > 0 && (
                <div className="bg-slate-950/80 border border-slate-800 rounded-xl overflow-hidden">
                  <div className="p-3 border-b border-slate-800 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <ListFilter className="w-4 h-4 text-emerald-400" />
                      <span className="text-xs font-semibold text-slate-200">
                        Candidate Progression &amp; Score Audit ({searchResponse.candidate_progression.length} items)
                      </span>
                    </div>
                    <button
                      onClick={() => setShowProgressionTable(!showProgressionTable)}
                      className="text-[11px] text-slate-400 hover:text-slate-200"
                    >
                      {showProgressionTable ? 'Collapse Table' : 'Expand Table'}
                    </button>
                  </div>

                  {showProgressionTable && (
                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-xs font-mono">
                        <thead className="bg-slate-900/90 text-slate-400 text-[10px] uppercase border-b border-slate-800">
                          <tr>
                            <th className="py-2 px-3">Final</th>
                            <th className="py-2 px-3 font-sans">Document &amp; Chunk</th>
                            <th className="py-2 px-3">Lexical Rank</th>
                            <th className="py-2 px-3">Dense Rank</th>
                            <th className="py-2 px-3">Fusion Rank / Score</th>
                            <th className="py-2 px-3">Cross-Encoder</th>
                            <th className="py-2 px-3 text-right">Status</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800/60">
                          {searchResponse.candidate_progression.map((item, idx) => (
                            <tr key={idx} className="hover:bg-slate-900/40">
                              <td className="py-2 px-3 font-bold text-blue-400">#{item.final_rank}</td>
                              <td className="py-2 px-3 font-sans text-slate-300 text-[11px]">
                                {item.document_name} <span className="text-slate-500 font-mono text-[10px]">Seq #{item.sequence_index}</span>
                              </td>
                              <td className="py-2 px-3 text-purple-400">
                                {item.lexical_rank != null ? `#${item.lexical_rank} (${item.lexical_score?.toFixed(3)})` : '-'}
                              </td>
                              <td className="py-2 px-3 text-blue-400">
                                {item.dense_rank != null ? `#${item.dense_rank} (${item.dense_score?.toFixed(3)})` : '-'}
                              </td>
                              <td className="py-2 px-3 text-amber-400">
                                {item.fusion_rank != null ? `#${item.fusion_rank} (${item.fusion_score?.toFixed(4)})` : '-'}
                              </td>
                              <td className="py-2 px-3 text-rose-400">
                                {item.rerank_score != null ? `${item.rerank_score.toFixed(4)}` : '-'}
                              </td>
                              <td className="py-2 px-3 text-right">
                                <span className={`px-2 py-0.5 rounded text-[10px] ${
                                  item.status === 'selected'
                                    ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                                    : 'bg-slate-800 text-slate-500'
                                }`}>
                                  {item.status}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}

              {/* Retrieved Chunks List */}
              <div className="space-y-3">
                {searchResponse.results.length === 0 ? (
                  <div className="bg-slate-900 border border-dashed border-slate-800 rounded-xl p-8 text-center text-slate-500 text-xs">
                    No matching chunks found for the current query and filters.
                  </div>
                ) : (
                  searchResponse.results.map((item, idx) => (
                    <div
                      key={idx}
                      className="bg-slate-900/80 border border-slate-800 hover:border-slate-700 rounded-xl p-4 space-y-3 transition-colors"
                    >
                      {/* Chunk Header */}
                      <div className="flex items-center justify-between text-xs pb-2 border-b border-slate-800/80">
                        <div className="flex items-center gap-2">
                          <span className="font-bold text-blue-400 font-mono">#{idx + 1}</span>
                          <span className="font-medium text-slate-200 flex items-center gap-1">
                            <FileText className="w-3.5 h-3.5 text-slate-400" />
                            {item.document_name}
                          </span>
                          <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-400 text-[10px]">
                            {item.mime_type}
                          </span>
                          {item.page_number && (
                            <span className="px-2 py-0.5 rounded bg-blue-500/10 text-blue-300 border border-blue-500/20 text-[10px] font-mono">
                              Page {item.page_number}
                            </span>
                          )}
                          <span className="text-slate-500 text-[11px]">Seq #{item.sequence_index}</span>

                          {/* Phase 08 Channel & Neighbor Badges */}
                          {item.retrieval_channel && (
                            <span className="px-2 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-800 text-[10px] font-mono">
                              {item.retrieval_channel}
                            </span>
                          )}
                          {item.is_neighbor_expanded && (
                            <span className="px-2 py-0.5 rounded bg-teal-950 text-teal-300 border border-teal-800 text-[10px] font-mono">
                              Neighbor Stitched
                            </span>
                          )}
                        </div>

                        {/* Metric Scores */}
                        <div className="flex items-center gap-2 font-mono text-xs">
                          {item.lexical_score != null && (
                            <div className="text-right px-2 border-r border-slate-800">
                              <span className="text-[10px] text-slate-500 block uppercase font-mono">Lexical</span>
                              <span className="text-purple-400 font-bold">{item.lexical_score.toFixed(3)}</span>
                            </div>
                          )}
                          {item.dense_score != null && (
                            <div className="text-right px-2 border-r border-slate-800">
                              <span className="text-[10px] text-slate-500 block uppercase font-mono">Dense Sim</span>
                              <span className="text-blue-400 font-bold">{item.dense_score.toFixed(3)}</span>
                            </div>
                          )}
                          {item.rrf_score != null && (
                            <div className="text-right px-2 border-r border-slate-800">
                              <span className="text-[10px] text-slate-500 block uppercase font-mono">RRF</span>
                              <span className="text-amber-400 font-bold">{item.rrf_score.toFixed(4)}</span>
                            </div>
                          )}
                          {item.rerank_score != null && (
                            <div className="text-right px-2 border-r border-slate-800">
                              <span className="text-[10px] text-slate-500 block uppercase font-mono">Rerank</span>
                              <span className="text-rose-400 font-bold">{item.rerank_score.toFixed(4)}</span>
                            </div>
                          )}
                          <div className="text-right">
                            <span className="text-[10px] text-slate-500 block uppercase font-mono">
                              Final Score
                            </span>
                            <span className="text-xs font-bold font-mono text-emerald-400">
                              {item.similarity_score.toFixed(4)}
                            </span>
                          </div>
                        </div>
                      </div>

                      {/* Snippet Content */}
                      <p className="text-xs text-slate-300 leading-relaxed font-sans bg-slate-950/60 p-3 rounded-lg border border-slate-800/60 whitespace-pre-wrap">
                        {item.content}
                      </p>

                      <div className="flex items-center justify-between text-[11px] text-slate-500">
                        <span>Tokens: {item.token_count} &bull; Chars: {item.char_count}</span>
                        <span className="font-mono text-[10px] text-slate-600">ID: {item.chunk_id}</span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 2: INDEX EXPLORER */}
      {activeTab === 'index_explorer' && (
        <div className="space-y-6">
          {/* Index Creation Card */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
            <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <Database className="w-4 h-4 text-blue-400" />
              Build Native PostgreSQL pgvector Index
            </h3>
            <p className="text-xs text-slate-400">
              Create an HNSW proximity graph or IVFFlat inverted file index matching the active model's dimension.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div>
                <label className="text-xs font-medium text-slate-300 block mb-1">Index Algorithm</label>
                <select
                  value={newIndexType}
                  onChange={(e) => setNewIndexType(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 text-xs rounded p-2 text-slate-200"
                >
                  <option value="hnsw">HNSW (Hierarchical Navigable Small World)</option>
                  <option value="ivfflat">IVFFlat (Inverted File Flat)</option>
                </select>
              </div>

              <div>
                <label className="text-xs font-medium text-slate-300 block mb-1">Distance Operator Class</label>
                <select
                  value={newIndexMetric}
                  onChange={(e) => setNewIndexMetric(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 text-xs rounded p-2 text-slate-200"
                >
                  <option value="cosine">vector_cosine_ops (&lt;=&gt;)</option>
                  <option value="l2">vector_l2_ops (&lt;-&gt;)</option>
                  <option value="inner_product">vector_ip_ops (&lt;#&gt;)</option>
                </select>
              </div>

              {newIndexType === 'hnsw' && (
                <>
                  <div>
                    <label className="text-xs font-medium text-slate-300 block mb-1">
                      M (Max Edges / Node): <span className="text-blue-400 font-mono">{hnswM}</span>
                    </label>
                    <input
                      type="number"
                      min="4"
                      max="64"
                      value={hnswM}
                      onChange={(e) => setHnswM(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 text-xs rounded p-2 text-slate-200"
                    />
                  </div>
                  <div>
                    <label className="text-xs font-medium text-slate-300 block mb-1">
                      ef_construction: <span className="text-emerald-400 font-mono">{hnswEfC}</span>
                    </label>
                    <input
                      type="number"
                      min="16"
                      max="512"
                      value={hnswEfC}
                      onChange={(e) => setHnswEfC(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 text-xs rounded p-2 text-slate-200"
                    />
                  </div>
                </>
              )}

              {newIndexType === 'ivfflat' && (
                <div className="md:col-span-2">
                  <label className="text-xs font-medium text-slate-300 block mb-1">
                    lists (Number of Clusters): <span className="text-amber-400 font-mono">{ivfLists}</span>
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="500"
                    value={ivfLists}
                    onChange={(e) => setIvfLists(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 text-xs rounded p-2 text-slate-200"
                  />
                  <span className="text-[10px] text-slate-500 mt-1 block">
                    Recommended: lists ~ sqrt(N) for small datasets, or N/1000 for large collections.
                  </span>
                </div>
              )}
            </div>

            <div className="pt-2">
              <button
                onClick={handleCreateIndex}
                disabled={creatingIndex}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded-lg text-xs font-medium flex items-center gap-2"
              >
                {creatingIndex ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                {creatingIndex ? 'Building Index in PostgreSQL...' : 'Build Index'}
              </button>
            </div>

            {indexActionMsg && (
              <div
                className={`p-3 rounded-lg text-xs flex items-center gap-2 ${
                  indexActionMsg.type === 'success'
                    ? 'bg-emerald-500/10 border border-emerald-500/20 text-emerald-300'
                    : 'bg-red-500/10 border border-red-500/20 text-red-300'
                }`}
              >
                {indexActionMsg.type === 'success' ? (
                  <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0" />
                ) : (
                  <AlertCircle className="w-4 h-4 text-red-400 shrink-0" />
                )}
                {indexActionMsg.text}
              </div>
            )}
          </div>

          {/* Active Indexes Table */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
            <div className="px-5 py-4 border-b border-slate-800 flex items-center justify-between">
              <div>
                <h3 className="text-sm font-semibold text-slate-200">Active PostgreSQL pgvector Indexes</h3>
                <p className="text-xs text-slate-400">
                  Managed physical indexes on the <code>chunk_embeddings</code> table.
                </p>
              </div>
              <button
                onClick={loadIndexes}
                disabled={loadingIndexes}
                className="text-xs px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 flex items-center gap-1.5"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${loadingIndexes ? 'animate-spin' : ''}`} />
                Refresh
              </button>
            </div>

            {indexes.length === 0 ? (
              <div className="p-8 text-center text-slate-500 text-xs">
                No custom ANN indexes created yet. Exact KNN sequential search is used by default.
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-950 text-slate-400 uppercase font-mono text-[10px]">
                    <tr>
                      <th className="px-4 py-3">Index Name</th>
                      <th className="px-4 py-3">Type</th>
                      <th className="px-4 py-3">Operator</th>
                      <th className="px-4 py-3">Dimension</th>
                      <th className="px-4 py-3">Parameters</th>
                      <th className="px-4 py-3">Size on Disk</th>
                      <th className="px-4 py-3">Build Latency</th>
                      <th className="px-4 py-3 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800">
                    {indexes.map((idx) => (
                      <tr key={idx.id} className="hover:bg-slate-800/40 font-mono">
                        <td className="px-4 py-3 font-semibold text-blue-300">{idx.index_name}</td>
                        <td className="px-4 py-3">
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                              idx.index_type === 'hnsw'
                                ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                                : 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                            }`}
                          >
                            {idx.index_type.toUpperCase()}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-slate-300">{idx.opclass}</td>
                        <td className="px-4 py-3 text-slate-300">{idx.vector_dimension}d</td>
                        <td className="px-4 py-3 text-slate-400 font-sans text-[11px]">
                          {JSON.stringify(idx.parameters)}
                        </td>
                        <td className="px-4 py-3 text-slate-200 font-bold">{idx.index_size_pretty}</td>
                        <td className="px-4 py-3 text-slate-400">{idx.build_duration_ms}ms</td>
                        <td className="px-4 py-3 text-right font-sans">
                          <button
                            onClick={() => handleDropIndex(idx.id)}
                            className="text-red-400 hover:text-red-300 text-xs flex items-center gap-1 ml-auto"
                          >
                            <Trash2 className="w-3.5 h-3.5" /> Drop
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 3: BENCHMARK LAB */}
      {activeTab === 'benchmark_lab' && (
        <div className="space-y-6">
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-blue-400" />
                  Comparative Algorithm Benchmark (Exact KNN vs HNSW vs IVFFlat)
                </h3>
                <p className="text-xs text-slate-400 mt-1">
                  Executes queries against Exact KNN, HNSW, and IVFFlat side-by-side, calculating Recall@K, latency distributions (p50, p95), and QPS throughput.
                </p>
              </div>

              <button
                onClick={handleRunBenchmark}
                disabled={benchmarking}
                className="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded-lg text-xs font-medium flex items-center gap-2 shadow-lg shadow-blue-500/20"
              >
                {benchmarking ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                {benchmarking ? 'Running Multi-Iteration Benchmark...' : 'Run Comparative Benchmark'}
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 pt-2">
              <div>
                <label className="text-xs text-slate-400 block mb-1">Repeated Iterations</label>
                <select
                  value={benchmarkRuns}
                  onChange={(e) => setBenchmarkRuns(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 text-xs rounded p-2 text-slate-200"
                >
                  <option value="3">3 Runs</option>
                  <option value="5">5 Runs (Default)</option>
                  <option value="10">10 Runs (Accurate)</option>
                </select>
              </div>

              <div>
                <label className="text-xs text-slate-400 block mb-1">HNSW ef_search Depth</label>
                <input
                  type="number"
                  min="10"
                  max="200"
                  value={efSearch}
                  onChange={(e) => setEfSearch(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 text-xs rounded p-2 text-slate-200"
                />
              </div>

              <div>
                <label className="text-xs text-slate-400 block mb-1">IVFFlat Probes</label>
                <input
                  type="number"
                  min="1"
                  max="20"
                  value={probes}
                  onChange={(e) => setProbes(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 text-xs rounded p-2 text-slate-200"
                />
              </div>

              <div className="flex flex-col justify-end">
                <button
                  onClick={handleGenerateSynthetic}
                  disabled={generatingSynthetic}
                  className="w-full px-3 py-2 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-slate-200 rounded text-xs border border-slate-700 flex items-center justify-center gap-1.5"
                >
                  <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                  {generatingSynthetic ? 'Generating Vectors...' : 'Generate 300 Synthetic Vectors'}
                </button>
              </div>
            </div>

            {syntheticStatus && (
              <div
                className={`p-3 rounded-lg text-xs flex items-center gap-2 ${
                  syntheticStatus.type === 'success'
                    ? 'bg-emerald-500/10 border border-emerald-500/20 text-emerald-300'
                    : 'bg-red-500/10 border border-red-500/20 text-red-300'
                }`}
              >
                <Info className="w-4 h-4 shrink-0" />
                {syntheticStatus.text}
              </div>
            )}
          </div>

          {/* Benchmark Results */}
          {benchmarkResponse && (
            <div className="space-y-4">
              {/* Latency Separation & Summary Header */}
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4 text-xs font-mono">
                <div className="flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-purple-400" />
                  <span className="text-slate-400 font-sans">Query Embedding Latency:</span>
                  <span className="text-purple-300 font-bold">
                    {benchmarkResponse.embedding_latency_ms != null
                      ? `${Number(benchmarkResponse.embedding_latency_ms).toFixed(2)} ms`
                      : 'N/A'}
                  </span>
                  <span className="text-[10px] text-slate-500 font-sans">(SentenceTransformers inference)</span>
                </div>
                <div className="flex items-center gap-2">
                  <Database className="w-4 h-4 text-blue-400" />
                  <span className="text-slate-400 font-sans">Corpus Evaluated:</span>
                  <span className="text-blue-300 font-bold">{benchmarkResponse.corpus_size} vectors</span>
                </div>
                <div className="flex items-center gap-2">
                  <Clock className="w-4 h-4 text-emerald-400" />
                  <span className="text-slate-400 font-sans">Benchmarking Mode:</span>
                  <span className="text-emerald-300 font-bold">Isolated Session Table ({benchmarkResponse.runs} runs)</span>
                </div>
              </div>

              {/* Warnings Banner */}
              {benchmarkResponse.warnings?.length > 0 && (
                <div className="bg-amber-500/10 border border-amber-500/20 text-amber-300 p-3.5 rounded-xl space-y-1">
                  <div className="flex items-center gap-2 font-medium text-xs text-amber-400">
                    <AlertTriangle className="w-4 h-4 shrink-0" /> Benchmark Dataset Notice:
                  </div>
                  {benchmarkResponse.warnings.map((w, idx) => (
                    <div key={idx} className="text-xs text-amber-200/90 pl-6">
                      {w}
                    </div>
                  ))}
                </div>
              )}

              {/* Side-by-Side Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                {/* 1. Exact KNN */}
                <div className="bg-slate-900 border border-blue-500/30 rounded-xl p-5 space-y-4">
                  <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                    <span className="text-xs font-bold text-blue-400 uppercase tracking-wide">
                      Exact KNN (Brute-Force)
                    </span>
                    <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-blue-500/20 text-blue-300 border border-blue-500/30">
                      Baseline
                    </span>
                  </div>

                  <div className="space-y-2.5 font-mono text-xs">
                    <div className="flex justify-between">
                      <span className="text-slate-400 font-sans">Recall@K:</span>
                      <span className="text-white font-bold">100.0% (1.0000)</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400 font-sans">DB Latency (p50):</span>
                      <span className="text-blue-300">{benchmarkResponse.algorithms.exact.latency_p50_ms} ms</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400 font-sans">DB Latency (p95):</span>
                      <span className="text-blue-300">{benchmarkResponse.algorithms.exact.latency_p95_ms} ms</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400 font-sans">DB Latency (Mean):</span>
                      <span className="text-blue-300">{benchmarkResponse.algorithms.exact.latency_mean_ms} ms</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400 font-sans">Throughput (QPS):</span>
                      <span className="text-emerald-400 font-bold">{benchmarkResponse.algorithms.exact.throughput_qps}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400 font-sans">Index Build Time:</span>
                      <span className="text-slate-400">
                        {benchmarkResponse.algorithms.exact.index_build_ms != null
                          ? `${benchmarkResponse.algorithms.exact.index_build_ms} ms`
                          : '0.00 ms (None)'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400 font-sans">Postgres Plan:</span>
                      <span className="text-slate-300">{benchmarkResponse.algorithms.exact.plan_node_type}</span>
                    </div>

                    {/* Plan Discrepancy Check */}
                    {!benchmarkResponse.algorithms.exact.plan_node_type?.toLowerCase().includes('seq scan') && (
                      <div className="p-2 rounded bg-amber-500/10 border border-amber-500/20 text-amber-400 text-[11px] flex items-center gap-1.5 font-sans">
                        <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                        <span>Plan Discrepancy: Expected Seq Scan</span>
                      </div>
                    )}

                    {benchmarkResponse.algorithms.exact.explain_plan_summary && (
                      <div className="pt-2 border-t border-slate-800">
                        <div className="text-[10px] text-slate-500 font-sans mb-1">Execution Plan:</div>
                        <div className="text-[10px] text-slate-300 font-mono bg-slate-950 p-2 rounded border border-slate-800/80 truncate" title={benchmarkResponse.algorithms.exact.explain_plan_summary}>
                          {benchmarkResponse.algorithms.exact.explain_plan_summary}
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* 2. HNSW */}
                <div className="bg-slate-900 border border-emerald-500/30 rounded-xl p-5 space-y-4">
                  <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                    <span className="text-xs font-bold text-emerald-400 uppercase tracking-wide">
                      HNSW (Proximity Graph)
                    </span>
                    <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                      ANN
                    </span>
                  </div>

                  <div className="space-y-2.5 font-mono text-xs">
                    <div className="flex justify-between">
                      <span className="text-slate-400 font-sans">Recall@K vs Exact:</span>
                      <span className="text-emerald-400 font-bold">
                        {(benchmarkResponse.algorithms.hnsw.recall_at_k * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400 font-sans">Jaccard Overlap:</span>
                      <span className="text-slate-200">
                        {benchmarkResponse.algorithms.hnsw.jaccard_overlap.toFixed(4)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400 font-sans">DB Latency (p50):</span>
                      <span className="text-emerald-300">{benchmarkResponse.algorithms.hnsw.latency_p50_ms} ms</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400 font-sans">DB Latency (p95):</span>
                      <span className="text-emerald-300">{benchmarkResponse.algorithms.hnsw.latency_p95_ms} ms</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400 font-sans">DB Latency (Mean):</span>
                      <span className="text-emerald-300">{benchmarkResponse.algorithms.hnsw.latency_mean_ms} ms</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400 font-sans">Throughput (QPS):</span>
                      <span className="text-emerald-400 font-bold">{benchmarkResponse.algorithms.hnsw.throughput_qps}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400 font-sans">Index Build Time:</span>
                      <span className="text-emerald-300">
                        {benchmarkResponse.algorithms.hnsw.index_build_ms != null
                          ? `${benchmarkResponse.algorithms.hnsw.index_build_ms} ms`
                          : 'N/A'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400 font-sans">Postgres Plan:</span>
                      <span className="text-slate-300">{benchmarkResponse.algorithms.hnsw.plan_node_type}</span>
                    </div>

                    {/* Plan Discrepancy Check */}
                    {!benchmarkResponse.algorithms.hnsw.plan_node_type?.toLowerCase().includes('hnsw') && (
                      <div className="p-2 rounded bg-amber-500/10 border border-amber-500/20 text-amber-400 text-[11px] flex items-center gap-1.5 font-sans">
                        <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                        <span>Plan Discrepancy: Expected HNSW index scan</span>
                      </div>
                    )}

                    {benchmarkResponse.algorithms.hnsw.explain_plan_summary && (
                      <div className="pt-2 border-t border-slate-800">
                        <div className="text-[10px] text-slate-500 font-sans mb-1">Execution Plan:</div>
                        <div className="text-[10px] text-slate-300 font-mono bg-slate-950 p-2 rounded border border-slate-800/80 truncate" title={benchmarkResponse.algorithms.hnsw.explain_plan_summary}>
                          {benchmarkResponse.algorithms.hnsw.explain_plan_summary}
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* 3. IVFFlat */}
                <div className="bg-slate-900 border border-amber-500/30 rounded-xl p-5 space-y-4">
                  <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                    <span className="text-xs font-bold text-amber-400 uppercase tracking-wide">
                      IVFFlat (Voronoi Cells)
                    </span>
                    <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-amber-500/20 text-amber-300 border border-amber-500/30">
                      ANN
                    </span>
                  </div>

                  <div className="space-y-2.5 font-mono text-xs">
                    <div className="flex justify-between">
                      <span className="text-slate-400 font-sans">Recall@K vs Exact:</span>
                      <span className="text-amber-400 font-bold">
                        {(benchmarkResponse.algorithms.ivfflat.recall_at_k * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400 font-sans">Jaccard Overlap:</span>
                      <span className="text-slate-200">
                        {benchmarkResponse.algorithms.ivfflat.jaccard_overlap.toFixed(4)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400 font-sans">DB Latency (p50):</span>
                      <span className="text-amber-300">{benchmarkResponse.algorithms.ivfflat.latency_p50_ms} ms</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400 font-sans">DB Latency (p95):</span>
                      <span className="text-amber-300">{benchmarkResponse.algorithms.ivfflat.latency_p95_ms} ms</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400 font-sans">DB Latency (Mean):</span>
                      <span className="text-amber-300">{benchmarkResponse.algorithms.ivfflat.latency_mean_ms} ms</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400 font-sans">Throughput (QPS):</span>
                      <span className="text-amber-400 font-bold">{benchmarkResponse.algorithms.ivfflat.throughput_qps}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400 font-sans">Index Build Time:</span>
                      <span className="text-amber-300">
                        {benchmarkResponse.algorithms.ivfflat.index_build_ms != null
                          ? `${benchmarkResponse.algorithms.ivfflat.index_build_ms} ms`
                          : 'N/A'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400 font-sans">Postgres Plan:</span>
                      <span className="text-slate-300">{benchmarkResponse.algorithms.ivfflat.plan_node_type}</span>
                    </div>

                    {/* Plan Discrepancy Check */}
                    {!benchmarkResponse.algorithms.ivfflat.plan_node_type?.toLowerCase().includes('ivfflat') && (
                      <div className="p-2 rounded bg-amber-500/10 border border-amber-500/20 text-amber-400 text-[11px] flex items-center gap-1.5 font-sans">
                        <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                        <span>Plan Discrepancy: Expected IVFFlat index scan</span>
                      </div>
                    )}

                    {benchmarkResponse.algorithms.ivfflat.explain_plan_summary && (
                      <div className="pt-2 border-t border-slate-800">
                        <div className="text-[10px] text-slate-500 font-sans mb-1">Execution Plan:</div>
                        <div className="text-[10px] text-slate-300 font-mono bg-slate-950 p-2 rounded border border-slate-800/80 truncate" title={benchmarkResponse.algorithms.ivfflat.explain_plan_summary}>
                          {benchmarkResponse.algorithms.ivfflat.explain_plan_summary}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 4: LEARNING CENTER */}
      {activeTab === 'learning_center' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Module 1: Exact KNN */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
              <h3 className="text-sm font-bold text-blue-400 flex items-center gap-2">
                <Search className="w-4 h-4" /> 1. Exact KNN (Brute-Force Search)
              </h3>
              <p className="text-xs text-slate-300 leading-relaxed">
                Exact K-Nearest Neighbors evaluates distances between the query vector and every stored embedding in the dataset (time complexity <code className="text-blue-300">O(N &bull; d)</code>).
              </p>
              <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 text-[11px] font-mono text-slate-300">
                SELECT * FROM chunk_embeddings<br />
                ORDER BY vector &lt;=&gt; :query_vector<br />
                LIMIT 5;
              </div>
              <ul className="text-xs text-slate-400 space-y-1 list-disc pl-4">
                <li><strong>Recall:</strong> 100% exact mathematical ground truth.</li>
                <li><strong>Tradeoff:</strong> Linear scaling latency. Impractical when N &gt; 100,000 vectors.</li>
              </ul>
            </div>

            {/* Module 2: HNSW Graphs */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
              <h3 className="text-sm font-bold text-emerald-400 flex items-center gap-2">
                <Network className="w-4 h-4" /> 2. HNSW (Hierarchical Navigable Small World)
              </h3>
              <p className="text-xs text-slate-300 leading-relaxed">
                HNSW builds a multi-layer graph reminiscent of a probabilistic skip-list. High layers provide coarse, long-range routing edges, while lower layers refine local neighborhoods (time complexity <code className="text-emerald-300">O(log N)</code>).
              </p>
              <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 text-[11px] font-mono text-slate-300">
                CREATE INDEX idx_hnsw ON chunk_embeddings<br />
                USING hnsw (vector vector_cosine_ops)<br />
                WITH (m = 16, ef_construction = 64);
              </div>
              <ul className="text-xs text-slate-400 space-y-1 list-disc pl-4">
                <li><strong>m:</strong> Number of bi-directional connection links per node (4-64).</li>
                <li><strong>ef_construction:</strong> Candidate queue size during graph building (higher = better graph quality).</li>
                <li><strong>ef_search:</strong> Query-time search depth. Higher values approach 99%+ recall at slight latency increase.</li>
              </ul>
            </div>

            {/* Module 3: IVFFlat Inverted Lists */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
              <h3 className="text-sm font-bold text-amber-400 flex items-center gap-2">
                <Database className="w-4 h-4" /> 3. IVFFlat (Inverted File Flat)
              </h3>
              <p className="text-xs text-slate-300 leading-relaxed">
                Partitions the high-dimensional vector space into Voronoi cells using k-means clustering. At query time, only the closest <code className="text-amber-300">probes</code> cluster lists are scanned.
              </p>
              <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 text-[11px] font-mono text-slate-300">
                SET ivfflat.probes = 5;<br />
                SELECT * FROM chunk_embeddings<br />
                ORDER BY vector &lt;=&gt; :query_vector<br />
                LIMIT 5;
              </div>
              <ul className="text-xs text-slate-400 space-y-1 list-disc pl-4">
                <li><strong>Memory footprint:</strong> Very compact compared to HNSW graph pointers.</li>
                <li><strong>Prerequisite:</strong> Requires substantial training data (N &ge; 1,000 vectors) to form representative centroids.</li>
              </ul>
            </div>

            {/* Module 4: Distance Metrics & Normalization */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
              <h3 className="text-sm font-bold text-purple-400 flex items-center gap-2">
                <Zap className="w-4 h-4" /> 4. Distance Metrics &amp; Unit Normalization
              </h3>
              <p className="text-xs text-slate-300 leading-relaxed">
                When vectors are L2-normalized (||u|| = ||v|| = 1.0), Cosine Distance, Euclidean L2 distance, and Inner Product are mathematically directly related:
              </p>
              <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 text-[11px] font-mono text-purple-300 space-y-1">
                <div>&bull; Cosine Distance: d_cos = 1 - (u &bull; v)</div>
                <div>&bull; L2 Distance: d_l2 = sqrt(2 &bull; (1 - u &bull; v)) = sqrt(2 &bull; d_cos)</div>
                <div>&bull; Inner Product: pgvector &lt;#&gt; returns -(u &bull; v)</div>
              </div>
              <p className="text-xs text-slate-400">
                On normalized embeddings, all three operators produce identical top-k rankings!
              </p>
            </div>
          </div>

          {/* Module 5: Interactive Reciprocal Rank Fusion (RRF) Simulator */}
          <div className="bg-slate-900 border border-indigo-500/30 rounded-xl p-5 space-y-4">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <h3 className="text-sm font-bold text-indigo-400 flex items-center gap-2">
                <GitMerge className="w-4 h-4" /> 5. Interactive Reciprocal Rank Fusion (RRF) Simulator
              </h3>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-800">
                Formula: RRF(d) = Σ 1 / (k + r_c(d))
              </span>
            </div>

            <p className="text-xs text-slate-300 leading-relaxed">
              Reciprocal Rank Fusion merges ranked candidate lists without needing score normalization. Unlike linear combination which suffers from disparate score distributions (e.g. cosine in [-1, 1] vs BM25 in [0, ∞)), RRF relies strictly on ordinal rank positions.
            </p>

            {/* Rank Constant K Slider */}
            <div className="p-3 bg-slate-950/80 rounded-lg border border-slate-800 space-y-1">
              <div className="flex justify-between items-center text-xs">
                <span className="font-medium text-slate-300">Rank Constant (k)</span>
                <span className="font-mono text-indigo-400 font-bold">{simK}</span>
              </div>
              <input
                type="range"
                min="1"
                max="100"
                value={simK}
                onChange={(e) => setSimK(parseInt(e.target.value))}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
              />
              <div className="flex justify-between text-[10px] text-slate-500">
                <span>k=1 (Extreme rank bias)</span>
                <span>k=60 (Standard Information Retrieval default)</span>
                <span>k=100 (Smoothed rank penalties)</span>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Document A Simulator */}
              {(() => {
                const scoreA = (1 / (simK + simRankDense)) + (1 / (simK + simRankLexical));
                const scoreB = (1 / (simK + simRankDenseB)) + (1 / (simK + simRankLexicalB));
                const isWinner = scoreA >= scoreB;
                return (
                  <div className={`p-4 rounded-xl border space-y-3 ${
                    isWinner
                      ? 'bg-emerald-950/20 border-emerald-500/40'
                      : 'bg-slate-950/60 border-slate-800'
                  }`}>
                    <div className="flex justify-between items-center">
                      <span className="text-xs font-bold text-slate-200">Candidate Document A</span>
                      {isWinner && (
                        <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 text-[10px] font-mono border border-emerald-500/30">
                          Winner (Rank #1)
                        </span>
                      )}
                    </div>

                    <div className="space-y-2 text-xs">
                      <div>
                        <div className="flex justify-between text-[11px] mb-1">
                          <span className="text-blue-400">Dense Rank (r1):</span>
                          <span className="font-mono font-bold text-blue-300">#{simRankDense}</span>
                        </div>
                        <input
                          type="range"
                          min="1"
                          max="50"
                          value={simRankDense}
                          onChange={(e) => setSimRankDense(parseInt(e.target.value))}
                          className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-blue-500"
                        />
                      </div>

                      <div>
                        <div className="flex justify-between text-[11px] mb-1">
                          <span className="text-purple-400">Lexical GIN Rank (r2):</span>
                          <span className="font-mono font-bold text-purple-300">#{simRankLexical}</span>
                        </div>
                        <input
                          type="range"
                          min="1"
                          max="50"
                          value={simRankLexical}
                          onChange={(e) => setSimRankLexical(parseInt(e.target.value))}
                          className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-purple-500"
                        />
                      </div>
                    </div>

                    <div className="pt-2 border-t border-slate-800 font-mono text-[11px] flex justify-between">
                      <span className="text-slate-400">RRF Score:</span>
                      <span className="text-emerald-400 font-bold">{scoreA.toFixed(5)}</span>
                    </div>
                  </div>
                );
              })()}

              {/* Document B Simulator */}
              {(() => {
                const scoreA = (1 / (simK + simRankDense)) + (1 / (simK + simRankLexical));
                const scoreB = (1 / (simK + simRankDenseB)) + (1 / (simK + simRankLexicalB));
                const isWinner = scoreB > scoreA;
                return (
                  <div className={`p-4 rounded-xl border space-y-3 ${
                    isWinner
                      ? 'bg-emerald-950/20 border-emerald-500/40'
                      : 'bg-slate-950/60 border-slate-800'
                  }`}>
                    <div className="flex justify-between items-center">
                      <span className="text-xs font-bold text-slate-200">Candidate Document B</span>
                      {isWinner && (
                        <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 text-[10px] font-mono border border-emerald-500/30">
                          Winner (Rank #1)
                        </span>
                      )}
                    </div>

                    <div className="space-y-2 text-xs">
                      <div>
                        <div className="flex justify-between text-[11px] mb-1">
                          <span className="text-blue-400">Dense Rank (r1):</span>
                          <span className="font-mono font-bold text-blue-300">#{simRankDenseB}</span>
                        </div>
                        <input
                          type="range"
                          min="1"
                          max="50"
                          value={simRankDenseB}
                          onChange={(e) => setSimRankDenseB(parseInt(e.target.value))}
                          className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-blue-500"
                        />
                      </div>

                      <div>
                        <div className="flex justify-between text-[11px] mb-1">
                          <span className="text-purple-400">Lexical GIN Rank (r2):</span>
                          <span className="font-mono font-bold text-purple-300">#{simRankLexicalB}</span>
                        </div>
                        <input
                          type="range"
                          min="1"
                          max="50"
                          value={simRankLexicalB}
                          onChange={(e) => setSimRankLexicalB(parseInt(e.target.value))}
                          className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-purple-500"
                        />
                      </div>
                    </div>

                    <div className="pt-2 border-t border-slate-800 font-mono text-[11px] flex justify-between">
                      <span className="text-slate-400">RRF Score:</span>
                      <span className="text-emerald-400 font-bold">{scoreB.toFixed(5)}</span>
                    </div>
                  </div>
                );
              })()}
            </div>

            <div className="p-3 bg-indigo-950/30 border border-indigo-900/40 rounded-lg text-xs text-indigo-200/90 leading-relaxed">
              <strong>Key Finding:</strong> Notice how Document B ranking #8 in Dense and #9 in Lexical can defeat Document A that ranks #1 in Dense but only #40 in Lexical. Dual-channel consensus reliably bubbles the most robust relevant chunks to the top!
            </div>
          </div>
        </div>
      )}

      {/* TAB 5: EXPERIMENT HISTORY */}
      {activeTab === 'experiment_history' && (
        <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
          <div className="px-5 py-4 border-b border-slate-800 flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-slate-200">Historical Retrieval Experiments</h3>
              <p className="text-xs text-slate-400">
                Persisted search configurations, latencies, and candidate chunk rankings.
              </p>
            </div>
            <button
              onClick={loadExperiments}
              disabled={loadingExperiments}
              className="text-xs px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 flex items-center gap-1.5"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loadingExperiments ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </div>

          {experiments.length === 0 ? (
            <div className="p-8 text-center text-slate-500 text-xs">
              No retrieval experiments saved yet. Perform a search and click "Save Run" to record an experiment.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-950 text-slate-400 uppercase font-mono text-[10px]">
                  <tr>
                    <th className="px-4 py-3">Experiment Name</th>
                    <th className="px-4 py-3">Query Text</th>
                    <th className="px-4 py-3">Model</th>
                    <th className="px-4 py-3">Algorithm</th>
                    <th className="px-4 py-3">Top-K</th>
                    <th className="px-4 py-3">Latency</th>
                    <th className="px-4 py-3">Plan Node</th>
                    <th className="px-4 py-3 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800">
                  {experiments.map((exp) => (
                    <tr key={exp.id} className="hover:bg-slate-800/40 font-mono">
                      <td className="px-4 py-3 font-semibold text-slate-200 font-sans">{exp.name}</td>
                      <td className="px-4 py-3 text-slate-300 max-w-xs truncate font-sans">{exp.query_text}</td>
                      <td className="px-4 py-3 text-blue-400 text-[11px]">{exp.model_id.split('/').pop()}</td>
                      <td className="px-4 py-3 text-slate-300 uppercase text-[10px]">
                        {exp.metrics?.search_algorithm || 'exact'}
                      </td>
                      <td className="px-4 py-3 text-slate-300">{exp.top_k}</td>
                      <td className="px-4 py-3 text-emerald-400">
                        {exp.metrics?.total_latency_ms || exp.metrics?.query_latency_ms || '-'}ms
                      </td>
                      <td className="px-4 py-3 text-slate-400 text-[11px]">
                        {exp.metrics?.plan_node_type || 'Seq Scan'}
                      </td>
                      <td className="px-4 py-3 text-right font-sans">
                        <button
                          onClick={() => handleDeleteExperiment(exp.id)}
                          className="text-red-400 hover:text-red-300 text-xs flex items-center gap-1 ml-auto"
                        >
                          <Trash2 className="w-3.5 h-3.5" /> Delete
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default RetrievalLab;
