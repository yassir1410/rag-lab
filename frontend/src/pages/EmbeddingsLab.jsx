import React, { useState, useEffect } from 'react';
import {
  Network,
  Cpu,
  Layers,
  Sparkles,
  BarChart3,
  Search,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  Copy,
  Check,
  ChevronRight,
  BookOpen,
  ArrowRightLeft,
  Gauge,
  Sliders,
  Database,
  ExternalLink,
  Flame,
  Trash2,
} from 'lucide-react';
import api from '../services/api';

export function EmbeddingsLab() {
  // Navigation tabs
  const [activeTab, setActiveTab] = useState('overview'); // overview | tokenizer | vector | similarity | projection | benchmark | learning
  const [learningMode, setLearningMode] = useState(true);

  // Global state
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('sentence-transformers/all-MiniLM-L6-v2');
  const [documents, setDocuments] = useState([]);
  const [selectedDocId, setSelectedDocId] = useState('');
  const [chunkExperiments, setChunkExperiments] = useState([]);
  const [selectedChunkExpId, setSelectedChunkExpId] = useState('');

  // Status & error states
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [copiedIndex, setCopiedIndex] = useState(null);

  // Tab 2: Tokenizer state
  const [tokenText, setTokenText] = useState(
    "L'architecture de microservices avec FastAPI et PostgreSQL améliore l'isolation des données."
  );
  const [tokenResult, setTokenResult] = useState(null);
  const [tokenCompareResults, setTokenCompareResults] = useState(null);
  const [isTokenizing, setIsTokenizing] = useState(false);

  // Tab 3: Vector Explorer state
  const [vectorInputText, setVectorInputText] = useState(
    'Dense vector embeddings map semantic concepts into geometric space.'
  );
  const [vectorResult, setVectorResult] = useState(null);
  const [isGeneratingVector, setIsGeneratingVector] = useState(false);

  // Tab 4: Similarity Lab state
  const [simTextA, setSimTextA] = useState(
    'FastAPI is a modern, high-performance web framework for building APIs.'
  );
  const [simTextB, setSimTextB] = useState(
    'A fast Python web framework designed for creating asynchronous REST APIs is FastAPI.'
  );
  const [simNormalize, setSimNormalize] = useState(true);
  const [simResult, setSimResult] = useState(null);
  const [isCalculatingSim, setIsCalculatingSim] = useState(false);

  // Similarity Matrix state
  const [matrixTexts, setMatrixTexts] = useState([
    'FastAPI builds fast asynchronous APIs.',
    'Python web frameworks enable rapid REST service development.',
    'PostgreSQL with pgvector stores high-dimensional embeddings.',
    'Baking chocolate cake requires flour, sugar and high heat.',
  ]);
  const [matrixMetric, setMatrixMetric] = useState('cosine');
  const [matrixResult, setMatrixResult] = useState(null);
  const [isCalculatingMatrix, setIsCalculatingMatrix] = useState(false);

  // Tab 5: Projection Lab state
  const [projAlgo, setProjAlgo] = useState('pca');
  const [projPerplexity, setProjPerplexity] = useState(15);
  const [projNeighbors, setProjNeighbors] = useState(5);
  const [projectionData, setProjectionData] = useState(null);
  const [isProjecting, setIsProjecting] = useState(false);
  const [hoveredPoint, setHoveredPoint] = useState(null);

  // Tab 6: Benchmark Lab state
  const [benchmarkResult, setBenchmarkResult] = useState(null);
  const [isBenchmarking, setIsBenchmarking] = useState(false);

  // Tab 7: Saved Experiments
  const [savedExperiments, setSavedExperiments] = useState([]);

  // Load initial data
  useEffect(() => {
    loadModels();
    loadDocuments();
    loadSavedExperiments();
  }, []);

  // When document changes, load its chunking experiments
  useEffect(() => {
    if (selectedDocId) {
      loadChunkExperiments(selectedDocId);
    }
  }, [selectedDocId]);

  const loadModels = async () => {
    try {
      const data = await api.getEmbeddingModels();
      setModels(data);
      if (data.length > 0 && !selectedModel) {
        setSelectedModel(data[0].model_id);
      }
    } catch (err) {
      console.error('Failed to load models:', err);
    }
  };

  const loadDocuments = async () => {
    try {
      const res = await api.getDocuments();
      const docs = res?.items || (Array.isArray(res) ? res : []);
      setDocuments(docs);
      if (docs.length > 0 && !selectedDocId) {
        setSelectedDocId(docs[0].id);
      }
    } catch (err) {
      console.error('Failed to load documents:', err);
    }
  };

  const loadChunkExperiments = async (docId) => {
    if (!docId) {
      setChunkExperiments([]);
      setSelectedChunkExpId('');
      return;
    }
    try {
      const res = await api.getChunkingExperiments(docId);
      const exps = res?.items || (Array.isArray(res) ? res : []);
      setChunkExperiments(exps);
      if (exps.length > 0) {
        setSelectedChunkExpId(exps[0].id);
      } else {
        setSelectedChunkExpId('');
      }
    } catch (err) {
      console.error('Failed to load chunk experiments:', err);
      setChunkExperiments([]);
      setSelectedChunkExpId('');
    }
  };

  const loadSavedExperiments = async () => {
    try {
      const res = await api.getEmbeddingExperiments();
      setSavedExperiments(res.items || []);
    } catch (err) {
      console.error('Failed to load embedding experiments:', err);
    }
  };

  // Actions
  const handleTokenize = async () => {
    if (!tokenText.trim()) return;
    setIsTokenizing(true);
    setError(null);
    try {
      const res = await api.tokenizeText(tokenText, selectedModel);
      setTokenResult(res);
      setTokenCompareResults(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsTokenizing(false);
    }
  };

  const handleCompareTokenizers = async () => {
    if (!tokenText.trim()) return;
    setIsTokenizing(true);
    setError(null);
    try {
      const modelIds = models.map((m) => m.model_id);
      const res = await api.compareTokenizers(tokenText, modelIds);
      setTokenCompareResults(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsTokenizing(false);
    }
  };

  const handleGenerateVector = async () => {
    if (!vectorInputText.trim()) return;
    setIsGeneratingVector(true);
    setError(null);
    try {
      const res = await api.generateEmbeddings({
        texts: [vectorInputText],
        model_id: selectedModel,
        normalize: true,
      });
      setVectorResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsGeneratingVector(false);
    }
  };

  const handleCalculateSimilarity = async () => {
    if (!simTextA.trim() || !simTextB.trim()) return;
    setIsCalculatingSim(true);
    setError(null);
    try {
      const res = await api.calculateSimilarity(
        simTextA,
        simTextB,
        selectedModel,
        simNormalize
      );
      setSimResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsCalculatingSim(false);
    }
  };

  const handleCalculateMatrix = async () => {
    const validTexts = matrixTexts.filter((t) => t.trim().length > 0);
    if (validTexts.length < 2) return;
    setIsCalculatingMatrix(true);
    setError(null);
    try {
      const res = await api.calculateSimilarityMatrix(
        validTexts,
        selectedModel,
        matrixMetric,
        simNormalize
      );
      setMatrixResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsCalculatingMatrix(false);
    }
  };

  const handleRunProjection = async () => {
    setIsProjecting(true);
    setError(null);
    try {
      const payload = {
        model_id: selectedModel,
        algorithm: projAlgo,
        n_components: 2,
        perplexity: projPerplexity,
        n_neighbors: projNeighbors,
      };

      if (selectedChunkExpId) {
        payload.chunking_experiment_id = selectedChunkExpId;
      } else {
        payload.texts = matrixTexts.filter((t) => t.trim().length > 0);
      }

      const res = await api.projectEmbeddings(payload);
      setProjectionData(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsProjecting(false);
    }
  };

  const handleRunBenchmark = async () => {
    setIsBenchmarking(true);
    setError(null);
    try {
      const res = await api.runEmbeddingBenchmark(selectedModel);
      setBenchmarkResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsBenchmarking(false);
    }
  };

  const handleSaveExperiment = async () => {
    if (!selectedChunkExpId) {
      setError('Please select a Chunking Experiment to embed and persist.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      await api.createEmbeddingExperiment({
        chunking_experiment_id: selectedChunkExpId,
        model_id: selectedModel,
        normalize: true,
        batch_size: 32,
      });
      await loadSavedExperiments();
      alert('Embedding experiment saved successfully to PostgreSQL!');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteExperiment = async (expId) => {
    if (!confirm('Are you sure you want to delete this embedding experiment?')) return;
    try {
      await api.deleteEmbeddingExperiment(expId);
      await loadSavedExperiments();
    } catch (err) {
      setError(err.message);
    }
  };

  const copyToClipboard = (text, idx) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(idx);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  const currentModelSpec = models.find((m) => m.model_id === selectedModel);

  return (
    <div className="flex-1 flex flex-col min-w-0 overflow-y-auto bg-slate-900 text-slate-100 p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 text-[11px] font-bold tracking-wider uppercase rounded bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
              Phase 04
            </span>
            <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
              <Network className="w-6 h-6 text-indigo-400" />
              Embeddings Engineering Lab
            </h1>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Explore dense vector geometry, tokenization boundaries, similarity mathematics, and dimensionality reductions.
          </p>
        </div>

        {/* Global Controls */}
        <div className="flex items-center gap-3">
          {/* Mode Toggle */}
          <button
            onClick={() => setLearningMode(!learningMode)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium border flex items-center gap-1.5 transition-colors ${
              learningMode
                ? 'bg-indigo-600/20 text-indigo-300 border-indigo-500/40'
                : 'bg-slate-800 text-slate-400 border-slate-700'
            }`}
          >
            <BookOpen className="w-3.5 h-3.5" />
            {learningMode ? 'Learning Mode: ON' : 'Expert Mode'}
          </button>

          {/* Model Selector */}
          <div className="flex items-center gap-2 bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5">
            <Cpu className="w-3.5 h-3.5 text-slate-400" />
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="bg-transparent text-xs text-slate-200 outline-none cursor-pointer max-w-[220px]"
            >
              {models.map((m) => (
                <option key={m.model_id} value={m.model_id} className="bg-slate-900 text-slate-200">
                  {m.display_name} ({m.dimension}d)
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Global Error Banner */}
      {error && (
        <div className="p-3 bg-red-950/50 border border-red-500/40 rounded-lg flex items-center justify-between text-xs text-red-300">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
            <span>{error}</span>
          </div>
          <button onClick={() => setError(null)} className="text-red-400 hover:text-white font-bold">
            &times;
          </button>
        </div>
      )}

      {/* Navigation Tabs */}
      <div className="flex items-center gap-1 border-b border-slate-800 overflow-x-auto pb-1 text-xs font-medium">
        {[
          { id: 'overview', label: 'Overview', icon: LayoutDashboardIcon },
          { id: 'tokenizer', label: 'Tokenizer Lab', icon: Gauge },
          { id: 'vector', label: 'Vector Explorer', icon: Layers },
          { id: 'similarity', label: 'Similarity Lab', icon: ArrowRightLeft },
          { id: 'projection', label: 'Projection Lab (PCA/t-SNE/UMAP)', icon: Network },
          { id: 'benchmark', label: 'Benchmark Lab', icon: Flame },
          { id: 'learning', label: 'Learning Center', icon: BookOpen },
        ].map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-t-lg transition-colors border-b-2 whitespace-nowrap ${
                isActive
                  ? 'border-indigo-500 text-indigo-300 bg-indigo-500/10 font-semibold'
                  : 'border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* TAB 1: OVERVIEW */}
      {activeTab === 'overview' && (
        <div className="space-y-6">
          {learningMode && (
            <div className="p-4 bg-indigo-950/30 border border-indigo-500/30 rounded-xl space-y-2">
              <h3 className="text-sm font-semibold text-indigo-300 flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-indigo-400" />
                How Dense Vector Embeddings Work
              </h3>
              <p className="text-xs text-slate-300 leading-relaxed">
                Dense embeddings map raw unstructured text into continuous real-valued vector spaces where geometric proximity reflects conceptual similarity. Unlike sparse lexical representations (TF-IDF, BM25) where orthogonal keywords cannot match synonyms, dense transformers represent latent semantic features across 384+ hidden dimensions.
              </p>
            </div>
          )}

          {/* Model Registry Cards */}
          <div>
            <h2 className="text-sm font-semibold text-slate-200 mb-3 flex items-center gap-2">
              <Cpu className="w-4 h-4 text-indigo-400" />
              Registered Embedding Models (Local CPU Inference)
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {models.map((m) => {
                const isSelected = selectedModel === m.model_id;
                return (
                  <div
                    key={m.model_id}
                    onClick={() => setSelectedModel(m.model_id)}
                    className={`p-4 rounded-xl border cursor-pointer transition-all ${
                      isSelected
                        ? 'bg-indigo-600/15 border-indigo-500 shadow-lg shadow-indigo-500/10'
                        : 'bg-slate-950/60 border-slate-800 hover:border-slate-700'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                        {m.provider}
                      </span>
                      <span className="text-xs font-semibold text-indigo-400">{m.dimension}d</span>
                    </div>
                    <h3 className="font-semibold text-sm text-slate-100 mb-1">{m.display_name}</h3>
                    <p className="text-[11px] text-slate-400 line-clamp-2 mb-3">{m.description}</p>
                    <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-[10px] text-slate-500 font-mono">
                      <span>Max: {m.max_seq_length} toks</span>
                      <span>~{m.approx_size_mb} MB</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Transformation Pipeline Diagram */}
          <div className="p-5 bg-slate-950 border border-slate-800 rounded-xl space-y-3">
            <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
              Dense Vector Generation Pipeline
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-5 gap-3 text-center text-xs">
              <div className="p-3 bg-slate-900 border border-slate-800 rounded-lg">
                <span className="text-indigo-400 font-bold block mb-1">1. Raw Text</span>
                <span className="text-[11px] text-slate-400">Arbitrary string or chunk</span>
              </div>
              <div className="p-3 bg-slate-900 border border-slate-800 rounded-lg">
                <span className="text-indigo-400 font-bold block mb-1">2. Tokenizer</span>
                <span className="text-[11px] text-slate-400">Subword token IDs + CLS/SEP</span>
              </div>
              <div className="p-3 bg-slate-900 border border-slate-800 rounded-lg">
                <span className="text-indigo-400 font-bold block mb-1">3. Transformer</span>
                <span className="text-[11px] text-slate-400">Multi-head self-attention</span>
              </div>
              <div className="p-3 bg-slate-900 border border-slate-800 rounded-lg">
                <span className="text-indigo-400 font-bold block mb-1">4. Mean Pooling</span>
                <span className="text-[11px] text-slate-400">Aggregate token hidden states</span>
              </div>
              <div className="p-3 bg-slate-900 border border-slate-800 rounded-lg">
                <span className="text-indigo-400 font-bold block mb-1">5. L2 Norm</span>
                <span className="text-[11px] text-slate-400">Unit sphere projection (||v||=1)</span>
              </div>
            </div>
          </div>

          {/* Persisted Embedding Experiments */}
          <div className="p-5 bg-slate-950 border border-slate-800 rounded-xl space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
                <Database className="w-3.5 h-3.5 text-indigo-400" />
                Persisted Embedding Experiments (PostgreSQL + pgvector)
              </h3>
              <span className="text-xs text-slate-400">
                {savedExperiments.length} experiment{savedExperiments.length === 1 ? '' : 's'} stored
              </span>
            </div>

            {savedExperiments.length === 0 ? (
              <div className="p-6 border border-dashed border-slate-800 rounded-lg text-center space-y-1">
                <p className="text-xs text-slate-400">No embedding experiments saved to the database yet.</p>
                <p className="text-[11px] text-slate-500">
                  Select a document and chunking experiment in Projection Lab or call POST /api/v1/embeddings/experiments to encode and persist vectors.
                </p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-xs text-left">
                  <thead className="bg-slate-900/80 text-slate-400 uppercase font-mono text-[10px] border-b border-slate-800">
                    <tr>
                      <th className="py-2.5 px-3">Experiment Name</th>
                      <th className="py-2.5 px-3">Model</th>
                      <th className="py-2.5 px-3 text-center">Dimension</th>
                      <th className="py-2.5 px-3 text-center">Chunks Embedded</th>
                      <th className="py-2.5 px-3 text-center">Total Latency</th>
                      <th className="py-2.5 px-3 text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono">
                    {savedExperiments.map((exp) => (
                      <tr key={exp.id} className="hover:bg-slate-900/40 transition-colors">
                        <td className="py-2.5 px-3 font-sans font-medium text-slate-200">
                          {exp.name}
                        </td>
                        <td className="py-2.5 px-3 text-slate-300">
                          {exp.model_id.split('/').pop()}
                        </td>
                        <td className="py-2.5 px-3 text-center text-indigo-400 font-bold">
                          {exp.vector_dimension}d
                        </td>
                        <td className="py-2.5 px-3 text-center text-slate-300">
                          {exp.embeddings_count || exp.metrics?.total_items || 0}
                        </td>
                        <td className="py-2.5 px-3 text-center text-emerald-400">
                          {exp.metrics?.latency_ms ? `${exp.metrics.latency_ms.toFixed(1)} ms` : '—'}
                        </td>
                        <td className="py-2.5 px-3 text-right">
                          <button
                            onClick={() => handleDeleteExperiment(exp.id)}
                            className="p-1 text-slate-500 hover:text-red-400 rounded transition-colors"
                            title="Delete experiment"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
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

      {/* TAB 2: TOKENIZER LAB */}
      {activeTab === 'tokenizer' && (
        <div className="space-y-6">
          {learningMode && (
            <div className="p-4 bg-indigo-950/30 border border-indigo-500/30 rounded-xl space-y-1">
              <h3 className="text-xs font-semibold text-indigo-300">
                Tokens vs. Characters vs. Words
              </h3>
              <p className="text-xs text-slate-300">
                Transformers do not process letters or dictionary words. They process subword units (WordPiece, BPE) mapping text into finite vocabulary indexes. Punctuation, capitalization, and language alter subword splits.
              </p>
            </div>
          )}

          {/* Text Input */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-xs font-medium text-slate-300">Input Text for Inspection</label>
              <div className="flex items-center gap-2">
                <button
                  onClick={() =>
                    setTokenText(
                      "L'intelligence artificielle générative et les représentations vectorielles denses transforment l'architecture RAG."
                    )
                  }
                  className="text-[11px] text-indigo-400 hover:text-indigo-300 underline"
                >
                  Load French Sample
                </button>
                <span className="text-slate-600">&bull;</span>
                <button
                  onClick={() =>
                    setTokenText(
                      "High-dimensional vector indexing using HNSW and product quantization achieves sub-millisecond retrieval latency on million-scale datasets."
                    )
                  }
                  className="text-[11px] text-indigo-400 hover:text-indigo-300 underline"
                >
                  Load Technical Sample
                </button>
              </div>
            </div>
            <textarea
              rows={4}
              value={tokenText}
              onChange={(e) => setTokenText(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs text-slate-100 focus:border-indigo-500 outline-none"
              placeholder="Type or paste text here to inspect its tokenization..."
            />
            <div className="flex items-center gap-3">
              <button
                onClick={handleTokenize}
                disabled={isTokenizing || !tokenText.trim()}
                className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-medium transition-colors flex items-center gap-2 disabled:opacity-50"
              >
                {isTokenizing ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Gauge className="w-3.5 h-3.5" />}
                Tokenize with {currentModelSpec?.display_name || selectedModel}
              </button>
              <button
                onClick={handleCompareTokenizers}
                disabled={isTokenizing || !tokenText.trim()}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg text-xs font-medium transition-colors flex items-center gap-2 disabled:opacity-50"
              >
                <ArrowRightLeft className="w-3.5 h-3.5" />
                Compare Across All Models
              </button>
            </div>
          </div>

          {/* Single Model Token Result */}
          {tokenResult && !tokenCompareResults && (
            <div className="space-y-4">
              {/* Stats Grid */}
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg text-center">
                  <span className="text-[10px] text-slate-500 uppercase block">Characters</span>
                  <span className="text-lg font-bold text-slate-100">{tokenResult.char_count}</span>
                </div>
                <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg text-center">
                  <span className="text-[10px] text-slate-500 uppercase block">Words</span>
                  <span className="text-lg font-bold text-slate-100">{tokenResult.word_count}</span>
                </div>
                <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg text-center">
                  <span className="text-[10px] text-slate-500 uppercase block">Tokens</span>
                  <span className="text-lg font-bold text-indigo-400">{tokenResult.token_count}</span>
                </div>
                <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg text-center">
                  <span className="text-[10px] text-slate-500 uppercase block">Chars / Token</span>
                  <span className="text-lg font-bold text-slate-200">{tokenResult.ratio_char_to_token}</span>
                </div>
                <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg text-center">
                  <span className="text-[10px] text-slate-500 uppercase block">Model Limit</span>
                  <span className="text-lg font-bold text-slate-200">{tokenResult.max_seq_length} toks</span>
                </div>
              </div>

              {/* Truncation Warning */}
              {tokenResult.is_truncated && (
                <div className="p-3 bg-amber-950/50 border border-amber-500/40 rounded-lg flex items-center gap-2 text-xs text-amber-300">
                  <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
                  <span>
                    Warning: Input exceeds maximum context limit! {tokenResult.truncated_tokens_count} tokens will be truncated by this model.
                  </span>
                </div>
              )}

              {/* Tokens Chip Visualization */}
              <div className="p-4 bg-slate-950 border border-slate-800 rounded-xl space-y-3">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-semibold text-slate-300">Token Chips & Token IDs</span>
                  <span>{tokenResult.tokens.length} tokens displayed</span>
                </div>
                <div className="flex flex-wrap gap-1.5 max-h-72 overflow-y-auto p-2 bg-slate-900/60 rounded-lg">
                  {tokenResult.tokens.map((tok, idx) => {
                    const id = tokenResult.token_ids[idx];
                    return (
                      <div
                        key={idx}
                        className="px-2 py-1 rounded bg-slate-800 border border-slate-700/60 flex flex-col items-center text-center hover:border-indigo-500 transition-colors"
                      >
                        <span className="text-xs font-mono font-medium text-slate-100">
                          {tok === ' ' ? '␣' : tok}
                        </span>
                        <span className="text-[9px] font-mono text-slate-500">{id}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          {/* Multi-Model Comparison Table */}
          {tokenCompareResults && (
            <div className="space-y-3">
              <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                Cross-Model Tokenizer Comparison
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {tokenCompareResults.map((res) => (
                  <div key={res.model_id} className="p-4 bg-slate-950 border border-slate-800 rounded-xl space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-slate-200">{res.model_id.split('/').pop()}</span>
                      <span className="text-xs font-mono text-indigo-400 font-bold">{res.token_count} toks</span>
                    </div>
                    <div className="text-[11px] text-slate-400 flex items-center justify-between pt-1 border-t border-slate-800">
                      <span>Chars/Tok: {res.ratio_char_to_token}</span>
                      <span>Limit: {res.max_seq_length}</span>
                    </div>
                    <div className="flex flex-wrap gap-1 max-h-48 overflow-y-auto p-1.5 bg-slate-900 rounded">
                      {res.tokens.slice(0, 40).map((tok, i) => (
                        <span key={i} className="px-1.5 py-0.5 rounded bg-slate-800 text-[10px] font-mono text-slate-300">
                          {tok}
                        </span>
                      ))}
                      {res.tokens.length > 40 && (
                        <span className="text-[10px] text-slate-500 self-center">
                          +{res.tokens.length - 40} more...
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 3: VECTOR EXPLORER */}
      {activeTab === 'vector' && (
        <div className="space-y-6">
          {learningMode && (
            <div className="p-4 bg-indigo-950/30 border border-indigo-500/30 rounded-xl space-y-1">
              <h3 className="text-xs font-semibold text-indigo-300">
                Vector Dimensions and Numeric Representation
              </h3>
              <p className="text-xs text-slate-300">
                Each dimension represents an abstract semantic latent axis. Standard models output 384 or 768 floating point numbers. Unit normalization constrains the vector to the surface of a unit hypersphere (L2 norm = 1.0).
              </p>
            </div>
          )}

          {/* Generator Controls */}
          <div className="space-y-2">
            <label className="text-xs font-medium text-slate-300">Text to Encode</label>
            <textarea
              rows={3}
              value={vectorInputText}
              onChange={(e) => setVectorInputText(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs text-slate-100 focus:border-indigo-500 outline-none"
            />
            <button
              onClick={handleGenerateVector}
              disabled={isGeneratingVector || !vectorInputText.trim()}
              className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-medium transition-colors flex items-center gap-2 disabled:opacity-50"
            >
              {isGeneratingVector ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Layers className="w-3.5 h-3.5" />}
              Generate Embedding Vector
            </button>
          </div>

          {/* Vector Visualizer */}
          {vectorResult && vectorResult.items.length > 0 && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg text-center">
                  <span className="text-[10px] text-slate-500 uppercase block">Model</span>
                  <span className="text-xs font-bold text-slate-200 truncate block">{vectorResult.model_id.split('/').pop()}</span>
                </div>
                <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg text-center">
                  <span className="text-[10px] text-slate-500 uppercase block">Dimension</span>
                  <span className="text-lg font-bold text-indigo-400">{vectorResult.dimension}d</span>
                </div>
                <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg text-center">
                  <span className="text-[10px] text-slate-500 uppercase block">L2 Norm</span>
                  <span className="text-lg font-bold text-slate-200">{vectorResult.items[0].norm}</span>
                </div>
                <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg text-center">
                  <span className="text-[10px] text-slate-500 uppercase block">Inference Latency</span>
                  <span className="text-lg font-bold text-emerald-400">{vectorResult.latency_ms} ms</span>
                </div>
              </div>

              {/* Heatmap Preview of First 64 Dimensions */}
              <div className="p-4 bg-slate-950 border border-slate-800 rounded-xl space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-300">
                    Dimension Value Intensity (First 64 Values)
                  </span>
                  <span className="text-[10px] text-slate-500 font-mono">Range: [-0.15, +0.15]</span>
                </div>
                <div className="grid grid-cols-8 md:grid-cols-16 gap-1 p-2 bg-slate-900 rounded-lg">
                  {vectorResult.items[0].full_vector.slice(0, 64).map((val, idx) => {
                    const isPos = val >= 0;
                    const intensity = Math.min(Math.abs(val) * 6, 1);
                    return (
                      <div
                        key={idx}
                        title={`Dim ${idx}: ${val.toFixed(5)}`}
                        style={{
                          backgroundColor: isPos
                            ? `rgba(16, 185, 129, ${intensity})`
                            : `rgba(59, 130, 246, ${intensity})`,
                        }}
                        className="h-8 rounded flex items-center justify-center text-[9px] font-mono text-white/80 cursor-default"
                      >
                        {val.toFixed(2)}
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 4: SIMILARITY LAB */}
      {activeTab === 'similarity' && (
        <div className="space-y-6">
          {learningMode && (
            <div className="p-4 bg-indigo-950/30 border border-indigo-500/30 rounded-xl space-y-1">
              <h3 className="text-xs font-semibold text-indigo-300">
                Cosine Similarity vs. Dot Product vs. Euclidean Distance
              </h3>
              <p className="text-xs text-slate-300">
                <strong>Cosine Similarity</strong> measures the angle between vectors independent of magnitude (cos &theta; = (u &middot; v) / (||u|| &times; ||v||)).
                <strong>Dot Product</strong> multiplies coordinates (u &middot; v); when vectors are L2-normalized (||u|| = ||v|| = 1), Dot Product equals Cosine Similarity.
                <strong>Euclidean Distance</strong> is a distance (||u - v||), where 0 means identical and larger values mean greater divergence.
              </p>
            </div>
          )}

          {/* Sentence Pair Comparison */}
          <div className="p-5 bg-slate-950 border border-slate-800 rounded-xl space-y-4">
            <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <ArrowRightLeft className="w-4 h-4 text-indigo-400" />
              Pairwise Sentence Comparison
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-1">
                <label className="text-[11px] font-medium text-slate-400">Sentence A</label>
                <textarea
                  rows={3}
                  value={simTextA}
                  onChange={(e) => setSimTextA(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2.5 text-xs text-slate-100 outline-none focus:border-indigo-500"
                />
              </div>
              <div className="space-y-1">
                <label className="text-[11px] font-medium text-slate-400">Sentence B</label>
                <textarea
                  rows={3}
                  value={simTextB}
                  onChange={(e) => setSimTextB(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2.5 text-xs text-slate-100 outline-none focus:border-indigo-500"
                />
              </div>
            </div>

            <div className="flex items-center justify-between pt-2">
              <div className="flex items-center gap-3">
                <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={simNormalize}
                    onChange={(e) => setSimNormalize(e.target.checked)}
                    className="rounded bg-slate-900 border-slate-700 text-indigo-600 focus:ring-0"
                  />
                  <span>Apply L2 Normalization (||v|| = 1.0)</span>
                </label>
              </div>

              <button
                onClick={handleCalculateSimilarity}
                disabled={isCalculatingSim}
                className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-medium transition-colors flex items-center gap-2"
              >
                {isCalculatingSim ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Sparkles className="w-3.5 h-3.5" />}
                Calculate Similarities
              </button>
            </div>

            {/* Results Grid */}
            {simResult && (
              <div className="pt-4 border-t border-slate-800 space-y-3">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <div className="p-3 bg-slate-900 border border-slate-800 rounded-lg text-center">
                    <span className="text-[10px] text-slate-500 uppercase block">Cosine Similarity</span>
                    <span className="text-xl font-bold text-indigo-400">{simResult.cosine_similarity}</span>
                    <span className="text-[10px] text-slate-500 block mt-0.5">Range: [-1.0, 1.0]</span>
                  </div>
                  <div className="p-3 bg-slate-900 border border-slate-800 rounded-lg text-center">
                    <span className="text-[10px] text-slate-500 uppercase block">Dot Product</span>
                    <span className="text-xl font-bold text-slate-200">{simResult.dot_product}</span>
                    <span className="text-[10px] text-slate-500 block mt-0.5">
                      {simNormalize ? 'Equal to Cosine when normalized' : 'Unconstrained magnitude'}
                    </span>
                  </div>
                  <div className="p-3 bg-slate-900 border border-slate-800 rounded-lg text-center">
                    <span className="text-[10px] text-slate-500 uppercase block">Euclidean Distance</span>
                    <span className="text-xl font-bold text-emerald-400">{simResult.euclidean_distance}</span>
                    <span className="text-[10px] text-slate-500 block mt-0.5">0 = Identical (Distance metric)</span>
                  </div>
                </div>
                <p className="text-xs text-slate-400 bg-slate-900/60 p-2.5 rounded-lg border border-slate-800/80">
                  {simResult.explanation}
                </p>
              </div>
            )}
          </div>

          {/* Pairwise NxN Matrix */}
          <div className="p-5 bg-slate-950 border border-slate-800 rounded-xl space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-indigo-400" />
                Pairwise Similarity Matrix (NxN Heatmap)
              </h3>
              <div className="flex items-center gap-2">
                <select
                  value={matrixMetric}
                  onChange={(e) => setMatrixMetric(e.target.value)}
                  className="bg-slate-900 border border-slate-800 text-xs text-slate-200 rounded px-2.5 py-1"
                >
                  <option value="cosine">Cosine Similarity</option>
                  <option value="dot_product">Dot Product</option>
                  <option value="euclidean">Euclidean Distance</option>
                </select>
                <button
                  onClick={handleCalculateMatrix}
                  disabled={isCalculatingMatrix}
                  className="px-3 py-1 bg-indigo-600 hover:bg-indigo-500 text-white rounded text-xs font-medium"
                >
                  {isCalculatingMatrix ? 'Computing...' : 'Compute Matrix'}
                </button>
              </div>
            </div>

            {/* Matrix Result Table */}
            {matrixResult && (
              <div className="overflow-x-auto">
                <table className="w-full text-xs text-left">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400">
                      <th className="py-2 px-3 font-mono text-[10px]">#</th>
                      {matrixResult.texts.map((_, i) => (
                        <th key={i} className="py-2 px-3 font-mono text-[10px] text-center">
                          S{i + 1}
                        </th>
                      ))}
                      <th className="py-2 px-3 text-slate-300">Sentence Text</th>
                    </tr>
                  </thead>
                  <tbody>
                    {matrixResult.matrix.map((row, i) => (
                      <tr key={i} className="border-b border-slate-800/50 hover:bg-slate-900/40">
                        <td className="py-2.5 px-3 font-mono text-[10px] text-slate-400 font-bold">
                          S{i + 1}
                        </td>
                        {row.map((score, j) => {
                          const isDiagonal = i === j;
                          return (
                            <td
                              key={j}
                              className={`py-2.5 px-3 text-center font-mono text-xs font-semibold ${
                                isDiagonal
                                  ? 'bg-slate-800/60 text-slate-400'
                                  : score > 0.7
                                  ? 'bg-emerald-950/40 text-emerald-400'
                                  : score > 0.4
                                  ? 'bg-indigo-950/40 text-indigo-300'
                                  : 'text-slate-400'
                              }`}
                            >
                              {score.toFixed(3)}
                            </td>
                          );
                        })}
                        <td className="py-2.5 px-3 text-slate-300 truncate max-w-xs">
                          {matrixResult.texts[i]}
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

      {/* TAB 5: PROJECTION LAB */}
      {activeTab === 'projection' && (
        <div className="space-y-6">
          {learningMode && (
            <div className="p-4 bg-indigo-950/30 border border-indigo-500/30 rounded-xl space-y-1">
              <h3 className="text-xs font-semibold text-indigo-300">
                Dimensionality Reduction: PCA vs. t-SNE vs. UMAP
              </h3>
              <p className="text-xs text-slate-300">
                High-dimensional vector spaces (384d+) cannot be directly rendered in human perception.
                <strong>PCA</strong> performs linear orthogonal projection maximizing preserved global variance.
                <strong>t-SNE</strong> minimizes KL-divergence of neighbor probabilities, grouping local clusters but distorting global inter-cluster distances.
                <strong>UMAP</strong> optimizes topological manifold representation, balancing local clusters and global structure.
              </p>
            </div>
          )}

          {/* Parameter Controls & Empty State Guidance */}
          <div className="p-4 bg-slate-950 border border-slate-800 rounded-xl space-y-4">
            {documents.length === 0 ? (
              <div className="p-4 bg-slate-900 border border-dashed border-slate-800 rounded-lg text-center space-y-2">
                <Database className="w-8 h-8 text-slate-500 mx-auto" />
                <h4 className="text-xs font-semibold text-slate-200">No Ingested Documents Available</h4>
                <p className="text-xs text-slate-400 max-w-md mx-auto">
                  To project and visualize chunk embeddings in 2D space, you first need documents in PostgreSQL. Go to the <strong>Documents</strong> tab (Phase 02) to upload a PDF, DOCX, TXT, or Markdown file.
                </p>
              </div>
            ) : chunkExperiments.length === 0 ? (
              <div className="p-4 bg-amber-950/25 border border-amber-500/30 rounded-lg flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
                <div className="space-y-1 text-xs">
                  <span className="font-semibold text-amber-300">
                    No Chunking Experiments Found for Selected Document
                  </span>
                  <p className="text-slate-300">
                    To compute 2D projections (PCA, t-SNE, UMAP), the system requires previously generated chunks.
                    Go to <strong>Chunking Lab</strong> (Phase 03), select this document, choose a strategy (e.g., <em>Recursive</em> or <em>Semantic</em>), and click <strong>Create Experiment & Save Chunks</strong>.
                  </p>
                </div>
              </div>
            ) : null}

            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div>
                <label className="text-[11px] font-medium text-slate-400 block mb-1">Source Document</label>
                <select
                  value={selectedDocId}
                  onChange={(e) => setSelectedDocId(e.target.value)}
                  disabled={documents.length === 0}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 disabled:opacity-50"
                >
                  {documents.length === 0 ? (
                    <option value="">No documents uploaded</option>
                  ) : (
                    documents.map((d) => (
                      <option key={d.id} value={d.id}>
                        {d.original_filename}
                      </option>
                    ))
                  )}
                </select>
              </div>

              <div>
                <label className="text-[11px] font-medium text-slate-400 block mb-1">Chunking Experiment</label>
                <select
                  value={selectedChunkExpId}
                  onChange={(e) => setSelectedChunkExpId(e.target.value)}
                  disabled={chunkExperiments.length === 0}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 disabled:opacity-50"
                >
                  {chunkExperiments.length === 0 ? (
                    <option value="">No chunking experiments</option>
                  ) : (
                    chunkExperiments.map((exp) => (
                      <option key={exp.id} value={exp.id}>
                        {exp.strategy} ({exp.metrics?.total_chunks || exp.chunks_count || 0} chunks)
                      </option>
                    ))
                  )}
                </select>
              </div>

              <div>
                <label className="text-[11px] font-medium text-slate-400 block mb-1">Algorithm</label>
                <select
                  value={projAlgo}
                  onChange={(e) => setProjAlgo(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2 text-xs text-slate-200"
                >
                  <option value="pca">PCA (Principal Component Analysis)</option>
                  <option value="tsne">t-SNE (t-Distributed Stochastic Neighbor)</option>
                  <option value="umap">UMAP (Uniform Manifold Approximation)</option>
                </select>
              </div>

              <div className="flex items-end gap-2">
                <button
                  onClick={handleRunProjection}
                  disabled={isProjecting || (!selectedChunkExpId && matrixTexts.length === 0)}
                  className="w-full py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-medium transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
                >
                  {isProjecting ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Network className="w-3.5 h-3.5" />}
                  {selectedChunkExpId ? 'Project Chunks' : 'Project Sample Texts'}
                </button>
              </div>
            </div>
          </div>

          {/* Interactive 2D Scatter Plot */}
          {projectionData && (
            <div className="p-5 bg-slate-950 border border-slate-800 rounded-xl space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-slate-200">
                    2D Semantic Map ({projectionData.algorithm.toUpperCase()})
                  </h3>
                  <span className="text-[11px] text-slate-400">
                    {projectionData.points.length} points rendered
                  </span>
                </div>
                {projectionData.explained_variance_ratio && (
                  <span className="text-xs font-mono text-indigo-400">
                    Explained Variance: {(projectionData.explained_variance_ratio.reduce((a, b) => a + b, 0) * 100).toFixed(1)}%
                  </span>
                )}
              </div>

              {/* Warnings */}
              {projectionData.warnings && projectionData.warnings.length > 0 && (
                <div className="p-3 bg-amber-950/40 border border-amber-500/30 rounded-lg text-xs text-amber-300">
                  {projectionData.warnings.join(' ')}
                </div>
              )}

              {/* SVG 2D Scatter Plot */}
              <div className="relative w-full h-96 bg-slate-900 rounded-xl border border-slate-800/80 p-4 flex items-center justify-center overflow-hidden">
                {(() => {
                  const pts = projectionData.points;
                  if (!pts || pts.length === 0) return <span className="text-xs text-slate-500">No data points.</span>;

                  const xs = pts.map((p) => p.coordinates[0]);
                  const ys = pts.map((p) => p.coordinates[1]);
                  const minX = Math.min(...xs);
                  const maxX = Math.max(...xs);
                  const minY = Math.min(...ys);
                  const maxY = Math.max(...ys);
                  const rangeX = maxX - minX || 1.0;
                  const rangeY = maxY - minY || 1.0;

                  return (
                    <svg className="w-full h-full" viewBox="0 0 800 400">
                      {/* Grid lines */}
                      <line x1="50" y1="200" x2="750" y2="200" stroke="#334155" strokeDasharray="4" strokeWidth="1" />
                      <line x1="400" y1="30" x2="400" y2="370" stroke="#334155" strokeDasharray="4" strokeWidth="1" />

                      {/* Points */}
                      {pts.map((pt, i) => {
                        const cx = 50 + ((pt.coordinates[0] - minX) / rangeX) * 700;
                        const cy = 370 - ((pt.coordinates[1] - minY) / rangeY) * 340;
                        const isHovered = hoveredPoint && hoveredPoint.index === pt.index;

                        return (
                          <g key={i}>
                            <circle
                              cx={cx}
                              cy={cy}
                              r={isHovered ? 8 : 5}
                              fill={isHovered ? '#818cf8' : '#6366f1'}
                              stroke="#ffffff"
                              strokeWidth={isHovered ? 2 : 1}
                              className="cursor-pointer transition-all"
                              onMouseEnter={() => setHoveredPoint(pt)}
                              onMouseLeave={() => setHoveredPoint(null)}
                            />
                            <text
                              x={cx + 8}
                              y={cy - 8}
                              fill="#94a3b8"
                              fontSize="10"
                              fontFamily="monospace"
                            >
                              #{pt.index + 1}
                            </text>
                          </g>
                        );
                      })}
                    </svg>
                  );
                })()}

                {/* Hover Tooltip Overlay */}
                {hoveredPoint && (
                  <div className="absolute bottom-4 left-4 max-w-sm p-3 bg-slate-950/95 border border-indigo-500/60 rounded-lg shadow-xl text-xs space-y-1 z-10">
                    <span className="font-bold text-indigo-400 block">Chunk #{hoveredPoint.index + 1}</span>
                    <p className="text-slate-200 line-clamp-3">{hoveredPoint.label}</p>
                    <span className="text-[10px] text-slate-500 font-mono block">
                      Coords: [{hoveredPoint.coordinates[0].toFixed(3)}, {hoveredPoint.coordinates[1].toFixed(3)}]
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 6: BENCHMARK LAB */}
      {activeTab === 'benchmark' && (
        <div className="space-y-6">
          {learningMode && (
            <div className="p-4 bg-indigo-950/30 border border-indigo-500/30 rounded-xl space-y-1">
              <h3 className="text-xs font-semibold text-indigo-300">
                Evaluation Benchmarks & Discriminative Margin
              </h3>
              <p className="text-xs text-slate-300">
                A high-quality embedding model must maximize similarity for positive semantic pairs ($\mu_{pos}$) while keeping distractor/unrelated pairs low ($\mu_{neg}$).
                The <strong>Discriminative Margin ($\Delta = \mu_{pos} - \mu_{neg}$)</strong> measures the model's separation capacity.
              </p>
            </div>
          )}

          <div className="flex items-center justify-between p-4 bg-slate-950 border border-slate-800 rounded-xl">
            <div>
              <h3 className="text-sm font-semibold text-slate-200">
                Run Model Benchmark: {currentModelSpec?.display_name || selectedModel}
              </h3>
              <p className="text-xs text-slate-400">
                Tests semantic equivalence, cross-lingual French/English alignment, latency, and memory footprint.
              </p>
            </div>
            <button
              onClick={handleRunBenchmark}
              disabled={isBenchmarking}
              className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-medium transition-colors flex items-center gap-2"
            >
              {isBenchmarking ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Flame className="w-3.5 h-3.5" />}
              Execute Benchmark
            </button>
          </div>

          {/* Benchmark Results */}
          {benchmarkResult && (
            <div className="space-y-4">
              {/* Metrics Cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg text-center">
                  <span className="text-[10px] text-slate-500 uppercase block">Pos. Pair Mean</span>
                  <span className="text-lg font-bold text-emerald-400">
                    {benchmarkResult.quality_metrics.mean_positive_similarity}
                  </span>
                </div>
                <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg text-center">
                  <span className="text-[10px] text-slate-500 uppercase block">Neg. Pair Mean</span>
                  <span className="text-lg font-bold text-red-400">
                    {benchmarkResult.quality_metrics.mean_negative_similarity}
                  </span>
                </div>
                <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg text-center">
                  <span className="text-[10px] text-slate-500 uppercase block">Margin (Δ)</span>
                  <span className="text-lg font-bold text-indigo-400">
                    {benchmarkResult.quality_metrics.discriminative_margin}
                  </span>
                </div>
                <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg text-center">
                  <span className="text-[10px] text-slate-500 uppercase block">Throughput</span>
                  <span className="text-lg font-bold text-slate-200">
                    {benchmarkResult.performance_metrics.throughput_tokens_per_sec} tok/s
                  </span>
                </div>
              </div>

              {/* Storage Footprint Breakdown */}
              <div className="p-4 bg-slate-950 border border-slate-800 rounded-xl space-y-2">
                <span className="text-xs font-semibold text-slate-300 block">
                  Estimated Vector Memory & Storage Footprint ({benchmarkResult.dimension}d float32)
                </span>
                <div className="grid grid-cols-3 gap-3 text-center text-xs">
                  <div className="p-2.5 bg-slate-900 rounded-lg border border-slate-800">
                    <span className="text-slate-400 block text-[10px]">10,000 Chunks</span>
                    <span className="font-bold text-slate-200">{benchmarkResult.storage_footprint.storage_10k_vectors_mb} MB</span>
                  </div>
                  <div className="p-2.5 bg-slate-900 rounded-lg border border-slate-800">
                    <span className="text-slate-400 block text-[10px]">100,000 Chunks</span>
                    <span className="font-bold text-slate-200">{benchmarkResult.storage_footprint.storage_100k_vectors_mb} MB</span>
                  </div>
                  <div className="p-2.5 bg-slate-900 rounded-lg border border-slate-800">
                    <span className="text-slate-400 block text-[10px]">1,000,000 Chunks</span>
                    <span className="font-bold text-indigo-400">{benchmarkResult.storage_footprint.storage_1m_vectors_mb} MB</span>
                  </div>
                </div>
              </div>

              {/* Detailed Pairs Table */}
              <div className="p-4 bg-slate-950 border border-slate-800 rounded-xl overflow-x-auto space-y-3">
                <span className="text-xs font-semibold text-slate-300 block">Test Pair Breakdown</span>
                <table className="w-full text-xs text-left">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400">
                      <th className="py-2 px-3">Category</th>
                      <th className="py-2 px-3">Type</th>
                      <th className="py-2 px-3">Sentence Pair</th>
                      <th className="py-2 px-3 text-center">Cosine Sim</th>
                      <th className="py-2 px-3 text-center">Euclidean Dist</th>
                    </tr>
                  </thead>
                  <tbody>
                    {benchmarkResult.pairs.map((p, i) => (
                      <tr key={i} className="border-b border-slate-800/40 hover:bg-slate-900/40">
                        <td className="py-2.5 px-3 font-semibold text-slate-200">{p.category}</td>
                        <td className="py-2.5 px-3">
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                              p.is_positive
                                ? 'bg-emerald-950/60 text-emerald-400 border border-emerald-500/30'
                                : 'bg-red-950/60 text-red-400 border border-red-500/30'
                            }`}
                          >
                            {p.is_positive ? 'POSITIVE' : 'NEGATIVE'}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 max-w-sm space-y-1">
                          <p className="text-slate-200 truncate">A: {p.text_a}</p>
                          <p className="text-slate-400 truncate">B: {p.text_b}</p>
                        </td>
                        <td className="py-2.5 px-3 text-center font-mono font-bold text-indigo-300">
                          {p.cosine_similarity}
                        </td>
                        <td className="py-2.5 px-3 text-center font-mono text-slate-300">
                          {p.euclidean_distance}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 7: LEARNING CENTER */}
      {activeTab === 'learning' && (
        <div className="space-y-6 max-w-4xl">
          <div className="p-5 bg-slate-950 border border-slate-800 rounded-xl space-y-4">
            <h2 className="text-lg font-bold text-indigo-400 flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-indigo-400" />
              Mathematical Foundations of Vector Spaces
            </h2>

            <div className="space-y-4 text-xs text-slate-300 leading-relaxed">
              <div className="p-3.5 bg-slate-900 border border-slate-800 rounded-lg space-y-1">
                <span className="font-bold text-slate-100 block">1. Cosine Similarity</span>
                <p>
                  Measures the cosine of the angle between two multi-dimensional vectors:
                </p>
                <div className="p-2 bg-slate-950 font-mono text-indigo-300 rounded text-center">
                  cos(u, v) = (u &middot; v) / (||u|| &times; ||v||)
                </div>
                <p className="text-[11px] text-slate-400 mt-1">
                  Bounded in [-1, +1]. In dense text embeddings, values typically cluster in [0.0, 1.0].
                </p>
              </div>

              <div className="p-3.5 bg-slate-900 border border-slate-800 rounded-lg space-y-1">
                <span className="font-bold text-slate-100 block">2. Dot Product (Inner Product)</span>
                <p>
                  Sum of element-wise coordinate products:
                </p>
                <div className="p-2 bg-slate-950 font-mono text-indigo-300 rounded text-center">
                  u &middot; v = &sum; (u_i &times; v_i)
                </div>
                <p className="text-[11px] text-slate-400 mt-1">
                  When vectors are L2-normalized (||u|| = ||v|| = 1), dot product is mathematically identical to Cosine Similarity.
                </p>
              </div>

              <div className="p-3.5 bg-slate-900 border border-slate-800 rounded-lg space-y-1">
                <span className="font-bold text-slate-100 block">3. Euclidean Distance (L2)</span>
                <p>
                  Standard geometric distance between vector endpoints in Euclidean space:
                </p>
                <div className="p-2 bg-slate-950 font-mono text-indigo-300 rounded text-center">
                  d(u, v) = &radic;(&sum; (u_i - v_i)&sup2;)
                </div>
                <p className="text-[11px] text-slate-400 mt-1">
                  Notice: Euclidean distance is a <em>distance</em> (0 = identical), not a percentage or similarity score.
                </p>
              </div>

              <div className="p-3.5 bg-slate-900 border border-slate-800 rounded-lg space-y-1">
                <span className="font-bold text-slate-100 block">4. The Anisotropy Problem</span>
                <p>
                  Pre-trained transformer embeddings often occupy a narrow cone in vector space rather than dispersing uniformly across all directions. As a consequence, even completely unrelated sentences may exhibit positive cosine similarities (e.g. 0.35-0.50). Post-processing (whitening or normalization) or contrastive fine-tuning mitigates this effect.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function LayoutDashboardIcon(props) {
  return <BarChart3 {...props} />;
}
