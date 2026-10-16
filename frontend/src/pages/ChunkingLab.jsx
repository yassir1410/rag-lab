import React, { useState, useEffect, useCallback } from 'react';
import {
  Scissors,
  Layers,
  FileText,
  Sliders,
  Play,
  Save,
  RotateCcw,
  Sparkles,
  ArrowRightLeft,
  CheckCircle2,
  AlertTriangle,
  Info,
  Clock,
  Eye,
  GitFork,
  BarChart2,
  HelpCircle,
} from 'lucide-react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { api } from '../services/api';

export function ChunkingLab() {
  // Data state
  const [documents, setDocuments] = useState([]);
  const [strategies, setStrategies] = useState([]);
  const [selectedDocId, setSelectedDocId] = useState('');
  const [selectedDocContent, setSelectedDocContent] = useState(null);
  const [selectedStrategy, setSelectedStrategy] = useState('fixed_size');

  // Strategy parameters state
  const [config, setConfig] = useState({
    chunk_size: 400,
    chunk_overlap: 50,
    measurement_unit: 'character',
    sentence_overlap: 1,
    similarity_threshold: 0.70,
    min_chunk_size: 100,
    max_chunk_size: 1000,
    parent_chunk_size: 1200,
    parent_overlap: 100,
    child_chunk_size: 300,
    child_overlap: 30,
    preserve_tables: true,
    preserve_code_blocks: true,
  });

  // Execution & results state
  const [previewResult, setPreviewResult] = useState(null);
  const [selectedChunk, setSelectedChunk] = useState(null);
  const [activeView, setActiveView] = useState('chunks'); // 'chunks', 'source', 'distribution', 'compare', 'experiments'
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [saveSuccess, setSaveSuccess] = useState(null);

  // Comparison state
  const [compareStrategyA, setCompareStrategyA] = useState('fixed_size');
  const [compareStrategyB, setCompareStrategyB] = useState('sentence');
  const [comparisonResults, setComparisonResults] = useState(null);
  const [isComparing, setIsComparing] = useState(false);

  // Experiments history
  const [savedExperiments, setSavedExperiments] = useState([]);

  // Initial load
  useEffect(() => {
    async function loadInitialData() {
      try {
        const [docsData, stratData] = await Promise.all([
          api.listDocuments(100, 0),
          api.getChunkingStrategies(),
        ]);
        setDocuments(docsData.items || []);
        if (docsData.items?.length > 0) {
          setSelectedDocId(docsData.items[0].id);
        }
        setStrategies(stratData || []);
      } catch (err) {
        setError(err.message);
      }
    }
    loadInitialData();
  }, []);

  // When selected document changes, fetch its content
  useEffect(() => {
    if (!selectedDocId) return;
    async function fetchDoc() {
      try {
        const contentData = await api.getDocumentContent(selectedDocId);
        setSelectedDocContent(contentData);
        setPreviewResult(null);
        setSelectedChunk(null);
        loadExperiments(selectedDocId);
      } catch (err) {
        setError(err.message);
      }
    }
    fetchDoc();
  }, [selectedDocId]);

  const loadExperiments = async (docId) => {
    try {
      const data = await api.listChunkingExperiments(docId);
      setSavedExperiments(data.items || []);
    } catch {
      // Ignore background error
    }
  };

  // Run preview chunking
  const handleRunPreview = async () => {
    if (!selectedDocId) {
      setError('Please select an ingested document first.');
      return;
    }
    setIsLoading(true);
    setError(null);
    setSaveSuccess(null);
    try {
      const result = await api.previewChunking(selectedDocId, selectedStrategy, config);
      setPreviewResult(result);
      if (result.chunks?.length > 0) {
        setSelectedChunk(result.chunks[0]);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  // Save experiment to DB
  const handleSaveExperiment = async () => {
    if (!selectedDocId) return;
    setIsLoading(true);
    setError(null);
    try {
      const exp = await api.createChunkingExperiment(selectedDocId, selectedStrategy, config);
      setSaveSuccess(`Experiment successfully saved in database (ID: ${exp.id.substring(0, 8)}...)`);
      loadExperiments(selectedDocId);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  // Run comparison
  const handleRunComparison = async () => {
    if (!selectedDocId) return;
    setIsComparing(true);
    setError(null);
    try {
      const data = await api.compareChunkingStrategies(selectedDocId, [
        { strategy: compareStrategyA, config },
        { strategy: compareStrategyB, config },
      ]);
      setComparisonResults(data.results);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsComparing(false);
    }
  };

  const currentStrategyObj = strategies.find((s) => s.strategy_name === selectedStrategy);

  return (
    <div className="space-y-6 max-w-7xl mx-auto py-2">
      {/* Title & Badge */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-slate-100 flex items-center gap-2">
              <Scissors className="w-6 h-6 text-blue-400" />
              Text Chunking Laboratory
            </h1>
            <Badge variant="primary">Phase 03</Badge>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Test, inspect, and benchmark splitting algorithms (Fixed-size, Recursive, Sentence, Semantic, Parent-Child, Structure-Aware) on ingested documents.
          </p>
        </div>

        {selectedDocId && (
          <div className="flex items-center gap-2">
            <button
              onClick={handleRunPreview}
              disabled={isLoading}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold shadow-md shadow-blue-500/20 transition-colors disabled:opacity-50"
            >
              <Play className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
              Run Chunking
            </button>
            <button
              onClick={handleSaveExperiment}
              disabled={isLoading || !previewResult}
              className="flex items-center gap-2 px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition-colors disabled:opacity-40"
              title="Save experiment to database"
            >
              <Save className="w-3.5 h-3.5 text-emerald-400" />
              Save Experiment
            </button>
          </div>
        )}
      </div>

      {/* Notifications */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800/60 text-rose-300 text-xs flex items-start gap-3">
          <AlertTriangle className="w-4 h-4 text-rose-400 mt-0.5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {saveSuccess && (
        <div className="p-4 rounded-xl bg-emerald-950/40 border border-emerald-800/60 text-emerald-300 text-xs flex items-center gap-3">
          <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
          <span>{saveSuccess}</span>
        </div>
      )}

      {/* Control Strip: Document & Strategy Selectors */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Document Selector */}
        <Card>
          <CardHeader className="py-3 px-4">
            <CardTitle className="text-sm">
              <FileText className="w-4 h-4 text-blue-400" />
              Source Document
            </CardTitle>
          </CardHeader>
          <CardContent className="p-4 pt-1">
            {documents.length === 0 ? (
              <p className="text-xs text-slate-500">
                No documents available. Ingest a document in Phase 02 Documents tab first.
              </p>
            ) : (
              <select
                value={selectedDocId}
                onChange={(e) => setSelectedDocId(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-blue-500"
              >
                {documents.map((doc) => (
                  <option key={doc.id} value={doc.id}>
                    {doc.original_filename} ({doc.extraction_method} &bull; {doc.sections_count} sections)
                  </option>
                ))}
              </select>
            )}
            {selectedDocContent && (
              <div className="mt-2 text-[11px] text-slate-400 flex items-center gap-3">
                <span>{selectedDocContent.raw_text?.length || 0} characters</span>
                <span>&bull;</span>
                <span>{selectedDocContent.sections_count} sections</span>
                <span>&bull;</span>
                <span>{selectedDocContent.page_count ? `${selectedDocContent.page_count} pages` : 'Reflowable'}</span>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Strategy Selector */}
        <Card>
          <CardHeader className="py-3 px-4">
            <CardTitle className="text-sm">
              <Scissors className="w-4 h-4 text-purple-400" />
              Chunking Algorithm
            </CardTitle>
          </CardHeader>
          <CardContent className="p-4 pt-1">
            <select
              value={selectedStrategy}
              onChange={(e) => {
                setSelectedStrategy(e.target.value);
                setPreviewResult(null);
              }}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-purple-500"
            >
              {strategies.map((strat) => (
                <option key={strat.strategy_name} value={strat.strategy_name}>
                  {strat.display_name} ({strat.strategy_name})
                </option>
              ))}
            </select>
            {currentStrategyObj && (
              <p className="mt-2 text-[11px] text-slate-400 line-clamp-2">
                {currentStrategyObj.description}
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Strategy Parameters Panel */}
      <Card>
        <CardHeader className="py-3 px-4 flex flex-row items-center justify-between">
          <CardTitle className="text-sm">
            <Sliders className="w-4 h-4 text-blue-400" />
            Algorithm Parameters: <span className="font-mono text-blue-300">{selectedStrategy}</span>
          </CardTitle>
          <span className="text-[11px] text-slate-500">Live configuration</span>
        </CardHeader>
        <CardContent className="p-4 pt-1">
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 text-xs">
            {/* Standard parameters */}
            {selectedStrategy !== 'parent_child' && (
              <div>
                <label className="text-slate-300 font-medium block mb-1">
                  Target Size ({config.measurement_unit})
                </label>
                <input
                  type="number"
                  min="20"
                  max="4000"
                  value={config.chunk_size}
                  onChange={(e) => setConfig({ ...config, chunk_size: parseInt(e.target.value) || 100 })}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-slate-200"
                />
              </div>
            )}

            {selectedStrategy === 'fixed_size' || selectedStrategy === 'recursive' || selectedStrategy === 'structure_aware' ? (
              <div>
                <label className="text-slate-300 font-medium block mb-1">
                  Chunk Overlap ({config.measurement_unit})
                </label>
                <input
                  type="number"
                  min="0"
                  max={config.chunk_size - 1}
                  value={config.chunk_overlap}
                  onChange={(e) => setConfig({ ...config, chunk_overlap: parseInt(e.target.value) || 0 })}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-slate-200"
                />
              </div>
            ) : null}

            {/* Measurement Unit */}
            {selectedStrategy !== 'semantic' && (
              <div>
                <label className="text-slate-300 font-medium block mb-1">Measurement Unit</label>
                <select
                  value={config.measurement_unit}
                  onChange={(e) => setConfig({ ...config, measurement_unit: e.target.value })}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-slate-200"
                >
                  <option value="character">Characters (len)</option>
                  <option value="token">Tokens (BPE/cl100k)</option>
                </select>
              </div>
            )}

            {/* Sentence Overlap */}
            {selectedStrategy === 'sentence' && (
              <div>
                <label className="text-slate-300 font-medium block mb-1">Sentence Overlap (count)</label>
                <input
                  type="number"
                  min="0"
                  max="5"
                  value={config.sentence_overlap}
                  onChange={(e) => setConfig({ ...config, sentence_overlap: parseInt(e.target.value) || 0 })}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-slate-200"
                />
              </div>
            )}

            {/* Semantic Threshold */}
            {selectedStrategy === 'semantic' && (
              <>
                <div>
                  <label className="text-slate-300 font-medium block mb-1">
                    Similarity Breakpoint Threshold: {config.similarity_threshold}
                  </label>
                  <input
                    type="range"
                    min="0.3"
                    max="0.95"
                    step="0.05"
                    value={config.similarity_threshold}
                    onChange={(e) => setConfig({ ...config, similarity_threshold: parseFloat(e.target.value) })}
                    className="w-full accent-blue-500"
                  />
                </div>
                <div>
                  <label className="text-slate-300 font-medium block mb-1">Max Chunk Size (chars)</label>
                  <input
                    type="number"
                    min="200"
                    max="3000"
                    value={config.max_chunk_size}
                    onChange={(e) => setConfig({ ...config, max_chunk_size: parseInt(e.target.value) || 1000 })}
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-slate-200"
                  />
                </div>
              </>
            )}

            {/* Parent-Child Parameters */}
            {selectedStrategy === 'parent_child' && (
              <>
                <div>
                  <label className="text-slate-300 font-medium block mb-1">Parent Size (chars)</label>
                  <input
                    type="number"
                    min="400"
                    max="4000"
                    value={config.parent_chunk_size}
                    onChange={(e) => setConfig({ ...config, parent_chunk_size: parseInt(e.target.value) || 1200 })}
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-slate-200"
                  />
                </div>
                <div>
                  <label className="text-slate-300 font-medium block mb-1">Child Size (chars)</label>
                  <input
                    type="number"
                    min="50"
                    max="800"
                    value={config.child_chunk_size}
                    onChange={(e) => setConfig({ ...config, child_chunk_size: parseInt(e.target.value) || 300 })}
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-slate-200"
                  />
                </div>
                <div>
                  <label className="text-slate-300 font-medium block mb-1">Child Overlap (chars)</label>
                  <input
                    type="number"
                    min="0"
                    max="150"
                    value={config.child_overlap}
                    onChange={(e) => setConfig({ ...config, child_overlap: parseInt(e.target.value) || 30 })}
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-slate-200"
                  />
                </div>
              </>
            )}

            {/* Structure-Aware flags */}
            {selectedStrategy === 'structure_aware' && (
              <div className="flex items-center gap-4 pt-4">
                <label className="flex items-center gap-2 cursor-pointer text-slate-300">
                  <input
                    type="checkbox"
                    checked={config.preserve_tables}
                    onChange={(e) => setConfig({ ...config, preserve_tables: e.target.checked })}
                    className="rounded bg-slate-900 border-slate-700 text-blue-600 focus:ring-0"
                  />
                  <span>Preserve Tables</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer text-slate-300">
                  <input
                    type="checkbox"
                    checked={config.preserve_code_blocks}
                    onChange={(e) => setConfig({ ...config, preserve_code_blocks: e.target.checked })}
                    className="rounded bg-slate-900 border-slate-700 text-blue-600 focus:ring-0"
                  />
                  <span>Preserve Code Blocks</span>
                </label>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Navigation Tabs for Views */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setActiveView('chunks')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
              activeView === 'chunks'
                ? 'bg-blue-600 text-white'
                : 'text-slate-400 hover:text-slate-200 bg-slate-900'
            }`}
          >
            Chunks List {previewResult ? `(${previewResult.chunks.length})` : ''}
          </button>
          <button
            onClick={() => setActiveView('source')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
              activeView === 'source'
                ? 'bg-blue-600 text-white'
                : 'text-slate-400 hover:text-slate-200 bg-slate-900'
            }`}
          >
            Boundary & Overlap Mapping
          </button>
          <button
            onClick={() => setActiveView('distribution')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors flex items-center gap-1.5 ${
              activeView === 'distribution'
                ? 'bg-blue-600 text-white'
                : 'text-slate-400 hover:text-slate-200 bg-slate-900'
            }`}
          >
            <BarChart2 className="w-3.5 h-3.5" />
            Size Distribution & Metrics
          </button>
          <button
            onClick={() => setActiveView('compare')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors flex items-center gap-1.5 ${
              activeView === 'compare'
                ? 'bg-blue-600 text-white'
                : 'text-slate-400 hover:text-slate-200 bg-slate-900'
            }`}
          >
            <ArrowRightLeft className="w-3.5 h-3.5 text-amber-400" />
            Compare Strategies
          </button>
          <button
            onClick={() => setActiveView('experiments')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
              activeView === 'experiments'
                ? 'bg-blue-600 text-white'
                : 'text-slate-400 hover:text-slate-200 bg-slate-900'
            }`}
          >
            Saved Experiments ({savedExperiments.length})
          </button>
        </div>

        {previewResult && (
          <span className="text-[11px] text-slate-400 font-mono">
            Generated in {previewResult.processing_duration_ms} ms
          </span>
        )}
      </div>

      {/* VIEW 1: CHUNKS LIST */}
      {activeView === 'chunks' && (
        <div>
          {!previewResult ? (
            <div className="py-16 text-center text-xs text-slate-500 bg-slate-900/40 border border-slate-800 rounded-2xl space-y-2">
              <Scissors className="w-8 h-8 text-slate-600 mx-auto" />
              <p className="text-slate-300 font-medium">No chunks generated yet</p>
              <p>Select a document, adjust parameters, and click "Run Chunking" above.</p>
            </div>
          ) : (
            <div className="space-y-4">
              {/* Quick Metrics Header */}
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                <div className="p-3 rounded-xl bg-slate-900 border border-slate-800 text-xs">
                  <span className="text-slate-400 text-[10px] uppercase font-semibold">Total Chunks</span>
                  <div className="text-lg font-bold font-mono text-slate-100">
                    {previewResult.metrics.total_chunks}
                  </div>
                </div>
                <div className="p-3 rounded-xl bg-slate-900 border border-slate-800 text-xs">
                  <span className="text-slate-400 text-[10px] uppercase font-semibold">Mean Tokens</span>
                  <div className="text-lg font-bold font-mono text-blue-400">
                    {previewResult.metrics.token_metrics?.mean || 0}
                  </div>
                </div>
                <div className="p-3 rounded-xl bg-slate-900 border border-slate-800 text-xs">
                  <span className="text-slate-400 text-[10px] uppercase font-semibold">Token Range (Min-Max)</span>
                  <div className="text-lg font-bold font-mono text-slate-200">
                    {previewResult.metrics.token_metrics?.min || 0} &ndash; {previewResult.metrics.token_metrics?.max || 0}
                  </div>
                </div>
                <div className="p-3 rounded-xl bg-slate-900 border border-slate-800 text-xs">
                  <span className="text-slate-400 text-[10px] uppercase font-semibold">Duplicated Text</span>
                  <div className="text-lg font-bold font-mono text-emerald-400">
                    {(previewResult.metrics.duplicated_text_ratio * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="p-3 rounded-xl bg-slate-900 border border-slate-800 text-xs">
                  <span className="text-slate-400 text-[10px] uppercase font-semibold">Observed Overlap</span>
                  <div className="text-lg font-bold font-mono text-purple-400">
                    {previewResult.metrics.overlap_metrics?.mean_observed_chars || 0} chars
                  </div>
                </div>
              </div>

              {/* Chunks Grid */}
              <div className="space-y-3">
                {previewResult.chunks.map((chunk) => {
                  const isSelected = selectedChunk?.id === chunk.id;
                  const isParent = chunk.metadata?.custom?.tier === 'parent';
                  const isChild = chunk.metadata?.custom?.tier === 'child';

                  return (
                    <div
                      key={chunk.id}
                      onClick={() => setSelectedChunk(chunk)}
                      className={`p-4 rounded-xl border transition-all cursor-pointer ${
                        isSelected
                          ? 'bg-blue-950/40 border-blue-500 shadow-md shadow-blue-500/10'
                          : isParent
                          ? 'bg-purple-950/20 border-purple-800/40 hover:border-purple-600'
                          : isChild
                          ? 'bg-slate-900/60 border-slate-800 hover:border-slate-700 ml-6'
                          : 'bg-slate-900/80 border-slate-800 hover:border-slate-700'
                      }`}
                    >
                      <div className="flex items-center justify-between text-xs mb-2">
                        <div className="flex items-center gap-2">
                          <span className="px-2 py-0.5 rounded bg-slate-800 font-mono text-[10px] text-slate-300">
                            #{chunk.sequence_index}
                          </span>
                          {isParent && <Badge variant="primary">Parent Chunk</Badge>}
                          {isChild && <Badge variant="default">Child Chunk</Badge>}
                          {chunk.metadata?.content_type && chunk.metadata.content_type !== 'text' && (
                            <Badge variant="warning">{chunk.metadata.content_type}</Badge>
                          )}
                          {chunk.metadata?.page_numbers?.length > 0 && (
                            <Badge variant="default">
                              Page {chunk.metadata.page_numbers.join(', ')}
                            </Badge>
                          )}
                          {chunk.metadata?.headings?.length > 0 && (
                            <span className="text-slate-300 font-semibold text-[11px] truncate max-w-sm">
                              {chunk.metadata.headings[0]}
                            </span>
                          )}
                        </div>

                        <div className="flex items-center gap-3 text-[11px] text-slate-400 font-mono">
                          <span>{chunk.token_count} tokens</span>
                          <span>&bull;</span>
                          <span>{chunk.char_count} chars</span>
                        </div>
                      </div>

                      <div className="bg-slate-950 p-3 rounded-lg border border-slate-900 font-mono text-xs text-slate-300 whitespace-pre-wrap leading-relaxed select-text">
                        {chunk.content}
                      </div>

                      {/* Breakpoint metadata for semantic chunking */}
                      {chunk.metadata?.breakpoint_distance !== undefined &&
                        chunk.metadata?.breakpoint_distance !== null && (
                          <div className="mt-2 text-[10px] text-amber-400 flex items-center gap-1.5 font-mono">
                            <Sparkles className="w-3 h-3" />
                            <span>
                              Semantic distance to next: {chunk.metadata.breakpoint_distance}
                            </span>
                          </div>
                        )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* VIEW 2: BOUNDARY & OVERLAP MAPPING */}
      {activeView === 'source' && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">
              <Eye className="w-4 h-4 text-emerald-400" />
              Source Document Text & Chunk Mapping
            </CardTitle>
            <CardDescription>
              Click any chunk below to inspect its exact character span inside the original document.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {selectedDocContent ? (
              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 font-mono text-xs text-slate-300 leading-relaxed max-h-[500px] overflow-y-auto whitespace-pre-wrap select-text">
                {selectedDocContent.normalized_text || selectedDocContent.raw_text}
              </div>
            ) : (
              <p className="text-xs text-slate-500">No document content loaded.</p>
            )}

            {selectedChunk && (
              <div className="p-4 rounded-xl bg-blue-950/30 border border-blue-800/40 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-semibold text-blue-300">
                    Selected Chunk #{selectedChunk.sequence_index}
                  </span>
                  <span className="text-[11px] font-mono text-slate-400">
                    Offsets: {selectedChunk.metadata?.start_char ?? 'N/A'} &rarr;{' '}
                    {selectedChunk.metadata?.end_char ?? 'N/A'}
                  </span>
                </div>
                <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 font-mono text-xs text-slate-200">
                  {selectedChunk.content}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* VIEW 3: SIZE DISTRIBUTION & METRICS */}
      {activeView === 'distribution' && (
        <div className="space-y-6">
          {!previewResult ? (
            <div className="py-16 text-center text-xs text-slate-500 bg-slate-900/40 border border-slate-800 rounded-2xl">
              Run chunking on a document to visualize token distribution metrics.
            </div>
          ) : (
            <div className="space-y-6">
              {/* Educational Notice */}
              <div className="p-4 rounded-xl bg-blue-950/30 border border-blue-800/40 text-blue-200 text-xs space-y-2">
                <div className="flex items-center gap-2 font-bold text-blue-300">
                  <Info className="w-4 h-4 text-blue-400 shrink-0" />
                  Chunk Size Distribution & Downstream RAG Precision
                </div>
                <p className="text-blue-200/90 leading-relaxed">
                  High variance in chunk sizes can cause embedding distortion. Chunks that are too small lack semantic context, leading to false-positive retrieval. Chunks that are too large dilute the embedding vector and overwhelm the LLM prompt budget.
                </p>
              </div>

              {/* Statistical Percentiles Grid */}
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800 text-xs space-y-1">
                  <span className="text-slate-500 font-semibold text-[10px] uppercase">P10 (Bottom 10%)</span>
                  <div className="text-lg font-bold font-mono text-slate-200">
                    {previewResult.metrics.token_metrics?.percentiles?.p10 || 0} tokens
                  </div>
                </div>
                <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800 text-xs space-y-1">
                  <span className="text-slate-500 font-semibold text-[10px] uppercase">P25 (First Quartile)</span>
                  <div className="text-lg font-bold font-mono text-slate-200">
                    {previewResult.metrics.token_metrics?.percentiles?.p25 || 0} tokens
                  </div>
                </div>
                <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800 text-xs space-y-1">
                  <span className="text-slate-500 font-semibold text-[10px] uppercase">P50 (Median)</span>
                  <div className="text-lg font-bold font-mono text-emerald-400">
                    {previewResult.metrics.token_metrics?.percentiles?.p50 || 0} tokens
                  </div>
                </div>
                <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800 text-xs space-y-1">
                  <span className="text-slate-500 font-semibold text-[10px] uppercase">P75 (Third Quartile)</span>
                  <div className="text-lg font-bold font-mono text-slate-200">
                    {previewResult.metrics.token_metrics?.percentiles?.p75 || 0} tokens
                  </div>
                </div>
                <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800 text-xs space-y-1">
                  <span className="text-slate-500 font-semibold text-[10px] uppercase">P90 (Top 10%)</span>
                  <div className="text-lg font-bold font-mono text-slate-200">
                    {previewResult.metrics.token_metrics?.percentiles?.p90 || 0} tokens
                  </div>
                </div>
              </div>

              {/* Visual Histogram Chart */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">
                    <BarChart2 className="w-4 h-4 text-purple-400" />
                    Chunk Token Length Distribution (Histogram)
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2 pt-2">
                    {previewResult.chunks.map((c, i) => {
                      const maxLen = previewResult.metrics.token_metrics?.max || 1;
                      const pct = Math.max(5, Math.round((c.token_count / maxLen) * 100));

                      return (
                        <div key={i} className="flex items-center gap-3 text-xs">
                          <span className="w-12 text-slate-400 font-mono text-[10px] shrink-0">
                            #{c.sequence_index}
                          </span>
                          <div className="flex-1 bg-slate-950 rounded-full h-4 overflow-hidden border border-slate-800">
                            <div
                              className="bg-blue-600 h-full rounded-full transition-all flex items-center justify-end pr-2 text-[10px] font-mono text-white"
                              style={{ width: `${pct}%` }}
                            >
                              {c.token_count > 15 ? `${c.token_count}t` : ''}
                            </div>
                          </div>
                          <span className="w-16 text-right font-mono text-slate-300 text-[11px] shrink-0">
                            {c.token_count} tokens
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </div>
      )}

      {/* VIEW 4: COMPARE STRATEGIES */}
      {activeView === 'compare' && (
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">
                <ArrowRightLeft className="w-4 h-4 text-amber-400" />
                Side-by-Side Algorithm Comparison
              </CardTitle>
              <CardDescription>
                Execute two different chunking configurations on the same document to compare chunk counts, size consistency, and text duplication.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label className="text-xs text-slate-300 font-medium block mb-1">Strategy A</label>
                  <select
                    value={compareStrategyA}
                    onChange={(e) => setCompareStrategyA(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200"
                  >
                    {strategies.map((s) => (
                      <option key={s.strategy_name} value={s.strategy_name}>
                        {s.display_name}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="text-xs text-slate-300 font-medium block mb-1">Strategy B</label>
                  <select
                    value={compareStrategyB}
                    onChange={(e) => setCompareStrategyB(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200"
                  >
                    {strategies.map((s) => (
                      <option key={s.strategy_name} value={s.strategy_name}>
                        {s.display_name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <button
                onClick={handleRunComparison}
                disabled={isComparing}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-amber-600 hover:bg-amber-500 text-white text-xs font-semibold transition-colors disabled:opacity-50"
              >
                <ArrowRightLeft className={`w-3.5 h-3.5 ${isComparing ? 'animate-spin' : ''}`} />
                Run Benchmark Comparison
              </button>
            </CardContent>
          </Card>

          {/* Comparison Results Cards */}
          {comparisonResults && comparisonResults.length === 2 && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {comparisonResults.map((res, idx) => (
                <Card key={idx} className="border-slate-700">
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-base text-blue-300">
                        {idx === 0 ? 'Strategy A: ' : 'Strategy B: '}
                        {res.strategy}
                      </CardTitle>
                      <Badge variant="primary">{res.chunks.length} chunks</Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-4 text-xs">
                    <div className="grid grid-cols-2 gap-2">
                      <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                        <span className="text-slate-400 text-[10px] uppercase">Mean Tokens</span>
                        <p className="text-base font-bold font-mono text-slate-100">
                          {res.metrics.token_metrics?.mean || 0}
                        </p>
                      </div>
                      <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                        <span className="text-slate-400 text-[10px] uppercase">Duplication Ratio</span>
                        <p className="text-base font-bold font-mono text-emerald-400">
                          {(res.metrics.duplicated_text_ratio * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>

                    <div className="space-y-2 max-h-96 overflow-y-auto">
                      <p className="text-slate-400 font-semibold text-[11px]">First 3 Chunks Preview:</p>
                      {res.chunks.slice(0, 3).map((c) => (
                        <div key={c.id} className="p-3 rounded-lg bg-slate-950 border border-slate-900 font-mono text-xs text-slate-300">
                          <span className="text-blue-400 font-semibold block mb-1">
                            #{c.sequence_index} ({c.token_count} tokens):
                          </span>
                          {c.content.substring(0, 160)}...
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>
      )}

      {/* VIEW 5: SAVED EXPERIMENTS */}
      {activeView === 'experiments' && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">
              <Clock className="w-4 h-4 text-emerald-400" />
              Persisted Chunking Experiments ({savedExperiments.length})
            </CardTitle>
            <CardDescription>
              Experiments stored in PostgreSQL with reproducible metrics and parameters
            </CardDescription>
          </CardHeader>
          <CardContent className="p-0">
            {savedExperiments.length === 0 ? (
              <div className="p-8 text-center text-xs text-slate-500">
                No saved experiments for this document. Run a strategy and click "Save Experiment".
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-slate-700/80 bg-slate-900/60 text-slate-400 text-[10px] uppercase">
                      <th className="py-3 px-4">Experiment ID</th>
                      <th className="py-3 px-4">Strategy</th>
                      <th className="py-3 px-4">Total Chunks</th>
                      <th className="py-3 px-4">Mean Tokens</th>
                      <th className="py-3 px-4">Date</th>
                      <th className="py-3 px-4 text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800 text-slate-300">
                    {savedExperiments.map((exp) => (
                      <tr key={exp.id} className="hover:bg-slate-800/40">
                        <td className="py-3 px-4 font-mono text-blue-400">
                          {exp.id.substring(0, 8)}...
                        </td>
                        <td className="py-3 px-4 font-semibold">{exp.strategy}</td>
                        <td className="py-3 px-4 font-mono">{exp.chunks_count}</td>
                        <td className="py-3 px-4 font-mono">
                          {exp.metrics?.token_metrics?.mean || '—'}
                        </td>
                        <td className="py-3 px-4 text-slate-400">
                          {new Date(exp.created_at).toLocaleDateString()}
                        </td>
                        <td className="py-3 px-4 text-right">
                          <button
                            onClick={async () => {
                              await api.deleteChunkingExperiment(exp.id);
                              loadExperiments(selectedDocId);
                            }}
                            className="p-1 rounded hover:bg-rose-900/40 text-slate-400 hover:text-rose-400"
                            title="Delete experiment"
                          >
                            Delete
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
