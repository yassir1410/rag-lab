import React, { useState, useEffect } from 'react';
import {
  CheckCircle,
  FlaskConical,
  BarChart2,
  Database,
  Play,
  Layers,
  ArrowRight,
  TrendingUp,
  AlertTriangle,
  HelpCircle,
  Clock,
  Coins,
  ShieldCheck,
  FileText,
  Search,
  RefreshCw,
  Plus,
  Trash2,
  Download,
  Upload,
  ChevronDown,
  ChevronRight,
  Sparkles,
  Award,
  Sliders,
  Scale,
  X,
  ExternalLink,
  Edit3,
  GitMerge,
  Check,
  Eye,
  Save,
  Code,
  Copy,
} from 'lucide-react';
import { api } from '../services/api';

const RETRIEVAL_STRATEGIES = [
  { id: 'dense_only', label: '1. Dense Vector Only', desc: 'Exact KNN or HNSW cosine similarity' },
  { id: 'lexical_only', label: '2. Lexical Only (GIN)', desc: 'PostgreSQL tsvector full-text search' },
  { id: 'hybrid_linear', label: '3. Hybrid Linear', desc: 'Normalized weighted blend: α · Dense + (1-α) · Lex' },
  { id: 'hybrid_rrf', label: '4. Hybrid RRF', desc: 'Reciprocal Rank Fusion (k=60)' },
  { id: 'dense_rerank', label: '5. Dense + Reranker', desc: 'Dense retrieval + Cross-Encoder TinyBERT' },
  { id: 'hybrid_rrf_rerank', label: '6. Hybrid RRF + Reranker', desc: 'Dual-channel RRF + Cross-Encoder' },
  { id: 'multi_query_hybrid_rerank', label: '7. Multi-Query + Hybrid + Reranker', desc: 'Multi-query expansion + RRF + Cross-Encoder' },
  { id: 'adaptive_context', label: '8. Adaptive Context Selection', desc: 'Contiguous neighbor expansion + deduplication' },
];

export function EvaluationStudio() {
  const [activeTab, setActiveTab] = useState('overview'); // 'overview' | 'datasets' | 'launcher' | 'results' | 'routing' | 'traces' | 'comparison' | 'learning'
  const [datasets, setDatasets] = useState([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState('');
  const [runs, setRuns] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState('');
  const [selectedRun, setSelectedRun] = useState(null);
  const [runQuestions, setRunQuestions] = useState([]);
  const [selectedQuestionItem, setSelectedQuestionItem] = useState(null);
  const [metricsCatalog, setMetricsCatalog] = useState([]);

  // Phase 13 Traces & Progress State
  const [runTraces, setRunTraces] = useState([]);
  const [selectedTrace, setSelectedTrace] = useState(null);
  const [showTraceModal, setShowTraceModal] = useState(false);
  const [runProgress, setRunProgress] = useState(null);
  const [isExporting, setIsExporting] = useState(false);

  // Comparison State
  const [compareRunAId, setCompareRunAId] = useState('');
  const [compareRunBId, setCompareRunBId] = useState('');
  const [comparisonResult, setComparisonResult] = useState(null);

  // Loading and Error States
  const [isLoading, setIsLoading] = useState(false);
  const [isExecutingRun, setIsExecutingRun] = useState(false);
  const [error, setError] = useState(null);

  // New Dataset / Question Modal State
  const [showNewDatasetModal, setShowNewDatasetModal] = useState(false);
  const [newDatasetName, setNewDatasetName] = useState('');
  const [newDatasetDesc, setNewDatasetDesc] = useState('');

  const [questions, setQuestions] = useState([]);
  const [showAddQuestionModal, setShowAddQuestionModal] = useState(false);
  const [newQuestionText, setNewQuestionText] = useState('');
  const [newRefAnswer, setNewRefAnswer] = useState('');
  const [newCategory, setNewCategory] = useState('factual');
  const [newDifficulty, setNewDifficulty] = useState('medium');
  const [newModality, setNewModality] = useState('text');
  const [newIsAnswerable, setNewIsAnswerable] = useState(true);

  // Import / Export State
  const [showImportModal, setShowImportModal] = useState(false);
  const [importJsonText, setImportJsonText] = useState('');

  // Experiment Launcher Form State
  const [runName, setRunName] = useState('');
  const [runType, setRunType] = useState('unified'); // 'end_to_end' | 'retrieval_only' | 'unified'
  const [routingMode, setRoutingMode] = useState('auto');
  const [retrievalStrategy, setRetrievalStrategy] = useState('dense_only');
  const [searchAlgorithm, setSearchAlgorithm] = useState('exact');
  const [topK, setTopK] = useState(5);
  const [hybridAlpha, setHybridAlpha] = useState(0.5);
  const [rrfK, setRrfK] = useState(60);
  const [rerankerModel, setRerankerModel] = useState('cross-encoder/ms-marco-TinyBERT-L-2-v2');
  const [enableQueryExpansion, setEnableQueryExpansion] = useState(false);
  const [enableNeighborExpansion, setEnableNeighborExpansion] = useState(false);
  const [llmProvider, setLlmProvider] = useState('mock');
  const [llmModel, setLlmModel] = useState('mock-instruct-v1');
  const [promptTemplate, setPromptTemplate] = useState('grounded');
  const [judgeProvider, setJudgeProvider] = useState('heuristic');
  const [maxQuestionsLimit, setMaxQuestionsLimit] = useState(24);
  const [runAsync, setRunAsync] = useState(false);

  // Evidence Auditor Modal State
  const [showAuditorModal, setShowAuditorModal] = useState(false);
  const [auditingQuestion, setAuditingQuestion] = useState(null);
  const [auditLabels, setAuditLabels] = useState({});
  const [auditNotes, setAuditNotes] = useState('');
  const [auditCandidateChunks, setAuditCandidateChunks] = useState([]);
  const [isLoadingAuditChunks, setIsLoadingAuditChunks] = useState(false);
  const [isSavingAudit, setIsSavingAudit] = useState(false);

  // Educational Sandbox 1 State: Precision@K vs Recall@K
  const [sandboxK, setSandboxK] = useState(3);
  // Educational Sandbox 2 State: Top-K Tradeoff
  const [sandboxTradeoffTopK, setSandboxTradeoffTopK] = useState(5);

  // Initial load
  useEffect(() => {
    loadInitialData();
  }, []);

  // When selectedDatasetId changes, load questions
  useEffect(() => {
    if (selectedDatasetId) {
      loadQuestions(selectedDatasetId);
    }
  }, [selectedDatasetId]);

  // When selectedRunId changes, load run details and traces
  useEffect(() => {
    if (selectedRunId) {
      loadRunDetails(selectedRunId);
    }
  }, [selectedRunId]);

  // Polling for progress when a run is running
  useEffect(() => {
    let interval = null;
    if (selectedRun && selectedRun.status === 'running') {
      interval = setInterval(async () => {
        try {
          const prog = await api.getEvaluationRunProgress(selectedRun.id);
          setRunProgress(prog);
          if (prog.status !== 'running') {
            loadRunDetails(selectedRun.id);
            clearInterval(interval);
          }
        } catch {
          // ignore transient poll error
        }
      }, 2000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [selectedRun]);

  const loadInitialData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [dsList, rList, cat] = await Promise.all([
        api.listEvaluationDatasets(),
        api.listEvaluationRuns(),
        api.getEvaluationMetricsCatalog(),
      ]);
      setDatasets(dsList || []);
      setRuns(rList || []);
      setMetricsCatalog(cat || []);

      if (dsList && dsList.length > 0 && !selectedDatasetId) {
        // Prefer Phase 13 benchmark dataset if present
        const p13Ds = dsList.find((d) => d.name.includes('Phase 13'));
        setSelectedDatasetId(p13Ds ? p13Ds.id : dsList[0].id);
      }
      if (rList && rList.length > 0 && !selectedRunId) {
        setSelectedRunId(rList[0].id);
        if (rList.length > 1) {
          setCompareRunAId(rList[1].id);
          setCompareRunBId(rList[0].id);
        }
      }
    } catch (err) {
      setError(err.message || 'Failed to initialize evaluation studio.');
    } finally {
      setIsLoading(false);
    }
  };

  const loadQuestions = async (datasetId) => {
    try {
      const qList = await api.listEvaluationQuestions(datasetId);
      setQuestions(qList || []);
    } catch (err) {
      console.error('Failed to load questions:', err);
    }
  };

  const loadRunDetails = async (runId) => {
    try {
      const [rDetail, rItems, traces] = await Promise.all([
        api.getEvaluationRun(runId),
        api.getEvaluationRunQuestions(runId),
        api.getEvaluationRunTraces(runId).catch(() => []),
      ]);
      setSelectedRun(rDetail);
      setRunQuestions(rItems || []);
      setRunTraces(traces || []);
      if (rItems && rItems.length > 0) {
        setSelectedQuestionItem(rItems[0]);
      }
    } catch (err) {
      console.error('Failed to load run details:', err);
    }
  };

  const handleCreateDataset = async (e) => {
    e.preventDefault();
    if (!newDatasetName.trim()) return;
    try {
      const created = await api.createEvaluationDataset({
        name: newDatasetName.trim(),
        description: newDatasetDesc.trim() || null,
      });
      setDatasets([created, ...datasets]);
      setSelectedDatasetId(created.id);
      setShowNewDatasetModal(false);
      setNewDatasetName('');
      setNewDatasetDesc('');
    } catch (err) {
      alert(err.message || 'Error creating dataset');
    }
  };

  const handleAddQuestion = async (e) => {
    e.preventDefault();
    if (!newQuestionText.trim() || !newRefAnswer.trim()) return;
    try {
      const created = await api.addEvaluationQuestion(selectedDatasetId, {
        question: newQuestionText.trim(),
        reference_answer: newRefAnswer.trim(),
        category: newCategory,
        difficulty: newDifficulty,
        modality: newModality,
        is_answerable: newIsAnswerable,
        is_reviewed: false,
      });
      setQuestions([...questions, created]);
      setShowAddQuestionModal(false);
      setNewQuestionText('');
      setNewRefAnswer('');
    } catch (err) {
      alert(err.message || 'Error adding question');
    }
  };

  const handleToggleReview = async (question) => {
    try {
      const updated = await api.reviewEvaluationQuestion(question.id, {
        is_reviewed: !question.is_reviewed,
        reviewer_notes: !question.is_reviewed ? 'Human verified ground truth.' : null,
      });
      setQuestions(questions.map((q) => (q.id === question.id ? updated : q)));
    } catch (err) {
      alert(err.message || 'Error updating review status');
    }
  };

  const handleDeleteQuestion = async (questionId) => {
    if (!confirm('Are you sure you want to delete this evaluation question?')) return;
    try {
      await api.deleteEvaluationQuestion(questionId);
      setQuestions(questions.filter((q) => q.id !== questionId));
    } catch (err) {
      alert(err.message || 'Error deleting question');
    }
  };

  const handleExportDataset = async (format = 'json') => {
    if (!selectedDatasetId) return;
    try {
      const payload = await api.exportEvaluationDataset(selectedDatasetId, format);
      const isJsonl = format === 'jsonl';
      const blob = new Blob([isJsonl ? payload : JSON.stringify(payload, null, 2)], {
        type: isJsonl ? 'application/x-ndjson' : 'application/json',
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `dataset_${selectedDatasetId}.${isJsonl ? 'jsonl' : 'json'}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      alert(err.message || 'Error exporting dataset');
    }
  };

  const handleExportRun = async (format = 'json') => {
    if (!selectedRunId) return;
    setIsExporting(true);
    try {
      const res = await api.exportEvaluationRun(selectedRunId, format);
      const isCsv = format === 'csv';
      const content = isCsv ? res : JSON.stringify(res, null, 2);
      const blob = new Blob([content], {
        type: isCsv ? 'text/csv' : 'application/json',
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `run_${selectedRunId}.${isCsv ? 'csv' : 'json'}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      alert(`Error exporting run: ${err.message}`);
    } finally {
      setIsExporting(false);
    }
  };

  const handleImportJson = async () => {
    if (!selectedDatasetId || !importJsonText.trim()) return;
    try {
      let parsed;
      if (importJsonText.trim().startsWith('{')) {
        parsed = JSON.parse(importJsonText);
      } else {
        // Try parsing JSONL lines
        const lines = importJsonText
          .trim()
          .split('\n')
          .filter(Boolean)
          .map((line) => JSON.parse(line));
        parsed = { questions: lines };
      }
      const res = await api.importEvaluationQuestions(selectedDatasetId, parsed);
      alert(`Imported ${res.imported_questions} questions (Skipped ${res.skipped_questions} duplicates).`);
      setShowImportModal(false);
      setImportJsonText('');
      loadQuestions(selectedDatasetId);
    } catch (err) {
      alert(err.message || 'Invalid JSON/JSONL format or import error');
    }
  };

  const handleOpenAuditor = async (question) => {
    setAuditingQuestion(question);
    setAuditLabels(question.relevance_labels || {});
    setAuditNotes(question.reviewer_notes || '');
    setShowAuditorModal(true);
    setIsLoadingAuditChunks(true);
    try {
      const res = await api.searchVectors({
        query: question.question,
        top_k: 5,
        explain_plan: false,
      });
      setAuditCandidateChunks(res?.results || []);
    } catch (err) {
      console.error('Failed to load candidate chunks for question:', err);
      setAuditCandidateChunks([]);
    } finally {
      setIsLoadingAuditChunks(false);
    }
  };

  const handleSaveAudit = async () => {
    if (!auditingQuestion) return;
    setIsSavingAudit(true);
    try {
      await api.labelEvaluationQuestion(auditingQuestion.id, {
        relevance_labels: auditLabels,
        reviewer_notes: auditNotes,
      });
      await loadQuestions(selectedDatasetId);
      setShowAuditorModal(false);
    } catch (err) {
      alert(`Failed to save evidence audit: ${err.message}`);
    } finally {
      setIsSavingAudit(false);
    }
  };

  const handleLaunchRun = async (e) => {
    e.preventDefault();
    if (!selectedDatasetId) {
      alert('Please select a dataset first.');
      return;
    }
    setIsExecutingRun(true);
    setError(null);

    try {
      const defaultName = `${runType.toUpperCase()} Run - ${new Date().toLocaleTimeString()}`;
      const payload = {
        dataset_id: selectedDatasetId,
        name: runName.trim() || defaultName,
        run_type: runType,
        routing_mode: routingMode,
        retrieval_strategy: retrievalStrategy,
        search_algorithm: searchAlgorithm,
        top_k: parseInt(topK),
        hybrid_alpha: parseFloat(hybridAlpha),
        rrf_k: parseInt(rrfK),
        reranker_model: rerankerModel,
        enable_query_expansion: Boolean(enableQueryExpansion),
        enable_neighbor_expansion: Boolean(enableNeighborExpansion),
        llm_provider: llmProvider,
        llm_model: llmModel,
        prompt_template: promptTemplate,
        judge_provider: judgeProvider,
        max_questions: parseInt(maxQuestionsLimit) || null,
      };

      const run = await api.createEvaluationRun(payload, runAsync);
      setRuns([run, ...runs]);
      setSelectedRunId(run.id);
      await loadRunDetails(run.id);
      setActiveTab('results');
    } catch (err) {
      setError(err.message || 'Evaluation run failed.');
    } finally {
      setIsExecutingRun(false);
    }
  };

  const handleCancelRun = async (runId) => {
    try {
      await api.cancelEvaluationRun(runId);
      await loadRunDetails(runId);
    } catch (err) {
      alert(`Failed to cancel run: ${err.message}`);
    }
  };

  const handleCompareRuns = async () => {
    if (!compareRunAId || !compareRunBId || compareRunAId === compareRunBId) {
      alert('Please select two distinct runs to compare.');
      return;
    }
    setIsLoading(true);
    try {
      const comp = await api.compareEvaluationRuns(compareRunAId, compareRunBId);
      setComparisonResult(comp);
    } catch (err) {
      alert(err.message || 'Error comparing runs');
    } finally {
      setIsLoading(false);
    }
  };

  const handleProviderChange = (e) => {
    const p = e.target.value;
    setLlmProvider(p);
    if (p === 'mock') setLlmModel('mock-instruct-v1');
    else if (p === 'gemini') setLlmModel('gemini-3.8-flash');
    else if (p === 'openai') setLlmModel('gpt-4o-mini');
    else if (p === 'ollama') setLlmModel('qwen2.5:3b');
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header Banner */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 bg-slate-950/80 p-6 rounded-xl border border-slate-800 backdrop-blur-sm">
        <div>
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-emerald-600/20 text-emerald-400 border border-emerald-500/30 flex items-center justify-center font-bold">
              <CheckCircle className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-slate-100 flex items-center gap-2">
                Phase 13 — RAG Evaluation, Observability &amp; Quality Engineering
                <span className="text-xs font-mono font-medium px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800">
                  Ready &bull; v13.0
                </span>
              </h1>
              <p className="text-xs text-slate-400 mt-0.5">
                Scientific evaluation of multi-modal RAG, routing accuracy, deterministic generation accuracy (EM, Token F1), claim-level grounding, structured traces, and versioned cost accounting.
              </p>
            </div>
          </div>
        </div>

        {/* Global Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => {
              loadInitialData();
              if (selectedDatasetId) loadQuestions(selectedDatasetId);
              if (selectedRunId) loadRunDetails(selectedRunId);
            }}
            disabled={isLoading}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-slate-900 text-slate-300 hover:text-white rounded-lg border border-slate-800 transition"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
          <button
            onClick={() => setActiveTab('launcher')}
            className="flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg shadow-md shadow-emerald-600/20 transition"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            New Benchmark Run
          </button>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex items-center gap-1 border-b border-slate-800 pb-2 overflow-x-auto text-xs font-medium">
        {[
          { id: 'overview', label: 'Overview & KPIs', icon: BarChart2 },
          { id: 'datasets', label: 'Golden Datasets', icon: Database, badge: datasets.length },
          { id: 'launcher', label: 'Experiment Launcher', icon: Play },
          { id: 'results', label: 'Results & Question Inspector', icon: Search, badge: runs.length },
          { id: 'routing', label: 'Routing Quality', icon: GitMerge },
          { id: 'traces', label: 'Traces & Latency', icon: Clock },
          { id: 'comparison', label: 'Run Comparison', icon: Scale },
          { id: 'learning', label: 'Educational Learning Mode', icon: Award },
        ].map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg transition shrink-0 ${
                isActive
                  ? 'bg-blue-600/20 text-blue-400 border border-blue-500/30'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/40'
              }`}
            >
              <Icon className="w-4 h-4" />
              <span>{tab.label}</span>
              {tab.badge !== undefined && (
                <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-slate-800 text-slate-400 font-mono">
                  {tab.badge}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {error && (
        <div className="p-4 rounded-lg bg-rose-950/40 border border-rose-800/60 text-rose-300 text-xs flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{error}</span>
          </div>
          <button onClick={() => setError(null)} className="text-rose-400 hover:text-rose-200">
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 1: OVERVIEW & KPIS */}
      {/* ========================================================================= */}
      {activeTab === 'overview' && (
        <div className="space-y-6">
          {/* Latest Completed Run Scorecard */}
          {selectedRun ? (
            <div>
              <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3 flex items-center justify-between">
                <span>Latest Benchmark Scorecard &bull; {selectedRun.name}</span>
                <span className="text-[10px] font-mono text-slate-500">
                  Type: {selectedRun.run_type} | Strategy: {selectedRun.config?.retrieval_strategy || 'N/A'} | Model: {selectedRun.config?.llm_model}
                </span>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <MetricCard
                  title="F1@5"
                  value={selectedRun.aggregate_metrics?.['mean_f1@5'] ?? '0.00'}
                  subtitle="Harmonic mean of P@5 &amp; R@5"
                  badge={selectedRun.run_type === 'retrieval_only' ? 'IR' : 'Full'}
                />
                <MetricCard
                  title="Recall@5"
                  value={selectedRun.aggregate_metrics?.['mean_recall@5'] ?? '0.00'}
                  subtitle="Evidence retrieval completeness"
                />
                <MetricCard
                  title="MRR"
                  value={selectedRun.aggregate_metrics?.mrr ?? '0.00'}
                  subtitle="Mean Reciprocal Rank"
                />
                <MetricCard
                  title="nDCG@5"
                  value={selectedRun.aggregate_metrics?.['mean_ndcg@5'] ?? '0.00'}
                  subtitle="Graded ranking quality"
                />
                <MetricCard
                  title="Exact Match"
                  value={selectedRun.aggregate_metrics?.mean_exact_match != null ? `${(selectedRun.aggregate_metrics.mean_exact_match * 100).toFixed(0)}%` : 'N/A'}
                  subtitle="Normalized string equality"
                />
                <MetricCard
                  title="Token F1"
                  value={selectedRun.aggregate_metrics?.mean_token_f1 != null ? `${(selectedRun.aggregate_metrics.mean_token_f1 * 100).toFixed(1)}%` : 'N/A'}
                  subtitle="Word unigram overlap"
                />
                <MetricCard
                  title="Answer Faithfulness"
                  value={selectedRun.aggregate_metrics?.mean_faithfulness != null ? `${(selectedRun.aggregate_metrics.mean_faithfulness * 100).toFixed(0)}%` : 'N/A'}
                  subtitle="Context entailment ratio"
                />
                <MetricCard
                  title="Citation Validity"
                  value={selectedRun.aggregate_metrics?.mean_citation_id_validity != null ? `${(selectedRun.aggregate_metrics.mean_citation_id_validity * 100).toFixed(0)}%` : 'N/A'}
                  subtitle="Evidence source grounding"
                />
              </div>

              {/* Routing & System Safety Row */}
              {selectedRun.routing_summary && Object.keys(selectedRun.routing_summary).length > 0 && (
                <div className="mt-4 p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-3">
                  <div className="text-xs font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                    <GitMerge className="w-4 h-4 text-purple-400" />
                    <span>Intelligent Retrieval Routing Telemetry</span>
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
                      <span className="text-[11px] text-slate-400 block">Route Selection Accuracy</span>
                      <span className="text-xl font-bold font-mono text-emerald-400">
                        {((selectedRun.routing_summary.mean_route_accuracy || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
                      <span className="text-[11px] text-slate-400 block">Unnecessary Vision Rate</span>
                      <span className="text-xl font-bold font-mono text-amber-400">
                        {((selectedRun.routing_summary.unnecessary_vision_rate || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
                      <span className="text-[11px] text-slate-400 block">Unnecessary Agentic Rate</span>
                      <span className="text-xl font-bold font-mono text-blue-400">
                        {((selectedRun.routing_summary.unnecessary_agentic_rate || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
                      <span className="text-[11px] text-slate-400 block">Cross-Document Leakage</span>
                      <span className="text-xl font-bold font-mono text-rose-400">
                        {((selectedRun.routing_summary.cross_document_leakage_rate || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Versioned Cost & Pricing Banner */}
              <div className="mt-4 p-4 rounded-xl bg-blue-950/20 border border-blue-900/40 flex items-start justify-between text-xs">
                <div className="flex items-start gap-3">
                  <Coins className="w-5 h-5 text-blue-400 shrink-0 mt-0.5" />
                  <div>
                    <span className="font-semibold text-blue-200">Versioned Cost Accounting &bull; ModelPricingRegistry</span>
                    <p className="text-slate-400 text-[11px] mt-0.5">
                      Configured Pricing: <strong className="text-slate-200">gemini-3.8-flash</strong> ($0.10 input / $0.40 output per 1M tokens), <strong className="text-slate-200">mock-instruct-v1</strong> ($0.00). Unregistered models are flagged as <code className="text-amber-400">unknown</code> without silent zeroing.
                    </p>
                  </div>
                </div>
                <div className="text-right font-mono">
                  <span className="text-slate-400 text-[10px] block">Run Estimated Cost</span>
                  <span className="text-sm font-bold text-emerald-400">
                    ${selectedRun.cost_summary?.estimated_cost_usd?.toFixed(6) ?? '0.000000'}
                  </span>
                </div>
              </div>
            </div>
          ) : (
            <div className="p-12 text-center bg-slate-950/60 rounded-xl border border-slate-800 text-slate-400 space-y-3">
              <FlaskConical className="w-8 h-8 text-slate-500 mx-auto" />
              <p className="text-sm">No completed evaluation runs found.</p>
              <button
                onClick={() => setActiveTab('launcher')}
                className="px-4 py-2 bg-emerald-600 text-white rounded-lg text-xs font-semibold"
              >
                Launch First Benchmark
              </button>
            </div>
          )}
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 2: GOLDEN DATASETS */}
      {/* ========================================================================= */}
      {activeTab === 'datasets' && (
        <div className="space-y-6">
          {/* Dataset Header & Actions */}
          <div className="flex items-center justify-between bg-slate-950/60 p-4 rounded-xl border border-slate-800">
            <div className="flex items-center gap-3">
              <Database className="w-5 h-5 text-blue-400" />
              <div>
                <select
                  value={selectedDatasetId}
                  onChange={(e) => setSelectedDatasetId(e.target.value)}
                  className="bg-slate-900 text-slate-100 text-xs px-3 py-1.5 rounded-lg border border-slate-700 font-semibold"
                >
                  {datasets.map((d) => (
                    <option key={d.id} value={d.id}>
                      {d.name} ({d.question_count || 0} questions &bull; v{d.version || '1.0'})
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={() => handleExportDataset('json')}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-slate-900 hover:bg-slate-800 text-slate-300 rounded-lg border border-slate-800"
              >
                <Download className="w-3.5 h-3.5" />
                Export JSON
              </button>
              <button
                onClick={() => handleExportDataset('jsonl')}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-slate-900 hover:bg-slate-800 text-slate-300 rounded-lg border border-slate-800 font-mono"
              >
                <Download className="w-3.5 h-3.5" />
                JSONL
              </button>
              <button
                onClick={() => setShowImportModal(true)}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-slate-900 hover:bg-slate-800 text-slate-300 rounded-lg border border-slate-800"
              >
                <Upload className="w-3.5 h-3.5" />
                Import
              </button>
              <button
                onClick={() => setShowAddQuestionModal(true)}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-medium shadow-sm"
              >
                <Plus className="w-3.5 h-3.5" />
                Add Question
              </button>
            </div>
          </div>

          {/* Questions Table */}
          <div className="bg-slate-950/80 rounded-xl border border-slate-800 overflow-hidden">
            <div className="p-4 border-b border-slate-800 flex items-center justify-between text-xs">
              <span className="font-semibold text-slate-300 uppercase tracking-wider">
                Benchmark Questions ({questions.length})
              </span>
              <span className="text-slate-500 font-mono">
                {questions.filter((q) => q.is_reviewed).length} human-reviewed
              </span>
            </div>

            <div className="divide-y divide-slate-800/60 max-h-[600px] overflow-y-auto">
              {questions.map((q, idx) => (
                <div key={q.id} className="p-4 hover:bg-slate-900/40 transition space-y-2">
                  <div className="flex items-start justify-between gap-4">
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-900 text-slate-400 border border-slate-800">
                          #{idx + 1}
                        </span>
                        <span className="text-xs font-semibold text-slate-100">
                          {q.question}
                        </span>
                        <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-purple-950 text-purple-300 border border-purple-800">
                          {q.modality || 'text'}
                        </span>
                        <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-blue-950 text-blue-300 border border-blue-800">
                          {q.category}
                        </span>
                      </div>
                      <p className="text-xs text-slate-400 pl-8">
                        <strong className="text-slate-300">Reference:</strong> {q.reference_answer}
                      </p>
                    </div>

                    <div className="flex items-center gap-2 shrink-0">
                      <button
                        onClick={() => handleToggleReview(q)}
                        className={`text-[10px] px-2 py-1 rounded font-medium border flex items-center gap-1 ${
                          q.is_reviewed
                            ? 'bg-emerald-950 text-emerald-400 border-emerald-800'
                            : 'bg-slate-900 text-slate-400 border-slate-800'
                        }`}
                      >
                        <Check className="w-3 h-3" />
                        {q.is_reviewed ? 'Reviewed' : 'Pending'}
                      </button>
                      <button
                        onClick={() => handleOpenAuditor(q)}
                        className="text-[10px] px-2 py-1 rounded bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800 flex items-center gap-1"
                      >
                        <ShieldCheck className="w-3 h-3 text-indigo-400" />
                        Auditor
                      </button>
                      <button
                        onClick={() => handleDeleteQuestion(q.id)}
                        className="p-1 text-slate-500 hover:text-rose-400 transition"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 3: EXPERIMENT LAUNCHER */}
      {/* ========================================================================= */}
      {activeTab === 'launcher' && (
        <div className="bg-slate-950/80 rounded-xl border border-slate-800 p-6 space-y-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div>
              <h2 className="text-base font-bold text-slate-100 flex items-center gap-2">
                <Play className="w-4 h-4 text-emerald-400 fill-current" />
                Configure &amp; Execute Benchmark Experiment
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Dispatch evaluation queries through Unified RAG intelligent routing, pure vector retrieval, or full Phase 06 generation.
              </p>
            </div>
          </div>

          <form onSubmit={handleLaunchRun} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Target Dataset
                </label>
                <select
                  value={selectedDatasetId}
                  onChange={(e) => setSelectedDatasetId(e.target.value)}
                  className="w-full bg-slate-900 text-slate-100 text-xs p-2.5 rounded-lg border border-slate-700"
                >
                  {datasets.map((d) => (
                    <option key={d.id} value={d.id}>
                      {d.name} ({d.question_count} questions)
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Run Name
                </label>
                <input
                  type="text"
                  placeholder="e.g. Unified P12.5 vs Dense Baseline"
                  value={runName}
                  onChange={(e) => setRunName(e.target.value)}
                  className="w-full bg-slate-900 text-slate-100 text-xs p-2.5 rounded-lg border border-slate-700 font-medium"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Execution Pipeline Mode
                </label>
                <select
                  value={runType}
                  onChange={(e) => setRunType(e.target.value)}
                  className="w-full bg-slate-900 text-slate-100 text-xs p-2.5 rounded-lg border border-slate-700 font-semibold text-emerald-400"
                >
                  <option value="unified">Unified RAG (Intelligent Routing P12.5)</option>
                  <option value="end_to_end">Full End-to-End RAG (Phase 06)</option>
                  <option value="retrieval_only">Retrieval Only (No LLM Inference)</option>
                </select>
              </div>
            </div>

            {/* Unified RAG Specific Settings */}
            {runType === 'unified' && (
              <div className="p-4 rounded-xl bg-purple-950/20 border border-purple-900/40 space-y-4">
                <div className="text-xs font-semibold text-purple-300 uppercase tracking-wider flex items-center gap-2">
                  <GitMerge className="w-4 h-4 text-purple-400" />
                  <span>Phase 12.5 Routing &amp; Multimodal Settings</span>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
                  <div>
                    <label className="block text-slate-400 mb-1">Routing Mode</label>
                    <select
                      value={routingMode}
                      onChange={(e) => setRoutingMode(e.target.value)}
                      className="w-full bg-slate-950 text-slate-100 p-2 rounded-lg border border-slate-700 font-mono"
                    >
                      <option value="auto">auto (Intent Classifier + Capability Audit)</option>
                      <option value="manual">manual (Force specific routes)</option>
                      <option value="comparative">comparative (Execute candidates side-by-side)</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-slate-400 mb-1">Top-K Evidence</label>
                    <select
                      value={topK}
                      onChange={(e) => setTopK(parseInt(e.target.value))}
                      className="w-full bg-slate-950 text-slate-100 p-2 rounded-lg border border-slate-700 font-mono"
                    >
                      <option value="1">1</option>
                      <option value="3">3</option>
                      <option value="5">5</option>
                      <option value="10">10</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-slate-400 mb-1">Max Question Limit</label>
                    <input
                      type="number"
                      value={maxQuestionsLimit}
                      onChange={(e) => setMaxQuestionsLimit(e.target.value)}
                      className="w-full bg-slate-950 text-slate-100 p-2 rounded-lg border border-slate-700 font-mono"
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Provider & Model Selection */}
            {runType !== 'retrieval_only' && (
              <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-4">
                <div className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                  LLM Provider &amp; Model Selection
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label className="block text-xs font-medium text-slate-400 mb-1">
                      LLM Provider
                    </label>
                    <select
                      value={llmProvider}
                      onChange={handleProviderChange}
                      className="w-full bg-slate-950 text-slate-100 text-xs p-2 rounded-lg border border-slate-700"
                    >
                      <option value="mock">Mock LLM (Offline / Free $0.00)</option>
                      <option value="gemini">Google Gemini (gemini-3.8-flash)</option>
                      <option value="openai">OpenAI (gpt-4o-mini)</option>
                      <option value="ollama">Ollama (Local compute)</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-slate-400 mb-1">
                      Model ID
                    </label>
                    <input
                      type="text"
                      value={llmModel}
                      onChange={(e) => setLlmModel(e.target.value)}
                      className="w-full bg-slate-950 text-slate-100 text-xs p-2 rounded-lg border border-slate-700 font-mono"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-slate-400 mb-1">
                      Evaluation Judge
                    </label>
                    <select
                      value={judgeProvider}
                      onChange={(e) => setJudgeProvider(e.target.value)}
                      className="w-full bg-slate-950 text-slate-100 text-xs p-2 rounded-lg border border-slate-700"
                    >
                      <option value="heuristic">Rule-Based / Heuristic (Deterministic &amp; Free)</option>
                      <option value="none">None (Skip LLM Judge)</option>
                    </select>
                  </div>
                </div>
              </div>
            )}

            {/* Execution Options */}
            <div className="flex items-center justify-between p-4 bg-slate-900/40 rounded-xl border border-slate-800">
              <label className="flex items-center gap-2 cursor-pointer text-xs text-slate-300">
                <input
                  type="checkbox"
                  checked={runAsync}
                  onChange={(e) => setRunAsync(e.target.checked)}
                  className="rounded bg-slate-950 border-slate-700 text-emerald-600 focus:ring-0"
                />
                <span>Execute in background with live progress tracking</span>
              </label>

              <button
                type="submit"
                disabled={isExecutingRun}
                className="flex items-center gap-2 px-6 py-2.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-xs font-bold rounded-lg shadow-md transition"
              >
                {isExecutingRun ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    Executing Benchmark...
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 fill-current" />
                    Launch Benchmark Run
                  </>
                )}
              </button>
            </div>
          </form>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 4: RESULTS & QUESTION INSPECTOR */}
      {/* ========================================================================= */}
      {activeTab === 'results' && (
        <div className="space-y-6">
          {/* Run Selector & Export Bar */}
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 bg-slate-950/60 p-4 rounded-xl border border-slate-800">
            <div className="flex items-center gap-3">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                Active Run:
              </span>
              <select
                value={selectedRunId}
                onChange={(e) => setSelectedRunId(e.target.value)}
                className="bg-slate-900 text-slate-100 text-xs px-3 py-1.5 rounded-lg border border-slate-700 font-medium"
              >
                {runs.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.name} ({r.status}) &bull; {r.run_type}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={() => handleExportRun('json')}
                disabled={isExporting}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-slate-900 hover:bg-slate-800 text-slate-300 rounded-lg border border-slate-800"
              >
                <Download className="w-3.5 h-3.5" />
                Export JSON
              </button>
              <button
                onClick={() => handleExportRun('csv')}
                disabled={isExporting}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-slate-900 hover:bg-slate-800 text-slate-300 rounded-lg border border-slate-800 font-mono"
              >
                <Download className="w-3.5 h-3.5" />
                Export CSV
              </button>
            </div>
          </div>

          {/* Granular Questions Breakdown & Inspector Layout */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left Column: Questions List (5 cols) */}
            <div className="lg:col-span-5 bg-slate-950/80 rounded-xl border border-slate-800 p-4 space-y-3">
              <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 pb-2 border-b border-slate-800 flex items-center justify-between">
                <span>Evaluated Questions ({runQuestions.length})</span>
                <span className="text-[10px] text-slate-500 font-mono">Select to inspect</span>
              </div>

              <div className="space-y-2 max-h-[600px] overflow-y-auto pr-1">
                {runQuestions.map((qItem, idx) => {
                  const isSelected = selectedQuestionItem?.id === qItem.id;
                  const faith = qItem.faithfulness_score;
                  const em = qItem.accuracy_metrics?.exact_match;
                  return (
                    <div
                      key={qItem.id}
                      onClick={() => setSelectedQuestionItem(qItem)}
                      className={`p-3 rounded-lg border cursor-pointer transition ${
                        isSelected
                          ? 'bg-blue-600/15 border-blue-500/40 text-blue-200'
                          : 'bg-slate-900/60 border-slate-800/80 hover:bg-slate-900 hover:border-slate-700 text-slate-300'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <span className="text-xs font-semibold text-slate-100 line-clamp-1">
                          #{idx + 1} {qItem.question_text}
                        </span>
                        <div className="flex items-center gap-1 shrink-0">
                          {em !== undefined && (
                            <span
                              className={`text-[9px] font-mono px-1 py-0.2 rounded border ${
                                em === 1.0
                                  ? 'bg-emerald-950 text-emerald-400 border-emerald-800'
                                  : 'bg-slate-950 text-slate-400 border-slate-800'
                              }`}
                            >
                              EM: {em}
                            </span>
                          )}
                          {faith != null && (
                            <span
                              className={`text-[9px] font-mono px-1 py-0.2 rounded border ${
                                faith >= 0.8
                                  ? 'bg-emerald-950 text-emerald-400 border-emerald-800'
                                  : 'bg-rose-950 text-rose-400 border-rose-800'
                              }`}
                            >
                              Faith: {faith}
                            </span>
                          )}
                        </div>
                      </div>

                      <div className="flex items-center gap-3 mt-2 text-[10px] text-slate-500 font-mono">
                        <span>P: {qItem.retrieval_metrics?.['precision@5'] ?? '0.0'}</span>
                        <span>R: {qItem.retrieval_metrics?.['recall@5'] ?? '0.0'}</span>
                        <span>F1: {qItem.retrieval_metrics?.['f1@5'] ?? '0.0'}</span>
                        <span>Time: {qItem.latency_breakdown?.total_latency_ms ?? 0} ms</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Right Column: Question Inspector Detail (7 cols) */}
            <div className="lg:col-span-7 bg-slate-950/80 rounded-xl border border-slate-800 p-6 space-y-6">
              {selectedQuestionItem ? (
                <>
                  <div className="flex items-start justify-between">
                    <div>
                      <span className="text-[10px] font-mono uppercase tracking-wider text-blue-400 block mb-1">
                        Target Query Question
                      </span>
                      <h3 className="text-sm font-bold text-slate-100">
                        {selectedQuestionItem.question_text}
                      </h3>
                    </div>
                    {selectedQuestionItem.trace_id && (
                      <button
                        onClick={async () => {
                          const tr = await api.getEvaluationTrace(selectedQuestionItem.trace_id);
                          setSelectedTrace(tr);
                          setShowTraceModal(true);
                        }}
                        className="flex items-center gap-1 px-2.5 py-1 text-[11px] bg-purple-950/60 hover:bg-purple-900/80 text-purple-300 border border-purple-800 rounded-lg font-mono transition"
                      >
                        <Clock className="w-3 h-3" />
                        Trace
                      </button>
                    )}
                  </div>

                  {/* Reference vs Generated Answer */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="p-3.5 rounded-lg bg-slate-900/60 border border-slate-800">
                      <span className="text-[10px] font-mono text-slate-400 block mb-1">
                        Ground Truth Reference Answer
                      </span>
                      <p className="text-xs text-slate-300">
                        {selectedQuestionItem.reference_answer || 'N/A'}
                      </p>
                    </div>

                    <div className="p-3.5 rounded-lg bg-slate-900/60 border border-slate-800">
                      <span className="text-[10px] font-mono text-emerald-400 block mb-1">
                        Generated Answer
                      </span>
                      <p className="text-xs text-slate-200">
                        {selectedQuestionItem.generated_answer || 'N/A (Retrieval only run)'}
                      </p>
                    </div>
                  </div>

                  {/* Accuracy Metrics Row */}
                  {selectedQuestionItem.accuracy_metrics && Object.keys(selectedQuestionItem.accuracy_metrics).length > 0 && (
                    <div className="p-3.5 bg-slate-900/40 rounded-lg border border-slate-800 space-y-2">
                      <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 block">
                        Deterministic Generation Accuracy Metrics
                      </span>
                      <div className="grid grid-cols-3 gap-3 text-xs font-mono">
                        <div>
                          <span className="text-slate-500 block text-[10px]">Exact Match</span>
                          <span className={selectedQuestionItem.accuracy_metrics.exact_match === 1.0 ? 'text-emerald-400' : 'text-slate-400'}>
                            {selectedQuestionItem.accuracy_metrics.exact_match}
                          </span>
                        </div>
                        <div>
                          <span className="text-slate-500 block text-[10px]">Token F1</span>
                          <span className="text-blue-400">
                            {selectedQuestionItem.accuracy_metrics.token_f1}
                          </span>
                        </div>
                        <div>
                          <span className="text-slate-500 block text-[10px]">Normalized Match</span>
                          <span className={selectedQuestionItem.accuracy_metrics.normalized_answer_match === 1.0 ? 'text-emerald-400' : 'text-slate-400'}>
                            {selectedQuestionItem.accuracy_metrics.normalized_answer_match}
                          </span>
                        </div>
                      </div>
                      {selectedQuestionItem.accuracy_metrics.numerical_correctness && (
                        <div className="mt-2 text-xs font-mono p-2 bg-slate-950 rounded border border-slate-800 text-slate-300 flex items-center justify-between">
                          <span>
                            Numerical Match: {selectedQuestionItem.accuracy_metrics.numerical_correctness.matched_value} (Error: {selectedQuestionItem.accuracy_metrics.numerical_correctness.error})
                          </span>
                          <span className={selectedQuestionItem.accuracy_metrics.numerical_correctness.is_correct ? 'text-emerald-400' : 'text-rose-400'}>
                            {selectedQuestionItem.accuracy_metrics.numerical_correctness.is_correct ? 'CORRECT' : 'INCORRECT'}
                          </span>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Claim Judgments */}
                  {selectedQuestionItem.claim_judgments && selectedQuestionItem.claim_judgments.length > 0 && (
                    <div className="space-y-2">
                      <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 block">
                        Claim-Level Grounding Judgments ({selectedQuestionItem.claim_judgments.length})
                      </span>
                      <div className="space-y-2 max-h-[200px] overflow-y-auto">
                        {selectedQuestionItem.claim_judgments.map((cj, idx) => (
                          <div
                            key={idx}
                            className={`p-2.5 rounded-lg border text-xs ${
                              cj.status === 'supported'
                                ? 'bg-emerald-950/30 border-emerald-800/60 text-slate-200'
                                : 'bg-rose-950/30 border-rose-800/60 text-slate-200'
                            }`}
                          >
                            <div className="flex items-center justify-between">
                              <span className="font-medium text-slate-100">{cj.claim}</span>
                              <span className="text-[10px] font-mono uppercase font-bold text-current">
                                {cj.status}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="p-12 text-center text-xs text-slate-500">
                  Select a question from the left column to inspect its evaluation breakdown.
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 5: ROUTING QUALITY */}
      {/* ========================================================================= */}
      {activeTab === 'routing' && (
        <div className="space-y-6">
          <div className="p-4 bg-slate-950/80 rounded-xl border border-slate-800 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <GitMerge className="w-5 h-5 text-purple-400" />
              <div>
                <h3 className="text-sm font-bold text-slate-100">Intelligent Retrieval Routing Telemetry</h3>
                <p className="text-xs text-slate-400">
                  Verification of route selection accuracy, unnecessary vision/agentic overhead, and document isolation.
                </p>
              </div>
            </div>
          </div>

          <div className="bg-slate-950/80 rounded-xl border border-slate-800 overflow-hidden">
            <div className="p-4 border-b border-slate-800 text-xs font-semibold uppercase tracking-wider text-slate-300">
              Routing Decisions Across Evaluated Questions
            </div>
            <div className="divide-y divide-slate-800 max-h-[600px] overflow-y-auto text-xs">
              {runQuestions.map((qItem, idx) => {
                const rm = qItem.routing_metrics || {};
                const isAccurate = rm.route_accuracy === 1.0;
                return (
                  <div key={qItem.id} className="p-4 hover:bg-slate-900/40 transition space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-slate-900 text-slate-400">
                          #{idx + 1}
                        </span>
                        <span className="font-semibold text-slate-100">{qItem.question_text}</span>
                      </div>
                      <span className={`text-[10px] font-mono px-2 py-0.5 rounded font-bold ${
                        isAccurate ? 'bg-emerald-950 text-emerald-400 border border-emerald-800' : 'bg-rose-950 text-rose-400 border border-rose-800'
                      }`}>
                        {isAccurate ? 'ROUTE HIT' : 'MISROUTED'}
                      </span>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-[11px] font-mono pl-6 text-slate-400">
                      <div>
                        <span className="text-slate-500 block text-[10px]">Selected Route</span>
                        <span className="text-purple-300 font-bold">{rm.selected_routes?.join(', ') || 'N/A'}</span>
                      </div>
                      <div>
                        <span className="text-slate-500 block text-[10px]">Expected Route</span>
                        <span className="text-slate-200">{rm.expected_routes?.join(', ') || 'N/A'}</span>
                      </div>
                      <div>
                        <span className="text-slate-500 block text-[10px]">Unnecessary Vision</span>
                        <span className={rm.unnecessary_vision ? 'text-amber-400 font-bold' : 'text-slate-400'}>
                          {rm.unnecessary_vision ? 'YES (OVERHEAD)' : 'No'}
                        </span>
                      </div>
                      <div>
                        <span className="text-slate-500 block text-[10px]">Doc Isolation</span>
                        <span className={rm.document_isolation ? 'text-emerald-400' : 'text-rose-400 font-bold'}>
                          {rm.document_isolation ? 'Verified Isolated' : 'Leakage Detected'}
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 6: TRACES & LATENCY WATERFALL */}
      {/* ========================================================================= */}
      {activeTab === 'traces' && (
        <div className="space-y-6">
          <div className="p-4 bg-slate-950/80 rounded-xl border border-slate-800 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Clock className="w-5 h-5 text-purple-400" />
              <div>
                <h3 className="text-sm font-bold text-slate-100">Structured Traces &amp; Latency Waterfall</h3>
                <p className="text-xs text-slate-400">
                  Per-stage timing breakdown with secrets and base64 payloads strictly redacted.
                </p>
              </div>
            </div>
            {selectedRun && (
              <div className="flex items-center gap-4 text-xs font-mono">
                <div>
                  <span className="text-slate-500 text-[10px] block">p50 Latency</span>
                  <span className="text-slate-200 font-bold">{selectedRun.latency_summary?.p50_ms || 0} ms</span>
                </div>
                <div>
                  <span className="text-slate-500 text-[10px] block">p95 Latency</span>
                  <span className="text-amber-400 font-bold">{selectedRun.latency_summary?.p95_ms || 0} ms</span>
                </div>
                <div>
                  <span className="text-slate-500 text-[10px] block">p99 Latency</span>
                  <span className="text-rose-400 font-bold">{selectedRun.latency_summary?.p99_ms || 0} ms</span>
                </div>
              </div>
            )}
          </div>

          <div className="bg-slate-950/80 rounded-xl border border-slate-800 p-6 space-y-4">
            <div className="text-xs font-semibold uppercase tracking-wider text-slate-300">
              Execution Stage Waterfall ({runTraces.length} recorded traces)
            </div>

            <div className="space-y-3 max-h-[600px] overflow-y-auto">
              {runTraces.map((tr, idx) => {
                const totalMs = tr.total_latency_ms || 1;
                return (
                  <div
                    key={tr.id}
                    onClick={() => {
                      setSelectedTrace(tr);
                      setShowTraceModal(true);
                    }}
                    className="p-3.5 bg-slate-900/60 hover:bg-slate-900 rounded-lg border border-slate-800 cursor-pointer transition space-y-2"
                  >
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-semibold text-slate-200">
                        Trace #{idx + 1} &bull; {tr.trace_id.slice(0, 8)}...
                      </span>
                      <span className="font-mono text-purple-400 font-bold">{totalMs} ms</span>
                    </div>

                    {/* SVG Waterfall Bar */}
                    <div className="w-full bg-slate-950 h-5 rounded overflow-hidden flex items-center border border-slate-800/80">
                      {tr.stages?.map((stg, sIdx) => {
                        const widthPct = Math.max(3, Math.min(100, (stg.duration_ms / totalMs) * 100));
                        const colors = [
                          'bg-blue-500',
                          'bg-purple-500',
                          'bg-emerald-500',
                          'bg-amber-500',
                          'bg-cyan-500',
                          'bg-indigo-500',
                        ];
                        const col = colors[sIdx % colors.length];
                        return (
                          <div
                            key={sIdx}
                            style={{ width: `${widthPct}%` }}
                            title={`${stg.stage_name}: ${stg.duration_ms} ms`}
                            className={`h-full ${col} opacity-85 hover:opacity-100 transition`}
                          />
                        );
                      })}
                    </div>

                    <div className="flex flex-wrap items-center gap-3 text-[10px] font-mono text-slate-500">
                      {tr.stages?.map((stg, sIdx) => (
                        <span key={sIdx}>
                          {stg.stage_name}: <strong className="text-slate-300">{stg.duration_ms} ms</strong>
                        </span>
                      ))}
                    </div>
                  </div>
                );
              })}

              {runTraces.length === 0 && (
                <div className="p-8 text-center text-xs text-slate-500">
                  No execution traces available for this run.
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 7: RUN COMPARISON */}
      {/* ========================================================================= */}
      {activeTab === 'comparison' && (
        <div className="space-y-6">
          <div className="bg-slate-950/80 rounded-xl border border-slate-800 p-6 space-y-4">
            <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
              <Scale className="w-4 h-4 text-blue-400" />
              Side-by-Side Run Delta Comparison
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Baseline Run (A)</label>
                <select
                  value={compareRunAId}
                  onChange={(e) => setCompareRunAId(e.target.value)}
                  className="w-full bg-slate-900 text-slate-100 text-xs p-2.5 rounded-lg border border-slate-700"
                >
                  <option value="">Select Baseline Run</option>
                  {runs.map((r) => (
                    <option key={r.id} value={r.id}>
                      {r.name} ({r.run_type})
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Candidate Run (B)</label>
                <select
                  value={compareRunBId}
                  onChange={(e) => setCompareRunBId(e.target.value)}
                  className="w-full bg-slate-900 text-slate-100 text-xs p-2.5 rounded-lg border border-slate-700"
                >
                  <option value="">Select Candidate Run</option>
                  {runs.map((r) => (
                    <option key={r.id} value={r.id}>
                      {r.name} ({r.run_type})
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <button
              onClick={handleCompareRuns}
              disabled={isLoading || !compareRunAId || !compareRunBId}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-xs font-semibold rounded-lg"
            >
              Compare Runs
            </button>
          </div>

          {comparisonResult && (
            <div className="bg-slate-950/80 rounded-xl border border-slate-800 p-6 space-y-4">
              <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                Metric Deltas (Candidate B vs Baseline A)
              </h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {Object.entries(comparisonResult.metric_deltas || {}).map(([key, val]) => (
                  <div key={key} className="p-3 bg-slate-900 rounded-lg border border-slate-800">
                    <span className="text-[10px] text-slate-400 uppercase font-mono block">{key}</span>
                    <span className={`text-lg font-bold font-mono ${val >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {val >= 0 ? `+${val}` : val}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 8: EDUCATIONAL LEARNING MODE */}
      {/* ========================================================================= */}
      {activeTab === 'learning' && (
        <div className="space-y-6">
          <div className="bg-slate-950/80 rounded-xl border border-slate-800 p-6 space-y-4">
            <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
              <Award className="w-4 h-4 text-amber-400" />
              Information Retrieval &amp; RAG Evaluation Formulas
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
              <div className="p-4 bg-slate-900 rounded-lg border border-slate-800 space-y-2">
                <span className="font-semibold text-blue-300 block">F1@K Harmonic Mean</span>
                <code className="text-slate-300 block font-mono text-[11px] p-2 bg-slate-950 rounded">
                  F1@K = 2 · (Precision@K · Recall@K) / (Precision@K + Recall@K)
                </code>
                <p className="text-slate-400 text-[11px]">
                  Balances precision (minimizing distractors) and recall (capturing all documentary evidence).
                </p>
              </div>

              <div className="p-4 bg-slate-900 rounded-lg border border-slate-800 space-y-2">
                <span className="font-semibold text-purple-300 block">nDCG@K with Graded Relevance</span>
                <code className="text-slate-300 block font-mono text-[11px] p-2 bg-slate-950 rounded">
                  nDCG@K = DCG@K / IDCG@K, where gain = 2^rel - 1
                </code>
                <p className="text-slate-400 text-[11px]">
                  Measures ranking quality with penalties for relevant documents appearing lower in the candidate list.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Trace Detail Modal */}
      {showTraceModal && selectedTrace && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-950 border border-slate-800 rounded-xl max-w-2xl w-full max-h-[85vh] overflow-hidden flex flex-col shadow-2xl">
            <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/60">
              <div className="flex items-center gap-2">
                <Clock className="w-4 h-4 text-purple-400" />
                <span className="text-xs font-bold text-slate-100">
                  Execution Trace Detail &bull; {selectedTrace.trace_id}
                </span>
              </div>
              <button
                onClick={() => setShowTraceModal(false)}
                className="text-slate-400 hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="p-6 overflow-y-auto space-y-4 text-xs font-mono">
              <div className="flex items-center gap-4 text-slate-400">
                <span>Provider: <strong className="text-slate-200">{selectedTrace.provider}</strong></span>
                <span>Model: <strong className="text-slate-200">{selectedTrace.model}</strong></span>
                <span>Total Latency: <strong className="text-emerald-400">{selectedTrace.total_latency_ms} ms</strong></span>
              </div>

              <div className="space-y-3">
                <span className="text-slate-400 font-sans font-semibold text-xs uppercase tracking-wider block">
                  Stages &amp; Sanitized Metadata
                </span>
                {selectedTrace.stages?.map((stg, idx) => (
                  <div key={idx} className="p-3 bg-slate-900 rounded-lg border border-slate-800 space-y-1">
                    <div className="flex items-center justify-between text-slate-200">
                      <span className="font-bold text-purple-300">{stg.stage_name}</span>
                      <span>{stg.duration_ms} ms</span>
                    </div>
                    {stg.metadata && Object.keys(stg.metadata).length > 0 && (
                      <pre className="text-[10px] text-slate-400 bg-slate-950 p-2 rounded overflow-x-auto">
                        {JSON.stringify(stg.metadata, null, 2)}
                      </pre>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* New Dataset Modal */}
      {showNewDatasetModal && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-950 border border-slate-800 rounded-xl max-w-md w-full p-6 space-y-4">
            <h3 className="text-sm font-bold text-slate-100">Create Golden Dataset</h3>
            <form onSubmit={handleCreateDataset} className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">Name</label>
                <input
                  type="text"
                  required
                  value={newDatasetName}
                  onChange={(e) => setNewDatasetName(e.target.value)}
                  className="w-full bg-slate-900 text-slate-100 text-xs p-2 rounded-lg border border-slate-700"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">Description</label>
                <textarea
                  value={newDatasetDesc}
                  onChange={(e) => setNewDatasetDesc(e.target.value)}
                  className="w-full bg-slate-900 text-slate-100 text-xs p-2 rounded-lg border border-slate-700"
                />
              </div>
              <div className="flex items-center justify-end gap-2">
                <button
                  type="button"
                  onClick={() => setShowNewDatasetModal(false)}
                  className="px-3 py-1.5 text-xs text-slate-400"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-1.5 bg-blue-600 text-white text-xs font-semibold rounded-lg"
                >
                  Create
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Import Modal */}
      {showImportModal && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-950 border border-slate-800 rounded-xl max-w-xl w-full p-6 space-y-4">
            <h3 className="text-sm font-bold text-slate-100">Import Benchmark Questions (JSON / JSONL)</h3>
            <textarea
              rows={10}
              placeholder='Paste JSON or JSONL lines here...'
              value={importJsonText}
              onChange={(e) => setImportJsonText(e.target.value)}
              className="w-full bg-slate-900 text-slate-100 text-xs p-3 rounded-lg border border-slate-700 font-mono"
            />
            <div className="flex items-center justify-end gap-2">
              <button
                type="button"
                onClick={() => setShowImportModal(false)}
                className="px-3 py-1.5 text-xs text-slate-400"
              >
                Cancel
              </button>
              <button
                onClick={handleImportJson}
                className="px-4 py-1.5 bg-blue-600 text-white text-xs font-semibold rounded-lg"
              >
                Import Questions
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function MetricCard({ title, value, subtitle, trend, unit, badge }) {
  return (
    <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 flex flex-col justify-between">
      <div>
        <div className="flex items-center justify-between mb-1">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">
            {title}
          </span>
          {badge && (
            <span className="text-[9px] font-mono px-1 py-0.2 rounded bg-slate-900 text-slate-400 border border-slate-800">
              {badge}
            </span>
          )}
        </div>
        <div className="text-2xl font-bold font-mono text-slate-100 mt-1 flex items-baseline gap-1">
          <span>{value}</span>
          {unit && <span className="text-xs text-slate-400 font-normal">{unit}</span>}
        </div>
      </div>
      <p className="text-[10px] text-slate-500 mt-2">{subtitle}</p>
    </div>
  );
}
