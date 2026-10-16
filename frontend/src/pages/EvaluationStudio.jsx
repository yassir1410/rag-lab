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
  const [activeTab, setActiveTab] = useState('overview'); // 'overview' | 'datasets' | 'launcher' | 'results' | 'comparison' | 'learning'
  const [datasets, setDatasets] = useState([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState('');
  const [runs, setRuns] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState('');
  const [selectedRun, setSelectedRun] = useState(null);
  const [runQuestions, setRunQuestions] = useState([]);
  const [selectedQuestionItem, setSelectedQuestionItem] = useState(null);
  const [metricsCatalog, setMetricsCatalog] = useState([]);

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
  const [newIsAnswerable, setNewIsAnswerable] = useState(true);

  // Import / Export State
  const [showImportModal, setShowImportModal] = useState(false);
  const [importJsonText, setImportJsonText] = useState('');

  // Experiment Launcher Form State
  const [runName, setRunName] = useState('');
  const [runType, setRunType] = useState('end_to_end');
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
  const [maxQuestionsLimit, setMaxQuestionsLimit] = useState(10);
  const [confirmPaidRun, setConfirmPaidRun] = useState(false);

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

  // When selectedRunId changes, load run details
  useEffect(() => {
    if (selectedRunId) {
      loadRunDetails(selectedRunId);
    }
  }, [selectedRunId]);

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
        setSelectedDatasetId(dsList[0].id);
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
      const [rDetail, rItems] = await Promise.all([
        api.getEvaluationRun(runId),
        api.getEvaluationRunQuestions(runId),
      ]);
      setSelectedRun(rDetail);
      setRunQuestions(rItems || []);
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

  const handleExportJson = async () => {
    if (!selectedDatasetId) return;
    try {
      const payload = await api.exportEvaluationDataset(selectedDatasetId);
      const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(payload, null, 2));
      const downloadAnchor = document.createElement('a');
      downloadAnchor.setAttribute('href', dataStr);
      downloadAnchor.setAttribute('download', `dataset_${selectedDatasetId}.json`);
      document.body.appendChild(downloadAnchor);
      downloadAnchor.click();
      downloadAnchor.remove();
    } catch (err) {
      alert(err.message || 'Error exporting dataset');
    }
  };

  const handleImportJson = async () => {
    if (!selectedDatasetId || !importJsonText.trim()) return;
    try {
      const parsed = JSON.parse(importJsonText);
      const res = await api.importEvaluationQuestions(selectedDatasetId, parsed);
      alert(`Imported ${res.imported_questions} questions (Skipped ${res.skipped_questions} duplicates).`);
      setShowImportModal(false);
      setImportJsonText('');
      loadQuestions(selectedDatasetId);
    } catch (err) {
      alert(err.message || 'Invalid JSON format or import error');
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
      const defaultName = `${retrievalStrategy.toUpperCase()} ${runType === 'retrieval_only' ? 'Retrieval' : 'RAG'} Run - ${new Date().toLocaleTimeString()}`;
      const payload = {
        dataset_id: selectedDatasetId,
        name: runName.trim() || defaultName,
        run_type: runType,
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

      const run = await api.createEvaluationRun(payload);
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

  // Helper for provider model sync
  const handleProviderChange = (e) => {
    const p = e.target.value;
    setLlmProvider(p);
    if (p === 'mock') setLlmModel('mock-instruct-v1');
    else if (p === 'gemini') setLlmModel('gemini-3.8-flash');
    else if (p === 'openai') setLlmModel('gpt-4o-mini');
    else if (p === 'ollama') setLlmModel('qwen2.5:3b');
  };

  // Sandbox 1: Precision@K vs Recall@K calculations
  const sampleRetrieved = [
    { rank: 1, title: 'Profile Summary', isRelevant: true },
    { rank: 2, title: 'Project Details: Microservices', isRelevant: true },
    { rank: 3, title: 'Hobbies & Interests', isRelevant: false },
    { rank: 4, title: 'Frameworks: FastAPI & Docker', isRelevant: true },
    { rank: 5, title: 'Generic References', isRelevant: false },
    { rank: 6, title: 'High School Education', isRelevant: false },
    { rank: 7, title: 'Architecture Schema', isRelevant: true },
  ];
  const totalRelevantInCorpus = 4; // ranks 1, 2, 4, 7
  const currentRetrievedTopK = sampleRetrieved.slice(0, sandboxK);
  const relevantInTopK = currentRetrievedTopK.filter((item) => item.isRelevant).length;
  const sandboxPrecision = (relevantInTopK / sandboxK).toFixed(2);
  const sandboxRecall = (relevantInTopK / totalRelevantInCorpus).toFixed(2);

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
                Phase 07 — RAG Evaluation & Quality Benchmarking
                <span className="text-xs font-mono font-medium px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800">
                  Ready &bull; v0.7.0
                </span>
              </h1>
              <p className="text-xs text-slate-400 mt-0.5">
                Scientific evaluation of vector retrieval, claim-level faithfulness, citation entailment, latency percentiles & token budgets.
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
          {/* Top Metric Cards (Latest Completed Run) */}
          {selectedRun && selectedRun.status === 'completed' ? (
            <div>
              <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3 flex items-center justify-between">
                <span>Latest Benchmark Scorecard &bull; {selectedRun.name}</span>
                <span className="text-[10px] font-mono text-slate-500">
                  Algorithm: {selectedRun.config?.search_algorithm} | Top-K: {selectedRun.config?.top_k} | Model: {selectedRun.config?.llm_model}
                </span>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <MetricCard
                  title="Recall@5"
                  value={selectedRun.aggregate_metrics?.['mean_recall@5'] ?? selectedRun.aggregate_metrics?.['mean_recall@3'] ?? '0.00'}
                  subtitle="Evidence retrieval completeness"
                  trend="positive"
                  unit=""
                />
                <MetricCard
                  title="Mean Reciprocal Rank"
                  value={selectedRun.aggregate_metrics?.mrr ?? '0.00'}
                  subtitle="Rank of first relevant chunk"
                  trend="positive"
                  unit=""
                />
                <MetricCard
                  title="Answer Faithfulness"
                  value={selectedRun.aggregate_metrics?.mean_faithfulness ?? 'N/A'}
                  subtitle="Claim-level context entailment"
                  trend="positive"
                  unit=""
                />
                <MetricCard
                  title="Citation Support Precision"
                  value={selectedRun.aggregate_metrics?.mean_citation_support_precision ?? 'N/A'}
                  subtitle="Citations that truly back claims"
                  trend="positive"
                  unit=""
                />
                <MetricCard
                  title="Answer Relevance"
                  value={selectedRun.aggregate_metrics?.mean_answer_relevance ?? 'N/A'}
                  subtitle="Question answering accuracy"
                  trend="positive"
                  unit=""
                />
                <MetricCard
                  title="nDCG@5"
                  value={selectedRun.aggregate_metrics?.['mean_ndcg@5'] ?? selectedRun.aggregate_metrics?.['mean_ndcg@3'] ?? '0.00'}
                  subtitle="Normalized discounted cumulative gain"
                  trend="positive"
                  unit=""
                />
                <MetricCard
                  title="Correct Refusal Rate"
                  value={selectedRun.aggregate_metrics?.mean_correct_refusal_rate != null ? `${(selectedRun.aggregate_metrics.mean_correct_refusal_rate * 100).toFixed(0)}%` : 'N/A'}
                  subtitle="Unanswerable questions refused"
                  trend="positive"
                  unit=""
                />
                <MetricCard
                  title="False Answer Rate"
                  value={selectedRun.aggregate_metrics?.mean_false_answer_rate != null ? `${(selectedRun.aggregate_metrics.mean_false_answer_rate * 100).toFixed(0)}%` : '0%'}
                  subtitle="Hallucinated unanswerables"
                  trend="negative"
                  unit=""
                />
                <MetricCard
                  title="p95 Latency"
                  value={selectedRun.latency_summary?.p95_ms ?? '0.0'}
                  subtitle="95th percentile pipeline time"
                  trend="neutral"
                  unit="ms"
                />
                <MetricCard
                  title="Estimated Cost"
                  value={`$${(selectedRun.cost_summary?.estimated_cost_usd ?? 0).toFixed(4)}`}
                  subtitle="Estimated API expenditure"
                  trend="neutral"
                  unit=""
                />
                <MetricCard
                  title="Evaluation Success"
                  value={`${selectedRun.aggregate_metrics?.successful_questions ?? 0}/${selectedRun.aggregate_metrics?.evaluated_questions ?? 0}`}
                  subtitle="Test questions verified"
                  trend="positive"
                  unit=""
                />
              </div>
            </div>
          ) : (
            <div className="p-8 rounded-xl bg-slate-950/60 border border-slate-800 text-center">
              <FlaskConical className="w-10 h-10 text-slate-600 mx-auto mb-3" />
              <h3 className="text-sm font-semibold text-slate-300">No Benchmark Execution Selected</h3>
              <p className="text-xs text-slate-500 max-w-md mx-auto mt-1 mb-4">
                Execute an evaluation run or select an existing experiment from the list below to inspect retrieval and generation quality scores.
              </p>
              <button
                onClick={() => setActiveTab('launcher')}
                className="px-4 py-2 text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg transition"
              >
                Launch First Evaluation Run
              </button>
            </div>
          )}

          {/* Recent Runs Table */}
          <div className="bg-slate-950/80 rounded-xl border border-slate-800 overflow-hidden">
            <div className="p-4 border-b border-slate-800 flex items-center justify-between">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                Evaluation History & Runs ({runs.length})
              </h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-slate-900/60 text-slate-400 font-mono text-[11px] uppercase border-b border-slate-800">
                  <tr>
                    <th className="py-2.5 px-4">Run Name</th>
                    <th className="py-2.5 px-4">Dataset</th>
                    <th className="py-2.5 px-4">Type</th>
                    <th className="py-2.5 px-4">Algorithm & Top-K</th>
                    <th className="py-2.5 px-4">Model</th>
                    <th className="py-2.5 px-4">Status</th>
                    <th className="py-2.5 px-4">P@K / R@K</th>
                    <th className="py-2.5 px-4">Faithfulness</th>
                    <th className="py-2.5 px-4">p50 Latency</th>
                    <th className="py-2.5 px-4 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {runs.map((r) => {
                    const isSelected = selectedRunId === r.id;
                    return (
                      <tr
                        key={r.id}
                        onClick={() => {
                          setSelectedRunId(r.id);
                        }}
                        className={`cursor-pointer transition hover:bg-slate-900/50 ${
                          isSelected ? 'bg-blue-600/10' : ''
                        }`}
                      >
                        <td className="py-3 px-4 font-semibold text-slate-100 flex items-center gap-2">
                          {r.name}
                        </td>
                        <td className="py-3 px-4 text-slate-400">{r.dataset_name || 'N/A'}</td>
                        <td className="py-3 px-4">
                          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-900 border border-slate-800">
                            {r.run_type}
                          </span>
                        </td>
                        <td className="py-3 px-4 font-mono text-slate-400">
                          {r.config?.search_algorithm} (k={r.config?.top_k})
                        </td>
                        <td className="py-3 px-4 font-mono text-slate-400">
                          {r.config?.llm_model || 'Retrieval'}
                        </td>
                        <td className="py-3 px-4">
                          <span
                            className={`text-[10px] font-mono px-2 py-0.5 rounded border ${
                              r.status === 'completed'
                                ? 'bg-emerald-950 text-emerald-400 border-emerald-800'
                                : r.status === 'running'
                                ? 'bg-blue-950 text-blue-400 border-blue-800 animate-pulse'
                                : 'bg-rose-950 text-rose-400 border-rose-800'
                            }`}
                          >
                            {r.status}
                          </span>
                        </td>
                        <td className="py-3 px-4 font-mono">
                          {r.aggregate_metrics?.['mean_recall@5'] ?? r.aggregate_metrics?.['mean_recall@3'] ?? '—'}
                        </td>
                        <td className="py-3 px-4 font-mono text-emerald-400">
                          {r.aggregate_metrics?.mean_faithfulness ?? '—'}
                        </td>
                        <td className="py-3 px-4 font-mono text-slate-400">
                          {r.latency_summary?.p50_ms ? `${r.latency_summary.p50_ms} ms` : '—'}
                        </td>
                        <td className="py-3 px-4 text-right">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setSelectedRunId(r.id);
                              setActiveTab('results');
                            }}
                            className="px-2 py-1 text-[11px] rounded bg-slate-900 hover:bg-slate-800 text-slate-200 border border-slate-800"
                          >
                            Inspect
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                  {runs.length === 0 && (
                    <tr>
                      <td colSpan={10} className="py-8 text-center text-slate-500">
                        No evaluation runs recorded yet.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 2: GOLDEN DATASETS MANAGER */}
      {/* ========================================================================= */}
      {activeTab === 'datasets' && (
        <div className="space-y-6">
          {/* Dataset Selector Bar */}
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 bg-slate-950/70 p-4 rounded-xl border border-slate-800">
            <div className="flex items-center gap-3">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                Select Dataset:
              </span>
              <select
                value={selectedDatasetId}
                onChange={(e) => setSelectedDatasetId(e.target.value)}
                className="bg-slate-900 text-slate-100 text-xs px-3 py-1.5 rounded-lg border border-slate-700 focus:outline-none focus:border-blue-500 font-medium"
              >
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name} ({d.question_count} questions)
                  </option>
                ))}
              </select>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowNewDatasetModal(true)}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-slate-900 hover:bg-slate-800 text-slate-200 rounded-lg border border-slate-700"
              >
                <Plus className="w-3.5 h-3.5" />
                New Dataset
              </button>
              <button
                onClick={() => setShowAddQuestionModal(true)}
                disabled={!selectedDatasetId}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-blue-600 hover:bg-blue-500 text-white rounded-lg"
              >
                <Plus className="w-3.5 h-3.5" />
                Add Question
              </button>
              <button
                onClick={handleExportJson}
                disabled={!selectedDatasetId || questions.length === 0}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-slate-900 hover:bg-slate-800 text-slate-300 rounded-lg border border-slate-800"
              >
                <Download className="w-3.5 h-3.5" />
                Export JSON
              </button>
              <button
                onClick={() => setShowImportModal(true)}
                disabled={!selectedDatasetId}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-slate-900 hover:bg-slate-800 text-slate-300 rounded-lg border border-slate-800"
              >
                <Upload className="w-3.5 h-3.5" />
                Import JSON
              </button>
            </div>
          </div>

          {/* Questions Table */}
          <div className="bg-slate-950/80 rounded-xl border border-slate-800 overflow-hidden">
            <div className="p-4 border-b border-slate-800 flex items-center justify-between">
              <div>
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                  Golden Evaluation Questions ({questions.length})
                </h3>
                <p className="text-[11px] text-slate-500 mt-0.5">
                  Ground truth evidence, reference answers and human verification status.
                </p>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-slate-900/60 text-slate-400 font-mono text-[11px] uppercase border-b border-slate-800">
                  <tr>
                    <th className="py-2.5 px-4">Question</th>
                    <th className="py-2.5 px-4">Reference Answer</th>
                    <th className="py-2.5 px-4">Category</th>
                    <th className="py-2.5 px-4">Difficulty</th>
                    <th className="py-2.5 px-4">Answerable</th>
                    <th className="py-2.5 px-4">Review Status</th>
                    <th className="py-2.5 px-4 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {questions.map((q) => (
                    <tr key={q.id} className="hover:bg-slate-900/40 transition">
                      <td className="py-3 px-4 font-semibold text-slate-100 max-w-xs truncate">
                        {q.question}
                      </td>
                      <td className="py-3 px-4 text-slate-400 max-w-xs truncate">
                        {q.reference_answer}
                      </td>
                      <td className="py-3 px-4">
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-blue-950/60 text-blue-400 border border-blue-900/50">
                          {q.category}
                        </span>
                      </td>
                      <td className="py-3 px-4 font-mono text-slate-400 capitalize">
                        {q.difficulty}
                      </td>
                      <td className="py-3 px-4">
                        {q.is_answerable ? (
                          <span className="text-[10px] text-emerald-400">Yes</span>
                        ) : (
                          <span className="text-[10px] text-amber-400 font-semibold">Refusal Required</span>
                        )}
                      </td>
                      <td className="py-3 px-4">
                        <div className="flex flex-col gap-1.5">
                          <button
                            onClick={() => handleToggleReview(q)}
                            className={`text-[10px] font-mono px-2 py-0.5 rounded border transition flex items-center gap-1 ${
                              q.is_reviewed
                                ? 'bg-emerald-950 text-emerald-400 border-emerald-800 hover:bg-emerald-900/50'
                                : 'bg-amber-950/60 text-amber-400 border-amber-800/80 hover:bg-amber-900/50'
                            }`}
                          >
                            <ShieldCheck className="w-3 h-3" />
                            {q.is_reviewed ? 'Approved' : 'Unreviewed'}
                          </button>
                          <span className="text-[9px] font-mono text-slate-500">
                            Rev: v{q.revision || 1} &bull; {Object.keys(q.relevance_labels || {}).length} graded
                          </span>
                        </div>
                      </td>
                      <td className="py-3 px-4 text-right">
                        <div className="flex items-center justify-end gap-1.5">
                          <button
                            onClick={() => handleOpenAuditor(q)}
                            className="px-2.5 py-1 bg-indigo-950 hover:bg-indigo-900/60 text-indigo-300 border border-indigo-800 rounded text-[10px] flex items-center gap-1 font-mono transition"
                            title="Audit candidate chunks and assign 0/1/2 graded relevance"
                          >
                            <Edit3 className="w-3 h-3 text-indigo-400" />
                            Audit Evidence
                          </button>
                          <button
                            onClick={() => handleDeleteQuestion(q.id)}
                            className="p-1 rounded text-slate-500 hover:text-rose-400 hover:bg-rose-950/40 transition"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                  {questions.length === 0 && (
                    <tr>
                      <td colSpan={7} className="py-8 text-center text-slate-500">
                        No evaluation questions in this dataset. Click "Add Question" or "Import JSON".
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 3: EXPERIMENT LAUNCHER */}
      {/* ========================================================================= */}
      {activeTab === 'launcher' && (
        <div className="max-w-3xl mx-auto space-y-6">
          <div className="bg-slate-950/80 p-6 rounded-xl border border-slate-800">
            <h2 className="text-base font-bold text-slate-100 mb-1 flex items-center gap-2">
              <Play className="w-4 h-4 text-emerald-400" />
              Configure RAG Benchmark Run
            </h2>
            <p className="text-xs text-slate-400 mb-6">
              Launch an evaluation against your selected golden dataset to calculate mathematical retrieval scores, claim-level faithfulness, and latency percentiles.
            </p>

            <form onSubmit={handleLaunchRun} className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">
                    Evaluation Dataset
                  </label>
                  <select
                    value={selectedDatasetId}
                    onChange={(e) => setSelectedDatasetId(e.target.value)}
                    className="w-full bg-slate-900 text-slate-100 text-xs p-2.5 rounded-lg border border-slate-700"
                    required
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
                    Run Name (Optional)
                  </label>
                  <input
                    type="text"
                    placeholder="e.g. Exact KNN vs HNSW Benchmark"
                    value={runName}
                    onChange={(e) => setRunName(e.target.value)}
                    className="w-full bg-slate-900 text-slate-100 text-xs p-2.5 rounded-lg border border-slate-700"
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">
                    Evaluation Scope
                  </label>
                  <select
                    value={runType}
                    onChange={(e) => setRunType(e.target.value)}
                    className="w-full bg-slate-900 text-slate-100 text-xs p-2.5 rounded-lg border border-slate-700"
                  >
                    <option value="end_to_end">Full End-to-End RAG</option>
                    <option value="retrieval_only">Retrieval Only (No LLM)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">
                    Retrieval Strategy (P08)
                  </label>
                  <select
                    value={retrievalStrategy}
                    onChange={(e) => setRetrievalStrategy(e.target.value)}
                    className="w-full bg-slate-900 text-slate-100 text-xs p-2.5 rounded-lg border border-slate-700 font-medium"
                  >
                    {RETRIEVAL_STRATEGIES.map((s) => (
                      <option key={s.id} value={s.id}>
                        {s.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">
                    Vector Index Scan
                  </label>
                  <select
                    value={searchAlgorithm}
                    onChange={(e) => setSearchAlgorithm(e.target.value)}
                    className="w-full bg-slate-900 text-slate-100 text-xs p-2.5 rounded-lg border border-slate-700"
                  >
                    <option value="exact">Exact KNN (Brute-force)</option>
                    <option value="hnsw">HNSW (Graph Proximity)</option>
                    <option value="ivfflat">IVFFlat (Inverted File)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">
                    Top-K Candidates
                  </label>
                  <select
                    value={topK}
                    onChange={(e) => setTopK(e.target.value)}
                    className="w-full bg-slate-900 text-slate-100 text-xs p-2.5 rounded-lg border border-slate-700 font-mono"
                  >
                    <option value="1">1</option>
                    <option value="3">3</option>
                    <option value="5">5</option>
                    <option value="10">10</option>
                    <option value="20">20</option>
                  </select>
                </div>
              </div>

              {/* Dynamic Strategy Tuning Bar */}
              {(retrievalStrategy === 'hybrid_linear' || retrievalStrategy.includes('rrf') || retrievalStrategy.includes('rerank') || retrievalStrategy === 'multi_query_hybrid_rerank' || retrievalStrategy === 'adaptive_context') && (
                <div className="p-3 bg-slate-900/40 rounded-xl border border-slate-800 grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
                  {retrievalStrategy === 'hybrid_linear' && (
                    <div>
                      <label className="block text-[11px] text-slate-400 mb-1">
                        Linear Alpha (α): <strong className="text-purple-400">{hybridAlpha}</strong>
                      </label>
                      <input
                        type="range"
                        min="0.0"
                        max="1.0"
                        step="0.05"
                        value={hybridAlpha}
                        onChange={(e) => setHybridAlpha(parseFloat(e.target.value))}
                        className="w-full h-1.5 bg-slate-800 rounded accent-purple-500 cursor-pointer"
                      />
                    </div>
                  )}

                  {(retrievalStrategy.includes('rrf') || retrievalStrategy === 'multi_query_hybrid_rerank') && (
                    <div>
                      <label className="block text-[11px] text-slate-400 mb-1">
                        RRF Constant (k): <strong className="text-emerald-400">{rrfK}</strong>
                      </label>
                      <input
                        type="range"
                        min="1"
                        max="100"
                        value={rrfK}
                        onChange={(e) => setRrfK(parseInt(e.target.value))}
                        className="w-full h-1.5 bg-slate-800 rounded accent-emerald-500 cursor-pointer"
                      />
                    </div>
                  )}

                  {(retrievalStrategy.includes('rerank') || retrievalStrategy === 'adaptive_context') && (
                    <div>
                      <label className="block text-[11px] text-slate-400 mb-1">Cross-Encoder Model</label>
                      <select
                        value={rerankerModel}
                        onChange={(e) => setRerankerModel(e.target.value)}
                        className="w-full bg-slate-950 text-slate-200 text-xs p-1.5 rounded border border-slate-800"
                      >
                        <option value="cross-encoder/ms-marco-TinyBERT-L-2-v2">TinyBERT-L-2-v2</option>
                        <option value="mock">Mock Reranker</option>
                        <option value="none">Bypass / No-Op</option>
                      </select>
                    </div>
                  )}
                </div>
              )}

              {/* Advanced Phase 08 Toggles */}
              <div className="flex flex-wrap items-center gap-6 text-xs text-slate-300">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={enableQueryExpansion}
                    onChange={(e) => setEnableQueryExpansion(e.target.checked)}
                    className="rounded bg-slate-950 border-slate-800 text-blue-600 focus:ring-0"
                  />
                  <span>Query Expansion (Multi-Query Expansion)</span>
                </label>

                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={enableNeighborExpansion}
                    onChange={(e) => setEnableNeighborExpansion(e.target.checked)}
                    className="rounded bg-slate-950 border-slate-800 text-blue-600 focus:ring-0"
                  />
                  <span>Contiguous Neighbor Expansion</span>
                </label>
              </div>

              {runType === 'end_to_end' && (
                <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-4">
                  <div className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                    Generation & Judge Configuration
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <label className="block text-xs font-medium text-slate-400 mb-1">
                        Generator Provider
                      </label>
                      <select
                        value={llmProvider}
                        onChange={handleProviderChange}
                        className="w-full bg-slate-950 text-slate-100 text-xs p-2 rounded-lg border border-slate-700"
                      >
                        <option value="mock">Mock LLM (Offline / Free)</option>
                        <option value="gemini">Google Gemini</option>
                        <option value="openai">OpenAI</option>
                        <option value="ollama">Ollama (Local)</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-slate-400 mb-1">
                        Generator Model
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
                        Evaluator Judge
                      </label>
                      <select
                        value={judgeProvider}
                        onChange={(e) => setJudgeProvider(e.target.value)}
                        className="w-full bg-slate-950 text-slate-100 text-xs p-2 rounded-lg border border-slate-700"
                      >
                        <option value="heuristic">Rule-Based / Heuristic (Free & Deterministic)</option>
                        <option value="gemini">Gemini LLM Judge</option>
                        <option value="openai">OpenAI LLM Judge</option>
                        <option value="none">None (Skip generation evaluation)</option>
                      </select>
                    </div>
                  </div>
                </div>
              )}

              {/* Cost Guardrail Section */}
              <div className="p-4 rounded-xl bg-blue-950/20 border border-blue-900/40 flex items-start gap-3">
                <Coins className="w-5 h-5 text-blue-400 shrink-0 mt-0.5" />
                <div className="space-y-1">
                  <div className="text-xs font-semibold text-blue-200">
                    Budget Guardrail & Cost Preview
                  </div>
                  <p className="text-[11px] text-slate-400">
                    Limit the number of evaluation queries to prevent unexpected API costs. Mock and local Ollama runs are 100% free.
                  </p>
                  <div className="flex items-center gap-4 mt-2">
                    <label className="text-xs text-slate-300 flex items-center gap-2">
                      <span>Max Questions:</span>
                      <input
                        type="number"
                        min="1"
                        max="100"
                        value={maxQuestionsLimit}
                        onChange={(e) => setMaxQuestionsLimit(e.target.value)}
                        className="w-20 bg-slate-900 text-slate-100 text-xs p-1.5 rounded border border-slate-700 font-mono"
                      />
                    </label>
                    <span className="text-xs font-mono text-emerald-400">
                      Estimated Cost:{' '}
                      {llmProvider === 'mock' || llmProvider === 'ollama'
                        ? '$0.00 (Free Local)'
                        : `~$${(maxQuestionsLimit * 0.0003).toFixed(4)} USD`}
                    </span>
                  </div>
                </div>
              </div>

              <div className="pt-2">
                <button
                  type="submit"
                  disabled={isExecutingRun}
                  className="w-full py-2.5 px-4 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-800 disabled:text-slate-500 text-white font-semibold text-xs rounded-xl shadow-lg shadow-emerald-600/20 transition flex items-center justify-center gap-2"
                >
                  {isExecutingRun ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      Executing Benchmark Across Dataset...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 fill-current" />
                      Start Benchmark Execution
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 4: RESULTS & QUESTION INSPECTOR */}
      {/* ========================================================================= */}
      {activeTab === 'results' && (
        <div className="space-y-6">
          {/* Run Selection Header */}
          <div className="flex items-center justify-between bg-slate-950/80 p-4 rounded-xl border border-slate-800">
            <div className="flex items-center gap-3">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Inspect Run:
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

            {selectedRun && (
              <div className="flex items-center gap-2 font-mono text-xs">
                <span className="text-slate-400">Status:</span>
                <span className="text-emerald-400 font-semibold">{selectedRun.status}</span>
                <span className="text-slate-600">&bull;</span>
                <span className="text-slate-400">p50 Latency:</span>
                <span className="text-slate-200">{selectedRun.latency_summary?.p50_ms ?? 0} ms</span>
              </div>
            )}
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
                        {faith !== null && (
                          <span
                            className={`text-[10px] font-mono px-1.5 py-0.2 rounded border ${
                              faith >= 0.8
                                ? 'bg-emerald-950 text-emerald-400 border-emerald-800'
                                : faith >= 0.5
                                ? 'bg-amber-950 text-amber-400 border-amber-800'
                                : 'bg-rose-950 text-rose-400 border-rose-800'
                            }`}
                          >
                            Faith: {faith}
                          </span>
                        )}
                      </div>

                      <div className="flex items-center gap-3 mt-2 text-[10px] text-slate-500 font-mono">
                        <span>P: {qItem.retrieval_metrics?.['precision@5'] ?? qItem.retrieval_metrics?.['precision@3'] ?? '0.0'}</span>
                        <span>R: {qItem.retrieval_metrics?.['recall@5'] ?? qItem.retrieval_metrics?.['recall@3'] ?? '0.0'}</span>
                        <span>Time: {qItem.latency_breakdown?.total_latency_ms ?? 0} ms</span>
                      </div>
                    </div>
                  );
                })}

                {runQuestions.length === 0 && (
                  <div className="p-6 text-center text-xs text-slate-500">
                    No question records found for this run.
                  </div>
                )}
              </div>
            </div>

            {/* Right Column: Question Inspector Detail (7 cols) */}
            <div className="lg:col-span-7 bg-slate-950/80 rounded-xl border border-slate-800 p-6 space-y-6">
              {selectedQuestionItem ? (
                <>
                  <div>
                    <span className="text-[10px] font-mono uppercase tracking-wider text-blue-400 block mb-1">
                      Target Query Question
                    </span>
                    <h3 className="text-sm font-bold text-slate-100">
                      {selectedQuestionItem.question_text}
                    </h3>
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

                  {/* Retrieval Evidence Hit / Miss */}
                  <div>
                    <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 block mb-2">
                      Retrieved Chunks vs Ground Truth Evidence
                    </span>
                    <div className="flex flex-wrap gap-2">
                      {selectedQuestionItem.retrieved_chunk_ids?.map((cId, idx) => {
                        const isHit = selectedQuestionItem.relevant_chunk_ids?.includes(cId);
                        return (
                          <span
                            key={idx}
                            className={`text-[10px] font-mono px-2 py-1 rounded border flex items-center gap-1.5 ${
                              isHit
                                ? 'bg-emerald-950 text-emerald-300 border-emerald-800'
                                : 'bg-slate-900 text-slate-400 border-slate-800'
                            }`}
                          >
                            <span className="w-1.5 h-1.5 rounded-full bg-current" />
                            Chunk #{idx + 1} {isHit ? '(Ground Truth Hit)' : '(Distractor)'}
                          </span>
                        );
                      })}
                      {(!selectedQuestionItem.retrieved_chunk_ids || selectedQuestionItem.retrieved_chunk_ids.length === 0) && (
                        <span className="text-xs text-slate-500">No retrieved chunks recorded.</span>
                      )}
                    </div>
                  </div>

                  {/* Claim-Level Faithfulness Inspector */}
                  {selectedQuestionItem.claim_judgments && selectedQuestionItem.claim_judgments.length > 0 && (
                    <div className="space-y-2">
                      <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 block">
                        Claim-Level Grounding Judgments ({selectedQuestionItem.claim_judgments.length})
                      </span>
                      <div className="space-y-2">
                        {selectedQuestionItem.claim_judgments.map((cj, idx) => (
                          <div
                            key={idx}
                            className={`p-3 rounded-lg border text-xs ${
                              cj.status === 'supported'
                                ? 'bg-emerald-950/30 border-emerald-800/60 text-slate-200'
                                : cj.status === 'contradicted'
                                ? 'bg-rose-950/30 border-rose-800/60 text-slate-200'
                                : 'bg-amber-950/30 border-amber-800/60 text-slate-200'
                            }`}
                          >
                            <div className="flex items-center justify-between gap-2">
                              <span className="font-semibold text-slate-100">
                                &bull; {cj.claim}
                              </span>
                              <span
                                className={`text-[10px] font-mono uppercase px-2 py-0.5 rounded border ${
                                  cj.status === 'supported'
                                    ? 'bg-emerald-950 text-emerald-400 border-emerald-800'
                                    : cj.status === 'contradicted'
                                    ? 'bg-rose-950 text-rose-400 border-rose-800'
                                    : 'bg-amber-950 text-amber-400 border-amber-800'
                                }`}
                              >
                                {cj.status}
                              </span>
                            </div>
                            <p className="text-[11px] text-slate-400 mt-1 italic">
                              {cj.explanation}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Latency Waterfall Breakdown */}
                  <div>
                    <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 block mb-2">
                      Execution Latency Waterfall Breakdown
                    </span>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center font-mono text-[11px]">
                      <div className="p-2 rounded bg-slate-900 border border-slate-800">
                        <span className="text-slate-500 block text-[9px]">Embedding</span>
                        <span className="text-slate-200 font-semibold">{selectedQuestionItem.latency_breakdown?.embedding_latency_ms ?? 0} ms</span>
                      </div>
                      <div className="p-2 rounded bg-slate-900 border border-slate-800">
                        <span className="text-slate-500 block text-[9px]">Vector Search</span>
                        <span className="text-slate-200 font-semibold">{selectedQuestionItem.latency_breakdown?.retrieval_latency_ms ?? 0} ms</span>
                      </div>
                      <div className="p-2 rounded bg-slate-900 border border-slate-800">
                        <span className="text-slate-500 block text-[9px]">LLM Inference</span>
                        <span className="text-slate-200 font-semibold">{selectedQuestionItem.latency_breakdown?.llm_inference_ms ?? 0} ms</span>
                      </div>
                      <div className="p-2 rounded bg-slate-900 border border-slate-800">
                        <span className="text-slate-500 block text-[9px]">Total Pipeline</span>
                        <span className="text-emerald-400 font-semibold">{selectedQuestionItem.latency_breakdown?.total_latency_ms ?? 0} ms</span>
                      </div>
                    </div>
                  </div>
                </>
              ) : (
                <div className="py-16 text-center text-slate-500 text-xs">
                  Select a question from the left list to view detailed claim-level verdicts and metrics.
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 5: RUN COMPARISON */}
      {/* ========================================================================= */}
      {activeTab === 'comparison' && (
        <div className="space-y-6">
          <div className="bg-slate-950/80 p-6 rounded-xl border border-slate-800">
            <h2 className="text-base font-bold text-slate-100 mb-1 flex items-center gap-2">
              <Scale className="w-4 h-4 text-blue-400" />
              Side-by-Side Experiment Comparison
            </h2>
            <p className="text-xs text-slate-400 mb-6">
              Compare two evaluation runs to measure statistical improvements, regressions, and latency/cost trade-offs.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Baseline Run (Run A)
                </label>
                <select
                  value={compareRunAId}
                  onChange={(e) => setCompareRunAId(e.target.value)}
                  className="w-full bg-slate-900 text-slate-100 text-xs p-2.5 rounded-lg border border-slate-700"
                >
                  <option value="">Select baseline run</option>
                  {runs.map((r) => (
                    <option key={r.id} value={r.id}>
                      {r.name} &bull; {r.config?.search_algorithm} (k={r.config?.top_k})
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Candidate Run (Run B)
                </label>
                <select
                  value={compareRunBId}
                  onChange={(e) => setCompareRunBId(e.target.value)}
                  className="w-full bg-slate-900 text-slate-100 text-xs p-2.5 rounded-lg border border-slate-700"
                >
                  <option value="">Select candidate run</option>
                  {runs.map((r) => (
                    <option key={r.id} value={r.id}>
                      {r.name} &bull; {r.config?.search_algorithm} (k={r.config?.top_k})
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <button
              onClick={handleCompareRuns}
              disabled={isLoading || !compareRunAId || !compareRunBId}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 disabled:text-slate-500 text-white font-semibold text-xs rounded-lg transition"
            >
              Compare Runs
            </button>
          </div>

          {comparisonResult && (
            <div className="bg-slate-950/80 rounded-xl border border-slate-800 p-6 space-y-6">
              <div className="flex items-center justify-between border-b border-slate-800 pb-4">
                <div>
                  <h3 className="text-sm font-bold text-slate-100">
                    {comparisonResult.run_a.name} vs. {comparisonResult.run_b.name}
                  </h3>
                  <span className="text-[11px] text-slate-400">
                    Delta = Candidate (Run B) - Baseline (Run A)
                  </span>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs text-slate-300">
                  <thead className="bg-slate-900/60 text-slate-400 font-mono text-[11px] uppercase border-b border-slate-800">
                    <tr>
                      <th className="py-2.5 px-4">Evaluation Metric</th>
                      <th className="py-2.5 px-4">Run A (Baseline)</th>
                      <th className="py-2.5 px-4">Run B (Candidate)</th>
                      <th className="py-2.5 px-4">Delta</th>
                      <th className="py-2.5 px-4">Evaluation Verdict</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono">
                    {comparisonResult.metrics_comparison?.map((m, idx) => (
                      <tr key={idx} className="hover:bg-slate-900/40">
                        <td className="py-3 px-4 text-slate-200 capitalize">
                          {m.metric.replace(/_/g, ' ')}
                        </td>
                        <td className="py-3 px-4">{m.run_a_value}</td>
                        <td className="py-3 px-4">{m.run_b_value}</td>
                        <td
                          className={`py-3 px-4 font-bold ${
                            m.delta > 0
                              ? 'text-emerald-400'
                              : m.delta < 0
                              ? 'text-rose-400'
                              : 'text-slate-400'
                          }`}
                        >
                          {m.delta > 0 ? `+${m.delta}` : m.delta}
                        </td>
                        <td className="py-3 px-4">
                          <span
                            className={`text-[10px] uppercase px-2 py-0.5 rounded border ${
                              m.direction === 'improved'
                                ? 'bg-emerald-950 text-emerald-400 border-emerald-800'
                                : m.direction === 'regressed'
                                ? 'bg-rose-950 text-rose-400 border-rose-800'
                                : 'bg-slate-900 text-slate-400 border-slate-800'
                            }`}
                          >
                            {m.direction}
                          </span>
                        </td>
                      </tr>
                    ))}
                    {/* Latency row */}
                    <tr>
                      <td className="py-3 px-4 text-slate-200 font-sans font-semibold">
                        Median Latency (p50)
                      </td>
                      <td className="py-3 px-4">{comparisonResult.latency_comparison?.p50_a_ms} ms</td>
                      <td className="py-3 px-4">{comparisonResult.latency_comparison?.p50_b_ms} ms</td>
                      <td
                        className={`py-3 px-4 font-bold ${
                          comparisonResult.latency_comparison?.delta_ms < 0
                            ? 'text-emerald-400'
                            : comparisonResult.latency_comparison?.delta_ms > 0
                            ? 'text-rose-400'
                            : 'text-slate-400'
                        }`}
                      >
                        {comparisonResult.latency_comparison?.delta_ms > 0 ? `+${comparisonResult.latency_comparison.delta_ms}` : comparisonResult.latency_comparison?.delta_ms} ms
                      </td>
                      <td className="py-3 px-4">
                        <span
                          className={`text-[10px] uppercase px-2 py-0.5 rounded border ${
                            comparisonResult.latency_comparison?.direction === 'improved'
                              ? 'bg-emerald-950 text-emerald-400 border-emerald-800'
                              : comparisonResult.latency_comparison?.direction === 'regressed'
                              ? 'bg-rose-950 text-rose-400 border-rose-800'
                              : 'bg-slate-900 text-slate-400 border-slate-800'
                          }`}
                        >
                          {comparisonResult.latency_comparison?.direction === 'improved' ? 'Faster' : 'Slower'}
                        </span>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 6: EDUCATIONAL LEARNING MODE */}
      {/* ========================================================================= */}
      {activeTab === 'learning' && (
        <div className="space-y-8">
          <div className="bg-slate-950/80 p-6 rounded-xl border border-slate-800">
            <h2 className="text-base font-bold text-slate-100 mb-1 flex items-center gap-2">
              <Award className="w-5 h-5 text-amber-400" />
              Educational RAG Evaluation Laboratory
            </h2>
            <p className="text-xs text-slate-400">
              Interactive demonstrations and formal definitions designed to explain the mathematics, strengths, and common pitfalls of RAG benchmarking.
            </p>
          </div>

          {/* Interactive Sandbox 1: Precision@K vs Recall@K Slider */}
          <div className="bg-slate-950/80 p-6 rounded-xl border border-blue-900/40 space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                  <Sliders className="w-4 h-4 text-blue-400" />
                  Interactive Sandbox 1: Precision@K vs. Recall@K Tradeoff
                </h3>
                <p className="text-[11px] text-slate-400 mt-0.5">
                  Adjust K to observe how expanding the retrieval window increases evidence Recall while diluting Precision with distractor chunks.
                </p>
              </div>
              <div className="text-right font-mono text-xs">
                <span className="text-slate-400">Total Relevant Evidence: </span>
                <span className="text-emerald-400 font-bold">{totalRelevantInCorpus} chunks</span>
              </div>
            </div>

            {/* Slider Control */}
            <div className="flex items-center gap-4 py-2">
              <label className="text-xs font-semibold text-slate-300">Cutoff Top-K:</label>
              <input
                type="range"
                min="1"
                max="7"
                value={sandboxK}
                onChange={(e) => setSandboxK(parseInt(e.target.value))}
                className="w-64 accent-blue-500 cursor-pointer"
              />
              <span className="text-xs font-mono font-bold text-blue-400 px-2 py-0.5 rounded bg-blue-950 border border-blue-800">
                K = {sandboxK}
              </span>
            </div>

            {/* Live Metrics Output */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-3.5 rounded-lg bg-slate-900 border border-slate-800 text-center">
                <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Precision@{sandboxK}</span>
                <span className="text-lg font-bold font-mono text-blue-400 mt-1 block">{sandboxPrecision}</span>
                <span className="text-[10px] text-slate-500">({relevantInTopK} / {sandboxK} retrieved are relevant)</span>
              </div>
              <div className="p-3.5 rounded-lg bg-slate-900 border border-slate-800 text-center">
                <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Recall@{sandboxK}</span>
                <span className="text-lg font-bold font-mono text-emerald-400 mt-1 block">{sandboxRecall}</span>
                <span className="text-[10px] text-slate-500">({relevantInTopK} / {totalRelevantInCorpus} total found)</span>
              </div>
              <div className="p-3.5 rounded-lg bg-slate-900 border border-slate-800 text-center">
                <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Distractor Noise</span>
                <span className="text-lg font-bold font-mono text-amber-400 mt-1 block">
                  {sandboxK - relevantInTopK} chunks
                </span>
                <span className="text-[10px] text-slate-500">Irrelevant tokens sent to prompt</span>
              </div>
              <div className="p-3.5 rounded-lg bg-slate-900 border border-slate-800 text-center">
                <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Hit Rate@{sandboxK}</span>
                <span className="text-lg font-bold font-mono text-emerald-400 mt-1 block">
                  {relevantInTopK > 0 ? '1.0 (Success)' : '0.0 (Failure)'}
                </span>
                <span className="text-[10px] text-slate-500">At least one hit found</span>
              </div>
            </div>

            {/* Visual Ranking Bar */}
            <div className="space-y-2 pt-2">
              <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block">
                Ranked Retrieval Passages (Rank 1 to 7):
              </span>
              <div className="flex gap-2">
                {sampleRetrieved.map((item) => {
                  const isIncludedInTopK = item.rank <= sandboxK;
                  return (
                    <div
                      key={item.rank}
                      className={`flex-1 p-2 rounded-lg border text-center transition ${
                        isIncludedInTopK
                          ? item.isRelevant
                            ? 'bg-emerald-950/60 border-emerald-800 text-emerald-300'
                            : 'bg-amber-950/40 border-amber-800 text-amber-300'
                          : 'bg-slate-900/40 border-slate-800/50 text-slate-600 opacity-40'
                      }`}
                    >
                      <span className="text-[10px] font-mono font-bold block">Rank #{item.rank}</span>
                      <span className="text-[9px] block truncate">{item.title}</span>
                      <span className="text-[8px] font-mono uppercase font-semibold mt-1 block">
                        {item.isRelevant ? 'Relevant' : 'Noise'}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Interactive Sandbox 2: Citation Validity vs Faithfulness */}
          <div className="bg-slate-950/80 p-6 rounded-xl border border-slate-800 space-y-4">
            <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              Interactive Sandbox 2: Citation Identifier Validity vs. Semantic Support
            </h3>
            <p className="text-xs text-slate-400">
              A citation identifier (e.g. <code className="text-blue-300">[S1]</code>) may be <strong>syntactically valid</strong> (meaning the chunk exists in the prompt), yet completely <strong>unsupported</strong> or even <strong>contradicted</strong> by the text of that chunk.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
                <span className="text-xs font-semibold text-emerald-400 block">Scenario A: Supported Citation</span>
                <p className="text-xs text-slate-300">
                  <em>Passage [S1]:</em> "Yassir built REST APIs in Python and FastAPI during his internships."
                </p>
                <p className="text-xs text-slate-200">
                  <em>Generated Claim:</em> "The candidate has experience with FastAPI <span className="text-emerald-400 font-mono">[S1]</span>."
                </p>
                <div className="flex gap-2 text-[10px] font-mono pt-1">
                  <span className="px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800">
                    ID Valid: 1.0
                  </span>
                  <span className="px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800">
                    Entailment: Supported
                  </span>
                </div>
              </div>

              <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
                <span className="text-xs font-semibold text-rose-400 block">Scenario B: Citation Hallucination</span>
                <p className="text-xs text-slate-300">
                  <em>Passage [S2]:</em> "Graduated from INSEA Rabat with a degree in software engineering."
                </p>
                <p className="text-xs text-slate-200">
                  <em>Generated Claim:</em> "The candidate was the Chief Technology Officer at Microsoft <span className="text-rose-400 font-mono">[S2]</span>."
                </p>
                <div className="flex gap-2 text-[10px] font-mono pt-1">
                  <span className="px-2 py-0.5 rounded bg-blue-950 text-blue-300 border border-blue-800">
                    ID Valid: 1.0 (Tag exists!)
                  </span>
                  <span className="px-2 py-0.5 rounded bg-rose-950 text-rose-300 border border-rose-800">
                    Entailment: Unsupported / False
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Educational Metrics Reference Catalog */}
          <div className="space-y-4">
            <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
              <FileText className="w-4 h-4 text-blue-400" />
              Formal Metrics Catalog & Worked Formulas
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {metricsCatalog.map((metric) => (
                <div key={metric.id} className="p-5 rounded-xl bg-slate-950/80 border border-slate-800 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-slate-100">{metric.name}</span>
                    <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-400">
                      {metric.category}
                    </span>
                  </div>
                  <div className="p-2 rounded bg-slate-900 border border-slate-800 font-mono text-[11px] text-blue-300">
                    {metric.formula}
                  </div>
                  <p className="text-xs text-slate-300">{metric.description}</p>
                  <div className="text-[11px] text-slate-400 space-y-1 pt-1 border-t border-slate-800/60">
                    <p><strong className="text-slate-300">Interpretation:</strong> {metric.interpretation}</p>
                    <p><strong className="text-amber-400">Common Pitfall:</strong> {metric.common_pitfall}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* MODAL: CREATE DATASET */}
      {/* ========================================================================= */}
      {showNewDatasetModal && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-6 max-w-md w-full space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-slate-100">Create Golden Dataset</h3>
              <button onClick={() => setShowNewDatasetModal(false)} className="text-slate-400 hover:text-white">
                <X className="w-4 h-4" />
              </button>
            </div>
            <form onSubmit={handleCreateDataset} className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">Dataset Name</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. CV Verification Dataset"
                  value={newDatasetName}
                  onChange={(e) => setNewDatasetName(e.target.value)}
                  className="w-full bg-slate-900 text-slate-100 text-xs p-2.5 rounded-lg border border-slate-700"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">Description (Optional)</label>
                <textarea
                  rows={3}
                  placeholder="Purpose of this benchmark suite..."
                  value={newDatasetDesc}
                  onChange={(e) => setNewDatasetDesc(e.target.value)}
                  className="w-full bg-slate-900 text-slate-100 text-xs p-2.5 rounded-lg border border-slate-700"
                />
              </div>
              <div className="flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setShowNewDatasetModal(false)}
                  className="px-3 py-1.5 text-xs text-slate-400 hover:text-white"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold rounded-lg"
                >
                  Create Dataset
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* MODAL: ADD QUESTION */}
      {/* ========================================================================= */}
      {showAddQuestionModal && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-6 max-w-lg w-full space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-slate-100">Add Evaluation Question</h3>
              <button onClick={() => setShowAddQuestionModal(false)} className="text-slate-400 hover:text-white">
                <X className="w-4 h-4" />
              </button>
            </div>
            <form onSubmit={handleAddQuestion} className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">Question</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Which backend frameworks does the candidate use?"
                  value={newQuestionText}
                  onChange={(e) => setNewQuestionText(e.target.value)}
                  className="w-full bg-slate-900 text-slate-100 text-xs p-2.5 rounded-lg border border-slate-700"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Reference Ground Truth Answer
                </label>
                <textarea
                  rows={3}
                  required
                  placeholder="The factual answer expected from the evidence..."
                  value={newRefAnswer}
                  onChange={(e) => setNewRefAnswer(e.target.value)}
                  className="w-full bg-slate-900 text-slate-100 text-xs p-2.5 rounded-lg border border-slate-700"
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">Category</label>
                  <select
                    value={newCategory}
                    onChange={(e) => setNewCategory(e.target.value)}
                    className="w-full bg-slate-900 text-slate-100 text-xs p-2 rounded-lg border border-slate-700"
                  >
                    <option value="factual">Simple Factual Lookup</option>
                    <option value="multi_chunk">Multi-Chunk Reasoning</option>
                    <option value="multi_document">Multi-Document Reasoning</option>
                    <option value="paraphrasing">Semantic Paraphrasing</option>
                    <option value="unanswerable">Unanswerable Question</option>
                    <option value="conflicting">Conflicting Evidence</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">Difficulty</label>
                  <select
                    value={newDifficulty}
                    onChange={(e) => setNewDifficulty(e.target.value)}
                    className="w-full bg-slate-900 text-slate-100 text-xs p-2 rounded-lg border border-slate-700"
                  >
                    <option value="easy">Easy</option>
                    <option value="medium">Medium</option>
                    <option value="hard">Hard</option>
                  </select>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="isAnswerableCheckbox"
                  checked={newIsAnswerable}
                  onChange={(e) => setNewIsAnswerable(e.target.checked)}
                  className="rounded accent-blue-500"
                />
                <label htmlFor="isAnswerableCheckbox" className="text-xs text-slate-300 cursor-pointer">
                  Question is answerable from context (Uncheck if evidence is deliberately omitted)
                </label>
              </div>
              <div className="flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setShowAddQuestionModal(false)}
                  className="px-3 py-1.5 text-xs text-slate-400 hover:text-white"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold rounded-lg"
                >
                  Save Question
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* MODAL: IMPORT JSON */}
      {/* ========================================================================= */}
      {showImportModal && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-6 max-w-lg w-full space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-slate-100">Import Questions JSON</h3>
              <button onClick={() => setShowImportModal(false)} className="text-slate-400 hover:text-white">
                <X className="w-4 h-4" />
              </button>
            </div>
            <p className="text-xs text-slate-400">
              Paste a JSON array of questions with <code className="text-blue-300">question</code> and <code className="text-blue-300">reference_answer</code> fields.
            </p>
            <textarea
              rows={8}
              value={importJsonText}
              onChange={(e) => setImportJsonText(e.target.value)}
              placeholder={`{\n  "questions": [\n    {\n      "question": "Which backend frameworks does the candidate use?",\n      "reference_answer": "FastAPI, Express, Spring Boot.",\n      "category": "factual",\n      "is_answerable": true\n    }\n  ]\n}`}
              className="w-full bg-slate-900 text-slate-100 font-mono text-xs p-3 rounded-lg border border-slate-700"
            />
            <div className="flex justify-end gap-2 pt-2">
              <button
                type="button"
                onClick={() => setShowImportModal(false)}
                className="px-3 py-1.5 text-xs text-slate-400 hover:text-white"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleImportJson}
                className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold rounded-lg"
              >
                Import Questions
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* MODAL: EVIDENCE AUDITOR (PHASE 08 GRADED RELEVANCE) */}
      {/* ========================================================================= */}
      {showAuditorModal && auditingQuestion && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-950 border border-slate-800 rounded-xl max-w-2xl w-full max-h-[90vh] flex flex-col overflow-hidden shadow-2xl">
            {/* Modal Header */}
            <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/80">
              <div className="flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-indigo-400" />
                <h3 className="text-sm font-bold text-slate-100">
                  Evidence Relevance Auditor &bull; Revision v{auditingQuestion.revision || 1}
                </h3>
              </div>
              <button
                onClick={() => setShowAuditorModal(false)}
                className="text-slate-400 hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Modal Content */}
            <div className="p-5 overflow-y-auto space-y-4 flex-1 text-xs">
              {/* Question Context Card */}
              <div className="p-3 bg-slate-900/60 rounded-lg border border-slate-800 space-y-1.5">
                <div className="text-[10px] font-mono uppercase text-slate-400">Question</div>
                <div className="text-slate-100 font-semibold">{auditingQuestion.question}</div>
                <div className="text-[10px] font-mono uppercase text-slate-400 pt-1">Reference Answer</div>
                <div className="text-slate-300 font-sans">{auditingQuestion.reference_answer}</div>
              </div>

              {/* Audit History Log */}
              {auditingQuestion.audit_history?.length > 0 && (
                <div className="p-2.5 bg-slate-900/30 rounded border border-slate-800/60 text-[11px] text-slate-400 space-y-1">
                  <div className="font-semibold text-slate-300">Prior Audit Trail:</div>
                  {auditingQuestion.audit_history.slice(-3).map((log, idx) => (
                    <div key={idx} className="font-mono text-[10px] text-slate-500">
                      v{log.revision}: {log.reviewer} &bull; {log.timestamp?.slice(0, 16)} &bull; Notes: "{log.reviewer_notes || 'None'}"
                    </div>
                  ))}
                </div>
              )}

              {/* Chunks Labeling Section */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-slate-200 uppercase tracking-wide text-[11px]">
                    Candidate Evidence Chunks &amp; Graded Relevance
                  </span>
                  <span className="text-[10px] text-slate-500 font-mono">
                    Labels: 0 (Irrelevant), 1 (Partially Relevant), 2 (Direct Hit)
                  </span>
                </div>

                {isLoadingAuditChunks ? (
                  <div className="p-8 text-center text-slate-500">
                    <RefreshCw className="w-4 h-4 animate-spin mx-auto mb-2" />
                    Fetching candidate passages for evaluation question...
                  </div>
                ) : auditCandidateChunks.length === 0 ? (
                  <div className="p-6 text-center text-slate-500 bg-slate-900/40 rounded-lg border border-slate-800">
                    No retrieved candidate chunks found for this query.
                  </div>
                ) : (
                  auditCandidateChunks.map((chunk, idx) => {
                    const currentVal = auditLabels[chunk.chunk_id] !== undefined ? auditLabels[chunk.chunk_id] : 0;
                    return (
                      <div
                        key={chunk.chunk_id}
                        className="p-3 bg-slate-900/50 rounded-lg border border-slate-800 space-y-2"
                      >
                        <div className="flex items-center justify-between text-[11px]">
                          <span className="font-bold text-blue-400 font-mono">
                            Passage #{idx + 1} &bull; {chunk.document_name}
                          </span>
                          {/* Graded Relevance Selector Buttons */}
                          <div className="flex items-center gap-1 font-mono text-[10px]">
                            <button
                              type="button"
                              onClick={() => setAuditLabels({ ...auditLabels, [chunk.chunk_id]: 0 })}
                              className={`px-2 py-0.5 rounded border transition ${
                                currentVal === 0
                                  ? 'bg-rose-950 text-rose-300 border-rose-700 font-bold'
                                  : 'bg-slate-950 text-slate-500 border-slate-800 hover:text-slate-300'
                              }`}
                            >
                              0: Irrelevant
                            </button>
                            <button
                              type="button"
                              onClick={() => setAuditLabels({ ...auditLabels, [chunk.chunk_id]: 1 })}
                              className={`px-2 py-0.5 rounded border transition ${
                                currentVal === 1
                                  ? 'bg-amber-950 text-amber-300 border-amber-700 font-bold'
                                  : 'bg-slate-950 text-slate-500 border-slate-800 hover:text-slate-300'
                              }`}
                            >
                              1: Partial
                            </button>
                            <button
                              type="button"
                              onClick={() => setAuditLabels({ ...auditLabels, [chunk.chunk_id]: 2 })}
                              className={`px-2 py-0.5 rounded border transition ${
                                currentVal === 2
                                  ? 'bg-emerald-950 text-emerald-300 border-emerald-700 font-bold'
                                  : 'bg-slate-950 text-slate-500 border-slate-800 hover:text-slate-300'
                              }`}
                            >
                              2: Direct Hit
                            </button>
                          </div>
                        </div>

                        <p className="text-[11px] text-slate-300 font-sans line-clamp-3 bg-slate-950/70 p-2 rounded border border-slate-800/80">
                          {chunk.content}
                        </p>
                      </div>
                    );
                  })
                )}
              </div>

              {/* Reviewer Audit Notes */}
              <div>
                <label className="block text-[11px] font-medium text-slate-400 mb-1">
                  Auditor Notes &amp; Revision Rationale
                </label>
                <textarea
                  rows={2}
                  value={auditNotes}
                  onChange={(e) => setAuditNotes(e.target.value)}
                  placeholder="e.g. Verified chunk #1 contains explicit mention of backend frameworks; marked as direct hit (2)."
                  className="w-full bg-slate-900 text-slate-100 text-xs p-2.5 rounded-lg border border-slate-800 focus:outline-none focus:border-indigo-500"
                />
              </div>
            </div>

            {/* Modal Footer */}
            <div className="p-4 border-t border-slate-800 flex items-center justify-between bg-slate-900/60">
              <span className="text-[11px] text-slate-500">
                Incrementing revision to v{(auditingQuestion.revision || 1) + 1} upon save.
              </span>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setShowAuditorModal(false)}
                  className="px-3 py-1.5 text-xs text-slate-400 hover:text-white"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={handleSaveAudit}
                  disabled={isSavingAudit}
                  className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-xs font-semibold rounded-lg flex items-center gap-1.5"
                >
                  {isSavingAudit ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Save className="w-3.5 h-3.5" />}
                  Save Audit &amp; Increment Revision
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function MetricCard({ title, value, subtitle, trend, unit }) {
  return (
    <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 flex flex-col justify-between">
      <div>
        <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">
          {title}
        </span>
        <div className="text-2xl font-bold font-mono text-slate-100 mt-1 flex items-baseline gap-1">
          <span>{value}</span>
          {unit && <span className="text-xs text-slate-400 font-normal">{unit}</span>}
        </div>
      </div>
      <p className="text-[10px] text-slate-500 mt-2">{subtitle}</p>
    </div>
  );
}
