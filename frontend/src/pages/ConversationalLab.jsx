import React, { useState, useEffect, useRef } from 'react';
import {
  MessageSquare,
  Send,
  Plus,
  Trash2,
  Edit2,
  Check,
  X,
  Sliders,
  History,
  GitCompare,
  Cpu,
  Database,
  Layers,
  Sparkles,
  AlertTriangle,
  CheckCircle,
  Clock,
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
  Search,
  Settings,
  ArrowRight,
  Split,
  Eye,
  BarChart3,
  Flame,
} from 'lucide-react';
import api from '../services/api';

const PRESET_CONVERSATIONS = [
  {
    title: "Miacorp & Authentication Follow-up",
    q1: "What technologies did Yassir use during his Miacorp internship?",
    q2: "Which of them was used for authentication?",
    q3: "Did he also work on real-time messaging there?",
  },
  {
    title: "Education & Degree Coreference",
    q1: "Tell me about Yassir's profile and graduation.",
    q2: "Where did he earn his engineering degree?",
    q3: "What were his major areas of specialization?",
  },
  {
    title: "AI4U Real-time & Tools",
    q1: "What did the candidate build at AI4U / Seven Eleven Media?",
    q2: "Which tools were used for real-time communication?",
    q3: "How were these features tested under load?",
  },
];

export function ConversationalLab() {
  // Session list & active session
  const [conversations, setConversations] = useState([]);
  const [activeConvId, setActiveConvId] = useState(null);
  const [activeConv, setActiveConv] = useState(null);
  const [searchFilter, setSearchFilter] = useState('');
  const [loadingSessions, setLoadingSessions] = useState(false);

  // Rename session inline state
  const [editingTitleId, setEditingTitleId] = useState(null);
  const [newTitleText, setNewTitleText] = useState('');

  // Messages & Turn execution state
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');
  const [loadingTurn, setLoadingTurn] = useState(false);
  const [streamingToken, setStreamingToken] = useState('');
  const [streamingStage, setStreamingStage] = useState('');
  const [streamingAnswer, setStreamingAnswer] = useState('');

  // Settings & Tuning
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  const [activeStrategy, setActiveStrategy] = useState('sliding_window');
  const [windowTurns, setWindowTurns] = useState(3);
  const [includeAssistant, setIncludeAssistant] = useState(true);
  const [maxMemoryTokens, setMaxMemoryTokens] = useState(1200);
  const [retrievalStrategy, setRetrievalStrategy] = useState('hybrid_rrf');
  const [llmProvider, setLlmProvider] = useState('mock');
  const [enableRewriting, setEnableRewriting] = useState(true);
  const [documentFilter, setDocumentFilter] = useState('');

  // Available metadata
  const [availableDocs, setAvailableDocs] = useState([]);
  const [memoryStrategies, setMemoryStrategies] = useState([]);

  // Diagnostic Inspector Tabs
  // 'memory' | 'rewriting' | 'telemetry' | 'comparison' | 'benchmark' | 'learning'
  const [activeInspectorTab, setActiveInspectorTab] = useState('memory');
  const [latestTurnMeta, setLatestTurnMeta] = useState(null);
  const [memoryInspection, setMemoryInspection] = useState(null);
  const [rewritePreview, setRewritePreview] = useState(null);
  const [strategyComparison, setStrategyComparison] = useState([]);
  const [loadingComparison, setLoadingComparison] = useState(false);

  // Benchmark suite state
  const [benchmarkReport, setBenchmarkReport] = useState(null);
  const [loadingBenchmark, setLoadingBenchmark] = useState(false);

  // Sources drawer / viewer
  const [selectedSource, setSelectedSource] = useState(null);

  const messagesEndRef = useRef(null);

  // 1. Initial Load: Documents, Memory Strategies, Sessions
  useEffect(() => {
    loadInitialData();
  }, []);

  // 2. When active session changes, load messages & memory inspection
  useEffect(() => {
    if (activeConvId) {
      loadSessionDetails(activeConvId);
    } else {
      setMessages([]);
      setActiveConv(null);
      setMemoryInspection(null);
    }
  }, [activeConvId]);

  // Scroll to bottom on new message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingAnswer]);

  const loadInitialData = async () => {
    try {
      setLoadingSessions(true);
      const [docsRes, stratRes, convsRes] = await Promise.all([
        api.listDocuments(100, 0).catch(() => ({ documents: [] })),
        api.listMemoryStrategies().catch(() => []),
        api.listConversations({ limit: 50 }).catch(() => ({ items: [] })),
      ]);

      setAvailableDocs(docsRes.documents || docsRes || []);
      setMemoryStrategies(stratRes || []);

      const convList = convsRes.items || [];
      setConversations(convList);

      if (convList.length > 0) {
        setActiveConvId(convList[0].id);
      } else {
        // Auto-create initial session
        handleCreateSession("Session de Démonstration Multi-Tours");
      }
    } catch (err) {
      console.error("Failed to load initial conversational data", err);
    } finally {
      setLoadingSessions(false);
    }
  };

  const loadSessionDetails = async (convId) => {
    try {
      const [conv, msgsRes, memRes] = await Promise.all([
        api.getConversation(convId),
        api.listConversationMessages(convId, { limit: 100 }),
        api.inspectMemory(convId).catch(() => null),
      ]);

      setActiveConv(conv);
      setMessages(msgsRes.items || []);
      setMemoryInspection(memRes);

      // Populate local settings from session settings
      if (conv.settings) {
        setActiveStrategy(conv.settings.memory_strategy || 'sliding_window');
        if (conv.settings.memory_config) {
          setWindowTurns(conv.settings.memory_config.sliding_window_turns || 3);
          setIncludeAssistant(conv.settings.memory_config.include_assistant_replies !== false);
          setMaxMemoryTokens(conv.settings.memory_config.max_memory_tokens || 1200);
        }
        setRetrievalStrategy(conv.settings.retrieval_strategy || 'hybrid_rrf');
        setLlmProvider(conv.settings.llm_provider || 'mock');
        setEnableRewriting(conv.settings.enable_query_rewriting !== false);
        setDocumentFilter(conv.settings.document_filter || '');
      }

      // Check if last message has turn metadata
      const items = msgsRes.items || [];
      if (items.length > 0) {
        const lastAsst = [...items].reverse().find(m => m.role === 'assistant');
        if (lastAsst && lastAsst.metadata_json) {
          setLatestTurnMeta(lastAsst.metadata_json);
        }
      }
    } catch (err) {
      console.error("Failed to load session details", err);
    }
  };

  const handleCreateSession = async (title = "Nouvelle Conversation") => {
    try {
      const newConv = await api.createConversation({
        title,
        settings: {
          memory_strategy: activeStrategy,
          memory_config: {
            sliding_window_turns: windowTurns,
            include_assistant_replies: includeAssistant,
            max_memory_tokens: maxMemoryTokens,
          },
          retrieval_strategy: retrievalStrategy,
          llm_provider: llmProvider,
          enable_query_rewriting: enableRewriting,
          document_filter: documentFilter || null,
        },
      });

      setConversations([newConv, ...conversations]);
      setActiveConvId(newConv.id);
    } catch (err) {
      console.error("Failed to create conversation", err);
    }
  };

  const handleDeleteSession = async (convId, e) => {
    e.stopPropagation();
    if (!window.confirm("Supprimer cette conversation et tout son historique ?")) return;

    try {
      await api.deleteConversation(convId);
      const remaining = conversations.filter(c => c.id !== convId);
      setConversations(remaining);
      if (activeConvId === convId) {
        setActiveConvId(remaining.length > 0 ? remaining[0].id : null);
      }
    } catch (err) {
      console.error("Failed to delete conversation", err);
    }
  };

  const handleSaveTitle = async (convId) => {
    if (!newTitleText.trim()) {
      setEditingTitleId(null);
      return;
    }
    try {
      const updated = await api.updateConversation(convId, { title: newTitleText.trim() });
      setConversations(conversations.map(c => c.id === convId ? updated : c));
      if (activeConvId === convId) setActiveConv(updated);
      setEditingTitleId(null);
    } catch (err) {
      console.error("Failed to update title", err);
    }
  };

  const handleUpdateSettings = async () => {
    if (!activeConvId) return;
    try {
      const newSettings = {
        memory_strategy: activeStrategy,
        memory_config: {
          sliding_window_turns: Number(windowTurns),
          include_assistant_replies: includeAssistant,
          max_memory_tokens: Number(maxMemoryTokens),
        },
        retrieval_strategy: retrievalStrategy,
        llm_provider: llmProvider,
        enable_query_rewriting: enableRewriting,
        document_filter: documentFilter || null,
      };

      const updated = await api.updateConversation(activeConvId, { settings: newSettings });
      setActiveConv(updated);
      setConversations(conversations.map(c => c.id === activeConvId ? updated : c));
      setShowSettingsModal(false);

      // Refresh memory inspection with new settings
      const memRes = await api.inspectMemory(activeConvId, activeStrategy);
      setMemoryInspection(memRes);
    } catch (err) {
      console.error("Failed to update settings", err);
    }
  };

  // Preview query rewrite as user types
  const handleQueryInputChange = async (e) => {
    const val = e.target.value;
    setUserInput(val);

    if (val.trim().length >= 4 && activeConvId && enableRewriting) {
      try {
        const preview = await api.previewQueryRewrite(activeConvId, val.trim());
        setRewritePreview(preview);
      } catch {
        setRewritePreview(null);
      }
    } else {
      setRewritePreview(null);
    }
  };

  // Submit Turn (Synchronous Execution)
  const handleSubmitTurn = async (e) => {
    e?.preventDefault();
    const queryText = userInput.trim();
    if (!queryText || !activeConvId || loadingTurn) return;

    setUserInput('');
    setLoadingTurn(true);
    setStreamingAnswer('');

    // Optimistic user bubble
    const tempUserMsg = {
      id: 'temp-' + Date.now(),
      role: 'user',
      content: queryText,
      sequence_number: messages.length + 1,
      created_at: new Date().toISOString(),
    };
    setMessages(prev => [...prev, tempUserMsg]);

    try {
      const turnResp = await api.executeTurn(activeConvId, {
        message: queryText,
        override_settings: {
          memory_strategy: activeStrategy,
          retrieval_strategy: retrievalStrategy,
          llm_provider: llmProvider,
          enable_query_rewriting: enableRewriting,
          document_filter: documentFilter || null,
        },
      });

      // Update messages with finalized server versions
      setMessages(prev => [
        ...prev.filter(m => m.id !== tempUserMsg.id),
        turnResp.user_message,
        turnResp.assistant_message,
      ]);

      // Update diagnostic inspectors
      setLatestTurnMeta(turnResp.assistant_message.metadata_json);
      setMemoryInspection(turnResp.memory_selection);
      setRewritePreview(turnResp.rewritten_query_info);

      // Update session in list
      setConversations(conversations.map(c => {
        if (c.id === activeConvId) {
          return {
            ...c,
            message_count: c.message_count + 2,
            latest_message: turnResp.assistant_message.content,
            updated_at: new Date().toISOString(),
          };
        }
        return c;
      }));
    } catch (err) {
      console.error("Failed turn execution", err);
      // Append error message
      setMessages(prev => [
        ...prev,
        {
          id: 'err-' + Date.now(),
          role: 'assistant',
          content: `⚠️ Erreur lors de l'exécution du tour: ${err.message || 'Échec de connexion au pipeline RAG.'}`,
          sequence_number: messages.length + 2,
          created_at: new Date().toISOString(),
          isError: true,
        },
      ]);
    } finally {
      setLoadingTurn(false);
    }
  };

  // Run Strategy Comparison on current input or last question
  const handleRunComparison = async (queryText) => {
    const q = queryText || userInput || (messages.length > 0 ? messages[messages.length - 1].content : "What degree did he earn?");
    if (!activeConvId || !q) return;

    try {
      setLoadingComparison(true);
      setActiveInspectorTab('comparison');
      const compResults = await api.compareMemoryStrategies(activeConvId, q);
      setStrategyComparison(compResults);
    } catch (err) {
      console.error("Strategy comparison failed", err);
    } finally {
      setLoadingComparison(false);
    }
  };

  // Run 12-Scenario Benchmark
  const handleRunBenchmark = async () => {
    try {
      setLoadingBenchmark(true);
      setActiveInspectorTab('benchmark');
      const report = await api.runConversationalBenchmark({
        memory_strategy: activeStrategy,
        retrieval_strategy: retrievalStrategy,
        llm_provider: llmProvider,
        top_k: 5,
      });
      setBenchmarkReport(report);
    } catch (err) {
      console.error("Benchmark failed", err);
    } finally {
      setLoadingBenchmark(false);
    }
  };

  const filteredConversations = conversations.filter(c =>
    (c.title || '').toLowerCase().includes(searchFilter.toLowerCase())
  );

  return (
    <div className="flex-1 flex overflow-hidden bg-slate-950 text-slate-100 font-sans">
      {/* ========================================================= */}
      {/* COLUMN 1: SESSIONS & CONVERSATION MANAGEMENT */}
      {/* ========================================================= */}
      <div className="w-72 bg-slate-900/80 border-r border-slate-800 flex flex-col shrink-0">
        <div className="p-3 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <MessageSquare className="w-4 h-4 text-indigo-400" />
            <span className="font-semibold text-xs tracking-wider uppercase text-slate-300">
              Conversations
            </span>
          </div>
          <button
            onClick={() => handleCreateSession()}
            className="flex items-center gap-1 px-2.5 py-1 text-xs font-medium rounded-md bg-indigo-600 hover:bg-indigo-500 text-white transition-colors shadow-sm"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>Nouveau</span>
          </button>
        </div>

        {/* Search Bar */}
        <div className="p-2 border-b border-slate-800/60">
          <div className="relative">
            <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-slate-500" />
            <input
              type="text"
              placeholder="Filtrer les conversations..."
              value={searchFilter}
              onChange={(e) => setSearchFilter(e.target.value)}
              className="w-full pl-8 pr-3 py-1.5 bg-slate-950 border border-slate-800 rounded-md text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500"
            />
          </div>
        </div>

        {/* Session List */}
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {filteredConversations.length === 0 ? (
            <div className="text-center py-8 text-xs text-slate-500">
              Aucune conversation trouvée.
            </div>
          ) : (
            filteredConversations.map((c) => {
              const isActive = c.id === activeConvId;
              const isEditing = editingTitleId === c.id;
              const strat = c.settings?.memory_strategy || 'sliding_window';

              return (
                <div
                  key={c.id}
                  onClick={() => setActiveConvId(c.id)}
                  className={`group relative p-2.5 rounded-lg cursor-pointer transition-all border ${
                    isActive
                      ? 'bg-indigo-950/40 border-indigo-500/40 shadow-sm'
                      : 'hover:bg-slate-800/60 border-transparent text-slate-300'
                  }`}
                >
                  <div className="flex items-start justify-between gap-1.5">
                    {isEditing ? (
                      <div className="flex items-center gap-1 flex-1" onClick={(e) => e.stopPropagation()}>
                        <input
                          type="text"
                          value={newTitleText}
                          onChange={(e) => setNewTitleText(e.target.value)}
                          className="w-full px-2 py-0.5 text-xs bg-slate-950 border border-indigo-500 rounded text-slate-200 focus:outline-none"
                          autoFocus
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') handleSaveTitle(c.id);
                            if (e.key === 'Escape') setEditingTitleId(null);
                          }}
                        />
                        <button
                          onClick={() => handleSaveTitle(c.id)}
                          className="p-1 text-emerald-400 hover:text-emerald-300"
                        >
                          <Check className="w-3.5 h-3.5" />
                        </button>
                        <button
                          onClick={() => setEditingTitleId(null)}
                          className="p-1 text-slate-400 hover:text-slate-300"
                        >
                          <X className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    ) : (
                      <div className="flex-1 min-w-0">
                        <span className="text-xs font-medium text-slate-200 truncate block">
                          {c.title}
                        </span>
                        <div className="flex items-center gap-1.5 mt-1">
                          <span className="text-[10px] font-mono px-1 py-0.2 rounded bg-slate-800 text-slate-400">
                            {c.message_count || 0} tours
                          </span>
                          <span className="text-[10px] font-mono px-1 py-0.2 rounded bg-indigo-950/60 text-indigo-400 border border-indigo-900/40">
                            {strat.replace('_', ' ')}
                          </span>
                        </div>
                      </div>
                    )}

                    {!isEditing && (
                      <div className="opacity-0 group-hover:opacity-100 flex items-center gap-1 transition-opacity">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setEditingTitleId(c.id);
                            setNewTitleText(c.title);
                          }}
                          className="p-1 text-slate-400 hover:text-slate-200"
                          title="Renommer"
                        >
                          <Edit2 className="w-3 h-3" />
                        </button>
                        <button
                          onClick={(e) => handleDeleteSession(c.id, e)}
                          className="p-1 text-rose-400 hover:text-rose-300"
                          title="Supprimer"
                        >
                          <Trash2 className="w-3 h-3" />
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Preset Conversations Box */}
        <div className="p-3 border-t border-slate-800 bg-slate-950/60">
          <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
            <BookOpen className="w-3.5 h-3.5 text-indigo-400" />
            <span>Scénarios Préréglés</span>
          </div>
          <div className="space-y-1">
            {PRESET_CONVERSATIONS.map((preset, idx) => (
              <button
                key={idx}
                onClick={async () => {
                  await handleCreateSession(preset.title);
                  setUserInput(preset.q1);
                }}
                className="w-full text-left text-[11px] p-1.5 rounded hover:bg-slate-800/60 text-slate-300 hover:text-indigo-300 transition-colors truncate block"
              >
                &bull; {preset.title}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* ========================================================= */}
      {/* COLUMN 2: ACTIVE CHAT FEED & MESSAGE INPUT */}
      {/* ========================================================= */}
      <div className="flex-1 flex flex-col min-w-0 bg-slate-950 border-r border-slate-800">
        {/* Top Session Toolbar */}
        <div className="h-14 px-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/40">
          <div className="flex items-center gap-3 min-w-0">
            <span className="font-semibold text-sm text-slate-200 truncate max-w-xs">
              {activeConv?.title || "Sélectionnez une session"}
            </span>
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-950 text-emerald-400 font-mono border border-emerald-800/40">
              Actif
            </span>
          </div>

          <div className="flex items-center gap-2">
            {/* Quick Memory Strategy Selector */}
            <div className="flex items-center gap-1.5 bg-slate-900 border border-slate-800 px-2 py-1 rounded-md text-xs">
              <Layers className="w-3.5 h-3.5 text-indigo-400" />
              <select
                value={activeStrategy}
                onChange={(e) => {
                  setActiveStrategy(e.target.value);
                  setTimeout(() => handleUpdateSettings(), 50);
                }}
                className="bg-transparent text-slate-200 text-xs focus:outline-none cursor-pointer"
              >
                <option value="sliding_window">Fenêtre Glissante (Sliding Window)</option>
                <option value="token_bounded">Budget Borné (Token-Bounded)</option>
                <option value="summary_memory">Mémoire Résumé (Summary Memory)</option>
                <option value="no_memory">Sans Mémoire (Stateless RAG)</option>
              </select>
            </div>

            {/* Settings Modal Toggle */}
            <button
              onClick={() => setShowSettingsModal(true)}
              className="p-1.5 rounded-md bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 transition-colors"
              title="Paramètres avancés de la mémoire et du RAG"
            >
              <Sliders className="w-4 h-4 text-indigo-400" />
            </button>
          </div>
        </div>

        {/* Message Feed */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center p-6 text-slate-500">
              <div className="w-12 h-12 rounded-2xl bg-indigo-950/60 border border-indigo-800/40 flex items-center justify-center text-indigo-400 mb-3 shadow-lg shadow-indigo-500/10">
                <Sparkles className="w-6 h-6" />
              </div>
              <h3 className="text-sm font-semibold text-slate-300 mb-1">
                Laboratoire RAG Conversationnel & Ingénierie Mémoire
              </h3>
              <p className="text-xs max-w-md text-slate-400 mb-4">
                Posez des questions enchaînées sur le profil ingénieur. Observez la réécriture autonome de requête, le budget mémoire, et la citation des passages indexés.
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-w-lg w-full text-left">
                {PRESET_CONVERSATIONS.slice(0, 2).map((p, i) => (
                  <button
                    key={i}
                    onClick={() => setUserInput(p.q1)}
                    className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800 hover:border-indigo-500/50 hover:bg-indigo-950/20 text-xs text-slate-300 transition-all text-left"
                  >
                    <span className="font-semibold text-indigo-400 block mb-0.5">Scénario {i + 1}</span>
                    <span className="text-slate-400 text-[11px] block">{p.q1}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((m, idx) => {
              const isUser = m.role === 'user';
              const meta = m.metadata_json || {};
              const rewritten = meta.rewritten_query_info?.rewritten_query;
              const timing = meta.timing_breakdown || {};
              const citations = meta.citations || [];
              const sources = meta.sources || [];

              return (
                <div
                  key={m.id || idx}
                  className={`flex flex-col ${isUser ? 'items-end' : 'items-start'} max-w-3xl mx-auto w-full`}
                >
                  <div className="flex items-center gap-2 mb-1 px-1">
                    <span className="text-[10px] font-mono text-slate-500 uppercase">
                      {isUser ? 'Utilisateur' : 'Assistant RAG'} &bull; Tour #{m.sequence_number}
                    </span>
                  </div>

                  <div
                    className={`rounded-2xl p-3.5 text-xs leading-relaxed max-w-2xl shadow-sm ${
                      isUser
                        ? 'bg-indigo-600 text-white rounded-tr-none'
                        : m.isError
                        ? 'bg-rose-950/60 border border-rose-800/60 text-rose-200 rounded-tl-none'
                        : 'bg-slate-900/90 border border-slate-800 text-slate-200 rounded-tl-none'
                    }`}
                  >
                    <div className="whitespace-pre-wrap">{m.content}</div>

                    {/* Metadata Drawer in Assistant Bubble */}
                    {!isUser && !m.isError && (
                      <div className="mt-3 pt-2.5 border-t border-slate-800/80 flex flex-wrap items-center gap-1.5 text-[10px] font-mono">
                        {rewritten && rewritten.toLowerCase() !== m.content.toLowerCase() && (
                          <span
                            onClick={() => setActiveInspectorTab('rewriting')}
                            className="cursor-pointer px-1.5 py-0.5 rounded bg-purple-950 text-purple-300 border border-purple-800/40 hover:bg-purple-900/60 transition-colors"
                            title={`Requête réécrite: ${rewritten}`}
                          >
                            🎯 Réécriture: "{rewritten.slice(0, 30)}..."
                          </span>
                        )}

                        {timing.total_latency_ms && (
                          <span className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">
                            ⏱️ {timing.total_latency_ms} ms
                          </span>
                        )}

                        {sources.length > 0 && (
                          <span
                            onClick={() => setActiveInspectorTab('telemetry')}
                            className="cursor-pointer px-1.5 py-0.5 rounded bg-blue-950 text-blue-300 border border-blue-800/40 hover:bg-blue-900/60"
                          >
                            📄 {sources.length} sources [{sources.map(s => `S${s.citation_index}`).join(', ')}]
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              );
            })
          )}

          {/* Loading Indicator */}
          {loadingTurn && (
            <div className="flex flex-col items-start max-w-3xl mx-auto w-full">
              <div className="flex items-center gap-2 mb-1 px-1">
                <span className="text-[10px] font-mono text-indigo-400 animate-pulse">
                  Assistant &bull; Exécution du Pipeline RAG...
                </span>
              </div>
              <div className="rounded-2xl rounded-tl-none p-3.5 bg-slate-900/60 border border-slate-800 text-xs text-slate-400 flex items-center gap-2.5">
                <RefreshCw className="w-4 h-4 animate-spin text-indigo-400" />
                <span>Sélection de mémoire & réécriture de requête autonome...</span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Bar */}
        <div className="p-3 border-t border-slate-800 bg-slate-900/50">
          {/* Query Rewrite Floating Live Pill */}
          {rewritePreview && rewritePreview.is_rewritten && (
            <div className="mb-2 px-3 py-1.5 rounded-md bg-indigo-950/60 border border-indigo-800/50 flex items-center justify-between text-xs text-indigo-300">
              <div className="flex items-center gap-2 truncate">
                <Sparkles className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
                <span className="truncate">
                  <span className="text-slate-400">Réécriture autonome détectée :</span> "{rewritePreview.rewritten_query}"
                </span>
              </div>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-indigo-900 text-indigo-200 shrink-0">
                Prêt
              </span>
            </div>
          )}

          <form onSubmit={handleSubmitTurn} className="flex gap-2">
            <textarea
              rows={1}
              value={userInput}
              onChange={handleQueryInputChange}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmitTurn(e);
                }
              }}
              placeholder="Posez une question contextualisée sur le profil..."
              className="flex-1 px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500 resize-none"
            />
            <button
              type="submit"
              disabled={loadingTurn || !userInput.trim()}
              className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 disabled:text-slate-500 text-white font-medium rounded-lg text-xs flex items-center gap-1.5 transition-colors shadow-sm"
            >
              <Send className="w-3.5 h-3.5" />
              <span>Envoyer</span>
            </button>
          </form>
        </div>
      </div>

      {/* ========================================================= */}
      {/* COLUMN 3: MULTI-PANEL DIAGNOSTIC INSPECTOR */}
      {/* ========================================================= */}
      <div className="w-96 bg-slate-900/90 border-l border-slate-800 flex flex-col shrink-0">
        {/* Tab Headers */}
        <div className="flex border-b border-slate-800 bg-slate-950/60 overflow-x-auto text-xs">
          {[
            { id: 'memory', label: 'Mémoire', icon: History },
            { id: 'rewriting', label: 'Réécriture', icon: Sparkles },
            { id: 'telemetry', label: 'Télémétrie', icon: BarChart3 },
            { id: 'comparison', label: 'Comparateur', icon: Split },
            { id: 'benchmark', label: 'Benchmark', icon: Flame },
            { id: 'learning', label: 'Pédagogie', icon: BookOpen },
          ].map((tab) => {
            const Icon = tab.icon;
            const isActive = activeInspectorTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveInspectorTab(tab.id)}
                className={`flex items-center gap-1.5 px-3 py-2.5 font-medium border-b-2 whitespace-nowrap transition-colors ${
                  isActive
                    ? 'border-indigo-500 text-indigo-400 bg-indigo-950/20'
                    : 'border-transparent text-slate-400 hover:text-slate-200'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>

        {/* Tab Content Area */}
        <div className="flex-1 overflow-y-auto p-3 text-xs space-y-3">
          {/* TAB 1: MEMORY INSPECTOR */}
          {activeInspectorTab === 'memory' && (
            <div className="space-y-3">
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-semibold text-slate-300">Stratégie Active</span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-950 text-indigo-400 border border-indigo-800/40">
                    {memoryInspection?.strategy || activeStrategy}
                  </span>
                </div>

                {/* Token Capacity Bar */}
                <div className="space-y-1 mt-2">
                  <div className="flex justify-between text-[11px] text-slate-400 font-mono">
                    <span>Tokens Mémoire :</span>
                    <span>
                      {memoryInspection?.allocated_memory_tokens || 0} / {memoryInspection?.token_budget || maxMemoryTokens}
                    </span>
                  </div>
                  <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                    <div
                      className="bg-indigo-500 h-1.5 rounded-full transition-all"
                      style={{
                        width: `${Math.min(100, ((memoryInspection?.allocated_memory_tokens || 0) / (memoryInspection?.token_budget || maxMemoryTokens || 1)) * 100)}%`
                      }}
                    />
                  </div>
                </div>
              </div>

              {/* Rolling Summary Box (if present) */}
              {memoryInspection?.summary_used && (
                <div className="p-3 rounded-lg bg-purple-950/30 border border-purple-800/40">
                  <div className="flex items-center gap-1.5 text-purple-400 font-semibold mb-1 text-[11px]">
                    <Sparkles className="w-3.5 h-3.5" />
                    <span>Résumé Roulant Utilisé (Jusqu'au Tour #{memoryInspection.summary_covered_sequence})</span>
                  </div>
                  <p className="text-slate-300 text-xs leading-relaxed italic">
                    "{memoryInspection.summary_used}"
                  </p>
                </div>
              )}

              {/* Partitioned Message Breakdown */}
              <div className="space-y-2">
                <span className="font-semibold text-slate-300 text-xs block">
                  Partition de l'Historique ({memoryInspection?.inspection_items?.length || 0} messages)
                </span>
                {(!memoryInspection?.inspection_items || memoryInspection.inspection_items.length === 0) ? (
                  <div className="text-slate-500 text-center py-4 text-xs">
                    Aucun message dans l'historique de cette session.
                  </div>
                ) : (
                  memoryInspection.inspection_items.map((item, idx) => (
                    <div
                      key={idx}
                      className={`p-2 rounded-lg border text-xs space-y-1 ${
                        item.is_included
                          ? 'bg-emerald-950/20 border-emerald-800/40 text-emerald-300'
                          : 'bg-slate-950/60 border-slate-800 text-slate-400'
                      }`}
                    >
                      <div className="flex items-center justify-between text-[10px] font-mono">
                        <span className="font-semibold">
                          #{item.sequence_number} &bull; {item.role.toUpperCase()}
                        </span>
                        <span className={`px-1.5 py-0.2 rounded ${item.is_included ? 'bg-emerald-900/60 text-emerald-300' : 'bg-slate-800 text-slate-400'}`}>
                          {item.is_included ? 'Inclus en mémoire' : 'Exclu'} ({item.token_count} tok)
                        </span>
                      </div>
                      <div className="text-slate-300 truncate">
                        "{item.content_snippet}"
                      </div>
                      <div className="text-[10px] text-slate-400">
                        {item.inclusion_reason}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}

          {/* TAB 2: QUERY REWRITING INSPECTOR */}
          {activeInspectorTab === 'rewriting' && (
            <div className="space-y-3">
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-2">
                <span className="font-semibold text-slate-300 text-xs block">
                  Analyse de la Réécriture de Requête
                </span>

                <div className="p-2 rounded bg-slate-900 border border-slate-800 space-y-1">
                  <span className="text-[10px] font-mono text-slate-500 block">Question Utilisateur Initiale</span>
                  <p className="text-slate-300 text-xs">
                    {rewritePreview?.original_query || latestTurnMeta?.rewritten_query_info?.original_query || "Aucune requête active."}
                  </p>
                </div>

                <div className="p-2 rounded bg-indigo-950/40 border border-indigo-800/40 space-y-1">
                  <span className="text-[10px] font-mono text-indigo-400 block">Requête Autonome Envoyée au RAG</span>
                  <p className="text-indigo-200 text-xs font-medium">
                    {rewritePreview?.rewritten_query || latestTurnMeta?.rewritten_query_info?.rewritten_query || "Identique à l'originale."}
                  </p>
                </div>

                {/* Detected Coreferences */}
                <div className="space-y-1 pt-1">
                  <span className="text-[10px] font-mono text-slate-400">Références Détectées :</span>
                  <div className="flex flex-wrap gap-1">
                    {(rewritePreview?.detected_references || latestTurnMeta?.rewritten_query_info?.detected_references || []).map((ref, i) => (
                      <span key={i} className="px-1.5 py-0.5 rounded bg-purple-950 text-purple-300 font-mono text-[10px] border border-purple-800/40">
                        {ref}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Ambiguity Alert */}
                {(rewritePreview?.ambiguity_detected || latestTurnMeta?.rewritten_query_info?.ambiguity_detected) && (
                  <div className="p-2 rounded bg-amber-950/40 border border-amber-800/50 flex items-center gap-2 text-amber-300 text-xs">
                    <AlertTriangle className="w-4 h-4 shrink-0 text-amber-400" />
                    <span>Ambiguïté détectée dans les entités antérieures. Avertissement levé.</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* TAB 3: PIPELINE TELEMETRY */}
          {activeInspectorTab === 'telemetry' && (
            <div className="space-y-3">
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-2">
                <span className="font-semibold text-slate-300 text-xs block">
                  Décomposition de Latence (Waterfall)
                </span>
                {latestTurnMeta?.timing_breakdown ? (
                  <div className="space-y-1.5 font-mono text-xs">
                    {Object.entries(latestTurnMeta.timing_breakdown).map(([k, v]) => (
                      <div key={k} className="flex justify-between items-center text-slate-400">
                        <span className="text-[11px] truncate">{k.replace(/_/g, ' ')}</span>
                        <span className="text-slate-200 text-[11px]">{v} ms</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-slate-500 text-center py-4 text-xs">
                    Exécutez un tour pour afficher la télémétrie.
                  </div>
                )}
              </div>

              {/* Retrieved Sources Accordion */}
              {latestTurnMeta?.sources && (
                <div className="space-y-2">
                  <span className="font-semibold text-slate-300 text-xs block">
                    Sources Documentaires Retrouvées ({latestTurnMeta.sources.length})
                  </span>
                  {latestTurnMeta.sources.map((s, idx) => (
                    <div
                      key={idx}
                      className="p-2.5 rounded-lg bg-slate-950 border border-slate-800 hover:border-slate-700 transition-colors"
                    >
                      <div className="flex items-center justify-between text-[10px] font-mono text-indigo-400 mb-1">
                        <span>[S{s.citation_index}] {s.document_name}</span>
                        <span>Score: {s.similarity_score?.toFixed(4)}</span>
                      </div>
                      <p className="text-slate-300 text-[11px] line-clamp-3">
                        {s.content}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* TAB 4: STRATEGY COMPARISON MODE */}
          {activeInspectorTab === 'comparison' && (
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="font-semibold text-slate-300 text-xs">Comparatif Multi-Stratégies</span>
                <button
                  onClick={() => handleRunComparison()}
                  disabled={loadingComparison}
                  className="px-2 py-1 text-xs font-medium rounded bg-indigo-600 hover:bg-indigo-500 text-white flex items-center gap-1 shadow-sm"
                >
                  <RefreshCw className={`w-3 h-3 ${loadingComparison ? 'animate-spin' : ''}`} />
                  <span>Tester le Tour</span>
                </button>
              </div>

              {strategyComparison.length > 0 ? (
                <div className="space-y-2.5">
                  {strategyComparison.map((comp, idx) => (
                    <div key={idx} className="p-2.5 rounded-lg bg-slate-950 border border-slate-800 space-y-1.5">
                      <div className="flex items-center justify-between">
                        <span className="font-semibold text-xs text-indigo-300">
                          {comp.strategy.replace('_', ' ').toUpperCase()}
                        </span>
                        <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-slate-800 text-slate-400">
                          {comp.latency_ms} ms &bull; {comp.memory_tokens} tok
                        </span>
                      </div>
                      <div className="text-[11px] text-slate-300">
                        <span className="text-slate-500">Requête:</span> "{comp.rewritten_query}"
                      </div>
                      <div className="text-[10px] font-mono text-slate-400">
                        Chunks: [{comp.retrieved_chunk_sequences?.join(', ') || 'aucun'}]
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-slate-500 text-center py-6 text-xs">
                  Cliquez sur "Tester le Tour" pour exécuter les 4 stratégies de mémoire en parallèle sur l'historique courant.
                </div>
              )}
            </div>
          )}

          {/* TAB 5: MULTI-TURN BENCHMARK */}
          {activeInspectorTab === 'benchmark' && (
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="font-semibold text-slate-300 text-xs">Benchmark Multi-Tours (12 Scénarios)</span>
                <button
                  onClick={handleRunBenchmark}
                  disabled={loadingBenchmark}
                  className="px-2 py-1 text-xs font-medium rounded bg-indigo-600 hover:bg-indigo-500 text-white flex items-center gap-1 shadow-sm"
                >
                  <Flame className={`w-3 h-3 ${loadingBenchmark ? 'animate-spin' : ''}`} />
                  <span>Lancer</span>
                </button>
              </div>

              {benchmarkReport ? (
                <div className="space-y-2.5">
                  <div className="grid grid-cols-2 gap-2 text-center">
                    <div className="p-2 rounded bg-slate-950 border border-slate-800">
                      <span className="text-[10px] font-mono text-slate-500 block">Taux de Réussite</span>
                      <span className="text-sm font-bold text-emerald-400">
                        {(benchmarkReport.pass_rate * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="p-2 rounded bg-slate-950 border border-slate-800">
                      <span className="text-[10px] font-mono text-slate-500 block">Rappel Moyen @ 5</span>
                      <span className="text-sm font-bold text-indigo-400">
                        {benchmarkReport.mean_recall_at_5?.toFixed(4)}
                      </span>
                    </div>
                  </div>

                  <div className="space-y-1.5 max-h-72 overflow-y-auto">
                    {benchmarkReport.results.map((r, i) => (
                      <div key={i} className="p-2 rounded bg-slate-950 border border-slate-800 text-[11px] space-y-0.5">
                        <div className="flex items-center justify-between">
                          <span className="font-semibold text-slate-300">{r.scenario_id}: {r.name}</span>
                          <span className={`px-1 py-0.2 rounded font-mono text-[9px] ${r.passed ? 'bg-emerald-950 text-emerald-400' : 'bg-rose-950 text-rose-400'}`}>
                            {r.passed ? 'SUCCÈS' : 'ÉCHEC'}
                          </span>
                        </div>
                        <p className="text-slate-400 text-[10px] truncate">{r.original_query}</p>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="text-slate-500 text-center py-6 text-xs">
                  Lancez le benchmark pour évaluer la résolution de coréférences, l'injection historique, et le rappel documentaire sur 12 cas complexes.
                </div>
              )}
            </div>
          )}

          {/* TAB 6: LEARNING CENTER */}
          {activeInspectorTab === 'learning' && (
            <div className="space-y-3 text-slate-300 leading-relaxed text-xs">
              <div className="p-3 rounded-lg bg-indigo-950/20 border border-indigo-800/40 space-y-1.5">
                <span className="font-semibold text-indigo-300 block">1. RAG Sans État vs Conversationnel</span>
                <p className="text-slate-400 text-[11px]">
                  En RAG sans état (Phase 06), chaque question est traitée isolément. Dès qu'un utilisateur utilise un pronom (*« Où a-t-il étudié ? »*), la recherche dense échoue faute de contexte d'entité.
                </p>
              </div>

              <div className="p-3 rounded-lg bg-indigo-950/20 border border-indigo-800/40 space-y-1.5">
                <span className="font-semibold text-indigo-300 block">2. Réécriture Autonome de Requête</span>
                <p className="text-slate-400 text-[11px]">
                  Un moteur dédié analyse l'historique récent et reformule la question en une requête autonome (*« Où Yassir a-t-il obtenu son diplôme d'ingénieur ? »*), préservant la précision du moteur Phase 08.
                </p>
              </div>

              <div className="p-3 rounded-lg bg-indigo-950/20 border border-indigo-800/40 space-y-1.5">
                <span className="font-semibold text-indigo-300 block">3. Stratégies de Mémoire</span>
                <p className="text-slate-400 text-[11px]">
                  &bull; <strong>Fenêtre Glissante :</strong> Garde les $K$ derniers tours.<br />
                  &bull; <strong>Budget Borné :</strong> Ajuste dynamiquement l'historique selon les tokens restants.<br />
                  &bull; <strong>Mémoire Résumé :</strong> Compresse les vieux tours via LLM sans perte d'entités.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ========================================================= */}
      {/* ADVANCED SETTINGS MODAL */}
      {/* ========================================================= */}
      {showSettingsModal && (
        <div className="fixed inset-0 z-50 bg-black/70 flex items-center justify-center p-4">
          <div className="w-full max-w-md bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <Sliders className="w-4 h-4 text-indigo-400" />
                <span className="font-semibold text-sm text-slate-100">
                  Paramètres de Session & Mémoire
                </span>
              </div>
              <button
                onClick={() => setShowSettingsModal(false)}
                className="text-slate-400 hover:text-slate-200"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-3 text-xs">
              <div>
                <label className="text-slate-400 block mb-1">Stratégie de Mémoire</label>
                <select
                  value={activeStrategy}
                  onChange={(e) => setActiveStrategy(e.target.value)}
                  className="w-full p-2 bg-slate-950 border border-slate-800 rounded text-slate-200"
                >
                  <option value="sliding_window">Fenêtre Glissante (Sliding Window)</option>
                  <option value="token_bounded">Budget Borné (Token-Bounded)</option>
                  <option value="summary_memory">Mémoire Résumé (Summary Memory)</option>
                  <option value="no_memory">Sans Mémoire (Stateless)</option>
                </select>
              </div>

              {activeStrategy === 'sliding_window' && (
                <div>
                  <div className="flex justify-between text-slate-400 mb-1">
                    <span>Nombre de Tours (K) : {windowTurns}</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="10"
                    value={windowTurns}
                    onChange={(e) => setWindowTurns(Number(e.target.value))}
                    className="w-full accent-indigo-500"
                  />
                  <label className="flex items-center gap-2 mt-2 text-slate-300 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={includeAssistant}
                      onChange={(e) => setIncludeAssistant(e.target.checked)}
                      className="accent-indigo-500"
                    />
                    <span>Inclure les réponses de l'assistant dans la mémoire</span>
                  </label>
                </div>
              )}

              <div>
                <div className="flex justify-between text-slate-400 mb-1">
                  <span>Plafond Tokens Mémoire : {maxMemoryTokens}</span>
                </div>
                <input
                  type="range"
                  min="200"
                  max="3000"
                  step="100"
                  value={maxMemoryTokens}
                  onChange={(e) => setMaxMemoryTokens(Number(e.target.value))}
                  className="w-full accent-indigo-500"
                />
              </div>

              <div>
                <label className="text-slate-400 block mb-1">Moteur de Retrieval (Phase 08)</label>
                <select
                  value={retrievalStrategy}
                  onChange={(e) => setRetrievalStrategy(e.target.value)}
                  className="w-full p-2 bg-slate-950 border border-slate-800 rounded text-slate-200"
                >
                  <option value="hybrid_rrf">Recherche Hybride RRF (k=60)</option>
                  <option value="dense_only">Dense Seul (all-MiniLM-L6-v2)</option>
                  <option value="lexical_only">Lexical Seul (PostgreSQL tsvector)</option>
                  <option value="dense_rerank">Dense + Rerank TinyBERT</option>
                  <option value="hybrid_rrf_rerank">Hybride RRF + Rerank TinyBERT</option>
                  <option value="multi_query_hybrid_rerank">Multi-Query + Hybride RRF + Rerank</option>
                  <option value="adaptive_context">Adaptive Context (+-1 Voisins)</option>
                </select>
              </div>

              <div>
                <label className="text-slate-400 block mb-1">Fournisseur LLM</label>
                <select
                  value={llmProvider}
                  onChange={(e) => setLlmProvider(e.target.value)}
                  className="w-full p-2 bg-slate-950 border border-slate-800 rounded text-slate-200"
                >
                  <option value="mock">Simulateur Local (Mock LLM)</option>
                  <option value="gemini">Google Gemini API</option>
                  <option value="openai">OpenAI API</option>
                  <option value="ollama">Ollama Local</option>
                </select>
              </div>

              <div className="pt-2">
                <label className="flex items-center gap-2 text-slate-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={enableRewriting}
                    onChange={(e) => setEnableRewriting(e.target.checked)}
                    className="accent-indigo-500"
                  />
                  <span>Activer la réécriture de requête autonome (History-Aware)</span>
                </label>
              </div>
            </div>

            <div className="flex justify-end gap-2 pt-3 border-t border-slate-800">
              <button
                onClick={() => setShowSettingsModal(false)}
                className="px-3 py-1.5 rounded bg-slate-800 text-slate-300 hover:bg-slate-700 text-xs"
              >
                Annuler
              </button>
              <button
                onClick={handleUpdateSettings}
                className="px-3 py-1.5 rounded bg-indigo-600 text-white hover:bg-indigo-500 text-xs font-medium"
              >
                Appliquer
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
export default ConversationalLab;
