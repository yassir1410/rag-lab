import React, { useState, useEffect } from 'react';
import {
  Bot,
  Play,
  RotateCcw,
  Sparkles,
  Layers,
  Wrench,
  Search,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Cpu,
  ShieldAlert,
  ArrowRight,
  Sliders,
  ExternalLink,
  ChevronRight,
  Database,
  FileText,
  Activity,
  ListTree,
  Terminal,
  BookOpen,
} from 'lucide-react';
import { api } from '../services/api';

const SAMPLE_QUESTIONS = [
  { label: 'Single-Hop Fact', q: "What is the candidate's primary programming language?" },
  { label: 'Multi-Hop Technical', q: "Which company did the candidate work at in 2022, and what real-time messaging technology was deployed there?" },
  { label: 'Document Metadata', q: "How many pages and chunks does the candidate's resume have?" },
  { label: 'Decomposition & Synthesis', q: "Compare the candidate's university education with their professional backend engineering skills." },
  { label: 'Unanswerable', q: "What is the candidate's favorite musical artist?" },
];

export function AgenticLab() {
  const [activeTab, setActiveTab] = useState('playground'); // playground | timeline | tools | compare | traces | learning

  // State for Playground
  const [question, setQuestion] = useState(SAMPLE_QUESTIONS[1].q);
  const [strategy, setStrategy] = useState('iterative');
  const [providerId, setProviderId] = useState('mock');
  const [modelId, setModelId] = useState('mock-instruct-v1');
  const [maxIterations, setMaxIterations] = useState(4);
  const [maxToolCalls, setMaxToolCalls] = useState(8);
  const [timeBudget, setTimeBudget] = useState(30);

  // Runtime Execution State
  const [isRunning, setIsRunning] = useState(false);
  const [liveSteps, setLiveSteps] = useState([]);
  const [streamedAnswer, setStreamedAnswer] = useState('');
  const [activeEvidence, setActiveEvidence] = useState([]);
  const [activeCitations, setActiveCitations] = useState([]);
  const [lastTraceId, setLastTraceId] = useState(null);
  const [execMetadata, setExecMetadata] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  // Tools & Strategies Catalogs
  const [toolsList, setToolsList] = useState([]);
  const [strategiesList, setStrategiesList] = useState([]);

  // Comparison State
  const [compareResults, setCompareResults] = useState(null);
  const [isComparing, setIsComparing] = useState(false);

  // Trace Inspector State
  const [inspectTraceId, setInspectTraceId] = useState('');
  const [inspectedTrace, setInspectedTrace] = useState(null);
  const [isInspecting, setIsInspecting] = useState(false);

  // Benchmark State
  const [benchmarkReport, setBenchmarkReport] = useState(null);
  const [isBenchmarking, setIsBenchmarking] = useState(false);

  useEffect(() => {
    // Load tools and strategies
    api.listAgentStrategies().then(res => setStrategiesList(res || [])).catch(() => {});
    api.listAgentTools().then(res => setToolsList(res || [])).catch(() => {});
  }, []);

  const handleRunAgent = async () => {
    if (!question.trim()) return;
    setIsRunning(true);
    setLiveSteps([]);
    setStreamedAnswer('');
    setActiveEvidence([]);
    setActiveCitations([]);
    setErrorMsg(null);
    setExecMetadata(null);

    const payload = {
      question,
      strategy,
      provider_id: providerId,
      model_id: modelId,
      max_iterations: Number(maxIterations),
      max_tool_calls: Number(maxToolCalls),
      time_budget_seconds: Number(timeBudget),
    };

    try {
      await api.streamAgentQuery(
        payload,
        (eventType, data) => {
          if (eventType === 'meta') {
            setExecMetadata(data);
          } else if (eventType === 'step') {
            setLiveSteps(prev => [...prev, data]);
          } else if (eventType === 'token') {
            setStreamedAnswer(prev => prev + (data.token || ''));
          } else if (eventType === 'citations') {
            setActiveCitations(data.citations || []);
          } else if (eventType === 'done') {
            setLastTraceId(data.trace_id);
            setInspectTraceId(data.trace_id);
            setExecMetadata(prev => ({ ...prev, ...data }));
            setIsRunning(false);
          } else if (eventType === 'error') {
            setErrorMsg(data.error || 'Agent stream error occurred.');
            setIsRunning(false);
          }
        },
        (err) => {
          setErrorMsg(err.message || 'Stream connection failed.');
          setIsRunning(false);
        }
      );
    } catch (e) {
      setErrorMsg(e.message);
      setIsRunning(false);
    }
  };

  const handleRunComparison = async () => {
    setIsComparing(true);
    setCompareResults(null);
    try {
      const res = await api.compareAgentStrategies({
        question,
        provider_id: providerId,
        model_id: modelId,
        max_iterations: Number(maxIterations),
        max_tool_calls: Number(maxToolCalls),
      });
      setCompareResults(res);
    } catch (e) {
      setErrorMsg(e.message);
    } finally {
      setIsComparing(false);
    }
  };

  const handleInspectTrace = async () => {
    if (!inspectTraceId.trim()) return;
    setIsInspecting(true);
    setInspectedTrace(null);
    try {
      const data = await api.getAgentTrace(inspectTraceId.trim());
      setInspectedTrace(data);
    } catch (e) {
      setErrorMsg(e.message);
    } finally {
      setIsInspecting(false);
    }
  };

  const handleRunBenchmark = async () => {
    setIsBenchmarking(true);
    setBenchmarkReport(null);
    try {
      const report = await api.runAgentBenchmark({
        strategy,
        provider_id: providerId,
        model_id: modelId,
      });
      setBenchmarkReport(report);
    } catch (e) {
      setErrorMsg(e.message);
    } finally {
      setIsBenchmarking(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col h-screen overflow-hidden bg-slate-950 text-slate-100 font-sans">
      {/* Top Header */}
      <header className="h-14 border-b border-slate-800 px-6 flex items-center justify-between bg-slate-900/60 shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-indigo-600/30 border border-indigo-500/40 flex items-center justify-center text-indigo-400">
            <Bot className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
              Agentic RAG Laboratory
              <span className="text-[10px] px-2 py-0.5 rounded bg-indigo-950 text-indigo-400 border border-indigo-800/40 font-mono">
                Phase 10 &bull; Ready
              </span>
            </h1>
            <p className="text-[11px] text-slate-400">
              Constrained Plan-Act-Observe tool-orchestrated multi-hop retrieval
            </p>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs">
          {[
            { id: 'playground', label: 'Agent Playground', icon: Bot },
            { id: 'timeline', label: 'Agent Timeline', icon: Activity },
            { id: 'tools', label: 'Tool Explorer', icon: Wrench },
            { id: 'compare', label: 'Strategy Comparison', icon: Layers },
            { id: 'traces', label: 'Trace Inspector', icon: Terminal },
            { id: 'benchmark', label: '15-Scenario Suite', icon: CheckCircle2 },
            { id: 'learning', label: 'Learning Center', icon: BookOpen },
          ].map(tab => {
            const Icon = tab.icon;
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition-colors ${
                  active
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </header>

      {/* Main Content Area */}
      <div className="flex-1 overflow-y-auto p-6">
        {errorMsg && (
          <div className="mb-4 p-3 bg-rose-950/60 border border-rose-800/50 rounded-lg text-xs text-rose-300 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
              <span>{errorMsg}</span>
            </div>
            <button onClick={() => setErrorMsg(null)} className="text-slate-400 hover:text-slate-200">
              &times;
            </button>
          </div>
        )}

        {/* TAB 1: PLAYGROUND */}
        {activeTab === 'playground' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left Controls & Query Input (5 cols) */}
            <div className="lg:col-span-5 space-y-4">
              <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 space-y-4">
                <div className="flex items-center justify-between">
                  <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                    <Sliders className="w-3.5 h-3.5 text-indigo-400" />
                    Agent Configuration
                  </h2>
                  <span className="text-[10px] text-emerald-400 font-mono">SANDBOXED READ-ONLY</span>
                </div>

                {/* Preset Scenarios */}
                <div className="space-y-1.5">
                  <label className="text-[11px] text-slate-400">Preset Scenarios</label>
                  <div className="flex flex-wrap gap-1.5">
                    {SAMPLE_QUESTIONS.map((s, idx) => (
                      <button
                        key={idx}
                        onClick={() => setQuestion(s.q)}
                        className="text-[10px] px-2 py-1 rounded bg-slate-800/80 hover:bg-slate-700 text-slate-300 border border-slate-700/50 transition-colors"
                      >
                        {s.label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Question Input */}
                <div className="space-y-1.5">
                  <label className="text-[11px] text-slate-400 font-medium">User Question</label>
                  <textarea
                    rows={3}
                    value={question}
                    onChange={e => setQuestion(e.target.value)}
                    placeholder="Enter your query for the agent..."
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-xs text-slate-100 placeholder-slate-600 focus:outline-none focus:border-indigo-500 font-sans"
                  />
                </div>

                {/* Strategy Selector */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1">
                    <label className="text-[11px] text-slate-400 font-medium">Agent Strategy</label>
                    <select
                      value={strategy}
                      onChange={e => setStrategy(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
                    >
                      <option value="single_step">Single-Step (Tool Baseline)</option>
                      <option value="iterative">Iterative (Plan-Act-Observe)</option>
                      <option value="plan_execute">Plan-and-Execute (Decompose)</option>
                      <option value="adaptive">Adaptive Routing</option>
                    </select>
                  </div>

                  <div className="space-y-1">
                    <label className="text-[11px] text-slate-400 font-medium">LLM Provider</label>
                    <select
                      value={providerId}
                      onChange={e => {
                        setProviderId(e.target.value);
                        setModelId(e.target.value === 'mock' ? 'mock-instruct-v1' : 'gpt-4o-mini');
                      }}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
                    >
                      <option value="mock">Mock LLM (Deterministic)</option>
                      <option value="gemini">Google Gemini</option>
                      <option value="openai">OpenAI (Cloud)</option>
                      <option value="ollama">Ollama (Local)</option>
                    </select>
                  </div>
                </div>

                {/* Bounded Sliders */}
                <div className="space-y-3 pt-2 border-t border-slate-800">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-slate-400">Max Iterations:</span>
                    <span className="font-mono text-indigo-400 font-semibold">{maxIterations} loops</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="8"
                    value={maxIterations}
                    onChange={e => setMaxIterations(e.target.value)}
                    className="w-full accent-indigo-500 h-1.5 bg-slate-950 rounded-lg cursor-pointer"
                  />

                  <div className="flex items-center justify-between text-xs">
                    <span className="text-slate-400">Max Tool Calls:</span>
                    <span className="font-mono text-indigo-400 font-semibold">{maxToolCalls} calls</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="15"
                    value={maxToolCalls}
                    onChange={e => setMaxToolCalls(e.target.value)}
                    className="w-full accent-indigo-500 h-1.5 bg-slate-950 rounded-lg cursor-pointer"
                  />
                </div>

                {/* Run Button */}
                <button
                  onClick={handleRunAgent}
                  disabled={isRunning || !question.trim()}
                  className={`w-full py-2.5 rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition-all ${
                    isRunning
                      ? 'bg-indigo-900/50 text-indigo-300 border border-indigo-700/50 cursor-wait'
                      : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-md shadow-indigo-600/30'
                  }`}
                >
                  {isRunning ? (
                    <>
                      <RotateCcw className="w-4 h-4 animate-spin" />
                      <span>Executing Plan-Act-Observe Loop...</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 fill-white" />
                      <span>Launch Agentic Retrieval</span>
                    </>
                  )}
                </button>
              </div>

              {/* Telemetry Snapshot */}
              {execMetadata && (
                <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-3 grid grid-cols-3 gap-2 text-center text-xs">
                  <div>
                    <span className="text-[10px] text-slate-500 block">Total Steps</span>
                    <span className="font-mono font-bold text-slate-200">{execMetadata.total_steps || liveSteps.length}</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 block">Tool Calls</span>
                    <span className="font-mono font-bold text-indigo-400">{execMetadata.tool_calls_count || liveSteps.filter(s => s.action_type === 'TOOL_CALL').length}</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 block">Latency</span>
                    <span className="font-mono font-bold text-emerald-400">{execMetadata.latency_ms ? `${Math.round(execMetadata.latency_ms)} ms` : 'Streaming...'}</span>
                  </div>
                </div>
              )}
            </div>

            {/* Right Output: Plan-Act-Observe Stream + Grounded Answer (7 cols) */}
            <div className="lg:col-span-7 space-y-4">
              {/* Grounded Cited Answer Box */}
              <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-3">
                <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
                  <div className="flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-emerald-400" />
                    <h3 className="text-xs font-semibold text-slate-200">Grounded Agent Synthesis</h3>
                  </div>
                  {lastTraceId && (
                    <button
                      onClick={() => {
                        setInspectTraceId(lastTraceId);
                        setActiveTab('traces');
                      }}
                      className="text-[10px] text-indigo-400 hover:text-indigo-300 font-mono flex items-center gap-1"
                    >
                      Trace ID: {lastTraceId.slice(0, 8)}...
                      <ExternalLink className="w-2.5 h-2.5" />
                    </button>
                  )}
                </div>

                {/* Answer Content */}
                <div className="min-h-[140px] text-xs leading-relaxed text-slate-200 font-sans whitespace-pre-wrap">
                  {streamedAnswer || (
                    <span className="text-slate-600 italic">
                      Agent answer will stream here token-by-token after multi-step retrieval completes...
                    </span>
                  )}
                </div>

                {/* Citations Grid */}
                {activeCitations.length > 0 && (
                  <div className="pt-3 border-t border-slate-800/80 space-y-2">
                    <div className="text-[11px] font-semibold text-slate-400 flex items-center gap-1.5">
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                      Grounded Citations ({activeCitations.length})
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                      {activeCitations.map((cit, idx) => (
                        <div
                          key={idx}
                          className="bg-slate-950/80 border border-slate-800/80 rounded-lg p-2.5 text-[11px] space-y-1 hover:border-slate-700 transition-colors"
                        >
                          <div className="flex items-center justify-between font-mono">
                            <span className="text-emerald-400 font-bold">{cit.source_id}</span>
                            <span className="text-[10px] text-slate-500">{cit.retrieved_by_tool}</span>
                          </div>
                          <div className="text-slate-300 font-medium truncate">{cit.document_name}</div>
                          <div className="text-[10px] text-slate-400 line-clamp-2 italic">
                            "{cit.snippet}"
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Plan-Act-Observe Live Feed */}
              <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4 space-y-3">
                <div className="flex items-center justify-between text-xs">
                  <h3 className="font-semibold text-slate-300 flex items-center gap-2">
                    <Activity className="w-4 h-4 text-indigo-400" />
                    Observable Step Trace
                  </h3>
                  <span className="text-[10px] text-slate-500">Live SSE Events</span>
                </div>

                <div className="space-y-2 max-h-[360px] overflow-y-auto pr-1">
                  {liveSteps.length === 0 ? (
                    <div className="text-center py-8 text-xs text-slate-600">
                      No steps executed yet. Launch the agent above to see real-time tool calls.
                    </div>
                  ) : (
                    liveSteps.map((step, idx) => (
                      <div
                        key={idx}
                        className="bg-slate-950/70 border border-slate-800/70 rounded-lg p-3 space-y-2 text-xs"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <span className="w-5 h-5 rounded-full bg-slate-800 text-slate-300 flex items-center justify-center font-mono text-[10px] font-bold">
                              {step.step_number}
                            </span>
                            <span
                              className={`text-[10px] px-2 py-0.5 rounded font-mono font-semibold ${
                                step.action_type === 'PLAN'
                                  ? 'bg-amber-950 text-amber-400 border border-amber-800/40'
                                  : step.action_type === 'TOOL_CALL'
                                  ? 'bg-indigo-950 text-indigo-400 border border-indigo-800/40'
                                  : 'bg-emerald-950 text-emerald-400 border border-emerald-800/40'
                              }`}
                            >
                              {step.action_type}
                            </span>
                            {step.tool_call && (
                              <span className="text-slate-300 font-mono text-[11px] font-semibold">
                                {step.tool_call.tool_name}()
                              </span>
                            )}
                          </div>
                          <span className="text-[10px] text-slate-500 font-mono">
                            {step.latency_ms ? `${Math.round(step.latency_ms)} ms` : step.timestamp}
                          </span>
                        </div>

                        {step.thought && (
                          <div className="text-[11px] text-slate-400 italic bg-slate-900/60 p-2 rounded border border-slate-800/40">
                            &ldquo;{step.thought}&rdquo;
                          </div>
                        )}

                        {step.tool_call?.arguments && (
                          <div className="text-[10px] font-mono text-slate-400 bg-slate-950 p-2 rounded border border-slate-800/60 overflow-x-auto">
                            args: {JSON.stringify(step.tool_call.arguments)}
                          </div>
                        )}

                        {step.tool_result && (
                          <div className="flex items-center justify-between text-[10px] text-slate-400 pt-1">
                            <span className="flex items-center gap-1 text-emerald-400">
                              <CheckCircle2 className="w-3 h-3" />
                              {step.tool_result.raw_preview || `${step.tool_result.items_count} items retrieved`}
                            </span>
                            {step.evidence_added?.length > 0 && (
                              <span className="text-indigo-400 font-mono font-semibold">
                                + {step.evidence_added.join(', ')}
                              </span>
                            )}
                          </div>
                        )}
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: AGENT TIMELINE */}
        {activeTab === 'timeline' && (
          <div className="max-w-4xl mx-auto space-y-4">
            <div className="flex items-center justify-between bg-slate-900/60 p-4 rounded-xl border border-slate-800">
              <div>
                <h2 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
                  <Activity className="w-4 h-4 text-indigo-400" />
                  Plan-Act-Observe Execution Flow
                </h2>
                <p className="text-xs text-slate-400">
                  Step-by-step breakdown of agent decisions, tool invocations, and evidence accumulation.
                </p>
              </div>
              <span className="text-xs font-mono text-slate-400">
                {liveSteps.length} Steps Recorded
              </span>
            </div>

            {liveSteps.length === 0 ? (
              <div className="text-center py-16 bg-slate-900/30 rounded-xl border border-slate-800/60 text-xs text-slate-500">
                No active execution timeline to display. Run a query in the Agent Playground first.
              </div>
            ) : (
              <div className="relative pl-6 space-y-6 before:absolute before:left-3 before:top-2 before:bottom-2 before:w-0.5 before:bg-slate-800">
                {liveSteps.map((step, idx) => (
                  <div key={idx} className="relative group">
                    <div className="absolute -left-6 top-1.5 w-3 h-3 rounded-full bg-indigo-500 ring-4 ring-slate-950" />
                    <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-4 space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-slate-200">
                          Step #{step.step_number}: {step.action_type}
                        </span>
                        <span className="text-[10px] text-emerald-400 font-mono font-medium">
                          {step.latency_ms ? `${Math.round(step.latency_ms)} ms` : ''}
                        </span>
                      </div>
                      <p className="text-xs text-slate-300">{step.thought}</p>
                      {step.tool_call && (
                        <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 space-y-1 font-mono text-xs">
                          <div className="text-indigo-400 font-bold">{step.tool_call.tool_name}()</div>
                          <div className="text-slate-400 text-[11px]">{JSON.stringify(step.tool_call.arguments, null, 2)}</div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* TAB 3: TOOL EXPLORER */}
        {activeTab === 'tools' && (
          <div className="max-w-5xl mx-auto space-y-4">
            <div className="bg-slate-900/60 p-4 rounded-xl border border-slate-800 flex items-center justify-between">
              <div>
                <h2 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
                  <Wrench className="w-4 h-4 text-indigo-400" />
                  Approved Read-Only Retrieval Tools
                </h2>
                <p className="text-xs text-slate-400">
                  Allowlisted tools available to the Agent Controller with typed JSON schemas.
                </p>
              </div>
              <span className="text-[10px] px-2.5 py-1 rounded bg-emerald-950 text-emerald-400 font-mono border border-emerald-800/40">
                5 STRICTLY READ-ONLY TOOLS
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {toolsList.map((tool, idx) => (
                <div
                  key={idx}
                  className="bg-slate-900/40 border border-slate-800 rounded-xl p-4 space-y-3 flex flex-col justify-between"
                >
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-xs font-bold text-indigo-300">
                        {tool.name}()
                      </span>
                      <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-400 font-mono">
                        {tool.category}
                      </span>
                    </div>
                    <p className="text-xs text-slate-400 leading-relaxed">
                      {tool.description}
                    </p>
                  </div>

                  <div className="space-y-1 pt-2 border-t border-slate-800/80">
                    <span className="text-[10px] uppercase tracking-wider text-slate-500 font-mono font-semibold">
                      Parameters Schema
                    </span>
                    <pre className="text-[10px] font-mono text-slate-300 bg-slate-950 p-2 rounded-lg border border-slate-800/60 overflow-x-auto max-h-36">
                      {JSON.stringify(tool.parameters_schema?.properties || {}, null, 2)}
                    </pre>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* TAB 4: STRATEGY COMPARISON */}
        {activeTab === 'compare' && (
          <div className="max-w-5xl mx-auto space-y-4">
            <div className="bg-slate-900/60 p-4 rounded-xl border border-slate-800 flex items-center justify-between">
              <div>
                <h2 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
                  <Layers className="w-4 h-4 text-indigo-400" />
                  Strategy Comparison Benchmark
                </h2>
                <p className="text-xs text-slate-400">
                  Compare Standard RAG vs Single-Step vs Iterative vs Plan-and-Execute vs Adaptive Routing.
                </p>
              </div>
              <button
                onClick={handleRunComparison}
                disabled={isComparing || !question.trim()}
                className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors"
              >
                {isComparing ? <RotateCcw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                <span>Compare Strategies</span>
              </button>
            </div>

            {compareResults ? (
              <div className="bg-slate-900/40 border border-slate-800 rounded-xl overflow-hidden">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-slate-800 bg-slate-950/80 text-[11px] text-slate-400 uppercase font-mono">
                      <th className="p-3">Strategy</th>
                      <th className="p-3">Latency</th>
                      <th className="p-3">Steps</th>
                      <th className="p-3">Tool Calls</th>
                      <th className="p-3">Evidence Items</th>
                      <th className="p-3">Citations</th>
                      <th className="p-3">Stop Reason</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60">
                    {Object.entries(compareResults.results || {}).map(([stratName, item], idx) => (
                      <tr key={idx} className="hover:bg-slate-900/40 transition-colors">
                        <td className="p-3 font-medium text-slate-200 capitalize font-mono">{stratName}</td>
                        <td className="p-3 font-mono text-emerald-400">{item.latency_ms} ms</td>
                        <td className="p-3 font-mono text-slate-300">{item.total_steps}</td>
                        <td className="p-3 font-mono text-indigo-400">{item.tool_calls_count}</td>
                        <td className="p-3 font-mono text-slate-300">{item.evidence_count}</td>
                        <td className="p-3 font-mono text-amber-400">{item.citation_count}</td>
                        <td className="p-3 font-mono text-[10px] text-slate-400">{item.stop_reason}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="text-center py-16 bg-slate-900/30 rounded-xl border border-slate-800/60 text-xs text-slate-500">
                Click &ldquo;Compare Strategies&rdquo; to execute the prompt across all four agentic strategies.
              </div>
            )}
          </div>
        )}

        {/* TAB 5: TRACE INSPECTOR */}
        {activeTab === 'traces' && (
          <div className="max-w-4xl mx-auto space-y-4">
            <div className="bg-slate-900/60 p-4 rounded-xl border border-slate-800 flex items-center justify-between gap-4">
              <div className="flex-1">
                <label className="text-[11px] text-slate-400 font-medium block mb-1">
                  Lookup Trace by ID
                </label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={inspectTraceId}
                    onChange={e => setInspectTraceId(e.target.value)}
                    placeholder="Enter UUID trace id (e.g. 550e8400-e29b-41d4-a716-446655440000)"
                    className="flex-1 bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-indigo-500"
                  />
                  <button
                    onClick={handleInspectTrace}
                    disabled={isInspecting || !inspectTraceId.trim()}
                    className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold transition-colors"
                  >
                    Inspect
                  </button>
                </div>
              </div>
            </div>

            {inspectedTrace ? (
              <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-5 space-y-4">
                <div className="grid grid-cols-4 gap-3 text-xs bg-slate-950 p-3 rounded-lg border border-slate-800/80">
                  <div>
                    <span className="text-[10px] text-slate-500 block">Strategy</span>
                    <span className="font-mono font-semibold text-indigo-400">{inspectedTrace.strategy}</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 block">Stop Reason</span>
                    <span className="font-mono font-semibold text-emerald-400">{inspectedTrace.stop_reason}</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 block">Tool Calls</span>
                    <span className="font-mono font-semibold text-slate-200">{inspectedTrace.tool_calls_count}</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 block">Latency</span>
                    <span className="font-mono font-semibold text-slate-200">{Math.round(inspectedTrace.latency_ms)} ms</span>
                  </div>
                </div>

                <div className="space-y-1">
                  <span className="text-[11px] font-semibold text-slate-400">User Question:</span>
                  <p className="text-xs text-slate-200 bg-slate-950 p-2.5 rounded border border-slate-800/60">
                    {inspectedTrace.question}
                  </p>
                </div>

                <div className="space-y-1">
                  <span className="text-[11px] font-semibold text-slate-400">Final Cited Answer:</span>
                  <p className="text-xs text-slate-200 bg-slate-950 p-2.5 rounded border border-slate-800/60 whitespace-pre-wrap">
                    {inspectedTrace.final_answer}
                  </p>
                </div>

                <div className="space-y-1">
                  <span className="text-[11px] font-semibold text-slate-400">Full JSON Trace:</span>
                  <pre className="text-[10px] font-mono text-slate-300 bg-slate-950 p-3 rounded-lg border border-slate-800/60 max-h-64 overflow-y-auto">
                    {JSON.stringify(inspectedTrace, null, 2)}
                  </pre>
                </div>
              </div>
            ) : (
              <div className="text-center py-16 bg-slate-900/30 rounded-xl border border-slate-800/60 text-xs text-slate-500">
                Enter a trace ID to inspect full Plan-Act-Observe steps and persistence logs.
              </div>
            )}
          </div>
        )}

        {/* TAB 6: 15-SCENARIO SUITE */}
        {activeTab === 'benchmark' && (
          <div className="max-w-5xl mx-auto space-y-4">
            <div className="bg-slate-900/60 p-4 rounded-xl border border-slate-800 flex items-center justify-between">
              <div>
                <h2 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  15-Scenario Synthetic Benchmark Suite
                </h2>
                <p className="text-xs text-slate-400">
                  Automated validation across multi-hop, unanswerable, distractors, and budget boundaries.
                </p>
              </div>
              <button
                onClick={handleRunBenchmark}
                disabled={isBenchmarking}
                className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors"
              >
                {isBenchmarking ? <RotateCcw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                <span>Run 15 Scenarios</span>
              </button>
            </div>

            {benchmarkReport ? (
              <div className="space-y-4">
                <div className="grid grid-cols-4 gap-3 bg-slate-900/60 p-4 rounded-xl border border-slate-800 text-center text-xs">
                  <div>
                    <span className="text-[10px] text-slate-500 block">Pass Rate</span>
                    <span className="font-mono text-lg font-bold text-emerald-400">
                      {benchmarkReport.passed_scenarios} / {benchmarkReport.total_scenarios}
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 block">Avg Latency</span>
                    <span className="font-mono text-lg font-bold text-slate-200">
                      {benchmarkReport.average_latency_ms} ms
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 block">Avg Tool Calls</span>
                    <span className="font-mono text-lg font-bold text-indigo-400">
                      {benchmarkReport.average_tool_calls}
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 block">Avg Evidence</span>
                    <span className="font-mono text-lg font-bold text-slate-200">
                      {benchmarkReport.average_evidence_count}
                    </span>
                  </div>
                </div>

                <div className="bg-slate-900/40 border border-slate-800 rounded-xl overflow-hidden">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead>
                      <tr className="border-b border-slate-800 bg-slate-950/80 text-[11px] text-slate-400 uppercase font-mono">
                        <th className="p-3">ID</th>
                        <th className="p-3">Scenario</th>
                        <th className="p-3">Result</th>
                        <th className="p-3">Tool Calls</th>
                        <th className="p-3">Citations</th>
                        <th className="p-3">Latency</th>
                        <th className="p-3">Stop Reason</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60">
                      {benchmarkReport.scenario_results.map((res, idx) => (
                        <tr key={idx} className="hover:bg-slate-900/40 transition-colors">
                          <td className="p-3 font-mono text-indigo-400 font-semibold">{res.scenario_id}</td>
                          <td className="p-3 text-slate-200">{res.title}</td>
                          <td className="p-3">
                            <span className={`text-[10px] px-2 py-0.5 rounded font-mono font-semibold ${
                              res.success ? 'bg-emerald-950 text-emerald-400 border border-emerald-800/40' : 'bg-rose-950 text-rose-400 border border-rose-800/40'
                            }`}>
                              {res.success ? 'PASSED' : 'FAILED'}
                            </span>
                          </td>
                          <td className="p-3 font-mono text-slate-300">{res.tool_calls_count}</td>
                          <td className="p-3 font-mono text-amber-400">{res.citations_count}</td>
                          <td className="p-3 font-mono text-emerald-400">{res.latency_ms} ms</td>
                          <td className="p-3 font-mono text-[10px] text-slate-400">{res.stop_reason}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : (
              <div className="text-center py-16 bg-slate-900/30 rounded-xl border border-slate-800/60 text-xs text-slate-500">
                Click &ldquo;Run 15 Scenarios&rdquo; to benchmark multi-hop accuracy and resilience.
              </div>
            )}
          </div>
        )}

        {/* TAB 7: LEARNING CENTER */}
        {activeTab === 'learning' && (
          <div className="max-w-4xl mx-auto space-y-6">
            <div className="bg-slate-900/60 p-5 rounded-xl border border-slate-800 space-y-2">
              <h2 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
                <BookOpen className="w-4 h-4 text-indigo-400" />
                Curriculum Guide: Agentic RAG & Tool-Orchestrated Retrieval
              </h2>
              <p className="text-xs text-slate-400 leading-relaxed">
                Understand how autonomous retrieval agents differ from standard pipelines, how tools are safely sandboxed,
                and why resource bounds are vital for production reliability.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4 space-y-2">
                <h3 className="text-xs font-semibold text-indigo-400 flex items-center gap-1.5">
                  <Bot className="w-3.5 h-3.5" />
                  Standard RAG vs. Agentic RAG
                </h3>
                <p className="text-xs text-slate-300 leading-relaxed">
                  In standard RAG, the query executes in a static, one-shot pipeline regardless of evidence completeness.
                  In Agentic RAG, the LLM actively observes search results, identifies missing facts, executes targeted follow-up tools,
                  and halts only when evidence is deemed sufficient.
                </p>
              </div>

              <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4 space-y-2">
                <h3 className="text-xs font-semibold text-emerald-400 flex items-center gap-1.5">
                  <ShieldAlert className="w-3.5 h-3.5" />
                  Tool Selection vs Arbitrary Code Execution
                </h3>
                <p className="text-xs text-slate-300 leading-relaxed">
                  Our agent is strictly restricted to an allowlist of 5 read-only retrieval tools. The LLM cannot write code,
                  execute shell commands, or perform arbitrary SQL mutations. Untrusted document contents cannot alter system privileges.
                </p>
              </div>

              <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4 space-y-2">
                <h3 className="text-xs font-semibold text-amber-400 flex items-center gap-1.5">
                  <Clock className="w-3.5 h-3.5" />
                  Halting Problem & Bounded Budgets
                </h3>
                <p className="text-xs text-slate-300 leading-relaxed">
                  Because LLM loops can theoretically iterate indefinitely when seeking unanswerable information, strict resource budgets
                  (Max Iterations, Max Tool Calls, Timeouts) enforce deterministic halting with explicit stop reasons.
                </p>
              </div>

              <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4 space-y-2">
                <h3 className="text-xs font-semibold text-sky-400 flex items-center gap-1.5">
                  <ListTree className="w-3.5 h-3.5" />
                  Multi-Hop Query Decomposition
                </h3>
                <p className="text-xs text-slate-300 leading-relaxed">
                  Plan-and-Execute agents break compound questions into independent sub-queries, retrieving targeted facts per hop
                  before merging them into a unified, cross-referenced documentary context envelope.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
