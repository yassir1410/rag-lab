import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
  Share2,
  Play,
  RotateCcw,
  Sparkles,
  Layers,
  Search,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Cpu,
  FileText,
  Activity,
  ListTree,
  Terminal,
  BookOpen,
  GitFork,
  Boxes,
  Compass,
  Maximize2,
  Minimize2,
  ExternalLink,
  ChevronRight,
  Filter,
  Check,
  X,
  Users,
} from 'lucide-react';
import { api } from '../services/api';

const SAMPLE_QUESTIONS = [
  { label: 'Single Entity', q: "What is the candidate's name and university?" },
  { label: 'Two-Hop Skills', q: "Which programming languages and backend frameworks does the candidate use?" },
  { label: 'Relational Chain', q: "Connect the candidate's degree from INSEA to their backend development skills in PostgreSQL." },
  { label: 'Alias Matching', q: "Does the candidate know how to write JS, TS and Postgres code?" },
  { label: 'Global Thematic', q: "Summarize the core technical expertise and tooling clusters." },
];

const ENTITY_TYPE_COLORS = {
  PERSON: { bg: 'bg-emerald-950', text: 'text-emerald-400', border: 'border-emerald-800/60', hex: '#10b981' },
  ORGANIZATION: { bg: 'bg-blue-950', text: 'text-blue-400', border: 'border-blue-800/60', hex: '#3b82f6' },
  PROJECT: { bg: 'bg-indigo-950', text: 'text-indigo-400', border: 'border-indigo-800/60', hex: '#6366f1' },
  TECHNOLOGY: { bg: 'bg-purple-950', text: 'text-purple-400', border: 'border-purple-800/60', hex: '#a855f7' },
  JOB_ROLE: { bg: 'bg-amber-950', text: 'text-amber-400', border: 'border-amber-800/60', hex: '#f59e0b' },
  LOCATION: { bg: 'bg-rose-950', text: 'text-rose-400', border: 'border-rose-800/60', hex: '#f43f5e' },
  CERTIFICATION: { bg: 'bg-teal-950', text: 'text-teal-400', border: 'border-teal-800/60', hex: '#14b8a6' },
  DEFAULT: { bg: 'bg-slate-900', text: 'text-slate-400', border: 'border-slate-800', hex: '#94a3b8' },
};

export function GraphRAGLab() {
  const [activeTab, setActiveTab] = useState('explorer'); // builder | explorer | resolution | query | communities | compare | learning

  // Core Data State
  const [documents, setDocuments] = useState([]);
  const [selectedDocId, setSelectedDocId] = useState('');
  const [graphStats, setGraphStats] = useState(null);
  const [entities, setEntities] = useState([]);
  const [relations, setRelations] = useState([]);
  const [communities, setCommunities] = useState([]);
  const [mergeCandidates, setMergeCandidates] = useState([]);

  // Builder State
  const [extractStrategy, setExtractStrategy] = useState('rule');
  const [isExtracting, setIsExtracting] = useState(false);
  const [lastExtractionRun, setLastExtractionRun] = useState(null);

  // Explorer Interactive State
  const [selectedNode, setSelectedNode] = useState(null);
  const [filterType, setFilterType] = useState('ALL');
  const [searchQuery, setSearchQuery] = useState('');
  const [zoomLevel, setZoomLevel] = useState(1);

  // Query Lab State
  const [question, setQuestion] = useState(SAMPLE_QUESTIONS[1].q);
  const [queryStrategy, setQueryStrategy] = useState('hybrid_graph_vector');
  const [maxHops, setMaxHops] = useState(2);
  const [isQuerying, setIsQuerying] = useState(false);
  const [queryResponse, setQueryResponse] = useState(null);

  // Compare State
  const [compareResults, setCompareResults] = useState(null);
  const [isComparing, setIsComparing] = useState(false);

  // Benchmark State
  const [benchmarkReport, setBenchmarkReport] = useState(null);
  const [isBenchmarking, setIsBenchmarking] = useState(false);

  // Error / Toast
  const [errorMsg, setErrorMsg] = useState(null);
  const [successMsg, setSuccessMsg] = useState(null);

  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      // 1. Documents
      const docs = await api.getDocuments();
      const docList = docs?.documents || [];
      setDocuments(docList);
      if (docList.length > 0 && !selectedDocId) {
        setSelectedDocId(docList[0].id);
      }

      // 2. Graph Stats
      const stats = await api.getGraphStats();
      setGraphStats(stats);

      // 3. Entities & Relations
      const ents = await api.listGraphEntities({ limit: 100 });
      setEntities(ents || []);

      const rels = await api.listGraphRelations({ limit: 100 });
      setRelations(rels || []);

      // 4. Communities
      const comms = await api.getGraphCommunities();
      setCommunities(comms || []);

      // 5. Merge Candidates
      const candidates = await api.getGraphMergeCandidates();
      setMergeCandidates(candidates || []);
    } catch (e) {
      console.error('Error loading initial graph data:', e);
    }
  };

  const handleRunExtraction = async () => {
    if (!selectedDocId) return;
    setIsExtracting(true);
    setErrorMsg(null);
    setSuccessMsg(null);
    try {
      const res = await api.extractGraph({
        document_id: selectedDocId,
        strategy: extractStrategy,
        model_id: 'mock-instruct-v1',
        provider_id: 'mock',
      });
      setLastExtractionRun(res);
      setSuccessMsg(`Graph built successfully! Extracted ${res.metrics?.unique_entities || 0} unique entities and ${res.metrics?.relations_extracted || 0} relations.`);
      await loadInitialData();
    } catch (e) {
      setErrorMsg(e.message);
    } finally {
      setIsExtracting(false);
    }
  };

  const handleRunQuery = async () => {
    if (!question.trim()) return;
    setIsQuerying(true);
    setQueryResponse(null);
    setErrorMsg(null);
    try {
      const res = await api.queryGraph({
        question,
        strategy: queryStrategy,
        max_hops: Number(maxHops),
        top_k: 5,
        provider_id: 'mock',
        model_id: 'mock-instruct-v1',
      });
      setQueryResponse(res);
    } catch (e) {
      setErrorMsg(e.message);
    } finally {
      setIsQuerying(false);
    }
  };

  const handleRunComparison = async () => {
    setIsComparing(true);
    setCompareResults(null);
    try {
      const res = await api.compareGraphStrategies({
        question,
        top_k: 5,
        provider_id: 'mock',
        model_id: 'mock-instruct-v1',
      });
      setCompareResults(res);
    } catch (e) {
      setErrorMsg(e.message);
    } finally {
      setIsComparing(false);
    }
  };

  const handleMergeEntities = async (survivingId, mergedId) => {
    try {
      await api.mergeGraphEntities({
        surviving_entity_id: survivingId,
        merged_entity_id: mergedId,
      });
      setSuccessMsg('Entities merged successfully! Relationships and evidence transferred.');
      await loadInitialData();
    } catch (e) {
      setErrorMsg(e.message);
    }
  };

  const handleRunBenchmark = async () => {
    setIsBenchmarking(true);
    setBenchmarkReport(null);
    try {
      const rep = await api.runGraphBenchmark({
        strategy: queryStrategy,
        provider_id: 'mock',
        model_id: 'mock-instruct-v1',
      });
      setBenchmarkReport(rep);
    } catch (e) {
      setErrorMsg(e.message);
    } finally {
      setIsBenchmarking(false);
    }
  };

  // Filtered nodes for Knowledge Graph Explorer
  const filteredEntities = useMemo(() => {
    return entities.filter(e => {
      const matchType = filterType === 'ALL' || e.entity_type === filterType;
      const matchSearch = !searchQuery || e.canonical_name.toLowerCase().includes(searchQuery.toLowerCase());
      return matchType && matchSearch;
    });
  }, [entities, filterType, searchQuery]);

  // Compute node coordinates in a circular / concentric layout for interactive SVG
  const nodeLayout = useMemo(() => {
    const width = 800;
    const height = 520;
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) * 0.38;

    const coords = {};
    const count = filteredEntities.length;
    filteredEntities.forEach((node, idx) => {
      const angle = (2 * Math.PI * idx) / Math.max(count, 1);
      coords[node.id] = {
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle),
      };
    });
    return coords;
  }, [filteredEntities]);

  return (
    <div className="flex-1 flex flex-col h-screen overflow-hidden bg-slate-950 text-slate-100 font-sans">
      {/* Top Header */}
      <header className="h-14 border-b border-slate-800 px-6 flex items-center justify-between bg-slate-900/60 shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-purple-600/30 border border-purple-500/40 flex items-center justify-center text-purple-400">
            <Share2 className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
              Graph RAG Laboratory
              <span className="text-[10px] px-2 py-0.5 rounded bg-purple-950 text-purple-400 border border-purple-800/40 font-mono">
                Phase 11 &bull; Ready
              </span>
            </h1>
            <p className="text-[11px] text-slate-400">
              Knowledge graph construction, bounded relational traversal & hybrid vector retrieval
            </p>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs">
          {[
            { id: 'builder', label: 'Graph Builder', icon: Boxes },
            { id: 'explorer', label: 'Graph Explorer', icon: Share2 },
            { id: 'resolution', label: 'Entity Resolution', icon: Users },
            { id: 'query', label: 'Graph Query Lab', icon: Search },
            { id: 'communities', label: 'Communities', icon: Compass },
            { id: 'compare', label: 'Strategy Compare', icon: Layers },
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
                    ? 'bg-purple-600 text-white shadow-sm'
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

        {successMsg && (
          <div className="mb-4 p-3 bg-emerald-950/60 border border-emerald-800/50 rounded-lg text-xs text-emerald-300 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
              <span>{successMsg}</span>
            </div>
            <button onClick={() => setSuccessMsg(null)} className="text-slate-400 hover:text-slate-200">
              &times;
            </button>
          </div>
        )}

        {/* TAB 1: GRAPH BUILDER */}
        {activeTab === 'builder' && (
          <div className="max-w-4xl mx-auto space-y-6">
            <div className="bg-slate-900/60 p-5 rounded-xl border border-slate-800 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
                    <Boxes className="w-4 h-4 text-purple-400" />
                    Knowledge Graph Construction Pipeline
                  </h2>
                  <p className="text-xs text-slate-400">
                    Extract structured entities and relationship triples directly from ingested document chunks.
                  </p>
                </div>
                <span className="text-[10px] px-2.5 py-1 rounded bg-purple-950 text-purple-400 font-mono border border-purple-800/40">
                  POSTGRES PROPERTY GRAPH
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-xs text-slate-400 font-medium">Target Document</label>
                  <select
                    value={selectedDocId}
                    onChange={e => setSelectedDocId(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-xs text-slate-200 focus:outline-none focus:border-purple-500"
                  >
                    {documents.map(d => (
                      <option key={d.id} value={d.id}>
                        {d.original_filename} ({d.file_size ? `${Math.round(d.file_size / 1024)} KB` : 'PDF'})
                      </option>
                    ))}
                  </select>
                </div>

                <div className="space-y-1.5">
                  <label className="text-xs text-slate-400 font-medium">Extraction Strategy</label>
                  <select
                    value={extractStrategy}
                    onChange={e => setExtractStrategy(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-xs text-slate-200 focus:outline-none focus:border-purple-500"
                  >
                    <option value="rule">Rule-Based & Regex (Deterministic High-Speed)</option>
                    <option value="hybrid">Hybrid (Rules + LLM Structured Validation)</option>
                    <option value="llm">LLM Structured Extraction Only</option>
                  </select>
                </div>
              </div>

              <button
                onClick={handleRunExtraction}
                disabled={isExtracting || !selectedDocId}
                className="w-full py-2.5 bg-purple-600 hover:bg-purple-500 text-white rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition-colors shadow-md shadow-purple-600/20"
              >
                {isExtracting ? (
                  <>
                    <RotateCcw className="w-4 h-4 animate-spin" />
                    <span>Extracting Entities & Relations...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 fill-white" />
                    <span>Build Knowledge Graph</span>
                  </>
                )}
              </button>
            </div>

            {/* Extraction Telemetry & Live Stats */}
            {graphStats && (
              <div className="grid grid-cols-4 gap-4">
                <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4 text-center">
                  <span className="text-[10px] text-slate-500 block uppercase font-mono">Entities (Nodes)</span>
                  <span className="text-2xl font-bold font-mono text-purple-400">{graphStats.node_count}</span>
                </div>
                <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4 text-center">
                  <span className="text-[10px] text-slate-500 block uppercase font-mono">Relations (Edges)</span>
                  <span className="text-2xl font-bold font-mono text-indigo-400">{graphStats.edge_count}</span>
                </div>
                <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4 text-center">
                  <span className="text-[10px] text-slate-500 block uppercase font-mono">Evidence Passages</span>
                  <span className="text-2xl font-bold font-mono text-emerald-400">{graphStats.evidence_count}</span>
                </div>
                <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4 text-center">
                  <span className="text-[10px] text-slate-500 block uppercase font-mono">Graph Density</span>
                  <span className="text-2xl font-bold font-mono text-amber-400">{graphStats.density}</span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 2: KNOWLEDGE GRAPH EXPLORER */}
        {activeTab === 'explorer' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-full">
            {/* Visual Graph Canvas (8 cols) */}
            <div className="lg:col-span-8 flex flex-col bg-slate-900/50 border border-slate-800 rounded-xl overflow-hidden">
              {/* Canvas Controls Header */}
              <div className="p-3 border-b border-slate-800 bg-slate-950/70 flex items-center justify-between text-xs">
                <div className="flex items-center gap-2">
                  <Filter className="w-3.5 h-3.5 text-purple-400" />
                  <select
                    value={filterType}
                    onChange={e => setFilterType(e.target.value)}
                    className="bg-slate-900 border border-slate-800 rounded px-2 py-1 text-slate-300 text-xs focus:outline-none"
                  >
                    <option value="ALL">All Entity Types</option>
                    <option value="PERSON">Persons</option>
                    <option value="TECHNOLOGY">Technologies</option>
                    <option value="ORGANIZATION">Organizations</option>
                    <option value="JOB_ROLE">Job Roles</option>
                    <option value="LOCATION">Locations</option>
                  </select>
                </div>

                <div className="flex items-center gap-2">
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={e => setSearchQuery(e.target.value)}
                    placeholder="Search node..."
                    className="bg-slate-900 border border-slate-800 rounded px-2 py-1 text-xs text-slate-200 placeholder-slate-600 focus:outline-none w-36"
                  />
                  <span className="text-[10px] text-slate-500 font-mono">
                    {filteredEntities.length} Nodes
                  </span>
                </div>
              </div>

              {/* Interactive SVG Canvas */}
              <div className="flex-1 relative bg-slate-950 flex items-center justify-center overflow-hidden min-h-[460px]">
                {filteredEntities.length === 0 ? (
                  <div className="text-center text-xs text-slate-600">
                    No entities in graph yet. Run extraction in Graph Builder tab!
                  </div>
                ) : (
                  <svg className="w-full h-full" viewBox="0 0 800 520">
                    {/* Render Edge Lines */}
                    {relations.map((rel, idx) => {
                      const p1 = nodeLayout[rel.source_entity_id];
                      const p2 = nodeLayout[rel.target_entity_id];
                      if (!p1 || !p2) return null;
                      return (
                        <g key={idx}>
                          <line
                            x1={p1.x}
                            y1={p1.y}
                            x2={p2.x}
                            y2={p2.y}
                            stroke="#334155"
                            strokeWidth="1.5"
                            strokeDasharray="2,2"
                          />
                          <text
                            x={(p1.x + p2.x) / 2}
                            y={(p1.y + p2.y) / 2 - 4}
                            fill="#64748b"
                            fontSize="8"
                            textAnchor="middle"
                            fontFamily="monospace"
                          >
                            {rel.relation_type}
                          </text>
                        </g>
                      );
                    })}

                    {/* Render Nodes */}
                    {filteredEntities.map((ent, idx) => {
                      const pos = nodeLayout[ent.id];
                      if (!pos) return null;
                      const isSelected = selectedNode?.id === ent.id;
                      const colorMeta = ENTITY_TYPE_COLORS[ent.entity_type] || ENTITY_TYPE_COLORS.DEFAULT;

                      return (
                        <g
                          key={ent.id}
                          className="cursor-pointer transition-transform hover:scale-110"
                          onClick={() => setSelectedNode(ent)}
                        >
                          <circle
                            cx={pos.x}
                            cy={pos.y}
                            r={isSelected ? 16 : 12}
                            fill={colorMeta.hex}
                            fillOpacity="0.85"
                            stroke={isSelected ? '#ffffff' : '#1e293b'}
                            strokeWidth={isSelected ? 3 : 1.5}
                          />
                          <text
                            x={pos.x}
                            y={pos.y + 24}
                            fill="#e2e8f0"
                            fontSize="9"
                            fontWeight={isSelected ? 'bold' : 'normal'}
                            textAnchor="middle"
                          >
                            {ent.canonical_name}
                          </text>
                        </g>
                      );
                    })}
                  </svg>
                )}
              </div>
            </div>

            {/* Right Node Details & Provenance Inspector (4 cols) */}
            <div className="lg:col-span-4 bg-slate-900/60 border border-slate-800 rounded-xl p-4 space-y-4">
              <h3 className="text-xs font-semibold text-slate-300 flex items-center gap-1.5 border-b border-slate-800 pb-2">
                <Compass className="w-3.5 h-3.5 text-purple-400" />
                Entity Details & Provenance
              </h3>

              {selectedNode ? (
                <div className="space-y-3 text-xs">
                  <div>
                    <span className="text-[10px] text-slate-500 uppercase font-mono block">Canonical Name</span>
                    <span className="text-sm font-bold text-slate-100">{selectedNode.canonical_name}</span>
                  </div>

                  <div className="flex items-center gap-2">
                    <span className="text-[10px] text-slate-500 uppercase font-mono">Entity Type:</span>
                    <span className={`text-[10px] px-2 py-0.5 rounded font-mono font-semibold ${
                      ENTITY_TYPE_COLORS[selectedNode.entity_type]?.bg || 'bg-slate-800'
                    } ${ENTITY_TYPE_COLORS[selectedNode.entity_type]?.text || 'text-slate-300'}`}>
                      {selectedNode.entity_type}
                    </span>
                  </div>

                  {selectedNode.aliases?.length > 0 && (
                    <div>
                      <span className="text-[10px] text-slate-500 uppercase font-mono block">Known Aliases</span>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {selectedNode.aliases.map((a, idx) => (
                          <span key={idx} className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">
                            {a}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="pt-2 border-t border-slate-800 space-y-2">
                    <span className="text-[10px] text-slate-500 uppercase font-mono block">
                      Connected Relations ({selectedNode.degree})
                    </span>
                    <div className="space-y-1.5 max-h-40 overflow-y-auto pr-1">
                      {relations
                        .filter(r => r.source_entity_id === selectedNode.id || r.target_entity_id === selectedNode.id)
                        .map((rel, idx) => (
                          <div key={idx} className="p-2 rounded bg-slate-950 border border-slate-800/80 text-[11px] font-mono">
                            <span className="text-purple-400 font-semibold">{rel.relation_type}</span>
                            <div className="text-[10px] text-slate-400 truncate">
                              {rel.source_name || 'Node'} &rarr; {rel.target_name || 'Node'}
                            </div>
                          </div>
                        ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-12 text-xs text-slate-500">
                  Click on any node in the graph visualization to inspect properties and connected edges.
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 3: ENTITY RESOLUTION LAB */}
        {activeTab === 'resolution' && (
          <div className="max-w-4xl mx-auto space-y-4">
            <div className="bg-slate-900/60 p-4 rounded-xl border border-slate-800 flex items-center justify-between">
              <div>
                <h2 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
                  <Users className="w-4 h-4 text-purple-400" />
                  Entity Resolution & Deduplication
                </h2>
                <p className="text-xs text-slate-400">
                  Fuzzy candidate matching, alias resolution, and human-in-the-loop review guards.
                </p>
              </div>
              <span className="text-xs font-mono text-purple-400">
                {mergeCandidates.length} Merge Candidates
              </span>
            </div>

            {mergeCandidates.length === 0 ? (
              <div className="text-center py-16 bg-slate-900/30 rounded-xl border border-slate-800/60 text-xs text-slate-500">
                No duplicate or ambiguous entity candidates detected. The knowledge graph is normalized!
              </div>
            ) : (
              <div className="space-y-3">
                {mergeCandidates.map((cand, idx) => (
                  <div
                    key={idx}
                    className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 flex items-center justify-between text-xs"
                  >
                    <div className="space-y-1">
                      <div className="flex items-center gap-2 font-mono">
                        <span className="font-bold text-slate-200">{cand.source_entity.canonical_name}</span>
                        <span className="text-slate-500">&harr;</span>
                        <span className="font-bold text-slate-200">{cand.target_entity.canonical_name}</span>
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-950 text-purple-400">
                          {cand.source_entity.entity_type}
                        </span>
                      </div>
                      <div className="text-[11px] text-slate-400">
                        Match Reason: {cand.match_reason} (Similarity: {cand.similarity_score})
                      </div>
                      {cand.requires_human_review && (
                        <span className="text-[10px] text-amber-400 font-mono flex items-center gap-1">
                          <AlertTriangle className="w-3 h-3" /> Human Review Required (Protected Entity Type)
                        </span>
                      )}
                    </div>

                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => handleMergeEntities(cand.source_entity.id, cand.target_entity.id)}
                        className="px-3 py-1.5 bg-purple-600 hover:bg-purple-500 text-white rounded-lg text-xs font-semibold flex items-center gap-1"
                      >
                        <Check className="w-3 h-3" /> Merge
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* TAB 4: GRAPH QUERY LAB */}
        {activeTab === 'query' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left Controls & Query (5 cols) */}
            <div className="lg:col-span-5 space-y-4">
              <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 space-y-4">
                <div className="flex items-center justify-between">
                  <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                    <Search className="w-3.5 h-3.5 text-purple-400" />
                    Graph Retrieval Query
                  </h2>
                  <span className="text-[10px] text-emerald-400 font-mono">MULTI-HOP REASONING</span>
                </div>

                {/* Preset Scenarios */}
                <div className="space-y-1.5">
                  <label className="text-[11px] text-slate-400">Preset Sample Queries</label>
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

                {/* Question Text Area */}
                <div className="space-y-1.5">
                  <label className="text-[11px] text-slate-400 font-medium">Question</label>
                  <textarea
                    rows={3}
                    value={question}
                    onChange={e => setQuestion(e.target.value)}
                    placeholder="Ask a question requiring graph reasoning..."
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-xs text-slate-100 placeholder-slate-600 focus:outline-none focus:border-purple-500 font-sans"
                  />
                </div>

                {/* Retrieval Strategy Selector */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1">
                    <label className="text-[11px] text-slate-400 font-medium">Retrieval Strategy</label>
                    <select
                      value={queryStrategy}
                      onChange={e => setQueryStrategy(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 focus:outline-none focus:border-purple-500"
                    >
                      <option value="hybrid_graph_vector">Hybrid (Graph + Vector RRF)</option>
                      <option value="graph_traversal">Graph Traversal (Paths)</option>
                      <option value="entity_graph">Entity-Based Graph</option>
                      <option value="community_based">Community-Based (Thematic)</option>
                    </select>
                  </div>

                  <div className="space-y-1">
                    <label className="text-[11px] text-slate-400 font-medium">Max Traversal Hops</label>
                    <select
                      value={maxHops}
                      onChange={e => setMaxHops(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 focus:outline-none focus:border-purple-500"
                    >
                      <option value={1}>1 Hop (Direct Neighbors)</option>
                      <option value={2}>2 Hops (Relational Multi-hop)</option>
                      <option value={3}>3 Hops (Deep Chain)</option>
                    </select>
                  </div>
                </div>

                {/* Run Button */}
                <button
                  onClick={handleRunQuery}
                  disabled={isQuerying || !question.trim()}
                  className="w-full py-2.5 bg-purple-600 hover:bg-purple-500 text-white rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition-colors shadow-md shadow-purple-600/20"
                >
                  {isQuerying ? (
                    <>
                      <RotateCcw className="w-4 h-4 animate-spin" />
                      <span>Traversing Graph & Generating...</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 fill-white" />
                      <span>Execute Graph RAG Query</span>
                    </>
                  )}
                </button>
              </div>

              {/* Execution Telemetry Snapshot */}
              {queryResponse && (
                <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-3 grid grid-cols-3 gap-2 text-center text-xs">
                  <div>
                    <span className="text-[10px] text-slate-500 block">Entities Found</span>
                    <span className="font-mono font-bold text-purple-400">{queryResponse.retrieved_entities?.length || 0}</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 block">Paths Explored</span>
                    <span className="font-mono font-bold text-indigo-400">{queryResponse.traversal_paths?.length || 0}</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 block">Latency</span>
                    <span className="font-mono font-bold text-emerald-400">{Math.round(queryResponse.latency_ms)} ms</span>
                  </div>
                </div>
              )}
            </div>

            {/* Right Output: Grounded Cited Answer & Traversed Paths (7 cols) */}
            <div className="lg:col-span-7 space-y-4">
              {/* Answer Box */}
              <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-3">
                <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
                  <div className="flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-emerald-400" />
                    <h3 className="text-xs font-semibold text-slate-200">Graph-Grounded Answer</h3>
                  </div>
                  {queryResponse && (
                    <span className="text-[10px] font-mono text-purple-400 capitalize">
                      {queryResponse.strategy.replace('_', ' ')}
                    </span>
                  )}
                </div>

                <div className="min-h-[120px] text-xs leading-relaxed text-slate-200 whitespace-pre-wrap">
                  {queryResponse ? (
                    queryResponse.final_answer
                  ) : (
                    <span className="text-slate-600 italic">
                      Execute a query on the left to observe graph traversal, relational triples, and cited generation...
                    </span>
                  )}
                </div>

                {/* Grounded Citations */}
                {queryResponse?.citations?.length > 0 && (
                  <div className="pt-3 border-t border-slate-800/80 space-y-2">
                    <div className="text-[11px] font-semibold text-slate-400 flex items-center gap-1.5">
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                      Documentary Source Citations ({queryResponse.citations.length})
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                      {queryResponse.citations.map((c, idx) => (
                        <div
                          key={idx}
                          className="bg-slate-950/80 border border-slate-800/80 rounded-lg p-2.5 text-[11px] space-y-1"
                        >
                          <div className="font-mono text-emerald-400 font-bold">{c.source_id}</div>
                          <div className="text-slate-400 text-[10px] italic line-clamp-2">"{c.snippet}"</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Traversal Paths & Triples Card */}
              {queryResponse?.traversal_paths?.length > 0 && (
                <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4 space-y-2">
                  <h3 className="text-xs font-semibold text-slate-300 flex items-center gap-2">
                    <GitFork className="w-4 h-4 text-purple-400" />
                    Discovered Graph Traversal Paths ({queryResponse.traversal_paths.length})
                  </h3>
                  <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
                    {queryResponse.traversal_paths.map((p, idx) => (
                      <div
                        key={idx}
                        className="bg-slate-950 p-2 rounded border border-slate-800/70 font-mono text-[11px] flex items-center gap-2"
                      >
                        {p.length === 3 ? (
                          <>
                            <span className="text-purple-300 font-bold">{p[0]}</span>
                            <span className="text-slate-500">&mdash;[{p[1]}]&rarr;</span>
                            <span className="text-indigo-300 font-bold">{p[2]}</span>
                          </>
                        ) : (
                          <span className="text-slate-300">{p.join(' ')}</span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 5: COMMUNITY EXPLORER */}
        {activeTab === 'communities' && (
          <div className="max-w-5xl mx-auto space-y-4">
            <div className="bg-slate-900/60 p-4 rounded-xl border border-slate-800 flex items-center justify-between">
              <div>
                <h2 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
                  <Compass className="w-4 h-4 text-purple-400" />
                  Knowledge Communities & Thematic Clusters
                </h2>
                <p className="text-xs text-slate-400">
                  Global modularity clustering detecting thematic domains for high-level corpus queries.
                </p>
              </div>
              <span className="text-xs font-mono text-purple-400">
                {communities.length} Communities Detected
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {communities.map((comm, idx) => (
                <div
                  key={idx}
                  className="bg-slate-900/40 border border-slate-800 rounded-xl p-4 space-y-3 flex flex-col justify-between"
                >
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-xs text-purple-300">{comm.name}</span>
                      <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-400 font-mono">
                        {comm.size} entities
                      </span>
                    </div>
                    <p className="text-xs text-slate-300 leading-relaxed">{comm.summary_text}</p>
                  </div>

                  <div className="pt-2 border-t border-slate-800 space-y-1">
                    <span className="text-[10px] text-slate-500 uppercase font-mono block">Key Relations</span>
                    <div className="space-y-1 max-h-24 overflow-y-auto">
                      {comm.key_relations.slice(0, 4).map((r, rIdx) => (
                        <div key={rIdx} className="text-[10px] font-mono text-slate-400 bg-slate-950 p-1 rounded">
                          {r}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* TAB 6: STRATEGY COMPARISON */}
        {activeTab === 'compare' && (
          <div className="max-w-5xl mx-auto space-y-4">
            <div className="bg-slate-900/60 p-4 rounded-xl border border-slate-800 flex items-center justify-between">
              <div>
                <h2 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
                  <Layers className="w-4 h-4 text-purple-400" />
                  Graph vs. Vector Strategy Benchmark
                </h2>
                <p className="text-xs text-slate-400">
                  Compare Entity Graph, Graph Traversal, Hybrid Graph+Vector, and Community-Based retrieval.
                </p>
              </div>
              <button
                onClick={handleRunComparison}
                disabled={isComparing || !question.trim()}
                className="px-4 py-2 bg-purple-600 hover:bg-purple-500 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors"
              >
                {isComparing ? <RotateCcw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                <span>Benchmark Strategies</span>
              </button>
            </div>

            {compareResults ? (
              <div className="bg-slate-900/40 border border-slate-800 rounded-xl overflow-hidden">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-slate-800 bg-slate-950/80 text-[11px] text-slate-400 uppercase font-mono">
                      <th className="p-3">Strategy</th>
                      <th className="p-3">Latency</th>
                      <th className="p-3">Entities</th>
                      <th className="p-3">Relations</th>
                      <th className="p-3">Evidence Passages</th>
                      <th className="p-3">Citations</th>
                      <th className="p-3">Answer Snippet</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60">
                    {Object.entries(compareResults.results || {}).map(([stratName, item], idx) => (
                      <tr key={idx} className="hover:bg-slate-900/40 transition-colors">
                        <td className="p-3 font-medium text-slate-200 capitalize font-mono">{stratName.replace(/_/g, ' ')}</td>
                        <td className="p-3 font-mono text-emerald-400">{item.latency_ms} ms</td>
                        <td className="p-3 font-mono text-purple-400">{item.entities_count}</td>
                        <td className="p-3 font-mono text-indigo-400">{item.relations_count}</td>
                        <td className="p-3 font-mono text-slate-300">{item.evidence_count}</td>
                        <td className="p-3 font-mono text-amber-400">{item.citation_count}</td>
                        <td className="p-3 text-[11px] text-slate-400 max-w-xs truncate">{item.answer_snippet}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="text-center py-16 bg-slate-900/30 rounded-xl border border-slate-800/60 text-xs text-slate-500">
                Click &ldquo;Benchmark Strategies&rdquo; to evaluate retrieval quality across all 4 Graph RAG modes.
              </div>
            )}
          </div>
        )}

        {/* TAB 7: 15-SCENARIO BENCHMARK SUITE */}
        {activeTab === 'benchmark' && (
          <div className="max-w-5xl mx-auto space-y-4">
            <div className="bg-slate-900/60 p-4 rounded-xl border border-slate-800 flex items-center justify-between">
              <div>
                <h2 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  15-Scenario Synthetic Benchmark Suite
                </h2>
                <p className="text-xs text-slate-400">
                  Comprehensive validation across single-hop, multi-hop, cross-document, alias matching, and global queries.
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
                    <span className="text-[10px] text-slate-500 block">Avg Entities</span>
                    <span className="font-mono text-lg font-bold text-purple-400">
                      {benchmarkReport.average_entities}
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 block">Avg Relations</span>
                    <span className="font-mono text-lg font-bold text-indigo-400">
                      {benchmarkReport.average_relations}
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
                        <th className="p-3">Entities</th>
                        <th className="p-3">Relations</th>
                        <th className="p-3">Latency</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60">
                      {benchmarkReport.results.map((res, idx) => (
                        <tr key={idx} className="hover:bg-slate-900/40 transition-colors">
                          <td className="p-3 font-mono text-purple-400 font-semibold">{res.scenario_id}</td>
                          <td className="p-3 text-slate-200">{res.title}</td>
                          <td className="p-3">
                            <span className={`text-[10px] px-2 py-0.5 rounded font-mono font-semibold ${
                              res.success ? 'bg-emerald-950 text-emerald-400 border border-emerald-800/40' : 'bg-rose-950 text-rose-400 border border-rose-800/40'
                            }`}>
                              {res.success ? 'PASSED' : 'FAILED'}
                            </span>
                          </td>
                          <td className="p-3 font-mono text-purple-400">{res.entities_retrieved}</td>
                          <td className="p-3 font-mono text-indigo-400">{res.relations_retrieved}</td>
                          <td className="p-3 font-mono text-emerald-400">{res.latency_ms} ms</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : (
              <div className="text-center py-16 bg-slate-900/30 rounded-xl border border-slate-800/60 text-xs text-slate-500">
                Click &ldquo;Run 15 Scenarios&rdquo; to benchmark multi-hop reasoning, alias matching, and cross-document relational retrieval.
              </div>
            )}
          </div>
        )}

        {/* TAB 8: LEARNING CENTER */}
        {activeTab === 'learning' && (
          <div className="max-w-4xl mx-auto space-y-6">
            <div className="bg-slate-900/60 p-5 rounded-xl border border-slate-800 space-y-2">
              <h2 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
                <BookOpen className="w-4 h-4 text-purple-400" />
                Curriculum Guide: Graph RAG & Knowledge Graph Engineering
              </h2>
              <p className="text-xs text-slate-400 leading-relaxed">
                Understand how structured knowledge graphs complement unstructured vector search,
                how recursive SQL CTEs execute bounded graph traversals, and why entity resolution is crucial.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4 space-y-2">
                <h3 className="text-xs font-semibold text-purple-400 flex items-center gap-1.5">
                  <Share2 className="w-3.5 h-3.5" />
                  Property Graphs in PostgreSQL
                </h3>
                <p className="text-xs text-slate-300 leading-relaxed">
                  Instead of forcing external graph engine dependencies like Neo4j, our lab models entities as nodes
                  and relationships as foreign-key edges in PostgreSQL with JSONB property dictionaries. This keeps the
                  infrastructure lean, self-contained, and transactionally aligned with chunk embeddings.
                </p>
              </div>

              <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4 space-y-2">
                <h3 className="text-xs font-semibold text-indigo-400 flex items-center gap-1.5">
                  <GitFork className="w-3.5 h-3.5" />
                  Multi-Hop Relational Traversal
                </h3>
                <p className="text-xs text-slate-300 leading-relaxed">
                  Standard vector similarity search retrieves isolated snippets. Graph traversal traverses connecting edges
                  (e.g., Person &rarr; Project &rarr; Technology &rarr; Database), synthesizing multi-hop facts that vector
                  similarity would otherwise miss.
                </p>
              </div>

              <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4 space-y-2">
                <h3 className="text-xs font-semibold text-emerald-400 flex items-center gap-1.5">
                  <Layers className="w-3.5 h-3.5" />
                  Hybrid Graph + Vector Fusion
                </h3>
                <p className="text-xs text-slate-300 leading-relaxed">
                  The strongest retrieval paradigm fuses both modalities: graph paths supply structured entity relationships,
                  while vector search pulls descriptive unstructured paragraphs. Reciprocal Rank Fusion (RRF) reconciles both streams.
                </p>
              </div>

              <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4 space-y-2">
                <h3 className="text-xs font-semibold text-amber-400 flex items-center gap-1.5">
                  <Compass className="w-3.5 h-3.5" />
                  Community-Based Global Retrieval
                </h3>
                <p className="text-xs text-slate-300 leading-relaxed">
                  For broad thematic questions (*"Summarize the technical architecture across the entire corpus"*), community
                  detection clusters related nodes and creates high-level thematic summaries, bypassing the limits of localized search.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
