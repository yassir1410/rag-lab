import React from 'react';
import {
  LayoutDashboard,
  FileText,
  MessageSquare,
  Scissors,
  Network,
  Search,
  CheckCircle,
  FlaskConical,
  Settings,
  Layers,
} from 'lucide-react';

export const NAV_ITEMS = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard, phase: '01', ready: true },
  { id: 'documents', label: 'Documents', icon: FileText, phase: '02', ready: true },
  { id: 'chunking', label: 'Chunking Lab', icon: Scissors, phase: '03', ready: true },
  { id: 'embeddings', label: 'Embeddings Explorer', icon: Network, phase: '04', ready: true },
  { id: 'retrieval', label: 'Retrieval Lab', icon: Search, phase: '05', ready: true },
  { id: 'rag-chat', label: 'RAG Chat', icon: MessageSquare, phase: '06', ready: true },
  { id: 'evaluation', label: 'Evaluation Lab', icon: CheckCircle, phase: '07', ready: true },
  { id: 'experiments', label: 'Experiments', icon: FlaskConical, phase: 'Lab', ready: false },
  { id: 'settings', label: 'Settings', icon: Settings, phase: 'Cfg', ready: false },
];

export function Sidebar({ currentTab, onSelectTab }) {
  return (
    <aside className="w-64 bg-slate-950 border-r border-slate-800 flex flex-col shrink-0 min-h-screen">
      {/* Brand Header */}
      <div className="h-16 flex items-center gap-3 px-6 border-b border-slate-800">
        <div className="w-9 h-9 rounded-lg bg-blue-600 flex items-center justify-center text-white font-bold shadow-md shadow-blue-500/20">
          <Layers className="w-5 h-5" />
        </div>
        <div>
          <span className="font-bold text-sm tracking-wide text-slate-100 block">
            RAG Lab
          </span>
          <span className="text-[10px] text-emerald-400 font-mono tracking-wider block">
            ENGINEERING &bull; v0.7.0
          </span>
        </div>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 py-4 px-3 space-y-1 overflow-y-auto">
        <div className="px-3 pb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
          Curriculum Modules
        </div>
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const isActive = currentTab === item.id;

          return (
            <button
              key={item.id}
              onClick={() => onSelectTab(item.id)}
              className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
                isActive
                  ? 'bg-blue-600/20 text-blue-300 border border-blue-500/30'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
              }`}
            >
              <div className="flex items-center gap-3">
                <Icon className={`w-4 h-4 ${isActive ? 'text-blue-400' : 'text-slate-400'}`} />
                <span>{item.label}</span>
              </div>

              {item.ready ? (
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-400 font-mono border border-emerald-800/40">
                  Ready
                </span>
              ) : (
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-900 text-slate-500 font-mono border border-slate-800">
                  P{item.phase}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Footer Info */}
      <div className="p-4 border-t border-slate-800 text-[11px] text-slate-500">
        <div className="flex items-center justify-between">
          <span>Active Phase:</span>
          <span className="text-slate-300 font-medium">07 — Evaluation & Benchmarking</span>
        </div>
      </div>
    </aside>
  );
}
