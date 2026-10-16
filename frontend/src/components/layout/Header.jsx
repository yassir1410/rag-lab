import React from 'react';
import { Server, Database, Activity, RefreshCw } from 'lucide-react';
import { Badge } from '../ui/Badge';

export function Header({ healthState }) {
  const { isBackendOnline, isDatabaseOnline, isPgVectorAvailable, loading, refresh } = healthState;

  return (
    <header className="h-16 bg-slate-900/80 backdrop-blur-md border-b border-slate-800 px-6 flex items-center justify-between shrink-0">
      <div className="flex items-center gap-3">
        <h2 className="text-sm font-semibold text-slate-200">
          Retrieval-Augmented Generation Engineering Lab
        </h2>
      </div>

      <div className="flex items-center gap-3">
        {/* API Backend Status */}
        <div className="flex items-center gap-2 px-2.5 py-1 rounded-lg bg-slate-950 border border-slate-800 text-xs">
          <Server className="w-3.5 h-3.5 text-slate-400" />
          <span className="text-slate-400 text-[11px]">API:</span>
          {isBackendOnline ? (
            <span className="flex items-center gap-1.5 text-emerald-400 font-medium">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              Connected
            </span>
          ) : (
            <span className="flex items-center gap-1.5 text-rose-400 font-medium">
              <span className="w-2 h-2 rounded-full bg-rose-500" />
              Offline
            </span>
          )}
        </div>

        {/* Database & pgvector Status */}
        <div className="flex items-center gap-2 px-2.5 py-1 rounded-lg bg-slate-950 border border-slate-800 text-xs">
          <Database className="w-3.5 h-3.5 text-slate-400" />
          <span className="text-slate-400 text-[11px]">Database:</span>
          {isDatabaseOnline ? (
            <span className="flex items-center gap-1.5 text-emerald-400 font-medium">
              <span className="w-2 h-2 rounded-full bg-emerald-500" />
              PostgreSQL {isPgVectorAvailable && '(+pgvector)'}
            </span>
          ) : (
            <span className="flex items-center gap-1.5 text-amber-400 font-medium">
              <span className="w-2 h-2 rounded-full bg-amber-500" />
              Unavailable
            </span>
          )}
        </div>

        {/* Refresh Status Button */}
        <button
          onClick={refresh}
          disabled={loading}
          title="Refresh connectivity status"
          className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>
    </header>
  );
}
