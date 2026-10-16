import React from 'react';
import {
  Server,
  Database,
  Layers,
  ArrowRight,
  ShieldCheck,
  CheckCircle2,
  XCircle,
  FileText,
  Clock,
  Sparkles,
  BookOpen,
} from 'lucide-react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';

export function Overview({ healthState, onNavigateToDocuments }) {
  const { health, readiness, loading, error, lastChecked, refresh } = healthState;

  return (
    <div className="space-y-6 max-w-6xl mx-auto py-2">
      {/* Hero Banner */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-blue-900/40 via-indigo-900/30 to-slate-900 border border-blue-800/40 p-8 shadow-xl">
        <div className="relative z-10 max-w-2xl space-y-3">
          <Badge variant="primary">Laboratory Environment &bull; Phase 01 & 02</Badge>
          <h1 className="text-3xl font-extrabold text-slate-50 tracking-tight">
            RAG Engineering Laboratory
          </h1>
          <p className="text-slate-300 text-sm leading-relaxed">
            A modular monorepo designed to master Retrieval-Augmented Generation from the ground up:
            from document ingestion and normalization to vector retrieval, hybrid search, and agentic workflows.
          </p>
          <div className="pt-2 flex items-center gap-3">
            <button
              onClick={onNavigateToDocuments}
              className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold flex items-center gap-2 shadow-lg shadow-blue-600/30 transition-colors"
            >
              Open Documents Lab <ArrowRight className="w-3.5 h-3.5" />
            </button>
            <a
              href="/api/v1/docs"
              target="_blank"
              rel="noreferrer"
              className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold border border-slate-700 transition-colors"
            >
              Interactive API Docs
            </a>
          </div>
        </div>
      </div>

      {/* System Operational Health Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {/* API Liveness */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>
                <Server className="w-4 h-4 text-blue-400" />
                Backend Service
              </CardTitle>
              {health ? (
                <Badge variant="success">Online</Badge>
              ) : (
                <Badge variant="danger">Offline</Badge>
              )}
            </div>
            <CardDescription>FastAPI Liveness (/api/v1/health)</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-xs">
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span className="text-slate-400">Status</span>
              <span className="font-mono text-slate-200">{health?.status || 'Unreachable'}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span className="text-slate-400">Version</span>
              <span className="font-mono text-slate-200">{health?.version || '—'}</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-slate-400">Environment</span>
              <span className="font-mono text-slate-200">{health?.environment || '—'}</span>
            </div>
          </CardContent>
        </Card>

        {/* Database & pgvector Readiness */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>
                <Database className="w-4 h-4 text-emerald-400" />
                PostgreSQL & Vector
              </CardTitle>
              {readiness?.database_connected ? (
                <Badge variant="success">Connected</Badge>
              ) : (
                <Badge variant="danger">Disconnected</Badge>
              )}
            </div>
            <CardDescription>Readiness Probe (/api/v1/health/ready)</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-xs">
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span className="text-slate-400">Database</span>
              <span className="font-mono text-slate-200">
                {readiness?.database_connected ? 'PostgreSQL 16' : 'Offline'}
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span className="text-slate-400">pgvector Extension</span>
              <span className="font-mono text-slate-200">
                {readiness?.pgvector_available
                  ? `Installed (v${readiness.pgvector_version || '0.7+'})`
                  : 'Missing / Disabled'}
              </span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-slate-400">Storage Writable</span>
              <span className="font-mono text-slate-200">
                {readiness?.storage_writable ? 'Yes (data/uploads)' : 'No'}
              </span>
            </div>
          </CardContent>
        </Card>

        {/* Ingestion Engine Status */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>
                <FileText className="w-4 h-4 text-purple-400" />
                Phase 02 Ingestion
              </CardTitle>
              <Badge variant="primary">Operational</Badge>
            </div>
            <CardDescription>Multi-format parsers & Normalizer</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-xs">
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span className="text-slate-400">Supported Formats</span>
              <span className="font-mono text-slate-200">PDF, DOCX, TXT, MD</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span className="text-slate-400">Engines</span>
              <span className="font-mono text-slate-200">PyMuPDF &bull; python-docx</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-slate-400">Security</span>
              <span className="font-mono text-slate-200">Magic-bytes, 15MB, SHA-256</span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Curriculum Roadmap Progress */}
      <Card>
        <CardHeader>
          <CardTitle>
            <BookOpen className="w-4 h-4 text-blue-400" />
            RAG Curriculum Progress (16 Phases)
          </CardTitle>
          <CardDescription>
            Systematic progression from foundation to multimodal and agentic systems
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
            <div className="p-3 rounded-lg bg-emerald-950/40 border border-emerald-800/50 space-y-1">
              <div className="flex items-center justify-between">
                <span className="font-bold text-emerald-400 font-mono">Phase 01</span>
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
              </div>
              <p className="font-semibold text-slate-200">Foundation</p>
              <p className="text-[11px] text-slate-400">FastAPI, pgvector, React, Docker</p>
            </div>

            <div className="p-3 rounded-lg bg-emerald-950/40 border border-emerald-800/50 space-y-1">
              <div className="flex items-center justify-between">
                <span className="font-bold text-emerald-400 font-mono">Phase 02</span>
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
              </div>
              <p className="font-semibold text-slate-200">Document Ingestion</p>
              <p className="text-[11px] text-slate-400">PDF, DOCX, TXT, MD, Normalizer</p>
            </div>

            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 space-y-1 opacity-75">
              <div className="flex items-center justify-between">
                <span className="font-bold text-slate-400 font-mono">Phase 03</span>
                <Clock className="w-3.5 h-3.5 text-slate-500" />
              </div>
              <p className="font-semibold text-slate-300">Chunking Lab</p>
              <p className="text-[11px] text-slate-500">Fixed, Recursive, Semantic splits</p>
            </div>

            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 space-y-1 opacity-75">
              <div className="flex items-center justify-between">
                <span className="font-bold text-slate-400 font-mono">Phase 04</span>
                <Clock className="w-3.5 h-3.5 text-slate-500" />
              </div>
              <p className="font-semibold text-slate-300">Embeddings</p>
              <p className="text-[11px] text-slate-500">SentenceTransformers & Vector Space</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
