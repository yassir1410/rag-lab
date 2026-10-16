import React, { useState, useRef } from 'react';
import {
  UploadCloud,
  FileText,
  FileSpreadsheet,
  FileCode,
  File,
  Trash2,
  Eye,
  AlertTriangle,
  CheckCircle2,
  RefreshCw,
  X,
  Layers,
  ArrowRightLeft,
  Info,
  Clock,
  Sparkles,
} from 'lucide-react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';

export function Documents({ docState }) {
  const {
    documents,
    total,
    loading,
    error,
    refresh,
    selectedDocId,
    selectedDocDetails,
    selectedDocSections,
    selectedDocContent,
    loadingDetails,
    loadDocumentDetails,
    closeDetails,
    isUploading,
    uploadError,
    uploadSuccess,
    uploadFile,
    deleteDocument,
  } = docState;

  const [dragActive, setDragActive] = useState(false);
  const [activeTab, setActiveTab] = useState('sections'); // 'sections' or 'comparison'
  const [textMode, setTextMode] = useState('normalized'); // 'normalized' or 'raw'
  const [deleteConfirmId, setDeleteConfirmId] = useState(null);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      await handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = async (e) => {
    if (e.target.files && e.target.files[0]) {
      await handleFileUpload(e.target.files[0]);
      e.target.value = ''; // reset file input
    }
  };

  const handleFileUpload = async (file) => {
    try {
      await uploadFile(file);
    } catch {
      // Handled in state
    }
  };

  const formatFileSize = (bytes) => {
    if (!bytes && bytes !== 0) return '—';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const getFileIcon = (filename) => {
    const ext = filename?.split('.').pop()?.toLowerCase();
    switch (ext) {
      case 'pdf':
        return <FileText className="w-5 h-5 text-rose-400" />;
      case 'docx':
        return <FileSpreadsheet className="w-5 h-5 text-blue-400" />;
      case 'md':
      case 'markdown':
        return <FileCode className="w-5 h-5 text-emerald-400" />;
      default:
        return <File className="w-5 h-5 text-slate-400" />;
    }
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto py-2">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-slate-100">Document Ingestion & Parsing</h1>
            <Badge variant="primary">Phase 02</Badge>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Upload, validate, parse, normalize, and inspect multi-format documents prior to chunking & embeddings.
          </p>
        </div>

        <button
          onClick={refresh}
          disabled={loading}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition-colors self-start"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          Refresh Documents
        </button>
      </div>

      {/* Notifications */}
      {uploadError && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800/60 text-rose-300 text-xs flex items-start gap-3">
          <AlertTriangle className="w-4 h-4 text-rose-400 mt-0.5 shrink-0" />
          <div className="space-y-1">
            <span className="font-semibold block">Ingestion Error</span>
            <span>{uploadError}</span>
          </div>
        </div>
      )}

      {uploadSuccess && (
        <div className="p-4 rounded-xl bg-emerald-950/40 border border-emerald-800/60 text-emerald-300 text-xs flex items-center gap-3">
          <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
          <span>{uploadSuccess}</span>
        </div>
      )}

      {/* Upload Drag & Drop Area */}
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => !isUploading && fileInputRef.current?.click()}
        className={`relative border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all ${
          dragActive
            ? 'border-blue-500 bg-blue-950/30'
            : 'border-slate-700 hover:border-slate-600 bg-slate-900/40'
        } ${isUploading ? 'opacity-50 cursor-wait' : ''}`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.docx,.txt,.md,.markdown"
          onChange={handleFileSelect}
          className="hidden"
          disabled={isUploading}
        />

        <div className="flex flex-col items-center justify-center space-y-3">
          <div className="w-12 h-12 rounded-full bg-blue-600/10 border border-blue-500/20 flex items-center justify-center text-blue-400">
            {isUploading ? (
              <RefreshCw className="w-6 h-6 animate-spin" />
            ) : (
              <UploadCloud className="w-6 h-6" />
            )}
          </div>

          <div>
            <p className="text-sm font-semibold text-slate-200">
              {isUploading
                ? 'Validating, parsing, and normalizing document...'
                : 'Click to upload or drag and drop a file'}
            </p>
            <p className="text-xs text-slate-400 mt-1">
              Supported formats: <span className="text-blue-400 font-mono">PDF, DOCX, TXT, Markdown</span> &bull; Max limit: 15 MB
            </p>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-2 pt-2">
            <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono border border-slate-700">
              PDF (PyMuPDF)
            </span>
            <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono border border-slate-700">
              DOCX (python-docx)
            </span>
            <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono border border-slate-700">
              TXT (UTF-8 / BOM)
            </span>
            <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono border border-slate-700">
              Markdown (.md)
            </span>
          </div>
        </div>
      </div>

      {/* Documents Table */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>
              <FileText className="w-4 h-4 text-blue-400" />
              Ingested Documents ({total})
            </CardTitle>
          </div>
          <CardDescription>
            Documents stored in PostgreSQL with extracted sections and metadata
          </CardDescription>
        </CardHeader>
        <CardContent className="p-0">
          {loading && documents.length === 0 ? (
            <div className="py-12 text-center text-xs text-slate-400 flex items-center justify-center gap-2">
              <RefreshCw className="w-4 h-4 animate-spin text-blue-400" />
              Loading documents...
            </div>
          ) : documents.length === 0 ? (
            <div className="py-12 text-center text-xs text-slate-400 space-y-2">
              <FileText className="w-8 h-8 text-slate-600 mx-auto" />
              <p className="text-sm font-medium text-slate-300">No documents ingested yet</p>
              <p className="text-slate-500">
                Upload a PDF, DOCX, TXT, or Markdown file above to start the ingestion pipeline.
              </p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-slate-700/80 bg-slate-900/60 text-slate-400 uppercase tracking-wider text-[10px]">
                    <th className="py-3 px-4">Document</th>
                    <th className="py-3 px-4">Parser</th>
                    <th className="py-3 px-4">Size</th>
                    <th className="py-3 px-4">Pages / Sections</th>
                    <th className="py-3 px-4">Status</th>
                    <th className="py-3 px-4">Date</th>
                    <th className="py-3 px-4 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800 text-slate-300">
                  {documents.map((doc) => (
                    <tr
                      key={doc.id}
                      className={`hover:bg-slate-800/40 transition-colors ${
                        selectedDocId === doc.id ? 'bg-blue-950/20' : ''
                      }`}
                    >
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-3">
                          {getFileIcon(doc.original_filename)}
                          <div>
                            <span className="font-semibold text-slate-200 block">
                              {doc.original_filename}
                            </span>
                            <span className="text-[10px] text-slate-500 font-mono">
                              SHA256: {doc.sha256_hash?.substring(0, 12)}...
                            </span>
                          </div>
                        </div>
                      </td>
                      <td className="py-3 px-4 font-mono text-[11px] text-slate-400">
                        {doc.extraction_method}
                      </td>
                      <td className="py-3 px-4 font-mono text-[11px] text-slate-400">
                        {formatFileSize(doc.file_size)}
                      </td>
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-1.5">
                          {doc.page_count ? (
                            <Badge variant="default" className="text-[10px]">
                              {doc.page_count} {doc.page_count === 1 ? 'page' : 'pages'}
                            </Badge>
                          ) : null}
                          <Badge variant="primary" className="text-[10px]">
                            {doc.sections_count} sections
                          </Badge>
                          {doc.warnings && doc.warnings.length > 0 && (
                            <Badge variant="warning" className="text-[10px]" title={doc.warnings.join(' | ')}>
                              <AlertTriangle className="w-2.5 h-2.5 mr-1" />
                              {doc.warnings.length} warning(s)
                            </Badge>
                          )}
                        </div>
                      </td>
                      <td className="py-3 px-4">
                        <Badge
                          variant={doc.status === 'processed' ? 'success' : 'danger'}
                          className="capitalize text-[10px]"
                        >
                          {doc.status}
                        </Badge>
                      </td>
                      <td className="py-3 px-4 text-[11px] text-slate-400">
                        {new Date(doc.created_at).toLocaleDateString()}
                      </td>
                      <td className="py-3 px-4 text-right">
                        <div className="flex items-center justify-end gap-2">
                          <button
                            onClick={() => loadDocumentDetails(doc.id)}
                            className="p-1.5 rounded-lg bg-slate-800 hover:bg-blue-600/30 text-slate-300 hover:text-blue-300 border border-slate-700 transition-colors"
                            title="Inspect parsed sections and learning metrics"
                          >
                            <Eye className="w-3.5 h-3.5" />
                          </button>
                          <button
                            onClick={() => setDeleteConfirmId(doc.id)}
                            className="p-1.5 rounded-lg bg-slate-800 hover:bg-rose-900/40 text-slate-400 hover:text-rose-400 border border-slate-700 transition-colors"
                            title="Delete document"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Delete Confirmation Modal */}
      {deleteConfirmId && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-md w-full p-6 space-y-4 shadow-2xl">
            <h3 className="text-base font-bold text-slate-100 flex items-center gap-2">
              <Trash2 className="w-4 h-4 text-rose-400" />
              Confirm Document Deletion
            </h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              Are you sure you want to delete this document? This will remove its metadata, all extracted sections, and safely unlink the physical file from disk.
            </p>
            <div className="flex items-center justify-end gap-3 pt-2">
              <button
                onClick={() => setDeleteConfirmId(null)}
                className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium"
              >
                Cancel
              </button>
              <button
                onClick={async () => {
                  await deleteDocument(deleteConfirmId);
                  setDeleteConfirmId(null);
                }}
                className="px-3 py-1.5 rounded-lg bg-rose-600 hover:bg-rose-500 text-white text-xs font-semibold"
              >
                Delete Document
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Document Detail & Educational Inspection Drawer */}
      {selectedDocId && (
        <Card className="border-blue-700/50 shadow-2xl">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                {getFileIcon(selectedDocDetails?.original_filename)}
                <div>
                  <CardTitle className="text-lg">
                    {selectedDocDetails?.original_filename || 'Document Details'}
                  </CardTitle>
                  <CardDescription>
                    Parser: <span className="font-mono text-slate-300">{selectedDocDetails?.extraction_method}</span> &bull; 
                    {selectedDocDetails?.page_count ? ` ${selectedDocDetails.page_count} pages &bull;` : ''} 
                    {selectedDocSections.length} sections extracted
                  </CardDescription>
                </div>
              </div>

              <div className="flex items-center gap-2">
                {/* Mode Selector Tabs */}
                <div className="flex bg-slate-900 p-1 rounded-lg border border-slate-700 text-xs">
                  <button
                    onClick={() => setActiveTab('sections')}
                    className={`px-3 py-1 rounded-md font-medium transition-colors ${
                      activeTab === 'sections'
                        ? 'bg-blue-600 text-white'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    Parsed Sections ({selectedDocSections.length})
                  </button>
                  <button
                    onClick={() => setActiveTab('comparison')}
                    className={`px-3 py-1 rounded-md font-medium transition-colors flex items-center gap-1.5 ${
                      activeTab === 'comparison'
                        ? 'bg-blue-600 text-white'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <Sparkles className="w-3.5 h-3.5 text-amber-300" />
                    Learning & Comparison
                  </button>
                </div>

                <button
                  onClick={closeDetails}
                  className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-slate-200 transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>
          </CardHeader>

          <CardContent className="space-y-6">
            {/* Warnings Banner (e.g. Scanned/OCR needed) */}
            {selectedDocDetails?.warnings && selectedDocDetails.warnings.length > 0 && (
              <div className="p-4 rounded-xl bg-amber-950/30 border border-amber-800/60 text-amber-200 text-xs space-y-1.5">
                <div className="flex items-center gap-2 font-semibold text-amber-300">
                  <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
                  Parser Warnings & Limitations
                </div>
                <ul className="list-disc list-inside space-y-1 text-amber-200/90 pl-1">
                  {selectedDocDetails.warnings.map((w, idx) => (
                    <li key={idx}>{w}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* TAB 1: SECTIONS VIEW */}
            {activeTab === 'sections' && (
              <div className="space-y-3">
                <div className="flex items-center justify-between text-xs text-slate-400 pb-1">
                  <span>Ordered extracted sections (sequential document order)</span>
                  <div className="flex items-center gap-2">
                    <span className="text-slate-500">Text Display:</span>
                    <button
                      onClick={() => setTextMode(textMode === 'normalized' ? 'raw' : 'normalized')}
                      className="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 font-mono text-[10px]"
                    >
                      {textMode === 'normalized' ? 'Showing Normalized' : 'Showing Raw'}
                    </button>
                  </div>
                </div>

                {loadingDetails ? (
                  <div className="py-8 text-center text-xs text-slate-400">Loading sections...</div>
                ) : selectedDocSections.length === 0 ? (
                  <div className="py-8 text-center text-xs text-slate-500">No sections found</div>
                ) : (
                  <div className="space-y-3 max-h-[600px] overflow-y-auto pr-2">
                    {selectedDocSections.map((sec) => (
                      <div
                        key={sec.id}
                        className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-2 hover:border-slate-700 transition-colors"
                      >
                        <div className="flex items-center justify-between text-xs">
                          <div className="flex items-center gap-2">
                            <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono text-[10px]">
                              Section #{sec.section_index}
                            </span>
                            {sec.page_number && (
                              <Badge variant="primary" className="text-[10px]">
                                Page {sec.page_number}
                              </Badge>
                            )}
                            {sec.heading && (
                              <span className="text-slate-300 font-semibold text-xs truncate max-w-md">
                                {sec.heading}
                              </span>
                            )}
                          </div>
                          <Badge variant="default" className="text-[10px] font-mono">
                            {sec.content_type}
                          </Badge>
                        </div>

                        <div className="bg-slate-950 p-3 rounded-lg border border-slate-900 font-mono text-xs text-slate-300 whitespace-pre-wrap overflow-x-auto leading-relaxed">
                          {textMode === 'normalized'
                            ? (sec.normalized_content || sec.content)
                            : sec.content}
                        </div>

                        <div className="flex items-center gap-4 text-[11px] text-slate-500">
                          <span>Chars: {sec.content?.length || 0}</span>
                          <span>Words: {sec.content?.split(/\s+/).filter(Boolean).length || 0}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* TAB 2: EDUCATIONAL COMPARISON VIEW (Requirement 7) */}
            {activeTab === 'comparison' && selectedDocContent && (
              <div className="space-y-6">
                {/* Educational Callout */}
                <div className="p-4 rounded-xl bg-blue-950/30 border border-blue-800/40 text-blue-200 text-xs space-y-2">
                  <div className="flex items-center gap-2 font-bold text-blue-300">
                    <Info className="w-4 h-4 text-blue-400 shrink-0" />
                    Why Ingestion Quality Governs RAG Performance
                  </div>
                  <p className="text-blue-200/90 leading-relaxed">
                    A vector search can only be as accurate as the text segments ingested into it. 
                    Uncleaned whitespace, malformed headers, lost table cell boundaries, or unhandled scanned pages directly produce noisy embedding vectors, degraded retrieval scores, and downstream LLM hallucinations.
                  </p>
                </div>

                {/* Metrics Grid */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800 space-y-1">
                    <span className="text-[10px] uppercase font-semibold text-slate-500">
                      Raw vs Normalized Chars
                    </span>
                    <div className="text-lg font-bold font-mono text-slate-200">
                      {selectedDocContent.comparison_stats?.raw_character_count || 0} &rarr;{' '}
                      <span className="text-emerald-400">
                        {selectedDocContent.comparison_stats?.normalized_character_count || 0}
                      </span>
                    </div>
                    <span className="text-[10px] text-slate-400">
                      Reduction: {selectedDocContent.comparison_stats?.character_reduction_pct || 0}%
                    </span>
                  </div>

                  <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800 space-y-1">
                    <span className="text-[10px] uppercase font-semibold text-slate-500">
                      Extracted Words
                    </span>
                    <div className="text-lg font-bold font-mono text-slate-200">
                      {selectedDocContent.comparison_stats?.normalized_word_count || 0}
                    </div>
                    <span className="text-[10px] text-slate-400">
                      Across {selectedDocContent.sections_count} sections
                    </span>
                  </div>

                  <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800 space-y-1">
                    <span className="text-[10px] uppercase font-semibold text-slate-500">
                      Engine Used
                    </span>
                    <div className="text-lg font-bold font-mono text-blue-400 truncate">
                      {selectedDocContent.extraction_method}
                    </div>
                    <span className="text-[10px] text-slate-400">
                      Format: {selectedDocDetails?.mime_type?.split('/')[1] || 'text'}
                    </span>
                  </div>

                  <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800 space-y-1">
                    <span className="text-[10px] uppercase font-semibold text-slate-500">
                      Page Attribution
                    </span>
                    <div className="text-lg font-bold font-mono text-slate-200">
                      {selectedDocContent.page_count
                        ? `${selectedDocContent.page_count} Pages`
                        : 'Flow-based (N/A)'}
                    </div>
                    <span className="text-[10px] text-slate-400">
                      Citations ready: {selectedDocContent.page_count ? 'Yes' : 'Section level'}
                    </span>
                  </div>
                </div>

                {/* Side-by-Side Comparison */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Raw Extracted Text */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-xs text-slate-400">
                      <span className="font-semibold text-slate-300">Raw Extracted Text</span>
                      <span className="font-mono text-[10px]">
                        {selectedDocContent.raw_text?.length || 0} characters
                      </span>
                    </div>
                    <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 font-mono text-xs text-slate-400 h-96 overflow-y-auto whitespace-pre-wrap leading-relaxed select-text">
                      {selectedDocContent.raw_text || 'No text extracted'}
                    </div>
                  </div>

                  {/* Normalized Text */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-xs text-slate-400">
                      <span className="font-semibold text-emerald-300">Normalized Text (Ready for Chunking)</span>
                      <span className="font-mono text-[10px]">
                        {selectedDocContent.normalized_text?.length || 0} characters
                      </span>
                    </div>
                    <div className="bg-slate-950 p-4 rounded-xl border border-emerald-900/40 font-mono text-xs text-slate-200 h-96 overflow-y-auto whitespace-pre-wrap leading-relaxed select-text">
                      {selectedDocContent.normalized_text || 'No text extracted'}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
