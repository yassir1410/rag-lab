import React, { useState, useEffect } from 'react';
import {
  Layers,
  FileText,
  Scan,
  Table as TableIcon,
  Image as ImageIcon,
  Search,
  Eye,
  BarChart3,
  BookOpen,
  Play,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
  RefreshCw,
  Cpu,
  ArrowRight,
  ShieldCheck,
  Maximize2,
  FileCheck,
} from 'lucide-react';
import api from '../services/api';

const TABS = [
  { id: 'analyzer', label: 'Document Analyzer', icon: Scan },
  { id: 'ocr', label: 'OCR Lab', icon: FileCheck },
  { id: 'tables', label: 'Table Intelligence', icon: TableIcon },
  { id: 'embeddings', label: 'Image Embeddings', icon: ImageIcon },
  { id: 'search', label: 'Multimodal Search', icon: Search },
  { id: 'vision-qa', label: 'Vision Q&A', icon: Eye },
  { id: 'compare', label: 'Strategy Compare', icon: BarChart3 },
  { id: 'learning', label: 'Learning Center', icon: BookOpen },
];

export function MultimodalLab() {
  const [activeTab, setActiveTab] = useState('analyzer');
  const [documents, setDocuments] = useState([]);
  const [selectedDocId, setSelectedDocId] = useState('');
  const [pages, setPages] = useState([]);
  const [elements, setElements] = useState([]);
  const [selectedPageNum, setSelectedPageNum] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [statusMessage, setStatusMessage] = useState(null);

  // Tab 2: OCR State
  const [ocrLang, setOcrLang] = useState('eng');
  const [ocrPreview, setOcrPreview] = useState(null);
  const [ocrLoading, setOcrLoading] = useState(false);

  // Tab 3: Tables State
  const [extractedTables, setExtractedTables] = useState([]);
  const [tableLoading, setTableLoading] = useState(false);
  const [tableScope, setTableScope] = useState('all'); // 'all' or 'current'
  const [hasAttemptedTableExtraction, setHasAttemptedTableExtraction] = useState(false);

  // Tab 4: Embeddings State
  const [modelsList, setModelsList] = useState([]);
  const [embeddingText, setEmbeddingText] = useState('Diagram showing microservices architecture and cache layer');
  const [generatedVector, setGeneratedVector] = useState(null);
  const [vectorLoading, setVectorLoading] = useState(false);

  // Tab 5: Search State
  const [searchQuery, setSearchQuery] = useState('Software engineering, Python and cloud systems');
  const [searchStrategy, setSearchStrategy] = useState('hybrid');
  const [searchResults, setSearchResults] = useState(null);
  const [searchLoading, setSearchLoading] = useState(false);

  // Tab 6: Vision QA State
  const [qaQuestion, setQaQuestion] = useState('What profile and key technical competencies are presented in the document?');
  const [qaProvider, setQaProvider] = useState('mock');
  const [qaResponse, setQaResponse] = useState(null);
  const [qaLoading, setQaLoading] = useState(false);

  // Tab 7: Compare State
  const [compareQuery, setCompareQuery] = useState('Database engineering and high-performance APIs');
  const [compareResults, setCompareResults] = useState(null);
  const [compareLoading, setCompareLoading] = useState(false);

  // Tab 8 / Benchmark State
  const [benchmarkReport, setBenchmarkReport] = useState(null);
  const [benchmarkLoading, setBenchmarkLoading] = useState(false);

  // Fetch initial documents & models
  useEffect(() => {
    loadDocuments();
    loadModels();
  }, []);

  // When selected doc changes, fetch pages and elements
  useEffect(() => {
    if (selectedDocId) {
      loadDocData(selectedDocId);
      setExtractedTables([]);
      setHasAttemptedTableExtraction(false);
    }
  }, [selectedDocId]);

  const loadDocuments = async () => {
    try {
      const res = await api.getDocuments();
      const rawList = Array.isArray(res) ? res : (res?.items || []);
      const docs = rawList.filter((d) => d && d.id && d.original_filename);
      setDocuments(docs);
      if (docs.length > 0) {
        setSelectedDocId((prev) => (prev && docs.some((d) => d.id === prev) ? prev : docs[0].id));
      } else {
        setSelectedDocId('');
      }
    } catch (err) {
      setError('Failed to fetch documents: ' + err.message);
    }
  };

  const loadModels = async () => {
    try {
      const res = await api.getMultimodalModels();
      setModelsList(res || []);
    } catch (err) {
      console.warn('Could not load models list', err);
    }
  };

  const loadDocData = async (docId) => {
    setLoading(true);
    setError(null);
    try {
      const [pagesRes, elementsRes] = await Promise.all([
        api.getDocumentPages(docId).catch(() => []),
        api.getDocumentElements(docId).catch(() => []),
      ]);
      setPages(pagesRes);
      setElements(elementsRes);
      if (pagesRes.length > 0) {
        setSelectedPageNum(pagesRes[0].page_number);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRunAnalysis = async () => {
    if (!selectedDocId) return;
    setLoading(true);
    setError(null);
    setStatusMessage('Analyzing document layout, extracting pages, and computing visual embeddings...');
    try {
      const res = await api.analyzeDocumentMultimodal(selectedDocId, {
        dpi: 150,
        ocr_language: ocrLang,
        force_ocr: false,
      });
      setStatusMessage(res.message);
      await loadDocData(selectedDocId);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleOCRPreview = async () => {
    if (!selectedDocId) return;
    setOcrLoading(true);
    try {
      const res = await api.previewOCR({
        document_id: selectedDocId,
        page_number: selectedPageNum,
        language: ocrLang,
        run_engine: 'tesseract',
      });
      setOcrPreview(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setOcrLoading(false);
    }
  };

  const handleExtractTables = async () => {
    if (!selectedDocId) return;
    setTableLoading(true);
    setHasAttemptedTableExtraction(true);
    try {
      const payload = {
        document_id: selectedDocId,
      };
      if (tableScope === 'current' && selectedPageNum) {
        payload.page_number = selectedPageNum;
      }
      const res = await api.extractDocumentTables(payload);
      setExtractedTables(res.tables || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setTableLoading(false);
    }
  };

  const handleGenerateVector = async () => {
    setVectorLoading(true);
    try {
      const res = await api.generateMultimodalEmbeddings({ text_prompt: embeddingText });
      setGeneratedVector(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setVectorLoading(false);
    }
  };

  const handleSearch = async () => {
    if (!selectedDocId) return;
    setSearchLoading(true);
    try {
      const res = await api.searchMultimodal({
        document_id: selectedDocId,
        query: searchQuery,
        strategy: searchStrategy,
        top_k: 5,
        ocr_weight: 0.5,
        visual_weight: 0.5,
      });
      setSearchResults(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setSearchLoading(false);
    }
  };

  const handleVisionQA = async () => {
    if (!selectedDocId) return;
    setQaLoading(true);
    try {
      const res = await api.queryVision({
        document_id: selectedDocId,
        question: qaQuestion,
        provider: qaProvider,
      });
      setQaResponse(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setQaLoading(false);
    }
  };

  const handleCompare = async () => {
    if (!selectedDocId) return;
    setCompareLoading(true);
    try {
      const res = await api.compareMultimodalStrategies(selectedDocId, compareQuery);
      setCompareResults(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setCompareLoading(false);
    }
  };

  const handleRunBenchmark = async () => {
    setBenchmarkLoading(true);
    try {
      const res = await api.runMultimodalBenchmark();
      setBenchmarkReport(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setBenchmarkLoading(false);
    }
  };

  // Filter elements for current page
  const pageElements = elements.filter((e) => e.page_number === selectedPageNum);
  const currentPageAsset = pages.find((p) => p.page_number === selectedPageNum);

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-16">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 to-indigo-600 flex items-center justify-center text-white shadow-lg shadow-cyan-500/20">
              <Scan className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight text-slate-100 flex items-center gap-2">
                Multimodal RAG & Document Intelligence
                <span className="text-xs font-mono px-2 py-0.5 rounded-full bg-cyan-900/40 text-cyan-400 border border-cyan-800">
                  Phase 12
                </span>
              </h1>
              <p className="text-xs text-slate-400">
                Document layout analysis, OCR validation, table extraction, CLIP embeddings & grounded vision reasoning.
              </p>
            </div>
          </div>
        </div>

        {/* Global Document Selector */}
        <div className="flex items-center gap-2">
          <select
            id="multimodal-document-select"
            value={selectedDocId}
            onChange={(e) => setSelectedDocId(e.target.value)}
            disabled={loading || documents.length === 0}
            className="bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded-lg px-3 py-2 focus:ring-2 focus:ring-cyan-500 focus:outline-none disabled:opacity-50"
          >
            {documents.length === 0 ? (
              <option value="">No documents available</option>
            ) : (
              documents.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.original_filename}
                </option>
              ))
            )}
          </select>
          <button
            id="multimodal-run-analysis-btn"
            onClick={handleRunAnalysis}
            disabled={loading || !selectedDocId}
            className="flex items-center gap-1.5 px-3 py-2 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg text-xs font-medium transition shadow-md shadow-cyan-600/20"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            {loading ? 'Analyzing...' : 'Run Layout Analysis'}
          </button>
        </div>
      </div>

      {/* Notifications */}
      {statusMessage && (
        <div className="p-3 bg-emerald-950/40 border border-emerald-800/60 rounded-xl text-xs text-emerald-300 flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
          <span>{statusMessage}</span>
        </div>
      )}
      {error && (
        <div className="p-3 bg-rose-950/40 border border-rose-800/60 rounded-xl text-xs text-rose-300 flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Navigation Tabs */}
      <div className="flex border-b border-slate-800 space-x-1 overflow-x-auto pb-px">
        {TABS.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-3.5 py-2 text-xs font-medium rounded-t-lg transition border-b-2 whitespace-nowrap ${
                isActive
                  ? 'border-cyan-500 text-cyan-400 bg-slate-900/60'
                  : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-700'
              }`}
            >
              <Icon className="w-4 h-4" />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* TAB 1: Document Analyzer */}
      {activeTab === 'analyzer' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Visual Canvas with Bounding Boxes */}
            <div className="lg:col-span-2 bg-slate-900/80 border border-slate-800 rounded-xl p-4 flex flex-col">
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold text-slate-200">Page Visualizer & Geometry</span>
                  {pages.length > 0 && (
                    <div className="flex gap-1 ml-4">
                      {pages.map((p) => (
                        <button
                          key={p.page_number}
                          onClick={() => setSelectedPageNum(p.page_number)}
                          className={`px-2 py-0.5 text-[11px] rounded font-mono ${
                            selectedPageNum === p.page_number
                              ? 'bg-cyan-600 text-white font-bold'
                              : 'bg-slate-800 text-slate-400 hover:text-slate-200'
                          }`}
                        >
                          P{p.page_number}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
                <div className="flex items-center gap-2 text-[11px] text-slate-400">
                  <span className="w-2.5 h-2.5 rounded bg-blue-500 inline-block"></span> Title
                  <span className="w-2.5 h-2.5 rounded bg-emerald-500 inline-block"></span> Paragraph
                  <span className="w-2.5 h-2.5 rounded bg-amber-500 inline-block"></span> Table
                  <span className="w-2.5 h-2.5 rounded bg-purple-500 inline-block"></span> Figure
                </div>
              </div>

              {/* Rendered Canvas / Mock Frame */}
              <div className="relative mt-4 bg-slate-950 border border-slate-800/80 rounded-lg overflow-hidden flex items-center justify-center min-h-[500px] p-4">
                {pages.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-20 text-center px-4">
                    <Scan className="w-12 h-12 text-slate-600 mb-3 animate-pulse" />
                    <p className="text-sm font-medium text-slate-300">Document has not been analyzed yet</p>
                    <p className="text-xs text-slate-500 max-w-sm mt-1">
                      Click &quot;Run Layout Analysis&quot; above to render page bitmaps, detect bounding boxes, and extract reading order.
                    </p>
                  </div>
                ) : (
                  <div className="relative inline-block max-w-full">
                    <img
                      src={api.getPageImageUrl(selectedDocId, selectedPageNum)}
                      alt={`Page ${selectedPageNum}`}
                      className="max-h-[600px] w-auto object-contain select-none block rounded shadow-md"
                      onError={(e) => {
                        e.target.style.display = 'none';
                      }}
                    />
                    {/* Bounding box overlays */}
                    {pageElements.map((elem) => {
                      const b = elem.bounding_box;
                      const colorMap = {
                        title: 'border-blue-500 bg-blue-500/10 text-blue-300',
                        paragraph: 'border-emerald-500 bg-emerald-500/5 text-emerald-300',
                        table: 'border-amber-500 bg-amber-500/15 text-amber-300',
                        figure: 'border-purple-500 bg-purple-500/15 text-purple-300',
                        header: 'border-slate-500 bg-slate-500/10 text-slate-300',
                        footer: 'border-slate-500 bg-slate-500/10 text-slate-300',
                      };
                      const style = colorMap[elem.element_type] || 'border-cyan-500 bg-cyan-500/10';

                      return (
                        <div
                          key={elem.id}
                          style={{
                            position: 'absolute',
                            left: `${(b.x0 / (currentPageAsset?.width || 595)) * 100}%`,
                            top: `${(b.y0 / (currentPageAsset?.height || 842)) * 100}%`,
                            width: `${(b.width / (currentPageAsset?.width || 595)) * 100}%`,
                            height: `${(b.height / (currentPageAsset?.height || 842)) * 100}%`,
                          }}
                          className={`border rounded border-dashed pointer-events-none transition ${style}`}
                          title={`${elem.element_type} (Order: #${elem.reading_order})`}
                        >
                          <span className="text-[9px] font-mono px-1 bg-slate-900/90 rounded border border-slate-700 absolute -top-2 left-1 whitespace-nowrap">
                            #{elem.reading_order} {elem.element_type}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>

            {/* Elements & Reading Order Inspector */}
            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 flex flex-col h-[650px]">
              <h3 className="text-xs font-semibold text-slate-200 pb-2 border-b border-slate-800">
                Reading Order & Elements ({pageElements.length})
              </h3>
              <div className="flex-1 overflow-y-auto mt-3 space-y-2 pr-1">
                {pageElements.length === 0 ? (
                  <div className="text-center py-12 text-xs text-slate-500">
                    No layout elements analyzed yet. Click "Run Layout Analysis" above.
                  </div>
                ) : (
                  pageElements.map((elem) => (
                    <div
                      key={elem.id}
                      className="p-2.5 rounded-lg border border-slate-800 bg-slate-950/60 hover:border-slate-700 transition"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-[10px] font-mono text-cyan-400 font-bold">
                          #{elem.reading_order} • {elem.element_type.toUpperCase()}
                        </span>
                        <span className="text-[10px] font-mono text-slate-500">
                          conf: {elem.confidence}
                        </span>
                      </div>
                      <p className="text-xs text-slate-300 font-sans line-clamp-2">
                        {elem.content_text || '[No raw text / visual object]'}
                      </p>
                      <div className="text-[10px] font-mono text-slate-500 mt-1">
                        box: [{elem.bounding_box.x0}, {elem.bounding_box.y0}, {elem.bounding_box.x1}, {elem.bounding_box.y1}]
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: OCR Lab */}
      {activeTab === 'ocr' && (
        <div className="space-y-6">
          <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-xl flex items-center justify-between gap-4">
            <div>
              <h3 className="text-sm font-semibold text-slate-200">Optical Character Recognition Comparator</h3>
              <p className="text-xs text-slate-400">
                Measure Character Error Rate (CER) and Word Error Rate (WER) between native PDF text and Tesseract OCR.
              </p>
            </div>
            <div className="flex items-center gap-3">
              <label className="text-xs text-slate-400 font-medium">OCR Language:</label>
              <select
                value={ocrLang}
                onChange={(e) => setOcrLang(e.target.value)}
                className="bg-slate-950 border border-slate-700 text-slate-200 text-xs rounded-lg px-2.5 py-1.5"
              >
                <option value="eng">English (eng)</option>
                <option value="fra">French (fra)</option>
                <option value="fra+eng">Bilingual (fra+eng)</option>
              </select>
              <button
                onClick={handleOCRPreview}
                disabled={ocrLoading || !selectedDocId}
                className="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-medium transition"
              >
                {ocrLoading ? 'Running OCR...' : 'Run OCR Comparison'}
              </button>
            </div>
          </div>

          {ocrPreview && (
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="p-3 bg-slate-900 border border-slate-800 rounded-lg">
                <span className="text-[11px] text-slate-400 block">Character Error Rate (CER)</span>
                <span className="text-lg font-mono font-bold text-cyan-400">
                  {(ocrPreview.cer * 100).toFixed(2)}%
                </span>
              </div>
              <div className="p-3 bg-slate-900 border border-slate-800 rounded-lg">
                <span className="text-[11px] text-slate-400 block">Word Error Rate (WER)</span>
                <span className="text-lg font-mono font-bold text-indigo-400">
                  {(ocrPreview.wer * 100).toFixed(2)}%
                </span>
              </div>
              <div className="p-3 bg-slate-900 border border-slate-800 rounded-lg">
                <span className="text-[11px] text-slate-400 block">OCR Confidence</span>
                <span className="text-lg font-mono font-bold text-emerald-400">
                  {ocrPreview.ocr_confidence.toFixed(1)}%
                </span>
              </div>
              <div className="p-3 bg-slate-900 border border-slate-800 rounded-lg">
                <span className="text-[11px] text-slate-400 block">OCR Execution Time</span>
                <span className="text-lg font-mono font-bold text-slate-300">
                  {ocrPreview.execution_time_ms} ms
                </span>
              </div>
            </div>
          )}

          {ocrPreview && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4">
                <h4 className="text-xs font-semibold text-slate-300 pb-2 border-b border-slate-800">
                  Native PDF Text (Ground Truth)
                </h4>
                <pre className="text-xs text-slate-300 font-mono mt-3 whitespace-pre-wrap max-h-96 overflow-y-auto">
                  {ocrPreview.native_text || '[No native extractable text]'}
                </pre>
              </div>

              <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4">
                <h4 className="text-xs font-semibold text-slate-300 pb-2 border-b border-slate-800 flex justify-between">
                  <span>Tesseract OCR Output</span>
                  <span className="text-[10px] text-slate-400 font-mono">
                    Engine: {ocrPreview.is_available ? 'Tesseract Local' : 'Educational Baseline'}
                  </span>
                </h4>
                <pre className="text-xs text-emerald-300 font-mono mt-3 whitespace-pre-wrap max-h-96 overflow-y-auto">
                  {ocrPreview.ocr_text || '[No OCR text recognized]'}
                </pre>
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 3: Table Intelligence */}
      {activeTab === 'tables' && (
        <div className="space-y-6">
          <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-xl flex flex-wrap items-center justify-between gap-3">
            <div>
              <h3 className="text-sm font-semibold text-slate-200">Structural Table Intelligence</h3>
              <p className="text-xs text-slate-400">
                Multi-strategy extraction preserves column associations, units, and headers compared to naive text collapse.
              </p>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex items-center bg-slate-950 border border-slate-700/80 rounded-lg p-0.5 text-xs">
                <button
                  onClick={() => setTableScope('all')}
                  className={`px-2.5 py-1 rounded-md transition font-medium ${
                    tableScope === 'all'
                      ? 'bg-amber-600/30 text-amber-300 border border-amber-500/40'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  All Pages
                </button>
                <button
                  onClick={() => setTableScope('current')}
                  className={`px-2.5 py-1 rounded-md transition font-medium ${
                    tableScope === 'current'
                      ? 'bg-amber-600/30 text-amber-300 border border-amber-500/40'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Page {selectedPageNum}
                </button>
              </div>
              <button
                id="multimodal-extract-tables-btn"
                onClick={handleExtractTables}
                disabled={tableLoading || !selectedDocId}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-amber-600 hover:bg-amber-500 disabled:opacity-50 text-white rounded-lg text-xs font-medium transition shadow-md shadow-amber-600/20"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${tableLoading ? 'animate-spin' : ''}`} />
                {tableLoading ? 'Extracting...' : 'Extract Tables from Document'}
              </button>
            </div>
          </div>

          {extractedTables.length === 0 ? (
            !hasAttemptedTableExtraction ? (
              <div className="text-center py-16 bg-slate-900/40 border border-slate-800 rounded-xl px-4">
                <TableIcon className="w-10 h-10 text-slate-600 mx-auto mb-3" />
                <p className="text-sm font-medium text-slate-300">No tables loaded yet</p>
                <p className="text-xs text-slate-500 max-w-md mx-auto mt-1">
                  Click &quot;Extract Tables from Document&quot; above to run multi-strategy detection across bordered, rule-divided, and borderless tables.
                </p>
              </div>
            ) : (
              <div className="bg-slate-900/80 border border-amber-500/30 rounded-xl p-6 space-y-4">
                <div className="flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-amber-400 mt-0.5 shrink-0" />
                  <div>
                    <h4 className="text-sm font-semibold text-slate-200">
                      No Tables Detected in {tableScope === 'all' ? 'Document' : `Page ${selectedPageNum}`}
                    </h4>
                    <p className="text-xs text-slate-400 mt-0.5">
                      The multi-strategy table extraction pipeline analyzed the specified scope but did not detect any structured tables.
                    </p>
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-2 text-xs">
                  <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800">
                    <span className="font-semibold text-slate-300 block mb-1">Strategy A: Vector Line Grids</span>
                    <span className="text-slate-500">Checked for fully enclosed bounding cell boxes. (0 bordered tables found)</span>
                  </div>
                  <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800">
                    <span className="font-semibold text-slate-300 block mb-1">Strategy B: Rule Dividers</span>
                    <span className="text-slate-500">Checked for clusters of horizontal dividing lines. (0 clusters found)</span>
                  </div>
                  <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800">
                    <span className="font-semibold text-slate-300 block mb-1">Strategy C: Text Alignment</span>
                    <span className="text-slate-500">Checked for multi-column whitespace alignments. (0 columns resolved)</span>
                  </div>
                </div>
                <p className="text-xs text-slate-400 italic">
                  Tip: If this document has tables on other pages, switch the scope to &quot;All Pages&quot; and re-run extraction.
                </p>
              </div>
            )
          ) : (
            <div className="space-y-6">
              {extractedTables.map((t, idx) => (
                <div key={idx} className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 space-y-4">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-bold text-amber-400 font-mono">
                        {t.caption}
                      </span>
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-500/10 text-amber-300 border border-amber-500/20">
                        {t.num_rows} rows × {t.num_cols} cols
                      </span>
                    </div>
                    <span className="text-[11px] text-slate-500 font-mono">
                      Page #{t.page_number} • ID: {t.table_id}
                    </span>
                  </div>

                  {/* Interactive Structured HTML Grid */}
                  <div>
                    <span className="text-xs font-semibold text-slate-200 block mb-2 flex items-center gap-1.5">
                      <TableIcon className="w-3.5 h-3.5 text-amber-400" />
                      Structured Interactive Grid
                    </span>
                    <div className="overflow-x-auto border border-slate-800 rounded-lg bg-slate-950/80 shadow-inner max-h-96">
                      <table className="w-full text-left text-xs text-slate-300">
                        <thead className="bg-slate-900/90 text-amber-300 uppercase text-[10px] font-semibold tracking-wider sticky top-0 border-b border-slate-700">
                          <tr>
                            {t.headers?.map((h, hIdx) => (
                              <th key={hIdx} className="px-3 py-2.5 whitespace-nowrap border-r border-slate-800/80 last:border-r-0">
                                {h}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800/60 font-mono text-[11px]">
                          {t.rows?.map((row, rIdx) => (
                            <tr key={rIdx} className={rIdx % 2 === 0 ? 'bg-slate-900/40 hover:bg-slate-800/50' : 'bg-slate-950/40 hover:bg-slate-800/50'}>
                              {row.map((cell, cIdx) => (
                                <td key={cIdx} className="px-3 py-2 whitespace-nowrap border-r border-slate-800/50 last:border-r-0 text-slate-300">
                                  {cell || <span className="text-slate-600">—</span>}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 pt-2">
                    {/* Structured Markdown Render */}
                    <div>
                      <span className="text-xs font-semibold text-slate-300 block mb-2">
                        1. Structured Table Representation (Markdown)
                      </span>
                      <pre className="text-xs font-mono bg-slate-950 p-3 rounded-lg border border-slate-800 text-amber-200 whitespace-pre-wrap max-h-64 overflow-y-auto">
                        {t.markdown_table}
                      </pre>
                    </div>

                    {/* Naive Flattened Collapse */}
                    <div>
                      <span className="text-xs font-semibold text-rose-300 block mb-2">
                        2. Naive Flattened Text (Destroys Column Associations)
                      </span>
                      <pre className="text-xs font-mono bg-slate-950 p-3 rounded-lg border border-slate-800 text-rose-200/80 whitespace-pre-wrap max-h-64 overflow-y-auto">
                        {t.raw_flattened_text}
                      </pre>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* TAB 4: Image Embeddings */}
      {activeTab === 'embeddings' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="md:col-span-2 space-y-4">
              <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 space-y-3">
                <h3 className="text-sm font-semibold text-slate-200">
                  Generate Multimodal Embeddings in Aligned CLIP Space
                </h3>
                <p className="text-xs text-slate-400">
                  CLIP (ViT-B/32) maps natural language descriptions and pixel bitmaps into the exact same 512-dimensional vector space.
                </p>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={embeddingText}
                    onChange={(e) => setEmbeddingText(e.target.value)}
                    className="flex-1 bg-slate-950 border border-slate-700 text-slate-200 text-xs rounded-lg px-3 py-2"
                    placeholder="Enter natural language query to project into visual space..."
                  />
                  <button
                    onClick={handleGenerateVector}
                    disabled={vectorLoading}
                    className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg text-xs font-medium transition"
                  >
                    {vectorLoading ? 'Encoding...' : 'Encode into 512d'}
                  </button>
                </div>
              </div>

              {generatedVector && (
                <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <span className="text-xs font-semibold text-cyan-400 font-mono">
                      Model: {generatedVector.model_id} (512 dimensions)
                    </span>
                    <span className="text-[10px] text-emerald-400 font-mono">
                      L2-Normalized: Yes (Cosine = Dot Product)
                    </span>
                  </div>
                  <div>
                    <span className="text-[11px] text-slate-400 block mb-1">Vector Coordinates (First 10 of 512):</span>
                    <pre className="text-xs font-mono bg-slate-950 p-2.5 rounded border border-slate-800 text-slate-300 overflow-x-auto">
                      {JSON.stringify(generatedVector.vector_sample, null, 2)}
                    </pre>
                  </div>
                </div>
              )}
            </div>

            {/* Model Catalog */}
            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 space-y-3">
              <h3 className="text-xs font-semibold text-slate-200 border-b border-slate-800 pb-2">
                Multimodal Registry
              </h3>
              <div className="space-y-3">
                {modelsList.map((m) => (
                  <div key={m.model_id} className="p-3 bg-slate-950/70 border border-slate-800 rounded-lg">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-mono text-cyan-400 font-bold">{m.model_id}</span>
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">
                        {m.dimension > 0 ? `${m.dimension}d` : 'Vision LLM'}
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-400">{m.description}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 5: Multimodal Search */}
      {activeTab === 'search' && (
        <div className="space-y-6">
          <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 space-y-4">
            <h3 className="text-sm font-semibold text-slate-200">Cross-Modal & Hybrid Retrieval Engine</h3>
            <div className="flex flex-col md:flex-row gap-3">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="flex-1 bg-slate-950 border border-slate-700 text-slate-200 text-xs rounded-lg px-3 py-2"
                placeholder="Search across documents via OCR text, visual diagrams, or hybrid fusion..."
              />
              <select
                value={searchStrategy}
                onChange={(e) => setSearchStrategy(e.target.value)}
                className="bg-slate-950 border border-slate-700 text-slate-200 text-xs rounded-lg px-3 py-2"
              >
                <option value="hybrid">Hybrid (RRF OCR + Visual)</option>
                <option value="ocr_text">OCR Text Retrieval</option>
                <option value="text_to_image">Text-to-Image (CLIP)</option>
                <option value="page_level">Page-Level Retrieval</option>
              </select>
              <button
                onClick={handleSearch}
                disabled={searchLoading || !selectedDocId}
                className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg text-xs font-medium transition"
              >
                {searchLoading ? 'Searching...' : 'Search'}
              </button>
            </div>
          </div>

          {searchResults && (
            <div className="space-y-3">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span>
                  Found {searchResults.total_found} evidence items via <strong>{searchResults.strategy}</strong>
                </span>
                <span>Latency: {searchResults.latency_ms} ms</span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {searchResults.results.map((item, idx) => (
                  <div
                    key={idx}
                    className="p-4 bg-slate-900 border border-slate-800 rounded-xl hover:border-slate-700 transition space-y-2"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-mono font-bold text-cyan-400">
                        {item.citation_tag}
                      </span>
                      <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-emerald-400">
                        Score: {(item.score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <p className="text-xs text-slate-300 font-sans line-clamp-3">
                      {item.content_text}
                    </p>
                    <div className="text-[10px] text-slate-500 font-mono">
                      Type: {item.element_type.toUpperCase()} • Page #{item.page_number}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 6: Vision Q&A */}
      {activeTab === 'vision-qa' && (
        <div className="space-y-6">
          <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 space-y-4">
            <h3 className="text-sm font-semibold text-slate-200">Grounded Vision-Language Question Answering</h3>
            <p className="text-xs text-slate-400">
              Submit questions grounded in document page bitmaps and extracted elements. Every answer is verified against citation tags.
            </p>
            <div className="space-y-3">
              <input
                type="text"
                value={qaQuestion}
                onChange={(e) => setQaQuestion(e.target.value)}
                className="w-full bg-slate-950 border border-slate-700 text-slate-200 text-xs rounded-lg px-3 py-2.5"
                placeholder="Ask about candidate experience, tables, diagrams, or page layouts..."
              />
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-slate-400">Provider:</span>
                  <select
                    value={qaProvider}
                    onChange={(e) => setQaProvider(e.target.value)}
                    className="bg-slate-950 border border-slate-700 text-slate-200 text-xs rounded-lg px-2.5 py-1.5"
                  >
                    <option value="mock">Deterministic Mock Vision (Tests & Offline)</option>
                    <option value="gemini">Google Gemini 1.5 Flash (Cloud)</option>
                  </select>
                </div>
                <button
                  onClick={handleVisionQA}
                  disabled={qaLoading || !selectedDocId}
                  className="px-5 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-medium transition"
                >
                  {qaLoading ? 'Reasoning...' : 'Answer Question with Vision Evidence'}
                </button>
              </div>
            </div>
          </div>

          {qaResponse && (
            <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <span className="text-xs font-semibold text-slate-300">Grounded Answer</span>
                <span className="text-xs font-mono text-cyan-400">
                  Model: {qaResponse.model_used} ({qaResponse.latency_ms} ms)
                </span>
              </div>
              <p className="text-sm text-slate-100 font-sans leading-relaxed">
                {qaResponse.answer}
              </p>
              <div className="flex items-center gap-2 pt-2 border-t border-slate-800">
                <span className="text-xs text-slate-400">Verified Citations:</span>
                {qaResponse.citations.map((c, i) => (
                  <span key={i} className="text-xs font-mono px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800">
                    {c}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 7: Strategy Compare */}
      {activeTab === 'compare' && (
        <div className="space-y-6">
          <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 flex flex-col md:flex-row gap-3 items-center justify-between">
            <input
              type="text"
              value={compareQuery}
              onChange={(e) => setCompareQuery(e.target.value)}
              className="flex-1 bg-slate-950 border border-slate-700 text-slate-200 text-xs rounded-lg px-3 py-2"
              placeholder="Query to test across OCR, Text-to-Image, Page-Level and Hybrid..."
            />
            <button
              onClick={handleCompare}
              disabled={compareLoading || !selectedDocId}
              className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg text-xs font-medium transition"
            >
              {compareLoading ? 'Comparing...' : 'Compare 4 Strategies'}
            </button>
          </div>

          {compareResults && (
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              {Object.entries(compareResults.strategies).map(([name, data]) => (
                <div key={name} className="p-4 bg-slate-900 border border-slate-800 rounded-xl space-y-3">
                  <span className="text-xs font-bold font-mono text-cyan-400 uppercase">{name}</span>
                  <div className="space-y-1 text-xs text-slate-300">
                    <div>Latency: <strong>{data.latency_ms} ms</strong></div>
                    <div>Results: <strong>{data.results_count}</strong></div>
                    <div>Top Citation: <strong>{data.top_citation}</strong></div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Benchmark Runner */}
          <div className="p-5 bg-slate-900/90 border border-slate-800 rounded-xl space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="text-sm font-semibold text-slate-200">15-Scenario Synthetic Multimodal Benchmark</h4>
                <p className="text-xs text-slate-400">
                  Tests native text, scanned pages, bilingual OCR, table matrices, diagram retrieval, and refusal handling.
                </p>
              </div>
              <button
                onClick={handleRunBenchmark}
                disabled={benchmarkLoading}
                className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-medium transition"
              >
                {benchmarkLoading ? 'Running...' : 'Run 15 Scenarios'}
              </button>
            </div>

            {benchmarkReport && (
              <div className="space-y-4 pt-3 border-t border-slate-800">
                <div className="grid grid-cols-3 gap-4">
                  <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
                    <span className="text-[11px] text-slate-400 block">Pass Rate</span>
                    <span className="text-xl font-bold font-mono text-emerald-400">
                      {benchmarkReport.pass_rate}% ({benchmarkReport.passed_count}/{benchmarkReport.total_scenarios})
                    </span>
                  </div>
                  <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
                    <span className="text-[11px] text-slate-400 block">Average CER</span>
                    <span className="text-xl font-bold font-mono text-cyan-400">
                      {(benchmarkReport.avg_cer * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
                    <span className="text-[11px] text-slate-400 block">Table Accuracy</span>
                    <span className="text-xl font-bold font-mono text-amber-400">
                      {(benchmarkReport.avg_table_accuracy * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 8: Learning Center */}
      {activeTab === 'learning' && (
        <div className="space-y-6">
          <div className="p-6 bg-slate-900/80 border border-slate-800 rounded-xl space-y-4">
            <h3 className="text-base font-bold text-slate-100">
              Multimodal RAG Engineering Principles
            </h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              Standard RAG systems treat all documents as continuous streams of flat text tokens. However, real-world business and technical documents rely heavily on visual layout, tables, diagrams, headers, and scanned pages. Multimodal RAG bridges this gap through three foundational layers:
            </p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
              <div className="p-4 bg-slate-950 rounded-lg border border-slate-800 space-y-2">
                <span className="text-xs font-bold text-cyan-400">1. Spatial Document Layout</span>
                <p className="text-[11px] text-slate-400">
                  Two-column reading orders and visual bounding boxes ensure paragraphs are not concatenated horizontally across column gutters.
                </p>
              </div>

              <div className="p-4 bg-slate-950 rounded-lg border border-slate-800 space-y-2">
                <span className="text-xs font-bold text-indigo-400">2. Cross-Modal Embeddings</span>
                <p className="text-[11px] text-slate-400">
                  CLIP aligns vision and language in a single metric space ($R^{512}$), enabling queries like "architecture diagram" to match visual elements without relying on manual tags.
                </p>
              </div>

              <div className="p-4 bg-slate-950 rounded-lg border border-slate-800 space-y-2">
                <span className="text-xs font-bold text-amber-400">3. Table Structure Preservation</span>
                <p className="text-[11px] text-slate-400">
                  Flattening tables into unstructured strings destroys cell-header coordinate bindings. Structured Markdown preserves row-column matrix integrity.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default MultimodalLab;
