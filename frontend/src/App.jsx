import React, { useState } from 'react';
import { Sidebar } from './components/layout/Sidebar';
import { Header } from './components/layout/Header';
import { Overview } from './pages/Overview';
import { Documents } from './pages/Documents';
import { ChunkingLab } from './pages/ChunkingLab';
import { EmbeddingsLab } from './pages/EmbeddingsLab';
import { RetrievalLab } from './pages/RetrievalLab';
import { RagChat } from './pages/RagChat';
import { EvaluationStudio } from './pages/EvaluationStudio';
import {
  ExperimentsPage,
  SettingsPage,
} from './pages/ComingSoonPages';
import { useHealth } from './hooks/useHealth';
import { useDocuments } from './hooks/useDocuments';

export default function App() {
  const [currentTab, setCurrentTab] = useState('overview');
  const healthState = useHealth(15000);
  const docState = useDocuments();

  const renderContent = () => {
    switch (currentTab) {
      case 'overview':
        return (
          <Overview
            healthState={healthState}
            onNavigateToDocuments={() => setCurrentTab('documents')}
          />
        );
      case 'documents':
        return <Documents docState={docState} />;
      case 'chunking':
        return <ChunkingLab />;
      case 'embeddings':
        return <EmbeddingsLab />;
      case 'retrieval':
        return <RetrievalLab />;
      case 'rag-chat':
        return <RagChat />;
      case 'evaluation':
        return <EvaluationStudio />;
      case 'experiments':
        return <ExperimentsPage />;
      case 'settings':
        return <SettingsPage />;
      default:
        return <Overview healthState={healthState} />;
    }
  };

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 overflow-hidden font-sans">
      {/* Sidebar */}
      <Sidebar currentTab={currentTab} onSelectTab={setCurrentTab} />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <Header healthState={healthState} />
        <main className="flex-1 overflow-y-auto p-6 md:p-8 bg-slate-900/60">
          {renderContent()}
        </main>
      </div>
    </div>
  );
}
