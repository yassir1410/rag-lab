import { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';

export function useDocuments() {
  const [documents, setDocuments] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Selected document detail state
  const [selectedDocId, setSelectedDocId] = useState(null);
  const [selectedDocDetails, setSelectedDocDetails] = useState(null);
  const [selectedDocSections, setSelectedDocSections] = useState([]);
  const [selectedDocContent, setSelectedDocContent] = useState(null);
  const [loadingDetails, setLoadingDetails] = useState(false);

  // Upload state
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [uploadSuccess, setUploadSuccess] = useState(null);

  const fetchDocuments = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.listDocuments(50, 0);
      setDocuments(data.items || []);
      setTotal(data.total || 0);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  const loadDocumentDetails = useCallback(async (docId) => {
    setSelectedDocId(docId);
    setLoadingDetails(true);
    try {
      const [docData, sectionsData, contentData] = await Promise.all([
        api.getDocument(docId),
        api.getDocumentSections(docId),
        api.getDocumentContent(docId),
      ]);
      setSelectedDocDetails(docData);
      setSelectedDocSections(sectionsData || []);
      setSelectedDocContent(contentData);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingDetails(false);
    }
  }, []);

  const closeDetails = () => {
    setSelectedDocId(null);
    setSelectedDocDetails(null);
    setSelectedDocSections([]);
    setSelectedDocContent(null);
  };

  const uploadFile = async (file) => {
    setIsUploading(true);
    setUploadError(null);
    setUploadSuccess(null);
    try {
      const result = await api.uploadDocument(file);
      setUploadSuccess(`"${result.original_filename}" processed successfully (${result.sections_count} sections extracted).`);
      await fetchDocuments();
      return result;
    } catch (err) {
      setUploadError(err.message);
      throw err;
    } finally {
      setIsUploading(false);
    }
  };

  const deleteDocument = async (docId) => {
    try {
      await api.deleteDocument(docId);
      if (selectedDocId === docId) {
        closeDetails();
      }
      await fetchDocuments();
    } catch (err) {
      setError(err.message);
      throw err;
    }
  };

  return {
    documents,
    total,
    loading,
    error,
    refresh: fetchDocuments,
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
  };
}
