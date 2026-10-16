import { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';

export function useHealth(pollIntervalMs = 15000) {
  const [health, setHealth] = useState(null);
  const [readiness, setReadiness] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastChecked, setLastChecked] = useState(null);

  const checkStatus = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [healthData, readinessData] = await Promise.allSettled([
        api.getHealth(),
        api.getReadiness(),
      ]);

      if (healthData.status === 'fulfilled') {
        setHealth(healthData.value);
      } else {
        setHealth(null);
        setError(healthData.reason?.message || 'Backend unreachable');
      }

      if (readinessData.status === 'fulfilled') {
        setReadiness(readinessData.value);
      } else {
        setReadiness(null);
      }
      setLastChecked(new Date());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    checkStatus();
    if (pollIntervalMs > 0) {
      const interval = setInterval(checkStatus, pollIntervalMs);
      return () => clearInterval(interval);
    }
  }, [checkStatus, pollIntervalMs]);

  return {
    health,
    readiness,
    loading,
    error,
    lastChecked,
    refresh: checkStatus,
    isBackendOnline: !!health,
    isDatabaseOnline: !!readiness?.database_connected,
    isPgVectorAvailable: !!readiness?.pgvector_available,
  };
}
