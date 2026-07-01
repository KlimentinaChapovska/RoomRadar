/**
 * Generic fetch hook with loading, error, and abort-on-unmount support.
 * Wraps the centralised api service.
 */
import { useState, useCallback, useRef } from 'react';

const BASE = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '');

export function useApi(path, method = 'GET') {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const abortRef = useRef(null);

  const run = useCallback(
    async (body = null) => {
      if (abortRef.current) abortRef.current.abort();
      abortRef.current = new AbortController();

      setLoading(true);
      setError(null);

      try {
        const res = await fetch(`${BASE}${path}`, {
          method,
          headers: { 'Content-Type': 'application/json' },
          body: body ? JSON.stringify(body) : undefined,
          signal: abortRef.current.signal,
        });

        if (!res.ok) {
          const payload = await res.json().catch(() => ({}));
          const msg =
            payload?.detail?.message ?? payload?.detail ?? `HTTP ${res.status}`;
          throw Object.assign(new Error(msg), { status: res.status });
        }

        const json = await res.json();
        setData(json);
        return json;
      } catch (err) {
        if (err.name !== 'AbortError') {
          setError(err.message ?? 'Unknown error');
        }
      } finally {
        setLoading(false);
      }
    },
    [path, method]
  );

  return { data, loading, error, run };
}
