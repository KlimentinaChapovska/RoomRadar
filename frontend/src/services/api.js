/**
 * Centralised API service.
 * All requests go through this module so the base URL is configured in one place.
 * Set VITE_API_URL in .env.local; falls back to an empty string (same origin).
 */

const BASE = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '');

function log(level, msg, data) {
  if (import.meta.env.DEV) {
    // eslint-disable-next-line no-console
    console[level](`[RoomRadar API] ${msg}`, data ?? '');
  }
}

async function request(path, options = {}) {
  const url = `${BASE}${path}`;
  log('info', `${options.method ?? 'GET'} ${url}`);

  let res;
  try {
    res = await fetch(url, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    });
  } catch (networkErr) {
    log('error', 'Network error', networkErr);
    throw new Error('Cannot reach the API server. Check that the backend is running.');
  }

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const payload = await res.json();
      const d = payload?.detail;
      if (Array.isArray(d)) {
        // Pydantic 422: [{loc, msg, type}, ...]
        detail = d.map(e => e.msg ?? String(e)).join('; ');
      } else {
        detail = d?.message ?? (typeof d === 'string' ? d : detail);
      }
    } catch (_) { /* ignore parse errors */ }
    log('error', `${res.status} from ${url}`, detail);
    throw Object.assign(new Error(detail), { status: res.status });
  }

  return res.json();
}

export const api = {
  /** POST /api/v1/predict */
  predict: (payload) =>
    request('/api/v1/predict', { method: 'POST', body: JSON.stringify(payload) }),

  /** GET /health */
  health: () => request('/health'),

  /** GET /api/v1/model-info */
  modelInfo: () => request('/api/v1/model-info'),
};
