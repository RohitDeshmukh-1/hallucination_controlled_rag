const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Request failed');
  }
  return res.json();
}

export const api = {
  // Health
  health: () => request('/health'),

  // Index
  indexStatus: () => request('/index/status'),
  clearIndex: () => request('/index', { method: 'DELETE' }),

  // Upload
  uploadPdf: (file, sessionId = 'default') => {
    const form = new FormData();
    form.append('file', file);
    return fetch(`${BASE}/upload?session_id=${sessionId}`, {
      method: 'POST',
      body: form,
    }).then(r => {
      if (!r.ok) return r.json().then(e => Promise.reject(new Error(e.detail)));
      return r.json();
    });
  },

  // Ask
  ask: (question, sessionId = 'default', opts = {}) =>
    request('/ask', {
      method: 'POST',
      body: JSON.stringify({
        question,
        session_id: sessionId,
        enable_nli: opts.enableNli ?? false,
        use_memory_context: opts.useMemory ?? true,
      }),
    }),

  // Sessions
  createSession: (name) =>
    request('/session/create', {
      method: 'POST',
      body: JSON.stringify({ session_name: name }),
    }),
  getSession: (id) => request(`/session/${id}`),
  getHistory: (id) => request(`/session/${id}/history`),
  listSessions: () => request('/sessions'),
  clearSession: (id) => request(`/session/${id}/clear`, { method: 'DELETE' }),

  // Pins
  addPin: (sessionId, text, sourceQuestion, fromDoc) =>
    request(`/session/${sessionId}/pin`, {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId, text, source_question: sourceQuestion, from_doc: fromDoc }),
    }),
  removePin: (sessionId, pinId) =>
    request(`/session/${sessionId}/pin/${pinId}`, { method: 'DELETE' }),
};
