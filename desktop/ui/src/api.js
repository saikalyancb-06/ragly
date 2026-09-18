const BASE = import.meta.env.DEV ? 'http://127.0.0.1:8765' : ''
const TOKEN_KEY = 'edgevault.token'

export const setToken = (t) => (t ? localStorage.setItem(TOKEN_KEY, t) : localStorage.removeItem(TOKEN_KEY))
export const getToken = () => {
  try { return localStorage.getItem(TOKEN_KEY) || '' } catch { return '' }
}

function authHeaders(extra = {}) {
  const token = getToken()
  return token ? { ...extra, Authorization: `Bearer ${token}` } : extra
}

async function req(path, options = {}) {
  const res = await fetch(BASE + path, {
    ...options,
    headers: authHeaders({ 'Content-Type': 'application/json', ...(options.headers || {}) }),
  })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}))
    throw new Error(detail.detail || `${res.status} ${res.statusText}`)
  }
  return res.json()
}
const post = (path, body) => req(path, { method: 'POST', body: JSON.stringify(body ?? {}) })

export const api = {
  base: BASE,
  // auth (local passcode)
  authStatus: () => req('/api/auth/status'),
  register: (passcode, display_name) => post('/api/auth/register', { passcode, display_name }),
  login: (passcode) => post('/api/auth/login', { passcode }),
  checkToken: (token) => post('/api/auth/check', { token }),
  changePasscode: (old, next) => post('/api/auth/change', { old, new: next }),
  disableAuth: (passcode) => post('/api/auth/disable', { passcode }),
  resetAll: (passcode) => post('/api/admin/reset', { passcode, confirm: 'DELETE' }),
  // projects
  projects: () => req('/api/projects'),
  createProject: (name, description = '') => post('/api/projects', { name, description }),
  activateProject: (id) => post(`/api/projects/${id}/activate`),
  renameProject: (id, name, description = '') =>
    req(`/api/projects/${id}`, { method: 'PATCH', body: JSON.stringify({ name, description }) }),
  deleteProject: (id) => req(`/api/projects/${id}`, { method: 'DELETE' }),
  pageText: (docId, page) => req(`/api/documents/${docId}/pagetext?page=${page}`),
  // system
  system: () => req('/api/system'),
  diagnostics: () => req('/api/diagnostics'),
  hardware: () => req('/api/hardware'),
  settings: (values) => (values ? post('/api/settings', values) : req('/api/settings')),
  setEngine: (mode) => post('/api/engine', { mode }),
  restartEngine: () => post('/api/engine/restart'),
  benchmark: (iterations = 3, includeLlm = true) =>
    post(`/api/benchmark?iterations=${iterations}&include_llm=${includeLlm}`),
  benchmarkHistory: () => req('/api/benchmark'),
  evalReports: () => req('/api/eval/reports'),
  evalReport: (name) => req(`/api/eval/report/${name}`),
  // knowledge base
  documents: () => req('/api/documents'),
  deleteDoc: (id) => req(`/api/documents/${id}`, { method: 'DELETE' }),
  reindex: (id) => post(`/api/documents/${id}/reindex`),
  addFromPath: (path) => post('/api/documents/from-path', { path }),
  pack: () => req('/api/pack'),
  retune: () => post('/api/pack/retune'),
  suggestions: () => req('/api/suggestions'),
  visualize: (q, docIds) => req(`/api/visualize${q || docIds ? '?' : ''}`
    + (q ? `q=${encodeURIComponent(q)}` : '')
    + (docIds?.length ? `${q ? '&' : ''}doc_ids=${docIds.join(',')}` : '')),
  async upload(files) {
    const form = new FormData()
    for (const f of files) form.append('files', f)
    const res = await fetch(BASE + '/api/documents', { method: 'POST', body: form, headers: authHeaders() })
    if (!res.ok) throw new Error(`upload failed: ${res.status}`)
    return res.json()
  },
  // retrieval
  search: (q, { topK = 8, tuning = true } = {}) =>
    req(`/api/search?q=${encodeURIComponent(q)}&top_k=${topK}&tuning=${tuning}`),
  route: (question, docIds) => post('/api/route', { question, doc_ids: docIds }),
  // tables
  tables: (docId) => req(`/api/tables${docId ? `?doc_id=${docId}` : ''}`),
  tableCompute: (question, docIds, tableId) =>
    post('/api/tables/compute', { question, doc_ids: docIds, table_id: tableId }),
  // graph
  graphNodes: (kind, limit = 60) =>
    req(`/api/graph/nodes?limit=${limit}${kind ? `&kind=${kind}` : ''}`),
  graphNode: (id, hops = 1) => req(`/api/graph/node/${id}?hops=${hops}`),
  graphSummary: (q) => req(`/api/graph/summary?q=${encodeURIComponent(q)}`),
  // images
  images: (docId) => req(`/api/images${docId ? `?doc_id=${docId}` : ''}`),
  imageFile: (id) => `${BASE}/api/images/${id}/file` +
    (getToken() ? `?token=${encodeURIComponent(getToken())}` : ''),
  // five results is what a person can actually look at; the ranking decides which five
  imageSearch: (query, limit = 5) => post('/api/images/search', { query, limit }),
  async imageSearchByImage(file, mode = 'images', limit = 5) {
    const form = new FormData()
    form.append('file', file)
    const res = await fetch(`${BASE}/api/images/search-by-image?mode=${mode}&limit=${limit}`,
      { method: 'POST', body: form, headers: authHeaders() })
    if (!res.ok) throw new Error(`image search failed: ${res.status}`)
    return res.json()
  },
  // compare
  compare: (left_id, right_id) => post('/api/compare', { left_id, right_id }),
  // pages & voice
  // images are loaded by <img src>, which cannot send headers, so the token rides in the query string
  pageImage: (docId, page, chunkId, dpi = 150) =>
    `${BASE}/api/documents/${docId}/pages/${page}?dpi=${dpi}${chunkId ? `&chunk_id=${chunkId}` : ''}` +
    (getToken() ? `&token=${encodeURIComponent(getToken())}` : ''),
  listen: () => post('/api/voice/listen'),
  speakUrl: () => `${BASE}/api/voice/speak`,
  // streaming answer
  async askStream(body, { onRoute, onSources, onToken, onDone, onError }) {
    const res = await fetch(BASE + '/api/ask/stream', {
      method: 'POST', headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify(body),
    })
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}))
      onError?.(new Error(detail.detail || `${res.status}`))
      return
    }
    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const parts = buffer.split('\n\n')
      buffer = parts.pop()
      for (const part of parts) {
        const line = part.split('\n').find((l) => l.startsWith('data: '))
        if (!line) continue
        let ev
        try { ev = JSON.parse(line.slice(6)) } catch { continue }
        if (ev.type === 'route') onRoute?.(ev.route)
        else if (ev.type === 'sources') onSources?.(ev)
        else if (ev.type === 'token') onToken?.(ev.token)
        else if (ev.type === 'done') onDone?.(ev)
        else if (ev.type === 'error') onError?.(new Error(ev.error))
      }
    }
  },
}
