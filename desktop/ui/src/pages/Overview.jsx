import React, { useState } from 'react'
import { api } from '../api.js'
import { Btn, Card, Empty, Err, Note, Pill, Stat } from '../ui.jsx'

const PERSONAS = [
  { id: 'business', label: 'Business', path: 'samples/business', hint: 'invoices, spreadsheet, contracts' },
  { id: 'manuals', label: 'Field / technical', path: 'samples/manuals', hint: 'manuals, scanned data sheet' },
  { id: 'mixed', label: 'Mixed', path: 'samples', hint: 'contract, lab report, scan, notes' },
]
const kindOf = (name = '') => (name.split('.').pop() || '').toUpperCase()

export default function Overview({ sys, docs, reload, setPage, setSource, onAdd }) {
  const [q, setQ] = useState('')
  const [hits, setHits] = useState(null)
  const [route, setRoute] = useState(null)
  const [computed, setComputed] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const run = async () => {
    if (!q.trim()) return
    setBusy(true); setError(''); setComputed(null)
    try {
      const [r, s, c] = await Promise.all([
        api.route(q), api.search(q, { topK: 8 }), api.tableCompute(q).catch(() => null),
      ])
      setRoute(r); setHits(s.hits)
      if (c?.answered) setComputed(c)
    } catch (e) { setError(String(e.message || e)) }
    setBusy(false)
  }

  const load = async (p) => {
    setBusy(true)
    try { await api.addFromPath(p.path) } catch (e) { setError(String(e.message || e)) }
    setBusy(false); reload()
  }

  return (
    <div className="wrap">
      <div className="grid four" style={{ marginBottom: 14 }}>
        <Stat label="Documents" value={sys?.index?.documents ?? '—'} hint={`${sys?.index?.pages ?? 0} pages indexed`} />
        <Stat label="Text chunks" value={sys?.index?.chunks ?? '—'} hint="Each one keeps its page" />
        <Stat label="Images" value={sys?.images?.images ?? 0} hint={`${sys?.images?.with_vectors ?? 0} searchable`} />
        <Stat label="Entities" value={sys?.graph?.nodes ?? 0} hint={`${sys?.graph?.edges ?? 0} relationships`} />
      </div>

      <Card>
        <div className="searchbig">
          <span style={{ color: 'var(--muted)' }}>⌕</span>
          <input placeholder="Search your private knowledge…" value={q}
                 onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && run()} />
          <Btn kind="primary" onClick={run} disabled={busy}>{busy ? '…' : 'Search'}</Btn>
        </div>
        {error && <div style={{ marginTop: 10 }}><Err>{error}</Err></div>}
        {route && (
          <div className="chips" style={{ marginTop: 11 }}>
            <Pill tone="accent">Routed to {route.routes.join(' + ')}</Pill>
            {Object.entries(route.reasons || {}).map(([k, v]) => <Pill key={k} title={v}>{k}</Pill>)}
          </div>
        )}
      </Card>

      {computed && (
        <Card title="Computed from a table — not generated">
          <div style={{ fontSize: 22, fontWeight: 600 }}>{computed.table.formatted}</div>
          <Note>{computed.table.workings}</Note>
          <div className="chips" style={{ marginTop: 8 }}>
            <Pill tone="good">Deterministic</Pill>
            <Pill>{computed.source.doc_name} · page {computed.source.page}</Pill>
            <Pill>{computed.table.row_count} rows</Pill>
          </div>
        </Card>
      )}

      {hits && (
        <Card title={`Evidence (${hits.length})`}>
          {hits.length === 0 && <Empty>Nothing matched.</Empty>}
          {hits.map((h) => (
            <div className="item click" key={h.chunk_id} onClick={() => setSource({ ...h, n: 1 })}>
              <div className="row">
                <span className="name">{h.doc_name}</span>
                <Pill>Page {h.page}</Pill>
                {h.exact_match && <Pill tone="accent">Exact</Pill>}
                {h.graph_reason && <Pill tone="accent" title={h.graph_reason}>graph</Pill>}
                <div className="spacer" />
                <Pill>{Math.round((h.vector_score || 0) * 100)}%</Pill>
              </div>
              <div className="snippet">{h.text.slice(0, 320)}{h.text.length > 320 ? '…' : ''}</div>
            </div>
          ))}
        </Card>
      )}

      <Card title="Recent documents" actions={<Btn sm onClick={onAdd}>+ Add</Btn>}>
        {docs.length === 0 ? (
          <>
            <Empty>Nothing indexed yet. Load a demo workspace to try it:</Empty>
            <div className="row" style={{ justifyContent: 'center' }}>
              {PERSONAS.map((p) => (
                <Btn key={p.id} onClick={() => load(p)} disabled={busy} title={p.hint}>{p.label}</Btn>
              ))}
            </div>
          </>
        ) : (
          <div className="files">
            {docs.slice(0, 12).map((d) => (
              <div className="file" key={d.id} onClick={() => setPage('documents')}>
                <div className="kind">{kindOf(d.name)}</div>
                <div className="fname">{d.name}</div>
                <div className="fmeta">
                  {d.pages} pages · {d.chunk_count} chunks
                  {d.table_count > 0 ? ` · ${d.table_count} tables` : ''}
                  {d.image_count > 0 ? ` · ${d.image_count} images` : ''}
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  )
}
