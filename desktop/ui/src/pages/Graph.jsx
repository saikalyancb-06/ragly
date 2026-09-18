import React, { useEffect, useState } from 'react'
import { api } from '../api.js'
import GraphCanvas from '../components/GraphCanvas.jsx'
import { Btn, Card, Empty, Err, Note, Pill, Switch } from '../ui.jsx'

const KINDS = ['org', 'person', 'invoice', 'contract', 'project', 'place', 'amount', 'date']

export default function Graph({ setPage, setPrefill, setSource }) {
  const [kind, setKind] = useState('')
  const [nodes, setNodes] = useState([])
  const [pool, setPool] = useState([])
  const [edges, setEdges] = useState([])
  const [counts, setCounts] = useState(null)
  const [selected, setSelected] = useState(null)
  const [weak, setWeak] = useState(false)
  const [summary, setSummary] = useState(null)
  const [q, setQ] = useState('')
  const [error, setError] = useState('')

  const loadNodes = async (k) => {
    try {
      const r = await api.graphNodes(k || undefined, 60)
      setCounts(r.counts); setPool(r.nodes)
      if (r.nodes.length) select(r.nodes[0])
      else { setNodes([]); setEdges([]); setSelected(null); setSummary(null) }
    } catch (e) { setError(String(e.message || e)) }
  }
  useEffect(() => { loadNodes(kind) }, [kind])

  const select = async (node) => {
    setSelected(node); setError('')
    try {
      const [n, s] = await Promise.all([api.graphNode(node.id, 1), api.graphSummary(node.name)])
      setSummary(s)
      // the canvas shows this entity's neighbourhood only — the whole graph is unreadable at once
      const known = new Map((n.nodes || []).map((x) => [x.id, x]))
      known.set(node.id, node)
      setNodes([...known.values()])
      setEdges(n.edges || [])
    } catch (e) { setError(String(e.message || e)) }
  }

  const search = async () => {
    if (!q.trim()) return
    try {
      const s = await api.graphSummary(q)
      setSummary(s)
      if (s.entity) await select(s.entity)
    } catch (e) { setError(String(e.message || e)) }
  }

  const shownEdges = weak ? edges : edges.filter((e) => e.rel !== 'co_occurs_with')
  const rel = summary?.related || {}
  return (
    <div className="wrap">
      <Card>
        <div className="row">
          <input placeholder="Everything related to…  e.g. ABC Ltd" value={q}
                 onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && search()}
                 style={{ flex: 1, minWidth: 220 }} />
          <Btn kind="primary" onClick={search}>Explore</Btn>
          <div className="spacer" />
          <Pill tone={kind === '' ? 'accent' : ''} onClick={() => setKind('')}>all</Pill>
          {KINDS.map((k) => <Pill key={k} tone={kind === k ? 'accent' : ''} onClick={() => setKind(k)}>{k}</Pill>)}
        </div>
        <div className="row" style={{ marginTop: 10 }}>
          <Switch on={weak} onChange={setWeak} label="Show weak links (co-occurrence)" />
          {counts && <Note>
            {counts.nodes} entities · {counts.edges} relationships · the canvas shows the selected entity's
            neighbourhood — click a node to move there, scroll to zoom, drag to pan
          </Note>}
        </div>
        {pool.length > 0 && (
          <div className="chips" style={{ marginTop: 10 }}>
            {pool.slice(0, 24).map((n) => (
              <Pill key={n.id} tone={selected?.id === n.id ? 'accent' : ''} onClick={() => select(n)}
                    title={`${n.kind} · seen ${n.mentions}×`}>{n.name}</Pill>
            ))}
          </div>
        )}
        {error && <div style={{ marginTop: 10 }}><Err>{error}</Err></div>}
      </Card>

      {nodes.length === 0 ? (
        <Card><Empty>No entities yet — import documents first.</Empty></Card>
      ) : (
        <div className="grid split">
          <GraphCanvas
            nodes={nodes.filter((n) => weak || shownEdges.some((e) => e.src === n.id || e.dst === n.id) || n.id === selected?.id)}
            edges={shownEdges} selectedId={selected?.id} onSelect={(n) => select(n)} />
          <div>
            {summary && (
              <Card title={summary.entity?.name || q}>
                <div className="row">
                  <Pill tone="accent">{summary.entity?.kind}</Pill>
                  <Pill>Seen {summary.entity?.mentions}×</Pill>
                  <Pill>{(summary.documents || []).length} documents</Pill>
                  {(summary.invoices || []).length > 0 && <Pill>{summary.invoices.length} invoices</Pill>}
                  <Pill>{(summary.edges || []).length} links</Pill>
                </div>
                <Btn sm style={{ marginTop: 10 }}
                     onClick={() => { setPrefill(`Everything related to ${summary.entity?.name}`); setPage('ask') }}>
                  Ask about this entity
                </Btn>
                {Object.entries(rel).filter(([, v]) => Array.isArray(v) && v.length).map(([group, vals]) => (
                  <div key={group} style={{ marginTop: 12 }}>
                    <Note>{group}</Note>
                    <div className="chips" style={{ marginTop: 5 }}>
                      {vals.slice(0, 14).map((v, i) => <Pill key={i}>{String(v)}</Pill>)}
                    </div>
                  </div>
                ))}
                {(summary.invoices || []).length > 0 && (
                  <div style={{ marginTop: 12 }}>
                    <Note>invoices</Note>
                    {summary.invoices.slice(0, 6).map((inv, i) => (
                      <div className="kv" key={i}>
                        <span>{inv.invoice}{inv.issued_by ? ` · ${inv.issued_by}` : ''}</span>
                        <span>{(inv.amounts || []).slice(-1)[0] || '—'}</span>
                      </div>
                    ))}
                  </div>
                )}
                {(summary.documents || []).length > 0 && (
                  <div style={{ marginTop: 12 }}>
                    <Note>documents</Note>
                    <div className="chips" style={{ marginTop: 5 }}>
                      {summary.documents.slice(0, 10).map((d) => (
                        <Pill key={d.doc_id} onClick={() => setSource({ doc_id: d.doc_id, doc_name: d.doc_name,
                          page: (d.pages || [1])[0], n: 1, text: '', match_reason: 'entity mentioned on this page' })}>
                          {d.doc_name}
                        </Pill>
                      ))}
                    </div>
                  </div>
                )}
              </Card>
            )}
            {shownEdges.length > 0 && (
              <Card title="Relationships">
                <table className="tbl">
                  <thead><tr><th>from</th><th>relation</th><th>to</th><th className="num">page</th></tr></thead>
                  <tbody>
                    {shownEdges.slice(0, 20).map((e, i) => (
                      <tr key={i}><td>{e.src_name}</td><td><Pill>{e.rel}</Pill></td><td>{e.dst_name}</td>
                        <td className="num">{e.page ?? '—'}</td></tr>
                    ))}
                  </tbody>
                </table>
                <Note style={{ marginTop: 8 }}>
                  Extracted by local rules (company suffixes, honorifics, invoice and amount patterns) — not a
                  trained NER model.
                </Note>
              </Card>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
