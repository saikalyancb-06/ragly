import React, { useEffect, useMemo, useState } from 'react'
import { api } from '../api.js'
import GraphCanvas from '../components/GraphCanvas.jsx'
import Graph from './Graph.jsx'
import { Btn, Card, Empty, Err, Note, Pill } from '../ui.jsx'

// The questions a person actually asks of a document. Each one makes the planner choose a
// different picture — that is the point of the planner, so the buttons demonstrate it.
const MODES = [
  { id: 'auto', label: 'Auto', q: '' },
  { id: 'financial', label: 'Financial', q: 'Show financial information' },
  { id: 'dates', label: 'Dates', q: 'Show important dates' },
  { id: 'relationships', label: 'Relationships', q: 'Show relationships' },
  { id: 'charts', label: 'Charts', q: 'Show the charts' },
  { id: 'tables', label: 'Tables', q: 'Show the tables' },
]

const TYPE_LABEL = {
  document_overview: 'Overview',
  financial_summary: 'Financial summary',
  timeline: 'Timeline',
  relationship_graph: 'Relationships',
  focused_path: 'Focused relationships',
  table_visualization: 'Tables',
  image_map: 'Images',
  document_entity_map: 'What is in the documents',
  comparison_view: 'Comparison',
}

const prettyDate = (iso, fallback) => {
  if (!iso) return fallback
  const d = new Date(iso)
  return Number.isNaN(d.getTime())
    ? fallback
    : d.toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' })
}

/** One labelled value. The label is the document's own wording, never a node id.
 *  `big` is the headline panel: one card per meaning, not one per number found. */
function FactCard({ item, onEvidence, big }) {
  const ev = item.evidence || {}
  const refs = item.references || 1
  return (
    <button className={`vcard${big ? ' big' : ''}`} onClick={() => onEvidence({ ...item, evidence: ev })}>
      <div className="vcard-label">{item.label}</div>
      <div className="vcard-value">{item.value}</div>
      <div className="vcard-src">
        {ev.doc_name ? `${ev.doc_name}${ev.page ? ` · page ${ev.page}` : ''}` : ''}
        {refs > 1 && <span className="vchip">{refs} references</span>}

      </div>
    </button>
  )
}

function Timeline({ items, onEvidence }) {
  if (!items?.length) return <Empty>No dates could be read from these documents.</Empty>
  return (
    <ol className="vtimeline">
      {items.map((it, i) => (
        <li key={`${it.date}-${it.label}-${i}`}>
          <span className="vdot" />
          <div className="vwhen">{prettyDate(it.date, it.value)}</div>
          <button className="vwhat" onClick={() => onEvidence(it)}>
            <span>{it.label}</span>
            {it.evidence?.doc_name && (
              <span className="vsrc">{it.evidence.doc_name} · page {it.evidence.page}</span>)}
          </button>
        </li>
      ))}
    </ol>
  )
}

/** Charts, drawn as plain SVG — no chart library, nothing fetched, works offline.
 *  Three shapes only: bars to compare, grouped bars to compare two or three columns, and a
 *  line when the x axis is time. Every bar carries its value as the document printed it. */

const PALETTE = ['#5b8cff', '#3ddc97', '#ffb454', '#c58bff']

const fmt = (value, unit) => {
  const n = Number(value)
  const shown = Math.abs(n) >= 1000 ? n.toLocaleString('en-IN') : String(n)
  if (unit === '₹') return `₹${shown}`
  if (unit === '%') return `${shown}%`
  return unit ? `${shown} ${unit}` : shown
}

function Bars({ chart, onPoint }) {
  const series = chart.series || []
  const labels = series[0].points.map((p) => p.label)
  const max = Math.max(...series.flatMap((s) => s.points.map((p) => Math.abs(p.value)))) || 1
  return (
    <div className="vbars">
      {labels.map((label, i) => (
        <div className="vbar-row" key={`${label}-${i}`}>
          <span className="vbar-label" title={label}>{label}</span>
          <span className="vbar-stack">
            {series.map((s, j) => {
              const point = s.points[i]
              if (!point) return null
              return (
                <button className="vbar-track" key={s.name}
                        title={`${s.name}: ${point.display || fmt(point.value, chart.unit)}`}
                        onClick={() => onPoint?.({ ...point, series: s.name, unit: chart.unit })}>
                  <span className="vbar-fill"
                        style={{ width: `${Math.max(1.5, (Math.abs(point.value) / max) * 100)}%`,
                                 background: PALETTE[j % PALETTE.length] }} />
                </button>
              )
            })}
          </span>
          <span className="vbar-value">
            {series.map((s) => s.points[i]).filter(Boolean)
              .map((p) => p.display || fmt(p.value, chart.unit)).join(' · ')}
          </span>
        </div>
      ))}
    </div>
  )
}

function Line({ chart }) {
  const W = 640, H = 220, padL = 52, padR = 16, padT = 14, padB = 34
  const series = chart.series || []
  const points = series[0].points
  const values = series.flatMap((s) => s.points.map((p) => p.value))
  const max = Math.max(...values), min = Math.min(...values, 0)
  const span = max - min || 1
  const x = (i) => padL + (i * (W - padL - padR)) / Math.max(1, points.length - 1)
  const y = (v) => H - padB - ((v - min) / span) * (H - padT - padB)
  return (
    <svg className="vline" viewBox={`0 0 ${W} ${H}`} role="img"
         aria-label={chart.title || 'chart'} preserveAspectRatio="xMidYMid meet">
      {[0, 0.5, 1].map((t) => {
        const value = min + span * t
        return (
          <g key={t}>
            <line x1={padL} x2={W - padR} y1={y(value)} y2={y(value)} className="vgridline" />
            <text x={padL - 8} y={y(value) + 4} className="vaxis" textAnchor="end">
              {fmt(Math.round(value), chart.unit)}
            </text>
          </g>
        )
      })}
      {series.map((s, j) => (
        <g key={s.name}>
          <polyline fill="none" stroke={PALETTE[j % PALETTE.length]} strokeWidth="2.5"
                    points={s.points.map((p, i) => `${x(i)},${y(p.value)}`).join(' ')} />
          {s.points.map((p, i) => (
            <circle key={i} cx={x(i)} cy={y(p.value)} r="3.5" fill={PALETTE[j % PALETTE.length]}>
              <title>{`${p.label}: ${p.display || fmt(p.value, chart.unit)}`}</title>
            </circle>
          ))}
        </g>
      ))}
      {points.map((p, i) => (
        (points.length <= 8 || i % Math.ceil(points.length / 8) === 0) && (
          <text key={i} x={x(i)} y={H - 10} className="vaxis" textAnchor="middle">{p.label}</text>
        )
      ))}
    </svg>
  )
}

function Donut({ chart }) {
  const points = chart.series[0].points
  const total = chart.total || points.reduce((a, p) => a + p.value, 0)
  let angle = -Math.PI / 2
  const R = 80, r = 48, C = 100
  const arcs = points.map((p, i) => {
    const slice = (p.value / total) * Math.PI * 2
    const [a0, a1] = [angle, angle + slice]
    angle = a1
    const big = slice > Math.PI ? 1 : 0
    const pt = (rad, ang) => `${C + rad * Math.cos(ang)},${C + rad * Math.sin(ang)}`
    return {
      d: `M ${pt(R, a0)} A ${R} ${R} 0 ${big} 1 ${pt(R, a1)} L ${pt(r, a1)} A ${r} ${r} 0 ${big} 0 ${pt(r, a0)} Z`,
      colour: PALETTE[i % PALETTE.length], p, share: Math.round((p.value / total) * 100),
    }
  })
  return (
    <div className="vdonut">
      <svg viewBox="0 0 200 200" role="img" aria-label={chart.title}>
        {arcs.map((a, i) => (
          <path key={i} d={a.d} fill={a.colour}>
            <title>{`${a.p.label}: ${a.p.display || fmt(a.p.value, chart.unit)} (${a.share}%)`}</title>
          </path>
        ))}
      </svg>
      <div className="vlegend col">
        {arcs.map((a, i) => (
          <span key={i}><i style={{ background: a.colour }} />{a.p.label}
            <b>{a.p.display || fmt(a.p.value, chart.unit)}</b>
            <em>{a.share}%</em></span>
        ))}
      </div>
    </div>
  )
}

function Chart({ chart, onPoint }) {
  if (!chart?.series?.length) return null
  const multi = chart.series.length > 1
  return (
    <div className="vchart">
      {multi && (
        <div className="vlegend">
          {chart.series.map((s, j) => (
            <span key={s.name}><i style={{ background: PALETTE[j % PALETTE.length] }} />{s.name}</span>
          ))}
        </div>
      )}
      {chart.kind === 'line' ? <Line chart={chart} />
        : chart.kind === 'donut' ? <Donut chart={chart} />
        : <Bars chart={chart} onPoint={onPoint} />}
      {(chart.note || chart.x) && (
        <Note>{[chart.x && `${chart.y || 'Value'} by ${chart.x}`, chart.note].filter(Boolean).join(' · ')}</Note>
      )}
    </div>
  )
}

function Relationships({ graph, onEvidence }) {
  const nodes = (graph?.nodes || []).map((n) => ({ ...n, name: n.name, kind: n.kind }))
  const edges = (graph?.edges || []).map((e) => ({ src: e.source, dst: e.target, rel: e.label }))
  const byId = new Map(nodes.map((n) => [n.id, n]))
  if (!nodes.length) return <Empty>No connection here can be stated in plain English yet.</Empty>
  return (
    <>
      <GraphCanvas nodes={nodes} edges={edges} height={360}
                   onSelect={(n) => onEvidence({
                     label: n.type || 'Entity', value: n.name,
                     evidence: { method: 'entity extraction', confidence: 1,
                                 text: `Mentioned ${n.mentions || 0} times in these documents.` } })} />
      <div className="vrels">
        {(graph.edges || []).map((e, i) => (
          <button className={`vrel ${e.origin}`} key={i} onClick={() => onEvidence({
            label: 'Relationship',
            value: `${byId.get(e.source)?.name || '?'} ${e.label} ${byId.get(e.target)?.name || '?'}`,
            role: e.origin === 'inferred' ? 'INFERRED' : 'STATED',
            evidence: { ...e.evidence, confidence: e.confidence, method: e.origin === 'inferred'
              ? 'Inferred from document context' : 'Stated in the document' },
          })}>
            <b>{byId.get(e.source)?.name || '?'}</b>
            <span className="vrel-verb">{e.label}</span>
            <b>{byId.get(e.target)?.name || '?'}</b>
            {e.origin === 'inferred' && <span className="vtag">inferred</span>}
          </button>
        ))}
      </div>
      {graph.hidden > 0 && <Note>{graph.hidden} less-mentioned entities are hidden to keep this readable.</Note>}
    </>
  )
}

export default function Visualize({ sys, docs, setSource, setPage }) {
  const [q, setQ] = useState('')
  const [asked, setAsked] = useState('')
  const [plan, setPlan] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [evidence, setEvidence] = useState(null)
  const [raw, setRaw] = useState(false)
  const [mode, setMode] = useState('auto')
  const [scope, setScope] = useState('')   // '' = let the planner scope it, 'all' = every document

  const load = async (question, docScope = scope) => {
    setBusy(true); setError(''); setEvidence(null)
    try {
      const ids = docScope && docScope !== 'all' ? [Number(docScope)] : undefined
      // "all" asks the planner to look across documents; a single id pins it to one
      setPlan(await api.visualize(docScope === 'all' ? `${question || ''} all documents`.trim() : question, ids))
      setAsked(question || '')
    } catch (e) { setError(String(e.message || e)) } finally { setBusy(false) }
  }
  useEffect(() => { load('') }, [])

  const sections = plan?.sections || []
  const openSource = (ev) => {
    if (!ev?.doc_id && !ev?.doc_name) return
    setSource({ doc_id: ev.doc_id, doc_name: ev.doc_name, page: ev.page || 1, n: 1, text: ev.text || '' })
  }

  return (
    <div className="wrap">
      <Card>
        <div className="row" style={{ gap: 10, alignItems: 'center', marginBottom: 12 }}>
          <div>
            <h2 style={{ margin: 0 }}>{plan?.title || 'Visualize'}</h2>
            <Note>{plan?.summary || 'Ask for the picture you want — dates, money, relationships, tables.'}</Note>
          </div>
          <div className="spacer" />
          {plan?.type && <Pill tone="accent">{TYPE_LABEL[plan.type] || plan.type}</Pill>}
        </div>

        {(docs || []).length > 1 && (
          <div className="row" style={{ gap: 8, alignItems: 'center', marginBottom: 10 }}>
            <span className="sectionLabel" style={{ margin: 0 }}>Document</span>
            <select className="vselect" value={scope}
                    onChange={(e) => { setScope(e.target.value); load(q, e.target.value) }}>
              <option value="">Most recent</option>
              {(docs || []).map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
              <option value="all">All documents</option>
            </select>
          </div>
        )}

        <div className="composer" style={{ marginBottom: 12 }}>
          <input placeholder="What do you want to see? e.g. show the loan terms"
                 value={q} disabled={busy}
                 onChange={(e) => setQ(e.target.value)}
                 onKeyDown={(e) => e.key === 'Enter' && (setMode(''), load(q))} />
          <Btn kind="primary" onClick={() => { setMode(''); load(q) }} disabled={busy}>Show</Btn>
        </div>
        <div className="vmodes">
          {MODES.map((m) => (
            <button key={m.id} className={`vmode${mode === m.id ? ' on' : ''}`}
                    onClick={() => { setMode(m.id); setQ(m.q); load(m.q) }}>{m.label}</button>
          ))}
        </div>
        {asked && <Note>Showing: “{asked}”</Note>}
        <Err>{error}</Err>
      </Card>

      {busy && <Card><Empty>Reading the documents…</Empty></Card>}

      {!busy && sections.map((s, i) => (
        <Card key={`${s.title}-${i}`} title={s.title}>
          {s.kind === 'kpi' && (
            <div className="vgrid kpi">
              {s.items.map((it, j) => (
                <FactCard key={`${it.label}-${it.value}-${j}`} item={it} onEvidence={setEvidence} big />
              ))}
            </div>
          )}
          {s.kind === 'facts' && (
            <div className="vgrid">
              {s.items.map((it, j) => <FactCard key={`${it.label}-${j}`} item={it} onEvidence={setEvidence} />)}
            </div>
          )}
          {s.kind === 'timeline' && <Timeline items={s.items} onEvidence={setEvidence} />}
          {s.kind === 'chart' && <Chart chart={s.chart} onPoint={(p) => setEvidence({
            label: p.label, value: p.display || p.value,
            evidence: { ...(p.evidence || s.evidence || {}), method: 'read from the document' } })} />}
          {s.kind === 'graph' && <Relationships graph={s.graph} onEvidence={setEvidence} />}
          {s.kind === 'table' && (
            <>
              <div className="tablewrap">
                <table className="tbl">
                  <thead><tr>{(s.headers || []).map((h, k) => <th key={k}>{h}</th>)}</tr></thead>
                  <tbody>
                    {(s.rows || []).map((r, k) => (
                      <tr key={k}>{(s.headers || []).map((_, c) => <td key={c}>{r[c] ?? ''}</td>)}</tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {s.chart && <Chart chart={s.chart} />}
              {s.evidence?.doc_name && (
                <Note>{s.evidence.doc_name} · page {s.evidence.page}</Note>)}
            </>
          )}
        </Card>
      ))}

      {!busy && !sections.length && plan && (
        <Card><Empty>{plan.notes?.[0] || 'Nothing to show yet — add a document first.'}</Empty></Card>
      )}

      {plan?.notes?.length > 0 && (
        <Card><div className="vnotes">{plan.notes.map((n, i) => <Note key={i}>{n}</Note>)}</div></Card>
      )}

      {evidence && (
        <Card title="Where this comes from" actions={<Btn sm onClick={() => setEvidence(null)}>Close</Btn>}>
          <div className="vev">
            <div><span className="vev-k">What is this?</span><span>{evidence.label}{evidence.value ? ` — ${evidence.value}` : ''}</span></div>
            <div><span className="vev-k">Source</span>
              <span>{evidence.evidence?.doc_name || 'this workspace'}
                {evidence.evidence?.page ? ` · page ${evidence.evidence.page}` : ''}</span></div>
            <div><span className="vev-k">How it was found</span><span>{evidence.evidence?.method || 'extraction'}</span></div>
            <div><span className="vev-k">Confidence</span><span>{evidence.evidence?.confidence ?? '—'}</span></div>
            {evidence.evidence?.text && (
              <div><span className="vev-k">Evidence</span><span className="vev-quote">“{evidence.evidence.text}”</span></div>)}
            {evidence.conflict && (
              <Note>This document states more than one value under this heading. Both are shown;
                open each to see the page it came from.</Note>)}
            {evidence.role === 'INFERRED' && <Note>Inferred from document context — not stated outright.</Note>}
          </div>
          {(evidence.evidence?.doc_id || evidence.evidence?.doc_name) && (
            <Btn sm onClick={() => openSource(evidence.evidence)}>Open source</Btn>)}
        </Card>
      )}

      {plan?.debug && (
        <Card>
          <h3 className="sectionLabel" style={{ marginBottom: 10 }}>Technical details</h3>
          {true && (
            <div className="vev" style={{ marginTop: 12 }}>
              <div><span className="vev-k">Visualization</span><span>{plan.debug.visualization}</span></div>
              <div><span className="vev-k">Why</span><span>{plan.debug.reason}</span></div>
              <div><span className="vev-k">Documents in scope</span>
                <span>{(plan.debug.documents_in_scope || []).join(', ') || '—'}
                  {plan.debug.documents_ignored > 0 && ` (${plan.debug.documents_ignored} not included)`}</span></div>
              <div><span className="vev-k">Facts used</span><span>{plan.debug.facts_kept}</span></div>
              <div><span className="vev-k">Values with no stated label</span>
                <span>{plan.debug.facts_without_a_stated_label} — not shown
                  {plan.debug.unlabelled_examples?.length ? `: ${plan.debug.unlabelled_examples.join(', ')}` : ''}</span></div>
              <div><span className="vev-k">Chart</span>
                <span>{plan.debug.chart || 'none'} — {plan.debug.chart_reason}</span></div>
              <div><span className="vev-k">Relationships</span>
                <span>{plan.debug.relationship_edges} drawn — {plan.debug.relationship_rule}</span></div>
            </div>
          )}
        </Card>
      )}

      <Card>
        <Btn sm onClick={() => setRaw(!raw)}>{raw ? 'Hide' : 'Show'} the raw knowledge graph</Btn>
        <Note>The graph is how the documents are stored internally. The views above are built from it.</Note>
        {raw && <div style={{ marginTop: 12 }}><Graph setPage={setPage} setSource={setSource} /></div>}
      </Card>
    </div>
  )
}
