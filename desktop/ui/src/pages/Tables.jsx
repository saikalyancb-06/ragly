import React, { useEffect, useRef, useState } from 'react'
import { api } from '../api.js'
import { Btn, Card, Empty, Err, Note, Pill } from '../ui.jsx'

export default function Tables({ setSource, settings }) {
  const [tables, setTables] = useState([])
  const [open, setOpen] = useState(null)
  const [q, setQ] = useState('')
  const [res, setRes] = useState(null)
  const [error, setError] = useState('')
  const showWorking = Boolean(settings?.show_evidence)
  const panel = useRef(null)

  useEffect(() => { api.tables().then((r) => setTables(r.tables)).catch(() => {}) }, [])

  const t = open != null ? tables.find((x) => x.id === open) : null

  // A column of bare figures reads best right-aligned and in tabular figures; one carrying
  // units ("18 cm") or words reads best left-aligned. The header follows its column, so the
  // label always sits over the values it belongs to.
  const align = (t?.headers || []).map((_h, j) => {
    const cells = (t?.rows || []).map((r) => String(r[j] ?? '').trim()).filter(Boolean)
    if (!cells.length) return ''
    const bare = cells.filter((c) => /^[₹$€£]?\s*-?[\d,]+(\.\d+)?%?$/.test(c) || c === '—' || c === '-')
    return bare.length === cells.length ? 'num' : ''
  })

  // Opening a table scrolls straight to it: otherwise it appears below the fold and looks
  // as though the click did nothing.
  useEffect(() => {
    if (open != null) {
      const id = setTimeout(() => panel.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 60)
      return () => clearTimeout(id)
    }
    return undefined
  }, [open])

  // A question asked from inside a table is answered from that document's tables only.
  const compute = async () => {
    if (!q.trim() || !t) return
    setError(''); setRes(null)
    try {
      const r = await api.tableCompute(q, [t.doc_id], t.id)
      if (r.answered) setRes(r)
      else setError(r.reason || 'That table does not answer this question.')
    } catch (e) { setError(String(e.message || e)) }
  }

  const select = (id) => {
    setOpen(open === id ? null : id)
    setQ(''); setRes(null); setError('')
  }

  return (
    <div className="wrap">
      <Card title={`Tables found in your documents (${tables.length})`}>
        <Note>
          Click a table to see its rows, then ask a question about it — totals, averages, counts
          and filters are calculated in Python over the extracted rows, never by the language model.
        </Note>
        {tables.length === 0 && (
          <Empty>No tables yet. Import a spreadsheet, or a PDF containing tables.</Empty>
        )}
        <div style={{ marginTop: 10 }}>
          {tables.map((tb) => (
            <div className={`item click ${open === tb.id ? 'sel' : ''}`} key={tb.id}
                 onClick={() => select(tb.id)}>
              <div className="row">
                <span className="name">{tb.title || `Table on page ${tb.page}`}</span>
                <Pill>{tb.doc_name}</Pill>
                <Pill>Page {tb.page}</Pill>
                <Pill>{tb.n_rows} × {tb.n_cols}</Pill>
                <div className="spacer" />
                {tb.numeric_cols?.length > 0 && (
                  <Pill tone="accent">numbers: {tb.numeric_cols.join(', ')}</Pill>)}
                <Pill>{open === tb.id ? 'Close' : 'Open'}</Pill>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {t && (
        <div ref={panel}>
        <Card title={`${t.title || 'Table'} — ${t.doc_name}, page ${t.page}`}
              actions={<Btn sm kind="ghost"
                            onClick={() => setSource({ doc_id: t.doc_id, doc_name: t.doc_name, page: t.page, n: 1,
                                                       chunk_id: t.chunk_id, text: '',
                                                       match_reason: 'table on this page' })}>
                Open the page</Btn>}>
          <div style={{ overflowX: 'auto' }}>
            <table className="tbl">
              <thead>
                <tr>{t.headers.map((h, i) => <th key={i} className={align[i]}>{h}</th>)}</tr>
              </thead>
              <tbody>
                {t.rows.slice(0, 60).map((r, i) => (
                  <tr key={i}>
                    {t.headers.map((_h, j) => (
                      <td key={j} className={align[j]}>{r[j] ?? ''}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {t.rows.length > 60 && <Note>Showing the first 60 of {t.n_rows} rows.</Note>}

          <h3 className="sectionLabel">Ask about this table</h3>
          <div className="row">
            <input placeholder={t.numeric_cols?.length
                     ? `e.g. "total ${t.numeric_cols[0]}", "how many rows"`
                     : 'e.g. "how many rows", "rows containing Mumbai"'}
                   value={q} onChange={(e) => setQ(e.target.value)}
                   onKeyDown={(e) => e.key === 'Enter' && compute()}
                   style={{ flex: 1, minWidth: 240 }} />
            <Btn kind="primary" onClick={compute} disabled={!q.trim()}>Answer</Btn>
          </div>
          {error && <div style={{ marginTop: 10 }}><Err>{error}</Err></div>}
          {res && (
            <div style={{ marginTop: 12 }}>
              <div style={{ fontSize: 26, fontWeight: 600 }}>{res.table.formatted}</div>
              {showWorking && <Note>{res.table.workings}</Note>}
              {showWorking && (
                <div className="chips" style={{ marginTop: 8 }}>
                  <Pill tone="good">Computed in Python</Pill>
                  <Pill>{res.source.doc_name} · page {res.source.page}</Pill>
                  <Pill>{res.table.row_count} rows</Pill>
                </div>
              )}
              {showWorking && res.table.matched_rows?.length > 0 && (
                <table className="tbl" style={{ marginTop: 10 }}>
                  <thead><tr>{Object.keys(res.table.matched_rows[0]).map((h) => <th key={h}>{h}</th>)}</tr></thead>
                  <tbody>
                    {res.table.matched_rows.slice(0, 10).map((row, i) => (
                      <tr key={i}>{Object.values(row).map((v, j) => <td key={j}>{String(v)}</td>)}</tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          )}
        </Card>
        </div>
      )}
    </div>
  )
}
