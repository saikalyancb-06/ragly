import React, { useState } from 'react'
import { api } from '../api.js'
import { Btn, Card, Empty, Err, Note, Pill } from '../ui.jsx'

/** "money" -> "amount", so the table reads like a person wrote it. */
const WHAT = { money: 'amount', date: 'date', duration: 'period', measurement: 'measurement',
               code: 'code', clause_ref: 'reference', part_number: 'part number', number: 'number' }

const sentence = (s) => {
  const bits = []
  if (s.value_changes) bits.push(`${s.value_changes} value${s.value_changes === 1 ? '' : 's'} changed`)
  if (s.added) bits.push(`${s.added} section${s.added === 1 ? '' : 's'} added`)
  if (s.removed) bits.push(`${s.removed} section${s.removed === 1 ? '' : 's'} removed`)
  const worded = (s.modified || 0) - 0
  if (!bits.length && worded) bits.push('only wording changed')
  if (!bits.length) return 'These two documents say the same thing.'
  return bits.join(', ').replace(/^./, (c) => c.toUpperCase()) + '.'
}

export default function Compare({ docs, setSource }) {
  const [left, setLeft] = useState('')
  const [right, setRight] = useState('')
  const [res, setRes] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [details, setDetails] = useState(false)

  const run = async () => {
    if (!left || !right || left === right) { setError('Pick two different documents.'); return }
    setBusy(true); setError(''); setDetails(false)
    try { setRes(await api.compare(Number(left), Number(right))) }
    catch (e) { setError(String(e.message || e)) }
    setBusy(false)
  }

  const open = (side, page) => setSource({ doc_id: res[side].doc_id, doc_name: res[side].name,
                                           page: page || 1, n: 1, text: '' })

  const all = res?.changes || []
  const values = all.flatMap((c) => (c.value_changes || []).map((v) => ({ ...v, section: c.title, change: c })))
  const added = all.filter((c) => c.kind === 'added')
  const removed = all.filter((c) => c.kind === 'removed')
  // Sections whose wording moved but where no figure, date or amount changed.
  const worded = all.filter((c) => c.kind === 'modified' && !(c.value_changes || []).length)

  return (
    <div className="wrap">
      <Card>
        <div className="row">
          <select value={left} onChange={(e) => setLeft(e.target.value)} style={{ flex: 1 }}>
            <option value="">Original…</option>
            {docs.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
          </select>
          <span style={{ color: 'var(--muted)' }}>→</span>
          <select value={right} onChange={(e) => setRight(e.target.value)} style={{ flex: 1 }}>
            <option value="">Revised…</option>
            {docs.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
          </select>
          <Btn kind="primary" onClick={run} disabled={busy}>{busy ? 'Comparing…' : 'What changed?'}</Btn>
        </div>
        {error && <div style={{ marginTop: 10 }}><Err>{error}</Err></div>}
      </Card>

      {res && (
        <Card title={sentence(res.summary)}
              actions={<Btn sm kind="ghost" onClick={() => setDetails(!details)}>
                {details ? 'Hide detail' : 'Show detail'}</Btn>}>

          {values.length > 0 && (
            <table className="tbl">
              <thead>
                <tr><th>What changed</th><th>Was</th><th>Now</th><th>Where</th></tr>
              </thead>
              <tbody>
                {values.map((v, i) => (
                  <tr key={i}>
                    <td>{v.section} · {WHAT[v.kind] || v.kind}</td>
                    <td className="mono">{v.from ?? '—'}</td>
                    <td className="mono" style={{ color: 'var(--warn)' }}>{v.to ?? '—'}</td>
                    <td>
                      {v.change.right_page
                        ? <Pill onClick={() => open('right', v.change.right_page)}>page {v.change.right_page}</Pill>
                        : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {added.map((c, i) => (
            <div className="item" key={`a${i}`}>
              <div className="row">
                <Pill tone="good">New</Pill>
                <span className="name">{c.title}</span>
                <div className="spacer" />
                {c.right_page && <Pill onClick={() => open('right', c.right_page)}>page {c.right_page}</Pill>}
              </div>
              <div className="snippet">{(c.text_diff?.[0] || '').replace(/^added:\s*/, '').replace(/[“”]/g, '')}</div>
            </div>
          ))}

          {removed.map((c, i) => (
            <div className="item" key={`r${i}`}>
              <div className="row">
                <Pill tone="bad">Dropped</Pill>
                <span className="name">{c.title}</span>
                <div className="spacer" />
                {c.left_page && <Pill onClick={() => open('left', c.left_page)}>page {c.left_page}</Pill>}
              </div>
              <div className="snippet">{(c.text_diff?.[0] || '').replace(/^removed:\s*/, '').replace(/[“”]/g, '')}</div>
            </div>
          ))}

          {worded.length > 0 && (
            <Note style={{ marginTop: 12 }}>
              Wording changed with no figure altered in: {worded.map((c) => c.title).join(', ')}.
            </Note>
          )}

          {values.length === 0 && added.length === 0 && removed.length === 0 && worded.length === 0 && (
            <Empty>Nothing differs between these two documents.</Empty>
          )}

          {details && (
            <div style={{ marginTop: 16 }}>
              <h3>Line by line</h3>
              {all.filter((c) => c.kind !== 'unchanged').map((c, i) => (
                <div className="item" key={`d${i}`}>
                  <div className="row">
                    <Pill>{c.kind}</Pill>
                    <span className="name">{c.title}</span>
                  </div>
                  {c.text_diff?.map((d, j) => <div className="diffline" key={j}>{d}</div>)}
                </div>
              ))}
              <Note style={{ marginTop: 8 }}>
                Sections are matched by heading, then by content, and values are compared exactly. No language
                model takes part in this comparison, so nothing here can be invented.
              </Note>
            </div>
          )}
        </Card>
      )}
    </div>
  )
}
