import React, { useEffect, useState } from 'react'
import { api } from '../api.js'
import { Btn, Empty, Note, Pill } from '../ui.jsx'

/** Split view: the page image on the left, why it matched on the right. */
export default function DocViewer({ source, docs, onClose, onAsk }) {
  const [page, setPage] = useState(source?.page || 1)
  const [fallback, setFallback] = useState(null)
  useEffect(() => { setPage(source?.page || 1); setFallback(null) }, [source])
  useEffect(() => { setFallback(null) }, [page])
  if (!source) return null

  const doc = docs.find((d) => d.id === source.doc_id)
  const total = doc?.pages || page
  const conf = source.vector_score != null
    ? Math.round(Math.max(0, Math.min(1, source.vector_score)) * 100)
    : (source.exact_match ? 100 : null)
  const related = docs.filter((d) => d.id !== source.doc_id).slice(0, 4)

  return (
    <div className="viewer" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="panel">
        <div className="vhead">
          <b>{source.doc_name}</b>
          <Pill>Page {page} of {total}</Pill>
          <Btn sm kind="ghost" disabled={page <= 1} onClick={() => setPage(page - 1)}>←</Btn>
          <Btn sm kind="ghost" disabled={page >= total} onClick={() => setPage(page + 1)}>→</Btn>
          <div className="spacer" />
          <Btn sm kind="ghost" onClick={onClose}>Close</Btn>
        </div>
        <div className="vbody">
          <div className="pane">
            {fallback === null ? (
              <img className="page-img" alt={`${source.doc_name} page ${page}`}
                   src={api.pageImage(source.doc_id, page, page === source.page ? source.chunk_id : null, 160)}
                   onError={() => api.pageText(source.doc_id, page)
                     .then((r) => setFallback(r.text || 'no text on this page'))
                     .catch(() => setFallback('this page cannot be displayed'))} />
            ) : (
              <>
                <Note>This file has no page image (spreadsheet or note), so here is its text:</Note>
                <div className="snippet" style={{ maxHeight: 'none', marginTop: 8, whiteSpace: 'pre-wrap' }}>
                  {fallback}
                </div>
              </>
            )}
            {page !== source.page && (
              <Note style={{ marginTop: 8 }}>Highlighting is shown on Page {source.page}, where the answer came from.</Note>
            )}
          </div>
          <div className="pane right">
            <h3>AI evidence</h3>
            <div style={{ marginTop: 10 }}>
              <Note>Why this page?</Note>
              <div className="chips" style={{ marginTop: 6 }}>
                {source.exact_match && <Pill tone="accent">exact code / value match</Pill>}
                {source.keyword_rank != null && <Pill>Keyword rank {source.keyword_rank + 1}</Pill>}
                {source.vector_score != null && <Pill>meaning match {source.vector_score}</Pill>}
                {source.graph_reason && <Pill tone="accent">{source.graph_reason}</Pill>}
                {source.match_reason && <Pill>{source.match_reason}</Pill>}
                {source.content_type && <Pill>{String(source.content_type).replace('_', ' ')}</Pill>}
              </div>
            </div>
            {conf != null && (
              <div style={{ marginTop: 14 }}>
                <Note>Confidence</Note>
                <div style={{ fontSize: 24, fontWeight: 600 }}>{conf}%</div>
              </div>
            )}
            <div style={{ marginTop: 14 }}>
              <Note>Matched text</Note>
              <div className="snippet">{source.text || source.snippet || '—'}</div>
            </div>
            {related.length > 0 && (
              <div style={{ marginTop: 14 }}>
                <Note>Other documents in this workspace</Note>
                <div className="chips" style={{ marginTop: 6 }}>
                  {related.map((d) => <Pill key={d.id}>{d.name}</Pill>)}
                </div>
              </div>
            )}
            {onAsk && (
              <Btn sm style={{ marginTop: 16 }} onClick={() => onAsk(source)}>Ask about this page</Btn>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
