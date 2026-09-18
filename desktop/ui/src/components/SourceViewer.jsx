import React from 'react'
import { api } from '../api.js'

export default function SourceViewer({ source, onClose, settings, onSettings }) {
  return (
    <div className="col right viewer">
      <div className="head">
        <h3 style={{ margin: 0 }}>Source</h3>
        {source && <button onClick={onClose}>Close</button>}
      </div>

      {!source ? (
        <div className="empty">Click a citation like [1] to see the page it came from, highlighted.</div>
      ) : (
        <>
          <div className="row" style={{ marginTop: 0 }}>
            <b style={{ color: 'var(--text)' }}>{source.doc_name}</b> · Page {source.page}
            {source.heading ? ` · ${source.heading}` : ''}
          </div>
          <img
            alt={`${source.doc_name} page ${source.page}`}
            src={api.pageImage(source.doc_id, source.page, source.chunk_id)}
            onError={(e) => { e.target.style.display = 'none' }}
          />
          <div className="snippet">{source.text || source.snippet}</div>
          <div className="row">
            match {source.vector_score}
            {source.exact_match ? ' · exact code match' : ''}
            {source.keyword_rank != null ? ` · keyword rank ${source.keyword_rank + 1}` : ''}
          </div>
        </>
      )}

      <h3>Retrieval settings</h3>
      <div className="row">
        <span>passages sent ({settings.top_k})</span>
        <input
          type="range" min="1" max="10" value={settings.top_k}
          onChange={(e) => onSettings({ top_k: Number(e.target.value) })}
        />
      </div>
      <div className="row">
        <span>refuse below ({settings.min_score.toFixed(2)})</span>
        <input
          type="range" min="0.2" max="0.8" step="0.01" value={settings.min_score}
          onChange={(e) => onSettings({ min_score: Number(e.target.value) })}
        />
      </div>
      <div className="row">
        <span>Answer length ({settings.max_tokens})</span>
        <input
          type="range" min="128" max="1024" step="64" value={settings.max_tokens}
          onChange={(e) => onSettings({ max_tokens: Number(e.target.value) })}
        />
      </div>
      <div className="row" style={{ color: 'var(--muted)' }}>
        Fewer passages = faster first word. Higher refuse threshold = stricter “not found”.
      </div>
    </div>
  )
}
