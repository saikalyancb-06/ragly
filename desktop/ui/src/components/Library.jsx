import React, { useRef, useState } from 'react'

function statusChip(d) {
  const map = { ready: 'ok', error: 'bad', indexing: 'warn', queued: 'warn' }
  return <span className={`chip ${map[d.status] || ''}`}>{d.status}</span>
}

export default function Library({ docs, pack, onUpload, onDelete, onRetune, indexing, error }) {
  const [over, setOver] = useState(false)
  const input = useRef(null)

  return (
    <div className="col left">
      <h3>Documents</h3>
      <div
        className={`drop ${over ? 'over' : ''}`}
        onClick={() => input.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setOver(true) }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); onUpload([...e.dataTransfer.files]) }}
      >
        Drop PDFs, scans, images or notes here<br />
        <small>or click to choose files</small>
      </div>
      <input
        ref={input}
        type="file"
        multiple
        style={{ display: 'none' }}
        onChange={(e) => { onUpload([...e.target.files]); e.target.value = '' }}
      />
      {error && <div className="err">{error}</div>}
      {indexing?.current && (
        <div className="row">indexing {indexing.current.name}: {indexing.current.stage}</div>
      )}

      <div style={{ marginTop: 10 }}>
        {docs.length === 0 && <div className="empty">No documents yet</div>}
        {docs.map((d) => (
          <div className="doc" key={d.id}>
            <button className="x" title="remove" onClick={() => onDelete(d.id)}>×</button>
            <div className="name">{d.name}</div>
            <div className="meta">
              {statusChip(d)}
              <span>{d.pages} pages</span>
              <span>{d.chunk_count} chunks</span>
              {d.ocr_pages > 0 && <span>{d.ocr_pages} OCR</span>}
              {d.index_seconds != null && <span>{d.index_seconds}s</span>}
            </div>
            {d.error && <div className="err">{d.error}</div>}
          </div>
        ))}
      </div>

      <h3>What Ragly learned</h3>
      {!pack ? (
        <div className="empty">Import documents to auto-tune</div>
      ) : (
        <div className="pack">
          <div className="kv"><span>Detected</span><b>{pack.doc_type}</b></div>
          <div className="kv"><span>Glossary</span><span>{Object.keys(pack.glossary || {}).length} terms</span></div>
          <div className="kv"><span>Answer style</span><span>{pack.answer_format}</span></div>
          <div className="kv"><span>Keyword weight</span><span>{pack.keyword_weight}×</span></div>
          {Object.entries(pack.entity_counts || {}).map(([k, v]) => (
            <div className="kv" key={k}><span>{k.replace('_', ' ')}s found</span><span>{v}</span></div>
          ))}
          {pack.sample_codes?.length > 0 && (
            <div className="terms">{pack.sample_codes.map((c) => <span className="term" key={c}>{c}</span>)}</div>
          )}
          {Object.entries(pack.glossary || {}).filter(([, v]) => v).slice(0, 8).map(([k, v]) => (
            <div className="kv" key={k}><span>{k}</span><span>{v}</span></div>
          ))}
          <button style={{ marginTop: 8, width: '100%' }} onClick={onRetune}>Re-tune from documents</button>
        </div>
      )}
    </div>
  )
}
