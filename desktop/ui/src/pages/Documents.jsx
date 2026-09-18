import React, { useRef, useState } from 'react'
import { api } from '../api.js'
import { Btn, Card, Empty, Err, KV, Note, Pill } from '../ui.jsx'

const TONE = { ready: 'good', error: 'bad', indexing: 'warn', queued: 'warn' }

export default function Documents({ sys, docs, pack, reload, setSource, addRef }) {
  const [over, setOver] = useState(false)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const input = useRef(null)
  if (addRef) addRef.current = () => input.current?.click()

  const upload = async (files) => {
    if (!files?.length) return
    setBusy(true); setError('')
    try {
      const r = await api.upload(files)
      const failed = r.results.filter((x) => x.error)
      if (failed.length) setError(failed.map((f) => `${f.filename}: ${f.error}`).join(' · '))
    } catch (e) { setError(String(e.message || e)) }
    setBusy(false); reload()
  }

  const ct = sys?.content_types || {}
  return (
    <div className="wrap grid split">
      <div>
        <Card>
          <div className={`drop ${over ? 'over' : ''}`} onClick={() => input.current?.click()}
               onDragOver={(e) => { e.preventDefault(); setOver(true) }}
               onDragLeave={() => setOver(false)}
               onDrop={(e) => { e.preventDefault(); setOver(false); upload([...e.dataTransfer.files]) }}>
            <div style={{ fontSize: 14, color: 'var(--ink)' }}>{busy ? 'Indexing…' : 'Drop files here'}</div>
            <Note>PDF · DOCX · XLSX · CSV · TXT · MD · PNG · JPG · scanned PDFs</Note>
          </div>
          <input ref={input} type="file" multiple style={{ display: 'none' }}
                 onChange={(e) => { upload([...e.target.files]); e.target.value = '' }} />
          {error && <div style={{ marginTop: 10 }}><Err>{error}</Err></div>}
          {sys?.indexing?.current && (
            <Note style={{ marginTop: 8 }}>
              Indexing {sys.indexing.current.name} — {sys.indexing.current.stage}
            </Note>
          )}
        </Card>

        <Card title={`All files (${docs.length})`}>
          {docs.length === 0 && <Empty>No documents yet.</Empty>}
          {docs.map((d) => (
            <div className="item" key={d.id}>
              <div className="row">
                <span className="name">{d.name}</span>
                <Pill tone={TONE[d.status] || ''}>
                  {d.status === 'indexing' && sys?.indexing?.current?.doc_id === d.id
                    ? sys.indexing.current.stage
                    : d.status}
                </Pill>
                <div className="spacer" />
                <Btn sm kind="ghost"
                     onClick={() => setSource({ doc_id: d.id, doc_name: d.name, page: 1, n: 1, text: '' })}>
                  open
                </Btn>
                <Btn sm kind="ghost" onClick={async () => { await api.reindex(d.id); reload() }}>Re-index</Btn>
                <Btn sm kind="ghost" onClick={async () => { await api.deleteDoc(d.id); reload() }}>Remove</Btn>
              </div>
              <div className="meta">
                <span>{d.pages} pages</span><span>{d.chunk_count} chunks</span>
                {d.table_count > 0 && <span>{d.table_count} tables</span>}
                {d.image_count > 0 && <span>{d.image_count} images</span>}
                {d.ocr_pages > 0 && <span>{d.ocr_pages} OCR</span>}
                {d.index_seconds != null && <span>{d.index_seconds}s</span>}
              </div>
              {d.error && <Err>{d.error}</Err>}
            </div>
          ))}
        </Card>
      </div>

      <div>
        <Card title="Page content types">
          {Object.keys(ct).length === 0 ? <Empty>No pages yet.</Empty> :
            Object.entries(ct).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
              <KV key={k} k={k.replace('_', ' ')} v={v} />))}
          <Note style={{ marginTop: 8 }}>
            Pages are classified on import, so a scan or a table is never treated as plain text.
          </Note>
        </Card>
        <Card title="Auto-tuned for this workspace"
              actions={<Btn sm onClick={async () => { await api.retune(); reload() }}>Re-tune</Btn>}>
          {!pack ? <Empty>Import documents to auto-tune.</Empty> : (
            <>
              <KV k="Detected" v={pack.doc_type} />
              <KV k="Glossary" v={`${Object.keys(pack.glossary || {}).length} terms`} />
              <KV k="Answer style" v={pack.answer_format} />
              <KV k="Keyword weight" v={`${pack.keyword_weight}×`} />
              {Object.entries(pack.entity_counts || {}).slice(0, 4).map(([k, v]) => (
                <KV key={k} k={`${k.replace('_', ' ')}s`} v={v} />))}
              {pack.sample_codes?.length > 0 && (
                <div className="chips" style={{ marginTop: 10 }}>
                  {pack.sample_codes.map((c) => <Pill key={c}>{c}</Pill>)}
                </div>
              )}
            </>
          )}
        </Card>
      </div>
    </div>
  )
}
