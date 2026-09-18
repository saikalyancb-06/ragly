import React, { useRef, useState } from 'react'
import { api } from '../api.js'
import { Btn, Card, Empty, Err, Note, Pill, Seg } from '../ui.jsx'

const pct = (r) => {
  if (r.similarity != null) {
    const scaled = Math.min(99, Math.max(35, Math.round(((r.similarity - 0.16) / 0.18) * 40 + 55)))
    return `${scaled}%`
  }
  if (r.phash_distance != null) return `${Math.max(0, 100 - r.phash_distance * 8)}%`
  if (r.score != null) return `${Math.round(Math.min(95, r.score * 100))}%`
  return '—'
}

export default function PhotoSearch({ sys, setSource, settings }) {
  const [q, setQ] = useState('')
  const [mode, setMode] = useState('images')
  const [results, setResults] = useState(null)
  const [docResults, setDocResults] = useState(null)
  const [preview, setPreview] = useState(null)
  const [busy, setBusy] = useState(false)
  const [over, setOver] = useState(false)
  const [error, setError] = useState('')
  const [caps, setCaps] = useState(sys?.vision || null)
  const [debug, setDebug] = useState(null)
  const file = useRef(null)
  const lastUpload = useRef(null)   // so switching mode re-runs the same picture

  const byText = async () => {
    if (!q.trim()) return
    setBusy(true); setError(''); setDocResults(null); setPreview(null); lastUpload.current = null
    try {
      const r = await api.imageSearch(q)
      setResults(r.results); setCaps(r.capabilities); setDebug(r.debug || null)
    } catch (e) { setError(String(e.message || e)) }
    setBusy(false)
  }

  const byImage = async (f, which = mode) => {
    if (!f) return
    lastUpload.current = f
    setBusy(true); setError(''); setPreview(URL.createObjectURL(f))
    // only one kind of result can be on screen at a time, so the page always matches the switch
    setResults(null); setDocResults(null)
    try {
      const r = await api.imageSearchByImage(f, which)
      setCaps(r.capabilities); setDebug(r.debug || null)
      if (which === 'documents') setDocResults(r.results)
      else setResults(r.results)
    } catch (e) { setError(String(e.message || e)) }
    setBusy(false)
  }

  const switchMode = (next) => {
    setMode(next)
    setResults(null); setDocResults(null); setDebug(null)
    if (lastUpload.current) byImage(lastUpload.current, next)   // same picture, other question
  }

  const clip = caps?.mode === 'clip'
  const missingVision = caps && !clip

  return (
    <div className="wrap">
      <Card>
        <div className="searchbig">
          <span style={{ color: 'var(--muted)' }}>⌕</span>
          <input placeholder='"damaged laptop near a desk"' value={q}
                 onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && byText()} />
          <Btn kind="primary" onClick={byText} disabled={busy}>Search</Btn>
        </div>
        <div className="row" style={{ marginTop: 12, justifyContent: 'center' }}><Note>or</Note></div>
        <div className={`drop ${over ? 'over' : ''}`} style={{ marginTop: 8 }}
             onClick={() => file.current?.click()}
             onDragOver={(e) => { e.preventDefault(); setOver(true) }}
             onDragLeave={() => setOver(false)}
             onDrop={(e) => { e.preventDefault(); setOver(false); byImage(e.dataTransfer.files?.[0]) }}>
          {preview
            ? <img src={preview} alt="reference" style={{ height: 110, borderRadius: 8 }} />
            : <div style={{ fontSize: 14, color: 'var(--ink)' }}>Drop a Reference image ↑</div>}
          <Note>image → image, or image → the documents it belongs to</Note>
        </div>
        <input ref={file} type="file" accept="image/*" style={{ display: 'none' }}
               onChange={(e) => { byImage(e.target.files?.[0]); e.target.value = '' }} />
        <div className="row" style={{ marginTop: 12 }}>
          <Seg value={mode} onChange={switchMode} options={[
            { value: 'images', label: 'Similar images' }, { value: 'documents', label: 'Which document' }]} />
          <Pill tone={clip ? 'good' : 'warn'}>
            {clip ? `Local vision model · ${caps.vision_model}` : 'No vision model installed'}
          </Pill>
          <Pill>{sys?.images?.images ?? 0} images indexed</Pill>
          <div className="spacer" />
          <Pill tone="good"><span className="dot" />0 cloud requests</Pill>
        </div>
        {missingVision && (
          <div style={{ marginTop: 10 }}>
            <Err>
              The vision model is not installed, so image search is falling back to near-duplicate
              matching only. Close the app and run scripts\doctor.ps1 once — it downloads the model
              (about 150 MB), then re-index your documents so their images get visual embeddings.
            </Err>
          </div>
        )}
        {error && <div style={{ marginTop: 10 }}><Err>{error}</Err></div>}
      </Card>

      {docResults && (
        <Card title={`Documents matched by this image (${docResults.length})`}>
          {docResults.length === 0 && <Empty>No document matched.</Empty>}
          {docResults.map((d) => (
            <div className="item click" key={d.doc_id}
                 onClick={() => setSource({ doc_id: d.doc_id, doc_name: d.doc_name, page: d.pages?.[0] || 1, n: 1,
                                            text: '', match_reason: (d.reasons || []).join(' · ') })}>
              <div className="row">
                <span className="name">{d.doc_name}</span>
                <Pill>score {d.score.toFixed(3)}</Pill>
                {d.pages?.length > 0 && <Pill>pages {d.pages.join(', ')}</Pill>}
              </div>
              <div className="meta">{(d.reasons || []).join(' · ')}</div>
            </div>
          ))}
        </Card>
      )}

      {debug && settings?.show_evidence && (
        <Card title="Image search debug">
          <div className="chips">
            <Pill>query type {debug.query_type}</Pill>
            <Pill>{debug.embedding_model || 'no vision model'}</Pill>
            <Pill>{debug.embedding_provider}</Pill>
            <Pill>dimension {debug.embedding_dimension ?? '—'}</Pill>
            <Pill>query norm {debug.query_embedding_norm ?? '—'}</Pill>
            <Pill>{debug.indexed_images} indexed · {debug.images_with_vectors} with vectors</Pill>
            <Pill>{debug.candidate_count} candidates · top {debug.top_k}</Pill>
          </div>
          {debug.similarity_scores?.length > 0 && (
            <table className="tbl" style={{ marginTop: 10 }}>
              <thead><tr><th>image</th><th>document</th><th className="num">page</th>
                <th className="num">similarity</th><th>why</th></tr></thead>
              <tbody>
                {debug.similarity_scores.map((row, i) => (
                  <tr key={i}>
                    <td className="mono">{row.image_id}</td><td>{row.document}</td>
                    <td className="num">{row.page}</td>
                    <td className="num mono">{row.similarity ?? row.phash_distance ?? '—'}</td>
                    <td>{row.why}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Card>
      )}

      {results && (
        <Card title={`Results (${results.length})`}>
          {results.length === 0 && <Empty>Nothing matched. Try different words, or index more images.</Empty>}
          <div className="thumbs">
            {results.map((r) => (
              <div className="thumb" key={r.image_id} title={r.match_reason}
                   onClick={() => setSource({ doc_id: r.doc_id, doc_name: r.doc_name, page: r.page, n: 1,
                                              text: '', match_reason: r.match_reason })}>
                <img src={api.imageFile(r.image_id)} alt={`${r.doc_name} page ${r.page}`} loading="lazy" />
                <div className="cap">
                  <div className="score">{pct(r)}</div>
                  <div style={{ color: 'var(--ink)' }}>{r.doc_name}</div>
                  page {r.page} · {r.match_reason}
                  {r.objects?.length > 0 && (
                    <div className="chips" style={{ marginTop: 6 }}>
                      {r.objects.slice(0, 4).map((o) => (
                        <span className="pill" key={o}
                              style={(r.matched_concepts || []).includes(o)
                                ? { borderColor: 'var(--accent)', color: 'var(--ink)' } : undefined}>
                          {(r.matched_concepts || []).includes(o) ? '✓ ' : ''}{o}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {!results && !docResults && (
        <Card><Empty>Search by words, or drop a photo to find visually similar images and their documents.</Empty></Card>
      )}
    </div>
  )
}
