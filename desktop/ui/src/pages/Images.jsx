import React, { useEffect, useState } from 'react'
import { api } from '../api.js'
import { Card, Empty, Note, Pill } from '../ui.jsx'

export default function Images({ sys, setSource, docs }) {
  const [images, setImages] = useState([])
  const [caps, setCaps] = useState(null)
  useEffect(() => {
    api.images().then((r) => { setImages(r.images); setCaps(r.capabilities) }).catch(() => {})
  }, [sys?.images?.images])

  return (
    <div className="wrap">
      <Card title={`Indexed images (${images.length})`}>
        <div className="row" style={{ marginBottom: 12 }}>
          <Pill tone={caps?.mode === 'clip' ? 'good' : 'warn'}>
            {caps?.mode === 'clip' ? `Vision model ${caps.vision_model} · ${caps.provider}` : 'OCR text + perceptual hash only'}
          </Pill>
          <Pill>{sys?.images?.with_vectors ?? 0} with vectors</Pill>
          {caps?.download_hint && <Note>Enable semantic image search with: {caps.download_hint}</Note>}
        </div>
        {images.length === 0 ? <Empty>No images extracted yet. Import a PDF with photos or a scanned file.</Empty> : (
          <div className="thumbs">
            {images.map((im) => (
              <div className="thumb" key={im.id}
                   onClick={() => setSource({ doc_id: im.doc_id, doc_name: im.doc_name, page: im.page, n: 1,
                                              text: im.ocr_text, match_reason: 'image on this page' })}>
                <img src={api.imageFile(im.id)} alt={`${im.doc_name} page ${im.page}`} loading="lazy" />
                <div className="cap">
                  <div style={{ color: 'var(--ink)' }}>{im.doc_name || 'upload'}</div>
                  Page {im.page} · {im.width}×{im.height}
                  {im.ocr_text ? ' · has text' : ''}
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  )
}
