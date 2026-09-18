import React, { useEffect, useState } from 'react'
import { api } from '../api.js'
import { Btn, Card, Empty, Err, KV, Note, Pill, Stat, fmtMs, fmtNum } from '../ui.jsx'

const tone = (b) => (b?.startsWith('NPU') ? 'accent' : b?.startsWith('GPU') ? 'accent' : '')

export default function EdgeAI({ sys }) {
  const [diag, setDiag] = useState(null)
  const [runs, setRuns] = useState([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const load = async () => {
    try {
      const [d, h] = await Promise.all([api.diagnostics(), api.benchmarkHistory()])
      setDiag(d); setRuns(h.runs || [])
    } catch (e) { setError(String(e.message || e)) }
  }
  useEffect(() => { load() }, [])

  const run = async () => {
    setBusy(true); setError('')
    try { await api.benchmark(3, true); await load() } catch (e) { setError(String(e.message || e)) }
    setBusy(false)
  }

  const hw = diag?.hardware
  const backends = diag?.backends || {}
  const latest = {}
  for (const r of runs) if (!latest[r.task]) latest[r.task] = r
  const models = diag?.models || []
  const modelRows = [
    { name: models.find((m) => m.role === 'text_generation')?.name || 'answer model',
      backend: diag?.llm?.compute || '—',
      quant: models.find((m) => m.role === 'text_generation')?.quantization || '—' },
    { name: diag?.embedding_model?.id || 'embeddings',
      backend: backends.actual?.text_embedding?.backend || 'CPU',
      quant: diag?.embedding_model?.quantization || '—' },
    { name: diag?.vision?.vision_model || 'vision model (not installed)',
      backend: backends.actual?.image_embedding?.backend || (diag?.vision?.mode === 'clip' ? 'CPU' : '—'),
      quant: diag?.vision?.mode === 'clip' ? 'int8' : '—' },
    { name: `OCR — ${diag?.ocr?.backend || 'unavailable'}`, backend: 'CPU (OS engine)', quant: '—' },
  ]

  return (
    <div className="wrap">
      <Card title="Device">
        {!hw ? <Empty>Detecting…</Empty> : (
          <>
            <div className="row" style={{ marginBottom: 12 }}>
              <Pill tone={hw.is_snapdragon ? 'accent' : ''}>
                {hw.is_snapdragon ? 'Qualcomm Snapdragon' : 'x86 development machine'}
              </Pill>
              <Pill tone={hw.npu_available ? 'good' : 'warn'}>
                Hexagon NPU {hw.npu_available ? 'available' : 'not detected'}
              </Pill>
              <Pill tone={hw.gpu_available ? 'good' : ''}>GPU {hw.gpu_available ? 'available' : 'not detected'}</Pill>
              <Pill tone="good">CPU available</Pill>
              <div className="spacer" />
              <Pill tone="good"><span className="dot" />All processing local · internet off</Pill>
            </div>
            <KV k="Processor" v={hw.cpu} />
            <KV k="OS / architecture" v={`${hw.os} · ${hw.machine}`} />
            <KV k="Cores / memory" v={`${hw.cores} cores · ${(hw.ram_mb / 1024).toFixed(1)} GB`} />
            <KV k="ONNX Runtime" v={`${hw.ort_version} · ${hw.available_providers.join(', ')}`} />
            <KV k="Backend mode" v={backends.mode} />
            <div style={{ marginTop: 10 }}>
              <Note>{hw.summary}</Note>
              {(hw.notes || []).map((n, i) => <Note key={i}>{n}</Note>)}
            </div>
          </>
        )}
      </Card>

      <Card title="Inference">
        <table className="tbl">
          <thead><tr><th>Model</th><th>Backend</th><th>Quantization</th></tr></thead>
          <tbody>
            {modelRows.map((m, i) => (
              <tr key={i}>
                <td>{m.name}</td>
                <td><Pill tone={tone(m.backend)}>{m.backend}</Pill></td>
                <td className="mono">{m.quant}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <Note style={{ marginTop: 8 }}>
          Backends come from the live runtime — an operation is never labelled NPU unless the provider reports it.
        </Note>
      </Card>

      <Card title="Performance" actions={<Btn sm kind="primary" onClick={run} disabled={busy}>
        {busy ? 'Measuring…' : 'Run benchmark'}</Btn>}>
        {error && <Err>{error}</Err>}
        {Object.keys(latest).length === 0 ? (
          <Empty>No measurements yet. Run the benchmark — every number shown is measured on this machine.</Empty>
        ) : (
          <div className="grid four">
            <Stat label="Embedding latency" value={fmtMs(latest.embed_query?.latency_ms)}
                  hint={latest.embed_query?.backend} />
            <Stat label="Retrieval latency" value={fmtMs(latest.retrieval?.latency_ms)} hint="Hybrid search" />
            <Stat label="Generation"
                  value={latest.llm_generate?.throughput ? `${fmtNum(latest.llm_generate.throughput)} tok/s` : '—'}
                  hint={latest.llm_generate?.backend} />
            <Stat label="Process memory" value={latest.embed_text?.memory_mb ? `${fmtNum(latest.embed_text.memory_mb, 0)} MB` : '—'}
                  hint="Resident memory" />
          </div>
        )}
      </Card>
    </div>
  )
}
