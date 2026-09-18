import React, { useEffect, useState } from 'react'
import { api } from '../api.js'
import { Card, Empty, Note, Pill } from '../ui.jsx'

export default function Models() {
  const [models, setModels] = useState([])
  const [vision, setVision] = useState(null)
  useEffect(() => {
    api.diagnostics().then((d) => { setModels(d.models || []); setVision(d.vision) }).catch(() => {})
  }, [])
  return (
    <div className="wrap">
      <Card title="Model registry">
        <Note>
          One place defines every model: role, format, quantization, memory, supported backends and Snapdragon
          compatibility. A Qualcomm AI Hub optimised model can replace a generic one by editing a single entry.
        </Note>
        {models.length === 0 ? <Empty>Loading…</Empty> : (
          <table className="tbl" style={{ marginTop: 12 }}>
            <thead>
              <tr><th>Model</th><th>Role</th><th>Format</th><th>Quantization</th><th>Backends</th>
                <th>Snapdragon</th><th>Status</th></tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.key}>
                  <td>
                    {m.name}
                    <div className="meta"><Note>{m.notes}</Note></div>
                    {m.source && <div className="meta mono">{m.source}</div>}
                  </td>
                  <td>{m.role.replace('_', ' ')}</td>
                  <td>{m.format}</td>
                  <td className="mono">{m.quantization}</td>
                  <td>{(m.backends || []).join(', ')}</td>
                  <td><Pill tone={m.snapdragon === 'npu' ? 'accent' : ''}>{String(m.snapdragon).toUpperCase()}</Pill></td>
                  <td>
                    {m.installed
                      ? <Pill tone="good">Installed{m.size_mb ? ` · ${m.size_mb} MB` : ''}</Pill>
                      : <Pill tone="warn">Not installed</Pill>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
      {vision && vision.mode !== 'clip' && (
        <Card title="Vision model">
          <Note>{vision.note}</Note>
          {vision.download_hint && <Note style={{ marginTop: 6 }}>Install with: {vision.download_hint}</Note>}
        </Card>
      )}
    </div>
  )
}
