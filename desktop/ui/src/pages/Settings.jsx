import React, { useEffect, useState } from 'react'
import { api } from '../api.js'
import { Btn, Card, Err, KV, Note, Pill, Switch } from '../ui.jsx'

export default function Settings({ sys, settings, setSettings, reload }) {
  const [diag, setDiag] = useState(null)
  const [wiping, setWiping] = useState(false)
  const [wipeError, setWipeError] = useState('')

  const wipe = async () => {
    const pass = window.prompt(
      'This deletes every project, document, index, image and entity on this device.\n' +
      'Enter your passcode to confirm (leave blank if none is set):')
    if (pass === null) return
    setWiping(true); setWipeError('')
    try {
      const r = await api.resetAll(pass)
      window.alert(`Erased ${r.documents_removed} documents and ${r.projects_removed} projects.`)
      reload()
    } catch (e) { setWipeError(String(e.message || e)) }
    setWiping(false)
  }
  useEffect(() => { api.diagnostics().then(setDiag).catch(() => {}) }, [])
  return (
    <div className="wrap grid two">
      <div>
        <Card title="Retrieval">
          <label className="field">Passages sent to the model ({settings.top_k})</label>
          <input type="range" min="1" max="10" value={settings.top_k}
                 onChange={(e) => setSettings({ top_k: Number(e.target.value) })} />
          <label className="field" style={{ marginTop: 12 }}>
            Refuse below similarity ({Number(settings.min_score).toFixed(2)})
          </label>
          <input type="range" min="0.2" max="0.8" step="0.01" value={settings.min_score}
                 onChange={(e) => setSettings({ min_score: Number(e.target.value) })} />
          <label className="field" style={{ marginTop: 12 }}>Answer length ({settings.max_tokens} tokens)</label>
          <input type="range" min="128" max="1024" step="64" value={settings.max_tokens}
                 onChange={(e) => setSettings({ max_tokens: Number(e.target.value) })} />
          <div className="row" style={{ marginTop: 14 }}>
            <Switch on={settings.strict_grounding} label="Strict grounding"
                    onChange={(v) => setSettings({ strict_grounding: v })} />
            <Switch on={settings.tuning} label="Auto-tuning" onChange={(v) => setSettings({ tuning: v })} />
            <Switch on={settings.show_evidence} label="Show sources and working"
                    onChange={(v) => setSettings({ show_evidence: v })} />
          </div>
          <Note style={{ marginTop: 10 }}>
            Strict grounding verifies every number, code and date against the retrieved text and refuses the
            answer otherwise.
          </Note>
        </Card>

        <Card title="Answer engine">
          <KV k="Mode" v={sys?.engine?.mode} />
          <KV k="Endpoint" v={sys?.engine?.base_url} />
          <KV k="Model" v={String(sys?.engine?.model || '—').split(/[\\/]/).pop()} />
          <KV k="Compute" v={sys?.engine?.compute} />
          <div className="row" style={{ marginTop: 10 }}>
            <Btn sm onClick={async () => { await api.restartEngine(); reload() }}>Restart engine</Btn>
          </div>
        </Card>
      </div>

      <div>
        <Card title="Privacy and storage">
          <KV k="Offline guard" v={diag?.offline_guard?.installed ? 'active' : 'off'} />
          <KV k="Blocked outbound attempts" v={diag?.offline_guard?.blocked_attempts ?? 0} />
          <KV k="Cloud API calls" v={0} />
          <KV k="Internet required" v="no" />
          <KV k="Documents" v={diag?.documents_processed ?? 0} />
          <KV k="Embedding cache" v={diag?.embedding_cache ?? 0} />
          <div className="row" style={{ marginTop: 10 }}>
            <Pill tone="good"><span className="dot" />all processing local</Pill>
          </div>
          <Note style={{ marginTop: 10 }}>
            Documents, index, images and the entity graph are stored in this app's local data folder. Deleting
            that folder erases everything.
          </Note>
        </Card>

        <Card title="Danger zone">
          <Note>
            Erase every project, document, index, image and entity stored by this app on this device. There is
            no undo and nothing is backed up anywhere.
          </Note>
          <div className="row" style={{ marginTop: 10 }}>
            <Btn onClick={wipe} disabled={wiping} style={{ borderColor: 'var(--bad)', color: 'var(--bad)' }}>
              {wiping ? 'Erasing…' : 'Erase all data'}
            </Btn>
          </div>
          {wipeError && <div style={{ marginTop: 10 }}><Err>{wipeError}</Err></div>}
        </Card>

        <Card title="Voice">
          <KV k="Speech to text" v={sys?.voice?.stt ? 'available' : 'unavailable'} />
          <KV k="Text to speech" v={sys?.voice?.tts ? 'available' : 'unavailable'} />
          {!sys?.voice?.stt && <Note>{sys?.voice?.stt_error}</Note>}
        </Card>
      </div>
    </div>
  )
}
