import React, { useEffect, useRef, useState } from 'react'
import { api } from '../api.js'
import { Btn, Err, Note, Pill } from '../ui.jsx'

export default function AuthScreen({ auth, onDone, onBack }) {
  const mode = auth?.enabled ? 'login' : 'register'
  const [passcode, setPasscode] = useState('')
  const [confirm, setConfirm] = useState('')
  const [name, setName] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [shake, setShake] = useState(false)
  const input = useRef(null)
  useEffect(() => { input.current?.focus() }, [])

  const fail = (msg) => {
    setError(msg); setShake(true); setTimeout(() => setShake(false), 400)
  }

  const submit = async () => {
    if (busy) return
    if (mode === 'register' && passcode !== confirm) return fail('the two passcodes do not match')
    setBusy(true); setError('')
    try {
      const r = mode === 'register'
        ? await api.register(passcode, name)
        : await api.login(passcode)
      onDone(r.token, r)
    } catch (e) { fail(String(e.message || e)) }
    setBusy(false)
  }

  return (
    <div className="auth-screen">
      <div className="aurora"><span /><span /><span /></div>
      <div className="grid-lines" />
      <div className={`auth-card ${shake ? 'shake' : ''}`}>
        <div className="brand" style={{ padding: 0, marginBottom: 14 }}>
          <div className="mark">EV</div>
          <div className="name">EdgeVault</div>
        </div>
        <h2>{mode === 'register' ? 'Set a passcode for this device' : 'Welcome back'}</h2>
        <Note style={{ marginTop: 6 }}>
          {mode === 'register'
            ? 'A passcode is required to open the workspace. It is stored as a hash in your local data folder — there is no account server, and nothing is sent anywhere.'
            : `Unlock the workspace${auth?.display_name ? `, ${auth.display_name}` : ''}.`}
        </Note>

        {mode === 'register' && (
          <div className="field-row">
            <label className="field">Your name (optional)</label>
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Krishna" />
          </div>
        )}
        <div className="field-row">
          <label className="field">Passcode</label>
          <input ref={input} type="password" value={passcode} autoComplete="current-password"
                 onChange={(e) => setPasscode(e.target.value)}
                 onKeyDown={(e) => e.key === 'Enter' && submit()} placeholder="At least 4 characters" />
        </div>
        {mode === 'register' && (
          <div className="field-row">
            <label className="field">Confirm passcode</label>
            <input type="password" value={confirm} onChange={(e) => setConfirm(e.target.value)}
                   onKeyDown={(e) => e.key === 'Enter' && submit()} />
          </div>
        )}
        {error && <div style={{ marginTop: 12 }}><Err>{error}</Err></div>}

        <div className="row" style={{ marginTop: 18 }}>
          <Btn kind="primary" onClick={submit} disabled={busy || passcode.length < 4}>
            {busy ? 'Checking…' : mode === 'register' ? 'Create workspace' : 'Unlock'}
          </Btn>
          <div className="spacer" style={{ flex: 1 }} />
          <Btn kind="ghost" onClick={onBack}>Back</Btn>
        </div>
        <div className="row" style={{ marginTop: 16 }}>
          <Pill tone="good"><span className="dot" />Offline · nothing leaves this device</Pill>
        </div>
        <Note style={{ marginTop: 10 }}>
          The passcode locks the interface. It does not encrypt the database file — deleting the data folder
          is still what erases everything.
        </Note>
      </div>
    </div>
  )
}
