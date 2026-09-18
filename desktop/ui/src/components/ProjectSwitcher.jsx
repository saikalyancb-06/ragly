import React, { useEffect, useRef, useState } from 'react'
import { Btn, Note } from '../ui.jsx'

export default function ProjectSwitcher({ projects, active, onSwitch, onCreate, onDelete }) {
  const [open, setOpen] = useState(false)
  const [creating, setCreating] = useState(false)
  const [name, setName] = useState('')
  const box = useRef(null)

  useEffect(() => {
    const away = (e) => { if (box.current && !box.current.contains(e.target)) { setOpen(false); setCreating(false) } }
    document.addEventListener('mousedown', away)
    return () => document.removeEventListener('mousedown', away)
  }, [])

  const current = projects.find((p) => p.id === active)
  const create = async () => {
    if (!name.trim()) return
    await onCreate(name.trim())
    setName(''); setCreating(false); setOpen(false)
  }

  return (
    <div className="proj" ref={box}>
      <button className="current" onClick={() => setOpen(!open)} title="Switch project">
        <span className="ic">◆</span>
        <span className="nm">{current?.name || 'Workspace'}</span>
        <span style={{ color: 'var(--muted)', fontSize: 11 }}>{current?.documents ?? 0}</span>
        <span style={{ color: 'var(--muted)' }}>▾</span>
      </button>
      {open && (
        <div className="menu">
          {projects.map((p) => (
            <button key={p.id} className={p.id === active ? 'on' : ''}
                    onClick={() => { onSwitch(p.id); setOpen(false) }}>
              <span style={{ color: p.id === active ? 'var(--accent)' : 'transparent' }}>●</span>
              <span style={{ flex: 1 }}>{p.name}</span>
              <span style={{ color: 'var(--muted)', fontSize: 11 }}>{p.documents}</span>
            </button>
          ))}
          <div className="sep" />
          {creating ? (
            <div style={{ padding: 6 }}>
              <input autoFocus placeholder="Project name" value={name}
                     onChange={(e) => setName(e.target.value)}
                     onKeyDown={(e) => e.key === 'Enter' && create()} />
              <div className="row" style={{ marginTop: 7 }}>
                <Btn sm kind="primary" onClick={create}>Create</Btn>
                <Btn sm kind="ghost" onClick={() => setCreating(false)}>Cancel</Btn>
              </div>
              <Note style={{ marginTop: 6 }}>A project keeps its own documents, index and entities.</Note>
            </div>
          ) : (
            <>
              <button onClick={() => setCreating(true)}><span>＋</span>New project</button>
              {projects.length > 1 && current && (
                <button onClick={() => { if (confirm(`Delete “${current.name}” and everything in it?`)) { onDelete(current.id); setOpen(false) } }}>
                  <span>🗑</span>Delete “{current.name}”
                </button>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}
