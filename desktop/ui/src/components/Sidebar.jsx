import React from 'react'
import FalconLogo from './FalconLogo.jsx'
import ProjectSwitcher from './ProjectSwitcher.jsx'

export const GROUPS = [
  { title: 'Workspace', items: [
    { id: 'overview', label: 'Overview', ic: '◱' },
    { id: 'documents', label: 'Documents', ic: '▤' },
    { id: 'images', label: 'Images', ic: '▣' },
    { id: 'tables', label: 'Tables', ic: '☷' },
  ] },
  { title: 'Intelligence', items: [
    { id: 'ask', label: 'Ask', ic: '✦' },
    { id: 'photo', label: 'Photo Search', ic: '⌕' },
    { id: 'compare', label: 'Compare', ic: '⇄' },
    { id: 'graph', label: 'Visualize', ic: '◈' },
  ] },
  { title: 'System', items: [
    { id: 'edge', label: 'Edge AI', ic: '▲' },
    { id: 'bench', label: 'Accuracy & Speed', ic: '◷' },
    { id: 'models', label: 'Models', ic: '◇' },
    { id: 'settings', label: 'Settings', ic: '⚙' },
  ] },
]

export default function Sidebar({ page, setPage, sys, onAdd, projects, onSwitchProject, onCreateProject, onDeleteProject, user, onLock, onHome }) {
  const counts = {
    documents: sys?.index?.documents,
    images: sys?.images?.images,
    tables: sys?.tables,
    graph: sys?.graph?.nodes,
  }
  return (
    <aside className="side">
      <div className="brand" onClick={onHome} style={{ cursor: onHome ? 'pointer' : 'default' }}>
        <FalconLogo size={26} />
        <div className="name">Falcon</div>
      </div>
      <ProjectSwitcher projects={projects || []} active={sys?.project?.id}
                       onSwitch={onSwitchProject} onCreate={onCreateProject} onDelete={onDeleteProject} />
      <button className="add-btn" onClick={onAdd}>+ Add Documents</button>
      {GROUPS.map((g) => (
        <div className="navgroup" key={g.title}>
          <h3>{g.title}</h3>
          <div className="nav">
            {g.items.map((it) => (
              <button key={it.id} className={page === it.id ? 'on' : ''} onClick={() => setPage(it.id)}>
                <span className="ic">{it.ic}</span>{it.label}
                {counts[it.id] ? <span className="count">{counts[it.id]}</span> : null}
              </button>
            ))}
          </div>
        </div>
      ))}
      <div className="foot">
        <span className="local"><span className="dot" />Local processing</span>
        <div style={{ marginTop: 8 }}>{sys?.hardware?.summary || 'Detecting hardware…'}</div>
        <div style={{ marginTop: 6, opacity: .7 }} title="the build this install is running">
          Build {sys?.build || '—'}{sys?.vision?.vision_model ? ' · vision model installed' : ' · no vision model'}
        </div>
        <div className="row" style={{ marginTop: 10, justifyContent: 'space-between' }}>
          <span>{user?.display_name || 'Local user'}</span>
          <span className="row" style={{ gap: 4 }}>
            <button className="btn sm ghost" onClick={onHome} title="back to the start page">Home</button>
            <button className="btn sm ghost" onClick={onLock} title="sign out of this device">Sign out</button>
          </span>
        </div>
      </div>
    </aside>
  )
}
