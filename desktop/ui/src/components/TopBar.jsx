import React from 'react'
import { Pill, Seg } from '../ui.jsx'
import { GROUPS } from './Sidebar.jsx'

const TITLES = Object.fromEntries(GROUPS.flatMap((g) => g.items.map((i) => [i.id, i.label])))
const SUBS = {
  overview: 'Everything indexed on this device',
  documents: 'Import files and see how each page was read',
  images: 'Every image extracted from your documents',
  tables: 'Extracted tables, and arithmetic computed from them',
  ask: 'Answers grounded in your own documents',
  photo: 'Text → image, image → image, image → document',
  compare: 'What changed between two documents',
  graph: 'Entities and how they connect',
  edge: 'Hardware, backends, and what actually ran where',
  bench: 'Measured on this machine, never estimated',
  models: 'What is installed, and where each model can run',
  settings: 'Retrieval, engine, privacy and data',
}

export default function TopBar({ page, sys, onEngine, switching }) {
  const engine = sys?.engine
  const hw = sys?.hardware
  const state = engine?.state
  return (
    <header className="topbar">
      <span className="title">{TITLES[page] || ''}</span>
      <span className="sub">{SUBS[page] || ''}</span>
      <div className="spacer" />
      <Pill tone={state === 'ready' ? '' : state === 'error' ? 'bad' : 'warn'}>
        {state === 'ready' ? engine?.compute : state === 'starting' ? 'Loading the model…' : state || '…'}
      </Pill>
      <Seg
        value={engine?.mode || 'normal'}
        onChange={onEngine}
        options={[
          { value: 'normal', label: 'CPU', disabled: switching },
          { value: 'snapdragon', label: 'NPU', disabled: switching || !hw?.is_snapdragon,
            title: hw?.is_snapdragon ? 'Snapdragon / GenieX' : 'Snapdragon device required' },
        ]}
      />
      <Pill tone="good"><span className="dot" />LOCAL / OFFLINE</Pill>
    </header>
  )
}
