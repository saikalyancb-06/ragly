import React from 'react'

export default function Header({ sys, onEngine, tuning, setTuning, strict, setStrict, busy }) {
  const engine = sys?.engine
  const snapdragon = sys?.device?.is_snapdragon
  const state = engine?.state
  return (
    <header className="top">
      <div className="brand">Rag<span>ly</span></div>
      <span className="chip">Offline · nothing leaves this device</span>

      <div className="toggle" title={snapdragon ? '' : 'Snapdragon mode needs a Snapdragon PC'}>
        <button className={engine?.mode === 'normal' ? 'on' : ''} disabled={busy} onClick={() => onEngine('normal')}>
          Normal (llama.cpp)
        </button>
        <button
          className={engine?.mode === 'snapdragon' ? 'on' : ''}
          disabled={busy || !snapdragon}
          onClick={() => onEngine('snapdragon')}
        >
          Snapdragon (GenieX)
        </button>
      </div>

      <span className={`chip ${state === 'ready' ? 'ok' : state === 'error' ? 'bad' : 'warn'}`}>
        {state === 'ready' ? engine?.compute : state === 'starting' ? 'Loading the model…' : state || '…'}
      </span>
      {sys?.embedder && <span className="chip">embed: {sys.embedder.provider.replace('ExecutionProvider', '')}</span>}
      {sys?.index && <span className="chip">{sys.index.documents} docs · {sys.index.chunks} chunks</span>}

      <div className="spacer" />
      <span className={`chip click ${tuning ? 'ok' : ''}`} onClick={() => setTuning(!tuning)}>
        auto-tuning: {tuning ? 'on' : 'off'}
      </span>
      <span className={`chip click ${strict ? 'ok' : 'warn'}`} onClick={() => setStrict(!strict)}>
        strict grounding: {strict ? 'on' : 'off'}
      </span>
    </header>
  )
}
