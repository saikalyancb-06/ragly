import React, { useEffect, useRef } from 'react'

function Answer({ text, onCite }) {
  // render [1] [2] as clickable chips
  const parts = String(text).split(/(\[\d+\])/g)
  return (
    <>
      {parts.map((p, i) => {
        const m = p.match(/^\[(\d+)\]$/)
        if (!m) return <span key={i}>{p}</span>
        return (
          <span key={i} className="cite" onClick={() => onCite(Number(m[1]))}>
            [{m[1]}]
          </span>
        )
      })}
    </>
  )
}

function Verdict({ g, timing, model }) {
  if (!g) return null
  const speed = timing?.llm_tokens_per_sec
  return (
    <div className="verdict">
      {g.verified ? <span className="good">✓ verified against sources</span> : null}
      {g.flag === 'low_relevance' && <span className="bad">no relevant passage (score {g.best_score})</span>}
      {g.flag === 'unsupported_values' && (
        <span className="bad">blocked: values not in sources ({g.unsupported_values.join(', ')})</span>
      )}
      {g.flag === 'hedged_language' && <span className="bad">Blocked: the answer hedged instead of citing ({g.hedges.join(', ')})</span>}
      {g.flag === 'uncited' && <span className="bad">blocked: no citation</span>}
      {g.dropped_sentences?.length > 0 && <span>dropped {g.dropped_sentences.length} uncited sentence(s)</span>}
      <span>Match {g.best_score}</span>
      {timing?.exact_hits > 0 && <span>{timing.exact_hits} exact code hit(s)</span>}
      {timing && <span>search {Math.round(timing.embed_ms + timing.search_ms)} ms</span>}
      {speed && <span>{speed} tok/s</span>}
      {timing?.llm_ttft_ms && <span>first word {(timing.llm_ttft_ms / 1000).toFixed(1)}s</span>}
      {model && <span title={model}>{String(model).split(/[\\/]/).pop()}</span>}
    </div>
  )
}

export default function Chat({
  messages, input, setInput, onSend, busy, suggestions, onCite, onSource,
  voice, recording, onMic, speakOn, setSpeakOn,
}) {
  const end = useRef(null)
  useEffect(() => { end.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  return (
    <div className="col chat">
      <div className="messages">
        {messages.length === 0 && (
          <div className="empty">
            Ask a question about your documents.<br />
            Answers come only from them, with the page cited.
          </div>
        )}
        {messages.map((m, i) => (
          <div className={`msg ${m.role}`} key={i}>
            <div className="who">{m.role === 'user' ? 'You' : 'Ragly'}{m.heard ? ` · heard: “${m.heard}”` : ''}</div>
            <div className={`bubble ${m.refused ? 'refused' : ''}`}>
              {m.role === 'user' ? m.text : <Answer text={m.text || '…'} onCite={(n) => onCite(m, n)} />}
            </div>
            {m.role === 'assistant' && m.sources?.length > 0 && (
              <div className="sources">
                {m.sources.map((s) => (
                  <div className="source" key={s.n} onClick={() => onSource(s)}>
                    [{s.n}] {s.doc_name} · p.{s.page}{s.exact_match ? ' · exact' : ''}
                  </div>
                ))}
              </div>
            )}
            {m.role === 'assistant' && <Verdict g={m.grounding} timing={m.timing} model={m.model} />}
          </div>
        ))}
        <div ref={end} />
      </div>

      {suggestions?.length > 0 && messages.length === 0 && (
        <div className="suggest">
          {suggestions.map((q) => (
            <span className="chip" key={q} onClick={() => onSend(q)}>{q}</span>
          ))}
        </div>
      )}

      <div className="composer">
        <input
          placeholder="Ask about your documents…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && !busy && onSend()}
          disabled={busy}
        />
        <button
          className={`mic ${recording ? 'rec' : ''}`}
          onClick={onMic}
          disabled={busy || !voice?.stt}
          title={voice?.stt ? 'Ask by voice' : voice?.stt_error || 'voice unavailable'}
        >
          {recording ? '● Listening' : '🎤 speak'}
        </button>
        <button
          className={`chip click ${speakOn ? 'ok' : ''}`}
          onClick={() => setSpeakOn(!speakOn)}
          disabled={!voice?.tts}
          title="Read answers aloud"
        >
          🔊 {speakOn ? 'on' : 'off'}
        </button>
        <button className="primary" onClick={() => onSend()} disabled={busy || !input.trim()}>
          {busy ? '…' : 'Ask'}
        </button>
      </div>
    </div>
  )
}
