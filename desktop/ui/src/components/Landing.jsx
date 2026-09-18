import React, { useEffect, useRef, useState } from 'react'
import { Btn, Pill } from '../ui.jsx'

const WORDS = ['contracts.', 'invoices.', 'manuals.', 'lab reports.', 'site photos.', 'spreadsheets.']

const TILES = [
  { ic: '◱', t: 'Multimodal indexing', d: 'PDFs, scans, images, spreadsheets and notes — every page classified, not flattened into text.' },
  { ic: '⌕', t: 'Hybrid retrieval', d: 'Meaning, keywords, exact codes and an entity graph, fused into one ranked evidence set.' },
  { ic: '☷', t: 'Table-aware answers', d: 'Numeric questions are computed from the real rows. The model never does arithmetic.' },
  { ic: '❏', t: 'Photo search', d: 'Text → image, image → image, image → document, with a vision model that runs on your device.' },
  { ic: '✓', t: 'Verified citations', d: 'Every number, date and code must exist in a retrieved page, or the answer is refused.' },
  { ic: '▲', t: 'Edge AI', d: 'CPU today, Hexagon NPU when a Snapdragon device is detected — never claimed otherwise.' },
]

const FACTS = ['0 cloud calls', 'no OpenAI API', 'no Gemini API', 'no cloud vector database',
  'works in airplane mode', 'documents never leave the device', 'local CLIP + local LLM',
  'evidence for every claim']

function useReveal() {
  const ref = useRef(null)
  useEffect(() => {
    const els = ref.current?.querySelectorAll('.reveal') || []
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => e.isIntersecting && e.target.classList.add('in'))
    }, { threshold: 0.12 })
    els.forEach((el) => io.observe(el))
    return () => io.disconnect()
  }, [])
  return ref
}

function Typewriter() {
  const [i, setI] = useState(0)
  const [text, setText] = useState('')
  const [back, setBack] = useState(false)
  useEffect(() => {
    const word = WORDS[i % WORDS.length]
    const done = !back && text === word
    const empty = back && text === ''
    const delay = done ? 1400 : empty ? 220 : back ? 34 : 68
    const t = setTimeout(() => {
      if (done) setBack(true)
      else if (empty) { setBack(false); setI(i + 1) }
      else setText(back ? word.slice(0, text.length - 1) : word.slice(0, text.length + 1))
    }, delay)
    return () => clearTimeout(t)
  }, [text, back, i])
  return <span className="type">{text}</span>
}

function Counter({ to, suffix = '' }) {
  const [n, setN] = useState(0)
  useEffect(() => {
    let raf, start
    const step = (ts) => {
      start = start || ts
      const p = Math.min(1, (ts - start) / 1100)
      setN(Math.round(to * (1 - Math.pow(1 - p, 3))))
      if (p < 1) raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [to])
  return <span className="counter">{n}{suffix}</span>
}

export default function Landing({ onEnter, auth, unlocked }) {
  const ref = useReveal()
  return (
    <div className="landing" ref={ref}>
      <div className="aurora"><span /><span /><span /></div>
      <div className="grid-lines" />
      <div className="land-wrap">
        <nav className="land-nav">
          <div className="brand">
            <div className="mark">EV</div>
            <div className="name">EdgeVault</div>
          </div>
          <div className="spacer" style={{ flex: 1 }} />
          <Pill tone="good"><span className="dot" />Local processing</Pill>
          <Btn onClick={onEnter}>{unlocked ? 'Continue' : auth?.enabled ? 'Unlock' : 'Open workspace'}</Btn>
        </nav>

        <header className="hero">
          <h1>Your private knowledge engine<br />for <Typewriter /></h1>
          <p className="sub">
            EdgeVault indexes documents, images, tables and the relationships between them on your own
            device, retrieves evidence with hybrid search, and answers with local AI — with citations you
            can open, and nothing sent to a cloud service.
          </p>
          <div className="hero-cta">
            <span className="glow"><Btn kind="primary" className="btn-xl" onClick={onEnter}>
              {unlocked ? 'Continue to your workspace →'
                        : auth?.enabled ? 'Unlock workspace →' : 'Create your workspace →'}
            </Btn></span>
            <Btn onClick={() => document.getElementById('how')?.scrollIntoView({ behavior: 'smooth' })}>
              See how it works
            </Btn>
          </div>
        </header>

        <div className="marquee">
          <div className="track">
            {[...FACTS, ...FACTS].map((f, i) => <span className="pill" key={i}>{f}</span>)}
          </div>
        </div>

        <section id="how" className="tiles three">
          {TILES.map((t, i) => (
            <div className="tile reveal" key={t.t} style={{ transitionDelay: `${i * 70}ms` }}>
              <div className="ic">{t.ic}</div>
              <h4>{t.t}</h4>
              <p>{t.d}</p>
            </div>
          ))}
        </section>

        <h3 className="steps-title reveal">How it works</h3>
        <section className="tiles three steps reveal">
          <div className="tile">
            <h4>1 · Create a project</h4>
            <p>Each project is its own workspace, with its own documents, index, entities and answers.</p>
          </div>
          <div className="tile">
            <h4>2 · Drop in your files</h4>
            <p>Text is extracted, scans are read with OCR, tables keep their rows, images are indexed.</p>
          </div>
          <div className="tile">
            <h4>3 · Ask, search, compare</h4>
            <p>Ask in words or by voice, search by photo, compare two versions and explore the entity graph.</p>
          </div>
        </section>

        <div className="land-foot">
          Runs on CPU today · uses the Hexagon NPU automatically on a Qualcomm Snapdragon device
        </div>
      </div>
    </div>
  )
}
