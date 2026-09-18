import React, { useEffect, useMemo, useRef, useState } from 'react'

const KIND_COLOR = {
  org: '#5b8cff', person: '#8b7cff', invoice: '#3ddc97', contract: '#ffb454',
  project: '#57d2ff', place: '#c58bff', amount: '#8fa0b4', date: '#6f7d8c',
  id: '#a0d468', code: '#a0d468', default: '#7b8794',
}
const colour = (kind) => KIND_COLOR[kind] || KIND_COLOR.default

/** Tiny force-directed layout (no dependency): repulsion + spring edges, a few hundred ticks. */
function layout(nodes, edges, w, h) {
  const pos = new Map()
  nodes.forEach((n, i) => {
    const a = (i / Math.max(1, nodes.length)) * Math.PI * 2
    pos.set(n.id, { x: w / 2 + Math.cos(a) * (w / 3.2), y: h / 2 + Math.sin(a) * (h / 3.2), vx: 0, vy: 0 })
  })
  const links = edges.filter((e) => pos.has(e.src) && pos.has(e.dst))
  for (let step = 0; step < 240; step++) {
    for (const a of nodes) {
      const pa = pos.get(a.id)
      for (const b of nodes) {
        if (a.id === b.id) continue
        const pb = pos.get(b.id)
        let dx = pa.x - pb.x, dy = pa.y - pb.y
        let d2 = dx * dx + dy * dy || 0.01
        const f = 2600 / d2
        pa.vx += dx * f * 0.002
        pa.vy += dy * f * 0.002
      }
      pa.vx += (w / 2 - pa.x) * 0.0016
      pa.vy += (h / 2 - pa.y) * 0.0016
    }
    for (const e of links) {
      const a = pos.get(e.src), b = pos.get(e.dst)
      const dx = b.x - a.x, dy = b.y - a.y
      const d = Math.hypot(dx, dy) || 0.01
      const f = (d - 120) * 0.008
      a.vx += dx / d * f; a.vy += dy / d * f
      b.vx -= dx / d * f; b.vy -= dy / d * f
    }
    for (const n of nodes) {
      const p = pos.get(n.id)
      p.x += p.vx; p.y += p.vy
      p.vx *= 0.82; p.vy *= 0.82
      p.x = Math.max(30, Math.min(w - 30, p.x))
      p.y = Math.max(26, Math.min(h - 26, p.y))
    }
  }
  return pos
}

export default function GraphCanvas({ nodes, edges, selectedId, onSelect, height = 520 }) {
  const W = 1000, H = height
  const [view, setView] = useState({ x: 0, y: 0, k: 1 })
  const pos = useMemo(() => layout(nodes, edges, W, H), [nodes, edges, H])

  useEffect(() => { setView({ x: 0, y: 0, k: 1 }) }, [nodes.length])

  // pan by dragging the canvas
  const panStart = useRef(null)
  const startPan = (e) => { panStart.current = { mx: e.clientX, my: e.clientY, x: view.x, y: view.y } }
  const doPan = (e) => {
    if (!panStart.current) return
    setView((v) => ({ ...v, x: panStart.current.x + (e.clientX - panStart.current.mx),
                      y: panStart.current.y + (e.clientY - panStart.current.my) }))
  }
  const endPan = () => { panStart.current = null }

  const zoom = (f) => setView((v) => ({ ...v, k: Math.max(0.4, Math.min(2.6, v.k * f)) }))
  const kinds = [...new Set(nodes.map((n) => n.kind))]

  return (
    <div className="canvas-wrap">
      <svg viewBox={`0 0 ${W} ${H}`} onMouseDown={startPan} onMouseMove={doPan}
           onMouseUp={endPan} onMouseLeave={endPan}
           onWheel={(e) => zoom(e.deltaY < 0 ? 1.1 : 0.9)}>
        <g transform={`translate(${view.x} ${view.y}) scale(${view.k})`}>
          {edges.map((e, i) => {
            const a = pos.get(e.src), b = pos.get(e.dst)
            if (!a || !b) return null
            const on = selectedId && (e.src === selectedId || e.dst === selectedId)
            return (
              <g key={i}>
                <line x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                      stroke={on ? 'rgba(91,140,255,.75)' : 'rgba(255,255,255,.10)'} strokeWidth={on ? 1.6 : 1} />
                {on && (
                  <text x={(a.x + b.x) / 2} y={(a.y + b.y) / 2 - 4} fill="#8fa0b4" fontSize="9"
                        textAnchor="middle">{e.rel}</text>
                )}
              </g>
            )
          })}
          {nodes.map((n) => {
            const p = pos.get(n.id)
            if (!p) return null
            const r = Math.min(18, 7 + Math.sqrt(n.mentions || 1) * 2.2)
            return (
              <g key={n.id} className={`gnode ${selectedId === n.id ? 'sel' : ''}`}
                 onClick={(e) => { e.stopPropagation(); onSelect(n) }}>
                <circle cx={p.x} cy={p.y} r={r} fill={colour(n.kind)} fillOpacity={selectedId === n.id ? 1 : .8} />
                <text x={p.x} y={p.y + r + 11} textAnchor="middle">
                  {n.name.length > 22 ? `${n.name.slice(0, 21)}…` : n.name}
                </text>
              </g>
            )
          })}
        </g>
      </svg>
      <div className="zoomers">
        <button className="btn sm" onClick={() => zoom(1.2)}>+</button>
        <button className="btn sm" onClick={() => zoom(0.83)}>−</button>
        <button className="btn sm" onClick={() => setView({ x: 0, y: 0, k: 1 })}>⟲</button>
      </div>
      <div className="legend">
        {kinds.map((k) => (
          <span className="pill" key={k}>
            <span className="dot" style={{ background: colour(k) }} />{k}
          </span>
        ))}
      </div>
    </div>
  )
}
