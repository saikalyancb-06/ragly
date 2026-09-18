import React from 'react'

export const Card = ({ title, actions, children, style }) => (
  <div className="card" style={style}>
    {(title || actions) && (
      <div className="head">
        {title && <h2>{title}</h2>}
        <div className="spacer" />
        {actions}
      </div>
    )}
    {children}
  </div>
)

export const Stat = ({ label, value, hint }) => (
  <div className="stat">
    <div className="label">{label}</div>
    <div className="value">{value}</div>
    {hint && <div className="hint">{hint}</div>}
  </div>
)

export const Pill = ({ tone = '', children, onClick, title }) => (
  <span className={`pill ${tone} ${onClick ? 'click' : ''}`} onClick={onClick} title={title}>
    {children}
  </span>
)

export const Btn = ({ kind = '', sm, className = '', children, ...rest }) => (
  <button className={`btn ${kind} ${sm ? 'sm' : ''} ${className}`} {...rest}>{children}</button>
)

export const Switch = ({ on, onChange, label }) => (
  <span className={`switch ${on ? 'on' : ''}`} onClick={() => onChange(!on)}>
    <span className="track" />
    {label}
  </span>
)

export const Seg = ({ value, options, onChange }) => (
  <span className="seg">
    {options.map((o) => (
      <button key={o.value} className={value === o.value ? 'on' : ''} onClick={() => onChange(o.value)}
              disabled={o.disabled} title={o.title || ''}>
        {o.label}
      </button>
    ))}
  </span>
)

export const Empty = ({ children }) => <div className="empty">{children}</div>
export const Spin = () => <span className="spin" />
export const Note = ({ children }) => <div className="note">{children}</div>
export const Err = ({ children }) => (children ? <div className="err">{children}</div> : null)

export const KV = ({ k, v }) => (
  <div className="kv"><span>{k}</span><span>{v}</span></div>
)

/** Single-hue bar for one measure across rows; the value is always labelled. */
export const Bar = ({ value, max, format }) => {
  const pct = max > 0 ? Math.max(2, Math.round((value / max) * 100)) : 2
  return (
    <div className="row" style={{ gap: 8, flexWrap: 'nowrap' }}>
      <div className="bar track" style={{ maxWidth: 120 }}>
        <div className="bar" style={{ width: `${pct}%` }} />
      </div>
      <span className="mono" style={{ whiteSpace: 'nowrap' }}>{format ? format(value) : value}</span>
    </div>
  )
}

export const fmtMs = (v) => (v == null ? '—' : v < 1000 ? `${Math.round(v)} ms` : `${(v / 1000).toFixed(2)} s`)
export const fmtNum = (v, d = 1) => (v == null ? '—' : Number(v).toFixed(d))
