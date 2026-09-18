import React, { useEffect, useState } from 'react'
import { api } from '../api.js'
import { Bar, Btn, Card, Empty, Err, Note, Pill, fmtMs, fmtNum } from '../ui.jsx'

const METRICS = [
  { key: 'answer_correct_pct', label: 'Answer correctness', hint: 'All 25 questions, including the ones your documents cannot answer' },
  { key: 'answerable_correct_pct', label: 'Answerable correct', hint: 'Only the questions the corpus does answer' },
  { key: 'retrieval_recall_pct', label: 'Retrieval recall', hint: 'Share of the expected pages that reached the model' },
  { key: 'retrieval_precision_pct', label: 'Retrieval precision', hint: 'Share of retrieved passages that were relevant' },
  { key: 'retrieval_top1_pct', label: 'Top-1 correct', hint: 'The best passage was the right one' },
  { key: 'citation_correct_pct', label: 'Citation correctness', hint: 'Cited page matches the page the fact came from' },
  { key: 'groundedness_pct', label: 'Groundedness', hint: 'Every value in the answer traced back to a source' },
  { key: 'unsupported_claim_rate_pct', label: 'Unsupported claims', hint: 'Answers containing a value absent from the sources', lowerIsBetter: true },
  { key: 'abstention_accuracy_pct', label: 'Abstention accuracy', hint: 'Correctly refused when the documents do not say' },
  { key: 'hallucination_pct', label: 'Hallucination rate', hint: 'Answered anyway when the documents do not say', lowerIsBetter: true },
  { key: 'over_refusal_pct', label: 'Over-refusal rate', hint: 'Refused although the evidence was there', lowerIsBetter: true },
  { key: 'numerical_accuracy_pct', label: 'Numerical accuracy', hint: 'Numbers, tables and dates' },
]

const tone = (m, v) => {
  if (v == null) return ''
  const good = m.lowerIsBetter ? v <= 5 : v >= 90
  const bad = m.lowerIsBetter ? v > 20 : v < 70
  return good ? 'good' : bad ? 'bad' : 'warn'
}

function Evaluation() {
  const [reports, setReports] = useState([])
  const [command, setCommand] = useState('')
  const [pick, setPick] = useState(0)
  const [error, setError] = useState('')

  useEffect(() => {
    api.evalReports()
      .then((r) => { setReports(r.reports || []); setCommand(r.command || '') })
      .catch((e) => setError(String(e.message || e)))
  }, [])

  const full = reports.filter((r) => !r.retrieval_only)
  const report = full[pick]
  const previous = full[pick + 1]

  return (
    <Card title="RAG accuracy — measured on a fixed question set">
      {error && <Err>{error}</Err>}
      <Note>
        These numbers come from the last evaluation run on this machine, not from a leaderboard. Each question has a
        known answer and a known source page, so a wrong page and a wrong answer are counted separately.
      </Note>
      {full.length === 0 ? (
        <Empty>
          No evaluation has been run yet. Run <span className="mono">{command || 'python scripts/rag_eval.py --ingest'}</span> to
          produce one.
        </Empty>
      ) : (
        <>
          <div className="chips" style={{ marginTop: 12 }}>
            {full.map((r, i) => (
              <Pill key={r.file} tone={i === pick ? 'accent' : ''} onClick={() => setPick(i)}>
                {r.label} · {new Date(r.timestamp).toLocaleString()}
              </Pill>
            ))}
          </div>

          <div className="chips" style={{ marginTop: 12 }}>
            <Pill title={report.model || ''}>{String(report.model || 'unknown model').split(/[\\/]/).pop()}</Pill>
            <Pill>{report.questions} questions</Pill>
            <Pill>{report.hardware?.backend || 'CPU'}</Pill>
            {report.model_load_s != null && <Pill>Model load {report.model_load_s}s</Pill>}
            {report.peak_ram_mb != null && <Pill>Peak RAM {fmtNum(report.peak_ram_mb, 0)} MB</Pill>}
            {report.metrics?.latency_p50_ms != null && <Pill>Median answer {fmtMs(report.metrics.latency_p50_ms)}</Pill>}
            {report.metrics?.latency_p95_ms != null && <Pill>p95 {fmtMs(report.metrics.latency_p95_ms)}</Pill>}
          </div>

          <table className="tbl" style={{ marginTop: 14 }}>
            <thead>
              <tr><th>Metric</th><th>Score</th><th className="num">Value</th>
                {previous && <th className="num">vs {previous.label}</th>}</tr>
            </thead>
            <tbody>
              {METRICS.map((m) => {
                const v = report.metrics?.[m.key]
                if (v == null) return null
                const before = previous?.metrics?.[m.key]
                const delta = before == null ? null : Math.round((v - before) * 10) / 10
                return (
                  <tr key={m.key}>
                    <td title={m.hint}>{m.label}</td>
                    <td><Bar value={v} max={100} format={(x) => `${x}%`} /></td>
                    <td className="num"><Pill tone={tone(m, v)}>{v}%</Pill></td>
                    {previous && (
                      <td className="num mono">
                        {delta == null ? '—' : delta === 0 ? '±0' : `${delta > 0 ? '+' : ''}${delta}`}
                      </td>
                    )}
                  </tr>
                )
              })}
            </tbody>
          </table>

          {report.metrics?.failures && (
            <>
              <h3 style={{ marginTop: 16 }}>Where the failures came from</h3>
              <div className="chips">
                <Pill tone={report.metrics.failures.retrieval ? 'warn' : 'good'}>
                  Retrieval {report.metrics.failures.retrieval}</Pill>
                <Pill tone={report.metrics.failures.generation ? 'warn' : 'good'}>
                  Generation {report.metrics.failures.generation}</Pill>
                <Pill tone={report.metrics.failures.grounding ? 'warn' : 'good'}>
                  Grounding {report.metrics.failures.grounding}</Pill>
              </div>
            </>
          )}

          {report.metrics?.by_category && (
            <>
              <h3 style={{ marginTop: 16 }}>By question type</h3>
              <table className="tbl">
                <thead><tr><th>Type</th><th className="num">Correct</th><th>Score</th></tr></thead>
                <tbody>
                  {Object.entries(report.metrics.by_category).map(([cat, v]) => (
                    <tr key={cat}>
                      <td>{cat.replace(/_/g, ' ')}</td>
                      <td className="num mono">{v.correct}/{v.n}</td>
                      <td><Bar value={v.pct} max={100} format={(x) => `${x}%`} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </>
      )}
    </Card>
  )
}

export default function Benchmarks() {
  const [runs, setRuns] = useState([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const load = async () => {
    try { setRuns((await api.benchmarkHistory()).runs || []) } catch (e) { setError(String(e.message || e)) }
  }
  useEffect(() => { load() }, [])

  const run = async () => {
    setBusy(true); setError('')
    try { await api.benchmark(5, true); await load() } catch (e) { setError(String(e.message || e)) }
    setBusy(false)
  }

  const max = Math.max(1, ...runs.map((r) => r.latency_ms || 0))
  return (
    <div className="wrap">
      <Evaluation />
      <Card title="Speed — measured on this machine"
            actions={<Btn sm kind="primary" onClick={run} disabled={busy}>{busy ? 'Measuring…' : 'Run benchmark'}</Btn>}>
        {error && <Err>{error}</Err>}
        <Note>
          Each row is a real workload timed just now. Run the same benchmark on a Snapdragon device to compare
          CPU with the Hexagon NPU — nothing here is estimated or copied from a datasheet.
        </Note>
        {runs.length === 0 ? <Empty>No runs yet.</Empty> : (
          <table className="tbl" style={{ marginTop: 12 }}>
            <thead>
              <tr><th>Task</th><th>Model</th><th>Backend</th><th>Latency</th>
                <th className="num">Throughput</th><th className="num">Memory</th><th className="num">When</th></tr>
            </thead>
            <tbody>
              {runs.map((r, i) => (
                <tr key={i}>
                  <td>{r.task.replace(/_/g, ' ')}</td>
                  <td className="mono" style={{ maxWidth: 240, overflowWrap: 'anywhere' }}>
                    {String(r.model).split(/[\\/]/).pop()}
                  </td>
                  <td><Pill tone={r.backend?.startsWith('NPU') ? 'accent' : ''}>{r.backend}</Pill></td>
                  <td><Bar value={r.latency_ms || 0} max={max} format={fmtMs} /></td>
                  <td className="num">{r.throughput ? `${fmtNum(r.throughput)} ${r.extra?.unit || ''}` : '—'}</td>
                  <td className="num">{r.memory_mb ? `${fmtNum(r.memory_mb, 0)} MB` : '—'}</td>
                  <td className="num">{new Date(r.ran_at * 1000).toLocaleTimeString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  )
}
