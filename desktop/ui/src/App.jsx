import React, { useCallback, useEffect, useRef, useState } from 'react'
import { api, getToken, setToken } from './api.js'
import AuthScreen from './components/AuthScreen.jsx'
import DocViewer from './components/DocViewer.jsx'
import Landing from './components/Landing.jsx'
import Sidebar from './components/Sidebar.jsx'
import TopBar from './components/TopBar.jsx'
import Ask from './pages/Ask.jsx'
import Benchmarks from './pages/Benchmarks.jsx'
import Compare from './pages/Compare.jsx'
import Documents from './pages/Documents.jsx'
import EdgeAI from './pages/EdgeAI.jsx'
import Graph from './pages/Graph.jsx'
import Images from './pages/Images.jsx'
import Models from './pages/Models.jsx'
import Overview from './pages/Overview.jsx'
import PhotoSearch from './pages/PhotoSearch.jsx'
import Settings from './pages/Settings.jsx'
import Tables from './pages/Tables.jsx'
import { Err } from './ui.jsx'

export default function App() {
  const [screen, setScreen] = useState('landing')     // landing | auth | app
  const [auth, setAuth] = useState(null)
  const [unlocked, setUnlocked] = useState(false)
  const [projects, setProjects] = useState([])
  const [page, setPage] = useState('overview')
  const [sys, setSys] = useState(null)
  const [docs, setDocs] = useState([])
  const [pack, setPack] = useState(null)
  const [suggestions, setSuggestions] = useState([])
  const [source, setSource] = useState(null)
  const [prefill, setPrefill] = useState('')
  const [switching, setSwitching] = useState(false)
  const [error, setError] = useState('')
  const [settings, setLocalSettings] = useState({
    top_k: 5, min_score: 0.45, max_tokens: 512, strict_grounding: true, tuning: true,
    // Display only: the answer stands alone unless someone asks to see where it came from.
    show_evidence: false,
  })
  const addRef = useRef(null)

  const reload = useCallback(async () => {
    try {
      const s = await api.system()
      setSys(s); setPack(s.pack); setProjects(s.projects || [])
      setLocalSettings((cur) => ({ ...cur, ...s.settings }))
      setDocs((await api.documents()).documents)
      setError('')
    } catch (e) { setError(String(e.message || e)) }
  }, [])

  // first load: is a passcode set, and is the stored session still valid?
  useEffect(() => {
    (async () => {
      try {
        api.system().then(setSys).catch(() => {})   // the landing page shows real counts
        const st = await api.authStatus()
        setAuth(st)
        const token = getToken()
        if (!st.enabled) return                    // no passcode yet: the landing page invites setup
        // The landing page is always what opens first. A still-valid session just means the
        // Enter button goes straight in instead of asking for the passcode again.
        if (token && (await api.checkToken(token)).valid) setUnlocked(true)
        else setToken(null)
      } catch { /* backend not ready yet */ }
    })()
  }, [])

  useEffect(() => { if (screen === 'app') reload() }, [reload, screen])
  useEffect(() => {
    if (screen !== 'app') return undefined
    const t = setInterval(reload, 4000)
    return () => clearInterval(t)
  }, [reload, screen])
  useEffect(() => { api.suggestions().then((r) => setSuggestions(r.items || (r.questions || []).map((q) => ({ question: q })))).catch(() => {}) }, [pack])

  const setSettings = async (values) => {
    setLocalSettings((cur) => ({ ...cur, ...values }))
    const { tuning, show_evidence, ...server } = values
    if (Object.keys(server).length) {
      try { await api.settings(server) } catch (e) { setError(String(e.message || e)) }
    }
  }

  const onEngine = async (mode) => {
    setSwitching(true); setError('')
    try { await api.setEngine(mode) } catch (e) { setError(`Could not switch: ${e.message}`) }
    setSwitching(false); reload()
  }

  const openAdd = () => {
    setPage('documents')
    setTimeout(() => addRef.current?.(), 60)
  }

  const switchProject = async (id) => {
    try { await api.activateProject(id) } catch (e) { setError(String(e.message || e)) }
    setSource(null); setPage('overview'); reload()
  }
  const createProject = async (name) => {
    try { await api.createProject(name) } catch (e) { setError(String(e.message || e)) }
    setPage('documents'); reload()
  }
  const deleteProject = async (id) => {
    try { await api.deleteProject(id) } catch (e) { setError(String(e.message || e)) }
    reload()
  }
  const lock = () => { setToken(null); setUnlocked(false); setScreen('landing') }

  const shared = { sys, docs, pack, reload, setPage, setSource, settings, setSettings, setPrefill }
  const PAGES = {
    overview: <Overview {...shared} onAdd={openAdd} />,
    documents: <Documents {...shared} addRef={addRef} />,
    images: <Images {...shared} />,
    tables: <Tables {...shared} />,
    ask: <Ask {...shared} suggestions={suggestions} prefill={prefill} clearPrefill={() => setPrefill('')} />,
    photo: <PhotoSearch {...shared} />,
    compare: <Compare {...shared} />,
    graph: <Graph {...shared} />,
    edge: <EdgeAI {...shared} />,
    bench: <Benchmarks />,
    models: <Models />,
    settings: <Settings {...shared} />,
  }

  if (screen === 'landing') {
    return (
      <Landing auth={auth} unlocked={unlocked}
               onEnter={() => setScreen(unlocked ? 'app' : 'auth')} />
    )
  }
  if (screen === 'auth') {
    return (
      <AuthScreen auth={auth} onBack={() => setScreen('landing')}
                  onDone={(token) => {
                    setToken(token)
                    setUnlocked(true)
                    api.authStatus().then(setAuth).catch(() => {})
                    setScreen('app')
                  }} />
    )
  }

  return (
    <div className="shell">
      <Sidebar page={page} setPage={setPage} sys={sys} onAdd={openAdd} projects={projects}
               onSwitchProject={switchProject} onCreateProject={createProject}
               onDeleteProject={deleteProject} user={auth} onLock={lock}
               onHome={() => setScreen('landing')} />
      <div className="main">
        <TopBar page={page} sys={sys} onEngine={onEngine} switching={switching}
                onHome={() => setScreen('landing')} />
        <div className="page">
          {error && <div className="wrap" style={{ marginBottom: 12 }}><Err>{error}</Err></div>}
          {PAGES[page]}
        </div>
      </div>
      <DocViewer source={source} docs={docs} onClose={() => setSource(null)}
                 onAsk={(s) => { setPrefill(`In ${s.doc_name} page ${s.page}: `); setSource(null); setPage('ask') }} />
    </div>
  )
}
