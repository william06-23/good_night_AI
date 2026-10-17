import { useCallback, useEffect, useRef, useState } from 'react'
import { LOADING_EXCUSES, STRINGS, loadLang, saveLang } from './i18n'
import './App.css'

const API = '/api'

/** Drop your own files at public/avatars/mia.png and user.png to replace these. */
const MIA_AVATAR = '/avatars/mia.png'
const USER_AVATAR = '/avatars/user.png'
const MIA_AVATAR_FALLBACK = '/avatars/mia.svg'
const USER_AVATAR_FALLBACK = '/avatars/user.svg'

function Avatar({ primary, fallback, label, align }) {
  const [src, setSrc] = useState(primary)
  return (
    <div className={`avatar ${align}`} title={label} aria-label={label}>
      <img
        src={src}
        alt=""
        onError={() => {
          if (src !== fallback) setSrc(fallback)
        }}
      />
    </div>
  )
}

function TypingDots({ label }) {
  return (
    <div className="row them">
      <Avatar
        primary={MIA_AVATAR}
        fallback={MIA_AVATAR_FALLBACK}
        label="Mia"
        align="left"
      />
      <div className="bubble them typing" aria-label={label}>
        <span className="dot" />
        <span className="dot" />
        <span className="dot" />
      </div>
    </div>
  )
}

export default function App() {
  const [lang, setLang] = useState(loadLang)
  const t = STRINGS[lang] || STRINGS.zh
  const [excuseIdx] = useState(() =>
    Math.floor(Math.random() * LOADING_EXCUSES.zh.length),
  )
  const loadExcuse =
    (LOADING_EXCUSES[lang] || LOADING_EXCUSES.zh)[excuseIdx] ||
    t.loadingModels

  const [ready, setReady] = useState(false)
  const [factsCount, setFactsCount] = useState(0)
  const [statusKind, setStatusKind] = useState('loading') // ready | loading | offline
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [facts, setFacts] = useState([])
  const [ended, setEnded] = useState(false)
  const [memoryOpen, setMemoryOpen] = useState(true)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const bottomRef = useRef(null)
  const inputRef = useRef(null)
  const settingsRef = useRef(null)
  const everReachedApi = useRef(false)

  const refreshMemory = useCallback(async () => {
    try {
      const res = await fetch(`${API}/memory`)
      if (!res.ok) return
      const data = await res.json()
      setFacts(data.facts || [])
    } catch {
      /* ignore until API is up */
    }
  }, [])

  const pollHealth = useCallback(async () => {
    try {
      const res = await fetch(`${API}/health`)
      const data = await res.json()
      everReachedApi.current = true
      setReady(!!data.ready)
      setFactsCount(data.facts_count ?? 0)
      // ready:false = models still loading — keep showing wait + excuse
      setStatusKind(data.ready ? 'ready' : 'loading')
      if (data.ready) refreshMemory()
      if (data.ended) setEnded(true)
    } catch {
      setReady(false)
      // First open / API not up yet: still show loading excuses, not "offline"
      setStatusKind(everReachedApi.current ? 'offline' : 'loading')
    }
  }, [refreshMemory])

  useEffect(() => {
    pollHealth()
    const id = setInterval(pollHealth, 3000)
    return () => clearInterval(id)
  }, [pollHealth])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, busy])

  useEffect(() => {
    function onDocClick(e) {
      if (!settingsRef.current?.contains(e.target)) setSettingsOpen(false)
    }
    document.addEventListener('click', onDocClick)
    return () => document.removeEventListener('click', onDocClick)
  }, [])

  function changeLang(next) {
    setLang(next)
    saveLang(next)
    setSettingsOpen(false)
  }

  async function send(text) {
    const message = (text ?? input).trim()
    if (!message || busy || !ready || ended) return

    setInput('')
    setMessages((m) => [...m, { role: 'user', content: message }])
    setBusy(true)

    try {
      const res = await fetch(`${API}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        setMessages((m) => [
          ...m,
          { role: 'system', content: data.detail || `Error ${res.status}` },
        ])
        return
      }
      setMessages((m) => [
        ...m,
        {
          role: 'assistant',
          content: data.reply,
          recalled: data.recalled || [],
        },
      ])
      if (data.ended) setEnded(true)
      await refreshMemory()
    } catch (e) {
      setMessages((m) => [
        ...m,
        { role: 'system', content: String(e.message || e) },
      ])
    } finally {
      setBusy(false)
      pollHealth()
      inputRef.current?.focus()
    }
  }

  async function resetSession() {
    setBusy(true)
    try {
      await fetch(`${API}/session/reset`, { method: 'POST' })
      setMessages([])
      setEnded(false)
      await refreshMemory()
      await pollHealth()
    } finally {
      setBusy(false)
    }
  }

  function onKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  return (
    <div className="app">
      <div className="phone">
        <header className="chat-header">
          <div className="peer">
            <Avatar
              primary={MIA_AVATAR}
              fallback={MIA_AVATAR_FALLBACK}
              label={t.name}
              align="left"
            />
            <div>
              <h1>{t.name}</h1>
              <p className="status">
                {statusKind === 'ready' && t.online(factsCount)}
                {statusKind === 'offline' && t.apiOffline}
                {statusKind === 'loading' && (
                  <>
                    <span className="wait-hint">{t.loadingTitle}</span>
                    <span className="excuse">
                      {t.loadingJustThat} {loadExcuse}
                    </span>
                  </>
                )}
              </p>
            </div>
          </div>
          <div className="header-actions">
            <div className="settings-wrap" ref={settingsRef}>
              <button
                type="button"
                className="header-btn gear"
                aria-label={t.settings}
                aria-expanded={settingsOpen}
                title={t.settings}
                onClick={(e) => {
                  e.stopPropagation()
                  setSettingsOpen((v) => !v)
                }}
              >
                ⚙️
              </button>
              {settingsOpen && (
                <div className="settings-menu" role="menu">
                  <label className="settings-row" htmlFor="lang-select">
                    <span>{t.language}</span>
                    <select
                      id="lang-select"
                      value={lang}
                      onChange={(e) => changeLang(e.target.value)}
                      onClick={(e) => e.stopPropagation()}
                    >
                      <option value="zh">中文</option>
                      <option value="en">English</option>
                    </select>
                  </label>
                </div>
              )}
            </div>
            <button
              type="button"
              className="header-btn"
              onClick={() => setMemoryOpen((v) => !v)}
              aria-expanded={memoryOpen}
            >
              {memoryOpen ? t.hideMemory : t.showMemory}
            </button>
            <button
              type="button"
              className="header-btn"
              onClick={resetSession}
              disabled={busy}
            >
              {t.newSession}
            </button>
          </div>
        </header>

        <div className={`layout ${memoryOpen ? '' : 'memory-collapsed'}`}>
          <main className="chat">
            <div className="thread">
              {!ready && (
                <div className="loading-card" role="status">
                  <p className="loading-title">{t.loadingTitle}</p>
                  <p className="just-that">
                    {t.loadingJustThat} {loadExcuse}
                  </p>
                  <p className="wait-hint">{t.loadingWait}</p>
                  <div className="loading-dots" aria-hidden="true">
                    <span className="dot" />
                    <span className="dot" />
                    <span className="dot" />
                  </div>
                </div>
              )}
              {ready && messages.length === 0 && !busy && (
                <p className="empty">{t.emptyChat}</p>
              )}
              {messages.map((msg, i) => {
                if (msg.role === 'system') {
                  return (
                    <div key={i} className="system-pill">
                      {msg.content}
                    </div>
                  )
                }
                const mine = msg.role === 'user'
                return (
                  <div key={i} className={`row ${mine ? 'mine' : 'them'}`}>
                    {!mine && (
                      <Avatar
                        primary={MIA_AVATAR}
                        fallback={MIA_AVATAR_FALLBACK}
                        label={t.name}
                        align="left"
                      />
                    )}
                    <div className={`bubble ${mine ? 'mine' : 'them'}`}>
                      <div className="body">{msg.content}</div>
                      {msg.recalled?.length > 0 && (
                        <div className="recalled">
                          {msg.recalled
                            .map((r) => r.text)
                            .filter(Boolean)
                            .join(' · ')}
                        </div>
                      )}
                    </div>
                    {mine && (
                      <Avatar
                        primary={USER_AVATAR}
                        fallback={USER_AVATAR_FALLBACK}
                        label={t.you}
                        align="right"
                      />
                    )}
                  </div>
                )
              })}
              {busy && <TypingDots label={t.typing} />}
              <div ref={bottomRef} />
            </div>

            <div className="composer">
              <button
                type="button"
                className="night-btn"
                onClick={() => send(t.goodNightMsg)}
                disabled={!ready || busy || ended}
                title={t.goodNight}
              >
                {t.goodNight}
              </button>
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={onKeyDown}
                placeholder={ended ? t.placeholderEnded : t.placeholder}
                disabled={!ready || busy || ended}
              />
              <button
                type="button"
                className="send-btn"
                onClick={() => send()}
                disabled={!ready || busy || ended || !input.trim()}
              >
                {t.send}
              </button>
            </div>
          </main>

          {memoryOpen && (
            <aside className="memory">
              <div className="memory-head">
                <h2>{t.memoryTitle}</h2>
                <div className="memory-actions">
                  <button
                    type="button"
                    className="header-btn small"
                    onClick={refreshMemory}
                  >
                    {t.refresh}
                  </button>
                  <button
                    type="button"
                    className="header-btn small"
                    onClick={() => setMemoryOpen(false)}
                  >
                    {t.collapse}
                  </button>
                </div>
              </div>
              {facts.length === 0 ? (
                <p className="empty">{t.noMemory}</p>
              ) : (
                <ul>
                  {facts.map((f) => (
                    <li key={f.id}>
                      <span>{f.text}</span>
                      {typeof f.certainty === 'number' && (
                        <em>{f.certainty.toFixed(2)}</em>
                      )}
                    </li>
                  ))}
                </ul>
              )}
            </aside>
          )}
        </div>
      </div>
    </div>
  )
}
