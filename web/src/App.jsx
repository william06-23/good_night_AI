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

function formatMsgTime(ts, lang) {
  if (!ts) return ''
  return new Date(ts).toLocaleTimeString(lang === 'zh' ? 'zh-CN' : 'en-US', {
    hour: '2-digit',
    minute: '2-digit',
  })
}

function dayKey(ts) {
  if (!ts) return ''
  const d = new Date(ts)
  return `${d.getFullYear()}-${d.getMonth()}-${d.getDate()}`
}

function formatMsgDate(ts, lang) {
  if (!ts) return ''
  const d = new Date(ts)
  const now = new Date()
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate())
  const that = new Date(d.getFullYear(), d.getMonth(), d.getDate())
  const diffDays = Math.round((today - that) / 86400000)
  if (diffDays === 0) return lang === 'zh' ? '今天' : 'Today'
  if (diffDays === 1) return lang === 'zh' ? '昨天' : 'Yesterday'
  return d.toLocaleDateString(lang === 'zh' ? 'zh-CN' : 'en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    weekday: 'short',
  })
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
  const [statusKind, setStatusKind] = useState('offline') // offline | choose | loading | ready | error
  const [loadError, setLoadError] = useState('')
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
  const historyLoaded = useRef(false)

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

  const loadHistory = useCallback(async () => {
    try {
      const res = await fetch(`${API}/chat/history`)
      if (!res.ok) return
      const data = await res.json()
      const rows = (data.messages || []).map((m) => ({
        role: m.role,
        content: m.content,
        ts: m.ts || undefined,
        kind: m.kind,
      }))
      setMessages(rows)
      historyLoaded.current = true
    } catch {
      /* ignore until API is up */
    }
  }, [])

  const pollHealth = useCallback(async () => {
    try {
      const res = await fetch(`${API}/health`)
      const data = await res.json()
      everReachedApi.current = true
      if (!historyLoaded.current) loadHistory()
      setReady(!!data.ready)
      setFactsCount(data.facts_count ?? 0)
      setLoadError(data.load_error || '')
      const phase = data.phase || (data.ready ? 'ready' : 'choose')
      setStatusKind(
        phase === 'ready' || data.ready
          ? 'ready'
          : phase === 'loading'
            ? 'loading'
            : phase === 'error'
              ? 'error'
              : 'choose',
      )
      refreshMemory()
      if (data.ended) setEnded(true)
      else setEnded(false)
    } catch {
      setReady(false)
      setStatusKind('offline')
    }
  }, [refreshMemory, loadHistory])

  useEffect(() => {
    pollHealth()
    const id = setInterval(pollHealth, ready ? 3000 : 1000)
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

  async function pickModel(key) {
    setStatusKind('loading')
    setLoadError('')
    try {
      const res = await fetch(`${API}/model`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key }),
      })
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        setLoadError(data.detail || `Error ${res.status}`)
        setStatusKind('error')
      }
    } catch (e) {
      setLoadError(String(e.message || e))
      setStatusKind('error')
    }
  }

  async function retryPick() {
    try {
      await fetch(`${API}/model/retry`, { method: 'POST' })
    } catch {
      /* poll will recover */
    }
    setLoadError('')
    setStatusKind('choose')
  }

  function changeLang(next) {
    setLang(next)
    saveLang(next)
    setSettingsOpen(false)
  }

  async function send(text) {
    const message = (text ?? input).trim()
    if (!message || busy || !ready || ended) return

    setInput('')
    const sentAt = Date.now()
    setMessages((m) => [...m, { role: 'user', content: message, ts: sentAt }])
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
          {
            role: 'system',
            content: data.detail || `Error ${res.status}`,
            ts: Date.now(),
          },
        ])
        return
      }
      setMessages((m) => [
        ...m,
        {
          role: 'assistant',
          content: data.reply,
          recalled: data.recalled || [],
          ts: Date.now(),
        },
      ])
      if (data.ended) setEnded(true)
      await refreshMemory()
    } catch (e) {
      setMessages((m) => [
        ...m,
        { role: 'system', content: String(e.message || e), ts: Date.now() },
      ])
    } finally {
      setBusy(false)
      pollHealth()
      inputRef.current?.focus()
    }
  }

  async function resetSession() {
    // Clears model short-term context only; transcript stays on screen and on disk.
    setBusy(true)
    try {
      await fetch(`${API}/session/reset`, { method: 'POST' })
      setEnded(false)
      await loadHistory()
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
                {statusKind === 'choose' && t.pickTitle}
                {statusKind === 'error' && (loadError || t.loadFailed)}
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
            <div className="thread-wrap">
            <div className="thread">
              {ready && messages.length === 0 && !busy && (
                <p className="empty">{t.emptyChat}</p>
              )}
              {messages.map((msg, i) => {
                const timeLabel = formatMsgTime(msg.ts, lang)
                const prevKey = i > 0 ? dayKey(messages[i - 1].ts) : ''
                const thisKey = dayKey(msg.ts)
                const showDate = Boolean(thisKey && thisKey !== prevKey)
                const dateLabel = showDate ? formatMsgDate(msg.ts, lang) : ''
                const dateDivider = showDate ? (
                  <div key={`date-${thisKey}-${i}`} className="date-divider">
                    <time dateTime={new Date(msg.ts).toISOString()}>
                      {dateLabel}
                    </time>
                  </div>
                ) : null

                if (msg.role === 'system') {
                  const isSession = msg.kind === 'session'
                  return (
                    <div key={`wrap-${i}`} className="msg-block">
                      {dateDivider}
                      <div
                        className={
                          isSession ? 'date-divider session-break' : 'system-pill'
                        }
                      >
                        {isSession ? (
                          <time
                            dateTime={
                              msg.ts
                                ? new Date(msg.ts).toISOString()
                                : undefined
                            }
                          >
                            {t.sessionBreak}
                          </time>
                        ) : (
                          <>
                            <span>{msg.content}</span>
                            {timeLabel && (
                              <time
                                className="msg-time"
                                dateTime={new Date(msg.ts).toISOString()}
                              >
                                {timeLabel}
                              </time>
                            )}
                          </>
                        )}
                      </div>
                    </div>
                  )
                }
                const mine = msg.role === 'user'
                return (
                  <div key={`wrap-${i}`} className="msg-block">
                    {dateDivider}
                    <div className={`row ${mine ? 'mine' : 'them'}`}>
                      {!mine && (
                        <Avatar
                          primary={MIA_AVATAR}
                          fallback={MIA_AVATAR_FALLBACK}
                          label={t.name}
                          align="left"
                        />
                      )}
                      <div className={`bubble-wrap ${mine ? 'mine' : 'them'}`}>
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
                        {timeLabel && (
                          <time
                            className="msg-time"
                            dateTime={new Date(msg.ts).toISOString()}
                          >
                            {timeLabel}
                          </time>
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
                  </div>
                )
              })}
              {busy && <TypingDots label={t.typing} />}
              <div ref={bottomRef} />
            </div>
            {(statusKind === 'choose' ||
              statusKind === 'offline' ||
              statusKind === 'error' ||
              statusKind === 'loading') && (
              <div className="chat-overlay">
                {(statusKind === 'choose' || statusKind === 'offline') && (
                  <div className="pick-card">
                    <p className="loading-title">{t.pickTitle}</p>
                    <p className="wait-hint">
                      {statusKind === 'offline' ? t.apiStarting : t.pickHint}
                    </p>
                    <button
                      type="button"
                      className="model-btn"
                      disabled={statusKind === 'offline'}
                      onClick={() => pickModel('7b')}
                    >
                      <strong>{t.pick7}</strong>
                      <span>{t.pick7note}</span>
                    </button>
                    <button
                      type="button"
                      className="model-btn"
                      disabled={statusKind === 'offline'}
                      onClick={() => pickModel('14b')}
                    >
                      <strong>{t.pick14}</strong>
                      <span>{t.pick14note}</span>
                    </button>
                  </div>
                )}
                {statusKind === 'error' && (
                  <div className="pick-card">
                    <p className="loading-title">{t.loadFailed}</p>
                    <p className="excuse">{loadError}</p>
                    <button
                      type="button"
                      className="model-btn"
                      onClick={retryPick}
                    >
                      <strong>{t.pickRetry}</strong>
                    </button>
                  </div>
                )}
                {statusKind === 'loading' && (
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
              </div>
            )}
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
                <>
                  <p className="memory-legend">
                    <span className="swatch new" /> {t.memoryNew}
                    <span className="swatch old" /> {t.memoryOld}
                  </p>
                  <ul>
                    {facts.map((f) => (
                      <li
                        key={f.id}
                        className={f.session_new ? 'fact-new' : 'fact-old'}
                      >
                        <span>{f.text}</span>
                        {typeof f.certainty === 'number' && (
                          <em>{f.certainty.toFixed(2)}</em>
                        )}
                      </li>
                    ))}
                  </ul>
                </>
              )}
            </aside>
          )}
        </div>
      </div>
    </div>
  )
}
