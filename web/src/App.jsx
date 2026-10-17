import { useCallback, useEffect, useRef, useState } from 'react'
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

function TypingDots() {
  return (
    <div className="row them">
      <Avatar
        primary={MIA_AVATAR}
        fallback={MIA_AVATAR_FALLBACK}
        label="Mia"
        align="left"
      />
      <div className="bubble them typing" aria-label="Mia is typing">
        <span className="dot" />
        <span className="dot" />
        <span className="dot" />
      </div>
    </div>
  )
}

export default function App() {
  const [ready, setReady] = useState(false)
  const [status, setStatus] = useState('Connecting…')
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [facts, setFacts] = useState([])
  const [ended, setEnded] = useState(false)
  const [memoryOpen, setMemoryOpen] = useState(true)
  const bottomRef = useRef(null)
  const inputRef = useRef(null)

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
      setReady(!!data.ready)
      setStatus(
        data.ready
          ? `在线 · ${data.facts_count ?? 0} 条记忆`
          : '正在加载模型…',
      )
      if (data.ready) refreshMemory()
      if (data.ended) setEnded(true)
    } catch {
      setReady(false)
      setStatus('API 未连接')
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
              label="Mia"
              align="left"
            />
            <div>
              <h1>米娅</h1>
              <p className="status">{status}</p>
            </div>
          </div>
          <div className="header-actions">
            <button
              type="button"
              className="header-btn"
              onClick={() => setMemoryOpen((v) => !v)}
              aria-expanded={memoryOpen}
            >
              {memoryOpen ? '收起记忆' : '长期记忆'}
            </button>
            <button
              type="button"
              className="header-btn"
              onClick={resetSession}
              disabled={busy}
            >
              新会话
            </button>
          </div>
        </header>

        <div className={`layout ${memoryOpen ? '' : 'memory-collapsed'}`}>
          <main className="chat">
            <div className="thread">
              {messages.length === 0 && !busy && (
                <p className="empty">打个招呼吧 · 结束时说「晚安」</p>
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
                        label="Mia"
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
                        label="You"
                        align="right"
                      />
                    )}
                  </div>
                )
              })}
              {busy && <TypingDots />}
              <div ref={bottomRef} />
            </div>

            <div className="composer">
              <button
                type="button"
                className="night-btn"
                onClick={() => send('晚安')}
                disabled={!ready || busy || ended}
                title="晚安"
              >
                晚安
              </button>
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={onKeyDown}
                placeholder={
                  ended ? '会话已结束，点「新会话」' : '输入消息…'
                }
                disabled={!ready || busy || ended}
              />
              <button
                type="button"
                className="send-btn"
                onClick={() => send()}
                disabled={!ready || busy || ended || !input.trim()}
              >
                发送
              </button>
            </div>
          </main>

          {memoryOpen && (
            <aside className="memory">
              <div className="memory-head">
                <h2>长期记忆</h2>
                <div className="memory-actions">
                  <button
                    type="button"
                    className="header-btn small"
                    onClick={refreshMemory}
                  >
                    刷新
                  </button>
                  <button
                    type="button"
                    className="header-btn small"
                    onClick={() => setMemoryOpen(false)}
                  >
                    收起
                  </button>
                </div>
              </div>
              {facts.length === 0 ? (
                <p className="empty">暂无记忆</p>
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
