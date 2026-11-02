import { useCallback, useEffect, useRef, useState } from 'react'
import './StoryApp.css'

const API = '/api'

function lineClass(ev) {
  if (ev.type === 'narrator' || ev.kind === 'narrator') return 'line narrator'
  if (ev.type === 'player' || ev.speaker === 'Player') return 'line player'
  if (ev.type === 'wait_player') return 'line system'
  if (ev.type === 'cue') return 'line cue'
  if (ev.type === 'scene_start' || ev.type === 'story_complete') return 'line system'
  return 'line npc'
}

function formatEvent(ev) {
  if (ev.type === 'cue') {
    return `导演：下一位 ${ev.speaker} → ${(ev.to || []).join('、') || '—'}（${ev.reason || ''}）`
  }
  if (ev.type === 'wait_player') {
    return `轮到你了。建议对象：${(ev.suggested_to || []).join('、') || '任选角色'}`
  }
  if (ev.type === 'scene_start') {
    return `—— 第 ${ev.scene_index}/${ev.max_scenes} 幕 ——`
  }
  if (ev.type === 'story_complete') {
    return ev.text || '故事结束'
  }
  const speaker = ev.speaker || '?'
  const to = ev.to || []
  const text = ev.text || ''
  if (speaker === 'Narrator' || ev.type === 'narrator') {
    return `旁白：${text}`
  }
  if (speaker === 'Player' || ev.type === 'player') {
    return `你 → ${to.join('、') || '?'}：${text}`
  }
  return `${speaker} → ${to.join('、') || '?'}：${text}`
}

export default function StoryApp() {
  const [statusKind, setStatusKind] = useState('offline')
  const [loadError, setLoadError] = useState('')
  const [ready, setReady] = useState(false)
  const [busy, setBusy] = useState(false)
  const [theme, setTheme] = useState('')
  const [story, setStory] = useState(null)
  const [feed, setFeed] = useState([])
  const [input, setInput] = useState('')
  const [selectedTo, setSelectedTo] = useState([])
  const bottomRef = useRef(null)

  const pushEvents = useCallback((events) => {
    if (!events?.length) return
    setFeed((prev) => [
      ...prev,
      ...events.map((ev, i) => ({
        ...ev,
        _id: `${Date.now()}-${i}-${Math.random()}`,
      })),
    ])
  }, [])

  const pollHealth = useCallback(async () => {
    try {
      const res = await fetch(`${API}/health`)
      const data = await res.json()
      setReady(!!data.ready)
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
      if (data.story) setStory(data.story)
    } catch {
      setReady(false)
      setStatusKind('offline')
    }
  }, [])

  useEffect(() => {
    pollHealth()
    const id = setInterval(pollHealth, ready ? 4000 : 1000)
    return () => clearInterval(id)
  }, [pollHealth, ready])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [feed, busy])

  useEffect(() => {
    if (!story?.suggested_to?.length) return
    if (story.waiting_for_player) {
      setSelectedTo(story.suggested_to.filter((n) => (story.cast_names || []).includes(n)))
    }
  }, [story?.waiting_for_player, story?.suggested_to, story?.cast_names])

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
      /* ignore */
    }
    setStatusKind('choose')
    setLoadError('')
  }

  async function generateStory() {
    if (!ready || busy) return
    setBusy(true)
    setFeed([])
    try {
      const res = await fetch(`${API}/story/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ theme: theme || null }),
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) throw new Error(data.detail || `Error ${res.status}`)
      setStory(data.story)
      setSelectedTo((data.story?.cast_names || []).slice(0, 1))
      setFeed([
        {
          _id: 'setup',
          type: 'system',
          text: '故事已生成。点「开始第一幕」进入旁白与调度。',
        },
      ])
    } catch (e) {
      setFeed((f) => [
        ...f,
        { _id: String(Date.now()), type: 'system', text: String(e.message || e) },
      ])
    } finally {
      setBusy(false)
      pollHealth()
    }
  }

  async function startIntro() {
    if (!ready || busy || !story) return
    setBusy(true)
    try {
      const res = await fetch(`${API}/story/intro`, { method: 'POST' })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) throw new Error(data.detail || `Error ${res.status}`)
      setStory(data.story)
      pushEvents(data.events || [])
    } catch (e) {
      pushEvents([{ type: 'system', text: String(e.message || e) }])
    } finally {
      setBusy(false)
      pollHealth()
    }
  }

  async function advance() {
    if (!ready || busy || !story || story.story_complete) return
    if (story.waiting_for_player) return
    setBusy(true)
    try {
      const res = await fetch(`${API}/story/advance`, { method: 'POST' })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) throw new Error(data.detail || `Error ${res.status}`)
      setStory(data.story)
      pushEvents(data.events || [])
    } catch (e) {
      pushEvents([{ type: 'system', text: String(e.message || e) }])
    } finally {
      setBusy(false)
      pollHealth()
    }
  }

  function toggleTo(name) {
    setSelectedTo((prev) =>
      prev.includes(name) ? prev.filter((n) => n !== name) : [...prev, name],
    )
  }

  async function sendPlayer() {
    const message = input.trim()
    if (!message || busy || !ready || !story || story.story_complete) return
    if (selectedTo.length === 0) {
      pushEvents([{ type: 'system', text: '请至少勾选一个对话对象。' }])
      return
    }
    setInput('')
    setBusy(true)
    try {
      const res = await fetch(`${API}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, to: selectedTo }),
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) throw new Error(data.detail || `Error ${res.status}`)
      setStory(data.story)
      pushEvents(data.events || [])
    } catch (e) {
      pushEvents([{ type: 'system', text: String(e.message || e) }])
    } finally {
      setBusy(false)
      pollHealth()
    }
  }

  const cast = story?.cast_names || []
  const sceneLabel = story
    ? `第 ${story.scene_index}/${story.max_scenes} 幕 · 本幕 ${story.message_count_in_scene}/${story.scene_message_cap}`
    : ''

  return (
    <div className="story-app">
      <header className="story-header">
        <div>
          <h1>AI Text Game</h1>
          <p className="meta">
            {statusKind === 'offline' && 'API offline'}
            {statusKind === 'choose' && 'Choose a model'}
            {statusKind === 'loading' && 'Loading model…'}
            {statusKind === 'error' && (loadError || 'Load failed')}
            {statusKind === 'ready' && (sceneLabel || 'Model ready')}
            {story?.waiting_for_player && ' · 旁白点到你了'}
            {story?.story_complete && ' · 已完结'}
          </p>
        </div>
      </header>

      {(statusKind === 'choose' || statusKind === 'error' || statusKind === 'offline') && (
        <section className="panel">
          <h2>1. Load model</h2>
          <div className="row-btns">
            <button type="button" disabled={statusKind === 'offline'} onClick={() => pickModel('7b')}>
              7B
            </button>
            <button type="button" disabled={statusKind === 'offline'} onClick={() => pickModel('14b')}>
              14B
            </button>
            {statusKind === 'error' && (
              <button type="button" onClick={retryPick}>
                Retry
              </button>
            )}
          </div>
        </section>
      )}

      {statusKind === 'loading' && (
        <section className="panel">
          <p>Loading weights… keep this tab open.</p>
        </section>
      )}

      {ready && (
        <>
          <section className="panel">
            <h2>2. Generate story</h2>
            <input
              className="theme"
              value={theme}
              onChange={(e) => setTheme(e.target.value)}
              placeholder="Optional theme (e.g. 宫廷权谋 / campus)"
              disabled={busy}
            />
            <div className="row-btns">
              <button type="button" disabled={busy} onClick={generateStory}>
                Generate story
              </button>
              <button
                type="button"
                disabled={busy || !story || story.story_complete}
                onClick={startIntro}
              >
                Start scene 1
              </button>
              <button
                type="button"
                disabled={
                  busy || !story || story.story_complete || story.waiting_for_player
                }
                onClick={advance}
              >
                Advance (narrator cue)
              </button>
            </div>
          </section>

          {story && (
            <section className="panel setup">
              <h2>Setup</h2>
              <p>
                <strong>Player:</strong> {story.player?.name} — {story.player?.identity}
              </p>
              <p>
                <strong>Goal:</strong> {story.player?.goal}
              </p>
              <details>
                <summary>Background</summary>
                <p className="bg">{story.background}</p>
              </details>
              <ul className="cast">
                {(story.characters || []).map((c) => (
                  <li key={c.name}>
                    <strong>{c.name}</strong> ({c.slot}) — {c.description}
                  </li>
                ))}
              </ul>
            </section>
          )}

          <section className="panel feed-panel">
            <h2>Scene feed</h2>
            <div className="feed">
              {feed.length === 0 && <p className="empty">No lines yet.</p>}
              {feed.map((ev) => (
                <div key={ev._id} className={lineClass(ev)}>
                  {ev.text && ev.type === 'system' ? ev.text : formatEvent(ev)}
                </div>
              ))}
              <div ref={bottomRef} />
            </div>
          </section>

          {story && !story.story_complete && (
            <section className="panel composer">
              <h2>3. Your line (interrupt anytime)</h2>
              <div className="checks">
                {cast.map((name) => (
                  <label key={name} className="check">
                    <input
                      type="checkbox"
                      checked={selectedTo.includes(name)}
                      onChange={() => toggleTo(name)}
                    />
                    {name}
                  </label>
                ))}
              </div>
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                rows={3}
                placeholder="What do you say?"
                disabled={busy}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    sendPlayer()
                  }
                }}
              />
              <button
                type="button"
                disabled={busy || !input.trim() || selectedTo.length === 0}
                onClick={sendPlayer}
              >
                Send
              </button>
            </section>
          )}
        </>
      )}
    </div>
  )
}
