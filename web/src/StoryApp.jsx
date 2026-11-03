import { useCallback, useEffect, useRef, useState } from 'react'
import './StoryApp.css'

const API = '/api'

const UI = {
  zh: {
    title: 'AI Text Game',
    offline: 'API 离线',
    choose: '选择模型',
    loading: '加载模型中…',
    error: '加载失败',
    ready: '模型已就绪',
    cuedYou: ' · 轮到你了',
    complete: ' · 已完结',
    loadModel: '1. 加载模型',
    retry: '重试',
    loadingWeights: '正在加载权重…请保持此页打开。',
    generate: '2. 生成故事',
    themePh: '主题（写入故事设定），例如：雨夜旧书店的重逢',
    themeHint:
      '主题会进入故事与人设生成。建议约 8–80 字（一短语或一两句）。留空=自由发挥。',
    genBtn: '生成故事',
    startScene: '开始第一幕',
    advance: '推进（导演调度）',
    setup: '设定',
    player: '玩家',
    goal: '目标',
    theme: '主题',
    present: '本幕出场',
    background: '背景',
    persona: '人设',
    feed: '场景记录',
    empty: '尚无对白。',
    yourCued: '3. 你的台词（被点到 — 对方正常接话）',
    yourInterrupt: '3. 你的台词（打断 — 对方按被打断反应）',
    phCued: '导演点到你了 — 说出你的台词…',
    phInterrupt: '可随时打断 — NPC 会按打断来反应…',
    send: '发送',
    needTo: '请至少勾选一个对话对象。',
    storyReady: '故事已生成。点「开始第一幕」进入旁白与调度。',
    scene: (s, max, n, lo, hi) => `第 ${s}/${max} 幕 · 对白 ${n}/${lo}~${hi}`,
    director: (speaker, raw, to, reason) =>
      `导演：下一位 ${speaker}${raw} → ${to}（${reason}）`,
    waitPlayer: (sug) => `轮到你了（被点到）。建议对象：${sug}`,
    sceneStart: (s, max, present) =>
      `—— 第 ${s}/${max} 幕 ——${present ? ` 出场：${present}` : ''}`,
    storyEnd: '故事结束',
    narrator: (t) => `旁白：${t}`,
    you: (tag, to, text) => `你${tag ? `（${tag}）` : ''} → ${to}：${text}`,
    interrupt: '打断',
    cued: '被点到',
    reactInterrupt: ' · 回应打断',
    reactCued: ' · 接话',
    promptTitle: 'Prompt 调试',
    promptClose: '关闭',
    promptNone: '这条没有模型 prompt（玩家输入或系统消息）。',
    promptHint: '测试：点击旁白/角色台词查看 prompt',
    promptSystem: 'System',
    promptUser: 'User',
    promptPersona: '人设',
    promptMemory: '记忆',
    promptBackground: '背景',
    promptNarration: '当前旁白',
    promptScene: '场景记录',
    promptKind: '类型',
  },
  en: {
    title: 'AI Text Game',
    offline: 'API offline',
    choose: 'Choose a model',
    loading: 'Loading model…',
    error: 'Load failed',
    ready: 'Model ready',
    cuedYou: ' · your turn',
    complete: ' · complete',
    loadModel: '1. Load model',
    retry: 'Retry',
    loadingWeights: 'Loading weights… keep this tab open.',
    generate: '2. Generate story',
    themePh: 'Theme (fed into setup), e.g. reunion in a rainy old bookstore',
    themeHint:
      'Theme goes into story + persona generation. Best ~8–80 characters (a short phrase or 1–2 sentences). Empty = freeform.',
    genBtn: 'Generate story',
    startScene: 'Start scene 1',
    advance: 'Advance (director cue)',
    setup: 'Setup',
    player: 'Player',
    goal: 'Goal',
    theme: 'Theme',
    present: 'Present this scene',
    background: 'Background',
    persona: 'Persona',
    feed: 'Scene feed',
    empty: 'No lines yet.',
    yourCued: '3. Your line (cued — they reply normally)',
    yourInterrupt: '3. Your line (interrupt — they react to being cut off)',
    phCued: 'Director pointed at you — say your line…',
    phInterrupt: 'Interrupt anytime — NPCs react as interrupted…',
    send: 'Send',
    needTo: 'Select at least one character to speak to.',
    storyReady: 'Story generated. Click “Start scene 1” for narration and cues.',
    scene: (s, max, n, lo, hi) =>
      `Scene ${s}/${max} · dialogue ${n}/${lo}~${hi}`,
    director: (speaker, raw, to, reason) =>
      `Director: next ${speaker}${raw} → ${to} (${reason})`,
    waitPlayer: (sug) => `Your turn (cued). Suggested: ${sug}`,
    sceneStart: (s, max, present) =>
      `—— Scene ${s}/${max} ——${present ? ` Present: ${present}` : ''}`,
    storyEnd: 'Story complete',
    narrator: (t) => `Narrator: ${t}`,
    you: (tag, to, text) => `You${tag ? ` (${tag})` : ''} → ${to}: ${text}`,
    interrupt: 'interrupt',
    cued: 'cued',
    reactInterrupt: ' · reacting to interrupt',
    reactCued: ' · reply',
    promptTitle: 'Prompt debug',
    promptClose: 'Close',
    promptNone: 'No model prompt for this line (player input or system).',
    promptHint: 'Test: click narrator/NPC lines to inspect prompts',
    promptSystem: 'System',
    promptUser: 'User',
    promptPersona: 'Persona',
    promptMemory: 'Memory',
    promptBackground: 'Background',
    promptNarration: 'Narration',
    promptScene: 'Scene log',
    promptKind: 'Kind',
  },
}

function PromptInspect({ prompt, t, onClose }) {
  if (!prompt) return null
  const sections = [
    [t.promptKind, prompt.kind || '—'],
    [t.promptPersona, prompt.persona],
    [t.promptMemory, prompt.memory],
    [t.promptBackground, prompt.background],
    [t.promptNarration, prompt.narration],
    [t.promptScene, prompt.scene_log],
    [t.promptSystem, prompt.system],
    [t.promptUser, prompt.user],
  ]
  return (
    <aside className="prompt-panel" aria-label={t.promptTitle}>
      <div className="prompt-panel-head">
        <h2>
          {t.promptTitle}
          {prompt.id != null ? ` #${prompt.id}` : ''}
        </h2>
        <button type="button" onClick={onClose}>
          {t.promptClose}
        </button>
      </div>
      <div className="prompt-panel-body">
        {sections.map(([label, value]) =>
          value ? (
            <section key={label} className="prompt-block">
              <h3>{label}</h3>
              <pre>{typeof value === 'string' ? value : JSON.stringify(value, null, 2)}</pre>
            </section>
          ) : null,
        )}
      </div>
    </aside>
  )
}

function lineClass(ev) {
  if (ev.type === 'narrator' || ev.kind === 'narrator') return 'line narrator'
  if (ev.type === 'player' || ev.speaker === 'Player') return 'line player'
  if (ev.type === 'wait_player' || ev.type === 'system') return 'line system'
  if (ev.type === 'cue') return 'line cue'
  if (ev.type === 'scene_start' || ev.type === 'story_complete') return 'line system'
  return 'line npc'
}

const LIVE_POLL_MS = 250

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function parseSseChunk(chunk, onMessage) {
  const line = chunk
    .split('\n')
    .map((l) => l.trim())
    .find((l) => l.startsWith('data:'))
  if (!line) return
  const payload = line.slice(5).trim()
  if (!payload) return
  onMessage(JSON.parse(payload))
}

async function readSseBody(res, onMessage) {
  if (!res.body) throw new Error('No response body')
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    const chunks = buf.split('\n\n')
    buf = chunks.pop() || ''
    for (const chunk of chunks) parseSseChunk(chunk, onMessage)
  }
  buf += decoder.decode()
  if (buf.trim().startsWith('data:')) parseSseChunk(buf, onMessage)
}

function formatEvent(ev, t, join) {
  if (ev.type === 'cue') {
    const raw =
      ev.raw_speaker && ev.raw_speaker !== ev.speaker ? ` [raw:${ev.raw_speaker}]` : ''
    return t.director(
      ev.speaker,
      raw,
      (ev.to || []).join(join) || '—',
      ev.reason || '',
    )
  }
  if (ev.type === 'wait_player') {
    return t.waitPlayer((ev.suggested_to || []).join(join) || '—')
  }
  if (ev.type === 'scene_start') {
    return t.sceneStart(
      ev.scene_index,
      ev.max_scenes,
      (ev.present || []).join(join),
    )
  }
  if (ev.type === 'story_complete') {
    return ev.text || t.storyEnd
  }
  const speaker = ev.speaker || '?'
  const to = (ev.to || []).join(join) || '?'
  const text = ev.text || ''
  if (speaker === 'Narrator' || ev.type === 'narrator') {
    return t.narrator(text)
  }
  if (speaker === 'Player' || ev.type === 'player') {
    const tag =
      ev.player_mode === 'interrupt'
        ? t.interrupt
        : ev.player_mode === 'cued'
          ? t.cued
          : ''
    return t.you(tag, to, text)
  }
  const reactTag =
    ev.player_mode === 'interrupt'
      ? t.reactInterrupt
      : ev.player_mode === 'cued'
        ? t.reactCued
        : ''
  return `${speaker} → ${to}${reactTag}: ${text}`
}

export default function StoryApp() {
  const [uiLang, setUiLang] = useState('zh')
  const [statusKind, setStatusKind] = useState('offline')
  const [loadError, setLoadError] = useState('')
  const [ready, setReady] = useState(false)
  const [busy, setBusy] = useState(false)
  const [theme, setTheme] = useState('')
  const [story, setStory] = useState(null)
  const [feed, setFeed] = useState([])
  const [input, setInput] = useState('')
  const [selectedTo, setSelectedTo] = useState([])
  const [inspectPrompt, setInspectPrompt] = useState(null)
  const [apiLiveOk, setApiLiveOk] = useState(null)
  const bottomRef = useRef(null)
  const autoTimerRef = useRef(null)
  const advanceRef = useRef(null)

  const storyLang = story?.language || uiLang
  const t = UI[storyLang] || UI.zh
  const join = storyLang === 'en' ? ', ' : '、'

  const clearAutoContinue = useCallback(() => {
    if (autoTimerRef.current != null) {
      clearTimeout(autoTimerRef.current)
      autoTimerRef.current = null
    }
  }, [])

  const scheduleAutoContinue = useCallback(
    (seconds) => {
      clearAutoContinue()
      const ms = Math.max(0.5, Number(seconds) || 0.75) * 1000
      autoTimerRef.current = setTimeout(() => {
        autoTimerRef.current = null
        advanceRef.current?.()
      }, ms)
    },
    [clearAutoContinue],
  )

  useEffect(() => () => clearAutoContinue(), [clearAutoContinue])

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

  const handleStreamMessage = useCallback((msg) => {
    const kind = msg?.type
    if (kind === 'error') {
      throw new Error(msg.message || 'Stream error')
    }
    if (kind === 'event' && msg.event) {
      const ev = {
        ...msg.event,
        _id: `${Date.now()}-${Math.random()}`,
      }
      setFeed((prev) => [...prev, ev])
      return
    }
    if (kind === 'stream_start' && msg.event) {
      const ev = {
        ...msg.event,
        text: msg.event.text || '',
        _streaming: true,
        _id: `${Date.now()}-${Math.random()}`,
      }
      setFeed((prev) => [...prev, ev])
      return
    }
    if (kind === 'stream_delta' && msg.delta) {
      setFeed((prev) => {
        if (!prev.length) return prev
        const next = [...prev]
        const last = { ...next[next.length - 1] }
        if (!last._streaming) return prev
        last.text = (last.text || '') + msg.delta
        next[next.length - 1] = last
        return next
      })
      return
    }
    if (kind === 'stream_end' && msg.event) {
      setFeed((prev) => {
        if (!prev.length) return prev
        const next = [...prev]
        const last = next[next.length - 1]
        next[next.length - 1] = {
          ...msg.event,
          _id: last?._id || `${Date.now()}-${Math.random()}`,
          _streaming: false,
        }
        return next
      })
    }
  }, [])

  async function applyJobMessages(messages) {
    let finish = null
    for (const msg of messages || []) {
      if (msg.type === 'error') {
        throw new Error(msg.message || 'Story job failed')
      }
      if (msg.type === 'done') {
        finish = msg
        continue
      }
      handleStreamMessage(msg)
    }
    return finish
  }

  async function runLiveJob(startRes) {
    if (!startRes.ok) {
      const data = await startRes.json().catch(() => ({}))
      const detail = data.detail || data.message || `Error ${startRes.status}`
      throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
    }

    const ctype = (startRes.headers.get('content-type') || '').toLowerCase()
    let finish = null

    // New API: POST returns JSON {started:true}; poll /api/story/live every 0.25s.
    if (ctype.includes('application/json') || ctype.includes('text/json')) {
      const started = await startRes.json().catch(() => ({}))
      if (started?.started === false) {
        throw new Error(started.message || 'Story job did not start')
      }
      let since = 0
      // First poll immediately, then every 0.25s.
      while (true) {
        const res = await fetch(`${API}/story/live?since=${since}`)
        const live = await res.json().catch(() => ({}))
        if (!res.ok) {
          if (res.status === 404) {
            throw new Error(
              'API is outdated (no /api/story/live). Close the API window and run run_web.bat again.',
            )
          }
          const detail = live.detail || live.message || `Live poll error ${res.status}`
          throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
        }
        since = live.next_since ?? since
        const part = await applyJobMessages(live.messages || [])
        if (part) finish = part
        if (finish) break
        if (!live.job_busy) break
        await sleep(LIVE_POLL_MS)
      }
    } else if (ctype.includes('text/event-stream')) {
      // Old API still streaming on the POST — read that body so text appears.
      await readSseBody(startRes, async (msg) => {
        if (msg?.type === 'done') {
          finish = msg
          return
        }
        if (msg?.type === 'error') {
          throw new Error(msg.message || 'Story job failed')
        }
        handleStreamMessage(msg)
      })
    } else {
      // Unknown body — try JSON, else treat as error.
      const data = await startRes.json().catch(() => null)
      if (data?.started) {
        throw new Error(
          'API started a job but returned an unexpected content-type. Restart the API.',
        )
      }
      throw new Error(
        `Unexpected response type (${ctype || 'none'}). Restart API + refresh http://127.0.0.1:5173/`,
      )
    }

    if (finish?.story) setStory(finish.story)
    if (finish?.auto_continue) scheduleAutoContinue(finish.beat_seconds)
    else clearAutoContinue()
    return finish
  }

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
      setApiLiveOk(data.live_feed === true)
    } catch {
      setReady(false)
      setStatusKind('offline')
      setApiLiveOk(null)
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
      setSelectedTo(
        story.suggested_to.filter((n) => (story.cast_names || []).includes(n)),
      )
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
    clearAutoContinue()
    setInspectPrompt(null)
    setBusy(true)
    setFeed([])
    try {
      const res = await fetch(`${API}/story/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ theme: theme || null, language: uiLang }),
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) throw new Error(data.detail || `Error ${res.status}`)
      setStory(data.story)
      setSelectedTo((data.story?.cast_names || []).slice(0, 1))
      const copy = UI[data.story?.language || uiLang] || UI.zh
      setFeed([
        {
          _id: 'setup',
          type: 'system',
          text: copy.storyReady,
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
    clearAutoContinue()
    setBusy(true)
    try {
      const res = await fetch(`${API}/story/intro`, { method: 'POST' })
      await runLiveJob(res)
    } catch (e) {
      clearAutoContinue()
      pushEvents([{ type: 'system', text: String(e.message || e) }])
    } finally {
      setBusy(false)
      pollHealth()
    }
  }

  async function advance() {
    if (!ready || busy || !story || story.story_complete) return
    if (story.waiting_for_player) return
    clearAutoContinue()
    setBusy(true)
    try {
      const res = await fetch(`${API}/story/advance`, { method: 'POST' })
      await runLiveJob(res)
    } catch (e) {
      clearAutoContinue()
      pushEvents([{ type: 'system', text: String(e.message || e) }])
    } finally {
      setBusy(false)
      pollHealth()
    }
  }
  advanceRef.current = advance

  function toggleTo(name) {
    setSelectedTo((prev) =>
      prev.includes(name) ? prev.filter((n) => n !== name) : [...prev, name],
    )
  }

  async function sendPlayer() {
    const message = input.trim()
    if (!message || busy || !ready || !story || story.story_complete) return
    if (selectedTo.length === 0) {
      pushEvents([{ type: 'system', text: t.needTo }])
      return
    }
    // Interrupt window: cancel the pending auto-cue and speak now.
    clearAutoContinue()
    setInput('')
    setBusy(true)
    try {
      const res = await fetch(`${API}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, to: selectedTo }),
      })
      await runLiveJob(res)
    } catch (e) {
      clearAutoContinue()
      pushEvents([{ type: 'system', text: String(e.message || e) }])
    } finally {
      setBusy(false)
      pollHealth()
    }
  }

  const cast = story?.cast_names || []
  const sceneLabel = story
    ? t.scene(
        story.scene_index,
        story.max_scenes,
        story.dialogue_count_in_scene ?? 0,
        story.scene_dialogue_min ?? 8,
        story.scene_dialogue_max ?? 15,
      )
    : ''

  return (
    <div className={`story-app${inspectPrompt ? ' with-prompt' : ''}`}>
      <PromptInspect
        prompt={inspectPrompt}
        t={t}
        onClose={() => setInspectPrompt(null)}
      />
      <div className="story-main">
      <header className="story-header">
        <div className="header-row">
          <div>
            <h1>{t.title}</h1>
            <p className="meta">
              {statusKind === 'offline' && t.offline}
              {statusKind === 'choose' && t.choose}
              {statusKind === 'loading' && t.loading}
              {statusKind === 'error' && (loadError || t.error)}
              {statusKind === 'ready' && (sceneLabel || t.ready)}
              {story?.waiting_for_player && t.cuedYou}
              {story?.story_complete && t.complete}
              {statusKind === 'ready' && apiLiveOk === false && (
                <span className="api-warn">
                  {' '}
                  · API outdated — close API window & re-run run_web.bat
                </span>
              )}
            </p>
          </div>
          <div className="lang-toggle" role="group" aria-label="Language">
            <button
              type="button"
              className={uiLang === 'zh' ? 'active' : ''}
              disabled={busy}
              onClick={() => setUiLang('zh')}
              title="Story language for the next Generate"
            >
              中文
            </button>
            <button
              type="button"
              className={uiLang === 'en' ? 'active' : ''}
              disabled={busy}
              onClick={() => setUiLang('en')}
              title="Story language for the next Generate"
            >
              EN
            </button>
          </div>
        </div>
      </header>

      {(statusKind === 'choose' || statusKind === 'error' || statusKind === 'offline') && (
        <section className="panel">
          <h2>{t.loadModel}</h2>
          <div className="row-btns">
            <button type="button" disabled={statusKind === 'offline'} onClick={() => pickModel('7b')}>
              7B
            </button>
            <button type="button" disabled={statusKind === 'offline'} onClick={() => pickModel('14b')}>
              14B
            </button>
            {statusKind === 'error' && (
              <button type="button" onClick={retryPick}>
                {t.retry}
              </button>
            )}
          </div>
        </section>
      )}

      {statusKind === 'loading' && (
        <section className="panel">
          <p>{t.loadingWeights}</p>
        </section>
      )}

      {ready && (
        <>
          <section className="panel">
            <h2>{t.generate}</h2>
            <input
              className="theme"
              value={theme}
              onChange={(e) => setTheme(e.target.value)}
              placeholder={t.themePh}
              disabled={busy}
              maxLength={120}
            />
            <p className="theme-hint">{t.themeHint}</p>
            <div className="row-btns">
              <button type="button" disabled={busy} onClick={generateStory}>
                {t.genBtn}
              </button>
              <button
                type="button"
                disabled={busy || !story || story.story_complete}
                onClick={startIntro}
              >
                {t.startScene}
              </button>
              <button
                type="button"
                disabled={
                  busy || !story || story.story_complete || story.waiting_for_player
                }
                onClick={advance}
              >
                {t.advance}
              </button>
            </div>
          </section>

          {story && (
            <section className="panel setup">
              <h2>{t.setup}</h2>
              <p>
                <strong>{t.player}:</strong> {story.player?.name} — {story.player?.identity}
              </p>
              <p>
                <strong>{t.goal}:</strong> {story.player?.goal}
              </p>
              {story.theme ? (
                <p>
                  <strong>{t.theme}:</strong> {story.theme}
                </p>
              ) : null}
              {story.present_characters?.length ? (
                <p>
                  <strong>{t.present}:</strong>{' '}
                  {story.present_characters.join(join)}
                </p>
              ) : null}
              <details>
                <summary>{t.background}</summary>
                <p className="bg">{story.background}</p>
              </details>
              <ul className="cast">
                {(story.characters || []).map((c) => (
                  <li key={c.name}>
                    <strong>{c.name}</strong> ({c.slot}) — {c.description}
                    {c.persona ? (
                      <div className="persona">
                        {t.persona}: {c.persona}
                      </div>
                    ) : null}
                  </li>
                ))}
              </ul>
            </section>
          )}

          <section className="panel feed-panel">
            <h2>{t.feed}</h2>
            <p className="theme-hint">{t.promptHint}</p>
            <div className="feed">
              {feed.length === 0 && <p className="empty">{t.empty}</p>}
              {feed.map((ev) => {
                const hasPrompt = !!ev.debug_prompt
                const selected =
                  hasPrompt && inspectPrompt?.id === ev.debug_prompt.id
                return (
                  <div
                    key={ev._id}
                    role={hasPrompt ? 'button' : undefined}
                    tabIndex={hasPrompt ? 0 : undefined}
                    className={`${lineClass(ev)}${ev._streaming ? ' streaming' : ''}${
                      hasPrompt ? ' clickable' : ''
                    }${selected ? ' selected' : ''}`}
                    onClick={() => {
                      if (!hasPrompt) {
                        setInspectPrompt({
                          kind: ev.type || 'system',
                          system: t.promptNone,
                        })
                        return
                      }
                      setInspectPrompt(ev.debug_prompt)
                    }}
                    onKeyDown={(e) => {
                      if (!hasPrompt) return
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault()
                        setInspectPrompt(ev.debug_prompt)
                      }
                    }}
                  >
                    {ev.text && ev.type === 'system'
                      ? ev.text
                      : formatEvent(ev, t, join)}
                  </div>
                )
              })}
              <div ref={bottomRef} />
            </div>
          </section>

          {story && !story.story_complete && (
            <section className="panel composer">
              <h2>{story.waiting_for_player ? t.yourCued : t.yourInterrupt}</h2>
              <div className="checks">
                {(story.present_characters?.length ? story.present_characters : cast).map(
                  (name) => (
                    <label key={name} className="check">
                      <input
                        type="checkbox"
                        checked={selectedTo.includes(name)}
                        onChange={() => toggleTo(name)}
                      />
                      {name}
                    </label>
                  ),
                )}
              </div>
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                rows={3}
                placeholder={story.waiting_for_player ? t.phCued : t.phInterrupt}
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
                {t.send}
              </button>
            </section>
          )}
        </>
      )}
      </div>
    </div>
  )
}
