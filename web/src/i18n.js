export const LANG_KEY = 'mia_ui_lang'

/** Obviously fake cute excuses while the model loads — pick one at random. */
export const LOADING_EXCUSES = {
  zh: [
    '米娅在和云朵讨价还价带宽…',
    '她的发夹卡住了显卡风扇，正在解救…',
    '正在给神经元泡枸杞茶，马上好…',
    '小猫踩到电源键了，正在哄它下来…',
    '米娅去平行宇宙借了一点算力…',
    '正在给回复涂腮红，别催…',
    '月亮信号有点弱，她在挥旗子…',
    '记忆盒子打不开，她在找钥匙…（钥匙在冰箱里）',
    '正在和懒癌开会，预计五分钟结束…',
    '她说再睡一分钟——已经是第三十一分钟…',
  ],
  en: [
    'Mia spilled glitter on the motherboard… sweeping carefully…',
    'Warming up her “I’m listening” face in the mirror…',
    'The Wi‑Fi goblin asked for snacks first…',
    'Recharging via dramatic sigh… 47%…',
    'She lost a sock in the server rack…',
    'Braiding ethernet cables for good luck…',
    'Convincing the GPU it is a fashion accessory…',
    'Writing you a reply in invisible ink… drying…',
    'Her brain tabs are all open — closing ads…',
    'Almost ready! (This is a lie, but a cute one.)…',
  ],
}

export function randomLoadingExcuse(lang) {
  const list = LOADING_EXCUSES[lang] || LOADING_EXCUSES.zh
  return list[Math.floor(Math.random() * list.length)]
}

export const STRINGS = {
  zh: {
    name: '米娅',
    you: '你',
    online: (n) => `在线 · ${n} 条记忆`,
    loadingModels: '正在加载模型…',
    loadingWait: '请稍等一两分钟…',
    loadingTitle: '米娅正在“非常努力”地赶来跟你聊天…',
    loadingJustThat: '只不过呢……',
    pickTitle: '选一个模型',
    pickHint: '选好后才会开始加载，然后进入聊天。',
    apiStarting: '服务还在启动，按钮马上就能点…',
    pick7: 'Qwen 2.5 7B',
    pick7note: '这台电脑上更稳',
    pick14: 'Qwen 2.5 14B',
    pick14note: 'GGUF；多数层在显卡，其余在内存（较慢）',
    pickRetry: '换一个再试',
    loadFailed: '加载失败',
    apiOffline: 'API 未连接',
    hideMemory: '收起记忆',
    showMemory: '长期记忆',
    newSession: '新会话',
    sessionBreak: '—— 新会话（模型忘掉刚才的对话）——',
    emptyChat: '打个招呼吧 · 结束时说「晚安」',
    goodNight: '晚安',
    goodNightMsg: '晚安',
    placeholder: '输入消息…',
    placeholderEnded: '会话已结束，点「新会话」',
    send: '发送',
    memoryTitle: '长期记忆',
    memoryNew: '本会话新记',
    memoryOld: '已沉淀',
    refresh: '刷新',
    collapse: '收起',
    noMemory: '暂无记忆',
    settings: '设置',
    language: '界面语言',
    typing: '米娅正在输入',
  },
  en: {
    name: 'Mia',
    you: 'You',
    online: (n) => `Online · ${n} memories`,
    loadingModels: 'Loading models…',
    loadingWait: 'Please wait a minute or two…',
    loadingTitle: 'Mia is trying really hard to come talk to you…',
    loadingJustThat: 'It’s just that…',
    pickTitle: 'Choose a model',
    pickHint: 'Loading starts after you pick. Then chat opens.',
    apiStarting: 'Server is still starting — buttons unlock in a moment…',
    pick7: 'Qwen 2.5 7B',
    pick7note: 'Easier fit on this laptop',
    pick14: 'Qwen 2.5 14B',
    pick14note: 'GGUF; most layers on GPU, rest in RAM (slower)',
    pickRetry: 'Pick again',
    loadFailed: 'Failed to load',
    apiOffline: 'API offline',
    hideMemory: 'Hide memory',
    showMemory: 'Memory',
    newSession: 'New session',
    sessionBreak: '—— New session (model forgot earlier turns) ——',
    emptyChat: 'Say hi — type good night when you leave',
    goodNight: 'Night',
    goodNightMsg: 'good night',
    placeholder: 'Message…',
    placeholderEnded: 'Session ended — New session',
    send: 'Send',
    memoryTitle: 'Long-term memory',
    memoryNew: 'New this session',
    memoryOld: 'Settled',
    refresh: 'Refresh',
    collapse: 'Hide',
    noMemory: 'No memories yet',
    settings: 'Settings',
    language: 'Language',
    typing: 'Mia is typing',
  },
}

export function loadLang() {
  try {
    const v = localStorage.getItem(LANG_KEY)
    if (v === 'en' || v === 'zh') return v
  } catch {
    /* ignore */
  }
  return 'zh'
}

export function saveLang(lang) {
  try {
    localStorage.setItem(LANG_KEY, lang)
  } catch {
    /* ignore */
  }
}
