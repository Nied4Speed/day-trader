import { useState, useEffect, useRef, useCallback, useMemo } from 'react'

export interface Model {
  id: number; name: string; strategy_type: string; generation: number;
  rank: number; parent_ids: number[] | null; genetic_operation: string | null;
  initial_capital: number; current_capital: number;
  capital_deployed?: number; deployment_pct?: number;
  performance: {
    equity: number; total_pnl: number; return_pct: number;
    sharpe_ratio: number; max_drawdown: number; win_rate: number;
    total_trades: number; session_number: number | null;
  } | null;
  equity_curve: { timestamp: string; equity: number; session_number: number }[];
  positions: { symbol: string; quantity: number; avg_entry: number; current_price: number; unrealized_pnl: number }[];
}

export interface Trade {
  id: number; model_id: number; model_name: string; symbol: string;
  side: string; quantity: number; price: number; transaction_cost: number;
  session_number: number; filled_at: string | null;
}

export interface Generation {
  generation: number; session_date: string; model_count: number;
  eliminated_count: number; best_fitness: number | null; avg_fitness: number | null;
}

export interface SessionInfo {
  date: string; session_number: number; generation: number;
  started_at: string | null; ended_at: string | null;
  total_bars: number; total_trades: number;
}

export interface ArenaStatus {
  phase: string;
  detail: string;
  session_number: number;
  bar: number;
  total_bars: number | null;
  timestamp: string | null;
}

export interface ArenaRunState {
  state: 'idle' | 'running' | 'finished'
  config: { num_sessions: number; session_minutes: number } | null
  error: string | null
}

export interface DashboardData {
  models: Model[]; trades: Trade[]; generations: Generation[];
  sessions: SessionInfo[];
  arena_status: ArenaStatus | null;
  arena_log: string[];
  arena_run_state: ArenaRunState | null;
}

export function useWebSocket() {
  const [data, setData] = useState<DashboardData | null>(null)
  const [connected, setConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)

  const connect = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws`)
    wsRef.current = ws
    ws.onopen = () => setConnected(true)
    ws.onclose = () => { setConnected(false); setTimeout(connect, 3000) }
    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data)
        if (msg.type === 'update' && msg.data) setData(msg.data)
      } catch {}
    }
  }, [])

  useEffect(() => {
    fetch('/api/dashboard').then(r => r.json()).then(setData).catch(() => {})
    connect()
    return () => { wsRef.current?.close() }
  }, [connect])

  return { data, connected }
}

// --- Playback types & hook ---

interface SnapshotPoint {
  timestamp: string; equity: number; return_pct: number;
  sharpe: number; drawdown: number; trades: number; win_rate: number;
  session_number: number;
}

interface ModelSnapshots {
  model_id: number; name: string; strategy_type: string;
  points: SnapshotPoint[];
}

/** A single frame = all models' state at one timestamp */
export interface PlaybackFrame {
  timestamp: string;
  frameIndex: number;
  models: Model[];
}

export type PlaybackSpeed = 1 | 2 | 5 | 10

export interface PlaybackState {
  availableDates: string[];
  selectedDate: string | null;
  selectedSession: number | null;
  loading: boolean;
  frames: PlaybackFrame[];
  currentFrame: number;
  totalFrames: number;
  playing: boolean;
  speed: PlaybackSpeed;
  load: (date: string, session: number | null) => void;
  play: () => void;
  pause: () => void;
  seek: (frame: number) => void;
  setSpeed: (s: PlaybackSpeed) => void;
  reset: () => void;
  refreshDates: () => void;
}

const MAX_FRAMES = 500

function buildFrames(models: ModelSnapshots[]): PlaybackFrame[] {
  // Collect all unique timestamps across all models
  const tsSet = new Set<string>()
  for (const m of models) {
    for (const p of m.points) tsSet.add(p.timestamp)
  }
  const allTimestamps = [...tsSet].sort()

  // Downsample to MAX_FRAMES for smooth playback
  let timestamps: string[]
  if (allTimestamps.length <= MAX_FRAMES) {
    timestamps = allTimestamps
  } else {
    const step = allTimestamps.length / MAX_FRAMES
    timestamps = []
    for (let i = 0; i < MAX_FRAMES; i++) {
      timestamps.push(allTimestamps[Math.floor(i * step)])
    }
    // Always include the last timestamp
    if (timestamps[timestamps.length - 1] !== allTimestamps[allTimestamps.length - 1]) {
      timestamps.push(allTimestamps[allTimestamps.length - 1])
    }
  }

  // Track cursor per model for O(n) instead of O(n*t) filter
  const cursors = new Array(models.length).fill(0)

  return timestamps.map((ts, frameIndex) => {
    const frameModels: Model[] = models.map((m, mi) => {
      // Advance cursor to latest point <= ts
      while (cursors[mi] < m.points.length - 1 && m.points[cursors[mi] + 1].timestamp <= ts) {
        cursors[mi]++
      }
      const current = m.points.length > 0 && m.points[cursors[mi]].timestamp <= ts
        ? m.points[cursors[mi]]
        : null

      return {
        id: m.model_id,
        name: m.name,
        strategy_type: m.strategy_type,
        generation: 1,
        rank: mi + 1, // re-ranked below
        parent_ids: null,
        genetic_operation: null,
        initial_capital: 1000,
        current_capital: current?.equity ?? 1000,
        performance: current ? {
          equity: current.equity,
          total_pnl: current.equity - 1000,
          return_pct: current.return_pct,
          sharpe_ratio: current.sharpe,
          max_drawdown: current.drawdown,
          win_rate: current.win_rate,
          total_trades: current.trades,
          session_number: current.session_number,
        } : null,
        equity_curve: m.points.slice(0, cursors[mi] + 1)
          .filter(p => p.timestamp <= ts)
          .map(p => ({
            timestamp: p.timestamp,
            equity: p.equity,
            session_number: p.session_number,
          })),
        positions: [],
      }
    })

    // Rank by return_pct descending
    frameModels.sort((a, b) => (b.performance?.return_pct ?? 0) - (a.performance?.return_pct ?? 0))
    frameModels.forEach((m, i) => { m.rank = i + 1 })

    return { timestamp: ts, frameIndex, models: frameModels }
  })
}

export function usePlayback(): PlaybackState {
  const [availableDates, setAvailableDates] = useState<string[]>([])
  const [selectedDate, setSelectedDate] = useState<string | null>(null)
  const [selectedSession, setSelectedSession] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [frames, setFrames] = useState<PlaybackFrame[]>([])
  const [currentFrame, setCurrentFrame] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState<PlaybackSpeed>(1)
  const intervalRef = useRef<number | null>(null)

  const refreshDates = useCallback(() => {
    fetch('/api/sessions/dates')
      .then(r => r.json())
      .then(data => { if (Array.isArray(data)) setAvailableDates(data) })
      .catch(() => {})
  }, [])

  // Fetch available dates on mount
  useEffect(() => { refreshDates() }, [refreshDates])

  const load = useCallback(async (date: string, session: number | null) => {
    setLoading(true)
    setPlaying(false)
    setCurrentFrame(0)
    setSelectedDate(date)
    setSelectedSession(session)
    try {
      const url = session != null
        ? `/api/sessions/${date}/performance?session_number=${session}`
        : `/api/sessions/${date}/performance`
      const res = await fetch(url)
      const data: ModelSnapshots[] = await res.json()
      setFrames(buildFrames(data))
    } catch {
      setFrames([])
    }
    setLoading(false)
  }, [])

  // Playback interval
  useEffect(() => {
    if (playing && frames.length > 0) {
      const ms = Math.max(50, 500 / speed)
      intervalRef.current = window.setInterval(() => {
        setCurrentFrame(prev => {
          if (prev >= frames.length - 1) {
            setPlaying(false)
            return prev
          }
          return prev + 1
        })
      }, ms)
    }
    return () => {
      if (intervalRef.current != null) window.clearInterval(intervalRef.current)
    }
  }, [playing, speed, frames.length])

  const play = useCallback(() => {
    if (currentFrame >= frames.length - 1) setCurrentFrame(0)
    setPlaying(true)
  }, [currentFrame, frames.length])

  const pause = useCallback(() => setPlaying(false), [])
  const seek = useCallback((f: number) => { setPlaying(false); setCurrentFrame(f) }, [])
  const reset = useCallback(() => {
    setPlaying(false)
    setCurrentFrame(0)
    setFrames([])
    setSelectedDate(null)
    setSelectedSession(null)
  }, [])

  return {
    availableDates, selectedDate, selectedSession, loading,
    frames, currentFrame, totalFrames: frames.length,
    playing, speed,
    load, play, pause, seek, setSpeed, reset, refreshDates,
  }
}

// --- Model Summaries (Reflections) ---

export interface ModelSummary {
  model_id: number
  model_name: string
  strategy_type: string
  session_number: number | null
  summary_type: string // "post_session" | "post_improvement"
  return_pct: number | null
  sharpe_ratio: number | null
  max_drawdown: number | null
  total_trades: number | null
  win_rate: number | null
  fitness: number | null
  rank: number | null
  param_changes: Record<string, { old: unknown; new: unknown }> | null
  reflection: string
}

export type ReflectionTab = string // e.g. '1_session', '1_improve', '2_session', '2_improve'

// --- Model Trades (Round-Trip) ---

export interface ModelTrade {
  symbol: string
  buy_price: number
  buy_time: string | null
  sell_price: number | null
  sell_time: string | null
  quantity: number
  pnl: number
  pnl_pct: number
  status: 'closed' | 'open'
  reason?: string
}

export function useModelTrades(modelId: number | null, sessionDate: string | null) {
  const [trades, setTrades] = useState<ModelTrade[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (modelId == null || !sessionDate) { setTrades([]); return }
    setLoading(true)
    fetch(`/api/models/${modelId}/trades?session_date=${sessionDate}`)
      .then(r => r.json())
      .then(data => { if (Array.isArray(data)) setTrades(data) })
      .catch(() => setTrades([]))
      .finally(() => setLoading(false))
  }, [modelId, sessionDate])

  return { trades, loading }
}

// --- History View types & hooks ---

export interface HistoryModel {
  id: number; rank: number; name: string; strategy_type: string;
  start_capital: number; end_capital: number; return_pct: number;
  realized_pnl: number; trade_count: number; winning_trades: number; win_rate: number;
}

export interface HistoryTrade {
  model_name: string; symbol: string; side: string;
  quantity: number; fill_price: number; realized_pnl: number | null;
  filled_at: string | null;
}

export interface HistorySession {
  session_number: number;
  started_at: string | null; ended_at: string | null;
  total_bars: number; total_trades: number;
}

export interface DailyHistoryData {
  date: string;
  sessions: HistorySession[];
  models: HistoryModel[];
  portfolio: {
    total_pnl: number; return_pct: number; total_trades: number;
    win_rate: number; initial_capital: number; end_capital: number;
    alpaca_pnl: number | null;
    unattributed_pnl: number | null;
  };
  trades: HistoryTrade[];
  equity_curve: { time: string; value: number }[];
  cfa_grade: string | null;
  cfa_summary: string | null;
}

export function useHistoryDates() {
  const [dates, setDates] = useState<string[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    fetch('/api/history/dates')
      .then(r => r.json())
      .then(data => { if (Array.isArray(data)) setDates(data) })
      .catch(() => setDates([]))
      .finally(() => setLoading(false))
  }, [])

  return { dates, loading }
}

export function useDailyHistory(date: string | null) {
  const [data, setData] = useState<DailyHistoryData | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!date) { setData(null); return }
    setLoading(true)
    fetch(`/api/history/${date}`)
      .then(r => r.json())
      .then(d => setData(d as DailyHistoryData))
      .catch(() => setData(null))
      .finally(() => setLoading(false))
  }, [date])

  return { data, loading }
}

export function useHistoricalData(sessionDate: string | null) {
  const [data, setData] = useState<DashboardData | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!sessionDate) { setData(null); return }
    setLoading(true)
    fetch(`/api/dashboard?session_date=${sessionDate}`)
      .then(r => r.json())
      .then(d => setData(d as DashboardData))
      .catch(() => setData(null))
      .finally(() => setLoading(false))
  }, [sessionDate])

  return { data, loading }
}

export function useModelSummaries(sessionDate: string | null, reflectionTab: ReflectionTab) {
  const [summaries, setSummaries] = useState<ModelSummary[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!sessionDate) { setSummaries([]); return }
    setLoading(true)

    // Parse tab format: "N_session" or "N_improve" (e.g. "1_session", "3_improve")
    // Legacy support: "s1" -> session_number=1, type=post_session
    let sessionNumber: number | null = null
    let summaryType: string | null = null

    const match = reflectionTab.match(/^(\d+)_(session|improve)$/)
    if (match) {
      sessionNumber = parseInt(match[1])
      summaryType = match[2] === 'session' ? 'post_session' : 'post_improvement'
    } else if (reflectionTab === 's1') { sessionNumber = 1; summaryType = 'post_session' }
    else if (reflectionTab === 's1_improve') { sessionNumber = 1; summaryType = 'post_improvement' }
    else if (reflectionTab === 's2') { sessionNumber = 2; summaryType = 'post_session' }
    else if (reflectionTab === 's2_improve') { sessionNumber = 2; summaryType = 'post_improvement' }

    let url = `/api/model-summaries/${sessionDate}`
    const params: string[] = []
    if (sessionNumber != null) params.push(`session_number=${sessionNumber}`)
    if (summaryType != null) params.push(`summary_type=${summaryType}`)
    if (params.length) url += '?' + params.join('&')

    fetch(url)
      .then(r => r.json())
      .then(data => { if (Array.isArray(data)) setSummaries(data) })
      .catch(() => setSummaries([]))
      .finally(() => setLoading(false))
  }, [sessionDate, reflectionTab])

  return { summaries, loading }
}
