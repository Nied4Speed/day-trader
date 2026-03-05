import { useState, useMemo, useRef, useEffect } from 'react'
import { useWebSocket, usePlayback, useHistoricalData, useModelSummaries, useModelTrades, PlaybackSpeed, Model, Trade, Generation, SessionInfo, ReflectionTab, ArenaStatus, ArenaRunState, ModelTrade } from './hooks'

function formatPct(v: number) { return (v >= 0 ? '+' : '') + v.toFixed(3) + '%' }
function formatMoney(v: number) { return '$' + v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) }
function cls(v: number) { return v > 0 ? 'positive' : v < 0 ? 'negative' : 'neutral' }

type SessionTab = number | 'all'
type ModelFilter = 'all' | number  // 'all' or model id

function SessionTabs({ active, onChange, sessions }: { active: SessionTab; onChange: (t: SessionTab) => void; sessions: SessionInfo[] }) {
  // Build unique session numbers from data
  const sessionNums = [...new Set(sessions.map(s => s.session_number))].sort((a, b) => a - b)
  // If no sessions yet, show at least S1
  if (sessionNums.length === 0) sessionNums.push(1)

  return (
    <div className="session-tabs">
      {sessionNums.map(n => {
        const s = sessions.find(s => s.session_number === n)
        return (
          <button
            key={n}
            className={`session-tab ${active === n ? 'active' : ''}`}
            onClick={() => onChange(n)}
          >
            S{n} {s && s.ended_at ? '(Done)' : s ? '(Live)' : ''}
          </button>
        )
      })}
      <button className={`session-tab ${active === 'all' ? 'active' : ''}`} onClick={() => onChange('all')}>
        All
      </button>
    </div>
  )
}

function StatsBar({ models, sessions }: { models: Model[]; sessions: SessionInfo[] }) {
  const totalCapital = models.reduce((s, m) => s + m.current_capital, 0)
  const totalPnl = models.reduce((s, m) => s + (m.performance?.total_pnl ?? 0), 0)
  const totalTrades = models.reduce((s, m) => s + (m.performance?.total_trades ?? 0), 0)
  const totalBars = sessions.reduce((s, sess) => s + sess.total_bars, 0)
  const gen = sessions[0]?.generation ?? 1

  return (
    <div className="stats-row">
      <div className="stat">
        <span className="label">Generation</span>
        <span className="value">{gen}</span>
      </div>
      <div className="stat">
        <span className="label">Models</span>
        <span className="value">{models.length}</span>
      </div>
      <div className="stat">
        <span className="label">Total Capital</span>
        <span className="value">{formatMoney(totalCapital)}</span>
      </div>
      <div className="stat">
        <span className="label">Total P&L</span>
        <span className={`value ${cls(totalPnl)}`}>{formatMoney(totalPnl)}</span>
      </div>
      <div className="stat">
        <span className="label">Trades</span>
        <span className="value">{totalTrades}</span>
      </div>
      <div className="stat">
        <span className="label">Bars</span>
        <span className="value">{totalBars}</span>
      </div>
    </div>
  )
}

function Rankings({ models, selectedModelId, onSelectModel }: { models: Model[]; selectedModelId: number | null; onSelectModel: (id: number | null) => void }) {
  return (
    <table className="rankings-table">
      <thead><tr>
        <th>#</th><th>Model</th><th>Type</th><th>Gen</th>
        <th>Capital</th><th>P&L</th><th>Return</th><th>Sharpe</th>
        <th>Drawdown</th><th>Trades</th><th>Win Rate</th>
      </tr></thead>
      <tbody>
        {models.map(m => {
          const p = m.performance
          const isSelected = m.id === selectedModelId
          return (
            <tr
              key={m.id}
              className={`rankings-row-clickable ${isSelected ? 'rankings-row-selected' : ''}`}
              onClick={() => onSelectModel(isSelected ? null : m.id)}
            >
              <td>{m.rank}</td>
              <td className="model-name">{m.name}</td>
              <td><span className="strategy-badge">{m.strategy_type}</span></td>
              <td>{m.generation}</td>
              <td>{formatMoney(m.current_capital)}</td>
              <td className={cls(p?.total_pnl ?? 0)}>{p ? formatMoney(p.total_pnl) : '-'}</td>
              <td className={cls(p?.return_pct ?? 0)}>{p ? formatPct(p.return_pct) : '-'}</td>
              <td>{p ? p.sharpe_ratio.toFixed(3) : '-'}</td>
              <td>{p ? formatPct(-p.max_drawdown * 100) : '-'}</td>
              <td>{p?.total_trades ?? 0}</td>
              <td>{p ? (p.win_rate * 100).toFixed(1) + '%' : '-'}</td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

function TradeHistory({ trades, loading, modelName }: { trades: ModelTrade[]; loading: boolean; modelName: string }) {
  if (loading) return <div className="empty-state">Loading trades...</div>
  if (trades.length === 0) return <div className="empty-state">No trades for {modelName}</div>

  const totalPnl = trades.reduce((s, t) => s + t.pnl, 0)
  const closedTrades = trades.filter(t => t.status === 'closed')
  const wins = closedTrades.filter(t => t.pnl > 0).length

  return (
    <div className="trade-history">
      <div className="trade-history-summary">
        <span>Showing {trades.length} round-trip{trades.length !== 1 ? 's' : ''} for <strong>{modelName}</strong></span>
        <span className={cls(totalPnl)}>Net: {formatMoney(totalPnl)}</span>
        {closedTrades.length > 0 && <span>Win Rate: {((wins / closedTrades.length) * 100).toFixed(0)}%</span>}
      </div>
      <table className="rankings-table trade-history-table">
        <thead><tr>
          <th>Symbol</th><th>Status</th><th>Qty</th><th>Buy Price</th>
          <th>Sell Price</th><th>P&L</th><th>P&L %</th><th>Buy Time</th><th>Sell Time</th>
        </tr></thead>
        <tbody>
          {trades.map((t, i) => (
            <tr key={i}>
              <td>{t.symbol}</td>
              <td><span className={`trade-status-badge ${t.status}`}>{t.status.toUpperCase()}</span></td>
              <td>{t.quantity}</td>
              <td>${t.buy_price?.toFixed(2) ?? '-'}</td>
              <td>{t.sell_price != null ? '$' + t.sell_price.toFixed(2) : '-'}</td>
              <td className={cls(t.pnl)}>{formatMoney(t.pnl)}</td>
              <td className={cls(t.pnl_pct)}>{formatPct(t.pnl_pct)}</td>
              <td className="trade-time">{t.buy_time ? new Date(t.buy_time).toLocaleTimeString() : '-'}</td>
              <td className="trade-time">{t.sell_time ? new Date(t.sell_time).toLocaleTimeString() : '-'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function TradeFeed({ trades, tab }: { trades: Trade[]; tab: SessionTab }) {
  const filtered = tab === 'all' ? trades : trades.filter(t => t.session_number === tab)
  return (
    <div className="trade-feed">
      {filtered.length === 0 && <div className="empty-state">No trades yet</div>}
      {filtered.map(t => (
        <div className="trade-item" key={t.id}>
          <span>
            <span className={`trade-side ${t.side}`}>{t.side.toUpperCase()}</span>
            {' '}{t.quantity} {t.symbol}
          </span>
          <span className="trade-model">{t.model_name}</span>
          <span>${t.price?.toFixed(2) ?? '-'}</span>
          <span className="trade-session">S{t.session_number}</span>
          <span className="trade-time">{t.filled_at ? new Date(t.filled_at).toLocaleTimeString() : ''}</span>
        </div>
      ))}
    </div>
  )
}

function GenerationHistory({ generations }: { generations: Generation[] }) {
  return (
    <div>
      {generations.length === 0 && <div className="empty-state">No generations yet</div>}
      {generations.map(g => (
        <div className="gen-item" key={g.generation}>
          <span className="gen-label">Gen {g.generation}</span>
          <span>{g.model_count} models</span>
          <span className="negative">{g.eliminated_count} culled</span>
          <span>Best: {g.best_fitness?.toFixed(4) ?? '-'}</span>
          <span>Avg: {g.avg_fitness?.toFixed(4) ?? '-'}</span>
        </div>
      ))}
    </div>
  )
}

function Positions({ models }: { models: Model[] }) {
  const allPositions = models.flatMap(m =>
    m.positions.map(p => ({ ...p, model_name: m.name }))
  )
  if (allPositions.length === 0) return <div className="empty-state">No open positions</div>

  return (
    <table className="rankings-table">
      <thead><tr>
        <th>Model</th><th>Symbol</th><th>Qty</th><th>Avg Entry</th><th>Current</th><th>Unrealized P&L</th>
      </tr></thead>
      <tbody>
        {allPositions.map((p, i) => (
          <tr key={i}>
            <td>{p.model_name}</td>
            <td>{p.symbol}</td>
            <td>{p.quantity}</td>
            <td>${p.avg_entry.toFixed(2)}</td>
            <td>${p.current_price.toFixed(2)}</td>
            <td className={cls(p.unrealized_pnl)}>{formatMoney(p.unrealized_pnl)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function ReflectionsPanel({ reflectionTab, onTabChange, sessionDate, sessions }: {
  reflectionTab: ReflectionTab
  onTabChange: (t: ReflectionTab) => void
  sessionDate: string | null
  sessions: SessionInfo[]
}) {
  const { summaries, loading } = useModelSummaries(sessionDate, reflectionTab)
  const [expanded, setExpanded] = useState<number | null>(null)

  // Generate reflection tabs from sessions
  const sessionNums = [...new Set(sessions.map(s => s.session_number))].sort((a, b) => a - b)
  // Fallback: if no sessions yet, show S1/S2
  if (sessionNums.length === 0) { sessionNums.push(1); sessionNums.push(2) }

  const tabs: [string, string][] = []
  for (const n of sessionNums) {
    tabs.push([`${n}_session`, `S${n} Reflection`])
    tabs.push([`${n}_improve`, `S${n} Improvement`])
  }

  return (
    <div className="reflections-panel">
      <div className="reflection-tabs">
        {tabs.map(([t, label]) => (
          <button
            key={t}
            className={`reflection-tab ${reflectionTab === t ? 'active' : ''}`}
            onClick={() => onTabChange(t)}
          >
            {label}
          </button>
        ))}
      </div>
      {loading && <div className="empty-state">Loading reflections...</div>}
      {!loading && summaries.length === 0 && <div className="empty-state">No reflections yet for this tab</div>}
      {!loading && summaries.map(s => {
        const isExpanded = expanded === s.model_id
        return (
          <div
            key={`${s.model_id}-${s.summary_type}-${s.session_number}`}
            className={`reflection-card ${isExpanded ? 'expanded' : ''}`}
            onClick={() => setExpanded(isExpanded ? null : s.model_id)}
          >
            <div className="reflection-header">
              {s.rank != null && <span className="reflection-rank">#{s.rank}</span>}
              <span className="reflection-model">{s.model_name}</span>
              <span className="strategy-badge">{s.strategy_type}</span>
              <span className={`reflection-return ${cls(s.return_pct ?? 0)}`}>
                {s.return_pct != null ? formatPct(s.return_pct) : '-'}
              </span>
            </div>
            {isExpanded && (
              <div className="reflection-body">
                <div className="reflection-metrics">
                  <div className="r-metric">
                    <span className="label">Sharpe</span>
                    <span className="value">{s.sharpe_ratio?.toFixed(3) ?? '-'}</span>
                  </div>
                  <div className="r-metric">
                    <span className="label">Drawdown</span>
                    <span className="value">{s.max_drawdown != null ? formatPct(-s.max_drawdown * 100) : '-'}</span>
                  </div>
                  <div className="r-metric">
                    <span className="label">Trades</span>
                    <span className="value">{s.total_trades ?? 0}</span>
                  </div>
                  <div className="r-metric">
                    <span className="label">Win Rate</span>
                    <span className="value">{s.win_rate != null ? (s.win_rate * 100).toFixed(1) + '%' : '-'}</span>
                  </div>
                  <div className="r-metric">
                    <span className="label">Fitness</span>
                    <span className="value">{s.fitness?.toFixed(4) ?? '-'}</span>
                  </div>
                </div>
                <div className="reflection-text">{s.reflection}</div>
                {s.param_changes && Object.keys(s.param_changes).length > 0 && (
                  <div className="param-changes">
                    <div className="param-label">Parameter Changes</div>
                    {Object.entries(s.param_changes).map(([param, change]) => (
                      <div key={param} className="param-item">
                        <span className="param-name">{param}</span>
                        <span className="param-old">{String((change as {old: unknown}).old)}</span>
                        <span className="param-arrow">&rarr;</span>
                        <span className="param-new">{String((change as {new: unknown}).new)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

const PHASE_LABELS: Record<string, string> = {
  offline: 'Offline',
  starting: 'Starting',
  warmup: 'Warming Up',
  premarket: 'Pre-Market',
  waiting: 'Waiting for Open',
  session: 'Trading',
  reflecting: 'Reflecting',
  improving: 'Self-Improving',
  break: 'Break',
  complete: 'Day Complete',
}

const PHASE_COLORS: Record<string, string> = {
  offline: 'var(--text-muted)',
  starting: 'var(--yellow)',
  warmup: 'var(--yellow)',
  premarket: 'var(--yellow)',
  waiting: 'var(--blue)',
  session: 'var(--green)',
  reflecting: 'var(--purple)',
  improving: 'var(--purple)',
  break: 'var(--blue)',
  complete: 'var(--green)',
}

function StatusBanner({ status }: { status: ArenaStatus | null }) {
  if (!status) return null
  const label = PHASE_LABELS[status.phase] || status.phase
  const color = PHASE_COLORS[status.phase] || 'var(--text-secondary)'
  const isTrading = status.phase === 'session'
  const showPulse = ['session', 'premarket', 'warmup', 'starting'].includes(status.phase)

  return (
    <div className="status-banner">
      <div className="status-banner-left">
        <span className={`status-banner-dot ${showPulse ? 'pulse' : ''}`} style={{ background: color }} />
        <span className="status-banner-phase" style={{ color }}>{label}</span>
        <span className="status-banner-detail">{status.detail}</span>
      </div>
      {isTrading && status.bar > 0 && (
        <div className="status-banner-right">
          <span className="status-banner-bars">Bar {status.bar}{status.total_bars ? ` / ~${status.total_bars}` : ''}</span>
        </div>
      )}
    </div>
  )
}

function ActivityLog({ log, expanded, onToggle }: { log: string[]; expanded: boolean; onToggle: () => void }) {
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (expanded && endRef.current) {
      endRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [log.length, expanded])

  return (
    <div className="activity-log">
      <button className="activity-log-toggle" onClick={onToggle}>
        Activity Log ({log.length} lines) {expanded ? '\u25B2' : '\u25BC'}
      </button>
      {expanded && (
        <div className="activity-log-body">
          {log.length === 0 && <div className="empty-state">No log entries yet</div>}
          {log.map((line, i) => {
            const isError = line.includes('[ERROR]') || line.includes('Traceback')
            const isWarning = line.includes('[WARNING]')
            return (
              <div key={i} className={`log-line ${isError ? 'error' : isWarning ? 'warning' : ''}`}>
                {line}
              </div>
            )
          })}
          <div ref={endRef} />
        </div>
      )}
    </div>
  )
}

function ModelFilterDropdown({ models, value, onChange }: { models: Model[]; value: ModelFilter; onChange: (v: ModelFilter) => void }) {
  const strategyGroups = useMemo(() => {
    const groups: Record<string, Model[]> = {}
    for (const m of models) {
      const key = m.strategy_type
      if (!groups[key]) groups[key] = []
      groups[key].push(m)
    }
    return Object.entries(groups).sort(([a], [b]) => a.localeCompare(b))
  }, [models])

  return (
    <select
      className="model-filter-select"
      value={value === 'all' ? 'all' : String(value)}
      onChange={e => onChange(e.target.value === 'all' ? 'all' : Number(e.target.value))}
    >
      <option value="all">All Models ({models.length})</option>
      {strategyGroups.map(([strat, group]) => (
        <optgroup key={strat} label={strat}>
          {group.map(m => (
            <option key={m.id} value={m.id}>{m.name}</option>
          ))}
        </optgroup>
      ))}
    </select>
  )
}

function StartSessionForm({ onStart, onPlayback }: {
  onStart: (numSessions: number, sessionMinutes: number) => void
  onPlayback: () => void
}) {
  const [numSessions, setNumSessions] = useState(2)
  const [sessionMinutes, setSessionMinutes] = useState(60)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const totalMinutes = numSessions * sessionMinutes
  const totalHours = (totalMinutes / 60).toFixed(1)

  const handleStart = async () => {
    setSubmitting(true)
    setError(null)
    try {
      const res = await fetch('/api/arena/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ num_sessions: numSessions, session_minutes: sessionMinutes }),
      })
      if (!res.ok) {
        const data = await res.json().catch(() => ({ detail: 'Unknown error' }))
        setError(data.detail || 'Failed to start arena')
      } else {
        onStart(numSessions, sessionMinutes)
      }
    } catch (e) {
      setError('Failed to connect to server')
    }
    setSubmitting(false)
  }

  return (
    <div className="start-session-form">
      <h3>Launch Trading Session</h3>
      <div className="session-config-row">
        <label>
          <span className="config-label">Sessions</span>
          <input
            type="number"
            min={1}
            max={20}
            value={numSessions}
            onChange={e => setNumSessions(Math.max(1, Math.min(20, parseInt(e.target.value) || 1)))}
          />
        </label>
        <span className="config-x">x</span>
        <label>
          <span className="config-label">Duration (min)</span>
          <input
            type="number"
            min={5}
            max={360}
            value={sessionMinutes}
            onChange={e => setSessionMinutes(Math.max(5, Math.min(360, parseInt(e.target.value) || 5)))}
          />
        </label>
      </div>
      <div className="session-config-summary">
        {numSessions} session{numSessions !== 1 ? 's' : ''} x {sessionMinutes} min = {totalMinutes} min total ({totalHours} hours)
      </div>
      {error && <div className="session-config-error">{error}</div>}
      <div className="session-config-actions">
        <button className="start-btn" onClick={handleStart} disabled={submitting}>
          {submitting ? 'Starting...' : 'Start Arena'}
        </button>
        <button className="mode-btn" onClick={onPlayback}>Playback</button>
      </div>
    </div>
  )
}

function PlaybackControls({ pb, sessions }: { pb: ReturnType<typeof usePlayback>; sessions: SessionInfo[] }) {
  const speeds: PlaybackSpeed[] = [1, 2, 5, 10]
  const frame = pb.frames[pb.currentFrame]
  const ts = frame ? new Date(frame.timestamp).toLocaleTimeString() : '--:--:--'

  // Get unique session numbers for the selected date
  const sessionNums = [...new Set(sessions.map(s => s.session_number))].sort((a, b) => a - b)

  return (
    <div className="playback-panel">
      <div className="playback-top">
        <div className="playback-selectors">
          <select
            className="playback-select"
            value={pb.selectedDate ?? ''}
            onChange={e => e.target.value && pb.load(e.target.value, pb.selectedSession)}
          >
            <option value="">Select date...</option>
            {pb.availableDates.map(d => <option key={d} value={d}>{d}</option>)}
          </select>
          <select
            className="playback-select"
            value={pb.selectedSession ?? ''}
            onChange={e => {
              const v = e.target.value
              const sess = v === '' ? null : Number(v)
              if (pb.selectedDate) pb.load(pb.selectedDate, sess)
            }}
          >
            <option value="">All Sessions</option>
            {sessionNums.length > 0
              ? sessionNums.map(n => <option key={n} value={n}>Session {n}</option>)
              : <>
                  <option value="1">Session 1</option>
                  <option value="2">Session 2</option>
                </>
            }
          </select>
        </div>
        <div className="playback-time">{ts}</div>
        <div className="playback-frame-count">
          {pb.currentFrame + 1} / {pb.totalFrames || '?'}
        </div>
      </div>

      {pb.loading && <div className="playback-loading">Loading snapshots...</div>}

      {pb.totalFrames > 0 && (
        <div className="playback-bar">
          <button className="playback-btn" onClick={pb.playing ? pb.pause : pb.play}>
            {pb.playing ? '\u23F8' : '\u25B6'}
          </button>
          <input
            type="range"
            className="playback-slider"
            min={0}
            max={pb.totalFrames - 1}
            value={pb.currentFrame}
            onChange={e => pb.seek(Number(e.target.value))}
          />
          <div className="playback-speeds">
            {speeds.map(s => (
              <button
                key={s}
                className={`speed-btn ${pb.speed === s ? 'active' : ''}`}
                onClick={() => pb.setSpeed(s)}
              >
                {s}x
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

interface SwingPoint {
  index: number
  equity: number
  prevEquity: number
  changePct: number
  timestamp: string
  x: number
  y: number
  type: 'peak' | 'valley'
}

function findSwings(
  pts: { equity: number; timestamp: string }[],
  toX: (i: number) => number,
  toY: (eq: number) => number,
  minSwingPct: number = 5,
): SwingPoint[] {
  if (pts.length < 3) return []
  const swings: SwingPoint[] = []

  // Walk the curve, find direction reversals
  let anchor = 0
  let dir = 0 // 0=unknown, 1=up, -1=down

  for (let i = 1; i < pts.length; i++) {
    const move = pts[i].equity - pts[anchor].equity
    const movePct = (move / pts[anchor].equity) * 100

    if (dir === 0) {
      if (Math.abs(movePct) >= minSwingPct) {
        dir = movePct > 0 ? 1 : -1
      }
      continue
    }

    // Check for reversal
    if (dir === 1 && movePct < -minSwingPct) {
      // Was going up, now dropped enough - previous high was a peak
      // Find the actual peak between anchor and i
      let peakIdx = anchor
      for (let j = anchor; j < i; j++) {
        if (pts[j].equity > pts[peakIdx].equity) peakIdx = j
      }
      swings.push({
        index: peakIdx,
        equity: pts[peakIdx].equity,
        prevEquity: pts[anchor].equity,
        changePct: ((pts[peakIdx].equity - pts[anchor].equity) / pts[anchor].equity) * 100,
        timestamp: pts[peakIdx].timestamp,
        x: toX(peakIdx),
        y: toY(pts[peakIdx].equity),
        type: 'peak',
      })
      anchor = peakIdx
      dir = -1
    } else if (dir === -1 && movePct > minSwingPct) {
      // Was going down, now rose enough - previous low was a valley
      let valleyIdx = anchor
      for (let j = anchor; j < i; j++) {
        if (pts[j].equity < pts[valleyIdx].equity) valleyIdx = j
      }
      swings.push({
        index: valleyIdx,
        equity: pts[valleyIdx].equity,
        prevEquity: pts[anchor].equity,
        changePct: ((pts[valleyIdx].equity - pts[anchor].equity) / pts[anchor].equity) * 100,
        timestamp: pts[valleyIdx].timestamp,
        x: toX(valleyIdx),
        y: toY(pts[valleyIdx].equity),
        type: 'valley',
      })
      anchor = valleyIdx
      dir = 1
    }
  }

  return swings
}

function EquityChart({ models, tab }: { models: Model[]; tab: SessionTab }) {
  const colors = ['#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6','#ec4899','#06b6d4','#f97316','#14b8a6','#6366f1']
  const [hover, setHover] = useState<{ model: string; swing: SwingPoint; screenX: number; screenY: number } | null>(null)

  const filtered = models.map(m => ({
    ...m,
    equity_curve: tab === 'all'
      ? m.equity_curve
      : m.equity_curve.filter(p => p.session_number === tab),
  })).filter(m => m.equity_curve.length > 1)

  if (filtered.length === 0) {
    return <div className="empty-state">No equity data yet</div>
  }

  const allEquities = filtered.flatMap(m => m.equity_curve.map(p => p.equity))
  const globalMin = Math.min(...allEquities) * 0.999
  const globalMax = Math.max(...allEquities) * 1.001
  const globalRange = globalMax - globalMin || 1

  const toX = (i: number, total: number) => (i / (total - 1)) * 760 + 10
  const toY = (eq: number) => 240 - ((eq - globalMin) / globalRange) * 220

  return (
    <div style={{ width: '100%', padding: '1rem 0', position: 'relative' }}>
      <svg viewBox="0 0 800 250" style={{ width: '100%', height: '250px' }}>
        {[0, 0.25, 0.5, 0.75, 1].map(pct => {
          const y = 240 - pct * 220
          const val = globalMin + pct * globalRange
          return (
            <g key={pct}>
              <line x1="10" y1={y} x2="790" y2={y} stroke="var(--border)" strokeWidth="0.5" />
              <text x="792" y={y + 3} fill="var(--text-muted)" fontSize="8">${val.toFixed(0)}</text>
            </g>
          )
        })}
        {filtered.map((m, mi) => {
          const pts = m.equity_curve
          const color = colors[mi % colors.length]
          const path = pts.map((p, i) => {
            const x = toX(i, pts.length)
            const y = toY(p.equity)
            return `${i === 0 ? 'M' : 'L'}${x},${y}`
          }).join(' ')

          const swings = findSwings(
            pts,
            (i) => toX(i, pts.length),
            toY,
          )

          return (
            <g key={m.id}>
              <path d={path} fill="none" stroke={color} strokeWidth="1.5" opacity="0.85" />
              {swings.map((sw, si) => (
                <g key={si}>
                  <circle
                    cx={sw.x} cy={sw.y} r="3"
                    fill={sw.type === 'peak' ? 'var(--green)' : 'var(--red)'}
                    stroke={color} strokeWidth="1"
                  />
                  {/* Invisible larger hit area for hover */}
                  <circle
                    cx={sw.x} cy={sw.y} r="10"
                    fill="transparent"
                    style={{ cursor: 'pointer' }}
                    onMouseEnter={(e) => setHover({ model: m.name, swing: sw, screenX: e.clientX, screenY: e.clientY })}
                    onMouseLeave={() => setHover(null)}
                  />
                </g>
              ))}
            </g>
          )
        })}
      </svg>

      {hover && (
        <div className="chart-tooltip" style={{ left: hover.screenX, top: hover.screenY }}>
          <div className="chart-tooltip-model">{hover.model}</div>
          <div>{formatMoney(hover.swing.equity)}</div>
          <div className={hover.swing.changePct >= 0 ? 'positive' : 'negative'}>
            {hover.swing.changePct >= 0 ? '+' : ''}{hover.swing.changePct.toFixed(1)}% swing
          </div>
          <div className="chart-tooltip-time">
            {new Date(hover.swing.timestamp).toLocaleTimeString()}
          </div>
        </div>
      )}

      <div className="chart-legend">
        {filtered.map((m, mi) => (
          <span key={m.id} style={{ color: colors[mi % colors.length] }}>&#9632; {m.name}</span>
        ))}
      </div>
    </div>
  )
}

export default function App() {
  const { data, connected } = useWebSocket()
  const pb = usePlayback()
  const [tab, setTab] = useState<SessionTab>('all')
  const [mode, setMode] = useState<'live' | 'playback' | 'idle'>('idle')
  const [reflectionTab, setReflectionTab] = useState<ReflectionTab>('1_session')
  const [modelFilter, setModelFilter] = useState<ModelFilter>('all')
  const [logExpanded, setLogExpanded] = useState(false)
  const [selectedModelId, setSelectedModelId] = useState<number | null>(null)
  const today = new Date().toISOString().split('T')[0]

  // Derive mode from arena run state
  const arenaRunState: ArenaRunState = data?.arena_run_state ?? { state: 'idle', config: null, error: null }
  const isRunning = arenaRunState.state === 'running'
  const isPlayback = mode === 'playback'
  const showForm = !isRunning && !isPlayback

  // Auto-switch to live when arena starts running
  useEffect(() => {
    if (isRunning && mode !== 'live') setMode('live')
  }, [isRunning])

  const activeDate = isPlayback ? pb.selectedDate : today
  const frame = isPlayback ? pb.frames[pb.currentFrame] : null

  // Fetch historical dashboard data (trades, sessions, etc.) for playback date
  const { data: historicalData } = useHistoricalData(isPlayback ? pb.selectedDate : null)

  const allModels = isPlayback ? (frame?.models ?? []) : (data?.models ?? [])
  const allTrades = isPlayback ? (historicalData?.trades ?? []) : (data?.trades ?? [])
  const { trades: modelTrades, loading: tradesLoading } = useModelTrades(selectedModelId, activeDate)
  const selectedModel = allModels.find(m => m.id === selectedModelId)
  const generations = isPlayback ? (historicalData?.generations ?? []) : (data?.generations ?? [])
  const sessions = isPlayback ? (historicalData?.sessions ?? []) : (data?.sessions ?? [])
  const arenaStatus = data?.arena_status ?? null
  const arenaLog = data?.arena_log ?? []

  const models = useMemo(() =>
    modelFilter === 'all' ? allModels : allModels.filter(m => m.id === modelFilter),
    [allModels, modelFilter]
  )
  const trades = useMemo(() =>
    modelFilter === 'all' ? allTrades : allTrades.filter(t => t.model_id === modelFilter),
    [allTrades, modelFilter]
  )

  const activeSession = sessions.find(s => !s.ended_at)
  const runConfig = arenaRunState.config
  const sessionLabel = isRunning
    ? (activeSession
      ? `Session ${activeSession.session_number}${runConfig ? `/${runConfig.num_sessions}` : ''} Live`
      : arenaStatus?.phase === 'improving' ? 'Self-Improving'
      : arenaStatus?.phase === 'reflecting' ? 'Reflecting'
      : arenaStatus?.phase === 'break' ? 'Break'
      : 'Starting...')
    : arenaRunState.state === 'finished' ? 'Finished'
    : 'Idle'

  const handleStop = async () => {
    try {
      await fetch('/api/arena/stop', { method: 'POST' })
    } catch {}
  }

  const handlePlayback = () => {
    setMode('playback')
    setSelectedModelId(null)
    setModelFilter('all')
    pb.refreshDates()
  }

  const handleExitPlayback = () => {
    setMode('idle')
    setSelectedModelId(null)
    setModelFilter('all')
    pb.reset()
  }

  return (
    <div className="app">
      <div className="header">
        <div className="header-left">
          <h1>Day Trader Arena</h1>
          <span className="header-subtitle">Evolutionary Trading Competition</span>
        </div>
        <div className="header-right">
          {isRunning && (
            <button className="stop-btn" onClick={handleStop}>Stop</button>
          )}
          {isPlayback && (
            <button className="mode-btn active" onClick={handleExitPlayback}>Exit Playback</button>
          )}
          <div className="status">
            <div className={`status-dot ${isPlayback ? 'playback' : isRunning ? '' : connected ? 'idle-dot' : 'disconnected'}`} />
            {isPlayback ? 'Playback' : connected ? sessionLabel : 'Disconnected'}
          </div>
        </div>
      </div>

      {showForm && (
        <StartSessionForm
          onStart={() => setMode('live')}
          onPlayback={handlePlayback}
        />
      )}

      {isPlayback && <PlaybackControls pb={pb} sessions={sessions} />}

      {!isPlayback && <StatusBanner status={arenaStatus} />}
      {!isPlayback && <ActivityLog log={arenaLog} expanded={logExpanded} onToggle={() => setLogExpanded(e => !e)} />}

      <StatsBar models={models} sessions={sessions} />
      <div className="filter-row">
        <SessionTabs active={tab} onChange={setTab} sessions={sessions} />
        <ModelFilterDropdown models={allModels} value={modelFilter} onChange={setModelFilter} />
      </div>

      <div className="grid">
        <div className="card full-width">
          <h2>Model Reflections {isPlayback && pb.selectedDate ? `(${pb.selectedDate})` : ''}</h2>
          <ReflectionsPanel reflectionTab={reflectionTab} onTabChange={setReflectionTab} sessionDate={activeDate} sessions={sessions} />
        </div>

        <div className="card full-width">
          <h2>Leaderboard {isPlayback && frame ? `@ ${new Date(frame.timestamp).toLocaleTimeString()}` : ''}</h2>
          <Rankings models={models} selectedModelId={selectedModelId} onSelectModel={setSelectedModelId} />
        </div>

        {selectedModelId != null && (
          <div className="card full-width">
            <h2>Trade History - {selectedModel?.name ?? `Model ${selectedModelId}`}</h2>
            <TradeHistory trades={modelTrades} loading={tradesLoading} modelName={selectedModel?.name ?? `Model ${selectedModelId}`} />
          </div>
        )}

        <div className="card full-width">
          <h2>Equity Curves {isPlayback ? '(Playback)' : tab !== 'all' ? `(Session ${tab})` : '(All)'}</h2>
          <div className="chart-container">
            <EquityChart models={models} tab={isPlayback ? 'all' : tab} />
          </div>
        </div>

        <div className="card">
          <h2>{isPlayback ? 'Trades' : 'Live Trades'} {tab !== 'all' ? `(S${tab})` : ''}</h2>
          <TradeFeed trades={trades} tab={tab} />
        </div>

        <div className="card">
          <h2>Open Positions</h2>
          <Positions models={models} />
        </div>

        <div className="card full-width">
          <h2>Evolution History</h2>
          <GenerationHistory generations={generations} />
        </div>
      </div>
    </div>
  )
}
