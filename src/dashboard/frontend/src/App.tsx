import { useState, useMemo } from 'react'
import { useWebSocket, useModelTrades, Model, Trade } from './hooks'

const STRATEGY_DESCRIPTIONS: Record<string, string> = {
  ma_crossover: "Moving Average Crossover. Buys when a fast moving average (e.g. 10-bar) crosses above a slow one (e.g. 30-bar), sells on cross below. Classic trend-following approach. Each instance evolves its own fast/slow periods and allocation size through self-improvement.",
  rsi_reversion: "Relative Strength Index Reversion. Measures how overbought or oversold a stock is on a 0-100 scale. Buys when RSI drops below the oversold threshold (expects a bounce), sells when it rises above overbought. Each instance evolves its own RSI period, oversold/overbought levels, and position sizing.",
  momentum: "Momentum. Measures the rate of price change over a lookback window. Buys when upward momentum exceeds a threshold, sells when it reverses. Each instance evolves its own lookback period, entry/exit thresholds, and allocation.",
  bollinger_bands: "Bollinger Bands. Draws bands at N standard deviations above and below a moving average. Buys when price touches the lower band (statistically cheap), sells at the upper band (statistically expensive). Each instance evolves its own band width, moving average period, and sizing.",
  macd: "Moving Average Convergence Divergence. Compares a fast and slow exponential moving average. Buys when the MACD line crosses above its signal line (bullish shift), sells on cross below. Each instance evolves its own fast/slow/signal periods and allocation.",
  vwap_reversion: "Volume-Weighted Average Price Reversion. VWAP is the average price weighted by volume — institutional traders use it as a benchmark. Buys when price drops significantly below VWAP, sells when it rises above. Each instance evolves its own deviation threshold and sizing.",
  breakout: "Opening Range Breakout. Records the high and low from the first N bars of the session, then buys if price breaks above that range (momentum entry) or sells on breakdown below. Each instance evolves its own range window size and confirmation threshold.",
  mean_reversion: "Mean Reversion. Buys when price falls far below its simple moving average (betting it returns to the mean), sells when it rises far above. Unlike RSI reversion, this uses raw price distance. Each instance evolves its own moving average period, deviation threshold, and sizing.",
}

/** Build a model-specific description noting which variant it is among siblings */
function getModelDescription(model: Model, allModels: Model[]): string {
  const base = STRATEGY_DESCRIPTIONS[model.strategy_type] ?? model.strategy_type
  const siblings = allModels.filter(m => m.strategy_type === model.strategy_type)
  if (siblings.length <= 1) return base
  const idx = siblings.findIndex(m => m.id === model.id)
  return `Variant ${idx + 1} of ${siblings.length}. ${base}`
}

function formatMoney(v: number) {
  const abs = Math.abs(v)
  const str = abs.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })
  return (v < 0 ? '-$' : '$') + str
}
function formatPct(v: number) { return (v >= 0 ? '+' : '') + v.toFixed(2) + '%' }
function cls(v: number) { return v > 0 ? 'pos' : v < 0 ? 'neg' : '' }

export default function App() {
  const { data, connected } = useWebSocket()
  const [starting, setStarting] = useState(false)
  const [formMins, setFormMins] = useState(390)
  const [activityOpen, setActivityOpen] = useState(true)

  const models = data?.models ?? []
  const trades = data?.trades ?? []
  const status = data?.arena_status
  const runState = data?.arena_run_state
  const isRunning = runState?.state === 'running'

  // Sort models by P&L descending
  const sorted = useMemo(() =>
    [...models].sort((a, b) =>
      (b.performance?.total_pnl ?? 0) - (a.performance?.total_pnl ?? 0)
    ), [models])

  // Portfolio aggregates
  const totals = useMemo(() => {
    let capital = 0, pnl = 0, unrealized = 0, posCount = 0
    for (const m of models) {
      capital += m.current_capital
      pnl += m.performance?.total_pnl ?? 0
      for (const p of m.positions) {
        unrealized += p.unrealized_pnl
        posCount++
      }
    }
    const initial = models.reduce((s, m) => s + m.initial_capital, 0)
    const returnPct = initial > 0 ? (pnl / initial) * 100 : 0
    return { capital, pnl, unrealized, posCount, returnPct }
  }, [models])

  // Recent trades (last 20, newest first)
  const recentTrades = useMemo(() =>
    [...trades]
      .sort((a, b) => (b.filled_at ?? '').localeCompare(a.filled_at ?? ''))
      .slice(0, 20),
    [trades])

  async function handleStart() {
    setStarting(true)
    try {
      await fetch('/api/arena/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ num_sessions: 1, session_minutes: formMins }),
      })
    } catch {}
    setStarting(false)
  }

  async function handleStop() {
    try { await fetch('/api/arena/stop', { method: 'POST' }) } catch {}
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <h1>Day Trader Arena</h1>
          <span className={`dot ${connected ? 'on' : 'off'}`} />
          <span className="conn-label">{connected ? 'Connected' : 'Disconnected'}</span>
        </div>
        <div className="header-right">
          {!isRunning ? (
            <div className="start-form">
              <input
                type="number" min={1} max={390} value={formMins}
                onChange={e => setFormMins(+e.target.value)}
                className="mins-input"
              />
              <span className="mins-label">min</span>
              <button onClick={handleStart} disabled={starting} className="btn btn-start">
                {starting ? 'Starting...' : 'Start Session'}
              </button>
            </div>
          ) : (
            <button onClick={handleStop} className="btn btn-stop">Stop</button>
          )}
        </div>
      </header>

      {/* Status bar */}
      {status && (
        <div className="status-bar">
          <span className={`phase ${status.phase}`}>{status.phase}</span>
          <span className="detail">{status.detail}</span>
          {status.total_bars && (
            <span className="bars">Bar {status.bar}/{status.total_bars}</span>
          )}
        </div>
      )}

      {/* Portfolio summary */}
      <div className="portfolio-strip">
        <div className="portfolio-stat">
          <span className="label">Portfolio</span>
          <span className="value">{formatMoney(totals.capital)}</span>
        </div>
        <div className="portfolio-stat">
          <span className="label">Realized P&L</span>
          <span className={`value ${cls(totals.pnl)}`}>{formatMoney(totals.pnl)}</span>
        </div>
        <div className="portfolio-stat">
          <span className="label">Unrealized</span>
          <span className={`value ${cls(totals.unrealized)}`}>{formatMoney(totals.unrealized)}</span>
        </div>
        <div className="portfolio-stat">
          <span className="label">Return</span>
          <span className={`value ${cls(totals.returnPct)}`}>{formatPct(totals.returnPct)}</span>
        </div>
        <div className="portfolio-stat">
          <span className="label">Open Positions</span>
          <span className="value">{totals.posCount}</span>
        </div>
      </div>

      {/* Model cards */}
      <div className="model-grid">
        {sorted.map(m => <ModelCard key={m.id} model={m} allModels={sorted} />)}
        {sorted.length === 0 && (
          <div className="empty">No models loaded</div>
        )}
      </div>

      {/* Activity feed */}
      <div className="activity-section">
        <button className="activity-toggle" onClick={() => setActivityOpen(!activityOpen)}>
          Recent Activity ({recentTrades.length}) {activityOpen ? '\u25B2' : '\u25BC'}
        </button>
        {activityOpen && (
          <div className="activity-feed">
            {recentTrades.map(t => <TradeRow key={t.id} trade={t} />)}
            {recentTrades.length === 0 && <div className="empty">No trades yet</div>}
          </div>
        )}
      </div>
    </div>
  )
}

const TODAY = new Date().toISOString().slice(0, 10)

function ModelCard({ model: m, allModels }: { model: Model; allModels: Model[] }) {
  const [showDesc, setShowDesc] = useState(false)
  const [showTrades, setShowTrades] = useState(false)
  const perf = m.performance
  const pnl = perf?.total_pnl ?? 0
  const returnPct = perf?.return_pct ?? 0
  const totalUnrealized = m.positions.reduce((s, p) => s + p.unrealized_pnl, 0)
  const equity = m.current_capital + totalUnrealized
  const desc = getModelDescription(m, allModels)
  const hasTrades = (perf?.total_trades ?? 0) > 0
  const { trades: closedTrades, loading: tradesLoading } = useModelTrades(
    showTrades ? m.id : null, showTrades ? TODAY : null
  )
  const sortedTrades = useMemo(() =>
    closedTrades
      .filter(t => t.status === 'closed')
      .sort((a, b) => (b.sell_time ?? '').localeCompare(a.sell_time ?? '')),
    [closedTrades]
  )

  return (
    <div className={`model-card ${cls(pnl)}-border`}>
      <div className="model-header">
        <div className="model-name">{m.name}</div>
        <span className="strategy-tag" onClick={() => setShowDesc(!showDesc)}>{m.strategy_type} {showDesc ? '\u25B2' : '\u25BC'}</span>
      </div>
      {showDesc && desc && <div className="strategy-desc">{desc}</div>}
      <div className="model-stats">
        <div className="stat-group">
          <span className="label">Equity</span>
          <span className="value">{formatMoney(equity)}</span>
        </div>
        <div className="stat-group">
          <span className="label">P&L</span>
          <span className={`value ${cls(pnl)}`}>
            {formatMoney(pnl)} ({formatPct(returnPct)})
          </span>
        </div>
        {totalUnrealized !== 0 && (
          <div className="stat-group">
            <span className="label">Unrealized</span>
            <span className={`value ${cls(totalUnrealized)}`}>{formatMoney(totalUnrealized)}</span>
          </div>
        )}
        <div className="stat-group">
          <span className="label">Trades</span>
          <span className="value">
            {perf?.total_trades ?? 0}
            {(perf?.win_rate ?? 0) > 0 && ` (${perf!.win_rate.toFixed(0)}% WR)`}
          </span>
        </div>
      </div>

      {m.positions.length > 0 && (
        <table className="positions-table">
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Qty</th>
              <th>Entry</th>
              <th>Current</th>
              <th>P&L</th>
            </tr>
          </thead>
          <tbody>
            {m.positions.map(p => (
              <tr key={p.symbol} className={cls(p.unrealized_pnl)}>
                <td className="sym">{p.symbol}</td>
                <td>{p.quantity.toFixed(2)}</td>
                <td>{formatMoney(p.avg_entry)}</td>
                <td>{formatMoney(p.current_price)}</td>
                <td className={cls(p.unrealized_pnl)}>{formatMoney(p.unrealized_pnl)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {!m.positions.length && !showTrades && (
        <div className="no-positions">No open positions</div>
      )}

      {hasTrades && (
        <button className="trades-toggle" onClick={() => setShowTrades(!showTrades)}>
          {showTrades ? 'Hide' : 'Show'} Closed Trades {showTrades ? '\u25B2' : '\u25BC'}
        </button>
      )}

      {showTrades && (
        tradesLoading ? (
          <div className="no-positions">Loading trades...</div>
        ) : sortedTrades.length > 0 ? (
          <table className="positions-table">
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Qty</th>
                <th>Buy</th>
                <th>Sell</th>
                <th>P&L</th>
              </tr>
            </thead>
            <tbody>
              {sortedTrades.map((t, i) => (
                <tr key={`${t.symbol}-${t.buy_time}-${i}`} className={cls(t.pnl)}>
                  <td className="sym">{t.symbol}</td>
                  <td>{t.quantity.toFixed(2)}</td>
                  <td>{formatMoney(t.buy_price)}</td>
                  <td>{formatMoney(t.sell_price!)}</td>
                  <td className={cls(t.pnl)}>{formatMoney(t.pnl)} ({t.pnl_pct >= 0 ? '+' : ''}{t.pnl_pct.toFixed(1)}%)</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="no-positions">No closed trades yet</div>
        )
      )}
    </div>
  )
}

function TradeRow({ trade: t }: { trade: Trade }) {
  const time = t.filled_at ? new Date(t.filled_at).toLocaleTimeString() : ''
  const isBuy = t.side === 'buy'
  return (
    <div className={`trade-row ${isBuy ? 'buy' : 'sell'}`}>
      <span className="time">{time}</span>
      <span className="model">{t.model_name}</span>
      <span className={`side ${t.side}`}>{t.side}</span>
      <span className="qty">{t.quantity.toFixed(2)}</span>
      <span className="sym">{t.symbol}</span>
      <span className="price">@ {formatMoney(t.price)}</span>
    </div>
  )
}
