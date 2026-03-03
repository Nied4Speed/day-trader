import { useWebSocket, Model, Trade, Generation } from './hooks'

function formatPct(v: number) { return (v >= 0 ? '+' : '') + v.toFixed(4) + '%' }
function formatNum(v: number, d = 2) { return v.toFixed(d) }
function cls(v: number) { return v > 0 ? 'positive' : v < 0 ? 'negative' : 'neutral' }

function Rankings({ models }: { models: Model[] }) {
  return (
    <table className="rankings-table">
      <thead><tr>
        <th>#</th><th>Model</th><th>Type</th><th>Gen</th>
        <th>Return</th><th>Sharpe</th><th>Drawdown</th><th>Trades</th><th>Win Rate</th>
      </tr></thead>
      <tbody>
        {models.map(m => {
          const p = m.performance
          return (
            <tr key={m.id}>
              <td>{m.rank}</td>
              <td>{m.name}</td>
              <td>{m.strategy_type}</td>
              <td>{m.generation}</td>
              <td className={cls(p?.return_pct ?? 0)}>{p ? formatPct(p.return_pct) : '-'}</td>
              <td>{p ? formatNum(p.sharpe_ratio, 3) : '-'}</td>
              <td>{p ? formatPct(-p.max_drawdown * 100) : '-'}</td>
              <td>{p?.total_trades ?? 0}</td>
              <td>{p ? formatNum(p.win_rate * 100, 1) + '%' : '-'}</td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

function TradeFeed({ trades }: { trades: Trade[] }) {
  return (
    <div className="trade-feed">
      {trades.length === 0 && <div className="neutral" style={{padding:'1rem',textAlign:'center'}}>No trades yet</div>}
      {trades.map(t => (
        <div className="trade-item" key={t.id}>
          <span><span className={`trade-side ${t.side}`}>{t.side.toUpperCase()}</span> {t.quantity} {t.symbol}</span>
          <span>{t.model_name}</span>
          <span>${t.price?.toFixed(2) ?? '-'}</span>
          <span style={{color:'var(--text-muted)'}}>{t.filled_at ? new Date(t.filled_at).toLocaleTimeString() : ''}</span>
        </div>
      ))}
    </div>
  )
}

function GenerationHistory({ generations }: { generations: Generation[] }) {
  return (
    <div>
      {generations.length === 0 && <div className="neutral" style={{padding:'1rem',textAlign:'center'}}>No generations yet</div>}
      {generations.map(g => (
        <div className="gen-item" key={g.generation}>
          <span className="gen-label">Gen {g.generation}</span>
          <span>{g.model_count} models</span>
          <span className="negative">{g.eliminated_count} eliminated</span>
          <span>Best: {g.best_fitness?.toFixed(4) ?? '-'}</span>
          <span>Avg: {g.avg_fitness?.toFixed(4) ?? '-'}</span>
        </div>
      ))}
    </div>
  )
}

export default function App() {
  const { data, connected } = useWebSocket()
  const models = data?.models ?? []
  const trades = data?.trades ?? []
  const generations = data?.generations ?? []
  const session = data?.session

  return (
    <div className="app">
      <div className="header">
        <h1>Day Trader Arena</h1>
        <div className="status">
          <div className={`status-dot ${connected ? '' : 'disconnected'}`} />
          {connected ? 'Live' : 'Disconnected'}
          {session && <> | Gen {session.generation} | {session.total_bars} bars | {session.total_trades} trades</>}
        </div>
      </div>

      <div className="grid">
        <div className="card full-width">
          <h2>Model Rankings</h2>
          <Rankings models={models} />
        </div>

        <div className="card">
          <h2>Trade Feed</h2>
          <TradeFeed trades={trades} />
        </div>

        <div className="card">
          <h2>Generation History</h2>
          <GenerationHistory generations={generations} />
        </div>

        <div className="card full-width">
          <h2>Equity Curves</h2>
          <div className="chart-container" style={{display:'flex',alignItems:'center',justifyContent:'center',color:'var(--text-muted)'}}>
            {models.length > 0 && models.some(m => m.equity_curve.length > 0)
              ? <EquityChart models={models} />
              : 'Equity curves will appear once trading begins'}
          </div>
        </div>
      </div>
    </div>
  )
}

function EquityChart({ models }: { models: Model[] }) {
  const colors = ['#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6','#ec4899','#06b6d4','#f97316','#14b8a6','#6366f1']
  return (
    <div style={{width:'100%',padding:'1rem'}}>
      <svg viewBox="0 0 800 250" style={{width:'100%',height:'250px'}}>
        {models.filter(m => m.equity_curve.length > 1).map((m, mi) => {
          const pts = m.equity_curve
          const minE = Math.min(...pts.map(p => p.equity)) * 0.999
          const maxE = Math.max(...pts.map(p => p.equity)) * 1.001
          const rangeE = maxE - minE || 1
          const path = pts.map((p, i) => {
            const x = (i / (pts.length - 1)) * 780 + 10
            const y = 240 - ((p.equity - minE) / rangeE) * 220
            return `${i === 0 ? 'M' : 'L'}${x},${y}`
          }).join(' ')
          return <path key={m.id} d={path} fill="none" stroke={colors[mi % colors.length]} strokeWidth="1.5" opacity="0.8" />
        })}
      </svg>
      <div style={{display:'flex',gap:'1rem',flexWrap:'wrap',fontSize:'0.7rem',marginTop:'0.5rem'}}>
        {models.filter(m => m.equity_curve.length > 1).map((m, mi) => (
          <span key={m.id} style={{color:colors[mi % colors.length]}}>&#9632; {m.name}</span>
        ))}
      </div>
    </div>
  )
}
