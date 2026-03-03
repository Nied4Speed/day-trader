import { useState, useEffect, useRef, useCallback } from 'react'

export interface Model {
  id: number; name: string; strategy_type: string; generation: number;
  rank: number; parent_ids: number[] | null; genetic_operation: string | null;
  initial_capital: number; current_capital: number;
  performance: { equity: number; total_pnl: number; return_pct: number; sharpe_ratio: number; max_drawdown: number; win_rate: number; total_trades: number } | null;
  equity_curve: { timestamp: string; equity: number }[];
  positions: { symbol: string; quantity: number; avg_entry: number; current_price: number; unrealized_pnl: number }[];
}

export interface Trade {
  id: number; model_id: number; model_name: string; symbol: string;
  side: string; quantity: number; price: number; transaction_cost: number; filled_at: string | null;
}

export interface Generation {
  generation: number; session_date: string; model_count: number;
  eliminated_count: number; best_fitness: number | null; avg_fitness: number | null;
}

export interface DashboardData {
  models: Model[]; trades: Trade[]; generations: Generation[];
  session: { date: string; generation: number; started_at: string | null; total_bars: number; total_trades: number } | null;
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
    // Initial fetch via REST
    fetch('/api/dashboard').then(r => r.json()).then(setData).catch(() => {})
    connect()
    return () => { wsRef.current?.close() }
  }, [connect])

  return { data, connected }
}
