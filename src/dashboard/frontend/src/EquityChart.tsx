import { useEffect, useRef } from 'react'
import { createChart, IChartApi, ISeriesApi } from 'lightweight-charts'

export interface EquityPoint {
  time: number  // UNIX seconds
  value: number
}

export function EquityChart({ data }: { data: EquityPoint[] }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const seriesRef = useRef<ISeriesApi<'Line'> | null>(null)

  // Create chart once
  useEffect(() => {
    if (!containerRef.current) return
    const chart = createChart(containerRef.current, {
      height: 200,
      layout: {
        background: { color: '#1a2332' },
        textColor: '#9ca3af',
        fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
        fontSize: 10,
      },
      grid: {
        vertLines: { color: 'rgba(42, 52, 66, 0.5)' },
        horzLines: { color: 'rgba(42, 52, 66, 0.5)' },
      },
      rightPriceScale: {
        borderColor: '#2a3442',
      },
      timeScale: {
        borderColor: '#2a3442',
        timeVisible: true,
        secondsVisible: false,
      },
      crosshair: {
        horzLine: { color: '#3b82f6', labelBackgroundColor: '#3b82f6' },
        vertLine: { color: '#3b82f6', labelBackgroundColor: '#3b82f6' },
      },
    })
    const series = chart.addLineSeries({
      color: '#3b82f6',
      lineWidth: 2,
      priceFormat: { type: 'custom', formatter: (p: number) => '$' + p.toFixed(0) },
    })
    chartRef.current = chart
    seriesRef.current = series

    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        chart.applyOptions({ width: entry.contentRect.width })
      }
    })
    ro.observe(containerRef.current)

    return () => {
      ro.disconnect()
      chart.remove()
      chartRef.current = null
      seriesRef.current = null
    }
  }, [])

  // Update data
  useEffect(() => {
    if (!seriesRef.current || data.length === 0) return
    seriesRef.current.setData(data as any)
    chartRef.current?.timeScale().fitContent()
  }, [data])

  return <div ref={containerRef} className="equity-chart-inner" />
}
