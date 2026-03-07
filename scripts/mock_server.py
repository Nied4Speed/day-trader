"""Mock WebSocket server that simulates a full trading day for dashboard testing.

Usage: .venv/bin/python scripts/mock_server.py
Then open http://localhost:5173/ to watch the simulated day unfold.

Simulates 12 models trading 10 symbols over ~2 minutes (compressed from 6.5 hours).
Models open/close positions, P&L fluctuates, trades stream in.
"""

import asyncio
import json
import math
import random
import time
from datetime import datetime, timedelta

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Simulation config ---
SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "SPY"]
BASE_PRICES = {
    "AAPL": 258, "MSFT": 420, "GOOGL": 298, "AMZN": 217, "NVDA": 180,
    "META": 652, "TSLA": 401, "JPM": 294, "V": 319, "SPY": 677,
}
MODELS = [
    (12, "ma_crossover_gen1_1", "ma_crossover"),
    (13, "rsi_reversion_gen1_2", "rsi_reversion"),
    (14, "momentum_gen1_3", "momentum"),
    (15, "bollinger_bands_gen1_4", "bollinger_bands"),
    (17, "macd_gen1_6", "macd"),
    (18, "vwap_reversion_gen1_7", "vwap_reversion"),
    (20, "breakout_gen1_9", "breakout"),
    (21, "mean_reversion_gen1_10", "mean_reversion"),
    (23, "ma_crossover_gen1_12", "ma_crossover"),
    (24, "rsi_reversion_gen1_13", "rsi_reversion"),
    (26, "bollinger_bands_gen1_15", "bollinger_bands"),
    (28, "macd_gen1_17", "macd"),
]

TOTAL_TICKS = 240  # ~2 minutes at 0.5s per tick
TICK_INTERVAL = 0.5
TOTAL_BARS = 780  # simulated bar count for a full day


class SimState:
    def __init__(self):
        self.tick = 0
        self.prices = {s: float(p) for s, p in BASE_PRICES.items()}
        self.models: dict[int, dict] = {}
        self.trades: list[dict] = []
        self.trade_id = 0
        self.phase = "warmup"
        self.sim_time = datetime(2026, 3, 7, 9, 30, 0)

        for mid, name, stype in MODELS:
            self.models[mid] = {
                "id": mid, "name": name, "strategy_type": stype,
                "initial_capital": 1000.0, "current_capital": 1000.0,
                "realized_pnl": 0.0,
                "positions": {},  # symbol -> {qty, entry}
                "total_trades": 0, "wins": 0,
            }

    def advance(self):
        self.tick += 1
        progress = self.tick / TOTAL_TICKS

        # Update phase
        if progress < 0.02:
            self.phase = "warmup"
        elif progress < 0.05:
            self.phase = "premarket"
        elif progress < 0.95:
            self.phase = "session"
        else:
            self.phase = "complete"

        # Advance simulated clock (~2.4 min per tick to cover 9:30-4:00)
        self.sim_time += timedelta(seconds=97.5)

        # Move prices with random walk + slight drift
        for sym in SYMBOLS:
            pct = random.gauss(0.0001, 0.003)  # slight upward bias
            # Add some trending behavior
            if sym in ("NVDA", "TSLA"):
                pct += random.gauss(0.0003, 0.002)  # more volatile, slight up
            elif sym in ("SPY", "V"):
                pct += random.gauss(0.0001, 0.001)  # steadier
            self.prices[sym] *= (1 + pct)

        if self.phase != "session":
            return

        # Each model might trade
        for mid, m in self.models.items():
            self._maybe_trade(m)

    def _maybe_trade(self, m: dict):
        # Entry: ~15% chance per tick if has capital and < 3 positions
        if len(m["positions"]) < 3 and m["current_capital"] > 100:
            if random.random() < 0.15:
                sym = random.choice([s for s in SYMBOLS if s not in m["positions"]])
                price = self.prices[sym]
                alloc = m["current_capital"] * random.uniform(0.15, 0.30)
                qty = round(alloc / price, 4)
                if qty >= 0.01:
                    cost = qty * price
                    m["current_capital"] -= cost
                    m["positions"][sym] = {"qty": qty, "entry": price}
                    m["total_trades"] += 1
                    self._add_trade(m, sym, "buy", qty, price)

        # Exit: check each position
        for sym in list(m["positions"].keys()):
            pos = m["positions"][sym]
            price = self.prices[sym]
            pnl_pct = (price - pos["entry"]) / pos["entry"]

            # Take profit > 1.5%, stop loss > -1%, or random exit ~5%
            should_exit = (
                pnl_pct > 0.015 or
                pnl_pct < -0.01 or
                random.random() < 0.05
            )
            if should_exit:
                proceeds = pos["qty"] * price
                realized = proceeds - (pos["qty"] * pos["entry"])
                m["current_capital"] += proceeds
                m["realized_pnl"] += realized
                if realized > 0:
                    m["wins"] += 1
                m["total_trades"] += 1
                self._add_trade(m, sym, "sell", pos["qty"], price)
                del m["positions"][sym]

    def _add_trade(self, m: dict, sym: str, side: str, qty: float, price: float):
        self.trade_id += 1
        self.trades.append({
            "id": self.trade_id,
            "model_id": m["id"],
            "model_name": m["name"],
            "symbol": sym,
            "side": side,
            "quantity": round(qty, 4),
            "price": round(price, 2),
            "transaction_cost": 0,
            "session_number": 1,
            "filled_at": self.sim_time.isoformat(),
        })
        # Keep last 100 trades
        if len(self.trades) > 100:
            self.trades = self.trades[-100:]

    def to_dashboard(self) -> dict:
        models = []
        for mid, m in self.models.items():
            positions = []
            total_unrealized = 0.0
            for sym, pos in m["positions"].items():
                cur = self.prices[sym]
                unr = (cur - pos["entry"]) * pos["qty"]
                total_unrealized += unr
                positions.append({
                    "symbol": sym,
                    "quantity": round(pos["qty"], 4),
                    "avg_entry": round(pos["entry"], 2),
                    "current_price": round(cur, 2),
                    "unrealized_pnl": round(unr, 2),
                })

            equity = m["current_capital"] + total_unrealized
            total_pnl = m["realized_pnl"]
            return_pct = (total_pnl / m["initial_capital"]) * 100

            wr = (m["wins"] / m["total_trades"] * 100) if m["total_trades"] > 0 else 0

            models.append({
                "id": mid,
                "name": m["name"],
                "strategy_type": m["strategy_type"],
                "generation": 1,
                "rank": 0,
                "parent_ids": None,
                "genetic_operation": None,
                "initial_capital": m["initial_capital"],
                "current_capital": round(m["current_capital"], 2),
                "performance": {
                    "equity": round(equity, 2),
                    "total_pnl": round(total_pnl, 2),
                    "return_pct": round(return_pct, 4),
                    "sharpe_ratio": 0,
                    "max_drawdown": 0,
                    "win_rate": round(wr, 1),
                    "total_trades": m["total_trades"],
                    "session_number": 1,
                },
                "equity_curve": [],
                "positions": positions,
            })

        # Rank by P&L
        models.sort(key=lambda x: x["performance"]["total_pnl"], reverse=True)
        for i, m in enumerate(models):
            m["rank"] = i + 1

        bar = min(int((self.tick / TOTAL_TICKS) * TOTAL_BARS), TOTAL_BARS)

        return {
            "models": models,
            "trades": self.trades[-50:],
            "generations": [],
            "sessions": [{"date": "2026-03-07", "session_number": 1, "generation": 1,
                          "started_at": "2026-03-07T09:30:00", "ended_at": None,
                          "total_bars": bar, "total_trades": self.trade_id}],
            "arena_status": {
                "phase": self.phase,
                "detail": f"Simulated trading day ({self.sim_time.strftime('%I:%M %p')})",
                "session_number": 1,
                "bar": bar,
                "total_bars": TOTAL_BARS,
                "timestamp": datetime.now().isoformat(),
            },
            "arena_log": [],
            "arena_run_state": {
                "state": "running" if self.phase == "session" else ("idle" if self.tick == 0 else "finished" if self.phase == "complete" else "running"),
                "config": {"num_sessions": 1, "session_minutes": 390},
                "error": None,
            },
        }


sim = SimState()
connections: list[WebSocket] = []


@app.get("/api/dashboard")
async def dashboard():
    return JSONResponse(sim.to_dashboard())


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connections.remove(websocket)


async def broadcast_loop():
    """Advance simulation and broadcast every 0.5s."""
    await asyncio.sleep(1)  # let server start
    print(f"\n  Simulating a full trading day in ~{TOTAL_TICKS * TICK_INTERVAL:.0f}s")
    print(f"  Open http://localhost:5173/ to watch\n")

    while sim.tick < TOTAL_TICKS:
        sim.advance()
        data = sim.to_dashboard()
        msg = json.dumps({"type": "update", "data": data})

        for ws in list(connections):
            try:
                await ws.send_text(msg)
            except Exception:
                connections.remove(ws)

        pct = sim.tick / TOTAL_TICKS * 100
        bar_vis = "=" * int(pct / 2) + ">" + " " * (50 - int(pct / 2))
        total_pnl = sum(m["realized_pnl"] for m in sim.models.values())
        open_pos = sum(len(m["positions"]) for m in sim.models.values())
        print(f"\r  [{bar_vis}] {pct:5.1f}%  {sim.sim_time.strftime('%I:%M %p')}  "
              f"P&L: ${total_pnl:+.2f}  Positions: {open_pos}", end="", flush=True)

        await asyncio.sleep(TICK_INTERVAL)

    # Final broadcast
    sim.phase = "complete"
    data = sim.to_dashboard()
    msg = json.dumps({"type": "update", "data": data})
    for ws in list(connections):
        try:
            await ws.send_text(msg)
        except Exception:
            pass

    total_pnl = sum(m["realized_pnl"] for m in sim.models.values())
    print(f"\n\n  Simulation complete! Final P&L: ${total_pnl:+.2f}")
    print(f"  Total trades: {sim.trade_id}")
    print(f"  Dashboard will stay up — Ctrl+C to stop\n")


@app.on_event("startup")
async def startup():
    asyncio.create_task(broadcast_loop())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
