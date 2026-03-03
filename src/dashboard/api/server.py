"""FastAPI dashboard backend.

Read-only API serving model performance data, trade feed, rankings,
and generation history. Pushes live updates via WebSocket.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.core.database import (
    Bar,
    GenerationRecord,
    ModelStatus,
    Order,
    OrderStatus,
    PerformanceSnapshot,
    Position,
    SessionRecord,
    TradingModel,
    get_session,
    init_db,
)

logger = logging.getLogger(__name__)

DB_PATH = "day_trader.db"


class ConnectionManager:
    """Manages WebSocket connections for live updates."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._broadcast_task: Optional[asyncio.Task] = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, data: dict):
        """Send data to all connected clients."""
        message = json.dumps(data, default=str)
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            self.active_connections.remove(conn)

    async def start_broadcasting(self):
        """Periodically push updates to all connected clients."""
        while True:
            if self.active_connections:
                data = get_dashboard_data()
                await self.broadcast({"type": "update", "data": data})
            await asyncio.sleep(1)


manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(DB_PATH)
    task = asyncio.create_task(manager.start_broadcasting())
    yield
    task.cancel()


app = FastAPI(title="Day Trader Arena", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_dashboard_data() -> dict:
    """Assemble complete dashboard state from DB."""
    db = get_session(DB_PATH)
    try:
        # Active models with latest performance
        models = (
            db.query(TradingModel)
            .filter(TradingModel.status == ModelStatus.ACTIVE)
            .all()
        )

        model_data = []
        for model in models:
            # Get latest performance snapshot
            snapshot = (
                db.query(PerformanceSnapshot)
                .filter(PerformanceSnapshot.model_id == model.id)
                .order_by(PerformanceSnapshot.timestamp.desc())
                .first()
            )

            # Get positions
            positions = (
                db.query(Position)
                .filter(Position.model_id == model.id)
                .all()
            )

            # Get equity curve from all snapshots
            snapshots = (
                db.query(PerformanceSnapshot)
                .filter(PerformanceSnapshot.model_id == model.id)
                .order_by(PerformanceSnapshot.timestamp.asc())
                .all()
            )

            model_data.append({
                "id": model.id,
                "name": model.name,
                "strategy_type": model.strategy_type,
                "generation": model.generation,
                "parent_ids": model.parent_ids,
                "genetic_operation": model.genetic_operation,
                "initial_capital": model.initial_capital,
                "current_capital": model.current_capital,
                "performance": {
                    "equity": snapshot.equity if snapshot else model.current_capital,
                    "total_pnl": snapshot.total_pnl if snapshot else 0,
                    "return_pct": snapshot.return_pct if snapshot else 0,
                    "sharpe_ratio": snapshot.sharpe_ratio if snapshot else 0,
                    "max_drawdown": snapshot.max_drawdown if snapshot else 0,
                    "win_rate": snapshot.win_rate if snapshot else 0,
                    "total_trades": snapshot.total_trades if snapshot else 0,
                } if snapshot else None,
                "equity_curve": [
                    {"timestamp": s.timestamp.isoformat(), "equity": s.equity}
                    for s in snapshots
                ],
                "positions": [
                    {
                        "symbol": p.symbol,
                        "quantity": p.quantity,
                        "avg_entry": p.avg_entry_price,
                        "current_price": p.current_price,
                        "unrealized_pnl": p.unrealized_pnl,
                    }
                    for p in positions
                    if p.quantity != 0
                ],
            })

        # Sort by performance (return %)
        model_data.sort(
            key=lambda m: m["performance"]["return_pct"] if m.get("performance") else 0,
            reverse=True,
        )
        for i, m in enumerate(model_data):
            m["rank"] = i + 1

        # Recent trades (last 50)
        recent_orders = (
            db.query(Order)
            .filter(Order.status == OrderStatus.FILLED)
            .order_by(Order.filled_at.desc())
            .limit(50)
            .all()
        )

        trades = []
        for order in recent_orders:
            model = db.query(TradingModel).get(order.model_id)
            trades.append({
                "id": order.id,
                "model_id": order.model_id,
                "model_name": model.name if model else "Unknown",
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": order.fill_quantity or order.quantity,
                "price": order.fill_price,
                "transaction_cost": order.transaction_cost,
                "filled_at": order.filled_at.isoformat() if order.filled_at else None,
            })

        # Generation history
        generations = (
            db.query(GenerationRecord)
            .order_by(GenerationRecord.generation_number.desc())
            .all()
        )

        gen_history = [
            {
                "generation": g.generation_number,
                "session_date": g.session_date,
                "model_count": len(g.model_ids),
                "eliminated_count": len(g.eliminated_ids),
                "best_fitness": g.best_fitness,
                "avg_fitness": g.avg_fitness,
            }
            for g in generations
        ]

        # Current session
        session = (
            db.query(SessionRecord)
            .order_by(SessionRecord.id.desc())
            .first()
        )

        return {
            "models": model_data,
            "trades": trades,
            "generations": gen_history,
            "session": {
                "date": session.session_date if session else None,
                "generation": session.generation if session else 1,
                "started_at": session.started_at.isoformat() if session and session.started_at else None,
                "total_bars": session.total_bars if session else 0,
                "total_trades": session.total_trades if session else 0,
            } if session else None,
        }

    finally:
        db.close()


@app.get("/api/dashboard")
async def dashboard():
    """Get current dashboard state."""
    return get_dashboard_data()


@app.get("/api/models")
async def list_models():
    """List all models (active and eliminated)."""
    db = get_session(DB_PATH)
    try:
        models = db.query(TradingModel).all()
        return [
            {
                "id": m.id,
                "name": m.name,
                "strategy_type": m.strategy_type,
                "generation": m.generation,
                "status": m.status.value,
                "parent_ids": m.parent_ids,
                "genetic_operation": m.genetic_operation,
                "initial_capital": m.initial_capital,
                "current_capital": m.current_capital,
                "created_at": m.created_at.isoformat(),
                "eliminated_at": m.eliminated_at.isoformat() if m.eliminated_at else None,
            }
            for m in models
        ]
    finally:
        db.close()


@app.get("/api/models/{model_id}/equity")
async def model_equity(model_id: int):
    """Get equity curve for a specific model."""
    db = get_session(DB_PATH)
    try:
        snapshots = (
            db.query(PerformanceSnapshot)
            .filter(PerformanceSnapshot.model_id == model_id)
            .order_by(PerformanceSnapshot.timestamp.asc())
            .all()
        )
        return [
            {"timestamp": s.timestamp.isoformat(), "equity": s.equity}
            for s in snapshots
        ]
    finally:
        db.close()


@app.get("/api/generations")
async def list_generations():
    """Get generation history."""
    db = get_session(DB_PATH)
    try:
        gens = (
            db.query(GenerationRecord)
            .order_by(GenerationRecord.generation_number.desc())
            .all()
        )
        return [
            {
                "generation": g.generation_number,
                "session_date": g.session_date,
                "model_ids": g.model_ids,
                "eliminated_ids": g.eliminated_ids,
                "survivor_ids": g.survivor_ids,
                "offspring_ids": g.offspring_ids,
                "best_fitness": g.best_fitness,
                "avg_fitness": g.avg_fitness,
            }
            for g in gens
        ]
    finally:
        db.close()


@app.get("/api/trades")
async def list_trades(limit: int = 100):
    """Get recent trades."""
    db = get_session(DB_PATH)
    try:
        orders = (
            db.query(Order)
            .filter(Order.status == OrderStatus.FILLED)
            .order_by(Order.filled_at.desc())
            .limit(limit)
            .all()
        )
        result = []
        for o in orders:
            model = db.query(TradingModel).get(o.model_id)
            result.append({
                "id": o.id,
                "model_name": model.name if model else "Unknown",
                "symbol": o.symbol,
                "side": o.side.value,
                "quantity": o.fill_quantity,
                "price": o.fill_price,
                "cost": o.transaction_cost,
                "filled_at": o.filled_at.isoformat() if o.filled_at else None,
            })
        return result
    finally:
        db.close()


@app.get("/api/sessions")
async def list_sessions():
    """Get all past trading sessions with summaries."""
    db = get_session(DB_PATH)
    try:
        sessions = db.query(SessionRecord).order_by(SessionRecord.id.desc()).all()
        return [
            {
                "id": s.id,
                "date": s.session_date,
                "generation": s.generation,
                "started_at": s.started_at.isoformat() if s.started_at else None,
                "ended_at": s.ended_at.isoformat() if s.ended_at else None,
                "total_bars": s.total_bars,
                "total_trades": s.total_trades,
                "summary": s.summary,
            }
            for s in sessions
        ]
    finally:
        db.close()


@app.get("/api/sessions/{session_date}/performance")
async def session_performance(session_date: str):
    """Get all performance snapshots for a session (timelapse data)."""
    db = get_session(DB_PATH)
    try:
        snapshots = (
            db.query(PerformanceSnapshot)
            .filter(PerformanceSnapshot.session_date == session_date)
            .order_by(PerformanceSnapshot.timestamp.asc())
            .all()
        )
        by_model: dict = {}
        for s in snapshots:
            model = db.query(TradingModel).get(s.model_id)
            name = model.name if model else f"model_{s.model_id}"
            if name not in by_model:
                by_model[name] = {"model_id": s.model_id, "name": name, "strategy_type": model.strategy_type if model else "", "points": []}
            by_model[name]["points"].append({
                "timestamp": s.timestamp.isoformat(),
                "equity": s.equity,
                "return_pct": s.return_pct,
                "sharpe": s.sharpe_ratio,
                "drawdown": s.max_drawdown,
                "trades": s.total_trades,
            })
        return list(by_model.values())
    finally:
        db.close()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for live dashboard updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, handle client messages if any
            data = await websocket.receive_text()
            # Could handle client commands here (e.g., subscribe to specific model)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
