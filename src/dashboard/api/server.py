"""FastAPI dashboard backend.

API serving model performance data, trade feed, rankings,
and generation history. Supports session-level performance queries
(session 1, session 2, day total). Pushes live updates via WebSocket.
Provides arena start/stop control endpoints.
"""

import asyncio
import json
import logging
import os
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.core.arena import Arena
from src.core.config import Config
from src.core.cfa_review import run_cfa_review
from src.core.database import (
    Bar,
    CfaReview,
    GenerationRecord,
    ModelStatus,
    ModelSummary,
    Order,
    OrderSide,
    OrderStatus,
    PerformanceSnapshot,
    Position,
    SessionRecord,
    TradingModel,
    get_dashboard_session,
    get_session,
    init_db,
)

# Configure logging so arena/feed/execution logs go to file + console
# (main.py does this via basicConfig but uvicorn doesn't)
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/arena.log"),
    ],
)

logger = logging.getLogger(__name__)

DB_PATH = "day_trader.db"
ARENA_STATUS_PATH = "logs/arena_status.json"
ARENA_LOG_PATH = "logs/arena.log"


class ConnectionManager:
    """Manages WebSocket connections for live updates."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, data: dict):
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
        while True:
            if self.active_connections:
                try:
                    arena_running = (
                        _arena_task is not None and not _arena_task.done()
                    )
                    if arena_running:
                        # Arena is running — use cached model data + live status
                        # (zero DB queries, never blocks)
                        status_data = get_arena_status()
                        base = dict(_dashboard_cache) if _dashboard_cache else {
                            "models": [], "trades": [], "generations": [], "sessions": [],
                        }
                        base["arena_status"] = status_data["status"]
                        base["arena_log"] = status_data.get("log", [])
                        base["arena_run_state"] = get_arena_run_state()
                        payload = {
                            "type": "update",
                            "data": base,
                        }
                    else:
                        # Arena idle — safe to query DB for full dashboard
                        payload = {
                            "type": "update",
                            "data": get_cached_dashboard_data(),
                        }
                    await self.broadcast(payload)
                except Exception:
                    logger.exception("Error in WebSocket broadcast")
            await asyncio.sleep(3)


manager = ConnectionManager()

# --- Dashboard data cache (serve stale to avoid blocking) ---
_dashboard_cache: dict | None = None
_dashboard_cache_time: float = 0.0
_DASHBOARD_CACHE_TTL: float = 2.0  # seconds


def get_cached_dashboard_data(session_date: str | None = None) -> dict:
    """Return cached dashboard data, refreshing if stale.

    Uses a short busy_timeout so queries fail fast when arena holds DB lock,
    falling back to stale cache instead of blocking the thread pool.
    TTL is 2s when idle, 15s when arena is running (to avoid DB contention).
    """
    import time
    global _dashboard_cache, _dashboard_cache_time

    arena_running = _arena_task is not None and not _arena_task.done()
    ttl = 15.0 if arena_running else _DASHBOARD_CACHE_TTL

    now = time.monotonic()
    if (
        _dashboard_cache is not None
        and session_date is None  # only cache today's data
        and (now - _dashboard_cache_time) < ttl
    ):
        return _dashboard_cache

    try:
        data = get_dashboard_data(session_date)
    except Exception:
        # DB locked or error — return stale cache
        if _dashboard_cache is not None and session_date is None:
            return _dashboard_cache
        raise

    if session_date is None:
        _dashboard_cache = data
        _dashboard_cache_time = now
    return data


# --- Arena run state ---
_arena_task: asyncio.Task | None = None
_arena_instance: Arena | None = None
_arena_config: dict | None = None  # {num_sessions, session_minutes}


class ArenaStartRequest(BaseModel):
    num_sessions: int = Field(default=2, ge=1, le=20)
    session_minutes: int = Field(default=60, ge=5, le=360)


def get_arena_run_state() -> dict:
    """Return current arena run state for the frontend."""
    global _arena_task, _arena_config
    if _arena_task is not None and not _arena_task.done():
        return {"state": "running", "config": _arena_config, "error": None}
    if _arena_task is not None and _arena_task.done():
        error = None
        try:
            exc = _arena_task.exception()
            if exc and not isinstance(exc, asyncio.CancelledError):
                error = str(exc)
        except (asyncio.CancelledError, asyncio.InvalidStateError):
            pass
        return {"state": "finished", "config": _arena_config, "error": error}
    return {"state": "idle", "config": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(DB_PATH)
    # Clear stale status from crashed runs
    try:
        os.makedirs("logs", exist_ok=True)
        with open(ARENA_STATUS_PATH, "w") as f:
            json.dump({"phase": "offline", "detail": "Dashboard started", "timestamp": datetime.now(ZoneInfo("America/New_York")).isoformat()}, f)
    except Exception:
        pass
    task = asyncio.create_task(manager.start_broadcasting())
    yield
    task.cancel()
    # Cancel arena if running
    if _arena_task and not _arena_task.done():
        _arena_task.cancel()


app = FastAPI(title="Day Trader Arena", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_arena_status() -> dict:
    """Read arena status file and tail recent log lines."""
    status = {"phase": "offline", "detail": "Arena not running", "timestamp": None}
    try:
        if os.path.exists(ARENA_STATUS_PATH):
            with open(ARENA_STATUS_PATH) as f:
                status = json.load(f)
    except Exception:
        pass

    log_lines: list[str] = []
    try:
        if os.path.exists(ARENA_LOG_PATH):
            with open(ARENA_LOG_PATH, "rb") as f:
                # Read last ~8KB for recent lines
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 8192))
                tail = f.read().decode("utf-8", errors="replace")
                log_lines = tail.strip().split("\n")[-50:]
    except Exception:
        pass

    return {"status": status, "log": log_lines}


def get_dashboard_data(session_date: str | None = None) -> dict:
    """Assemble complete dashboard state from DB, scoped to a date (default today)."""
    today = session_date or datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    db = get_dashboard_session(DB_PATH)
    try:
        models = (
            db.query(TradingModel)
            .filter(TradingModel.status == ModelStatus.ACTIVE)
            .all()
        )

        model_data = []
        for model in models:
            snapshot = (
                db.query(PerformanceSnapshot)
                .filter(
                    PerformanceSnapshot.model_id == model.id,
                    PerformanceSnapshot.session_date == today,
                )
                .order_by(PerformanceSnapshot.timestamp.desc())
                .first()
            )

            positions = (
                db.query(Position)
                .filter(Position.model_id == model.id)
                .all()
            )

            snapshots = (
                db.query(PerformanceSnapshot)
                .filter(
                    PerformanceSnapshot.model_id == model.id,
                    PerformanceSnapshot.session_date == today,
                )
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
                    "session_number": snapshot.session_number if snapshot else None,
                } if snapshot else None,
                "equity_curve": [
                    {
                        "timestamp": s.timestamp.isoformat(),
                        "equity": s.equity,
                        "session_number": s.session_number,
                    }
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

        model_data.sort(
            key=lambda m: m["performance"]["return_pct"] if m.get("performance") else 0,
            reverse=True,
        )
        for i, m in enumerate(model_data):
            m["rank"] = i + 1

        # Recent trades (last 50, today only)
        recent_orders = (
            db.query(Order)
            .filter(
                Order.status == OrderStatus.FILLED,
                Order.session_date == today,
            )
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
                "session_number": order.session_number,
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

        # Current sessions (today only)
        sessions = (
            db.query(SessionRecord)
            .filter(SessionRecord.session_date == today)
            .order_by(SessionRecord.session_number.asc())
            .all()
        )

        session_data = [
            {
                "date": s.session_date,
                "session_number": s.session_number,
                "generation": s.generation,
                "started_at": s.started_at.isoformat() if s.started_at else None,
                "ended_at": s.ended_at.isoformat() if s.ended_at else None,
                "total_bars": s.total_bars,
                "total_trades": s.total_trades,
            }
            for s in sessions
        ]

        arena = get_arena_status()

        return {
            "models": model_data,
            "trades": trades,
            "generations": gen_history,
            "sessions": session_data,
            "arena_status": arena["status"],
            "arena_log": arena["log"],
            "arena_run_state": get_arena_run_state(),
        }

    finally:
        db.close()


_api_executor = None


def _get_api_executor():
    """Lazy-init a single-thread executor for API DB queries."""
    global _api_executor
    if _api_executor is None:
        import concurrent.futures
        _api_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="dashboard-api"
        )
    return _api_executor


@app.get("/api/dashboard")
async def dashboard(session_date: Optional[str] = None):
    """Get current dashboard state, optionally for a specific date."""
    # If arena is running and no specific date requested, skip DB entirely
    # (same approach the WebSocket broadcast already uses)
    arena_running = _arena_task is not None and not _arena_task.done()
    if arena_running and session_date is None:
        status = get_arena_status()
        base = _dashboard_cache or {
            "models": [], "trades": [], "generations": [], "sessions": [],
        }
        base["arena_status"] = status["status"]
        base["arena_log"] = status.get("log", [])
        base["arena_run_state"] = get_arena_run_state()
        return base

    try:
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(
                _get_api_executor(),
                get_cached_dashboard_data,
                session_date,
            ),
            timeout=3.0,
        )
    except asyncio.TimeoutError:
        if _dashboard_cache is not None and session_date is None:
            return _dashboard_cache
        return {"models": [], "trades": [], "generations": [], "sessions": [],
                "arena_status": get_arena_status()["status"], "arena_log": [],
                "arena_run_state": get_arena_run_state()}


@app.get("/api/arena-status")
async def arena_status():
    """Get current arena status and recent log lines."""
    return get_arena_status()


@app.get("/api/models")
async def list_models():
    """List all models (active and eliminated)."""
    db = get_dashboard_session(DB_PATH)
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
async def model_equity(model_id: int, session_number: Optional[int] = None):
    """Get equity curve for a specific model, optionally filtered by session."""
    db = get_dashboard_session(DB_PATH)
    try:
        query = db.query(PerformanceSnapshot).filter(
            PerformanceSnapshot.model_id == model_id
        )
        if session_number is not None:
            query = query.filter(PerformanceSnapshot.session_number == session_number)

        snapshots = query.order_by(PerformanceSnapshot.timestamp.asc()).all()
        return [
            {
                "timestamp": s.timestamp.isoformat(),
                "equity": s.equity,
                "session_number": s.session_number,
            }
            for s in snapshots
        ]
    finally:
        db.close()


@app.get("/api/generations")
async def list_generations():
    """Get generation history."""
    db = get_dashboard_session(DB_PATH)
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
async def list_trades(limit: int = 100, session_number: Optional[int] = None, session_date: Optional[str] = None):
    """Get recent trades, optionally filtered by session number and/or date."""
    db = get_dashboard_session(DB_PATH)
    try:
        query = db.query(Order).filter(Order.status == OrderStatus.FILLED)
        if session_date is not None:
            query = query.filter(Order.session_date == session_date)
        if session_number is not None:
            query = query.filter(Order.session_number == session_number)

        orders = query.order_by(Order.filled_at.desc()).limit(limit).all()
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
                "session_number": o.session_number,
                "filled_at": o.filled_at.isoformat() if o.filled_at else None,
            })
        return result
    finally:
        db.close()


@app.get("/api/sessions")
async def list_sessions():
    """Get all past trading sessions with summaries."""
    db = get_dashboard_session(DB_PATH)
    try:
        sessions = db.query(SessionRecord).order_by(SessionRecord.id.desc()).all()
        return [
            {
                "id": s.id,
                "date": s.session_date,
                "session_number": s.session_number,
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


@app.get("/api/sessions/dates")
async def session_dates():
    """Get distinct dates that have session data (for playback date picker)."""
    db = get_dashboard_session(DB_PATH)
    try:
        dates = (
            db.query(SessionRecord.session_date)
            .distinct()
            .order_by(SessionRecord.session_date.desc())
            .all()
        )
        return [d[0] for d in dates]
    finally:
        db.close()


@app.get("/api/sessions/{session_date}/performance")
async def session_performance(session_date: str, session_number: Optional[int] = None):
    """Get all performance snapshots for a session (timelapse data)."""
    db = get_dashboard_session(DB_PATH)
    try:
        query = db.query(PerformanceSnapshot).filter(
            PerformanceSnapshot.session_date == session_date
        )
        if session_number is not None:
            query = query.filter(PerformanceSnapshot.session_number == session_number)

        snapshots = query.order_by(PerformanceSnapshot.timestamp.asc()).all()

        by_model: dict = {}
        for s in snapshots:
            model = db.query(TradingModel).get(s.model_id)
            name = model.name if model else f"model_{s.model_id}"
            if name not in by_model:
                by_model[name] = {
                    "model_id": s.model_id,
                    "name": name,
                    "strategy_type": model.strategy_type if model else "",
                    "points": [],
                }
            by_model[name]["points"].append({
                "timestamp": s.timestamp.isoformat(),
                "equity": s.equity,
                "return_pct": s.return_pct,
                "sharpe": s.sharpe_ratio,
                "drawdown": s.max_drawdown,
                "trades": s.total_trades,
                "win_rate": s.win_rate,
                "session_number": s.session_number,
            })
        return list(by_model.values())
    finally:
        db.close()


@app.get("/api/model-summaries/{session_date}")
async def model_summaries(session_date: str, session_number: Optional[int] = None, summary_type: Optional[str] = None):
    """Get model reflections/summaries for a session date."""
    db = get_dashboard_session(DB_PATH)
    try:
        query = db.query(ModelSummary).filter(
            ModelSummary.session_date == session_date
        )
        if session_number is not None:
            query = query.filter(ModelSummary.session_number == session_number)
        if summary_type is not None:
            query = query.filter(ModelSummary.summary_type == summary_type)

        summaries = query.order_by(ModelSummary.rank.asc()).all()

        result = []
        for s in summaries:
            model = db.query(TradingModel).get(s.model_id)
            result.append({
                "model_id": s.model_id,
                "model_name": model.name if model else f"model_{s.model_id}",
                "strategy_type": model.strategy_type if model else "",
                "session_number": s.session_number,
                "summary_type": s.summary_type,
                "return_pct": s.return_pct,
                "sharpe_ratio": s.sharpe_ratio,
                "max_drawdown": s.max_drawdown,
                "total_trades": s.total_trades,
                "win_rate": s.win_rate,
                "fitness": s.fitness,
                "rank": s.rank,
                "param_changes": s.param_changes,
                "reflection": s.reflection,
            })
        return result
    finally:
        db.close()


@app.get("/api/models/{model_id}/trades")
async def model_trades(model_id: int, session_date: Optional[str] = None):
    """Get round-trip trades (buy->sell pairs) for a model on a given date."""
    if session_date is None:
        session_date = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

    db = get_dashboard_session(DB_PATH)
    try:
        orders = (
            db.query(Order)
            .filter(
                Order.model_id == model_id,
                Order.session_date == session_date,
                Order.status == OrderStatus.FILLED,
            )
            .order_by(Order.filled_at.asc())
            .all()
        )

        # Group orders by symbol, then pair buys with sells chronologically
        from collections import defaultdict
        buys_by_symbol: dict[str, list] = defaultdict(list)
        result = []

        for o in orders:
            if o.side == OrderSide.BUY:
                buys_by_symbol[o.symbol].append(o)
            elif o.side == OrderSide.SELL:
                qty_to_match = o.fill_quantity or o.quantity
                sell_price = o.fill_price
                sell_time = o.filled_at

                # Match against queued buys for this symbol (FIFO)
                while qty_to_match > 0 and buys_by_symbol[o.symbol]:
                    buy = buys_by_symbol[o.symbol][0]
                    buy_qty = buy.fill_quantity or buy.quantity
                    matched = min(qty_to_match, buy_qty)

                    pnl = (sell_price - buy.fill_price) * matched - (
                        buy.transaction_cost + o.transaction_cost
                    ) * (matched / (o.fill_quantity or o.quantity))
                    pnl_pct = ((sell_price / buy.fill_price) - 1) * 100 if buy.fill_price else 0

                    result.append({
                        "symbol": o.symbol,
                        "buy_price": buy.fill_price,
                        "buy_time": buy.filled_at.isoformat() if buy.filled_at else None,
                        "sell_price": sell_price,
                        "sell_time": sell_time.isoformat() if sell_time else None,
                        "quantity": matched,
                        "pnl": round(pnl, 4),
                        "pnl_pct": round(pnl_pct, 4),
                        "status": "closed",
                    })

                    qty_to_match -= matched
                    buy_remaining = buy_qty - matched
                    if buy_remaining <= 0:
                        buys_by_symbol[o.symbol].pop(0)
                    else:
                        # Partially matched - update remaining qty on the buy
                        buy.fill_quantity = buy_remaining

        # Remaining unmatched buys are open positions
        for symbol, buys in buys_by_symbol.items():
            for buy in buys:
                buy_qty = buy.fill_quantity or buy.quantity
                # Get current price from Position table if available
                pos = (
                    db.query(Position)
                    .filter(Position.model_id == model_id, Position.symbol == symbol)
                    .first()
                )
                current_price = pos.current_price if pos and pos.current_price else buy.fill_price
                pnl = (current_price - buy.fill_price) * buy_qty
                pnl_pct = ((current_price / buy.fill_price) - 1) * 100 if buy.fill_price else 0

                result.append({
                    "symbol": symbol,
                    "buy_price": buy.fill_price,
                    "buy_time": buy.filled_at.isoformat() if buy.filled_at else None,
                    "sell_price": None,
                    "sell_time": None,
                    "quantity": buy_qty,
                    "pnl": round(pnl, 4),
                    "pnl_pct": round(pnl_pct, 4),
                    "status": "open",
                })

        return result
    finally:
        db.close()


@app.get("/api/arena/state")
async def arena_state():
    """Get current arena run state (idle/running/finished)."""
    return get_arena_run_state()


@app.post("/api/arena/start")
async def arena_start(req: ArenaStartRequest):
    """Start the arena with custom session configuration."""
    global _arena_task, _arena_instance, _arena_config

    # Guard against double-start
    if _arena_task is not None and not _arena_task.done():
        raise HTTPException(status_code=409, detail="Arena is already running")

    try:
        config = Config.load()
        _arena_instance = Arena(config)
        _arena_config = {
            "num_sessions": req.num_sessions,
            "session_minutes": req.session_minutes,
        }
        _arena_task = asyncio.create_task(
            _arena_instance.run_custom(req.num_sessions, req.session_minutes)
        )
        logger.info(
            f"Arena started: {req.num_sessions} sessions x {req.session_minutes} min"
        )
        return {
            "status": "started",
            "config": _arena_config,
        }
    except Exception as e:
        logger.exception("Failed to start arena")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/arena/stop")
async def arena_stop():
    """Stop the running arena."""
    global _arena_task, _arena_instance

    if _arena_task is None or _arena_task.done():
        return {"status": "not_running"}

    _arena_task.cancel()
    try:
        await asyncio.wait_for(_arena_task, timeout=10.0)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass

    logger.info("Arena stopped via API")
    return {"status": "stopped"}


@app.get("/api/cfa-review/{session_date}")
async def get_cfa_review(session_date: str):
    """Retrieve a saved CFA review for a given date."""
    db = get_dashboard_session(DB_PATH)
    try:
        review = (
            db.query(CfaReview)
            .filter(CfaReview.session_date == session_date)
            .first()
        )
        if not review:
            raise HTTPException(status_code=404, detail="No review found for this date")
        return {
            "session_date": review.session_date,
            "review": review.review_json,
            "model_used": review.model_used,
            "created_at": review.created_at.isoformat() if review.created_at else None,
        }
    finally:
        db.close()


@app.post("/api/cfa-review/{session_date}/generate")
async def generate_cfa_review(session_date: str):
    """Manually trigger or re-run a CFA review for a given date."""
    config = Config.load()
    try:
        loop = asyncio.get_event_loop()
        review = await asyncio.wait_for(
            loop.run_in_executor(
                _get_api_executor(),
                run_cfa_review,
                config.db_path,
                session_date,
                config.arena.cfa_review_model,
                config.arena.cfa_review_timeout_sec,
                config.arena.cfa_review_lookback_days,
            ),
            timeout=float(config.arena.cfa_review_timeout_sec + 30),
        )
        if review is None:
            raise HTTPException(
                status_code=500,
                detail="Review generation failed (check logs for details)",
            )
        return {
            "session_date": session_date,
            "review": review,
            "status": "generated",
        }
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Review generation timed out")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("CFA review generation failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for live dashboard updates."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
