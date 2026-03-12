"""Real-time performance tracking and leaderboard.

Computes per-model metrics (P&L, Sharpe, drawdown, win rate) as bars arrive,
and maintains a ranked leaderboard of all active models.
Supports per-session tracking (session 1, session 2, day total).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from src.core.database import (
    Order,
    OrderStatus,
    PerformanceSnapshot,
    Position,
    TradingModel,
    get_session,
)
from src.core.fitness import FitnessScore, compute_fitness

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Live metrics for a single model."""

    model_id: int
    model_name: str
    strategy_type: str
    equity: float = 0.0
    total_pnl: float = 0.0
    return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    equity_curve: list[float] = field(default_factory=list)
    trade_returns: list[float] = field(default_factory=list)
    rank: int = 0
    fitness: Optional[FitnessScore] = None


class PerformanceTracker:
    """Tracks real-time performance for all active models."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._metrics: dict[int, ModelMetrics] = {}
        self._session_number: int = 1
        self._session_date: Optional[str] = None

    def set_session_number(self, session_number: int) -> None:
        """Set which session we're tracking (1 or 2)."""
        self._session_number = session_number

    def set_session_date(self, session_date: str) -> None:
        """Set the session date to scope order queries."""
        self._session_date = session_date

    def initialize_models(self, models: list[TradingModel]) -> None:
        """Set up tracking for a list of models at session start."""
        self._metrics.clear()
        for model in models:
            self._metrics[model.id] = ModelMetrics(
                model_id=model.id,
                model_name=model.name,
                strategy_type=model.strategy_type,
                equity=model.current_capital,
                equity_curve=[model.current_capital],
            )

    def set_last_prices(self, prices: dict[str, float]) -> None:
        """Update latest bar prices for unrealized P&L calculation."""
        self._last_prices = prices

    def update(self, model_id: int) -> Optional[ModelMetrics]:
        """Recompute metrics for a model based on current DB state."""
        if model_id not in self._metrics:
            return None

        metrics = self._metrics[model_id]
        db = get_session(self.db_path)

        try:
            model = db.query(TradingModel).get(model_id)
            if not model:
                return None

            # Get current positions and compute unrealized P&L from latest prices
            positions = (
                db.query(Position)
                .filter(Position.model_id == model_id)
                .all()
            )
            last_prices = getattr(self, '_last_prices', {})
            unrealized = 0.0
            for p in positions:
                if p.quantity > 0 and p.avg_entry_price > 0:
                    price = last_prices.get(p.symbol, p.current_price)
                    unrealized += (price - p.avg_entry_price) * p.quantity

            # Trade stats from filled orders (scoped to today's date, cumulative across sessions)
            order_query = db.query(Order).filter(
                Order.model_id == model_id,
                Order.status == OrderStatus.FILLED,
            )
            if self._session_date:
                order_query = order_query.filter(Order.session_date == self._session_date)
            filled_orders = order_query.all()
            metrics.total_trades = len(filled_orders)

            # Realized P&L from filled sell orders (not from positions table,
            # which can be stale/phantom). Each sell order has realized_pnl
            # computed at fill time from actual Alpaca execution prices.
            sell_orders = [o for o in filled_orders if o.side.value == "sell"]
            realized = sum((o.realized_pnl or 0.0) for o in sell_orders)

            # Compute equity (capital + position cost basis + unrealized)
            position_cost = sum(
                p.avg_entry_price * p.quantity for p in positions if p.quantity > 0
            )
            equity = model.current_capital + position_cost + unrealized
            metrics.equity = equity
            metrics.equity_curve.append(equity)

            # Total P&L
            metrics.total_pnl = realized + unrealized
            if model.initial_capital > 0:
                metrics.return_pct = metrics.total_pnl / model.initial_capital * 100

            # Compute per-trade returns from individual sell order P&L
            if sell_orders:
                metrics.trade_returns = [
                    (o.realized_pnl or 0.0) / model.initial_capital
                    for o in sell_orders
                ]
                metrics.winning_trades = sum(
                    1 for o in sell_orders if (o.realized_pnl or 0) > 0
                )
                metrics.win_rate = metrics.winning_trades / len(sell_orders)

            # Compute fitness score
            session_bars = len(metrics.equity_curve)
            metrics.fitness = compute_fitness(
                equity_curve=metrics.equity_curve,
                trade_returns=metrics.trade_returns,
                trade_count=metrics.total_trades,
                session_bars=session_bars,
                initial_capital=model.initial_capital,
            )
            metrics.sharpe_ratio = metrics.fitness.sharpe_ratio
            metrics.max_drawdown = metrics.fitness.max_drawdown

            return metrics

        finally:
            db.close()

    def update_all(self) -> list[ModelMetrics]:
        """Update metrics for all tracked models and compute rankings."""
        results = []
        for model_id in self._metrics:
            m = self.update(model_id)
            if m:
                results.append(m)

        # Rank by fitness composite score
        results.sort(
            key=lambda m: m.fitness.composite if m.fitness else 0,
            reverse=True,
        )
        for i, m in enumerate(results):
            m.rank = i + 1
            self._metrics[m.model_id].rank = i + 1

        return results

    def get_leaderboard(self) -> list[ModelMetrics]:
        """Return current rankings without recomputing."""
        ranked = sorted(
            self._metrics.values(),
            key=lambda m: m.fitness.composite if m.fitness else 0,
            reverse=True,
        )
        return ranked

    def get_model_metrics(self, model_id: int) -> Optional[ModelMetrics]:
        """Get metrics for a specific model."""
        return self._metrics.get(model_id)

    def save_snapshots(self, session_date: str, session_number: int = 1) -> None:
        """Persist current performance snapshots to DB."""
        db = get_session(self.db_path)
        now = datetime.utcnow()
        try:
            for metrics in self._metrics.values():
                snapshot = PerformanceSnapshot(
                    model_id=metrics.model_id,
                    session_date=session_date,
                    session_number=session_number,
                    timestamp=now,
                    equity=metrics.equity,
                    total_pnl=metrics.total_pnl,
                    return_pct=metrics.return_pct,
                    sharpe_ratio=metrics.sharpe_ratio,
                    max_drawdown=metrics.max_drawdown,
                    win_rate=metrics.win_rate,
                    total_trades=metrics.total_trades,
                    winning_trades=metrics.winning_trades,
                )
                db.add(snapshot)
            db.commit()
        except Exception:
            db.rollback()
            logger.exception("Failed to save performance snapshots")
        finally:
            db.close()

    def generate_session_summary(self) -> dict:
        """Generate end-of-session summary."""
        leaderboard = self.get_leaderboard()
        return {
            "model_count": len(leaderboard),
            "rankings": [
                {
                    "rank": m.rank,
                    "model_id": m.model_id,
                    "name": m.model_name,
                    "strategy_type": m.strategy_type,
                    "return_pct": round(m.return_pct, 4),
                    "sharpe": round(m.sharpe_ratio, 4),
                    "max_drawdown": round(m.max_drawdown, 6),
                    "trades": m.total_trades,
                    "fitness": round(m.fitness.composite, 4) if m.fitness else 0,
                }
                for m in leaderboard
            ],
            "best_model": leaderboard[0].model_name if leaderboard else None,
            "worst_model": leaderboard[-1].model_name if leaderboard else None,
        }
