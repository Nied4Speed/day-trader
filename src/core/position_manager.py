"""Central Position Manager: coordinates all model signals into net orders.

Instead of 20 models independently submitting orders (causing wash trades),
all models produce TradeSignal objects which flow here. The PositionManager:

1. Collects signals per symbol per bar
2. Resolves conflicts (net out buy vs sell signals, weighted by signal quality)
3. Submits one order per symbol per bar (eliminates wash trades)
4. Distributes fill P&L proportionally to contributing models
5. Enforces portfolio-level risk limits
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from src.core.strategy import TradeSignal

logger = logging.getLogger(__name__)


@dataclass
class ModelSignal:
    """A signal tagged with the model that produced it."""
    model_id: int
    signal: TradeSignal
    capital: float  # model's current capital at signal time


@dataclass
class RollingScore:
    """Exponential moving average score tracking signal quality."""
    hits: int = 0
    misses: int = 0
    ema: float = 0.5  # start neutral
    alpha: float = 0.15  # EMA smoothing factor

    def record(self, win: bool) -> None:
        val = 1.0 if win else 0.0
        self.ema = self.alpha * val + (1 - self.alpha) * self.ema
        if win:
            self.hits += 1
        else:
            self.misses += 1

    @property
    def score(self) -> float:
        return self.ema


@dataclass
class ResolvedOrder:
    """A net order to submit after conflict resolution."""
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    contributing_models: list[tuple[int, float]]  # (model_id, weight)
    net_strength: float  # 0-1, how strong the consensus is


class PositionManager:
    """Coordinates signals from all models into net orders per symbol."""

    MIN_SIGNAL_STRENGTH = 0.3  # minimum net strength to execute

    def __init__(self):
        self._signal_scores: dict[int, RollingScore] = {}  # model_id -> score
        self._symbol_locks: set[str] = set()  # symbols with pending orders
        self._pending_signals: dict[str, list[ModelSignal]] = {}  # symbol -> signals awaiting P&L

        # Portfolio risk limits
        self.max_exposure_per_symbol_pct = 0.15  # 15% max in one stock
        self.max_total_exposure_pct = 0.80  # 80% max invested
        self.drawdown_reduce_threshold = 0.03  # 3% drawdown -> reduce size 50%
        self.drawdown_stop_threshold = 0.05  # 5% drawdown -> stop new positions

        # Track portfolio state
        self._total_portfolio_value: float = 0.0
        self._symbol_exposure: dict[str, float] = {}  # symbol -> dollar exposure
        self._session_start_value: float = 0.0
        self._current_drawdown: float = 0.0

        # Correlation groups for concentration guard
        self._correlation_groups: dict[str, str] = {
            "AAPL": "tech", "MSFT": "tech", "GOOGL": "tech",
            "AMZN": "tech", "NVDA": "tech", "META": "tech", "TSLA": "tech",
            "JPM": "finance", "V": "finance",
            "SPY": "index",
        }
        self._group_exposure: dict[str, float] = {}

    def get_signal_score(self, model_id: int) -> float:
        """Get a model's current signal quality score (0-1)."""
        if model_id not in self._signal_scores:
            self._signal_scores[model_id] = RollingScore()
        return self._signal_scores[model_id].score

    def record_signal_outcome(self, model_id: int, win: bool) -> None:
        """Record whether a model's signal resulted in a profitable trade."""
        if model_id not in self._signal_scores:
            self._signal_scores[model_id] = RollingScore()
        self._signal_scores[model_id].record(win)

    def update_portfolio_state(
        self,
        total_value: float,
        symbol_exposure: dict[str, float],
    ) -> None:
        """Update current portfolio state for risk checks."""
        self._total_portfolio_value = total_value
        self._symbol_exposure = symbol_exposure

        # Compute group exposure
        self._group_exposure.clear()
        for sym, dollars in symbol_exposure.items():
            group = self._correlation_groups.get(sym, sym)
            self._group_exposure[group] = self._group_exposure.get(group, 0.0) + dollars

        # Update drawdown
        if self._session_start_value > 0:
            self._current_drawdown = max(
                0.0,
                (self._session_start_value - total_value) / self._session_start_value,
            )

    def start_session(self, total_value: float) -> None:
        """Reset session-level state."""
        self._session_start_value = total_value
        self._total_portfolio_value = total_value
        self._current_drawdown = 0.0
        self._symbol_locks.clear()
        self._pending_signals.clear()
        self._symbol_exposure.clear()
        self._group_exposure.clear()

    def is_symbol_locked(self, symbol: str) -> bool:
        return symbol in self._symbol_locks

    def lock_symbol(self, symbol: str) -> None:
        self._symbol_locks.add(symbol)

    def unlock_symbol(self, symbol: str) -> None:
        self._symbol_locks.discard(symbol)

    def resolve(
        self,
        signals: list[ModelSignal],
        regime_weights: Optional[dict[str, dict[str, float]]] = None,
    ) -> list[ResolvedOrder]:
        """Resolve conflicting signals into net orders per symbol.

        Args:
            signals: All model signals for this bar
            regime_weights: Optional per-symbol regime-based weight adjustments
                {symbol: {strategy_type: weight_multiplier}}

        Returns:
            List of resolved orders to submit
        """
        if not signals:
            return []

        # Group by symbol
        by_symbol: dict[str, list[ModelSignal]] = defaultdict(list)
        for ms in signals:
            if ms.signal.symbol == "__SKIP__":
                continue
            by_symbol[ms.signal.symbol].append(ms)

        resolved: list[ResolvedOrder] = []

        for symbol, sym_signals in by_symbol.items():
            # Skip locked symbols
            if self.is_symbol_locked(symbol):
                logger.debug(f"Skipping {symbol}: symbol locked (pending order)")
                continue

            # Check portfolio risk limits before processing
            order = self._resolve_symbol(symbol, sym_signals)
            if order:
                # Apply portfolio risk limits
                order = self._apply_risk_limits(order)
                if order:
                    resolved.append(order)

        return resolved

    def _resolve_symbol(
        self,
        symbol: str,
        signals: list[ModelSignal],
    ) -> Optional[ResolvedOrder]:
        """Resolve signals for a single symbol into a net order."""
        buy_weight = 0.0
        sell_weight = 0.0
        buy_models: list[tuple[int, float]] = []
        sell_models: list[tuple[int, float]] = []
        total_buy_qty = 0.0
        total_sell_qty = 0.0

        for ms in signals:
            score = self.get_signal_score(ms.model_id)
            weight = score

            if ms.signal.side == "buy":
                buy_weight += weight
                buy_models.append((ms.model_id, weight))
                total_buy_qty += ms.signal.quantity
            else:
                sell_weight += weight
                sell_models.append((ms.model_id, weight))
                total_sell_qty += ms.signal.quantity

        total_weight = buy_weight + sell_weight
        if total_weight == 0:
            return None

        # Determine net direction
        if buy_weight > sell_weight:
            net_strength = (buy_weight - sell_weight) / total_weight
            if net_strength < self.MIN_SIGNAL_STRENGTH:
                logger.debug(
                    f"{symbol}: net BUY strength {net_strength:.2f} below threshold"
                )
                return None

            # Weighted average quantity from buy signals
            avg_qty = round(total_buy_qty / len(buy_models), 4)
            return ResolvedOrder(
                symbol=symbol,
                side="buy",
                quantity=avg_qty,
                contributing_models=buy_models,
                net_strength=net_strength,
            )
        elif sell_weight > buy_weight:
            net_strength = (sell_weight - buy_weight) / total_weight
            if net_strength < self.MIN_SIGNAL_STRENGTH:
                logger.debug(
                    f"{symbol}: net SELL strength {net_strength:.2f} below threshold"
                )
                return None

            avg_qty = round(total_sell_qty / len(sell_models), 4)
            return ResolvedOrder(
                symbol=symbol,
                side="sell",
                quantity=avg_qty,
                contributing_models=sell_models,
                net_strength=net_strength,
            )

        return None  # Perfectly balanced = no action

    def _apply_risk_limits(self, order: ResolvedOrder) -> Optional[ResolvedOrder]:
        """Apply portfolio-level risk limits to a resolved order."""
        if self._total_portfolio_value <= 0:
            return order  # Can't compute limits without portfolio value

        # Circuit breaker: stop new buys if drawdown > 5%
        if order.side == "buy" and self._current_drawdown >= self.drawdown_stop_threshold:
            logger.warning(
                f"Circuit breaker: blocking BUY {order.symbol} "
                f"(drawdown {self._current_drawdown:.1%} >= {self.drawdown_stop_threshold:.0%})"
            )
            return None

        # Reduce size if drawdown > 3%
        if self._current_drawdown >= self.drawdown_reduce_threshold:
            order.quantity = max(0.01, order.quantity / 2)
            logger.info(
                f"Drawdown {self._current_drawdown:.1%}: reduced {order.symbol} "
                f"order to {order.quantity} shares"
            )

        if order.side == "buy":
            # Max exposure per symbol (15%)
            current_exposure = self._symbol_exposure.get(order.symbol, 0.0)
            max_dollars = self._total_portfolio_value * self.max_exposure_per_symbol_pct
            remaining = max_dollars - current_exposure
            if remaining <= 0:
                logger.debug(
                    f"Max exposure reached for {order.symbol}: "
                    f"${current_exposure:.0f} >= ${max_dollars:.0f}"
                )
                return None

            # Max total exposure (80%)
            total_invested = sum(self._symbol_exposure.values())
            max_total = self._total_portfolio_value * self.max_total_exposure_pct
            if total_invested >= max_total:
                logger.debug(
                    f"Max total exposure reached: "
                    f"${total_invested:.0f} >= ${max_total:.0f}"
                )
                return None

            # Correlation guard: reduce weight if same sector already heavy
            group = self._correlation_groups.get(order.symbol, order.symbol)
            group_exp = self._group_exposure.get(group, 0.0)
            # If sector already > 30% of portfolio, halve the order
            sector_limit = self._total_portfolio_value * 0.30
            if group_exp > sector_limit:
                order.quantity = max(0.01, order.quantity / 2)
                logger.debug(
                    f"Correlation guard: {group} sector at ${group_exp:.0f}, "
                    f"reduced {order.symbol} to {order.quantity}"
                )

        return order

    def distribute_fill(
        self,
        symbol: str,
        side: str,
        fill_price: float,
        fill_qty: float,
        contributing_models: list[tuple[int, float]],
    ) -> dict[int, float]:
        """Distribute fill cost/proceeds proportionally to contributing models.

        Returns: dict of model_id -> dollar amount to debit (buy) or credit (sell)
        """
        total_weight = sum(w for _, w in contributing_models)
        if total_weight == 0:
            return {}

        distribution: dict[int, float] = {}
        total_value = fill_price * fill_qty

        for model_id, weight in contributing_models:
            share = weight / total_weight
            distribution[model_id] = total_value * share

        return distribution

    def get_scores_summary(self) -> dict[int, dict]:
        """Get signal quality scores for all models."""
        return {
            mid: {
                "score": s.score,
                "hits": s.hits,
                "misses": s.misses,
            }
            for mid, s in self._signal_scores.items()
        }
