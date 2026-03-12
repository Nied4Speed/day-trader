"""Moving Average Crossover Strategy.

Buys when fast MA crosses above slow MA, sells when it crosses below.
Classic trend-following approach.
"""

import logging
from typing import Optional

import pandas as pd

from src.core.strategy import BarData, Strategy, TradeSignal

logger = logging.getLogger(__name__)


class MACrossoverStrategy(Strategy):
    strategy_type = "ma_crossover"

    def __init__(self, name: str, params: Optional[dict] = None):
        self.fast_period: int = 10
        self.slow_period: int = 30
        self.allocation_pct: float = 0.30  # 30% of capital per trade
        self._prev_fast: dict[str, float] = {}
        self._prev_slow: dict[str, float] = {}
        super().__init__(name, params)

    def on_bar(self, bar: BarData) -> Optional[TradeSignal]:
        self.record_bar(bar)
        liq = self.check_liquidation(bar)
        if liq:
            return liq if liq.quantity > 0 else None
        closes = self.get_close_series(bar.symbol)

        if len(closes) < self.slow_period:
            return None

        fast_ma = closes.rolling(self.fast_period).mean().iloc[-1]
        slow_ma = closes.rolling(self.slow_period).mean().iloc[-1]

        # Cache indicators for watch rule evaluation
        ma_spread = (fast_ma - slow_ma) / slow_ma if slow_ma != 0 else 0.0
        self._indicators[bar.symbol] = {
            "close": bar.close,
            "fast_ma": fast_ma,
            "slow_ma": slow_ma,
            "ma_spread": ma_spread,
        }

        prev_fast = self._prev_fast.get(bar.symbol)
        prev_slow = self._prev_slow.get(bar.symbol)

        self._prev_fast[bar.symbol] = fast_ma
        self._prev_slow[bar.symbol] = slow_ma

        if prev_fast is None or prev_slow is None:
            return None

        # Golden cross: fast crosses above slow
        if prev_fast <= prev_slow and fast_ma > slow_ma:
            qty = self.compute_quantity(bar.close, self.allocation_pct, symbol=bar.symbol)
            if qty > 0:
                return TradeSignal(symbol=bar.symbol, side="buy", quantity=qty)

        # Death cross: fast crosses below slow
        if prev_fast >= prev_slow and fast_ma < slow_ma:
            qty = self.compute_quantity(bar.close, self.allocation_pct)
            if qty > 0:
                return TradeSignal(symbol=bar.symbol, side="sell", quantity=qty)

        return None

    def get_params(self) -> dict:
        return {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "allocation_pct": self.allocation_pct,
        }

    def set_params(self, params: dict) -> None:
        self.fast_period = max(2, int(params.get("fast_period", self.fast_period)))
        self.slow_period = max(
            self.fast_period + 1,
            int(params.get("slow_period", self.slow_period)),
        )
        self.allocation_pct = max(0.05, min(1.0, float(params.get("allocation_pct", self.allocation_pct))))

    def adapt(self, recent_signals: list, recent_fills: list, realized_pnl: float) -> None:
        old_fast, old_slow = self.fast_period, self.slow_period
        if realized_pnl < 0:
            self.fast_period += 1
            self.slow_period += 1
        elif realized_pnl > 0:
            self.fast_period -= 1
        self.fast_period = max(3, min(20, self.fast_period))
        self.slow_period = max(15, min(50, self.slow_period))
        # Ensure slow always > fast
        if self.slow_period <= self.fast_period:
            self.slow_period = self.fast_period + 1
        logger.info(
            f"{self.name} adapt: fast_period {old_fast}->{self.fast_period}, "
            f"slow_period {old_slow}->{self.slow_period}, pnl={realized_pnl:.2f}"
        )

    def reset(self) -> None:
        super().reset()
        self._prev_fast.clear()
        self._prev_slow.clear()
