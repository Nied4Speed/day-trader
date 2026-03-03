"""Moving Average Crossover Strategy.

Buys when fast MA crosses above slow MA, sells when it crosses below.
Classic trend-following approach.
"""

from typing import Optional

import pandas as pd

from src.core.strategy import BarData, Strategy, TradeSignal


class MACrossoverStrategy(Strategy):
    strategy_type = "ma_crossover"

    def __init__(self, name: str, params: Optional[dict] = None):
        self.fast_period: int = 10
        self.slow_period: int = 30
        self.position_size: int = 1
        self._prev_fast: dict[str, float] = {}
        self._prev_slow: dict[str, float] = {}
        super().__init__(name, params)

    def on_bar(self, bar: BarData) -> Optional[TradeSignal]:
        self.record_bar(bar)
        closes = self.get_close_series(bar.symbol)

        if len(closes) < self.slow_period:
            return None

        fast_ma = closes.rolling(self.fast_period).mean().iloc[-1]
        slow_ma = closes.rolling(self.slow_period).mean().iloc[-1]

        prev_fast = self._prev_fast.get(bar.symbol)
        prev_slow = self._prev_slow.get(bar.symbol)

        self._prev_fast[bar.symbol] = fast_ma
        self._prev_slow[bar.symbol] = slow_ma

        if prev_fast is None or prev_slow is None:
            return None

        # Golden cross: fast crosses above slow
        if prev_fast <= prev_slow and fast_ma > slow_ma:
            return TradeSignal(
                symbol=bar.symbol,
                side="buy",
                quantity=self.position_size,
            )

        # Death cross: fast crosses below slow
        if prev_fast >= prev_slow and fast_ma < slow_ma:
            return TradeSignal(
                symbol=bar.symbol,
                side="sell",
                quantity=self.position_size,
            )

        return None

    def get_params(self) -> dict:
        return {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "position_size": self.position_size,
        }

    def set_params(self, params: dict) -> None:
        self.fast_period = max(2, int(params.get("fast_period", self.fast_period)))
        self.slow_period = max(
            self.fast_period + 1,
            int(params.get("slow_period", self.slow_period)),
        )
        self.position_size = max(1, int(params.get("position_size", self.position_size)))

    def reset(self) -> None:
        super().reset()
        self._prev_fast.clear()
        self._prev_slow.clear()
