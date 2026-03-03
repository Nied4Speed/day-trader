"""Momentum Strategy.

Buys when price shows strong upward momentum (rate of change above threshold),
sells when momentum reverses. Uses lookback period to smooth noise.
"""

from typing import Optional

import pandas as pd

from src.core.strategy import BarData, Strategy, TradeSignal


class MomentumStrategy(Strategy):
    strategy_type = "momentum"

    def __init__(self, name: str, params: Optional[dict] = None):
        self.lookback: int = 20
        self.buy_threshold: float = 0.02  # 2% momentum to buy
        self.sell_threshold: float = -0.01  # -1% momentum to sell
        self.position_size: int = 10
        self.volume_filter: float = 1.5  # volume must be 1.5x average
        super().__init__(name, params)

    def on_bar(self, bar: BarData) -> Optional[TradeSignal]:
        self.record_bar(bar)
        history = self.get_history(bar.symbol)

        if len(history) < self.lookback + 1:
            return None

        closes = [b.close for b in history]
        volumes = [b.volume for b in history]

        # Rate of change over lookback period
        roc = (closes[-1] - closes[-self.lookback]) / closes[-self.lookback]

        # Volume filter: current volume vs average
        avg_volume = sum(volumes[-self.lookback:]) / self.lookback
        volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 0

        if roc > self.buy_threshold and volume_ratio > self.volume_filter:
            return TradeSignal(
                symbol=bar.symbol,
                side="buy",
                quantity=self.position_size,
            )

        if roc < self.sell_threshold:
            return TradeSignal(
                symbol=bar.symbol,
                side="sell",
                quantity=self.position_size,
            )

        return None

    def get_params(self) -> dict:
        return {
            "lookback": self.lookback,
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
            "position_size": self.position_size,
            "volume_filter": self.volume_filter,
        }

    def set_params(self, params: dict) -> None:
        self.lookback = max(5, int(params.get("lookback", self.lookback)))
        self.buy_threshold = max(0.001, float(params.get("buy_threshold", self.buy_threshold)))
        self.sell_threshold = min(-0.001, float(params.get("sell_threshold", self.sell_threshold)))
        self.position_size = max(1, int(params.get("position_size", self.position_size)))
        self.volume_filter = max(0.5, float(params.get("volume_filter", self.volume_filter)))
