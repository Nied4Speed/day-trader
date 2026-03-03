"""MACD (Moving Average Convergence Divergence) Strategy.

Buys on MACD line crossing above signal line (bullish),
sells on MACD crossing below signal (bearish).
Uses histogram divergence for confirmation.
"""

from typing import Optional

import pandas as pd

from src.core.strategy import BarData, Strategy, TradeSignal


class MACDStrategy(Strategy):
    strategy_type = "macd"

    def __init__(self, name: str, params: Optional[dict] = None):
        self.fast_period: int = 12
        self.slow_period: int = 26
        self.signal_period: int = 9
        self.position_size: int = 1
        self.histogram_threshold: float = 0.0
        self._prev_histogram: dict[str, float] = {}
        super().__init__(name, params)

    def on_bar(self, bar: BarData) -> Optional[TradeSignal]:
        self.record_bar(bar)
        closes = self.get_close_series(bar.symbol)

        min_required = self.slow_period + self.signal_period
        if len(closes) < min_required:
            return None

        # Compute MACD
        fast_ema = closes.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = closes.ewm(span=self.slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        current_hist = histogram.iloc[-1]
        prev_hist = self._prev_histogram.get(bar.symbol)
        self._prev_histogram[bar.symbol] = current_hist

        if prev_hist is None:
            return None

        # Buy: histogram crosses above zero (MACD crosses above signal)
        if prev_hist <= self.histogram_threshold and current_hist > self.histogram_threshold:
            return TradeSignal(
                symbol=bar.symbol,
                side="buy",
                quantity=self.position_size,
            )

        # Sell: histogram crosses below zero (MACD crosses below signal)
        if prev_hist >= -self.histogram_threshold and current_hist < -self.histogram_threshold:
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
            "signal_period": self.signal_period,
            "position_size": self.position_size,
            "histogram_threshold": self.histogram_threshold,
        }

    def set_params(self, params: dict) -> None:
        self.fast_period = max(2, int(params.get("fast_period", self.fast_period)))
        self.slow_period = max(
            self.fast_period + 1,
            int(params.get("slow_period", self.slow_period)),
        )
        self.signal_period = max(2, int(params.get("signal_period", self.signal_period)))
        self.position_size = max(1, int(params.get("position_size", self.position_size)))
        self.histogram_threshold = max(
            0.0, float(params.get("histogram_threshold", self.histogram_threshold))
        )

    def reset(self) -> None:
        super().reset()
        self._prev_histogram.clear()
