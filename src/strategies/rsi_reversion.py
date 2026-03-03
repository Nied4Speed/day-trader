"""RSI Mean Reversion Strategy.

Buys when RSI drops below oversold threshold (assumes bounce),
sells when RSI rises above overbought threshold (assumes pullback).
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.core.strategy import BarData, Strategy, TradeSignal


class RSIReversionStrategy(Strategy):
    strategy_type = "rsi_reversion"

    def __init__(self, name: str, params: Optional[dict] = None):
        self.rsi_period: int = 14
        self.oversold: float = 30.0
        self.overbought: float = 70.0
        self.position_size: int = 10
        super().__init__(name, params)

    def _compute_rsi(self, closes: pd.Series) -> float:
        """Compute RSI from a series of close prices."""
        if len(closes) < self.rsi_period + 1:
            return 50.0  # neutral

        deltas = closes.diff().dropna()
        gains = deltas.where(deltas > 0, 0.0)
        losses = (-deltas.where(deltas < 0, 0.0))

        avg_gain = gains.rolling(self.rsi_period).mean().iloc[-1]
        avg_loss = losses.rolling(self.rsi_period).mean().iloc[-1]

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def on_bar(self, bar: BarData) -> Optional[TradeSignal]:
        self.record_bar(bar)
        closes = self.get_close_series(bar.symbol)

        if len(closes) < self.rsi_period + 1:
            return None

        rsi = self._compute_rsi(closes)

        if rsi < self.oversold:
            return TradeSignal(
                symbol=bar.symbol,
                side="buy",
                quantity=self.position_size,
            )

        if rsi > self.overbought:
            return TradeSignal(
                symbol=bar.symbol,
                side="sell",
                quantity=self.position_size,
            )

        return None

    def get_params(self) -> dict:
        return {
            "rsi_period": self.rsi_period,
            "oversold": self.oversold,
            "overbought": self.overbought,
            "position_size": self.position_size,
        }

    def set_params(self, params: dict) -> None:
        self.rsi_period = max(2, int(params.get("rsi_period", self.rsi_period)))
        self.oversold = max(5.0, min(45.0, float(params.get("oversold", self.oversold))))
        self.overbought = max(55.0, min(95.0, float(params.get("overbought", self.overbought))))
        self.position_size = max(1, int(params.get("position_size", self.position_size)))
