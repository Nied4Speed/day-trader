"""RSI Mean Reversion Strategy.

Buys when RSI drops below oversold threshold (assumes bounce),
sells when RSI rises above overbought threshold (assumes pullback).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.core.strategy import BarData, Strategy, TradeSignal

logger = logging.getLogger(__name__)


class RSIReversionStrategy(Strategy):
    strategy_type = "rsi_reversion"

    def __init__(self, name: str, params: Optional[dict] = None):
        self.rsi_period: int = 14
        self.oversold: float = 30.0
        self.overbought: float = 70.0
        self.allocation_pct: float = 0.25
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
        liq = self.check_liquidation(bar)
        if liq:
            return liq if liq.quantity > 0 else None
        closes = self.get_close_series(bar.symbol)

        if len(closes) < self.rsi_period + 1:
            return None

        rsi = self._compute_rsi(closes)

        # Cache indicators for watch rule evaluation
        self._indicators[bar.symbol] = {
            "close": bar.close,
            "rsi": rsi,
        }

        if rsi < self.oversold:
            qty = self.compute_quantity(bar.close, self.allocation_pct)
            if qty > 0:
                return TradeSignal(symbol=bar.symbol, side="buy", quantity=qty)

        if rsi > self.overbought:
            qty = self.compute_quantity(bar.close, self.allocation_pct)
            if qty > 0:
                return TradeSignal(symbol=bar.symbol, side="sell", quantity=qty)

        return None

    def get_params(self) -> dict:
        return {
            "rsi_period": self.rsi_period,
            "oversold": self.oversold,
            "overbought": self.overbought,
            "allocation_pct": self.allocation_pct,
        }

    def set_params(self, params: dict) -> None:
        self.rsi_period = max(2, int(params.get("rsi_period", self.rsi_period)))
        self.oversold = max(5.0, min(45.0, float(params.get("oversold", self.oversold))))
        self.overbought = max(55.0, min(95.0, float(params.get("overbought", self.overbought))))
        self.allocation_pct = max(0.05, min(1.0, float(params.get("allocation_pct", self.allocation_pct))))

    def adapt(self, recent_signals: list, recent_fills: list, realized_pnl: float) -> None:
        old_oversold, old_overbought = self.oversold, self.overbought
        if realized_pnl < 0:
            self.oversold -= 2
            self.overbought += 2
        elif realized_pnl > 0:
            self.oversold += 1
            self.overbought -= 1
        self.oversold = max(15.0, min(40.0, self.oversold))
        self.overbought = max(60.0, min(85.0, self.overbought))
        logger.info(
            f"{self.name} adapt: oversold {old_oversold}->{self.oversold}, "
            f"overbought {old_overbought}->{self.overbought}, pnl={realized_pnl:.2f}"
        )
