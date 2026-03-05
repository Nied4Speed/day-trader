"""MACD (Moving Average Convergence Divergence) Strategy.

Buys on MACD line crossing above signal line (bullish),
sells on MACD crossing below signal (bearish).
Uses histogram divergence for confirmation.
"""

import logging
from typing import Optional

import pandas as pd

from src.core.strategy import BarData, Strategy, TradeSignal

logger = logging.getLogger(__name__)


class MACDStrategy(Strategy):
    strategy_type = "macd"

    def __init__(self, name: str, params: Optional[dict] = None):
        self.fast_period: int = 12
        self.slow_period: int = 26
        self.signal_period: int = 9
        self.allocation_pct: float = 0.25
        self.histogram_threshold: float = 0.0
        self._prev_histogram: dict[str, float] = {}
        super().__init__(name, params)

    def on_bar(self, bar: BarData) -> Optional[TradeSignal]:
        self.record_bar(bar)
        liq = self.check_liquidation(bar)
        if liq:
            return liq if liq.quantity > 0 else None
        closes = self.get_close_series(bar.symbol)

        min_required = self.slow_period + self.signal_period
        if len(closes) < min_required:
            return None

        fast_ema = closes.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = closes.ewm(span=self.slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        current_hist = histogram.iloc[-1]
        prev_hist = self._prev_histogram.get(bar.symbol)
        self._prev_histogram[bar.symbol] = current_hist

        # Cache indicators for watch rule evaluation
        self._indicators[bar.symbol] = {
            "close": bar.close,
            "histogram": float(current_hist),
            "prev_histogram": float(prev_hist) if prev_hist is not None else 0.0,
            "macd_line": float(macd_line.iloc[-1]),
            "signal_line": float(signal_line.iloc[-1]),
        }

        if prev_hist is None:
            return None

        if prev_hist <= self.histogram_threshold and current_hist > self.histogram_threshold:
            qty = self.compute_quantity(bar.close, self.allocation_pct)
            if qty > 0:
                return TradeSignal(symbol=bar.symbol, side="buy", quantity=qty)

        if prev_hist >= -self.histogram_threshold and current_hist < -self.histogram_threshold:
            qty = self.compute_quantity(bar.close, self.allocation_pct)
            if qty > 0:
                return TradeSignal(symbol=bar.symbol, side="sell", quantity=qty)

        return None

    def get_params(self) -> dict:
        return {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "signal_period": self.signal_period,
            "allocation_pct": self.allocation_pct,
            "histogram_threshold": self.histogram_threshold,
        }

    def set_params(self, params: dict) -> None:
        self.fast_period = max(2, int(params.get("fast_period", self.fast_period)))
        self.slow_period = max(
            self.fast_period + 1,
            int(params.get("slow_period", self.slow_period)),
        )
        self.signal_period = max(2, int(params.get("signal_period", self.signal_period)))
        self.allocation_pct = max(0.05, min(1.0, float(params.get("allocation_pct", self.allocation_pct))))
        self.histogram_threshold = max(
            0.0, float(params.get("histogram_threshold", self.histogram_threshold))
        )

    def adapt(self, recent_signals: list, recent_fills: list, realized_pnl: float) -> None:
        old_thresh = self.histogram_threshold
        if realized_pnl < 0:
            self.histogram_threshold += 0.05
        elif realized_pnl > 0:
            self.histogram_threshold -= 0.02
        self.histogram_threshold = max(0.0, min(1.0, self.histogram_threshold))
        logger.debug(
            f"{self.name} adapt: histogram_threshold {old_thresh:.2f}->{self.histogram_threshold:.2f}, "
            f"pnl={realized_pnl:.2f}"
        )

    def reset(self) -> None:
        super().reset()
        self._prev_histogram.clear()
