"""Volatility Compression Strategy.

Detects periods of narrowing volatility (Bollinger Band squeeze / low ATR)
and enters when the compression breaks out. After prolonged low-volatility
consolidation, price tends to make a directional move. Buys on upward
expansion, sells on downward expansion.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.core.strategy import BarData, Strategy, TradeSignal

logger = logging.getLogger(__name__)


class VolatilityCompressionStrategy(Strategy):
    strategy_type = "volatility_compression"

    def __init__(self, name: str, params: Optional[dict] = None):
        self.bb_period: int = 20  # Bollinger Band lookback
        self.bb_std: float = 2.0
        self.atr_period: int = 14
        self.squeeze_lookback: int = 10  # bars to check for narrowing
        self.squeeze_threshold: float = 0.6  # band width < 60% of recent avg = squeeze
        self.expansion_pct: float = 0.005  # 0.5% move from squeeze to trigger entry
        self.allocation_pct: float = 0.30  # higher conviction on breakout
        self.min_squeeze_bars: int = 5  # minimum bars in squeeze before we care
        # Internal state
        self._in_squeeze: dict[str, int] = {}  # symbol -> bars in squeeze
        self._squeeze_midpoint: dict[str, float] = {}  # price at squeeze entry
        super().__init__(name, params)

    def _calc_bb_width(self, closes: pd.Series) -> float | None:
        """Calculate Bollinger Band width as % of middle band."""
        if len(closes) < self.bb_period:
            return None
        ma = closes.rolling(self.bb_period).mean().iloc[-1]
        std = closes.rolling(self.bb_period).std().iloc[-1]
        if ma <= 0 or np.isnan(std):
            return None
        upper = ma + self.bb_std * std
        lower = ma - self.bb_std * std
        return (upper - lower) / ma

    def _calc_atr(self, bars: list[BarData]) -> float:
        """Average True Range."""
        if len(bars) < 2:
            return 0.0
        trs = []
        for i in range(1, len(bars)):
            high_low = bars[i].high - bars[i].low
            high_close = abs(bars[i].high - bars[i - 1].close)
            low_close = abs(bars[i].low - bars[i - 1].close)
            trs.append(max(high_low, high_close, low_close))
        period = min(self.atr_period, len(trs))
        return float(np.mean(trs[-period:])) if trs else 0.0

    def on_bar(self, bar: BarData) -> Optional[TradeSignal]:
        self.record_bar(bar)
        liq = self.check_liquidation(bar)
        if liq:
            return liq if liq.quantity > 0 else None

        history = self.get_history(bar.symbol)
        if len(history) < self.bb_period + self.squeeze_lookback:
            return None

        closes = self.get_close_series(bar.symbol)
        current_width = self._calc_bb_width(closes)
        if current_width is None:
            return None

        # Calculate average BB width over squeeze_lookback period
        recent_widths = []
        for i in range(self.squeeze_lookback):
            idx = len(closes) - 1 - i
            if idx >= self.bb_period:
                subset = closes.iloc[:idx + 1]
                w = self._calc_bb_width(subset)
                if w is not None:
                    recent_widths.append(w)
        avg_width = np.mean(recent_widths) if recent_widths else current_width
        atr = self._calc_atr(history)

        # Cache indicators
        squeeze_ratio = current_width / avg_width if avg_width > 0 else 1.0
        self._indicators[bar.symbol] = {
            "close": bar.close,
            "bb_width": current_width,
            "avg_bb_width": avg_width,
            "squeeze_ratio": squeeze_ratio,
            "atr": atr,
        }

        # Detect squeeze
        is_squeeze = squeeze_ratio < self.squeeze_threshold
        sym = bar.symbol

        if is_squeeze:
            self._in_squeeze[sym] = self._in_squeeze.get(sym, 0) + 1
            if sym not in self._squeeze_midpoint:
                ma = closes.rolling(self.bb_period).mean().iloc[-1]
                self._squeeze_midpoint[sym] = float(ma)
        else:
            # Squeeze released — check for expansion breakout
            squeeze_bars = self._in_squeeze.get(sym, 0)
            midpoint = self._squeeze_midpoint.get(sym)

            if squeeze_bars >= self.min_squeeze_bars and midpoint and midpoint > 0:
                move_pct = (bar.close - midpoint) / midpoint

                # Upward expansion: buy
                if move_pct > self.expansion_pct:
                    self._in_squeeze.pop(sym, None)
                    self._squeeze_midpoint.pop(sym, None)
                    qty = self.compute_quantity(bar.close, self.allocation_pct, symbol=bar.symbol)
                    if qty > 0:
                        return TradeSignal(
                            symbol=sym, side="buy", quantity=qty,
                            reason="volatility_expansion_up",
                        )

                # Downward expansion: sell
                if move_pct < -self.expansion_pct:
                    self._in_squeeze.pop(sym, None)
                    self._squeeze_midpoint.pop(sym, None)
                    qty = self.compute_quantity(bar.close, self.allocation_pct)
                    if qty > 0:
                        return TradeSignal(
                            symbol=sym, side="sell", quantity=qty,
                            reason="volatility_expansion_down",
                        )

            # Clear squeeze state if no breakout triggered
            self._in_squeeze.pop(sym, None)
            self._squeeze_midpoint.pop(sym, None)

        return None

    def get_params(self) -> dict:
        return {
            "bb_period": self.bb_period,
            "bb_std": self.bb_std,
            "atr_period": self.atr_period,
            "squeeze_lookback": self.squeeze_lookback,
            "squeeze_threshold": self.squeeze_threshold,
            "expansion_pct": self.expansion_pct,
            "allocation_pct": self.allocation_pct,
            "min_squeeze_bars": self.min_squeeze_bars,
        }

    def set_params(self, params: dict) -> None:
        self.bb_period = max(10, min(50, int(params.get("bb_period", self.bb_period))))
        self.bb_std = max(1.0, min(3.0, float(params.get("bb_std", self.bb_std))))
        self.atr_period = max(5, min(30, int(params.get("atr_period", self.atr_period))))
        self.squeeze_lookback = max(5, min(30, int(params.get("squeeze_lookback", self.squeeze_lookback))))
        self.squeeze_threshold = max(0.3, min(0.9, float(params.get("squeeze_threshold", self.squeeze_threshold))))
        self.expansion_pct = max(0.001, min(0.02, float(params.get("expansion_pct", self.expansion_pct))))
        self.allocation_pct = max(0.05, min(1.0, float(params.get("allocation_pct", self.allocation_pct))))
        self.min_squeeze_bars = max(2, min(15, int(params.get("min_squeeze_bars", self.min_squeeze_bars))))

    def adapt(self, recent_signals: list, recent_fills: list, realized_pnl: float) -> None:
        old_exp = self.expansion_pct
        old_sq = self.squeeze_threshold
        if realized_pnl < 0:
            # Tighten: require more compression and bigger expansion
            self.expansion_pct += 0.001
            self.squeeze_threshold -= 0.05
        elif realized_pnl > 0:
            # Loosen slightly
            self.expansion_pct -= 0.0005
            self.squeeze_threshold += 0.02
        self.expansion_pct = max(0.002, min(0.02, self.expansion_pct))
        self.squeeze_threshold = max(0.3, min(0.8, self.squeeze_threshold))
        logger.info(
            f"{self.name} adapt: expansion_pct {old_exp:.4f}->{self.expansion_pct:.4f} "
            f"squeeze_threshold {old_sq:.2f}->{self.squeeze_threshold:.2f} pnl={realized_pnl:.2f}"
        )

    def reset(self) -> None:
        super().reset()
        self._in_squeeze.clear()
        self._squeeze_midpoint.clear()
