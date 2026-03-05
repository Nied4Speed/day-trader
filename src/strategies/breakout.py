"""Opening Range Breakout Strategy.

Tracks the high/low range from the first N bars of the session.
Buys on a breakout above the range high, sells on a breakdown
below the range low. Classic day trading pattern.
"""

import logging
from typing import Optional

from src.core.strategy import BarData, Strategy, TradeSignal

logger = logging.getLogger(__name__)


class BreakoutStrategy(Strategy):
    strategy_type = "breakout"

    def __init__(self, name: str, params: Optional[dict] = None):
        self.range_bars: int = 15  # first 15 bars define the range
        self.breakout_pct: float = 0.001  # 0.1% beyond range to confirm
        self.allocation_pct: float = 0.35  # breakouts deserve conviction
        self._range_high: dict[str, float] = {}
        self._range_low: dict[str, float] = {}
        self._range_set: dict[str, bool] = {}
        super().__init__(name, params)

    def on_bar(self, bar: BarData) -> Optional[TradeSignal]:
        self.record_bar(bar)
        liq = self.check_liquidation(bar)
        if liq:
            return liq if liq.quantity > 0 else None
        history = self.get_history(bar.symbol)
        bar_count = len(history)

        if bar_count <= self.range_bars:
            highs = [b.high for b in history]
            lows = [b.low for b in history]
            self._range_high[bar.symbol] = max(highs)
            self._range_low[bar.symbol] = min(lows)
            if bar_count == self.range_bars:
                self._range_set[bar.symbol] = True
            return None

        if not self._range_set.get(bar.symbol, False):
            return None

        range_high = self._range_high[bar.symbol]
        range_low = self._range_low[bar.symbol]
        threshold_high = range_high * (1 + self.breakout_pct)
        threshold_low = range_low * (1 - self.breakout_pct)

        # Cache indicators for watch rule evaluation
        distance_pct = (threshold_high - bar.close) / bar.close if bar.close > 0 else 0.0
        self._indicators[bar.symbol] = {
            "close": bar.close,
            "range_high": range_high,
            "range_low": range_low,
            "threshold_high": threshold_high,
            "threshold_low": threshold_low,
            "distance_to_breakout_pct": distance_pct,
        }

        if bar.close > threshold_high:
            qty = self.compute_quantity(bar.close, self.allocation_pct)
            if qty > 0:
                return TradeSignal(symbol=bar.symbol, side="buy", quantity=qty)

        if bar.close < threshold_low:
            qty = self.compute_quantity(bar.close, self.allocation_pct)
            if qty > 0:
                return TradeSignal(symbol=bar.symbol, side="sell", quantity=qty)

        return None

    def get_params(self) -> dict:
        return {
            "range_bars": self.range_bars,
            "breakout_pct": self.breakout_pct,
            "allocation_pct": self.allocation_pct,
        }

    def set_params(self, params: dict) -> None:
        self.range_bars = max(5, int(params.get("range_bars", self.range_bars)))
        self.breakout_pct = max(0.0001, float(params.get("breakout_pct", self.breakout_pct)))
        self.allocation_pct = max(0.05, min(1.0, float(params.get("allocation_pct", self.allocation_pct))))

    def adapt(self, recent_signals: list, recent_fills: list, realized_pnl: float) -> None:
        old_pct = self.breakout_pct
        # Count false breakouts: buy fills followed by a losing sell
        buys = [f for f in recent_fills if f.get("side") == "buy"]
        losing_sells = [f for f in recent_fills if f.get("side") == "sell" and f.get("pnl", 0) < 0]
        false_breakout_rate = len(losing_sells) / len(buys) if buys else 0.0
        if false_breakout_rate > 0.5:
            self.breakout_pct += 0.002
        elif false_breakout_rate < 0.3:
            self.breakout_pct -= 0.001
        self.breakout_pct = max(0.0005, min(0.02, self.breakout_pct))
        logger.debug(
            f"{self.name} adapt: breakout_pct {old_pct:.4f}->{self.breakout_pct:.4f}, "
            f"false_breakout_rate={false_breakout_rate:.2f}, pnl={realized_pnl:.2f}"
        )

    def reset(self) -> None:
        super().reset()
        self._range_high.clear()
        self._range_low.clear()
        self._range_set.clear()
