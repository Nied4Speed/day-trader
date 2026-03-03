"""Opening Range Breakout Strategy.

Tracks the high/low range from the first N bars of the session.
Buys on a breakout above the range high, sells on a breakdown
below the range low. Classic day trading pattern.
"""

from typing import Optional

from src.core.strategy import BarData, Strategy, TradeSignal


class BreakoutStrategy(Strategy):
    strategy_type = "breakout"

    def __init__(self, name: str, params: Optional[dict] = None):
        self.range_bars: int = 15  # first 15 bars define the range
        self.breakout_pct: float = 0.001  # 0.1% beyond range to confirm
        self.position_size: int = 1
        self._range_high: dict[str, float] = {}
        self._range_low: dict[str, float] = {}
        self._range_set: dict[str, bool] = {}
        super().__init__(name, params)

    def on_bar(self, bar: BarData) -> Optional[TradeSignal]:
        self.record_bar(bar)
        history = self.get_history(bar.symbol)
        bar_count = len(history)

        # Still building the opening range
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

        if bar.close > threshold_high:
            return TradeSignal(symbol=bar.symbol, side="buy", quantity=self.position_size)

        if bar.close < threshold_low:
            return TradeSignal(symbol=bar.symbol, side="sell", quantity=self.position_size)

        return None

    def get_params(self) -> dict:
        return {"range_bars": self.range_bars, "breakout_pct": self.breakout_pct, "position_size": self.position_size}

    def set_params(self, params: dict) -> None:
        self.range_bars = max(5, int(params.get("range_bars", self.range_bars)))
        self.breakout_pct = max(0.0001, float(params.get("breakout_pct", self.breakout_pct)))
        self.position_size = max(1, int(params.get("position_size", self.position_size)))

    def reset(self) -> None:
        super().reset()
        self._range_high.clear()
        self._range_low.clear()
        self._range_set.clear()
