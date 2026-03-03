"""Stochastic Oscillator Strategy.

Uses %K and %D lines to identify overbought/oversold conditions.
Buys when %K crosses above %D in oversold territory, sells when
%K crosses below %D in overbought territory.
"""

from typing import Optional

from src.core.strategy import BarData, Strategy, TradeSignal


class StochasticStrategy(Strategy):
    strategy_type = "stochastic"

    def __init__(self, name: str, params: Optional[dict] = None):
        self.k_period: int = 14
        self.d_period: int = 3
        self.oversold: float = 20.0
        self.overbought: float = 80.0
        self.position_size: int = 1
        self._prev_k: dict[str, float] = {}
        self._prev_d: dict[str, float] = {}
        super().__init__(name, params)

    def on_bar(self, bar: BarData) -> Optional[TradeSignal]:
        self.record_bar(bar)
        history = self.get_history(bar.symbol)
        if len(history) < self.k_period + self.d_period:
            return None

        # Compute %K values for the d_period window
        k_values = []
        for i in range(self.d_period):
            idx = len(history) - self.d_period + i
            window = history[idx - self.k_period + 1:idx + 1]
            highest = max(b.high for b in window)
            lowest = min(b.low for b in window)
            if highest == lowest:
                k_values.append(50.0)
            else:
                k_values.append((window[-1].close - lowest) / (highest - lowest) * 100)

        k = k_values[-1]
        d = sum(k_values) / len(k_values)

        prev_k = self._prev_k.get(bar.symbol)
        prev_d = self._prev_d.get(bar.symbol)
        self._prev_k[bar.symbol] = k
        self._prev_d[bar.symbol] = d

        if prev_k is None or prev_d is None:
            return None

        # Buy: %K crosses above %D in oversold zone
        if prev_k <= prev_d and k > d and k < self.oversold:
            return TradeSignal(symbol=bar.symbol, side="buy", quantity=self.position_size)

        # Sell: %K crosses below %D in overbought zone
        if prev_k >= prev_d and k < d and k > self.overbought:
            return TradeSignal(symbol=bar.symbol, side="sell", quantity=self.position_size)

        return None

    def get_params(self) -> dict:
        return {"k_period": self.k_period, "d_period": self.d_period, "oversold": self.oversold, "overbought": self.overbought, "position_size": self.position_size}

    def set_params(self, params: dict) -> None:
        self.k_period = max(5, int(params.get("k_period", self.k_period)))
        self.d_period = max(2, int(params.get("d_period", self.d_period)))
        self.oversold = max(5.0, min(40.0, float(params.get("oversold", self.oversold))))
        self.overbought = max(60.0, min(95.0, float(params.get("overbought", self.overbought))))
        self.position_size = max(1, int(params.get("position_size", self.position_size)))

    def reset(self) -> None:
        super().reset()
        self._prev_k.clear()
        self._prev_d.clear()
