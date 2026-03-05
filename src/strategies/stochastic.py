"""Stochastic Oscillator Strategy.

Uses %K and %D lines to identify overbought/oversold conditions.
Buys when %K crosses above %D in oversold territory, sells when
%K crosses below %D in overbought territory.
"""

import logging
from typing import Optional

from src.core.strategy import BarData, Strategy, TradeSignal

logger = logging.getLogger(__name__)


class StochasticStrategy(Strategy):
    strategy_type = "stochastic"

    def __init__(self, name: str, params: Optional[dict] = None):
        self.k_period: int = 14
        self.d_period: int = 3
        self.oversold: float = 20.0
        self.overbought: float = 80.0
        self.allocation_pct: float = 0.25
        self._prev_k: dict[str, float] = {}
        self._prev_d: dict[str, float] = {}
        super().__init__(name, params)

    def on_bar(self, bar: BarData) -> Optional[TradeSignal]:
        self.record_bar(bar)
        liq = self.check_liquidation(bar)
        if liq:
            return liq if liq.quantity > 0 else None
        history = self.get_history(bar.symbol)
        if len(history) < self.k_period + self.d_period:
            return None

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

        # Cache indicators for watch rule evaluation
        self._indicators[bar.symbol] = {
            "close": bar.close,
            "k": k,
            "d": d,
            "prev_k": prev_k if prev_k is not None else 50.0,
            "prev_d": prev_d if prev_d is not None else 50.0,
        }

        if prev_k is None or prev_d is None:
            return None

        if prev_k <= prev_d and k > d and k < self.oversold:
            qty = self.compute_quantity(bar.close, self.allocation_pct)
            if qty > 0:
                return TradeSignal(symbol=bar.symbol, side="buy", quantity=qty)

        if prev_k >= prev_d and k < d and k > self.overbought:
            qty = self.compute_quantity(bar.close, self.allocation_pct)
            if qty > 0:
                return TradeSignal(symbol=bar.symbol, side="sell", quantity=qty)

        return None

    def get_params(self) -> dict:
        return {
            "k_period": self.k_period,
            "d_period": self.d_period,
            "oversold": self.oversold,
            "overbought": self.overbought,
            "allocation_pct": self.allocation_pct,
        }

    def set_params(self, params: dict) -> None:
        self.k_period = max(5, int(params.get("k_period", self.k_period)))
        self.d_period = max(2, int(params.get("d_period", self.d_period)))
        self.oversold = max(5.0, min(40.0, float(params.get("oversold", self.oversold))))
        self.overbought = max(60.0, min(95.0, float(params.get("overbought", self.overbought))))
        self.allocation_pct = max(0.05, min(1.0, float(params.get("allocation_pct", self.allocation_pct))))

    def adapt(self, recent_signals: list, recent_fills: list, realized_pnl: float) -> None:
        old_os, old_ob = self.oversold, self.overbought
        if realized_pnl < 0:
            self.oversold -= 2.0
            self.overbought += 2.0
        else:
            self.oversold += 1.0
            self.overbought -= 1.0
        self.oversold = max(10.0, min(35.0, self.oversold))
        self.overbought = max(65.0, min(90.0, self.overbought))
        logger.debug(
            f"{self.name} adapt: pnl={realized_pnl:.2f} "
            f"oversold {old_os:.1f}->{self.oversold:.1f} "
            f"overbought {old_ob:.1f}->{self.overbought:.1f}"
        )

    def reset(self) -> None:
        super().reset()
        self._prev_k.clear()
        self._prev_d.clear()
