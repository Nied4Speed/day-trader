"""Momentum Strategy.

Buys when price shows strong upward momentum (rate of change above threshold),
sells when momentum reverses. Uses lookback period to smooth noise.
"""

import logging
from typing import Optional

import pandas as pd

from src.core.strategy import BarData, Strategy, TradeSignal

logger = logging.getLogger(__name__)


class MomentumStrategy(Strategy):
    strategy_type = "momentum"

    def __init__(self, name: str, params: Optional[dict] = None):
        self.lookback: int = 20
        self.buy_threshold: float = 0.02  # 2% momentum to buy
        self.sell_threshold: float = -0.01  # -1% momentum to sell
        self.allocation_pct: float = 0.25  # 25% of capital per trade
        self.volume_filter: float = 1.5  # volume must be 1.5x average
        super().__init__(name, params)

    def on_bar(self, bar: BarData) -> Optional[TradeSignal]:
        self.record_bar(bar)
        liq = self.check_liquidation(bar)
        if liq:
            return liq if liq.quantity > 0 else None
        history = self.get_history(bar.symbol)

        if len(history) < self.lookback + 1:
            return None

        closes = [b.close for b in history]
        volumes = [b.volume for b in history]

        # Rate of change over lookback period
        ref_price = closes[-self.lookback]
        roc = (closes[-1] - ref_price) / ref_price

        # Volume filter: current volume vs average
        avg_volume = sum(volumes[-self.lookback:]) / self.lookback
        volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 0

        # Cache indicators for watch rule evaluation
        self._indicators[bar.symbol] = {
            "close": bar.close,
            "roc": roc,
            "ref_price": ref_price,
            "volume_ratio": volume_ratio,
        }

        if roc > self.buy_threshold and volume_ratio > self.volume_filter:
            qty = self.compute_quantity(bar.close, self.allocation_pct)
            if qty > 0:
                return TradeSignal(symbol=bar.symbol, side="buy", quantity=qty)

        if roc < self.sell_threshold:
            qty = self.compute_quantity(bar.close, self.allocation_pct)
            if qty > 0:
                return TradeSignal(symbol=bar.symbol, side="sell", quantity=qty)

        return None

    def get_params(self) -> dict:
        return {
            "lookback": self.lookback,
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
            "allocation_pct": self.allocation_pct,
            "volume_filter": self.volume_filter,
        }

    def set_params(self, params: dict) -> None:
        self.lookback = max(5, int(params.get("lookback", self.lookback)))
        self.buy_threshold = max(0.001, float(params.get("buy_threshold", self.buy_threshold)))
        self.sell_threshold = min(-0.001, float(params.get("sell_threshold", self.sell_threshold)))
        self.allocation_pct = max(0.05, min(1.0, float(params.get("allocation_pct", self.allocation_pct))))
        self.volume_filter = max(0.5, float(params.get("volume_filter", self.volume_filter)))

    def adapt(self, recent_signals: list, recent_fills: list, realized_pnl: float) -> None:
        old_thresh = self.buy_threshold
        if realized_pnl < 0:
            self.buy_threshold += 0.005
        elif realized_pnl > 0:
            self.buy_threshold -= 0.002
        self.buy_threshold = max(0.005, min(0.10, self.buy_threshold))
        logger.info(
            f"{self.name} adapt: buy_threshold {old_thresh:.4f}->{self.buy_threshold:.4f}, "
            f"pnl={realized_pnl:.2f}"
        )

    def reset(self) -> None:
        super().reset()
