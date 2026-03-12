"""Bollinger Bands Strategy.

Buys when price touches the lower band (oversold), sells when price
touches the upper band (overbought). Uses standard deviation bands
around a moving average.
"""

import logging
from typing import Optional

import pandas as pd

from src.core.strategy import BarData, Strategy, TradeSignal

logger = logging.getLogger(__name__)


class BollingerBandsStrategy(Strategy):
    strategy_type = "bollinger_bands"

    def __init__(self, name: str, params: Optional[dict] = None):
        self.period: int = 20
        self.num_std: float = 2.0
        self.allocation_pct: float = 0.25
        super().__init__(name, params)

    def on_bar(self, bar: BarData) -> Optional[TradeSignal]:
        self.record_bar(bar)
        liq = self.check_liquidation(bar)
        if liq:
            return liq if liq.quantity > 0 else None
        closes = self.get_close_series(bar.symbol)

        if len(closes) < self.period:
            return None

        ma = closes.rolling(self.period).mean().iloc[-1]
        std = closes.rolling(self.period).std().iloc[-1]

        if std == 0:
            return None

        upper_band = ma + self.num_std * std
        lower_band = ma - self.num_std * std
        band_width = upper_band - lower_band
        band_position = (bar.close - lower_band) / band_width if band_width > 0 else 0.5

        # Cache indicators for watch rule evaluation
        self._indicators[bar.symbol] = {
            "close": bar.close,
            "lower_band": lower_band,
            "upper_band": upper_band,
            "band_width": band_width,
            "band_position": band_position,
        }

        if bar.close <= lower_band:
            qty = self.compute_quantity(bar.close, self.allocation_pct, symbol=bar.symbol)
            if qty > 0:
                return TradeSignal(symbol=bar.symbol, side="buy", quantity=qty)

        if bar.close >= upper_band:
            qty = self.compute_quantity(bar.close, self.allocation_pct)
            if qty > 0:
                return TradeSignal(symbol=bar.symbol, side="sell", quantity=qty)

        return None

    def get_params(self) -> dict:
        return {
            "period": self.period,
            "num_std": self.num_std,
            "allocation_pct": self.allocation_pct,
        }

    def set_params(self, params: dict) -> None:
        self.period = max(5, int(params.get("period", self.period)))
        self.num_std = max(0.5, min(4.0, float(params.get("num_std", self.num_std))))
        self.allocation_pct = max(0.05, min(1.0, float(params.get("allocation_pct", self.allocation_pct))))

    def adapt(self, recent_signals: list, recent_fills: list, realized_pnl: float) -> None:
        old_std = self.num_std
        if realized_pnl < 0:
            self.num_std += 0.1
        elif realized_pnl > 0:
            self.num_std -= 0.05
        self.num_std = max(1.0, min(3.5, self.num_std))
        logger.info(
            f"{self.name} adapt: num_std {old_std:.2f}->{self.num_std:.2f}, "
            f"pnl={realized_pnl:.2f}"
        )

    def reset(self) -> None:
        super().reset()
