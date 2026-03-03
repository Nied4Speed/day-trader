"""Bollinger Bands Strategy.

Buys when price touches the lower band (oversold), sells when price
touches the upper band (overbought). Uses standard deviation bands
around a moving average.
"""

from typing import Optional

import pandas as pd

from src.core.strategy import BarData, Strategy, TradeSignal


class BollingerBandsStrategy(Strategy):
    strategy_type = "bollinger_bands"

    def __init__(self, name: str, params: Optional[dict] = None):
        self.period: int = 20
        self.num_std: float = 2.0
        self.position_size: int = 1
        super().__init__(name, params)

    def on_bar(self, bar: BarData) -> Optional[TradeSignal]:
        self.record_bar(bar)
        closes = self.get_close_series(bar.symbol)

        if len(closes) < self.period:
            return None

        ma = closes.rolling(self.period).mean().iloc[-1]
        std = closes.rolling(self.period).std().iloc[-1]

        if std == 0:
            return None

        upper_band = ma + self.num_std * std
        lower_band = ma - self.num_std * std
        current_price = bar.close

        # Buy when price touches lower band
        if current_price <= lower_band:
            return TradeSignal(
                symbol=bar.symbol,
                side="buy",
                quantity=self.position_size,
            )

        # Sell when price touches upper band
        if current_price >= upper_band:
            return TradeSignal(
                symbol=bar.symbol,
                side="sell",
                quantity=self.position_size,
            )

        return None

    def get_params(self) -> dict:
        return {
            "period": self.period,
            "num_std": self.num_std,
            "position_size": self.position_size,
        }

    def set_params(self, params: dict) -> None:
        self.period = max(5, int(params.get("period", self.period)))
        self.num_std = max(0.5, min(4.0, float(params.get("num_std", self.num_std))))
        self.position_size = max(1, int(params.get("position_size", self.position_size)))
