"""Simple Mean Reversion Strategy.

Buys when price deviates significantly below its moving average,
sells when it deviates above. Different from RSI in that it uses
raw price deviation rather than relative strength.
"""

from typing import Optional

from src.core.strategy import BarData, Strategy, TradeSignal


class MeanReversionStrategy(Strategy):
    strategy_type = "mean_reversion"

    def __init__(self, name: str, params: Optional[dict] = None):
        self.ma_period: int = 30
        self.buy_deviation: float = -0.015  # buy when 1.5% below MA
        self.sell_deviation: float = 0.015  # sell when 1.5% above MA
        self.position_size: int = 1
        super().__init__(name, params)

    def on_bar(self, bar: BarData) -> Optional[TradeSignal]:
        self.record_bar(bar)
        closes = self.get_close_series(bar.symbol)
        if len(closes) < self.ma_period:
            return None

        ma = closes.rolling(self.ma_period).mean().iloc[-1]
        if ma == 0:
            return None

        deviation = (bar.close - ma) / ma

        if deviation < self.buy_deviation:
            return TradeSignal(symbol=bar.symbol, side="buy", quantity=self.position_size)
        if deviation > self.sell_deviation:
            return TradeSignal(symbol=bar.symbol, side="sell", quantity=self.position_size)
        return None

    def get_params(self) -> dict:
        return {"ma_period": self.ma_period, "buy_deviation": self.buy_deviation, "sell_deviation": self.sell_deviation, "position_size": self.position_size}

    def set_params(self, params: dict) -> None:
        self.ma_period = max(5, int(params.get("ma_period", self.ma_period)))
        self.buy_deviation = min(-0.001, float(params.get("buy_deviation", self.buy_deviation)))
        self.sell_deviation = max(0.001, float(params.get("sell_deviation", self.sell_deviation)))
        self.position_size = max(1, int(params.get("position_size", self.position_size)))
