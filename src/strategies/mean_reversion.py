"""Simple Mean Reversion Strategy.

Buys when price deviates significantly below its moving average,
sells when it deviates above. Different from RSI in that it uses
raw price deviation rather than relative strength.
"""

import logging
from typing import Optional

from src.core.strategy import BarData, Strategy, TradeSignal

logger = logging.getLogger(__name__)


class MeanReversionStrategy(Strategy):
    strategy_type = "mean_reversion"

    def __init__(self, name: str, params: Optional[dict] = None):
        self.ma_period: int = 30
        self.buy_deviation: float = -0.015  # buy when 1.5% below MA
        self.sell_deviation: float = 0.015  # sell when 1.5% above MA
        self.allocation_pct: float = 0.25
        super().__init__(name, params)

    def on_bar(self, bar: BarData) -> Optional[TradeSignal]:
        self.record_bar(bar)
        liq = self.check_liquidation(bar)
        if liq:
            return liq if liq.quantity > 0 else None
        closes = self.get_close_series(bar.symbol)
        if len(closes) < self.ma_period:
            return None

        ma = closes.rolling(self.ma_period).mean().iloc[-1]
        if ma == 0:
            return None

        deviation = (bar.close - ma) / ma

        # Cache indicators for watch rule evaluation
        self._indicators[bar.symbol] = {
            "close": bar.close,
            "ma": ma,
            "deviation_from_ma": deviation,
        }

        if deviation < self.buy_deviation:
            qty = self.compute_quantity(bar.close, self.allocation_pct)
            if qty > 0:
                return TradeSignal(symbol=bar.symbol, side="buy", quantity=qty)
        if deviation > self.sell_deviation:
            qty = self.compute_quantity(bar.close, self.allocation_pct)
            if qty > 0:
                return TradeSignal(symbol=bar.symbol, side="sell", quantity=qty)
        return None

    def get_params(self) -> dict:
        return {
            "ma_period": self.ma_period,
            "buy_deviation": self.buy_deviation,
            "sell_deviation": self.sell_deviation,
            "allocation_pct": self.allocation_pct,
        }

    def set_params(self, params: dict) -> None:
        self.ma_period = max(5, int(params.get("ma_period", self.ma_period)))
        self.buy_deviation = min(-0.001, float(params.get("buy_deviation", self.buy_deviation)))
        self.sell_deviation = max(0.001, float(params.get("sell_deviation", self.sell_deviation)))
        self.allocation_pct = max(0.05, min(1.0, float(params.get("allocation_pct", self.allocation_pct))))

    def adapt(self, recent_signals: list, recent_fills: list, realized_pnl: float) -> None:
        old_buy, old_sell = self.buy_deviation, self.sell_deviation
        if realized_pnl < 0:
            self.buy_deviation -= 0.003
            self.sell_deviation += 0.003
        else:
            self.buy_deviation += 0.0015
            self.sell_deviation -= 0.0015
        self.buy_deviation = max(-0.05, min(-0.005, self.buy_deviation))
        self.sell_deviation = max(0.005, min(0.05, self.sell_deviation))
        logger.debug(
            f"{self.name} adapt: pnl={realized_pnl:.2f} "
            f"buy_dev {old_buy:.4f}->{self.buy_deviation:.4f} "
            f"sell_dev {old_sell:.4f}->{self.sell_deviation:.4f}"
        )
