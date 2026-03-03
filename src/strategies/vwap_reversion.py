"""VWAP Reversion Strategy.

Buys when price drops significantly below VWAP (volume-weighted average price),
sells when price rises above VWAP. Institutional traders watch VWAP closely,
so price tends to revert to it.
"""

from typing import Optional

from src.core.strategy import BarData, Strategy, TradeSignal


class VWAPReversionStrategy(Strategy):
    strategy_type = "vwap_reversion"

    def __init__(self, name: str, params: Optional[dict] = None):
        self.deviation_buy: float = -0.003  # buy when 0.3% below VWAP
        self.deviation_sell: float = 0.003  # sell when 0.3% above VWAP
        self.position_size: int = 1
        self._cumulative_vol: dict[str, float] = {}
        self._cumulative_vp: dict[str, float] = {}
        super().__init__(name, params)

    def _get_vwap(self, symbol: str, bar: BarData) -> Optional[float]:
        typical_price = (bar.high + bar.low + bar.close) / 3.0
        self._cumulative_vp[symbol] = self._cumulative_vp.get(symbol, 0) + typical_price * bar.volume
        self._cumulative_vol[symbol] = self._cumulative_vol.get(symbol, 0) + bar.volume
        if self._cumulative_vol[symbol] == 0:
            return None
        return self._cumulative_vp[symbol] / self._cumulative_vol[symbol]

    def on_bar(self, bar: BarData) -> Optional[TradeSignal]:
        self.record_bar(bar)
        vwap = self._get_vwap(bar.symbol, bar)
        if vwap is None or vwap == 0:
            return None

        deviation = (bar.close - vwap) / vwap

        if deviation < self.deviation_buy:
            return TradeSignal(symbol=bar.symbol, side="buy", quantity=self.position_size)
        if deviation > self.deviation_sell:
            return TradeSignal(symbol=bar.symbol, side="sell", quantity=self.position_size)
        return None

    def get_params(self) -> dict:
        return {"deviation_buy": self.deviation_buy, "deviation_sell": self.deviation_sell, "position_size": self.position_size}

    def set_params(self, params: dict) -> None:
        self.deviation_buy = min(-0.0005, float(params.get("deviation_buy", self.deviation_buy)))
        self.deviation_sell = max(0.0005, float(params.get("deviation_sell", self.deviation_sell)))
        self.position_size = max(1, int(params.get("position_size", self.position_size)))

    def reset(self) -> None:
        super().reset()
        self._cumulative_vol.clear()
        self._cumulative_vp.clear()
