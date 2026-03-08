"""VWAP Reversion Strategy.

Buys when price drops significantly below VWAP (volume-weighted average price),
sells when price rises above VWAP. Institutional traders watch VWAP closely,
so price tends to revert to it.
"""

import logging
from typing import Optional

from src.core.strategy import BarData, Strategy, TradeSignal

logger = logging.getLogger(__name__)


class VWAPReversionStrategy(Strategy):
    strategy_type = "vwap_reversion"

    def __init__(self, name: str, params: Optional[dict] = None):
        self.deviation_buy: float = -0.003  # buy when 0.3% below VWAP
        self.deviation_sell: float = 0.003  # sell when 0.3% above VWAP
        self.allocation_pct: float = 0.25
        self._cumulative_vol: dict[str, float] = {}
        self._cumulative_vp: dict[str, float] = {}
        super().__init__(name, params)

    def _get_vwap(self, symbol: str, bar: BarData) -> float | None:
        typical_price = (bar.high + bar.low + bar.close) / 3.0
        self._cumulative_vp[symbol] = self._cumulative_vp.get(symbol, 0) + typical_price * bar.volume
        self._cumulative_vol[symbol] = self._cumulative_vol.get(symbol, 0) + bar.volume
        if self._cumulative_vol[symbol] == 0:
            return None
        return self._cumulative_vp[symbol] / self._cumulative_vol[symbol]

    def on_bar(self, bar: BarData) -> Optional[TradeSignal]:
        self.record_bar(bar)
        liq = self.check_liquidation(bar)
        if liq:
            return liq if liq.quantity > 0 else None
        vwap = self._get_vwap(bar.symbol, bar)
        if vwap is None or vwap == 0:
            return None

        deviation = (bar.close - vwap) / vwap

        # Cache indicators for watch rule evaluation
        self._indicators[bar.symbol] = {
            "close": bar.close,
            "vwap": vwap,
            "deviation_from_vwap": deviation,
        }

        if deviation < self.deviation_buy:
            qty = self.compute_quantity(bar.close, self.allocation_pct)
            if qty > 0:
                return TradeSignal(symbol=bar.symbol, side="buy", quantity=qty)
        if deviation > self.deviation_sell:
            qty = self.compute_quantity(bar.close, self.allocation_pct)
            if qty > 0:
                return TradeSignal(symbol=bar.symbol, side="sell", quantity=qty)
        return None

    def get_params(self) -> dict:
        return {
            "deviation_buy": self.deviation_buy,
            "deviation_sell": self.deviation_sell,
            "allocation_pct": self.allocation_pct,
        }

    def set_params(self, params: dict) -> None:
        self.deviation_buy = min(-0.0005, float(params.get("deviation_buy", self.deviation_buy)))
        self.deviation_sell = max(0.0005, float(params.get("deviation_sell", self.deviation_sell)))
        self.allocation_pct = max(0.05, min(1.0, float(params.get("allocation_pct", self.allocation_pct))))

    def adapt(self, recent_signals: list, recent_fills: list, realized_pnl: float) -> None:
        old_buy, old_sell = self.deviation_buy, self.deviation_sell
        if realized_pnl < 0:
            self.deviation_buy -= 0.001
            self.deviation_sell += 0.001
        else:
            self.deviation_buy += 0.0005
            self.deviation_sell -= 0.0005
        self.deviation_buy = max(-0.02, min(-0.001, self.deviation_buy))
        self.deviation_sell = max(0.001, min(0.02, self.deviation_sell))
        logger.info(
            f"{self.name} adapt: pnl={realized_pnl:.2f} "
            f"dev_buy {old_buy:.4f}->{self.deviation_buy:.4f} "
            f"dev_sell {old_sell:.4f}->{self.deviation_sell:.4f}"
        )

    def reset(self) -> None:
        super().reset()
        self._cumulative_vol.clear()
        self._cumulative_vp.clear()
