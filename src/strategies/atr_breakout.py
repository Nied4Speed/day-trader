"""ATR Breakout Strategy.

Buys when price breaks above the 10-period high AND current ATR is expanding
(ATR > 1.2x its 20-period average), confirming volatility expansion into the
breakout. Exits at 1.5% profit, 2.0% stop, or 5-bar no-follow-through.
Data-informed by CFA analysis of 2026-03-11 session.
"""

import logging
from typing import Optional

from src.core.strategy import BarData, Strategy, TradeSignal

logger = logging.getLogger(__name__)


class ATRBreakoutStrategy(Strategy):
    strategy_type = "atr_breakout"

    def __init__(self, name: str, params: Optional[dict] = None):
        self.breakout_period: int = 10
        self.atr_period: int = 20
        self.atr_expansion: float = 1.2
        self.follow_through_bars: int = 5
        self.allocation_pct: float = 0.15
        self.max_positions: int = 8
        # Track bars since entry for follow-through check
        self._bars_since_entry: dict[str, int] = {}
        super().__init__(name, params)
        if self.stop_loss_pct is None or self.stop_loss_pct == 2.0:
            self.stop_loss_pct = 2.0
        if self.take_profit_pct is None or self.take_profit_pct == 3.0:
            self.take_profit_pct = 1.5
        if not self.trailing_stop_tiers:
            self.trailing_stop_tiers = [(1.0, 0.75)]
        if not self.patience_stop_tiers:
            self.patience_stop_tiers = [(5, 0.0)]

    def _compute_atr(self, bars: list[BarData], period: int) -> float:
        """Average True Range over the given period."""
        if len(bars) < 2:
            return 0.0
        trs = []
        for i in range(1, len(bars)):
            tr = max(
                bars[i].high - bars[i].low,
                abs(bars[i].high - bars[i - 1].close),
                abs(bars[i].low - bars[i - 1].close),
            )
            trs.append(tr)
        recent = trs[-period:] if len(trs) >= period else trs
        return sum(recent) / len(recent) if recent else 0.0

    def on_bar(self, bar: BarData) -> Optional[TradeSignal]:
        self.record_bar(bar)
        liq = self.check_liquidation(bar)
        if liq:
            return liq if liq.quantity > 0 else None

        history = self.get_history(bar.symbol)
        min_bars = max(self.breakout_period, self.atr_period) + 1
        if not history or len(history) < min_bars:
            return None

        current_position = self._positions.get(bar.symbol, 0)

        # Track bars since entry for follow-through check
        if current_position > 0:
            self._bars_since_entry[bar.symbol] = self._bars_since_entry.get(bar.symbol, 0) + 1
        else:
            self._bars_since_entry.pop(bar.symbol, None)

        # Compute ATR
        current_atr = self._compute_atr(history[-self.atr_period - 1:], self.atr_period)
        # Longer-term ATR for expansion check
        long_atr = self._compute_atr(history, self.atr_period)

        # Period high
        period_bars = history[-self.breakout_period:]
        period_high = max(b.high for b in period_bars)

        atr_expanding = long_atr > 0 and current_atr > self.atr_expansion * long_atr

        self._indicators[bar.symbol] = {
            "close": bar.close,
            "current_atr": current_atr,
            "long_atr": long_atr,
            "atr_ratio": current_atr / long_atr if long_atr > 0 else 0,
            "period_high": period_high,
        }

        # SELL: no follow-through after N bars
        if current_position > 0:
            bars_held = self._bars_since_entry.get(bar.symbol, 0)
            entry = self._entry_prices.get(bar.symbol, bar.close)
            gain_pct = (bar.close - entry) / entry * 100 if entry > 0 else 0
            if bars_held >= self.follow_through_bars and gain_pct < 0.5:
                self._bars_since_entry.pop(bar.symbol, None)
                return TradeSignal(
                    symbol=bar.symbol,
                    side="sell",
                    quantity=current_position,
                    reason="no_follow_through",
                )

        # BUY: breakout above period high + ATR expanding
        if current_position <= 0:
            open_positions = sum(1 for q in self._positions.values() if q > 0)
            if open_positions >= self.max_positions:
                return None
            if bar.close > period_high and atr_expanding:
                qty = self.compute_quantity(bar.close, self.allocation_pct, symbol=bar.symbol)
                if qty > 0:
                    return TradeSignal(
                        symbol=bar.symbol,
                        side="buy",
                        quantity=qty,
                        reason="atr_breakout",
                    )

        return None

    def get_params(self) -> dict:
        return {
            "breakout_period": self.breakout_period,
            "atr_period": self.atr_period,
            "atr_expansion": self.atr_expansion,
            "follow_through_bars": self.follow_through_bars,
            "allocation_pct": self.allocation_pct,
            "max_positions": self.max_positions,
        }

    def set_params(self, params: dict) -> None:
        self.breakout_period = max(3, min(30, int(params.get("breakout_period", self.breakout_period))))
        self.atr_period = max(5, min(50, int(params.get("atr_period", self.atr_period))))
        self.atr_expansion = max(1.0, min(2.0, float(params.get("atr_expansion", self.atr_expansion))))
        self.follow_through_bars = max(2, min(15, int(params.get("follow_through_bars", self.follow_through_bars))))
        self.allocation_pct = max(0.05, min(0.50, float(params.get("allocation_pct", self.allocation_pct))))
        self.max_positions = max(1, min(15, int(params.get("max_positions", self.max_positions))))

    def adapt(self, recent_signals: list, recent_fills: list, realized_pnl: float) -> None:
        old_period = self.breakout_period
        old_expansion = self.atr_expansion

        sells = [f for f in recent_fills if f.get("side") == "sell"]
        wins = [f for f in sells if f.get("pnl", 0) > 0]
        win_rate = len(wins) / len(sells) if sells else 0.5

        # Tune breakout_period: shorter = more signals, longer = fewer/stronger
        if win_rate < 0.45:
            self.breakout_period = min(20, self.breakout_period + 1)
        elif win_rate > 0.65:
            self.breakout_period = max(5, self.breakout_period - 1)

        # Tune atr_expansion threshold
        if win_rate < 0.45:
            self.atr_expansion = min(1.5, self.atr_expansion + 0.05)
        elif win_rate > 0.65:
            self.atr_expansion = max(1.1, self.atr_expansion - 0.05)

        logger.info(
            f"{self.name} adapt: breakout_period {old_period}->{self.breakout_period}, "
            f"atr_expansion {old_expansion:.2f}->{self.atr_expansion:.2f}, "
            f"wr={win_rate:.2f}, pnl={realized_pnl:.2f}"
        )

    def reset(self) -> None:
        super().reset()
        self._bars_since_entry.clear()
