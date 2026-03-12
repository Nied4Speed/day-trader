"""Simple ATR Breakout Strategy.

Buys when price breaks above previous close + (ATR * breakout_factor) with
volume confirmation. Tight stops and quick trailing to lock profits. Designed
to replace vol_compress_gen1_14 with a simpler, more robust approach.
"""

import logging
from typing import Optional

import numpy as np

from src.core.strategy import BarData, Strategy, TradeSignal

logger = logging.getLogger(__name__)


class ATRBreakoutSimpleStrategy(Strategy):
    strategy_type = "atr_breakout_simple"

    def __init__(self, name: str, params: Optional[dict] = None):
        # Core parameters (from CFA spec)
        self.atr_period: int = 20
        self.breakout_factor: float = 2.2
        self.volume_confirm: float = 1.4
        self.min_price: float = 5.0
        self.max_price: float = 500.0
        self.allocation_pct: float = 0.12
        self.max_positions: int = 5
        # Adaptation state
        self._consecutive_losses: int = 0
        self._pause_bars: int = 0  # bars remaining in pause
        self._signal_count: int = 0
        self._signal_pnl_sum: float = 0.0
        super().__init__(name, params)
        # Override base class risk params per CFA spec
        self.stop_loss_pct = 1.5
        self.take_profit_pct = 2.5
        self.trailing_stop_tiers = [(1.2, 0.6), (2.5, 1.0)]
        self.patience_stop_tiers = [(6, 0.4), (12, 0.2)]

    def _calc_atr(self, bars: list[BarData]) -> float:
        """Average True Range over atr_period."""
        if len(bars) < 2:
            return 0.0
        trs = []
        for i in range(1, len(bars)):
            high_low = bars[i].high - bars[i].low
            high_close = abs(bars[i].high - bars[i - 1].close)
            low_close = abs(bars[i].low - bars[i - 1].close)
            trs.append(max(high_low, high_close, low_close))
        period = min(self.atr_period, len(trs))
        return float(np.mean(trs[-period:])) if trs else 0.0

    def _avg_volume(self, bars: list[BarData], lookback: int = 10) -> float:
        """Average volume over the last `lookback` bars."""
        recent = bars[-lookback:] if len(bars) >= lookback else bars
        if not recent:
            return 0.0
        return float(np.mean([b.volume for b in recent]))

    def on_bar(self, bar: BarData) -> Optional[TradeSignal]:
        self.record_bar(bar)
        liq = self.check_liquidation(bar)
        if liq:
            return liq if liq.quantity > 0 else None

        history = self._bar_history.get(bar.symbol, [])
        # Need at least atr_period + 1 bars for ATR + previous close
        if len(history) < self.atr_period + 1:
            return None

        # Price filter
        if bar.close < self.min_price or bar.close > self.max_price:
            return None

        # Pause after consecutive losses
        if self._pause_bars > 0:
            self._pause_bars -= 1
            return None

        atr = self._calc_atr(history)
        if atr <= 0:
            return None

        prev_close = history[-2].close
        avg_vol = self._avg_volume(history[:-1], lookback=10)
        breakout_distance = atr * self.breakout_factor
        volume_ok = avg_vol > 0 and bar.volume > self.volume_confirm * avg_vol

        # Cache indicators
        self._indicators[bar.symbol] = {
            "close": bar.close,
            "atr": atr,
            "prev_close": prev_close,
            "breakout_up": prev_close + breakout_distance,
            "breakout_down": prev_close - breakout_distance,
            "volume_ratio": bar.volume / avg_vol if avg_vol > 0 else 0.0,
        }

        current_position = self._positions.get(bar.symbol, 0)

        # SELL: breakdown with volume confirmation
        if current_position > 0:
            if bar.close < prev_close - breakout_distance and volume_ok:
                return TradeSignal(
                    symbol=bar.symbol,
                    side="sell",
                    quantity=current_position,
                    reason="atr_breakdown",
                )

        # BUY: breakout with volume confirmation
        if current_position <= 0:
            # Respect max positions
            open_positions = sum(1 for q in self._positions.values() if q > 0)
            if open_positions >= self.max_positions:
                return None

            if bar.close > prev_close + breakout_distance and volume_ok:
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
            "atr_period": self.atr_period,
            "breakout_factor": self.breakout_factor,
            "volume_confirm": self.volume_confirm,
            "min_price": self.min_price,
            "max_price": self.max_price,
            "allocation_pct": self.allocation_pct,
            "max_positions": self.max_positions,
        }

    def set_params(self, params: dict) -> None:
        self.atr_period = max(5, min(50, int(params.get("atr_period", self.atr_period))))
        self.breakout_factor = max(1.0, min(5.0, float(params.get("breakout_factor", self.breakout_factor))))
        self.volume_confirm = max(1.0, min(3.0, float(params.get("volume_confirm", self.volume_confirm))))
        self.min_price = max(1.0, min(50.0, float(params.get("min_price", self.min_price))))
        self.max_price = max(100.0, min(2000.0, float(params.get("max_price", self.max_price))))
        self.allocation_pct = max(0.05, min(0.50, float(params.get("allocation_pct", self.allocation_pct))))
        self.max_positions = max(1, min(10, int(params.get("max_positions", self.max_positions))))

    def adapt(self, recent_signals: list, recent_fills: list, realized_pnl: float) -> None:
        old_factor = self.breakout_factor
        old_alloc = self.allocation_pct

        # Track signal quality
        self._signal_count += len(recent_fills)
        self._signal_pnl_sum += realized_pnl

        # Count consecutive losses
        sells = [f for f in recent_fills if f.get("side") == "sell"]
        for f in sells:
            if f.get("pnl", 0) < 0:
                self._consecutive_losses += 1
            else:
                self._consecutive_losses = 0

        # Pause after 3+ consecutive losses
        if self._consecutive_losses >= 3:
            self._pause_bars = 5
            self._consecutive_losses = 0
            logger.info(f"{self.name} adapt: pausing 5 bars after 3 consecutive losses")

        # Adjust breakout_factor based on signal quality
        avg_pnl = self._signal_pnl_sum / max(1, self._signal_count)
        if avg_pnl < 0:
            # Signals are bad -- raise the bar
            self.breakout_factor = min(2.8, self.breakout_factor + 0.1)
        elif avg_pnl > 0.5:
            # Signals are good -- can be slightly less selective
            self.breakout_factor = max(1.8, self.breakout_factor - 0.05)

        # Reduce allocation if avg P&L is weak
        if avg_pnl < 0.2:
            self.allocation_pct = max(0.05, self.allocation_pct - 0.02)

        self.breakout_factor = max(1.8, min(2.8, self.breakout_factor))
        self.allocation_pct = max(0.05, min(0.50, self.allocation_pct))

        logger.info(
            f"{self.name} adapt: breakout_factor {old_factor:.2f}->{self.breakout_factor:.2f} "
            f"allocation_pct {old_alloc:.3f}->{self.allocation_pct:.3f} "
            f"avg_pnl={avg_pnl:.3f} consec_losses={self._consecutive_losses} "
            f"pnl={realized_pnl:.2f}"
        )

    def reset(self) -> None:
        super().reset()
        self._consecutive_losses = 0
        self._pause_bars = 0
        self._signal_count = 0
        self._signal_pnl_sum = 0.0
