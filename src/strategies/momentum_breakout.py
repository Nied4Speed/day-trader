"""Momentum Breakout Strategy.

Buys when price breaks above the 20-period high with volume confirmation
and RSI < 70 (not overbought). Exits on RSI > 80, stop/take-profit, or
5-bar patience timeout. Data-informed by CFA analysis of 2026-03-11 session.
"""

import logging
from typing import Optional

from src.core.strategy import BarData, Strategy, TradeSignal

logger = logging.getLogger(__name__)


class MomentumBreakoutStrategy(Strategy):
    strategy_type = "momentum_breakout"

    def __init__(self, name: str, params: Optional[dict] = None):
        self.lookback_high: int = 20
        self.volume_multiplier: float = 2.0
        self.rsi_threshold: float = 70.0
        self.rsi_exit: float = 80.0
        self.volume_window: int = 20
        self.allocation_pct: float = 0.20
        self.max_positions: int = 10
        super().__init__(name, params)
        if self.stop_loss_pct is None or self.stop_loss_pct == 2.0:
            self.stop_loss_pct = 1.0
        if self.take_profit_pct is None or self.take_profit_pct == 3.0:
            self.take_profit_pct = 2.0
        if not self.trailing_stop_tiers:
            self.trailing_stop_tiers = [(1.0, 0.5)]
        if not self.patience_stop_tiers:
            self.patience_stop_tiers = [(5, 0.0)]

    def _compute_rsi(self, closes, period: int = 14) -> float:
        """Compute RSI from a list/series of close prices."""
        if len(closes) < period + 1:
            return 50.0  # neutral default
        deltas = [closes[i] - closes[i - 1] for i in range(-period, 0)]
        gains = [d for d in deltas if d > 0]
        losses = [-d for d in deltas if d < 0]
        avg_gain = sum(gains) / period if gains else 0.0
        avg_loss = sum(losses) / period if losses else 0.0001
        rs = avg_gain / avg_loss if avg_loss > 0 else 100.0
        return 100.0 - (100.0 / (1.0 + rs))

    def on_bar(self, bar: BarData) -> Optional[TradeSignal]:
        self.record_bar(bar)
        liq = self.check_liquidation(bar)
        if liq:
            return liq if liq.quantity > 0 else None

        history = self.get_history(bar.symbol)
        min_bars = max(self.lookback_high, self.volume_window) + 1
        if not history or len(history) < min_bars:
            return None

        # Compute indicators
        closes = [b.close for b in history]
        rsi = self._compute_rsi(closes)
        period_high = max(b.high for b in history[-self.lookback_high:])
        vol_bars = history[-self.volume_window:]
        avg_volume = sum(b.volume for b in vol_bars) / len(vol_bars) if vol_bars else 0
        volume_ok = avg_volume > 0 and bar.volume > self.volume_multiplier * avg_volume

        self._indicators[bar.symbol] = {
            "close": bar.close,
            "rsi": rsi,
            "period_high": period_high,
            "volume_ratio": bar.volume / avg_volume if avg_volume > 0 else 0,
        }

        current_position = self._positions.get(bar.symbol, 0)

        # SELL: RSI overbought exit
        if current_position > 0 and rsi > self.rsi_exit:
            return TradeSignal(
                symbol=bar.symbol,
                side="sell",
                quantity=current_position,
                reason="rsi_overbought_exit",
            )

        # BUY: breakout above period high + volume + RSI filter
        if current_position <= 0:
            open_positions = sum(1 for q in self._positions.values() if q > 0)
            if open_positions >= self.max_positions:
                return None
            if bar.close > period_high and volume_ok and rsi < self.rsi_threshold:
                qty = self.compute_quantity(bar.close, self.allocation_pct, symbol=bar.symbol)
                if qty > 0:
                    return TradeSignal(
                        symbol=bar.symbol,
                        side="buy",
                        quantity=qty,
                        reason="momentum_breakout",
                    )

        return None

    def get_params(self) -> dict:
        return {
            "lookback_high": self.lookback_high,
            "volume_multiplier": self.volume_multiplier,
            "rsi_threshold": self.rsi_threshold,
            "rsi_exit": self.rsi_exit,
            "volume_window": self.volume_window,
            "allocation_pct": self.allocation_pct,
            "max_positions": self.max_positions,
        }

    def set_params(self, params: dict) -> None:
        self.lookback_high = max(5, min(50, int(params.get("lookback_high", self.lookback_high))))
        self.volume_multiplier = max(1.0, min(5.0, float(params.get("volume_multiplier", self.volume_multiplier))))
        self.rsi_threshold = max(50.0, min(85.0, float(params.get("rsi_threshold", self.rsi_threshold))))
        self.rsi_exit = max(60.0, min(95.0, float(params.get("rsi_exit", self.rsi_exit))))
        self.volume_window = max(5, min(50, int(params.get("volume_window", self.volume_window))))
        self.allocation_pct = max(0.05, min(0.50, float(params.get("allocation_pct", self.allocation_pct))))
        self.max_positions = max(1, min(15, int(params.get("max_positions", self.max_positions))))

    def adapt(self, recent_signals: list, recent_fills: list, realized_pnl: float) -> None:
        old_vol = self.volume_multiplier
        old_lookback = self.lookback_high
        old_rsi = self.rsi_threshold

        sells = [f for f in recent_fills if f.get("side") == "sell"]
        wins = [f for f in sells if f.get("pnl", 0) > 0]
        win_rate = len(wins) / len(sells) if sells else 0.5

        # Tune volume_multiplier: lower = more trades, higher = fewer but better
        if win_rate < 0.45:
            self.volume_multiplier = min(3.0, self.volume_multiplier + 0.1)
        elif win_rate > 0.70:
            self.volume_multiplier = max(1.5, self.volume_multiplier - 0.1)

        # Tune lookback_high
        if win_rate < 0.45:
            self.lookback_high = min(30, self.lookback_high + 2)
        elif win_rate > 0.70:
            self.lookback_high = max(10, self.lookback_high - 1)

        # Tune RSI threshold
        if win_rate < 0.45:
            self.rsi_threshold = max(55.0, self.rsi_threshold - 2.0)
        elif win_rate > 0.70:
            self.rsi_threshold = min(80.0, self.rsi_threshold + 1.0)

        logger.info(
            f"{self.name} adapt: vol_mult {old_vol:.1f}->{self.volume_multiplier:.1f}, "
            f"lookback {old_lookback}->{self.lookback_high}, "
            f"rsi_thresh {old_rsi:.0f}->{self.rsi_threshold:.0f}, "
            f"wr={win_rate:.2f}, pnl={realized_pnl:.2f}"
        )

    def reset(self) -> None:
        super().reset()
