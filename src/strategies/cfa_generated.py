"""CFA-Generated Macro-Aware Defensive VWAP Strategy.

Shift from pure mean reversion to macro regime-aware defensive trading.
Adds SPY trend filter and dramatically reduces position sizes to preserve
capital in adverse market conditions.

Key changes from previous version:
1. Add SPY 5-minute trend filter to avoid counter-market trades
2. Reduce allocation from 0.15 to 0.08 for defensive sizing
3. Tighter fast profit (0.5%) and regime-aware stop loss adjustment
"""

import logging
import pandas as pd
from typing import Optional

from src.core.strategy import BarData, Strategy, TradeSignal

logger = logging.getLogger(__name__)


class CfaGeneratedStrategy(Strategy):
    strategy_type = "cfa_generated"

    def __init__(self, name: str, params: Optional[dict] = None):
        # VWAP deviation thresholds
        self.vwap_deviation_threshold: float = 0.0015
        self.max_deviation: float = 0.006

        # Volume confirmation
        self.volume_confirmation: float = 1.3
        self.min_volume_ratio: float = 0.9

        # Momentum and trend filters
        self.momentum_window: int = 5
        self.max_momentum_threshold: float = 0.015  # Tightened from 0.02
        self.spy_trend_window: int = 5  # NEW: SPY trend filter

        # Defensive position management
        self.allocation_pct: float = 0.08  # Reduced from 0.15
        self.max_positions: int = 6  # Reduced from 8
        self.reversion_timeout: int = 8  # Faster timeout

        # Faster profit target
        self.fast_profit_pct: float = 0.5  # Reduced from 0.8

        # Internal state
        self._cumulative_vol: dict[str, float] = {}
        self._cumulative_vp: dict[str, float] = {}
        self._bars_since_entry: dict[str, int] = {}
        self._volume_history: dict[str, list[float]] = {}
        self._spy_history: list[float] = []  # NEW: SPY price history
        self._adapt_wins: int = 0
        self._adapt_total: int = 0

        super().__init__(name, params)

        # Regime-aware risk management
        if not params or "stop_loss_pct" not in params:
            self.stop_loss_pct = 1.2  # Tighter base stop
        if not params or "take_profit_pct" not in params:
            self.take_profit_pct = 1.8  # Lower target

        # Tighter defensive stops
        self.trailing_stop_tiers = [(0.5, 0.2), (1.2, 0.3)]
        self.patience_stop_tiers = [(5, -0.4), (10, -0.2)]

    def _get_vwap(self, symbol: str, bar: BarData) -> Optional[float]:
        """Compute cumulative intraday VWAP from typical price * volume."""
        typical_price = (bar.high + bar.low + bar.close) / 3.0
        self._cumulative_vp[symbol] = (
            self._cumulative_vp.get(symbol, 0.0) + typical_price * bar.volume
        )
        self._cumulative_vol[symbol] = (
            self._cumulative_vol.get(symbol, 0.0) + bar.volume
        )
        if self._cumulative_vol[symbol] == 0:
            return None
        return self._cumulative_vp[symbol] / self._cumulative_vol[symbol]

    def _get_avg_volume(self, symbol: str) -> float:
        """Return 20-bar average volume for the symbol."""
        history = self._volume_history.get(symbol, [])
        return sum(history) / len(history) if history else 0.0

    def _update_volume_history(self, symbol: str, volume: float) -> None:
        """Append volume and keep only the last 20 bars."""
        if symbol not in self._volume_history:
            self._volume_history[symbol] = []
        self._volume_history[symbol].append(volume)
        if len(self._volume_history[symbol]) > 20:
            self._volume_history[symbol] = self._volume_history[symbol][-20:]

    def _update_spy_trend(self, bar: BarData) -> None:
        """NEW: Update SPY price history for trend analysis."""
        if bar.symbol == 'SPY':
            self._spy_history.append(bar.close)
            if len(self._spy_history) > self.spy_trend_window:
                self._spy_history = self._spy_history[-self.spy_trend_window:]

    def _check_spy_trend(self) -> str:
        """NEW: Determine SPY trend direction."""
        if len(self._spy_history) < self.spy_trend_window:
            return 'neutral'
        
        start_price = self._spy_history[0]
        end_price = self._spy_history[-1]
        trend_pct = (end_price - start_price) / start_price
        
        if trend_pct > 0.002:  # >0.2% trend
            return 'bullish'
        elif trend_pct < -0.002:  # <-0.2% trend
            return 'bearish'
        else:
            return 'neutral'

    def _check_momentum_filter(self, symbol: str) -> bool:
        """Check if recent momentum is suitable for mean reversion."""
        history = self.get_history(symbol, self.momentum_window)
        if len(history) < self.momentum_window:
            return True
        
        start_price = history[0].close
        end_price = history[-1].close
        momentum = (end_price - start_price) / start_price
        
        return abs(momentum) < self.max_momentum_threshold

    def on_bar(self, bar: BarData) -> Optional[TradeSignal]:
        self.record_bar(bar)
        
        # Update SPY trend data
        self._update_spy_trend(bar)

        # Check liquidation first
        liq = self.check_liquidation(bar)
        if liq:
            if liq.quantity > 0 and liq.symbol in self._bars_since_entry:
                del self._bars_since_entry[liq.symbol]
            return liq if liq.quantity > 0 else None

        # Compute VWAP
        vwap = self._get_vwap(bar.symbol, bar)
        if vwap is None or vwap == 0:
            self._update_volume_history(bar.symbol, bar.volume)
            return None

        self._update_volume_history(bar.symbol, bar.volume)

        price = bar.close
        deviation = (price - vwap) / vwap
        abs_deviation = abs(deviation)
        avg_volume = self._get_avg_volume(bar.symbol)
        current_position = self._positions.get(bar.symbol, 0)
        entry_price = self._entry_prices.get(bar.symbol, 0)
        spy_trend = self._check_spy_trend()

        # Increment bars counter for positions
        if current_position > 0:
            self._bars_since_entry[bar.symbol] = (
                self._bars_since_entry.get(bar.symbol, 0) + 1
            )

        # Cache indicators
        self._indicators[bar.symbol] = {
            "close": price,
            "vwap": vwap,
            "deviation_from_vwap": deviation,
            "spy_trend": spy_trend
        }

        # --- EXIT LOGIC ---
        if current_position > 0 and entry_price > 0:
            # Fast profit-taking (tightened)
            pnl_pct = ((price - entry_price) / entry_price) * 100
            if pnl_pct >= self.fast_profit_pct:
                del self._bars_since_entry[bar.symbol]
                return TradeSignal(
                    symbol=bar.symbol,
                    side="sell",
                    quantity=current_position,
                    reason="fast_profit",
                )

            # Regime-aware early exit in bearish conditions
            if spy_trend == 'bearish' and pnl_pct < -0.5:
                del self._bars_since_entry[bar.symbol]
                return TradeSignal(
                    symbol=bar.symbol,
                    side="sell",
                    quantity=current_position,
                    reason="regime_defensive",
                )

            # Reversion timeout (faster)
            bars_held = self._bars_since_entry.get(bar.symbol, 0)
            if bars_held >= self.reversion_timeout:
                del self._bars_since_entry[bar.symbol]
                return TradeSignal(
                    symbol=bar.symbol,
                    side="sell",
                    quantity=current_position,
                    reason="reversion_timeout",
                )

            # VWAP reversion sell
            volume_ok = (
                avg_volume > 0
                and bar.volume > avg_volume * self.min_volume_ratio
            )
            if (
                deviation > self.vwap_deviation_threshold
                and abs_deviation < self.max_deviation
                and volume_ok
            ):
                del self._bars_since_entry[bar.symbol]
                return TradeSignal(
                    symbol=bar.symbol,
                    side="sell",
                    quantity=current_position,
                    reason="vwap_reversion",
                )

        # --- ENTRY LOGIC ---
        if current_position == 0 and len(self._positions) < self.max_positions:
            if avg_volume <= 0:
                return None

            # Apply momentum filter
            if not self._check_momentum_filter(bar.symbol):
                return None

            # NEW: SPY trend filter - be more selective in bearish conditions
            if spy_trend == 'bearish' and abs_deviation < 0.003:
                return None  # Skip small deviations in bear market

            volume_confirmed = bar.volume > avg_volume * self.volume_confirmation
            below_vwap = deviation < -self.vwap_deviation_threshold
            within_max = abs_deviation < self.max_deviation

            if below_vwap and within_max and volume_confirmed:
                # Regime-adjusted position sizing
                base_allocation = self.allocation_pct
                if spy_trend == 'bearish':
                    base_allocation *= 0.7  # Reduce size in bear market
                elif spy_trend == 'bullish':
                    base_allocation *= 1.2  # Slightly increase in bull market
                
                qty = self.compute_quantity(
                    price, base_allocation, symbol=bar.symbol
                )
                if qty > 0:
                    self._bars_since_entry[bar.symbol] = 0
                    return TradeSignal(
                        symbol=bar.symbol, side="buy", quantity=qty
                    )

        return None

    def get_params(self) -> dict:
        return {
            "vwap_deviation_threshold": self.vwap_deviation_threshold,
            "volume_confirmation": self.volume_confirmation,
            "max_deviation": self.max_deviation,
            "reversion_timeout": self.reversion_timeout,
            "min_volume_ratio": self.min_volume_ratio,
            "momentum_window": self.momentum_window,
            "max_momentum_threshold": self.max_momentum_threshold,
            "spy_trend_window": self.spy_trend_window,
            "fast_profit_pct": self.fast_profit_pct,
            "allocation_pct": self.allocation_pct,
            "max_positions": self.max_positions,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
        }

    def set_params(self, params: dict) -> None:
        self.vwap_deviation_threshold = max(
            0.0005, min(0.005, float(params.get("vwap_deviation_threshold", self.vwap_deviation_threshold)))
        )
        self.volume_confirmation = max(
            1.0, min(2.0, float(params.get("volume_confirmation", self.volume_confirmation)))
        )
        self.max_deviation = max(
            0.003, min(0.02, float(params.get("max_deviation", self.max_deviation)))
        )
        self.reversion_timeout = max(
            5, min(15, int(params.get("reversion_timeout", self.reversion_timeout)))
        )
        self.spy_trend_window = max(
            3, min(10, int(params.get("spy_trend_window", self.spy_trend_window)))
        )
        self.allocation_pct = max(
            0.03, min(0.15, float(params.get("allocation_pct", self.allocation_pct)))
        )
        self.max_positions = max(
            3, min(10, int(params.get("max_positions", self.max_positions)))
        )

    def adapt(self, recent_signals: list, recent_fills: list, realized_pnl: float) -> None:
        """Adapt based on regime and performance."""
        wins = sum(1 for f in recent_fills if isinstance(f, dict) and f.get("pnl", 0) > 0)
        total = sum(1 for f in recent_fills if isinstance(f, dict) and f.get("side") == "sell")
        
        self._adapt_wins += wins
        self._adapt_total += total
        
        if self._adapt_total >= 5:
            win_rate = self._adapt_wins / self._adapt_total
            
            # Adjust thresholds based on win rate and regime
            if win_rate < 0.4:
                self.vwap_deviation_threshold = min(0.003, self.vwap_deviation_threshold + 0.0003)
                self.max_momentum_threshold = max(0.01, self.max_momentum_threshold - 0.003)
            elif win_rate > 0.7:
                self.vwap_deviation_threshold = max(0.001, self.vwap_deviation_threshold - 0.0002)

    def reset(self) -> None:
        """Clear all internal state for a new session."""
        super().reset()
        self._cumulative_vol.clear()
        self._cumulative_vp.clear()
        self._bars_since_entry.clear()
        self._volume_history.clear()
        self._spy_history.clear()
        self._adapt_wins = 0
        self._adapt_total = 0
