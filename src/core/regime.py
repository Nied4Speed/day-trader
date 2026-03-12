"""Market regime detection for adaptive strategy weighting.

Detects current market regime per symbol and broadcasts to strategies:
- Trending Up / Down (ADX > 25)
- Ranging/Choppy (ADX < 20)
- High/Low Volatility (ATR vs average)

Strategies use regime to self-weight: momentum skips ranging markets,
mean reversion skips trending markets, etc.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from src.core.strategy import BarData

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    UNKNOWN = "unknown"


class VolatilityRegime(str, Enum):
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


@dataclass
class RegimeState:
    """Current regime for a symbol."""
    market: MarketRegime = MarketRegime.UNKNOWN
    volatility: VolatilityRegime = VolatilityRegime.NORMAL
    adx: float = 0.0
    atr: float = 0.0
    atr_ratio: float = 1.0  # current ATR / average ATR

    @property
    def is_trending(self) -> bool:
        return self.market in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN)

    @property
    def is_ranging(self) -> bool:
        return self.market == MarketRegime.RANGING

    @property
    def is_high_vol(self) -> bool:
        return self.volatility == VolatilityRegime.HIGH

    @property
    def is_low_vol(self) -> bool:
        return self.volatility == VolatilityRegime.LOW


# Strategy type -> regime weight multipliers
# Below 1.0 = reduce allocation, above 1.0 = boost
REGIME_STRATEGY_WEIGHTS: dict[str, dict[MarketRegime, float]] = {
    "momentum": {
        MarketRegime.TRENDING_UP: 1.2,
        MarketRegime.TRENDING_DOWN: 0.5,
        MarketRegime.RANGING: 0.3,
    },
    "breakout": {
        MarketRegime.TRENDING_UP: 1.2,
        MarketRegime.TRENDING_DOWN: 0.5,
        MarketRegime.RANGING: 0.4,
    },
    "ma_crossover": {
        MarketRegime.TRENDING_UP: 1.2,
        MarketRegime.TRENDING_DOWN: 0.8,
        MarketRegime.RANGING: 0.4,
    },
    "macd": {
        MarketRegime.TRENDING_UP: 1.1,
        MarketRegime.TRENDING_DOWN: 0.7,
        MarketRegime.RANGING: 0.5,
    },
    "rsi_reversion": {
        MarketRegime.TRENDING_UP: 0.5,
        MarketRegime.TRENDING_DOWN: 0.5,
        MarketRegime.RANGING: 1.3,
    },
    "bollinger_bands": {
        MarketRegime.TRENDING_UP: 0.5,
        MarketRegime.TRENDING_DOWN: 0.5,
        MarketRegime.RANGING: 1.3,
    },
    "mean_reversion": {
        MarketRegime.TRENDING_UP: 0.4,
        MarketRegime.TRENDING_DOWN: 0.4,
        MarketRegime.RANGING: 1.3,
    },
    "vwap_reversion": {
        MarketRegime.TRENDING_UP: 0.5,
        MarketRegime.TRENDING_DOWN: 0.5,
        MarketRegime.RANGING: 1.2,
    },
    "stochastic": {
        MarketRegime.TRENDING_UP: 0.6,
        MarketRegime.TRENDING_DOWN: 0.6,
        MarketRegime.RANGING: 1.2,
    },
    "ml_predictor": {
        MarketRegime.TRENDING_UP: 1.0,
        MarketRegime.TRENDING_DOWN: 1.0,
        MarketRegime.RANGING: 1.0,
    },
    "news_sentiment": {
        MarketRegime.TRENDING_UP: 1.0,
        MarketRegime.TRENDING_DOWN: 1.0,
        MarketRegime.RANGING: 1.0,
    },
    "collab": {
        MarketRegime.TRENDING_UP: 1.0,
        MarketRegime.TRENDING_DOWN: 1.0,
        MarketRegime.RANGING: 1.0,
    },
    "momentum_breakout": {
        MarketRegime.TRENDING_UP: 1.2,
        MarketRegime.TRENDING_DOWN: 0.5,
        MarketRegime.RANGING: 0.3,
    },
    "atr_breakout_simple": {
        MarketRegime.TRENDING_UP: 1.2,
        MarketRegime.TRENDING_DOWN: 0.5,
        MarketRegime.RANGING: 0.4,
    },
    "atr_breakout": {
        MarketRegime.TRENDING_UP: 1.2,
        MarketRegime.TRENDING_DOWN: 0.5,
        MarketRegime.RANGING: 0.4,
    },
    "cfa_generated": {
        MarketRegime.TRENDING_UP: 0.5,
        MarketRegime.TRENDING_DOWN: 0.5,
        MarketRegime.RANGING: 1.2,
    },
    "volume_profile_reversion": {
        MarketRegime.TRENDING_UP: 0.5,
        MarketRegime.TRENDING_DOWN: 0.5,
        MarketRegime.RANGING: 1.3,
    },
    "volatility_compression": {
        MarketRegime.TRENDING_UP: 1.1,
        MarketRegime.TRENDING_DOWN: 0.6,
        MarketRegime.RANGING: 1.2,
    },
}


class RegimeDetector:
    """Detects market regime per symbol from bar history."""

    ADX_TRENDING_THRESHOLD = 25.0
    ADX_RANGING_THRESHOLD = 20.0
    ATR_HIGH_RATIO = 1.5
    ATR_LOW_RATIO = 0.7
    MA_PERIOD = 20
    ATR_PERIOD = 14
    ADX_PERIOD = 14

    def __init__(self):
        self._regimes: dict[str, RegimeState] = {}
        self._bar_history: dict[str, list[BarData]] = {}

    def update(self, bar: BarData) -> RegimeState:
        """Update regime for a symbol given a new bar."""
        symbol = bar.symbol
        if symbol not in self._bar_history:
            self._bar_history[symbol] = []
        self._bar_history[symbol].append(bar)

        # Need enough bars for ADX computation
        bars = self._bar_history[symbol]
        min_bars = max(self.ADX_PERIOD * 2, self.MA_PERIOD) + 5
        if len(bars) < min_bars:
            state = RegimeState()
            self._regimes[symbol] = state
            return state

        state = self._compute_regime(bars)
        self._regimes[symbol] = state
        return state

    def get_regime(self, symbol: str) -> RegimeState:
        """Get current regime for a symbol."""
        return self._regimes.get(symbol, RegimeState())

    def get_strategy_weight(self, strategy_type: str, symbol: str) -> float:
        """Get regime-based weight multiplier for a strategy on a symbol."""
        regime = self.get_regime(symbol)
        weights = REGIME_STRATEGY_WEIGHTS.get(strategy_type, {})
        market_weight = weights.get(regime.market, 1.0)

        # Adjust for volatility
        if regime.is_high_vol:
            market_weight *= 0.8  # reduce in high vol
        elif regime.is_low_vol:
            market_weight *= 1.1  # slight boost in low vol

        return market_weight

    def reset(self) -> None:
        """Clear all state for new session."""
        self._regimes.clear()
        self._bar_history.clear()

    def _compute_regime(self, bars: list[BarData]) -> RegimeState:
        """Compute market and volatility regime from bar history."""
        highs = np.array([b.high for b in bars])
        lows = np.array([b.low for b in bars])
        closes = np.array([b.close for b in bars])

        # ADX calculation
        adx = self._compute_adx(highs, lows, closes, self.ADX_PERIOD)

        # ATR calculation
        atr = self._compute_atr(highs, lows, closes, self.ATR_PERIOD)
        avg_atr = np.mean([
            self._compute_atr(highs[:i+1], lows[:i+1], closes[:i+1], self.ATR_PERIOD)
            for i in range(max(self.ATR_PERIOD + 1, len(bars) - 20), len(bars))
        ]) if len(bars) > self.ATR_PERIOD + 1 else atr
        atr_ratio = atr / avg_atr if avg_atr > 0 else 1.0

        # MA for trend direction
        ma = np.mean(closes[-self.MA_PERIOD:])
        current_price = closes[-1]

        # Determine market regime
        if adx >= self.ADX_TRENDING_THRESHOLD:
            if current_price > ma:
                market = MarketRegime.TRENDING_UP
            else:
                market = MarketRegime.TRENDING_DOWN
        elif adx <= self.ADX_RANGING_THRESHOLD:
            market = MarketRegime.RANGING
        else:
            # Transition zone (20-25): lean toward current state
            if current_price > ma:
                market = MarketRegime.TRENDING_UP
            else:
                market = MarketRegime.RANGING

        # Determine volatility regime
        if atr_ratio >= self.ATR_HIGH_RATIO:
            volatility = VolatilityRegime.HIGH
        elif atr_ratio <= self.ATR_LOW_RATIO:
            volatility = VolatilityRegime.LOW
        else:
            volatility = VolatilityRegime.NORMAL

        return RegimeState(
            market=market,
            volatility=volatility,
            adx=adx,
            atr=atr,
            atr_ratio=atr_ratio,
        )

    @staticmethod
    def _compute_atr(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int,
    ) -> float:
        """Compute Average True Range."""
        if len(highs) < period + 1:
            return 0.0
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )
        return float(np.mean(tr[-period:]))

    @staticmethod
    def _compute_adx(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int,
    ) -> float:
        """Compute Average Directional Index (ADX)."""
        n = len(highs)
        if n < period * 2:
            return 0.0

        # +DM and -DM
        up_move = highs[1:] - highs[:-1]
        down_move = lows[:-1] - lows[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # True Range
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )

        # Smoothed via EMA
        def ema_smooth(data, p):
            result = np.zeros_like(data, dtype=float)
            result[0] = data[0]
            alpha = 1.0 / p
            for i in range(1, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result

        smoothed_tr = ema_smooth(tr, period)
        smoothed_plus = ema_smooth(plus_dm, period)
        smoothed_minus = ema_smooth(minus_dm, period)

        # +DI and -DI
        with np.errstate(divide='ignore', invalid='ignore'):
            plus_di = 100.0 * smoothed_plus / smoothed_tr
            minus_di = 100.0 * smoothed_minus / smoothed_tr
            dx = 100.0 * np.abs(plus_di - minus_di) / np.where(
                (plus_di + minus_di) == 0, 1.0, plus_di + minus_di
            )

        # ADX = EMA of DX
        adx_series = ema_smooth(dx, period)
        return float(adx_series[-1]) if len(adx_series) > 0 else 0.0


# ---------------------------------------------------------------------------
# Macro Regime Overlay — broad-market regime using SPY
# ---------------------------------------------------------------------------

class MacroRegime(str, Enum):
    STRONG_BULL = "strong_bull"
    MILD_BULL = "mild_bull"
    MILD_BEAR = "mild_bear"
    STRONG_BEAR = "strong_bear"


@dataclass
class MacroMultipliers:
    """Parameter multipliers for the current macro regime."""
    stop_loss: float = 1.0
    breakeven_activate_pct: float = 0.3
    breakeven_floor_pct: float = 0.3
    trailing_trail: float = 1.0
    patience_bars: float = 1.0
    allocation: float = 1.0
    buy_gate_open: bool = True  # False = block new buys (sells always allowed)

    # CFA-recommended tables per regime
    _TABLES: dict = None  # populated below

    @classmethod
    def for_regime(cls, regime: "MacroRegime") -> "MacroMultipliers":
        return _MACRO_MULTIPLIER_TABLE[regime]


_MACRO_MULTIPLIER_TABLE: dict[MacroRegime, MacroMultipliers] = {
    MacroRegime.STRONG_BULL: MacroMultipliers(
        stop_loss=1.2, breakeven_activate_pct=0.4, breakeven_floor_pct=0.3,
        trailing_trail=1.3, patience_bars=1.2, allocation=1.0, buy_gate_open=True,
    ),
    MacroRegime.MILD_BULL: MacroMultipliers(
        stop_loss=1.0, breakeven_activate_pct=0.3, breakeven_floor_pct=0.3,
        trailing_trail=1.0, patience_bars=1.0, allocation=0.8, buy_gate_open=True,
    ),
    MacroRegime.MILD_BEAR: MacroMultipliers(
        stop_loss=0.8, breakeven_activate_pct=0.15, breakeven_floor_pct=0.15,
        trailing_trail=0.7, patience_bars=0.8, allocation=0.5, buy_gate_open=True,
    ),
    MacroRegime.STRONG_BEAR: MacroMultipliers(
        stop_loss=0.6, breakeven_activate_pct=0.1, breakeven_floor_pct=0.1,
        trailing_trail=0.5, patience_bars=0.6, allocation=0.2, buy_gate_open=False,
    ),
}


class MacroRegimeOverlay:
    """Monitors SPY intraday to classify broad market regime.

    Updates each bar batch. Provides multipliers that arena applies to
    stop-loss, breakeven, trailing, patience, and allocation parameters.

    Classification uses:
    - SPY price vs intraday VWAP
    - SPY price vs session open
    - Drawdown from session high
    - 10-bar EMA slope
    """

    STRONG_THRESHOLD_PCT = 0.5  # CFA: bumped from 0.3 to reduce false switches
    DRAWDOWN_THRESHOLD_PCT = 0.5
    EMA_PERIOD = 10
    MIN_REGIME_BARS = 5  # minimum bars in a regime before switching (~5 min)
    MAX_SWITCHES_PER_HOUR = 5  # if exceeded, lock to MILD_BULL (neutral)

    def __init__(self):
        self._open_price: float = 0.0
        self._session_high: float = 0.0
        self._cum_vol: float = 0.0
        self._cum_vol_price: float = 0.0
        self._closes: list[float] = []
        self._regime: MacroRegime = MacroRegime.MILD_BULL
        self._bars_in_regime: int = 0
        self._switch_times: list[float] = []  # monotonic timestamps of switches
        self._regime_history: list[tuple[MacroRegime, int]] = []  # (regime, bar_count)
        # Gradual transition: blend multipliers over N bars
        self._transition_bars_remaining: int = 0
        self._prev_multipliers: MacroMultipliers = MacroMultipliers.for_regime(MacroRegime.MILD_BULL)
        self._target_multipliers: MacroMultipliers = self._prev_multipliers
        self._bar_count: int = 0
        # Emergency override
        self._emergency_bear: bool = False

    @property
    def regime(self) -> MacroRegime:
        if self._emergency_bear:
            return MacroRegime.STRONG_BEAR
        return self._regime

    @property
    def multipliers(self) -> MacroMultipliers:
        if self._emergency_bear:
            return MacroMultipliers.for_regime(MacroRegime.STRONG_BEAR)
        if self._transition_bars_remaining <= 0:
            return self._target_multipliers
        # Blend between prev and target
        t = self._transition_bars_remaining / 3.0  # 3-bar transition
        prev = self._prev_multipliers
        tgt = self._target_multipliers
        return MacroMultipliers(
            stop_loss=prev.stop_loss * t + tgt.stop_loss * (1 - t),
            breakeven_activate_pct=prev.breakeven_activate_pct * t + tgt.breakeven_activate_pct * (1 - t),
            breakeven_floor_pct=prev.breakeven_floor_pct * t + tgt.breakeven_floor_pct * (1 - t),
            trailing_trail=prev.trailing_trail * t + tgt.trailing_trail * (1 - t),
            patience_bars=prev.patience_bars * t + tgt.patience_bars * (1 - t),
            allocation=prev.allocation * t + tgt.allocation * (1 - t),
            buy_gate_open=tgt.buy_gate_open,  # gate is binary, no blend
        )

    def set_emergency_bear(self, active: bool) -> None:
        """Force STRONG_BEAR when portfolio daily P&L < -1%."""
        if active and not self._emergency_bear:
            logger.warning("[MACRO] Emergency STRONG_BEAR override activated (portfolio P&L < -1%)")
        elif not active and self._emergency_bear:
            logger.info("[MACRO] Emergency override cleared")
        self._emergency_bear = active

    def update(self, spy_bar: BarData) -> MacroRegime:
        """Feed a SPY bar and reclassify regime."""
        self._bar_count += 1
        price = spy_bar.close

        # Track session open from first bar
        if self._open_price == 0:
            self._open_price = spy_bar.open

        # Update VWAP
        bar_vol = spy_bar.volume or 1
        self._cum_vol += bar_vol
        self._cum_vol_price += price * bar_vol
        vwap = self._cum_vol_price / self._cum_vol if self._cum_vol > 0 else price

        # Session high
        if price > self._session_high:
            self._session_high = price

        # EMA for slope
        self._closes.append(price)
        ema_slope = self._compute_ema_slope()

        # Drawdown from session high
        drawdown_pct = (self._session_high - price) / self._session_high * 100.0 if self._session_high > 0 else 0.0

        # Price vs open
        open_dev_pct = (price - self._open_price) / self._open_price * 100.0 if self._open_price > 0 else 0.0

        # Above/below VWAP
        above_vwap = price > vwap
        below_vwap = price < vwap

        # Classify
        new_regime = self._classify(above_vwap, below_vwap, open_dev_pct, drawdown_pct, ema_slope)

        # Apply minimum time-in-regime guard
        self._bars_in_regime += 1
        if new_regime != self._regime:
            if self._bars_in_regime < self.MIN_REGIME_BARS:
                # Too soon to switch
                return self._regime

            # Check switch frequency (max 5/hour = 300 bars at 1/min)
            import time as _time
            now = _time.monotonic()
            self._switch_times = [t for t in self._switch_times if now - t < 3600]
            if len(self._switch_times) >= self.MAX_SWITCHES_PER_HOUR:
                # Too many switches — lock to neutral
                if self._regime != MacroRegime.MILD_BULL:
                    logger.warning(
                        f"[MACRO] Too many regime switches ({len(self._switch_times)}/hr), "
                        f"locking to MILD_BULL"
                    )
                    self._set_regime(MacroRegime.MILD_BULL)
                return self._regime

            self._switch_times.append(now)
            self._set_regime(new_regime)
            logger.info(
                f"[MACRO] Regime -> {new_regime.value} "
                f"(SPY={price:.2f} VWAP={vwap:.2f} open_dev={open_dev_pct:+.2f}% "
                f"dd={drawdown_pct:.2f}% ema_slope={'up' if ema_slope > 0 else 'down'})"
            )

        return self._regime

    def _classify(
        self, above_vwap: bool, below_vwap: bool,
        open_dev_pct: float, drawdown_pct: float, ema_slope: float,
    ) -> MacroRegime:
        # STRONG_BULL: above VWAP, well above open, positive momentum
        if above_vwap and open_dev_pct >= self.STRONG_THRESHOLD_PCT and ema_slope > 0:
            return MacroRegime.STRONG_BULL
        # STRONG_BEAR: below VWAP, well below open, significant drawdown
        if below_vwap and open_dev_pct <= -self.STRONG_THRESHOLD_PCT and drawdown_pct >= self.DRAWDOWN_THRESHOLD_PCT:
            return MacroRegime.STRONG_BEAR
        # MILD_BEAR: below VWAP and below open
        if below_vwap and open_dev_pct < 0:
            return MacroRegime.MILD_BEAR
        # MILD_BULL: anything else (above VWAP or above open, but not strongly)
        return MacroRegime.MILD_BULL

    def _set_regime(self, new_regime: MacroRegime) -> None:
        self._prev_multipliers = self.multipliers  # snapshot current blended state
        self._target_multipliers = MacroMultipliers.for_regime(new_regime)
        self._transition_bars_remaining = 3  # gradual over 3 bars
        self._regime = new_regime
        self._bars_in_regime = 0

    def _compute_ema_slope(self) -> float:
        """Compute slope of 10-bar EMA (positive = up, negative = down)."""
        if len(self._closes) < self.EMA_PERIOD + 1:
            return 0.0
        closes = self._closes[-(self.EMA_PERIOD + 5):]  # small buffer
        alpha = 2.0 / (self.EMA_PERIOD + 1)
        ema = closes[0]
        prev_ema = ema
        for p in closes[1:]:
            prev_ema = ema
            ema = alpha * p + (1 - alpha) * ema
        return ema - prev_ema

    def tick_transition(self) -> None:
        """Call once per bar batch to advance gradual transition."""
        if self._transition_bars_remaining > 0:
            self._transition_bars_remaining -= 1

    def reset(self) -> None:
        """Clear all state for new session."""
        self.__init__()
