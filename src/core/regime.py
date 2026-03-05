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
