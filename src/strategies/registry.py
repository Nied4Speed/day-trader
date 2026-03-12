"""Strategy registry: maps strategy types to classes and handles creation."""

from typing import Optional

from src.core.strategy import Strategy
from src.strategies.bollinger_bands import BollingerBandsStrategy
from src.strategies.ma_crossover import MACrossoverStrategy
from src.strategies.macd_strategy import MACDStrategy
from src.strategies.ml_predictor import MLPredictorStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.rsi_reversion import RSIReversionStrategy
from src.strategies.vwap_reversion import VWAPReversionStrategy
from src.strategies.stochastic import StochasticStrategy
from src.strategies.breakout import BreakoutStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.collab import CollabStrategy
from src.strategies.news_sentiment import NewsSentimentStrategy
from src.strategies.volume_profile_reversion import VolumeProfileReversionStrategy
from src.strategies.volatility_compression import VolatilityCompressionStrategy
from src.strategies.momentum_breakout import MomentumBreakoutStrategy
from src.strategies.atr_breakout_simple import ATRBreakoutSimpleStrategy
from src.strategies.atr_breakout import ATRBreakoutStrategy

STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "ma_crossover": MACrossoverStrategy,
    "rsi_reversion": RSIReversionStrategy,
    "momentum": MomentumStrategy,
    "bollinger_bands": BollingerBandsStrategy,
    "ml_predictor": MLPredictorStrategy,
    "macd": MACDStrategy,
    "vwap_reversion": VWAPReversionStrategy,
    "stochastic": StochasticStrategy,
    "breakout": BreakoutStrategy,
    "mean_reversion": MeanReversionStrategy,
    "news_sentiment": NewsSentimentStrategy,
    "collab": CollabStrategy,
    "volume_profile_reversion": VolumeProfileReversionStrategy,
    "volatility_compression": VolatilityCompressionStrategy,
    "momentum_breakout": MomentumBreakoutStrategy,
    "atr_breakout_simple": ATRBreakoutSimpleStrategy,
    "atr_breakout": ATRBreakoutStrategy,
}

# Dynamically load CFA-generated strategy if it exists on disk
try:
    from src.strategies.cfa_generated import CfaGeneratedStrategy
    STRATEGY_REGISTRY["cfa_generated"] = CfaGeneratedStrategy
except (ImportError, SyntaxError, Exception):
    pass  # CFA hasn't generated one yet, or it has errors — skip silently


def create_strategy(
    strategy_type: str,
    name: str,
    params: Optional[dict] = None,
) -> Strategy:
    """Create a strategy instance by type name."""
    cls = STRATEGY_REGISTRY.get(strategy_type)
    if cls is None:
        raise ValueError(
            f"Unknown strategy type: {strategy_type}. "
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    return cls(name=name, params=params)


def get_strategy_types() -> list[str]:
    """Return all registered strategy type names."""
    return list(STRATEGY_REGISTRY.keys())


def get_default_params(strategy_type: str) -> dict:
    """Get default parameters for a strategy type."""
    strategy = create_strategy(strategy_type, name="__default__")
    return strategy.get_params()
