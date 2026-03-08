"""News Sentiment Strategy: trades purely on news sentiment signals.

Buys when sentiment exceeds a bullish threshold, sells when it drops
below bearish threshold or positions exceed hold duration.
"""

import logging
from typing import Optional

from src.core.strategy import BarData, Strategy, TradeSignal

logger = logging.getLogger(__name__)


class NewsSentimentStrategy(Strategy):
    """Trades based on news sentiment attached to bar data."""

    strategy_type = "news_sentiment"

    def __init__(self, name: str, params: Optional[dict] = None):
        self.bullish_threshold: float = 0.3
        self.bearish_threshold: float = -0.2
        self.allocation_pct: float = 0.20
        self.hold_bars: int = 30  # max bars to hold a position
        self._entry_bar_count: dict[str, int] = {}  # symbol -> bar count at entry
        self._bar_count: int = 0
        super().__init__(name, params)

    def on_bar(self, bar: BarData) -> Optional[TradeSignal]:
        self.record_bar(bar)
        self._bar_count += 1

        # Check liquidation window
        liq = self.check_liquidation(bar)
        if liq:
            if liq.symbol == "__SKIP__":
                return None
            self._entry_bar_count.pop(bar.symbol, None)
            return liq

        sentiment = bar.news_sentiment
        position = self._positions.get(bar.symbol, 0)

        # Cache indicators for watch rule evaluation
        self._indicators[bar.symbol] = {
            "close": bar.close,
            "sentiment": sentiment if sentiment is not None else 0.0,
        }

        # Exit: bearish sentiment or held too long
        if position > 0:
            held = self._bar_count - self._entry_bar_count.get(bar.symbol, 0)
            if (sentiment is not None and sentiment < self.bearish_threshold) or held >= self.hold_bars:
                self._entry_bar_count.pop(bar.symbol, None)
                return TradeSignal(symbol=bar.symbol, side="sell", quantity=position)

        # Entry: bullish sentiment and no position
        if position == 0 and sentiment is not None and sentiment > self.bullish_threshold:
            qty = self.compute_quantity(bar.close, self.allocation_pct)
            if qty > 0:
                self._entry_bar_count[bar.symbol] = self._bar_count
                return TradeSignal(symbol=bar.symbol, side="buy", quantity=qty)

        return None

    def get_params(self) -> dict:
        return {
            "bullish_threshold": self.bullish_threshold,
            "bearish_threshold": self.bearish_threshold,
            "allocation_pct": self.allocation_pct,
            "hold_bars": self.hold_bars,
        }

    def set_params(self, params: dict) -> None:
        if "bullish_threshold" in params:
            self.bullish_threshold = max(0.05, min(0.8, float(params["bullish_threshold"])))
        if "bearish_threshold" in params:
            self.bearish_threshold = max(-0.8, min(-0.01, float(params["bearish_threshold"])))
        if "allocation_pct" in params:
            self.allocation_pct = max(0.05, min(0.50, float(params["allocation_pct"])))
        if "hold_bars" in params:
            self.hold_bars = max(5, min(120, int(params["hold_bars"])))

    def adapt(self, recent_signals: list, recent_fills: list, realized_pnl: float) -> None:
        old_bull, old_bear = self.bullish_threshold, self.bearish_threshold
        if realized_pnl < 0:
            self.bullish_threshold += 0.05
            self.bearish_threshold -= 0.05
        else:
            self.bullish_threshold -= 0.02
        self.bullish_threshold = max(0.1, min(0.8, self.bullish_threshold))
        self.bearish_threshold = max(-0.8, min(-0.1, self.bearish_threshold))
        logger.info(
            f"{self.name} adapt: pnl={realized_pnl:.2f} "
            f"bull {old_bull:.2f}->{self.bullish_threshold:.2f} "
            f"bear {old_bear:.2f}->{self.bearish_threshold:.2f}"
        )
