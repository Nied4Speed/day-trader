"""Volume Profile Reversion Strategy.

Tracks cumulative volume at price levels to build a volume profile.
Buys when price drops to high-volume support (point of control area),
sells when price reaches low-volume resistance zones. Institutional
flow tends to cluster at specific price levels, creating natural
support/resistance.
"""

import logging
from typing import Optional

import numpy as np

from src.core.strategy import BarData, Strategy, TradeSignal

logger = logging.getLogger(__name__)


class VolumeProfileReversionStrategy(Strategy):
    strategy_type = "volume_profile_reversion"

    def __init__(self, name: str, params: Optional[dict] = None):
        self.profile_bars: int = 30  # bars to build volume profile
        self.num_bins: int = 20  # price level buckets
        self.poc_deviation: float = -0.003  # buy within 0.3% of POC
        self.low_volume_threshold: float = 0.3  # sell at levels with <30% of avg volume
        self.allocation_pct: float = 0.25
        self.min_volume_ratio: float = 1.2  # current volume must exceed 1.2x avg
        # Internal state
        self._poc: dict[str, float] = {}  # point of control per symbol
        self._low_vol_zones: dict[str, list[tuple[float, float]]] = {}  # low volume resistance
        super().__init__(name, params)

    def _build_volume_profile(self, history: list[BarData]) -> tuple[float | None, list[tuple[float, float]]]:
        """Build volume profile from recent bars.

        Returns (point_of_control_price, list_of_low_volume_zones).
        Each zone is (zone_low, zone_high).
        """
        if len(history) < self.profile_bars:
            return None, []

        bars = history[-self.profile_bars:]
        prices = []
        volumes = []
        for b in bars:
            # Use typical price weighted by volume
            typical = (b.high + b.low + b.close) / 3.0
            prices.append(typical)
            volumes.append(b.volume)

        price_min = min(b.low for b in bars)
        price_max = max(b.high for b in bars)
        if price_max <= price_min:
            return None, []

        bin_size = (price_max - price_min) / self.num_bins
        if bin_size <= 0:
            return None, []

        # Accumulate volume per bin
        vol_bins = np.zeros(self.num_bins)
        for b in bars:
            typical = (b.high + b.low + b.close) / 3.0
            bin_idx = min(int((typical - price_min) / bin_size), self.num_bins - 1)
            vol_bins[bin_idx] += b.volume

        # Point of control = price level with highest volume
        poc_bin = int(np.argmax(vol_bins))
        poc_price = price_min + (poc_bin + 0.5) * bin_size

        # Low volume zones = bins with significantly less volume
        avg_vol = np.mean(vol_bins)
        low_vol_zones = []
        for i, v in enumerate(vol_bins):
            if v < avg_vol * self.low_volume_threshold:
                zone_low = price_min + i * bin_size
                zone_high = zone_low + bin_size
                low_vol_zones.append((zone_low, zone_high))

        return poc_price, low_vol_zones

    def on_bar(self, bar: BarData) -> Optional[TradeSignal]:
        self.record_bar(bar)
        liq = self.check_liquidation(bar)
        if liq:
            return liq if liq.quantity > 0 else None

        history = self.get_history(bar.symbol)
        if len(history) < self.profile_bars:
            return None

        poc, low_vol_zones = self._build_volume_profile(history)
        if poc is None:
            return None

        self._poc[bar.symbol] = poc
        self._low_vol_zones[bar.symbol] = low_vol_zones

        # Volume filter
        recent_volumes = [b.volume for b in history[-20:]]
        avg_vol = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 1
        vol_ratio = bar.volume / avg_vol if avg_vol > 0 else 0

        # Cache indicators
        poc_deviation = (bar.close - poc) / poc if poc > 0 else 0
        self._indicators[bar.symbol] = {
            "close": bar.close,
            "poc": poc,
            "poc_deviation": poc_deviation,
            "volume_ratio": vol_ratio,
        }

        # Buy: price near POC (support) with decent volume
        if poc_deviation < self.poc_deviation and vol_ratio > self.min_volume_ratio:
            qty = self.compute_quantity(bar.close, self.allocation_pct, symbol=bar.symbol)
            if qty > 0:
                return TradeSignal(symbol=bar.symbol, side="buy", quantity=qty)

        # Sell: price in a low-volume zone (resistance, likely to reverse)
        for zone_low, zone_high in low_vol_zones:
            if zone_low <= bar.close <= zone_high and bar.close > poc:
                qty = self.compute_quantity(bar.close, self.allocation_pct)
                if qty > 0:
                    return TradeSignal(symbol=bar.symbol, side="sell", quantity=qty)
                break

        return None

    def get_params(self) -> dict:
        return {
            "profile_bars": self.profile_bars,
            "num_bins": self.num_bins,
            "poc_deviation": self.poc_deviation,
            "low_volume_threshold": self.low_volume_threshold,
            "allocation_pct": self.allocation_pct,
            "min_volume_ratio": self.min_volume_ratio,
        }

    def set_params(self, params: dict) -> None:
        self.profile_bars = max(15, min(100, int(params.get("profile_bars", self.profile_bars))))
        self.num_bins = max(10, min(50, int(params.get("num_bins", self.num_bins))))
        self.poc_deviation = min(-0.001, float(params.get("poc_deviation", self.poc_deviation)))
        self.low_volume_threshold = max(0.1, min(0.8, float(params.get("low_volume_threshold", self.low_volume_threshold))))
        self.allocation_pct = max(0.05, min(1.0, float(params.get("allocation_pct", self.allocation_pct))))
        self.min_volume_ratio = max(0.5, min(3.0, float(params.get("min_volume_ratio", self.min_volume_ratio))))

    def adapt(self, recent_signals: list, recent_fills: list, realized_pnl: float) -> None:
        old_dev = self.poc_deviation
        if realized_pnl < 0:
            # Tighten: require price closer to POC
            self.poc_deviation -= 0.001
        elif realized_pnl > 0:
            # Loosen slightly
            self.poc_deviation += 0.0005
        self.poc_deviation = max(-0.02, min(-0.001, self.poc_deviation))
        logger.info(
            f"{self.name} adapt: poc_deviation {old_dev:.4f}->{self.poc_deviation:.4f}, "
            f"pnl={realized_pnl:.2f}"
        )

    def reset(self) -> None:
        super().reset()
        self._poc.clear()
        self._low_vol_zones.clear()
