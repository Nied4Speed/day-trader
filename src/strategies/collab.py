"""COLLAB - Ensemble strategy that combines all 10 strategy types via weighted voting.

This model is permanently protected from culling. Its voting weights evolve
through self-improvement while sub-strategy params come from the best active
models at creation time.
"""

from typing import Optional

from src.core.strategy import BarData, Strategy, TradeSignal

SUB_STRATEGY_TYPES = [
    "ma_crossover",
    "rsi_reversion",
    "momentum",
    "bollinger_bands",
    "ml_predictor",
    "macd",
    "vwap_reversion",
    "stochastic",
    "breakout",
    "mean_reversion",
]


class CollabStrategy(Strategy):
    """Ensemble strategy: collects signals from all 10 strategies, decides by weighted vote."""

    strategy_type = "collab"

    def __init__(self, name: str, params: Optional[dict] = None):
        # Defaults before super().__init__ calls set_params
        self.weights: dict[str, float] = {st: 1.0 for st in SUB_STRATEGY_TYPES}
        self.buy_threshold: float = 0.4
        self.sell_threshold: float = 0.4
        self.allocation_pct: float = 0.30
        self.max_allocation_pct: float = 0.50
        self._sub_strategies: dict[str, Strategy] = {}
        self._eligible_voters: set[str] = set(SUB_STRATEGY_TYPES)  # all eligible by default

        super().__init__(name, params)
        self._init_sub_strategies(params)

    def _init_sub_strategies(self, params: Optional[dict] = None) -> None:
        """Create internal sub-strategy instances."""
        from src.strategies.registry import create_strategy

        sub_params = {}
        if params and "sub_params" in params and isinstance(params["sub_params"], dict):
            sub_params = params["sub_params"]

        for st in SUB_STRATEGY_TYPES:
            sp = sub_params.get(st, {})
            self._sub_strategies[st] = create_strategy(st, f"{self.name}_sub_{st}", params=sp)

    def on_bar(self, bar: BarData) -> Optional[TradeSignal]:
        self.record_bar(bar)
        liq = self.check_liquidation(bar)
        if liq:
            return liq if liq.quantity > 0 else None

        buy_weight = 0.0
        sell_weight = 0.0
        total_weight = sum(
            w for st, w in self.weights.items()
            if st in self._eligible_voters
        )

        for st, sub in self._sub_strategies.items():
            if st not in self._eligible_voters:
                continue  # only top weekly performers vote
            sub.current_capital = self.current_capital
            try:
                signal = sub.on_bar(bar)
            except Exception:
                continue
            w = self.weights.get(st, 1.0)
            if signal:
                if signal.side == "buy":
                    buy_weight += w
                elif signal.side == "sell":
                    sell_weight += w

        if total_weight <= 0:
            return None

        buy_ratio = buy_weight / total_weight
        sell_ratio = sell_weight / total_weight

        # Buy on consensus
        if buy_ratio >= self.buy_threshold and buy_ratio > sell_ratio:
            alloc = self._scale_allocation(buy_ratio, self.buy_threshold)
            qty = self.compute_quantity(bar.close, alloc)
            if qty > 0:
                return TradeSignal(symbol=bar.symbol, side="buy", quantity=qty)

        # Sell on consensus
        if sell_ratio >= self.sell_threshold and sell_ratio > buy_ratio:
            alloc = self._scale_allocation(sell_ratio, self.sell_threshold)
            qty = self.compute_quantity(bar.close, alloc)
            if qty > 0:
                return TradeSignal(symbol=bar.symbol, side="sell", quantity=qty)

        return None

    def set_eligible_voters(self, strategy_types: list[str]) -> None:
        """Set which strategy types are eligible to vote in COLLAB.

        Only the top 5 weekly performers' strategy types should be eligible.
        Strategies not in this set get weight 0 during voting.
        """
        self._eligible_voters = set(strategy_types) if strategy_types else set(SUB_STRATEGY_TYPES)

    def _scale_allocation(self, ratio: float, threshold: float) -> float:
        """Scale allocation from base to max based on consensus strength."""
        if ratio >= 1.0:
            return self.max_allocation_pct
        span = 1.0 - threshold
        if span <= 0:
            return self.allocation_pct
        strength = (ratio - threshold) / span
        return self.allocation_pct + (self.max_allocation_pct - self.allocation_pct) * strength

    def get_params(self) -> dict:
        params: dict = {
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
            "allocation_pct": self.allocation_pct,
            "max_allocation_pct": self.max_allocation_pct,
        }
        # Flatten weights so self-improvement mutation can reach them
        for st, w in self.weights.items():
            params[f"weight_{st}"] = w
        # Sub-strategy params stored nested (not mutated by self-improve)
        params["sub_params"] = {
            st: sub.get_params() for st, sub in self._sub_strategies.items()
        }
        # Persist eligible voters so they survive restarts
        params["_eligible_voters"] = sorted(self._eligible_voters)
        return params

    def set_params(self, params: dict) -> None:
        self.buy_threshold = max(0.1, min(0.9, float(params.get("buy_threshold", self.buy_threshold))))
        self.sell_threshold = max(0.1, min(0.9, float(params.get("sell_threshold", self.sell_threshold))))
        self.allocation_pct = max(0.05, min(1.0, float(params.get("allocation_pct", self.allocation_pct))))
        self.max_allocation_pct = max(
            self.allocation_pct,
            min(1.0, float(params.get("max_allocation_pct", self.max_allocation_pct))),
        )

        # Reconstruct weights from flattened keys
        for st in SUB_STRATEGY_TYPES:
            key = f"weight_{st}"
            if key in params:
                self.weights[st] = max(0.0, float(params[key]))

        # Also accept nested "weights" dict
        if "weights" in params and isinstance(params["weights"], dict):
            for st in SUB_STRATEGY_TYPES:
                if st in params["weights"]:
                    self.weights[st] = max(0.0, float(params["weights"][st]))

        # Update sub-strategy params if provided
        if "sub_params" in params and isinstance(params["sub_params"], dict):
            for st, sp in params["sub_params"].items():
                if st in self._sub_strategies and isinstance(sp, dict):
                    self._sub_strategies[st].set_params(sp)

        # Restore eligible voters if persisted
        if "_eligible_voters" in params and isinstance(params["_eligible_voters"], list):
            self._eligible_voters = set(params["_eligible_voters"])

    def reset(self) -> None:
        super().reset()
        for sub in self._sub_strategies.values():
            sub.reset()
