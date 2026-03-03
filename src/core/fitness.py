"""Multi-objective fitness function for model evaluation.

Uses a composite score combining Sharpe ratio, max drawdown penalty,
trade frequency floor, and expected value per trade. This prevents
degenerate strategies (buy-and-hold, ultra-high-frequency noise)
from dominating the evolutionary selection.

The fitness function is locked before any strategy is written to prevent
changing the goal mid-run and invalidating cross-generation comparisons.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class FitnessScore:
    """Composite fitness score with component breakdown."""

    composite: float
    sharpe_component: float
    drawdown_component: float
    frequency_component: float
    ev_per_trade_component: float
    trade_count: int
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_return_pct: float


def compute_fitness(
    equity_curve: list[float],
    trade_returns: list[float],
    trade_count: int,
    session_bars: int,
    initial_capital: float = 100_000.0,
    weights: dict | None = None,
) -> FitnessScore:
    """Compute multi-objective fitness for a model's session performance.

    Args:
        equity_curve: List of portfolio values at each bar.
        trade_returns: List of per-trade percentage returns.
        trade_count: Number of completed round-trip trades.
        session_bars: Total number of bars in the session.
        initial_capital: Starting capital for return calculation.
        weights: Optional weight overrides for fitness components.

    Returns:
        FitnessScore with composite score and component breakdown.
    """
    w = {
        "sharpe": 0.35,
        "drawdown": 0.25,
        "frequency": 0.15,
        "ev_per_trade": 0.25,
    }
    if weights:
        w.update(weights)

    # Sharpe ratio (annualized from per-bar returns)
    if len(equity_curve) < 2:
        sharpe = 0.0
    else:
        returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
        if np.std(returns) > 0:
            # Annualize assuming 390 bars per day, 252 trading days
            sharpe = (np.mean(returns) / np.std(returns)) * math.sqrt(390)
        else:
            sharpe = 0.0

    # Sharpe component: sigmoid-like mapping, capped at [-2, 4]
    sharpe_clamped = max(-2.0, min(4.0, sharpe))
    sharpe_component = (sharpe_clamped + 2.0) / 6.0  # maps to [0, 1]

    # Max drawdown
    if len(equity_curve) < 2:
        max_dd = 0.0
    else:
        peak = equity_curve[0]
        max_dd = 0.0
        for val in equity_curve:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

    # Drawdown component: lower drawdown = higher score
    drawdown_component = max(0.0, 1.0 - (max_dd * 5.0))  # 20% dd -> score 0

    # Trade frequency: penalize both too few and too many trades
    if session_bars > 0:
        trades_per_bar = trade_count / session_bars
        # Sweet spot: 0.01 to 0.1 trades per bar (roughly 4-39 trades per session)
        if trades_per_bar < 0.005:
            frequency_component = trades_per_bar / 0.005 * 0.5  # too few
        elif trades_per_bar <= 0.1:
            frequency_component = 1.0  # sweet spot
        else:
            frequency_component = max(0.0, 1.0 - (trades_per_bar - 0.1) * 5)  # too many
    else:
        frequency_component = 0.0

    # Expected value per trade
    if trade_returns:
        ev = np.mean(trade_returns)
        # Map EV to [0, 1]: ev of 0% -> 0.5, ev of 2% -> 1.0, ev of -2% -> 0.0
        ev_component = max(0.0, min(1.0, (ev + 0.02) / 0.04))
    else:
        ev_component = 0.0

    # Win rate
    if trade_returns:
        wins = sum(1 for r in trade_returns if r > 0)
        win_rate = wins / len(trade_returns)
    else:
        win_rate = 0.0

    # Total return
    if equity_curve and initial_capital > 0:
        total_return = (equity_curve[-1] - initial_capital) / initial_capital * 100
    else:
        total_return = 0.0

    # Composite score
    composite = (
        w["sharpe"] * sharpe_component
        + w["drawdown"] * drawdown_component
        + w["frequency"] * frequency_component
        + w["ev_per_trade"] * ev_component
    )

    return FitnessScore(
        composite=composite,
        sharpe_component=sharpe_component,
        drawdown_component=drawdown_component,
        frequency_component=frequency_component,
        ev_per_trade_component=ev_component,
        trade_count=trade_count,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        total_return_pct=total_return,
    )
