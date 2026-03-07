"""Multi-objective fitness function for model evaluation.

Heavily weights total profit/return as the primary objective.
Sharpe ratio and drawdown serve as tiebreakers - the goal is
maximum profit above all else.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class FitnessScore:
    """Composite fitness score with component breakdown."""

    composite: float
    profit_component: float
    sharpe_component: float
    drawdown_component: float
    frequency_component: float
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
    initial_capital: float = 2_000.0,
    weights: dict | None = None,
) -> FitnessScore:
    """Compute fitness with profit as the dominant objective.

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
        "profit": 0.70,     # profit is king
        "sharpe": 0.15,     # risk-adjusted return as tiebreaker
        "drawdown": 0.15,   # don't blow up completely
        "frequency": 0.00,  # disabled — judge models on returns, not activity
    }
    if weights:
        w.update(weights)

    # Total return percentage
    if equity_curve and initial_capital > 0:
        total_return = (equity_curve[-1] - initial_capital) / initial_capital * 100
    else:
        total_return = 0.0

    # Profit component: sigmoid mapping centered at 0%
    # -10% -> ~0.15, 0% -> 0.5, +5% -> ~0.75, +20% -> ~0.95
    profit_component = 1.0 / (1.0 + math.exp(-total_return / 5.0))

    # Sharpe ratio (annualized from per-bar returns)
    if len(equity_curve) < 2:
        sharpe = 0.0
    else:
        returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
        if np.std(returns) > 0:
            sharpe = (np.mean(returns) / np.std(returns)) * math.sqrt(390)
        else:
            sharpe = 0.0

    sharpe_clamped = max(-2.0, min(4.0, sharpe))
    sharpe_component = (sharpe_clamped + 2.0) / 6.0

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

    drawdown_component = max(0.0, 1.0 - (max_dd * 3.0))  # more lenient

    # Trade frequency
    if session_bars > 0:
        trades_per_bar = trade_count / session_bars
        if trades_per_bar < 0.003:
            frequency_component = trades_per_bar / 0.003 * 0.5
        elif trades_per_bar <= 0.15:
            frequency_component = 1.0
        else:
            frequency_component = max(0.0, 1.0 - (trades_per_bar - 0.15) * 3)
    else:
        frequency_component = 0.0

    # Win rate
    if trade_returns:
        wins = sum(1 for r in trade_returns if r > 0)
        win_rate = wins / len(trade_returns)
    else:
        win_rate = 0.0

    # Composite score
    composite = (
        w["profit"] * profit_component
        + w["sharpe"] * sharpe_component
        + w["drawdown"] * drawdown_component
        + w["frequency"] * frequency_component
    )

    return FitnessScore(
        composite=composite,
        profit_component=profit_component,
        sharpe_component=sharpe_component,
        drawdown_component=drawdown_component,
        frequency_component=frequency_component,
        trade_count=trade_count,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        total_return_pct=total_return,
    )
