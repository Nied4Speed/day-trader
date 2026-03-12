"""Post-day CFA-style review powered by Claude.

Gathers the day's trading data, builds a structured prompt, calls Claude Opus
for analysis, and saves the review to DB + markdown file.
"""

import importlib
import inspect
import json
import logging
import os
import pathlib
import textwrap
from datetime import datetime
from typing import Any

from sqlalchemy import func

from src.core.database import (
    Bar,
    CfaReview,
    DailyLedger,
    ModelStatus,
    ModelSummary,
    NewsArticle,
    Order,
    OrderSide,
    OrderStatus,
    PerformanceSnapshot,
    SessionRecord,
    TradingModel,
    get_session,
)

logger = logging.getLogger(__name__)

# Changelog of system changes — most recent first. CFA sees the last 10.
# Prepend new entries when making changes that affect trading behavior.
SYSTEM_CHANGELOG: list[tuple[str, str]] = [
    ("2026-03-12", "DailyContext on every bar — strategies now receive bar.daily_context (daily open/high/low/VWAP, prev close, change_from_open_pct, daily_range_position) and bar.spy_daily_context (SPY as market trend proxy). Screener additions seeded from Alpaca snapshots. Zero extra API calls."),
    ("2026-03-11", "Added profit stagnation exit — sells positions +0.5% to <TP that go flat for 15+ bars"),
    ("2026-03-11", "Added position-aware sizing — compute_quantity tapers per-symbol exposure to prevent oversized positions"),
    ("2026-03-11", "New strategies: volume_profile_reversion (POC support/resistance), volatility_compression (BB squeeze breakout)"),
    ("2026-03-11", "Independent model trading — removed position manager netting, each model submits own orders"),
    ("2026-03-11", "Trailing TP — take_profit_pct now activates trailing stop tiers instead of hard sell"),
    ("2026-03-11", "Fixed arena stop endpoint crash — safe attribute access on detached SQLAlchemy objects"),
    ("2026-03-10", "Added ratcheting trailing stops (trailing_stop_tiers) — trail tightens as gain grows"),
    ("2026-03-10", "Added patience stops (patience_stop_tiers) — exits slow-bleed losers after N bars underwater"),
    ("2026-03-10", "CFA-seeded initial trailing_stop_tiers and patience_stop_tiers for all 13 models"),
    ("2026-03-10", "Fixed circuit breaker blocking sells — all guards now buy-only, sells always reach Alpaca"),
    ("2026-03-10", "Fixed wash trade cooldown blocking sells — cooldown now buy-only"),
    ("2026-03-10", "Fixed position symbols dropped from feed — session start merges held symbols into universe"),
    ("2026-03-10", "Added single-stock leveraged ETFs (NVD, TSLL, MSTU, CONL, etc.) to screener blocklist"),
    ("2026-03-10", "Added mission control health checks — data flow, subscription integrity, sell path, model state"),
    ("2026-03-09", "Wired up stop-losses (were dead code previously), fixed unit mismatch (0.02 vs 2.0)"),
    ("2026-03-09", "Fixed position duplication from shared Alpaca account reconciliation"),
    ("2026-03-09", "Fixed P&L tracking — equity = capital + position_cost + unrealized"),
    ("2026-03-09", "Added sell qty clamping to prevent overselling from shared Alpaca account"),
    ("2026-03-08", "CFA enriched with news sentiment, sector analysis, options data, corporate actions"),
    ("2026-03-06", "Fixed post-session freeze (freeze bug #3) — stream close timeout + timer await"),
    ("2026-03-06", "Capital tracking fixed — daily ledger saved BEFORE self-improvement"),
    ("2026-03-05", "Fixed event loop freeze #1 — bar batching, async trade updates"),
]


def _gather_bar_summaries(db, session_date: str) -> dict[str, Any]:
    """Build market data summaries from stored bars for the given date.

    Returns:
        - daily_stats: per-symbol summary (open, high, low, close, volume, pct_change)
        - intraday_30min: 30-minute aggregated OHLCV for core + most-active symbols
    """
    from sqlalchemy import func as sqla_func

    # All bars for this date
    bars = (
        db.query(Bar)
        .filter(sqla_func.date(Bar.timestamp) == session_date)
        .order_by(Bar.symbol, Bar.timestamp)
        .all()
    )

    if not bars:
        return {"daily_stats": {}, "intraday_30min": {}}

    # Group by symbol
    by_symbol: dict[str, list] = {}
    for b in bars:
        by_symbol.setdefault(b.symbol, []).append(b)

    # Build all daily stats first (needed for ranking)
    all_daily: dict[str, dict] = {}
    for sym, sym_bars in by_symbol.items():
        day_open = sym_bars[0].open
        day_close = sym_bars[-1].close
        day_high = max(b.high for b in sym_bars)
        day_low = min(b.low for b in sym_bars)
        day_volume = sum(b.volume for b in sym_bars)
        pct_change = round((day_close - day_open) / day_open * 100, 4) if day_open else 0.0
        intraday_range = round((day_high - day_low) / day_open * 100, 4) if day_open else 0.0

        # Microstructure stats
        # avg_bar_range_pct: mean (high-low)/open per bar — intrabar volatility proxy
        bar_ranges = [(b.high - b.low) / b.open * 100 for b in sym_bars if b.open > 0]
        avg_bar_range = round(sum(bar_ranges) / len(bar_ranges), 4) if bar_ranges else 0.0

        # volume_concentration: which 30-min window had peak volume
        vol_by_window: dict[str, int] = {}
        for b in sym_bars:
            window_min = (b.timestamp.minute // 30) * 30
            window_key = f"{b.timestamp.hour:02d}:{window_min:02d}"
            vol_by_window[window_key] = vol_by_window.get(window_key, 0) + b.volume
        peak_volume_window = max(vol_by_window, key=vol_by_window.get) if vol_by_window else None

        # trend_strength: (close-open)/intraday_range — how directional was the day
        # +1 = pure uptrend, -1 = pure downtrend, ~0 = choppy
        intraday_abs_range = day_high - day_low
        trend_strength = round(
            (day_close - day_open) / intraday_abs_range, 4
        ) if intraday_abs_range > 0 else 0.0

        all_daily[sym] = {
            "open": round(day_open, 2),
            "high": round(day_high, 2),
            "low": round(day_low, 2),
            "close": round(day_close, 2),
            "volume": day_volume,
            "pct_change": pct_change,
            "intraday_range_pct": intraday_range,
            "bar_count": len(sym_bars),
            "avg_bar_range_pct": avg_bar_range,
            "peak_volume_window": peak_volume_window,
            "trend_strength": trend_strength,
        }

    core_symbols = {
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "META", "TSLA", "JPM", "V", "SPY",
    }
    # Top 20 non-core symbols by volume
    dynamic_by_vol = sorted(
        ((sym, stats["volume"]) for sym, stats in all_daily.items() if sym not in core_symbols),
        key=lambda x: x[1],
        reverse=True,
    )
    top_dynamic = {sym for sym, _ in dynamic_by_vol[:20]}

    # Daily stats: core + top 20 dynamic (not all 93)
    included_symbols = core_symbols | top_dynamic
    daily_stats = {sym: stats for sym, stats in all_daily.items() if sym in included_symbols}

    # 30-min aggregated bars for core + top 5 dynamic
    intraday_symbols = core_symbols | {sym for sym, _ in dynamic_by_vol[:5]}

    intraday_30min: dict[str, list] = {}
    for sym in intraday_symbols:
        sym_bars = by_symbol.get(sym, [])
        if not sym_bars:
            continue
        # Filter to market hours only (4:00 AM - 8:00 PM ET covers extended hours)
        sym_bars = [b for b in sym_bars if 4 <= b.timestamp.hour < 20]
        if not sym_bars:
            continue

        # Aggregate into 30-min windows
        windows: list[dict] = []
        window_bars: list = []
        window_start = None

        for b in sym_bars:
            ts = b.timestamp
            # 30-min window: floor to nearest 30 min
            minute = ts.minute
            window_minute = (minute // 30) * 30
            window_key = ts.replace(minute=window_minute, second=0, microsecond=0)

            if window_start is None:
                window_start = window_key

            if window_key != window_start and window_bars:
                # Flush window
                windows.append({
                    "time": window_start.strftime("%H:%M"),
                    "open": round(window_bars[0].open, 2),
                    "high": round(max(wb.high for wb in window_bars), 2),
                    "low": round(min(wb.low for wb in window_bars), 2),
                    "close": round(window_bars[-1].close, 2),
                    "volume": sum(wb.volume for wb in window_bars),
                })
                window_bars = []
                window_start = window_key

            window_bars.append(b)

        # Flush last window
        if window_bars:
            windows.append({
                "time": window_start.strftime("%H:%M"),
                "open": round(window_bars[0].open, 2),
                "high": round(max(wb.high for wb in window_bars), 2),
                "low": round(min(wb.low for wb in window_bars), 2),
                "close": round(window_bars[-1].close, 2),
                "volume": sum(wb.volume for wb in window_bars),
            })

        intraday_30min[sym] = windows

    return {
        "daily_stats": daily_stats,
        "intraday_30min": intraday_30min,
    }


def _gather_news_sentiment(
    db, session_date: str, symbol_stats: dict[str, dict]
) -> dict[str, Any]:
    """Aggregate news sentiment from NewsArticle table for the session date.

    Returns per-symbol sentiment, market-wide stats, and sentiment-vs-PnL flags.
    """
    from sqlalchemy import func as sqla_func

    articles = (
        db.query(NewsArticle)
        .filter(sqla_func.date(NewsArticle.published_at) == session_date)
        .all()
    )

    if not articles:
        # Also try fetched_at in case published_at is null
        articles = (
            db.query(NewsArticle)
            .filter(sqla_func.date(NewsArticle.fetched_at) == session_date)
            .all()
        )

    if not articles:
        return {"per_symbol": {}, "market_wide": {}, "sentiment_pnl_divergence": []}

    # Per-symbol aggregation
    sym_scores: dict[str, list[float]] = {}
    sym_headlines: dict[str, str] = {}  # first/top headline per symbol
    for article in articles:
        syms = article.symbols or []
        if isinstance(syms, str):
            try:
                syms = json.loads(syms)
            except (json.JSONDecodeError, TypeError):
                syms = []
        for sym in syms:
            sym_scores.setdefault(sym, []).append(article.sentiment_score)
            if sym not in sym_headlines:
                sym_headlines[sym] = (article.headline or "")[:120]

    per_symbol: dict[str, dict] = {}
    for sym, scores in sym_scores.items():
        per_symbol[sym] = {
            "avg_sentiment": round(sum(scores) / len(scores), 4),
            "article_count": len(scores),
            "top_headline": sym_headlines.get(sym, ""),
        }

    # Market-wide stats
    all_scores = [a.sentiment_score for a in articles]
    bullish = sum(1 for s in all_scores if s > 0.1)
    bearish = sum(1 for s in all_scores if s < -0.1)
    neutral = len(all_scores) - bullish - bearish

    market_wide = {
        "total_articles": len(articles),
        "avg_sentiment": round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0,
        "bullish_count": bullish,
        "bearish_count": bearish,
        "neutral_count": neutral,
    }

    # Sentiment-vs-PnL divergence: flag symbols where sentiment and PnL disagree
    divergence: list[dict] = []
    for sym, sym_data in per_symbol.items():
        if sym not in symbol_stats:
            continue
        pnl = symbol_stats[sym].get("realized_pnl", 0.0)
        sentiment = sym_data["avg_sentiment"]
        # Bullish sentiment but negative PnL, or bearish sentiment but positive PnL
        if (sentiment > 0.2 and pnl < -1.0) or (sentiment < -0.2 and pnl > 1.0):
            divergence.append({
                "symbol": sym,
                "sentiment": sentiment,
                "realized_pnl": round(pnl, 2),
                "interpretation": "bullish sentiment + losses" if sentiment > 0 else "bearish sentiment + gains",
            })

    return {
        "per_symbol": per_symbol,
        "market_wide": market_wide,
        "sentiment_pnl_divergence": divergence,
    }


def _gather_cfa_memory(db_path: str, session_date: str, max_reviews: int = 5) -> list[dict]:
    """Query prior CFA reviews and cross-reference with next-day outcomes.

    Returns a list of dicts, most recent first, each containing:
    - date, grade, key action items, param rec summary
    - next_day_outcome: whether the portfolio improved/declined the day after
    - strategy_evolution fields if present
    """
    db = get_session(db_path)
    try:
        prior_reviews = (
            db.query(CfaReview)
            .filter(CfaReview.session_date < session_date, CfaReview.review_json.isnot(None))
            .order_by(CfaReview.session_date.desc())
            .limit(max_reviews)
            .all()
        )
        if not prior_reviews:
            return []

        memory: list[dict] = []
        for review in prior_reviews:
            rj = review.review_json or {}

            # Extract high-priority action items (top 5)
            action_items = rj.get("action_items", [])
            high_actions = [
                a.get("action", "") for a in action_items
                if isinstance(a, dict) and a.get("priority") == "high"
            ][:5]

            # Summarize param recs: model_name -> {param: value}
            param_recs = rj.get("parameter_recommendations", [])
            param_summary = {}
            for rec in (param_recs if isinstance(param_recs, list) else []):
                if isinstance(rec, dict) and rec.get("model_name"):
                    params = rec.get("recommendations", {})
                    if isinstance(params, dict):
                        param_summary[rec["model_name"]] = {
                            k: v.get("value") if isinstance(v, dict) else v
                            for k, v in params.items()
                        }

            # Cross-reference: what happened the day AFTER this review?
            next_day_outcome = None
            next_day_ledgers = (
                db.query(DailyLedger)
                .filter(DailyLedger.session_date > review.session_date)
                .order_by(DailyLedger.session_date.asc())
                .limit(25)  # up to 25 models for one day
                .all()
            )
            if next_day_ledgers:
                next_date = next_day_ledgers[0].session_date
                same_day = [l for l in next_day_ledgers if l.session_date == next_date]
                total_return = sum(l.daily_return_pct for l in same_day)
                avg_return = total_return / len(same_day) if same_day else 0

                # Per-model results so CFA can see how its specific recs played out
                model_results = {}
                for ledger in same_day:
                    m = db.query(TradingModel).filter(TradingModel.id == ledger.model_id).first()
                    if m:
                        model_results[m.name] = {
                            "return_pct": round(ledger.daily_return_pct, 4),
                            "trades": ledger.total_trades,
                        }

                next_day_outcome = {
                    "date": next_date,
                    "avg_return_pct": round(avg_return, 4),
                    "model_count": len(same_day),
                    "verdict": "improved" if avg_return > 0 else "declined",
                    "per_model": model_results,
                }

            entry = {
                "date": review.session_date,
                "grade": rj.get("portfolio_grade", "?"),
                "executive_summary": (rj.get("executive_summary", "") or "")[:200],
                "high_priority_actions": high_actions,
                "param_recs_summary": param_summary,
                "strategy_evolution": rj.get("strategy_evolution"),
                "roster_changes": rj.get("roster_changes"),
                "next_day_outcome": next_day_outcome,
            }
            memory.append(entry)

        return memory
    finally:
        db.close()


def _gather_review_data(
    db_path: str,
    session_date: str,
    lookback_days: int = 10,
) -> dict[str, Any]:
    """Query DB and pre-aggregate data for the review prompt."""
    db = get_session(db_path)
    try:
        # --- Models ---
        models = db.query(TradingModel).filter(
            TradingModel.status.in_(["active", "ACTIVE"])
        ).all()
        model_map = {m.id: m for m in models}

        # --- Latest snapshots for today ---
        model_rows = []
        for m in models:
            snap = (
                db.query(PerformanceSnapshot)
                .filter(
                    PerformanceSnapshot.model_id == m.id,
                    PerformanceSnapshot.session_date == session_date,
                )
                .order_by(PerformanceSnapshot.timestamp.desc())
                .first()
            )
            return_pct = snap.return_pct if snap else 0.0
            sharpe = snap.sharpe_ratio if snap else 0.0
            drawdown = snap.max_drawdown if snap else 0.0
            win_rate = snap.win_rate if snap else 0.0
            trades = snap.total_trades if snap else 0

            # Get fitness from model summary if available
            summary = (
                db.query(ModelSummary)
                .filter(
                    ModelSummary.model_id == m.id,
                    ModelSummary.session_date == session_date,
                    ModelSummary.summary_type == "post_session",
                )
                .order_by(ModelSummary.created_at.desc())
                .first()
            )
            fitness = summary.fitness if summary else None
            rank = summary.rank if summary else None

            # Use daily_ledger for accurate capital (m.current_capital resets)
            ledger = (
                db.query(DailyLedger)
                .filter(
                    DailyLedger.model_id == m.id,
                    DailyLedger.session_date == session_date,
                )
                .first()
            )
            start_cap = ledger.start_capital if ledger else m.initial_capital
            daily_ret = ledger.daily_return_pct if ledger else return_pct

            # Per-model realized PnL from orders (permanent, survives resets)
            model_realized = (
                db.query(func.sum(Order.realized_pnl))
                .filter(
                    Order.model_id == m.id,
                    Order.session_date == session_date,
                    Order.status == OrderStatus.FILLED,
                    Order.side == OrderSide.SELL,
                )
                .scalar() or 0.0
            )

            # Compute equity from realized PnL (not raw cash, which excludes
            # capital tied up in positions). After EOD liquidation all positions
            # are closed, so equity = start + realized_pnl.
            end_cap = start_cap + model_realized

            # Aggregate exit reasons for this model's sells today
            reason_rows = (
                db.query(Order.signal_reason, func.count(Order.id))
                .filter(
                    Order.model_id == m.id,
                    Order.session_date == session_date,
                    Order.status == OrderStatus.FILLED,
                    Order.side == OrderSide.SELL,
                    Order.signal_reason.isnot(None),
                )
                .group_by(Order.signal_reason)
                .all()
            )
            exit_reasons = {reason: count for reason, count in reason_rows}

            # Parse model parameters for risk control visibility
            params = {}
            if m.parameters:
                try:
                    params = json.loads(m.parameters) if isinstance(m.parameters, str) else m.parameters
                except (json.JSONDecodeError, TypeError):
                    pass

            model_rows.append({
                "id": m.id,
                "name": m.name,
                "strategy_type": m.strategy_type,
                "generation": m.generation,
                "return_pct": round(daily_ret, 4),
                "sharpe": round(sharpe, 4) if sharpe else 0.0,
                "max_drawdown": round(drawdown, 4),
                "win_rate": round(win_rate, 4),
                "trades": trades,
                "fitness": round(fitness, 4) if fitness else None,
                "rank": rank,
                "start_capital": round(start_cap, 2),
                "end_capital": round(end_cap, 2),
                "realized_pnl": round(model_realized, 2),
                "stop_loss_pct": params.get("stop_loss_pct"),
                "take_profit_pct": params.get("take_profit_pct"),
                "all_parameters": {
                    k: v for k, v in params.items()
                    if not k.startswith("_") and isinstance(v, (int, float, bool, list))
                },
                "exit_reasons": exit_reasons,
            })

        model_rows.sort(key=lambda r: r["return_pct"], reverse=True)

        # --- Reflections for top/bottom 3 ---
        top3_ids = [r["id"] for r in model_rows[:3]]
        bot3_ids = [r["id"] for r in model_rows[-3:] if r["trades"] > 0] or [
            r["id"] for r in model_rows[-3:]
        ]
        reflections = {}
        for mid in set(top3_ids + bot3_ids):
            s = (
                db.query(ModelSummary)
                .filter(
                    ModelSummary.model_id == mid,
                    ModelSummary.session_date == session_date,
                    ModelSummary.summary_type == "post_session",
                )
                .order_by(ModelSummary.created_at.desc())
                .first()
            )
            if s:
                reflections[mid] = s.reflection[:500]

        # --- Orders (today) ---
        orders = (
            db.query(Order)
            .filter(Order.session_date == session_date)
            .all()
        )

        filled = [o for o in orders if o.status == OrderStatus.FILLED]
        rejected = [o for o in orders if o.status == OrderStatus.REJECTED]

        sells_with_pnl = [
            o for o in filled
            if o.side == OrderSide.SELL and o.realized_pnl is not None
        ]
        wins = [o for o in sells_with_pnl if o.realized_pnl > 0]
        losses = [o for o in sells_with_pnl if o.realized_pnl <= 0]

        # Per-symbol aggregation
        symbol_stats: dict[str, dict] = {}
        for o in filled:
            sym = o.symbol
            if sym not in symbol_stats:
                symbol_stats[sym] = {"trades": 0, "realized_pnl": 0.0, "volume": 0.0}
            symbol_stats[sym]["trades"] += 1
            if o.realized_pnl is not None:
                symbol_stats[sym]["realized_pnl"] += o.realized_pnl
            if o.fill_price and o.fill_quantity:
                symbol_stats[sym]["volume"] += o.fill_price * o.fill_quantity

        # Rejected order reasons
        reject_reasons: dict[str, int] = {}
        for o in rejected:
            reason = o.rejected_reason or "unknown"
            # Truncate long reasons
            reason = reason[:80]
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1

        # --- Sessions ---
        sessions = (
            db.query(SessionRecord)
            .filter(SessionRecord.session_date == session_date)
            .order_by(SessionRecord.session_number.asc())
            .all()
        )

        # --- Strategy type aggregation ---
        strat_types: dict[str, dict] = {}
        for r in model_rows:
            st = r["strategy_type"]
            if st not in strat_types:
                strat_types[st] = {
                    "count": 0, "avg_return": 0.0, "total_trades": 0,
                    "models": [],
                }
            strat_types[st]["count"] += 1
            strat_types[st]["avg_return"] += r["return_pct"]
            strat_types[st]["total_trades"] += r["trades"]
            strat_types[st]["models"].append(r["name"])
        for st in strat_types:
            n = strat_types[st]["count"]
            if n:
                strat_types[st]["avg_return"] = round(
                    strat_types[st]["avg_return"] / n, 4
                )

        # --- Parameter drift from improvement summaries ---
        improvements = (
            db.query(ModelSummary)
            .filter(
                ModelSummary.session_date == session_date,
                ModelSummary.summary_type == "post_improvement",
            )
            .all()
        )
        param_changes = []
        for imp in improvements:
            if imp.param_changes:
                m = model_map.get(imp.model_id)
                param_changes.append({
                    "model": m.name if m else f"model_{imp.model_id}",
                    "changes": imp.param_changes,
                })

        # --- Multi-day trends ---
        multi_day = []
        ledger_entries = (
            db.query(DailyLedger)
            .filter(DailyLedger.session_date <= session_date)
            .order_by(DailyLedger.session_date.desc())
            .limit(lookback_days * 25)  # up to 25 models * N days
            .all()
        )

        # Group by date
        by_date: dict[str, list] = {}
        for entry in ledger_entries:
            d = entry.session_date
            if d not in by_date:
                by_date[d] = []
            m = model_map.get(entry.model_id)
            by_date[d].append({
                "model": m.name if m else f"model_{entry.model_id}",
                "daily_return": round(entry.daily_return_pct, 4),
                "cumulative_return": round(entry.cumulative_return_pct, 4),
                "trades": entry.total_trades,
            })

        # Build per-date aggregates
        sorted_dates = sorted(by_date.keys(), reverse=True)[:lookback_days]
        for d in sorted_dates:
            entries = by_date[d]
            returns = [e["daily_return"] for e in entries]
            multi_day.append({
                "date": d,
                "model_count": len(entries),
                "mean_return": round(sum(returns) / len(returns), 4) if returns else 0.0,
                "best_return": round(max(returns), 4) if returns else 0.0,
                "worst_return": round(min(returns), 4) if returns else 0.0,
                "total_trades": sum(e["trades"] for e in entries),
            })

        # --- CFA-generated model data (if it exists) ---
        cfa_model_data = None
        cfa_model = (
            db.query(TradingModel)
            .filter(
                TradingModel.name == "cfa_analyst",
                TradingModel.status.in_(["active", "ACTIVE"]),
            )
            .first()
        )
        if cfa_model:
            cfa_row = next(
                (r for r in model_rows if r["id"] == cfa_model.id), None
            )
            # Read current source code
            cfa_source_path = pathlib.Path(__file__).resolve().parent.parent / "strategies" / "cfa_generated.py"
            cfa_source = ""
            if cfa_source_path.exists():
                try:
                    cfa_source = cfa_source_path.read_text()
                except Exception:
                    pass
            cfa_model_data = {
                "performance": cfa_row,
                "current_source": cfa_source,
                "parameters": cfa_model.parameters or {},
            }

        # --- Inactive models ---
        inactive = [r for r in model_rows if r["trades"] == 0]

        # --- Aggregate portfolio ---
        total_start = sum(r["start_capital"] for r in model_rows)
        total_end = sum(r["end_capital"] for r in model_rows)
        returns = [r["return_pct"] for r in model_rows]
        total_realized = sum(
            o.realized_pnl for o in sells_with_pnl
        )

        # Alpaca ground-truth P&L from session records
        alpaca_pnl_total = None
        for s in sessions:
            if s.alpaca_pnl is not None:
                alpaca_pnl_total = (alpaca_pnl_total or 0.0) + s.alpaca_pnl
        unattributed_pnl = None
        if alpaca_pnl_total is not None:
            unattributed_pnl = round(alpaca_pnl_total - total_realized, 2)

        # Core vs dynamic symbols (core list from config)
        core_symbols = {
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
            "META", "TSLA", "JPM", "V", "SPY",
        }
        core_pnl = sum(
            v["realized_pnl"] for k, v in symbol_stats.items() if k in core_symbols
        )
        dynamic_pnl = sum(
            v["realized_pnl"] for k, v in symbol_stats.items() if k not in core_symbols
        )

        # --- News Sentiment ---
        news_sentiment = _gather_news_sentiment(db, session_date, symbol_stats)

        # --- Sector Analysis ---
        try:
            from src.data.sectors import compute_sector_stats
            sector_analysis = compute_sector_stats(db_path, session_date)
        except Exception:
            logger.debug("Failed to compute sector stats", exc_info=True)
            sector_analysis = {}

        # --- Options Data (IV + Greeks) ---
        try:
            from src.data.options import fetch_options_summary
            from src.core.config import Config
            cfg = Config.load()
            # Fetch for core symbols + top traded dynamic symbols
            top_traded = [sym for sym, _ in sorted(
                symbol_stats.items(), key=lambda x: x[1]["trades"], reverse=True
            )[:15]]
            options_symbols = list(set(list(core_symbols) + top_traded))
            options_data = fetch_options_summary(options_symbols, cfg)
        except Exception:
            logger.debug("Failed to fetch options data", exc_info=True)
            options_data = {}

        # --- Corporate Actions / Upcoming Events ---
        try:
            from src.data.corporate_actions import fetch_upcoming_events
            from src.core.config import Config
            cfg = Config.load()
            all_traded_symbols = list(symbol_stats.keys()) + list(core_symbols)
            upcoming_events = fetch_upcoming_events(
                list(set(all_traded_symbols)), cfg, lookahead_days=5
            )
        except Exception:
            logger.debug("Failed to fetch corporate actions", exc_info=True)
            upcoming_events = []

        return {
            "session_date": session_date,
            "portfolio": {
                "total_models": len(model_rows),
                "start_capital": round(total_start, 2),
                "end_capital": round(total_end, 2),
                "aggregate_pnl": round(total_end - total_start, 2),
                "aggregate_return_pct": round(
                    (total_end - total_start) / total_start * 100
                    if total_start else 0.0, 4
                ),
                "mean_return_pct": round(
                    sum(returns) / len(returns) if returns else 0.0, 4
                ),
                "median_return_pct": round(
                    sorted(returns)[len(returns) // 2] if returns else 0.0, 4
                ),
                "total_realized_pnl": round(total_realized, 2),
                "alpaca_pnl": round(alpaca_pnl_total, 2) if alpaca_pnl_total is not None else None,
                "unattributed_pnl": unattributed_pnl,
            },
            "models": model_rows,
            "reflections": {
                mid: text for mid, text in reflections.items()
            },
            "symbols": {
                sym: {
                    "trades": v["trades"],
                    "realized_pnl": round(v["realized_pnl"], 2),
                    "volume": round(v["volume"], 2),
                    "is_core": sym in core_symbols,
                }
                for sym, v in sorted(
                    symbol_stats.items(),
                    key=lambda x: abs(x[1]["realized_pnl"]),
                    reverse=True,
                )
            },
            "trade_quality": {
                "total_filled": len(filled),
                "total_sells_with_pnl": len(sells_with_pnl),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": round(
                    len(wins) / len(sells_with_pnl) * 100 if sells_with_pnl else 0.0, 2
                ),
                "avg_win": round(
                    sum(o.realized_pnl for o in wins) / len(wins) if wins else 0.0, 4
                ),
                "avg_loss": round(
                    sum(o.realized_pnl for o in losses) / len(losses)
                    if losses else 0.0, 4
                ),
                "largest_win": round(
                    max((o.realized_pnl for o in wins), default=0.0), 4
                ),
                "largest_loss": round(
                    min((o.realized_pnl for o in losses), default=0.0), 4
                ),
                "core_pnl": round(core_pnl, 2),
                "dynamic_pnl": round(dynamic_pnl, 2),
            },
            "risk_events": {
                "rejected_orders": len(rejected),
                "reject_reasons": dict(
                    sorted(reject_reasons.items(), key=lambda x: -x[1])[:20]
                ),
            },
            "sessions": [
                {
                    "number": s.session_number,
                    "started": s.started_at.isoformat() if s.started_at else None,
                    "ended": s.ended_at.isoformat() if s.ended_at else None,
                    "bars": s.total_bars,
                    "trades": s.total_trades,
                }
                for s in sessions
            ],
            "strategy_types": strat_types,
            "param_drift": param_changes[:10],  # cap to keep prompt manageable
            "multi_day": multi_day,
            "inactive_models": [
                {"name": r["name"], "strategy_type": r["strategy_type"]}
                for r in inactive
            ],
            "cfa_model": cfa_model_data,
            "market_data": _gather_bar_summaries(db, session_date),
            "news_sentiment": news_sentiment,
            "sector_analysis": sector_analysis,
            "options_data": options_data,
            "upcoming_events": upcoming_events,
            "cfa_memory": _gather_cfa_memory(db_path, session_date),
        }
    finally:
        db.close()


def _build_cfa_model_section(data: dict[str, Any]) -> str:
    """Build the CFA model context section for the prompt."""
    cfa = data.get("cfa_model")
    if not cfa:
        return "### Your previous strategy\nThis is the FIRST time you are generating a strategy. There is no previous code.\nState your hypothesis for the initial design."

    lines = ["### Your previous strategy"]
    perf = cfa.get("performance")
    if perf:
        lines.append(f"Return: {perf.get('return_pct', 0)}%, Trades: {perf.get('trades', 0)}, "
                      f"Win rate: {perf.get('win_rate', 0)}%, Realized PnL: ${perf.get('realized_pnl', 0)}")

    # Show prior evolution hypothesis if available (from cfa_memory)
    cfa_mem = data.get("cfa_memory", [])
    for entry in cfa_mem:
        evolution = entry.get("strategy_evolution")
        if evolution and isinstance(evolution, dict):
            hypo = evolution.get("hypothesis", "")
            predicted = evolution.get("predicted_performance", {})
            if hypo:
                lines.append(f"\nYour prior hypothesis ({entry['date']}): {hypo}")
            if predicted:
                lines.append(f"Your prediction: win_rate={predicted.get('expected_win_rate_pct', '?')}%, "
                             f"avg_pnl=${predicted.get('expected_avg_pnl_per_trade', '?')}")
            if perf:
                lines.append(f"Actual result: win_rate={perf.get('win_rate', '?')}%, "
                             f"realized_pnl=${perf.get('realized_pnl', '?')}")
                lines.append("Evaluate: did your hypothesis hold? What needs to change?")
            break  # Only show most recent evolution

    source = cfa.get("current_source", "")
    if source:
        lines.append("\n```python")
        lines.append(source)
        lines.append("```")
        lines.append("\nMake at most 3 targeted changes. State what you're changing and why.")
    else:
        lines.append("Previous source code not found. Generate fresh.")

    return "\n".join(lines)


def _build_changelog_section(since_date: str | None = None) -> str:
    """Build a changelog section, filtered to entries since last review date."""
    if since_date:
        entries = [(d, desc) for d, desc in SYSTEM_CHANGELOG if d > since_date]
    else:
        entries = SYSTEM_CHANGELOG[:10]
    if not entries:
        return ""
    lines = ["## Recent System Changes (since last review)"]
    lines.append("")
    lines.append("Evaluate whether each change helped or hurt. Reference in `changelog_assessment`.")
    lines.append("")
    for date, description in entries:
        lines.append(f"- **{date}**: {description}")
    lines.append("")
    return "\n".join(lines)


def _build_cfa_memory_section(data: dict[str, Any]) -> str:
    """Build the Prior Reviews & Accountability prompt section from cfa_memory."""
    memory = data.get("cfa_memory", [])
    if not memory:
        return ""

    lines = [
        "## Prior Reviews & Accountability",
        "",
        "You have access to your last reviews WITH next-day outcomes showing exactly how each",
        "model you recommended changes for actually performed. For each prior review:",
        "1. Check the per-model results — did models you tuned improve or get worse?",
        "2. Did roster changes you recommended (replacements, probation) pan out?",
        "3. Were your performance predictions (win rate, avg P&L) accurate?",
        "4. Course-correct failures. Do NOT repeat recommendations that clearly didn't work",
        "   without explaining what's different this time.",
        "5. Double down on recommendations that worked — similar logic, tighter params.",
        "",
    ]
    for entry in memory:
        date = entry.get("date", "?")
        grade = entry.get("grade", "?")
        lines.append(f"### Review: {date} (Grade: {grade})")
        summary = entry.get("executive_summary", "")
        if summary:
            lines.append(f"Summary: {summary}")

        actions = entry.get("high_priority_actions", [])
        if actions:
            lines.append("High-priority actions you recommended:")
            for a in actions:
                lines.append(f"  - {a}")

        param_summary = entry.get("param_recs_summary", {})
        if param_summary:
            lines.append("Parameter changes you recommended:")
            for model_name, params in param_summary.items():
                if isinstance(params, dict):
                    param_strs = [f"{k}={v}" for k, v in params.items()]
                    lines.append(f"  - {model_name}: {', '.join(param_strs)}")
                else:
                    lines.append(f"  - {model_name}: {params}")

        roster = entry.get("roster_changes")
        if roster and isinstance(roster, dict):
            replacements = roster.get("replace", [])
            if replacements:
                lines.append("Roster changes you recommended:")
                for r in replacements:
                    if isinstance(r, dict):
                        remove = r.get("remove", "?")
                        repl = r.get("replacement_type") or r.get("replacement_strategy_type", "?")
                        lines.append(f"  - Remove {remove} -> Add {repl}")
            probation = roster.get("probation", [])
            if probation:
                lines.append("Models you put on probation:")
                for p in probation:
                    if isinstance(p, dict):
                        lines.append(f"  - {p.get('model', '?')}: {p.get('reason', '')}")

        evolution = entry.get("strategy_evolution")
        if evolution and isinstance(evolution, dict):
            hypo = evolution.get("hypothesis", "")
            predicted = evolution.get("predicted_performance", {})
            if hypo:
                lines.append(f"Your hypothesis: {hypo}")
            if predicted:
                lines.append(f"Your predictions: {json.dumps(predicted)}")

        outcome = entry.get("next_day_outcome")
        if outcome:
            lines.append(
                f"**Next-day result ({outcome['date']}): {outcome['verdict']}** "
                f"(avg return {outcome['avg_return_pct']}% across {outcome['model_count']} models)"
            )
            # Show per-model outcomes for models the CFA specifically recommended changes for
            per_model = outcome.get("per_model", {})
            if per_model and param_summary:
                lines.append("How your recommended models performed next day:")
                for model_name in param_summary:
                    if model_name in per_model:
                        mr = per_model[model_name]
                        lines.append(
                            f"  - {model_name}: {mr['return_pct']}% return, {mr['trades']} trades"
                        )
            # Show any replaced models' successors
            if per_model and roster and isinstance(roster, dict):
                for r in roster.get("replace", []):
                    if isinstance(r, dict):
                        removed = r.get("remove", "")
                        # The successor model may have a different name
                        for mn, mr in per_model.items():
                            if mn not in param_summary:  # new model, not one we tuned
                                pass  # can't reliably match yet
        else:
            lines.append("Next-day result: not yet available (this was the most recent review)")
        lines.append("")

    return "\n".join(lines)


def _build_review_prompt(data: dict[str, Any], incident_notes: str = "") -> str:
    """Format gathered data into a structured LLM prompt."""
    # Exclude cfa_memory from the main data JSON (it's rendered separately above)
    data_for_json = {k: v for k, v in data.items() if k != "cfa_memory"}
    data_json = json.dumps(data_for_json, indent=2, default=str)

    # Only show changelog entries since last review
    last_review_date = None
    cfa_mem = data.get("cfa_memory", [])
    if cfa_mem:
        last_review_date = cfa_mem[0].get("date")

    return f"""\
You are the **Chief Strategy Architect** for an evolutionary trading arena. You own the
strategic direction of 13 AI strategy models competing on live Alpaca paper trading data.
Each model has $2,000 starting capital. Your goal: make this portfolio real-money ready
within 3 weeks.

You are NOT a passive reviewer — your recommendations WILL be applied. Every parameter
change, roster decision, and strategy evolution you output gets executed. Own the outcomes.
Diagnose causally, prescribe knowing changes take effect, iterate on what you've learned
from prior reviews, and architect the roster for consistent profitability.
## System Architecture

- **Models**: 13 active — 12 base models across 8 strategy types (ma_crossover, rsi_reversion,
  momentum, bollinger_bands, macd, vwap_reversion, breakout, mean_reversion) plus your
  cfa_generated model.
- **Capital**: $2,000 per model per day ($26,000 total portfolio). Capital resets daily.
- **Fractional shares**: Enabled (min 0.01 shares or $1 notional). Market orders only for fractional.
- **Risk defaults**: stop_loss_pct=2.0%, take_profit_pct=3.0% (evolvable via self-improvement).
- **Exit mechanisms**: Hard stop-loss, take-profit, ratcheting trailing stops (`trailing_stop_tiers`),
  patience stops (`patience_stop_tiers`). All CFA-tunable via parameter_recommendations.
  - `trailing_stop_tiers`: list of `[gain_threshold_pct, trail_distance_pct]` pairs. Trail tightens as gain grows.
  - `patience_stop_tiers`: list of `[bars_underwater, exit_pct]` pairs. Exits slow-bleed losers.
- **Position manager**: 15% max per symbol, 80% max total, correlation guards, 3%/5% drawdown circuit breaker.
- **Screener**: Top 20 most-active symbols ($5 floor), re-runs every 15 min. Leveraged ETF blocklist.
- **Wind-down**: NO_BUY_WINDOW=10 bars, LIQUIDATION_WINDOW=3 bars.
- **Self-improvement**: Between sessions — losers 10% mutation, mediocre 5%, winners 2%.
- **EOD liquidation**: All positions force-closed at session end.

NOTE: `realized_pnl` is the authoritative return figure (from filled sell orders).
`alpaca_pnl` is ground-truth P&L when available. `unattributed_pnl` = gap between the two.

{_build_cfa_memory_section(data)}

{_build_changelog_section(last_review_date)}

## Roster Architecture

You control 13 strategy slots. 12 base + your cfa_generated. Architect the roster for:
- **Diversity**: Uncorrelated strategy types covering different regimes (trend, mean-reversion, momentum, volatility).
- **Accountability**: If a strategy type lost money 2+ consecutive days, recommend replacement with
  a specific alternative.
- **Slot efficiency**: 0-trade models waste a slot. Replace with something that participates.
- Be direct: "replace X with Y because Z" not "consider removing X."

CRITICAL — SPECIFICITY REQUIREMENT: Every recommendation you make MUST include exact, implementable
details. You are the expert — your output gets executed directly. No vague suggestions.
- **Roster replacements**: Must include the full parameter set (every param with exact numeric values),
  entry/exit logic with specific indicator thresholds, adapt() tuning logic with bounds, and
  predicted performance. We will build the strategy from your spec — if you leave anything vague,
  we can't implement it.
- **Parameter recommendations**: Must include exact target values, not directions ("lower X" is not
  acceptable — "set X to 0.05" is).
- **Action items**: Must be specific enough that an engineer can execute without asking follow-up
  questions. Include the file, function, or parameter to change.
- You have full visibility into every model's parameters (`all_parameters` in the data). Use them.

## Cross-Strategy Synthesis

Analyze the top 3 performers' winning patterns: what indicators, parameters, entry/exit logic made
them profitable? Identify losing patterns across bottom performers. Your `cross_strategy_insights`
output should synthesize what works and what doesn't, and your generated strategy should incorporate
the winning elements.

Review the following data for {data['session_date']} and produce a structured JSON analysis.

## Today's Data

The data includes: market_data (daily_stats + intraday_30min with microstructure),
news_sentiment (per-symbol + market-wide + divergence flags), sector_analysis (returns +
correlations), options_data (IV, put/call ratio, delta skew), upcoming_events (dividends,
splits, mergers in next 5 days).

```json
{data_json}
```

## Your Analysis

Produce ONLY a JSON object (no markdown, no explanation outside JSON) with this schema:

{{
  "date": "{data['session_date']}",
  "executive_summary": "2-3 sentences summarizing the day",
  "portfolio_grade": "A/B/C/D/F",
  "portfolio_grade_justification": "1-2 sentences",
  "plain_english_verdict": {{
    "whats_working": "Plain English paragraph (no jargon) explaining what strategies, patterns, and behaviors are making money and why. Write this so a non-technical person can understand it.",
    "whats_not_working": "Plain English paragraph explaining what is losing money, why it's failing, and what the core problems are. Be blunt and specific — name the models and what they're doing wrong.",
    "bottom_line": "1-2 sentence gut-check verdict: is this portfolio heading in the right direction or not?"
  }},
  "sections": {{
    "performance_analysis": {{
      "headline": "one-line summary",
      "detail": "2-3 sentences",
      "concerns": ["list of concerns"],
      "positives": ["list of positives"]
    }},
    "best_performers": {{
      "models": [{{"name": "...", "why": "..."}}],
      "pattern": "what the best performers have in common"
    }},
    "worst_performers": {{
      "models": [{{"name": "...", "why": "..."}}],
      "pattern": "what the worst performers have in common"
    }},
    "symbol_analysis": {{
      "headline": "one-line summary",
      "concentration_risk": "assessment",
      "opportunities_missed": "what tickers could have been traded better"
    }},
    "trade_quality": {{
      "headline": "one-line summary",
      "expectancy_assessment": "is avg win/loss ratio healthy?"
    }},
    "risk_management": {{
      "headline": "one-line summary",
      "circuit_breaker_assessment": "were risk limits triggered appropriately?",
      "recommendations": ["actionable recommendations"]
    }},
    "strategy_rotation": {{
      "headline": "one-line summary",
      "trending_up": ["strategy types working well"],
      "trending_down": ["strategy types struggling"],
      "inactive_strategies": "assessment of models that didn't trade"
    }},
    "parameter_drift": {{
      "headline": "one-line summary",
      "convergence_warning": "are models converging to same params?",
      "recommendations": ["actionable recommendations"]
    }},
    "multi_day_trends": {{
      "headline": "one-line summary",
      "trajectory": "improving/stable/degrading",
      "detail": "2-3 sentences"
    }},
    "screener_review": {{
      "headline": "one-line summary",
      "dynamic_vs_core": "which performed better?",
      "recommendations": ["actionable recommendations"]
    }},
    "execution_quality": {{
      "headline": "one-line summary",
      "rejected_order_patterns": "analysis of rejections"
    }}
  }},
  "self_accountability": {{
    "prior_recs_assessed": [
      {{"recommendation": "what you recommended", "outcome": "helped|hurt|neutral|too_early", "evidence": "data points"}}
    ],
    "prediction_accuracy": "how accurate were your prior performance predictions",
    "lessons_learned": "what you'll do differently based on outcomes"
  }},
  "cross_strategy_insights": {{
    "winning_patterns": ["what top performers share"],
    "losing_patterns": ["what bottom performers share"],
    "synthesis_notes": "how you're incorporating these into your strategy"
  }},
  "strategy_evolution": {{
    "hypothesis": "what you're testing with this iteration",
    "changes_from_previous": ["max 3 specific logic changes"],
    "predicted_performance": {{
      "expected_win_rate_pct": 0.0,
      "expected_avg_pnl_per_trade": 0.0,
      "confidence": "high|medium|low"
    }}
  }},
  "roster_changes": {{
    "keep": ["model names performing well"],
    "probation": [{{"model": "name", "reason": "why", "conditions": "what would trigger replacement"}}],
    "replace": [{{
      "remove": "exact model name to remove",
      "replacement_strategy_type": "snake_case registry key for new strategy",
      "replacement_name": "human readable name for new model",
      "rationale": "why this replacement is better, citing specific data",
      "parameters": {{
        "param_name": "exact numeric value — ALL parameters the strategy needs",
        "allocation_pct": 0.0,
        "max_positions": 0,
        "stop_loss_pct": 0.0,
        "take_profit_pct": 0.0,
        "trailing_stop_tiers": [[0.0, 0.0]],
        "patience_stop_tiers": [[0, 0.0]]
      }},
      "entry_logic": "exact buy conditions with specific thresholds/indicators",
      "exit_logic": "exact sell conditions with specific thresholds",
      "adapt_logic": "what adapt() should tune, direction, and bounds",
      "predicted_win_rate": 0.0,
      "predicted_avg_pnl": 0.0
    }}]
  }},
  "action_items": [
    {{"priority": "high/medium/low", "action": "what to do", "rationale": "why"}}
  ],
  "red_flags": ["critical issues that need immediate attention"],
  "next_day_recommendations": "paragraph of recommendations for tomorrow",
  "research_notes": [
    {{"topic": "short name", "finding": "your analysis", "action_needed": "what to build/change/investigate"}}
  ],
  "parameter_recommendations": [
    {{
      "model_name": "<exact model name from the models list>",
      "recommendations": {{
        "<param_name>": {{
          "value": "<number or list — the target value>",
          "confidence": "high|medium|low",
          "rationale": "evidence-based reason for this change"
        }}
      }}
    }}
  ],
  "changelog_assessment": [
    {{
      "change": "<short description of the system change>",
      "assessment": "positive|negative|neutral|too_early",
      "evidence": "specific data points supporting your assessment"
    }}
  ],
  "infrastructure_recommendations": [
    {{
      "category": "order_execution|risk_management|data_pipeline|position_management|market_microstructure|other",
      "title": "short descriptive title",
      "description": "what the improvement is and how it works",
      "expected_impact": "how this would improve P&L or reduce losses",
      "priority": "high|medium|low",
      "complexity": "simple|moderate|complex"
    }}
  ]
}}

Be specific, data-driven, and actionable. Reference actual model names, symbols, and numbers.

CRITICAL: Base your analysis ONLY on the data provided. Never fabricate dollar amounts not in the data.

PARAMETER RECOMMENDATIONS — Each model's `all_parameters` dict shows its current tunable params.
- **Confidence levels**: `high` = override directly, `medium` = blend 70% toward target, `low` = bias mutation.
- Focus on underperformers. Leave top performers alone unless you have strong evidence.
- You can tune `stop_loss_pct`, `take_profit_pct`, `trailing_stop_tiers`, `patience_stop_tiers` on ANY model.
- Flag models with null stop_loss_pct/take_profit_pct as risk management concerns.
- Check for stop-loss re-entry loops (buy->stop->buy->stop on same symbol) and recommend cooldowns.

INFRASTRUCTURE RECOMMENDATIONS — Think beyond individual model parameters. Consider system-level
design patterns and execution strategies that haven't been implemented yet. Examples to consider:
- **Limit orders vs market orders**: Currently all orders are market orders. Would limit orders at
  favorable prices (e.g., VWAP deviation targets) reduce slippage and improve fill quality?
- **Time-of-day awareness**: Are certain hours consistently better/worse? Should models scale
  position sizes or pause trading during historically bad windows (e.g., first 15 min, lunch lull)?
- **Correlated position limits**: Models may independently accumulate positions in correlated assets
  (e.g., NVDA + SOXL, multiple tech stocks). Should there be cross-model correlation guards?
- **Screener quality**: Are screener-added symbols consistently worse than core symbols? Should
  the screener filter more aggressively or weight toward momentum quality over raw volume?
- **Order staging**: Would batching orders or using TWAP/VWAP execution reduce market impact?
- **Partial exits**: Instead of all-or-nothing sells, could scaling out (sell 50% at +1%, rest at trail)
  improve risk-adjusted returns?
- **Entry timing**: Could watching Level 2 / order flow data improve entry prices?
- **Capital rebalancing**: Should winning models get more capital between sessions?
- Anything else you observe in the data that suggests a structural improvement.

## Strategy Evolution Protocol

Generate a complete `CfaGeneratedStrategy` Python class. **Key constraint**: make at most
3 logic changes from the previous version. No wholesale rewrites unless this is the first
generation. State your hypothesis and predict expected performance (you'll be held accountable).

The `strategy_evolution` section MUST include specific parameter values and indicator thresholds,
not vague descriptions. Example: "buy when close < VWAP * 0.999 and volume > 1.2x avg_20" is
good. "buy on VWAP dips with volume confirmation" is NOT acceptable.

### Base Class API

```python
from src.core.strategy import BarData, Strategy, TradeSignal
# BarData: symbol, timestamp, open, high, low, close, volume, minutes_remaining,
#   daily_context: DailyContext (daily_open, running_high, running_low, daily_vwap, prev_close,
#     change_from_open_pct, change_from_prev_close_pct, daily_range_position) — None if unavailable
#   spy_daily_context: DailyContext — SPY's daily context as market trend proxy on every bar
# TradeSignal(symbol, side="buy"|"sell", quantity)
class Strategy(ABC):
    current_capital: float; _positions: dict[str, float]; _entry_prices: dict[str, float]
    _bar_history: dict[str, list[BarData]]
    def record_bar(self, bar) -> None: ...
    def check_liquidation(self, bar) -> TradeSignal | None: ...
    def get_close_series(self, symbol, lookback=0) -> pd.Series: ...
    def compute_quantity(self, price, allocation_pct=0.25) -> float: ...
    def get_history(self, symbol, lookback=0) -> list[BarData]: ...
```

### Requirements
1. Class: `CfaGeneratedStrategy`, `strategy_type = "cfa_generated"`, subclass `Strategy`
2. Must implement `on_bar`, `get_params`, `set_params`. Call `record_bar` + `check_liquidation` first.
3. Only import: numpy, pandas, math, logging, src.core.strategy
4. Include `adapt()` for intra-session tuning
5. `"generated_strategy"` key = complete Python source as string (\\n for newlines)

{_build_cfa_model_section(data)}
{f"""
## Incident Notes

{incident_notes}
""" if incident_notes else ""}"""


def _render_markdown(review: dict[str, Any]) -> str:
    """Convert review JSON into a readable markdown report."""
    lines = []
    lines.append(f"# CFA Review — {review.get('date', 'Unknown')}")
    lines.append("")
    lines.append(f"**Grade: {review.get('portfolio_grade', 'N/A')}** — "
                 f"{review.get('portfolio_grade_justification', '')}")
    lines.append("")
    lines.append(f"## Executive Summary")
    lines.append(review.get("executive_summary", ""))
    lines.append("")

    # Plain English verdict
    verdict = review.get("plain_english_verdict", {})
    if verdict:
        lines.append("## The Verdict (Plain English)")
        lines.append("")
        working = verdict.get("whats_working", "")
        if working:
            lines.append(f"**What's Working:** {working}")
            lines.append("")
        not_working = verdict.get("whats_not_working", "")
        if not_working:
            lines.append(f"**What's Not Working:** {not_working}")
            lines.append("")
        bottom_line = verdict.get("bottom_line", "")
        if bottom_line:
            lines.append(f"**Bottom Line:** {bottom_line}")
            lines.append("")

    # Self-accountability
    accountability = review.get("self_accountability", {})
    if accountability:
        lines.append("## Self-Accountability")
        prior_recs = accountability.get("prior_recs_assessed", [])
        if prior_recs:
            lines.append("")
            lines.append("| Recommendation | Outcome | Evidence |")
            lines.append("|---|---|---|")
            for rec in prior_recs:
                if isinstance(rec, dict):
                    lines.append(
                        f"| {rec.get('recommendation', '?')} "
                        f"| {rec.get('outcome', '?')} "
                        f"| {rec.get('evidence', '')} |"
                    )
            lines.append("")
        accuracy = accountability.get("prediction_accuracy", "")
        if accuracy:
            lines.append(f"*Prediction accuracy*: {accuracy}")
            lines.append("")
        lessons = accountability.get("lessons_learned", "")
        if lessons:
            lines.append(f"*Lessons learned*: {lessons}")
            lines.append("")

    # Cross-strategy insights
    insights = review.get("cross_strategy_insights", {})
    if insights:
        lines.append("## Cross-Strategy Insights")
        for field in ("winning_patterns", "losing_patterns"):
            items = insights.get(field, [])
            if items:
                lines.append(f"**{field.replace('_', ' ').title()}**:")
                for item in items:
                    lines.append(f"- {item}")
                lines.append("")
        synthesis = insights.get("synthesis_notes", "")
        if synthesis:
            lines.append(f"*Synthesis*: {synthesis}")
            lines.append("")

    # Standard sections
    sections = review.get("sections", {})
    section_titles = {
        "performance_analysis": "Performance Analysis",
        "best_performers": "Best Performers",
        "worst_performers": "Worst Performers",
        "symbol_analysis": "Symbol Analysis",
        "trade_quality": "Trade Quality",
        "risk_management": "Risk Management",
        "strategy_rotation": "Strategy Rotation",
        "parameter_drift": "Parameter Drift",
        "multi_day_trends": "Multi-Day Trends",
        "screener_review": "Screener Review",
        "execution_quality": "Execution Quality",
    }

    for key, title in section_titles.items():
        section = sections.get(key, {})
        if not section:
            continue

        lines.append(f"## {title}")
        headline = section.get("headline", "")
        if headline:
            lines.append(f"**{headline}**")
            lines.append("")

        for field in ("detail", "concentration_risk", "opportunities_missed",
                      "expectancy_assessment", "circuit_breaker_assessment",
                      "convergence_warning", "inactive_strategies",
                      "trajectory", "dynamic_vs_core", "rejected_order_patterns",
                      "pattern"):
            val = section.get(field)
            if val:
                lines.append(f"*{field.replace('_', ' ').title()}*: {val}")
                lines.append("")

        for field in ("concerns", "positives", "trending_up", "trending_down",
                      "recommendations"):
            items = section.get(field, [])
            if items:
                lines.append(f"**{field.replace('_', ' ').title()}**:")
                for item in items:
                    lines.append(f"- {item}")
                lines.append("")

        model_list = section.get("models", [])
        if model_list:
            for entry in model_list:
                if isinstance(entry, dict):
                    lines.append(f"- **{entry.get('name', '?')}**: {entry.get('why', '')}")
                else:
                    lines.append(f"- {entry}")
            lines.append("")

    # Strategy evolution
    evolution = review.get("strategy_evolution", {})
    if evolution:
        lines.append("## Strategy Evolution")
        hypo = evolution.get("hypothesis", "")
        if hypo:
            lines.append(f"**Hypothesis**: {hypo}")
            lines.append("")
        changes = evolution.get("changes_from_previous", [])
        if changes:
            lines.append("**Changes**:")
            for c in changes:
                lines.append(f"- {c}")
            lines.append("")
        predicted = evolution.get("predicted_performance", {})
        if predicted:
            lines.append(
                f"**Predicted**: win_rate={predicted.get('expected_win_rate_pct', '?')}%, "
                f"avg_pnl=${predicted.get('expected_avg_pnl_per_trade', '?')} "
                f"[{predicted.get('confidence', '?')}]"
            )
            lines.append("")

    # Roster changes
    roster = review.get("roster_changes", {})
    if roster:
        lines.append("## Roster Changes")
        keep = roster.get("keep", [])
        if keep:
            lines.append(f"**Keep**: {', '.join(keep)}")
            lines.append("")
        probation = roster.get("probation", [])
        if probation:
            lines.append("**Probation**:")
            for p in probation:
                if isinstance(p, dict):
                    lines.append(f"- **{p.get('model', '?')}**: {p.get('reason', '')} "
                                 f"(conditions: {p.get('conditions', '')})")
            lines.append("")
        replace = roster.get("replace", [])
        if replace:
            lines.append("**Replace**:")
            for r in replace:
                if isinstance(r, dict):
                    lines.append(f"- Remove **{r.get('remove', '?')}** -> "
                                 f"Add **{r.get('replacement_type', '?')}**: "
                                 f"{r.get('rationale', '')}")
                    logic = r.get("replacement_logic", "")
                    if logic:
                        lines.append(f"  Logic: {logic}")
            lines.append("")

    # Action items
    action_items = review.get("action_items", [])
    if action_items:
        lines.append("## Action Items")
        for item in action_items:
            if isinstance(item, dict):
                prio = item.get("priority", "medium").upper()
                lines.append(f"- [{prio}] **{item.get('action', '')}** — {item.get('rationale', '')}")
            else:
                lines.append(f"- {item}")
        lines.append("")

    # Red flags
    red_flags = review.get("red_flags", [])
    if red_flags:
        lines.append("## Red Flags")
        for flag in red_flags:
            lines.append(f"- {flag}")
        lines.append("")

    # Next day recommendations
    next_day = review.get("next_day_recommendations", "")
    if next_day:
        lines.append("## Next Day Recommendations")
        lines.append(next_day)
        lines.append("")

    # Research notes (replaces open_questions + data_requests)
    research = review.get("research_notes", [])
    if research:
        lines.append("## Research Notes")
        for note in research:
            if isinstance(note, dict):
                lines.append(f"### {note.get('topic', '?')}")
                lines.append(note.get("finding", ""))
                action = note.get("action_needed", "")
                if action:
                    lines.append(f"\n> **Action needed**: {action}")
                lines.append("")
            else:
                lines.append(f"- {note}")
        lines.append("")

    # Legacy: open_questions (backward compat with old reviews)
    open_qs = review.get("open_questions", [])
    if open_qs:
        lines.append("## Open Questions")
        for q in open_qs:
            if isinstance(q, dict):
                lines.append(f"### {q.get('topic', '?')}")
                lines.append(q.get("assessment", ""))
                lines.append("")
            else:
                lines.append(f"- {q}")
        lines.append("")

    # Legacy: data_requests (backward compat with old reviews)
    data_reqs = review.get("data_requests", [])
    if data_reqs:
        lines.append("## Data Requests")
        for req in data_reqs:
            if isinstance(req, dict):
                prio = req.get("priority", "medium").upper()
                lines.append(f"- [{prio}] **{req.get('data_type', '?')}**: {req.get('description', '')} — {req.get('rationale', '')}")
            else:
                lines.append(f"- {req}")
        lines.append("")

    # Parameter recommendations
    param_recs = review.get("parameter_recommendations", [])
    if param_recs:
        lines.append("## Parameter Recommendations")
        lines.append("")
        for entry in param_recs:
            if not isinstance(entry, dict):
                continue
            model_name = entry.get("model_name", "?")
            lines.append(f"### {model_name}")
            recs = entry.get("recommendations", {})
            if isinstance(recs, dict):
                for param, spec in recs.items():
                    if isinstance(spec, dict):
                        conf = spec.get("confidence", "?")
                        val = spec.get("value", "?")
                        rationale = spec.get("rationale", "")
                        lines.append(f"- **{param}** = {val} [{conf}] — {rationale}")
                    else:
                        lines.append(f"- **{param}**: {spec}")
            lines.append("")

    # Infrastructure recommendations
    infra_recs = review.get("infrastructure_recommendations", [])
    if infra_recs:
        lines.append("## Infrastructure Recommendations")
        lines.append("")
        for entry in infra_recs:
            if isinstance(entry, dict):
                title = entry.get("title", "?")
                cat = entry.get("category", "other")
                prio = entry.get("priority", "medium").upper()
                desc = entry.get("description", "")
                impact = entry.get("expected_impact", "")
                complexity = entry.get("complexity", "?")
                lines.append(f"### [{prio}] {title} ({cat})")
                lines.append(f"- **Description**: {desc}")
                lines.append(f"- **Expected impact**: {impact}")
                lines.append(f"- **Complexity**: {complexity}")
                lines.append("")
            else:
                lines.append(f"- {entry}")
        lines.append("")

    # Changelog assessment
    changelog = review.get("changelog_assessment", [])
    if changelog:
        lines.append("## Changelog Assessment")
        lines.append("")
        for entry in changelog:
            if isinstance(entry, dict):
                change = entry.get("change", "?")
                assessment = entry.get("assessment", "?").upper()
                evidence = entry.get("evidence", "")
                lines.append(f"- **{change}** [{assessment}] — {evidence}")
            else:
                lines.append(f"- {entry}")
        lines.append("")

    return "\n".join(lines)


def run_cfa_review(
    db_path: str,
    session_date: str,
    model_id: str = "claude-opus-4-20250514",
    timeout_sec: int = 120,
    lookback_days: int = 10,
    incident_notes: str = "",
) -> dict[str, Any] | None:
    """Run the full CFA review pipeline. Returns parsed review dict or None on failure."""
    try:
        import anthropic
    except ImportError:
        logger.error("anthropic package not installed, skipping CFA review")
        return None

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set, skipping CFA review")
        return None

    # 1. Gather data
    logger.info(f"CFA Review: gathering data for {session_date}...")
    data = _gather_review_data(db_path, session_date, lookback_days)

    # Quick sanity check
    if not data["models"]:
        logger.warning("CFA Review: no models found, skipping")
        return None

    # 2. Build prompt
    prompt = _build_review_prompt(data, incident_notes=incident_notes)

    # 3. Call LLM
    logger.info(f"CFA Review: calling {model_id}...")
    client = anthropic.Anthropic(api_key=api_key, timeout=timeout_sec)
    response = client.messages.create(
        model=model_id,
        max_tokens=16384,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_text = response.content[0].text.strip()

    # 4. Parse JSON
    review = _parse_review_json(raw_text)
    if review is None:
        logger.error("CFA Review: failed to parse response JSON")
        # Still save raw response
        _save_to_db(db_path, session_date, None, raw_text, model_id)
        return None

    # 5. Save to DB
    _save_to_db(db_path, session_date, review, raw_text, model_id)

    # 6. Write markdown file
    md = _render_markdown(review)
    os.makedirs("logs", exist_ok=True)
    md_path = f"logs/cfa_review_{session_date}.md"
    with open(md_path, "w") as f:
        f.write(md)
    logger.info(f"CFA Review saved to {md_path}")

    # 7. Extract generated strategy code (if present)
    generated_strategy = review.get("generated_strategy")
    if generated_strategy:
        logger.info("CFA Review: generated strategy code received")
    else:
        logger.warning("CFA Review: no generated_strategy in response")

    return review


def _parse_review_json(text: str) -> dict | None:
    """Extract and parse JSON from LLM response text."""
    # Try direct parse first (strict=False to allow control chars in strings)
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass

    # Try extracting from code block
    for marker in ("```json", "```"):
        if marker in text:
            start = text.index(marker) + len(marker)
            end = text.index("```", start)
            try:
                return json.loads(text[start:end].strip(), strict=False)
            except (json.JSONDecodeError, ValueError):
                pass

    # Last resort: find first { to last }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace:last_brace + 1], strict=False)
        except json.JSONDecodeError:
            pass

    return None


VALID_CONFIDENCE_LEVELS = {"high", "medium", "low"}


def extract_parameter_recommendations(
    review: dict[str, Any],
    valid_model_names: set[str] | None = None,
) -> dict[str, dict]:
    """Extract and validate parameter recommendations from a CFA review.

    Returns:
        Dict mapping model_name -> {param_name: {"value": num, "confidence": str, "rationale": str}}
        Invalid entries are silently dropped.
    """
    recs = review.get("parameter_recommendations", [])
    if not isinstance(recs, list):
        return {}

    result: dict[str, dict] = {}
    for entry in recs:
        if not isinstance(entry, dict):
            continue
        model_name = entry.get("model_name")
        if not isinstance(model_name, str) or not model_name:
            continue
        if valid_model_names and model_name not in valid_model_names:
            logger.debug(f"CFA param rec: unknown model '{model_name}', skipping")
            continue

        recommendations = entry.get("recommendations", {})
        if not isinstance(recommendations, dict):
            continue

        valid_recs = {}
        for param, spec in recommendations.items():
            if not isinstance(param, str) or not isinstance(spec, dict):
                continue
            value = spec.get("value")
            confidence = spec.get("confidence", "").lower()
            if confidence not in VALID_CONFIDENCE_LEVELS:
                continue

            # Special case: trailing_stop_tiers is a list of [gain, trail] pairs
            if param == "trailing_stop_tiers":
                if isinstance(value, list) and all(
                    isinstance(p, (list, tuple)) and len(p) == 2
                    for p in value
                ):
                    try:
                        validated = [[float(p[0]), float(p[1])] for p in value]
                        valid_recs[param] = {
                            "value": validated,
                            "confidence": confidence,
                            "rationale": str(spec.get("rationale", "")),
                        }
                    except (TypeError, ValueError):
                        pass
                continue

            # Special case: patience_stop_tiers is a list of [bars, exit_pct] pairs
            if param == "patience_stop_tiers":
                if isinstance(value, list) and all(
                    isinstance(p, (list, tuple)) and len(p) == 2
                    for p in value
                ):
                    try:
                        validated = [[int(p[0]), float(p[1])] for p in value]
                        valid_recs[param] = {
                            "value": validated,
                            "confidence": confidence,
                            "rationale": str(spec.get("rationale", "")),
                        }
                    except (TypeError, ValueError):
                        pass
                continue

            if not isinstance(value, (int, float)):
                # Try to parse string numbers
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
            valid_recs[param] = {
                "value": value,
                "confidence": confidence,
                "rationale": str(spec.get("rationale", "")),
            }

        if valid_recs:
            result[model_name] = valid_recs

    if result:
        logger.info(
            f"CFA parameter recommendations extracted for {len(result)} models: "
            + ", ".join(result.keys())
        )
    return result


def _save_to_db(
    db_path: str,
    session_date: str,
    review_json: dict | None,
    raw_response: str,
    model_used: str,
) -> None:
    """Save or upsert the CFA review into the database."""
    db = get_session(db_path)
    try:
        existing = (
            db.query(CfaReview)
            .filter(CfaReview.session_date == session_date)
            .first()
        )
        if existing:
            existing.review_json = review_json
            existing.raw_response = raw_response
            existing.model_used = model_used
            existing.created_at = datetime.utcnow()
        else:
            db.add(CfaReview(
                session_date=session_date,
                review_json=review_json,
                raw_response=raw_response,
                model_used=model_used,
            ))
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Failed to save CFA review to DB")
    finally:
        db.close()


def apply_cfa_strategy(
    code: str,
    db_path: str,
    initial_capital: float = 2_000.0,
) -> bool:
    """Validate and persist CFA-generated strategy code.

    1. Write code to src/strategies/cfa_generated.py
    2. Try importing it to validate syntax and structure
    3. Verify it subclasses Strategy and has required methods
    4. Register in STRATEGY_REGISTRY
    5. Create or update cfa_analyst model in DB

    Returns True on success, False on failure.
    """
    from src.core.strategy import Strategy as StrategyBase

    strategies_dir = pathlib.Path(__file__).resolve().parent.parent / "strategies"
    target_path = strategies_dir / "cfa_generated.py"

    # 1. Write code to disk
    try:
        target_path.write_text(code)
        logger.info(f"CFA strategy code written to {target_path}")
    except Exception:
        logger.exception("Failed to write CFA strategy code to disk")
        return False

    # 2. Try importing to validate
    module_name = "src.strategies.cfa_generated"
    try:
        # Remove stale cached module if any
        import sys
        if module_name in sys.modules:
            del sys.modules[module_name]
        mod = importlib.import_module(module_name)
    except (SyntaxError, ImportError, Exception) as exc:
        logger.error(f"CFA strategy code failed to import: {exc}")
        # Remove the broken file so it doesn't crash on next startup
        target_path.unlink(missing_ok=True)
        return False

    # 3. Verify class structure
    cls = getattr(mod, "CfaGeneratedStrategy", None)
    if cls is None:
        logger.error("CFA strategy module missing CfaGeneratedStrategy class")
        target_path.unlink(missing_ok=True)
        return False

    if not (inspect.isclass(cls) and issubclass(cls, StrategyBase)):
        logger.error("CfaGeneratedStrategy does not subclass Strategy")
        target_path.unlink(missing_ok=True)
        return False

    for method in ("on_bar", "get_params", "set_params"):
        if not hasattr(cls, method):
            logger.error(f"CfaGeneratedStrategy missing required method: {method}")
            target_path.unlink(missing_ok=True)
            return False

    # Quick instantiation test
    try:
        instance = cls(name="__validation_test__")
        _ = instance.get_params()
    except Exception as exc:
        logger.error(f"CfaGeneratedStrategy instantiation failed: {exc}")
        target_path.unlink(missing_ok=True)
        return False

    # 4. Register in STRATEGY_REGISTRY
    from src.strategies.registry import STRATEGY_REGISTRY
    STRATEGY_REGISTRY["cfa_generated"] = cls
    logger.info("Registered 'cfa_generated' in STRATEGY_REGISTRY")

    # 5. Create or update cfa_analyst model in DB
    db = get_session(db_path)
    try:
        existing = (
            db.query(TradingModel)
            .filter(TradingModel.name == "cfa_analyst")
            .first()
        )
        if existing:
            existing.strategy_type = "cfa_generated"
            existing.status = ModelStatus.ACTIVE
            existing.parameters = instance.get_params()
            logger.info(f"Updated existing cfa_analyst model (id={existing.id})")
        else:
            model = TradingModel(
                name="cfa_analyst",
                strategy_type="cfa_generated",
                parameters=instance.get_params(),
                generation=1,
                initial_capital=initial_capital,
                current_capital=initial_capital,
            )
            db.add(model)
            logger.info("Created new cfa_analyst model in DB")
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Failed to create/update cfa_analyst model in DB")
        return False
    finally:
        db.close()

    logger.info("CFA strategy applied successfully")
    return True
