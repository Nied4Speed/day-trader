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
            end_cap = ledger.end_capital if ledger else m.current_capital
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
                "reject_reasons": reject_reasons,
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
        }
    finally:
        db.close()


def _build_cfa_model_section(data: dict[str, Any]) -> str:
    """Build the CFA model context section for the prompt."""
    cfa = data.get("cfa_model")
    if not cfa:
        return "### Your previous strategy\nThis is the FIRST time you are generating a strategy. There is no previous code."

    lines = ["### Your previous strategy"]
    perf = cfa.get("performance")
    if perf:
        lines.append(f"Return: {perf.get('return_pct', 0)}%, Trades: {perf.get('trades', 0)}, "
                      f"Win rate: {perf.get('win_rate', 0)}%, Realized PnL: ${perf.get('realized_pnl', 0)}")
    source = cfa.get("current_source", "")
    if source:
        lines.append("\n```python")
        lines.append(source)
        lines.append("```")
        lines.append("\nAnalyze what worked and what didn't. You may refine or completely rewrite.")
    else:
        lines.append("Previous source code not found. Generate fresh.")

    return "\n".join(lines)


def _build_review_prompt(data: dict[str, Any]) -> str:
    """Format gathered data into a structured LLM prompt."""
    data_json = json.dumps(data, indent=2, default=str)

    return f"""\
You are a CFA charterholder and portfolio risk analyst reviewing the daily results
of an evolutionary trading arena. The arena runs 12 competitive AI strategy models
on live Alpaca paper trading data. Each model has $2,000 starting capital.

## System Architecture

- **Models**: 13 active — 12 base models across 8 strategy types (ma_crossover, rsi_reversion,
  momentum, bollinger_bands, macd, vwap_reversion, breakout, mean_reversion) plus your
  cfa_generated model.
- **Capital**: $2,000 per model per day ($24,000 total portfolio). Capital resets daily.
- **Fractional shares**: Enabled (min 0.01 shares or $1 notional). Market orders only for fractional.
- **Risk defaults**: stop_loss_pct=2.0%, take_profit_pct=3.0% (evolvable via self-improvement).
- **Position manager**: 15% max exposure per symbol, 80% max total exposure, correlation guards,
  3% drawdown halves position sizes, 5% drawdown blocks new buys.
- **Screener**: Discovers top 20 most-active symbols ($5 price floor), re-runs every 15 min.
  Leveraged ETF blocklist (39 tickers like TQQQ, SQQQ, SOXL, UVIX, etc.) filtered out.
- **Wind-down**: NO_BUY_WINDOW=10 bars before session end, LIQUIDATION_WINDOW=3 bars forces sells.
- **Self-improvement**: Between sessions — losers 10% mutation, mediocre 5%, winners 2%.
- **COLLAB model**: Disabled pending redesign as between-session synthesis from top performers.
- **EOD liquidation**: All positions force-closed at session end + Alpaca hard safety net.

NOTE: The "realized_pnl" field on each model is the authoritative return figure,
computed from filled sell orders. Use it over "return_pct" or capital-based calculations
if they disagree.

## Strategy Roster — You Have Full Control

The current 12 models use 8 common day-trading strategy types that were picked as a starting
point: ma_crossover, rsi_reversion, momentum, bollinger_bands, macd, vwap_reversion, breakout,
mean_reversion. These are NOT sacred — they were chosen as generic baselines to get the arena
running.

**You can and should recommend changes to the roster.** Specifically:
- **Eliminate underperformers**: If a strategy type is consistently losing, recommend we remove
  it and replace it with something better. Name the specific model(s) to cut.
- **Add new strategy types**: If you think a different approach would work better (pairs trading,
  statistical arbitrage, volume profile, order flow, gamma scalping, etc.), recommend it. We will
  implement it.
- **Rebalance the mix**: If one strategy type dominates, recommend adding more variants of it
  or reducing over-represented losers.
- **Your generated strategy counts as one of the 12 slots.** If you want to test a radically
  different approach, you can — just build it in your generated_strategy output.

The constraint: we want to keep exactly 13 models total (12 base + your cfa_generated model).
So any additions require removing an equal number. Think of it as portfolio construction — which
13 strategies give us the best risk-adjusted returns? You always keep your own slot, but you can
recommend replacing any of the other 12.

Include roster recommendations in your action_items if you think changes are warranted.

Review the following data for {data['session_date']} and produce a structured JSON analysis.

## Today's Data

The data includes several enriched sections:

### market_data
- **daily_stats**: Per-symbol daily OHLCV, % change, intraday range, PLUS microstructure stats:
  - `avg_bar_range_pct`: mean (high-low)/open per 1-min bar — intrabar volatility proxy
  - `peak_volume_window`: 30-min window with highest volume (e.g., "09:30" = open, "15:30" = close)
  - `trend_strength`: (close-open)/(high-low), ranges -1 to +1. +1=pure uptrend, -1=pure downtrend, ~0=choppy
- **intraday_30min**: 30-minute aggregated OHLCV bars for core symbols + top dynamic symbols

### news_sentiment
- **per_symbol**: avg sentiment score [-1,+1], article count, top headline for each mentioned symbol
  - Score interpretation: -1.0 = very bearish, 0 = neutral, +1.0 = very bullish
- **market_wide**: total articles, avg sentiment, bullish/bearish/neutral counts
- **sentiment_pnl_divergence**: symbols where sentiment disagreed with realized PnL (contrarian signals)

### sector_analysis
- **sector_returns**: per-sector avg return, volume, symbol count, best/worst performers
- **correlation_matrix**: cross-sector return correlations (from intraday bar data)
- **sector_concentration**: which sectors were most/least active

### options_data
- Per-symbol options chain summary (core + top traded symbols):
  - `implied_volatility_pct`: avg ATM call IV as percentage — market's expected annualized move
  - `put_call_ratio`: put volume / call volume. >1.0 = bearish sentiment, <0.7 = bullish
  - `delta_skew`: avg put delta - avg call delta. More negative = normal; less negative/positive = fear
  - `call_count`, `put_count`, `call_volume`, `put_volume` for flow analysis

### upcoming_events
- Corporate actions in next 5 trading days: dividends (ex-date, cash amount), splits (ratio), mergers, spinoffs
- Use to avoid entering positions before ex-dates or to anticipate volatility around events

### Data NOT available
- **Order book depth**: NOT available for equities via Alpaca (crypto only). Would need Polygon.io.
- **Options flow** (unusual activity detection): NOT available. We provide IV + Greeks instead.
- **Tick-level trade data**: Available in real-time (quote stream) but not persisted. Can add if needed.
- **Short interest / dark pool volume**: NOT available via Alpaca. Would need third-party data.

Use all this data when analyzing performance and designing your strategy.

```json
{data_json}
```

## Your Analysis

Produce ONLY a JSON object (no markdown, no explanation outside JSON) with this exact schema:

{{
  "date": "{data['session_date']}",
  "executive_summary": "2-3 sentences summarizing the day",
  "portfolio_grade": "A/B/C/D/F",
  "portfolio_grade_justification": "1-2 sentences",
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
  "action_items": [
    {{"priority": "high/medium/low", "action": "what to do", "rationale": "why"}}
  ],
  "red_flags": ["critical issues that need immediate attention"],
  "next_day_recommendations": "paragraph of recommendations for tomorrow",
  "open_questions": [
    {{"topic": "short name", "assessment": "your analysis/opinion on this topic"}}
  ],
  "data_requests": [
    {{
      "data_type": "short name (e.g. 'order_book_depth', 'tick_data', 'options_flow')",
      "description": "what this data is and how you'd use it",
      "priority": "critical/high/medium/low",
      "rationale": "why this would improve your strategy's returns or risk management"
    }}
  ]
}}

Be specific, data-driven, and actionable. Reference actual model names, symbols, and numbers.
If multi_day data is empty (first day), say so and focus on today only.
For inactive models, diagnose likely causes (short sessions, high thresholds, strategy type mismatch).

CRITICAL: Base your analysis ONLY on the data provided. If all models show 0% returns
with non-zero trade counts, report this as a data collection issue — do NOT invent or
extrapolate losses. Never fabricate specific dollar amounts not present in the data.

IMPORTANT: Each model has stop_loss_pct and take_profit_pct fields. If these are null/None, the model
has NO automatic exit discipline — it will hold positions until its strategy logic generates a sell
signal or the session ends. Flag any models with null stop_loss_pct or take_profit_pct as a risk
management concern and recommend they set values (e.g., 2% stop-loss, 3% take-profit) so that
self-improvement can evolve the thresholds over time.

DATA REQUESTS — Your goal is to maximize returns and minimize losses. Include a "data_requests"
array listing ANY additional data that would help you build a better strategy. Think about what
a top quantitative fund would want: order book depth, tick-level data, options flow/Greeks,
sector correlations, earnings calendars, economic indicators, short interest, dark pool volume,
news sentiment scores, pre-market/after-hours activity, etc. We can wire up new data sources.
Be specific about what you want, how you'd use it, and why it would improve performance.
Your strategy competes against 12 others — tell us what edge you need.

OPEN QUESTIONS — include an "open_questions" array in your response with your assessment of each:
1. "COLLAB model" — We are considering adding a collaborative model that is synthesized from the
   top performers between sessions (not a real-time voting ensemble). It would inherit the best
   parameters from winning strategies. Is this a good idea? What are the risks (overfitting,
   reduced diversity, regime sensitivity)? How would you structure it?

## Strategy Generation

In addition to your JSON analysis, you must ALSO generate a complete Python strategy class.
You have full creative freedom — use ANY combination of indicators and logic you want.
Base your design on what worked (and what didn't) in today's data.

You have access to the full market_data (daily stats + 30-min intraday bars) above.
Use it to understand what price action, volatility, and volume patterns looked like.
Design your strategy to exploit the patterns you observe. You are not limited to any
particular approach — combine multiple indicators, use regime detection, volume analysis,
multi-timeframe logic, whatever you think will work best.

### Base Class API

```python
from src.core.strategy import BarData, Strategy, TradeSignal

# BarData fields: symbol, timestamp, open, high, low, close, volume, minutes_remaining
# TradeSignal(symbol, side="buy"|"sell", quantity)

class Strategy(ABC):
    strategy_type: str = "base"
    current_capital: float          # available cash
    _positions: dict[str, float]    # symbol -> quantity held
    _entry_prices: dict[str, float] # symbol -> avg entry price
    _bar_history: dict[str, list[BarData]]

    def record_bar(self, bar: BarData) -> None: ...     # stores bar in history
    def check_liquidation(self, bar: BarData) -> TradeSignal | None: ...  # EOD wind-down
    def get_close_series(self, symbol: str, lookback: int = 0) -> pd.Series: ...  # close prices
    def compute_quantity(self, price: float, allocation_pct: float = 0.25) -> float: ...  # fractional shares
    def get_history(self, symbol: str, lookback: int = 0) -> list[BarData]: ...
```

### Available imports
numpy, pandas, math, logging (already in environment — do NOT import anything else)

### Example (RSI strategy for reference)

```python
class RSIReversionStrategy(Strategy):
    strategy_type = "rsi_reversion"

    def __init__(self, name, params=None):
        self.rsi_period = 14
        self.oversold = 30.0
        self.overbought = 70.0
        self.allocation_pct = 0.25
        super().__init__(name, params)

    def _compute_rsi(self, closes):
        if len(closes) < self.rsi_period + 1:
            return 50.0
        deltas = closes.diff().dropna()
        gains = deltas.where(deltas > 0, 0.0)
        losses = (-deltas.where(deltas < 0, 0.0))
        avg_gain = gains.rolling(self.rsi_period).mean().iloc[-1]
        avg_loss = losses.rolling(self.rsi_period).mean().iloc[-1]
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def on_bar(self, bar):
        self.record_bar(bar)
        liq = self.check_liquidation(bar)
        if liq:
            return liq if liq.quantity > 0 else None
        closes = self.get_close_series(bar.symbol)
        if len(closes) < self.rsi_period + 1:
            return None
        rsi = self._compute_rsi(closes)
        if rsi < self.oversold:
            qty = self.compute_quantity(bar.close, self.allocation_pct)
            if qty > 0:
                return TradeSignal(symbol=bar.symbol, side="buy", quantity=qty)
        if rsi > self.overbought:
            qty = self.compute_quantity(bar.close, self.allocation_pct)
            if qty > 0:
                return TradeSignal(symbol=bar.symbol, side="sell", quantity=qty)
        return None

    def get_params(self):
        return {{"rsi_period": self.rsi_period, "oversold": self.oversold,
                "overbought": self.overbought, "allocation_pct": self.allocation_pct}}

    def set_params(self, params):
        self.rsi_period = max(2, int(params.get("rsi_period", self.rsi_period)))
        self.oversold = max(5.0, min(45.0, float(params.get("oversold", self.oversold))))
        self.overbought = max(55.0, min(95.0, float(params.get("overbought", self.overbought))))
        self.allocation_pct = max(0.05, min(1.0, float(params.get("allocation_pct", self.allocation_pct))))

    def adapt(self, recent_signals, recent_fills, realized_pnl):
        pass  # optional
```

### Requirements for your generated class
1. Class name MUST be `CfaGeneratedStrategy`
2. `strategy_type = "cfa_generated"`
3. Must subclass `Strategy` and call `super().__init__(name, params)` in `__init__`
4. Must implement `on_bar(self, bar)` — call `self.record_bar(bar)` and `self.check_liquidation(bar)` first
5. Must implement `get_params(self)` and `set_params(self, params)`
6. Return `TradeSignal(symbol=bar.symbol, side="buy"|"sell", quantity=qty)` or `None`
7. Use `self.compute_quantity(price, allocation_pct)` for position sizing
8. Only import from: `numpy`, `pandas`, `math`, `logging`, and `src.core.strategy`
9. Include the full file with all imports at the top
10. You can use multiple indicators, regime detection, volume analysis — anything you want
11. The `adapt()` method is optional but recommended for intra-session tuning

{_build_cfa_model_section(data)}

### Output format
Include a `"generated_strategy"` key in your JSON response. Its value must be a STRING containing
the complete, valid Python source code for the file. Use \\n for newlines within the string.
The file should be self-contained and ready to write to `src/strategies/cfa_generated.py`.
"""


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

        # Render known sub-fields
        for field in ("detail", "concentration_risk", "opportunities_missed",
                      "expectancy_assessment", "circuit_breaker_assessment",
                      "convergence_warning", "inactive_strategies",
                      "trajectory", "dynamic_vs_core", "rejected_order_patterns",
                      "pattern"):
            val = section.get(field)
            if val:
                lines.append(f"*{field.replace('_', ' ').title()}*: {val}")
                lines.append("")

        # Render lists
        for field in ("concerns", "positives", "trending_up", "trending_down",
                      "recommendations"):
            items = section.get(field, [])
            if items:
                lines.append(f"**{field.replace('_', ' ').title()}**:")
                for item in items:
                    lines.append(f"- {item}")
                lines.append("")

        # Render model lists (best/worst performers)
        model_list = section.get("models", [])
        if model_list:
            for entry in model_list:
                if isinstance(entry, dict):
                    lines.append(f"- **{entry.get('name', '?')}**: {entry.get('why', '')}")
                else:
                    lines.append(f"- {entry}")
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

    # Open questions
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

    # Data requests
    data_reqs = review.get("data_requests", [])
    if data_reqs:
        lines.append("## Data Requests")
        lines.append("*What the CFA wants to build a better strategy:*")
        lines.append("")
        for req in data_reqs:
            if isinstance(req, dict):
                prio = req.get("priority", "medium").upper()
                lines.append(f"### [{prio}] {req.get('data_type', '?')}")
                lines.append(req.get("description", ""))
                lines.append(f"\n*Rationale*: {req.get('rationale', '')}")
                lines.append("")
            else:
                lines.append(f"- {req}")
        lines.append("")

    return "\n".join(lines)


def run_cfa_review(
    db_path: str,
    session_date: str,
    model_id: str = "claude-opus-4-20250514",
    timeout_sec: int = 120,
    lookback_days: int = 10,
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
    prompt = _build_review_prompt(data)

    # 3. Call LLM
    logger.info(f"CFA Review: calling {model_id}...")
    client = anthropic.Anthropic(api_key=api_key, timeout=timeout_sec)
    response = client.messages.create(
        model=model_id,
        max_tokens=8192,
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
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from code block
    for marker in ("```json", "```"):
        if marker in text:
            start = text.index(marker) + len(marker)
            end = text.index("```", start)
            try:
                return json.loads(text[start:end].strip())
            except (json.JSONDecodeError, ValueError):
                pass

    # Last resort: find first { to last }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass

    return None


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
