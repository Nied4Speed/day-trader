"""Post-day CFA-style review powered by Claude.

Gathers the day's trading data, builds a structured prompt, calls Claude Opus
for analysis, and saves the review to DB + markdown file.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any

from sqlalchemy import func

from src.core.database import (
    CfaReview,
    DailyLedger,
    ModelSummary,
    Order,
    OrderSide,
    OrderStatus,
    PerformanceSnapshot,
    SessionRecord,
    TradingModel,
    get_session,
)

logger = logging.getLogger(__name__)


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
        }
    finally:
        db.close()


def _build_review_prompt(data: dict[str, Any]) -> str:
    """Format gathered data into a structured LLM prompt."""
    data_json = json.dumps(data, indent=2, default=str)

    return f"""\
You are a CFA charterholder and portfolio risk analyst reviewing the daily results
of an evolutionary trading arena. The arena runs 12 competitive AI strategy models
on live Alpaca paper trading data. Each model has $1,000 starting capital.

Models self-improve between sessions by mutating their own parameters. Weekly evolution
culls the bottom performers and breeds new ones from top performers.

NOTE: The "realized_pnl" field on each model is the authoritative return figure,
computed from filled sell orders. Use it over "return_pct" or capital-based calculations
if they disagree.

Review the following data for {data['session_date']} and produce a structured JSON analysis.

## Today's Data

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

OPEN QUESTIONS — include an "open_questions" array in your response with your assessment of each:
1. "COLLAB model" — We are considering adding a collaborative model that is synthesized from the
   top performers between sessions (not a real-time voting ensemble). It would inherit the best
   parameters from winning strategies. Is this a good idea? What are the risks (overfitting,
   reduced diversity, regime sensitivity)? How would you structure it?
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
        max_tokens=4096,
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
