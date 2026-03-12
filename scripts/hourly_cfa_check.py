"""Hourly CFA check-in: gathers recent trading data and asks CFA for assessment.

Run in background: .venv/bin/python scripts/hourly_cfa_check.py &
Logs to logs/cfa_hourly_YYYYMMDD.log
"""

import json
import logging
import os
import time
from datetime import datetime, timezone

import anthropic
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Add parent to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.database import get_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"logs/cfa_hourly_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

INTERVAL_SEC = 3600  # 1 hour
MARKET_CLOSE_HOUR_ET = 16


def gather_data() -> dict:
    db = get_session()
    today = datetime.now().strftime("%Y-%m-%d")

    summary = db.execute(text("""
        SELECT
            count(*) as total_sells,
            sum(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
            sum(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as losses,
            sum(realized_pnl) as net_rpnl,
            avg(CASE WHEN realized_pnl > 0 THEN realized_pnl END) as avg_win,
            avg(CASE WHEN realized_pnl < 0 THEN realized_pnl END) as avg_loss
        FROM orders WHERE session_date = :d AND status = 'FILLED' AND side = 'SELL'
    """), {"d": today}).fetchone()

    by_reason = db.execute(text("""
        SELECT signal_reason, count(*), sum(realized_pnl)
        FROM orders WHERE session_date = :d AND status = 'FILLED' AND side = 'SELL'
        GROUP BY signal_reason
    """), {"d": today}).fetchall()

    by_model = db.execute(text("""
        SELECT o.model_id, m.name, m.strategy_type,
            count(*) as trades,
            sum(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
            sum(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as losses,
            sum(realized_pnl) as net
        FROM orders o JOIN models m ON o.model_id = m.id
        WHERE o.session_date = :d AND o.status = 'FILLED' AND o.side = 'SELL'
        GROUP BY o.model_id ORDER BY net DESC
    """), {"d": today}).fetchall()

    reject_counts = db.execute(text("""
        SELECT side, count(*) FROM orders
        WHERE session_date = :d AND status = 'REJECTED' GROUP BY side
    """), {"d": today}).fetchall()

    db.close()

    win_rate = (summary[1] / (summary[1] + summary[2]) * 100) if summary[1] and summary[2] else 0

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "date": today,
        "total_sells": summary[0],
        "wins": summary[1] or 0,
        "losses": summary[2] or 0,
        "win_rate_pct": round(win_rate, 1),
        "net_realized_pnl": round(summary[3] or 0, 2),
        "avg_win": round(summary[4] or 0, 2),
        "avg_loss": round(summary[5] or 0, 2),
        "sell_reasons": [
            {"reason": r[0] or "strategy_signal", "count": r[1], "pnl": round(r[2] or 0, 2)}
            for r in by_reason
        ],
        "models": [
            {"id": r[0], "name": r[1], "type": r[2], "sells": r[3],
             "wins": r[4] or 0, "losses": r[5] or 0, "net": round(r[6] or 0, 2)}
            for r in by_model
        ],
        "rejections": {r[0]: r[1] for r in reject_counts},
    }


def run_cfa_check(data: dict) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "ERROR: No ANTHROPIC_API_KEY"

    prompt = f"""You are the Chief Strategy Architect doing an hourly check-in on a live trading arena.

ARENA: 13 AI strategy models competing on live Alpaca paper trading, $2K capital each.
Exit mechanisms: hard stop-loss, take-profit, ratcheting trailing stops, patience stops.

CURRENT DATA:
{json.dumps(data, indent=2)}

In 3-5 sentences:
1. How are things going? Any red flags in win rate, avg_win/avg_loss ratio, or net PnL?
2. Any model that needs immediate attention (heavy losses, poor win rate)?
3. Are sell reasons distributed healthily (strategy signals vs stop-losses vs take-profits)?
4. Overall assessment: STABLE / CAUTION / DANGER

Be concise and data-driven."""

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def main():
    logger.info("Hourly CFA check started. Interval: %ds", INTERVAL_SEC)
    while True:
        try:
            now_et_hour = (datetime.now(timezone.utc).hour - 4) % 24  # rough ET
            if now_et_hour >= MARKET_CLOSE_HOUR_ET:
                logger.info("Market closed. Exiting.")
                break

            logger.info("Gathering trading data...")
            data = gather_data()
            logger.info(
                "Data: %d sells, %.0f%% WR, avg_win=$%.2f, avg_loss=$%.2f, net=$%.2f",
                data["total_sells"], data["win_rate_pct"],
                data["avg_win"], data["avg_loss"], data["net_realized_pnl"],
            )

            logger.info("Calling CFA...")
            assessment = run_cfa_check(data)
            logger.info("CFA Assessment:\n%s", assessment)

        except Exception:
            logger.exception("Hourly check failed")

        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
