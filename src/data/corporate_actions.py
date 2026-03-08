"""Alpaca corporate actions fetcher.

Queries upcoming dividends, splits, mergers, and spinoffs for traded symbols
to give CFA review context about events that may affect strategy performance.
"""

import logging
from datetime import date, timedelta
from typing import Any

from src.core.config import Config

logger = logging.getLogger(__name__)


def fetch_upcoming_events(
    symbols: list[str],
    config: Config,
    lookahead_days: int = 5,
) -> list[dict[str, Any]]:
    """Fetch upcoming corporate actions for the given symbols.

    Queries Alpaca's corporate announcements API for dividends, splits,
    mergers, and spinoffs within the lookahead window.

    Returns list of {symbol, event_type, date, details} dicts.
    """
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.enums import CorporateActionDateType, CorporateActionType
        from alpaca.trading.requests import GetCorporateAnnouncementsRequest
    except ImportError:
        logger.warning("alpaca trading imports not available, skipping corporate actions")
        return []

    try:
        client = TradingClient(
            api_key=config.alpaca.api_key,
            secret_key=config.alpaca.secret_key,
            paper=True,
        )
    except Exception:
        logger.exception("Failed to create TradingClient for corporate actions")
        return []

    today = date.today()
    end_date = today + timedelta(days=lookahead_days)

    ca_types = [
        CorporateActionType.DIVIDEND,
        CorporateActionType.SPLIT,
        CorporateActionType.MERGER,
        CorporateActionType.SPINOFF,
    ]

    events: list[dict[str, Any]] = []
    symbol_set = set(symbols)

    try:
        announcements = client.get_corporate_announcements(
            GetCorporateAnnouncementsRequest(
                ca_types=ca_types,
                since=today,
                until=end_date,
                date_type=CorporateActionDateType.EX_DATE,
            )
        )

        if not announcements:
            logger.info("No upcoming corporate actions found")
            return []

        for ann in announcements:
            # Filter to our traded symbols only
            ann_symbol = getattr(ann, "symbol", None)
            if not ann_symbol or ann_symbol not in symbol_set:
                continue

            event = {
                "symbol": ann_symbol,
                "event_type": str(getattr(ann, "ca_type", "unknown")),
                "ex_date": str(getattr(ann, "ex_date", "")),
                "record_date": str(getattr(ann, "record_date", "")),
                "payable_date": str(getattr(ann, "payable_date", "")),
                "details": {},
            }

            # Add type-specific details
            ca_type = str(getattr(ann, "ca_type", ""))
            if "dividend" in ca_type.lower():
                event["details"] = {
                    "cash_amount": float(getattr(ann, "cash", 0) or 0),
                    "frequency": str(getattr(ann, "frequency", "")),
                }
            elif "split" in ca_type.lower():
                event["details"] = {
                    "old_rate": float(getattr(ann, "old_rate", 0) or 0),
                    "new_rate": float(getattr(ann, "new_rate", 0) or 0),
                }

            events.append(event)

        logger.info(
            f"Corporate actions: {len(events)} events found for "
            f"{len(symbol_set)} symbols (next {lookahead_days} days)"
        )

    except Exception:
        logger.exception("Failed to fetch corporate announcements")
        return []

    return events
