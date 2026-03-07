"""Pre-market symbol screener using Alpaca's Screener API.

Discovers the day's most actively traded stocks and merges them into the
trading universe alongside the core symbol list.  Fetches a large pool of
most-active-by-volume, applies a price floor, and returns exactly
``screener_max_additions`` symbols.
"""

import logging

from alpaca.data.historical import ScreenerClient, StockHistoricalDataClient
from alpaca.data.enums import DataFeed, MostActivesBy
from alpaca.data.requests import (
    MostActivesRequest,
    StockSnapshotRequest,
)

from src.core.config import Config

logger = logging.getLogger(__name__)

# Leveraged / inverse ETFs — too volatile, dominated by decay, poor for
# short-horizon strategies.  Covers the tickers that showed up in our
# 2026-03-06 session plus common families (Direxion, ProShares, etc.).
LEVERAGED_ETF_BLOCKLIST: set[str] = {
    # ProShares
    "TQQQ", "SQQQ", "UPRO", "SPXU", "SOXL", "SOXS", "SPDN",
    "QLD", "QID", "SSO", "SDS", "UDOW", "SDOW", "UVXY", "SVXY",
    # Direxion
    "TNA", "TZA", "LABU", "LABD", "FAS", "FAZ", "NUGT", "DUST",
    "TECL", "TECS", "SPXL", "SPXS", "FNGU", "FNGD",
    # Volatility
    "UVIX", "SVIX", "VXX", "VIXY",
    # Crypto-linked
    "BITO", "BITI",
    # Other inverse
    "SH", "PSQ", "DOG", "RWM",
}


class SymbolScreener:
    """Queries Alpaca screener APIs for daily symbol discovery."""

    def __init__(self, config: Config):
        self.config = config
        self._screener = ScreenerClient(
            api_key=config.alpaca.api_key,
            secret_key=config.alpaca.secret_key,
        )
        self._historical = StockHistoricalDataClient(
            api_key=config.alpaca.api_key,
            secret_key=config.alpaca.secret_key,
        )

    def screen(self) -> list[str]:
        """Return up to ``screener_max_additions`` new symbols.

        Fetches top 100 most-active by volume, removes symbols already in the
        core list, applies the $5 price floor, and takes the first
        ``screener_max_additions`` that pass.  Volume-rank order is preserved.
        """
        cfg = self.config.arena
        max_add = cfg.screener_max_additions
        existing = set(cfg.symbols)

        # Fetch a large pool so we have enough after price filtering
        try:
            actives = self._screener.get_most_actives(
                MostActivesRequest(by=MostActivesBy.VOLUME, top=100)
            )
            candidates = [
                s.symbol for s in actives.most_actives
                if s.symbol not in existing
                and s.symbol not in LEVERAGED_ETF_BLOCKLIST
            ]
            logger.info(
                f"Screener: {len(actives.most_actives)} most-active fetched, "
                f"{len(candidates)} new candidates"
            )
        except Exception:
            logger.exception("Screener: failed to fetch most-actives")
            return []

        if not candidates:
            logger.info("Screener: no new candidates found")
            return []

        # Price filter then cap
        filtered = self._filter_by_price(candidates, cfg.screener_min_price)
        result = filtered[:max_add]

        logger.info(
            f"Screener: {len(result)} symbols passed "
            f"(from {len(candidates)} candidates, min_price=${cfg.screener_min_price})"
        )
        return result

    def _filter_by_price(
        self, symbols: list[str], min_price: float
    ) -> list[str]:
        """Remove symbols trading below min_price using latest snapshots."""
        if not symbols or min_price <= 0:
            return symbols

        try:
            snapshots = self._historical.get_stock_snapshot(
                StockSnapshotRequest(
                    symbol_or_symbols=symbols,
                    feed=DataFeed.SIP,
                )
            )
            passed: list[str] = []
            for sym in symbols:  # preserve original order
                snap = snapshots.get(sym)
                if not snap:
                    continue
                price = (
                    float(snap.latest_trade.price)
                    if snap.latest_trade
                    else 0.0
                )
                if price >= min_price:
                    passed.append(sym)
                else:
                    logger.debug(
                        f"Screener: {sym} filtered out "
                        f"(price=${price:.2f} < ${min_price})"
                    )
            return passed
        except Exception:
            logger.exception(
                "Screener: snapshot price filter failed, keeping all candidates"
            )
            return symbols
