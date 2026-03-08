"""Sector mapping and cross-sector correlation analysis.

Provides a hardcoded sector map for core symbols + common screener picks,
and computes per-sector statistics from stored bar data for CFA review.
"""

import logging
from collections import defaultdict
from typing import Any

import numpy as np

from src.core.database import Bar, get_session

logger = logging.getLogger(__name__)

# Hardcoded sector map — core 10 + ~40 common screener picks
SYMBOL_SECTORS: dict[str, str] = {
    # Core 10
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Communication Services",
    "AMZN": "Consumer Discretionary",
    "NVDA": "Technology",
    "META": "Communication Services",
    "TSLA": "Consumer Discretionary",
    "JPM": "Financials",
    "V": "Financials",
    "SPY": "ETF-Broad Market",
    # Technology
    "AMD": "Technology",
    "INTC": "Technology",
    "CRM": "Technology",
    "ORCL": "Technology",
    "AVGO": "Technology",
    "MU": "Technology",
    "QCOM": "Technology",
    "AMAT": "Technology",
    "ADBE": "Technology",
    "NOW": "Technology",
    "SHOP": "Technology",
    "SNOW": "Technology",
    "PLTR": "Technology",
    "DELL": "Technology",
    "SMCI": "Technology",
    "ARM": "Technology",
    "MRVL": "Technology",
    # Financials
    "BAC": "Financials",
    "GS": "Financials",
    "MS": "Financials",
    "WFC": "Financials",
    "C": "Financials",
    "SCHW": "Financials",
    "BLK": "Financials",
    "AXP": "Financials",
    # Healthcare
    "JNJ": "Healthcare",
    "UNH": "Healthcare",
    "PFE": "Healthcare",
    "ABBV": "Healthcare",
    "MRK": "Healthcare",
    "LLY": "Healthcare",
    "BMY": "Healthcare",
    "MRNA": "Healthcare",
    # Energy
    "XOM": "Energy",
    "CVX": "Energy",
    "OXY": "Energy",
    "SLB": "Energy",
    "COP": "Energy",
    "XLE": "Energy",
    # Consumer
    "WMT": "Consumer Staples",
    "COST": "Consumer Staples",
    "KO": "Consumer Staples",
    "PEP": "Consumer Staples",
    "PG": "Consumer Staples",
    "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary",
    "HD": "Consumer Discretionary",
    "LOW": "Consumer Discretionary",
    "TGT": "Consumer Discretionary",
    # Communication Services
    "NFLX": "Communication Services",
    "DIS": "Communication Services",
    "GOOG": "Communication Services",
    "SNAP": "Communication Services",
    # Industrials
    "BA": "Industrials",
    "CAT": "Industrials",
    "UPS": "Industrials",
    "GE": "Industrials",
    "HON": "Industrials",
    "RTX": "Industrials",
    # Sector ETFs
    "XLF": "ETF-Financials",
    "XLK": "ETF-Technology",
    "XLV": "ETF-Healthcare",
    "XLY": "ETF-Consumer Discretionary",
    "XLP": "ETF-Consumer Staples",
    "XLI": "ETF-Industrials",
    "XLB": "ETF-Materials",
    "XLU": "ETF-Utilities",
    "XLRE": "ETF-Real Estate",
    "QQQ": "ETF-Nasdaq 100",
    "IWM": "ETF-Small Cap",
    "DIA": "ETF-Dow Jones",
}


def get_sector(symbol: str) -> str:
    """Return the sector for a symbol, or 'Unknown'."""
    return SYMBOL_SECTORS.get(symbol, "Unknown")


def compute_sector_stats(db_path: str, session_date: str) -> dict[str, Any]:
    """Compute per-sector statistics from bar data for the given date.

    Returns:
        - sector_returns: per-sector avg return, total volume, symbol list
        - correlation_matrix: cross-sector return correlations
        - trade_concentration: which sectors were most/least traded
    """
    from sqlalchemy import func as sqla_func

    db = get_session(db_path)
    try:
        bars = (
            db.query(Bar)
            .filter(sqla_func.date(Bar.timestamp) == session_date)
            .order_by(Bar.symbol, Bar.timestamp)
            .all()
        )

        if not bars:
            return {"sector_returns": {}, "correlation_matrix": {}, "sector_concentration": {}}

        # Group bars by symbol
        by_symbol: dict[str, list] = {}
        for b in bars:
            by_symbol.setdefault(b.symbol, []).append(b)

        # Compute per-symbol returns and group by sector
        sector_data: dict[str, list[dict]] = defaultdict(list)
        for sym, sym_bars in by_symbol.items():
            if len(sym_bars) < 2:
                continue
            day_open = sym_bars[0].open
            day_close = sym_bars[-1].close
            day_volume = sum(b.volume for b in sym_bars)
            if day_open <= 0:
                continue
            ret_pct = (day_close - day_open) / day_open * 100
            sector = get_sector(sym)
            sector_data[sector].append({
                "symbol": sym,
                "return_pct": round(ret_pct, 4),
                "volume": day_volume,
                "close": day_close,
            })

        # Per-sector aggregates
        sector_returns: dict[str, dict] = {}
        for sector, entries in sorted(sector_data.items()):
            returns = [e["return_pct"] for e in entries]
            sector_returns[sector] = {
                "avg_return_pct": round(sum(returns) / len(returns), 4),
                "total_volume": sum(e["volume"] for e in entries),
                "symbol_count": len(entries),
                "symbols": [e["symbol"] for e in entries],
                "best": max(entries, key=lambda e: e["return_pct"])["symbol"],
                "worst": min(entries, key=lambda e: e["return_pct"])["symbol"],
            }

        # Cross-sector correlation matrix (from per-bar close returns)
        # Build per-symbol minute-return series, then average by sector
        sector_minute_returns: dict[str, list[float]] = {}
        min_bars = float("inf")

        for sector, entries in sector_data.items():
            # Use the symbol with most bars as the sector representative
            best_sym = max(entries, key=lambda e: len(by_symbol.get(e["symbol"], [])))["symbol"]
            sym_bars = by_symbol[best_sym]
            if len(sym_bars) < 10:
                continue
            closes = [b.close for b in sym_bars]
            returns = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes)) if closes[i - 1] > 0]
            if returns:
                sector_minute_returns[sector] = returns
                min_bars = min(min_bars, len(returns))

        # Compute correlation matrix (truncate all to same length)
        correlation_matrix: dict[str, dict[str, float]] = {}
        sectors_with_data = sorted(sector_minute_returns.keys())
        if len(sectors_with_data) >= 2 and min_bars >= 10:
            truncated = min(int(min_bars), 300)  # cap for performance
            arrays = []
            for s in sectors_with_data:
                arrays.append(sector_minute_returns[s][:truncated])
            mat = np.array(arrays)
            try:
                corr = np.corrcoef(mat)
                for i, s1 in enumerate(sectors_with_data):
                    correlation_matrix[s1] = {}
                    for j, s2 in enumerate(sectors_with_data):
                        correlation_matrix[s1][s2] = round(float(corr[i, j]), 3)
            except Exception:
                logger.debug("Failed to compute sector correlation matrix")

        return {
            "sector_returns": sector_returns,
            "correlation_matrix": correlation_matrix,
            "sector_concentration": {
                "most_active": max(sector_returns, key=lambda s: sector_returns[s]["total_volume"])
                if sector_returns else None,
                "least_active": min(sector_returns, key=lambda s: sector_returns[s]["total_volume"])
                if sector_returns else None,
                "sector_count": len(sector_returns),
            },
        }
    finally:
        db.close()
