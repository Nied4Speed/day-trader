"""Alpaca options data fetcher for IV and Greeks.

Fetches option chain snapshots to extract implied volatility, put/call ratios,
and delta skew for CFA review context. Uses OptionHistoricalDataClient from
alpaca-py.
"""

import logging
from datetime import date, timedelta
from typing import Any

from src.core.config import Config

logger = logging.getLogger(__name__)


def fetch_options_summary(symbols: list[str], config: Config) -> dict[str, Any]:
    """Fetch options IV and Greeks summary for given symbols.

    For each symbol, gets the nearest-expiry option chain and extracts:
    - implied_volatility: ATM call IV (proxy for expected move)
    - put_call_ratio: put volume / call volume
    - delta_skew: put delta - call delta (negative = normal, positive = fear)

    Returns {symbol: {iv, put_call_ratio, delta_skew}} or empty dict on failure.
    """
    try:
        from alpaca.data.historical.option import OptionHistoricalDataClient
        from alpaca.data.requests import OptionChainRequest
    except ImportError:
        logger.warning("alpaca option imports not available, skipping options data")
        return {}

    try:
        client = OptionHistoricalDataClient(
            api_key=config.alpaca.api_key,
            secret_key=config.alpaca.secret_key,
        )
    except Exception:
        logger.exception("Failed to create OptionHistoricalDataClient")
        return {}

    # Use expiration window: today to 30 days out (nearest-expiry focus)
    today = date.today()
    exp_max = today + timedelta(days=30)

    results: dict[str, Any] = {}
    for symbol in symbols:
        try:
            chain = client.get_option_chain(
                OptionChainRequest(
                    underlying_symbol=symbol,
                    expiration_date_gte=today.isoformat(),
                    expiration_date_lte=exp_max.isoformat(),
                )
            )

            if not chain:
                continue

            # chain is dict[str, Snapshot] keyed by option symbol
            calls = []
            puts = []
            for opt_symbol, snapshot in chain.items():
                # Determine call vs put from option symbol convention
                # Standard: symbol + expiry + C/P + strike
                opt_upper = opt_symbol.upper()
                greeks = snapshot.greeks if hasattr(snapshot, "greeks") else None
                iv = snapshot.implied_volatility if hasattr(snapshot, "implied_volatility") else None

                entry = {
                    "symbol": opt_symbol,
                    "iv": float(iv) if iv is not None else None,
                    "delta": float(greeks.delta) if greeks and hasattr(greeks, "delta") and greeks.delta is not None else None,
                    "gamma": float(greeks.gamma) if greeks and hasattr(greeks, "gamma") and greeks.gamma is not None else None,
                    "volume": int(snapshot.latest_trade.size) if hasattr(snapshot, "latest_trade") and snapshot.latest_trade and hasattr(snapshot.latest_trade, "size") else 0,
                }

                # Classify as call or put
                # Look for C or P after the date portion in the OCC symbol
                if "C" in opt_upper[len(symbol):]:
                    calls.append(entry)
                elif "P" in opt_upper[len(symbol):]:
                    puts.append(entry)

            if not calls and not puts:
                continue

            # Compute aggregates
            call_ivs = [c["iv"] for c in calls if c["iv"] is not None and c["iv"] > 0]
            put_ivs = [p["iv"] for p in puts if p["iv"] is not None and p["iv"] > 0]
            call_volume = sum(c["volume"] for c in calls)
            put_volume = sum(p["volume"] for p in puts)
            call_deltas = [c["delta"] for c in calls if c["delta"] is not None]
            put_deltas = [p["delta"] for p in puts if p["delta"] is not None]

            avg_iv = None
            if call_ivs:
                avg_iv = round(sum(call_ivs) / len(call_ivs) * 100, 2)  # as percentage

            put_call_ratio = None
            if call_volume > 0:
                put_call_ratio = round(put_volume / call_volume, 3)

            delta_skew = None
            if call_deltas and put_deltas:
                avg_call_delta = sum(call_deltas) / len(call_deltas)
                avg_put_delta = sum(put_deltas) / len(put_deltas)
                delta_skew = round(avg_put_delta - avg_call_delta, 4)

            results[symbol] = {
                "implied_volatility_pct": avg_iv,
                "put_call_ratio": put_call_ratio,
                "delta_skew": delta_skew,
                "call_count": len(calls),
                "put_count": len(puts),
                "call_volume": call_volume,
                "put_volume": put_volume,
            }

        except Exception:
            logger.debug(f"Options data unavailable for {symbol}", exc_info=True)
            continue

    logger.info(f"Options data fetched for {len(results)}/{len(symbols)} symbols")
    return results
