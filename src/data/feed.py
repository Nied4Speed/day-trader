"""Alpaca market data feed: WebSocket streaming and historical bar fetching.

Provides real-time 1-minute bars via WebSocket and historical data for
strategy warm-up periods. All data is persisted to SQLite as it arrives.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Callable, Optional

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from src.core.config import Config
from src.core.database import Bar, get_session
from src.core.strategy import BarData

logger = logging.getLogger(__name__)


class AlpacaDataFeed:
    """Manages Alpaca WebSocket streaming and historical data fetching."""

    def __init__(self, config: Config):
        self.config = config
        self._stream: Optional[StockDataStream] = None
        self._historical_client: Optional[StockHistoricalDataClient] = None
        self._bar_callbacks: list[Callable] = []
        self._running = False

    @property
    def historical_client(self) -> StockHistoricalDataClient:
        if self._historical_client is None:
            self._historical_client = StockHistoricalDataClient(
                api_key=self.config.alpaca.api_key,
                secret_key=self.config.alpaca.secret_key,
            )
        return self._historical_client

    @property
    def stream(self) -> StockDataStream:
        if self._stream is None:
            self._stream = StockDataStream(
                api_key=self.config.alpaca.api_key,
                secret_key=self.config.alpaca.secret_key,
            )
        return self._stream

    def on_bar(self, callback: Callable) -> None:
        """Register a callback to receive completed bar data."""
        self._bar_callbacks.append(callback)

    async def _handle_bar(self, bar) -> None:
        """Process an incoming bar from Alpaca WebSocket."""
        bar_data = BarData(
            symbol=bar.symbol,
            timestamp=bar.timestamp,
            open=float(bar.open),
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
            volume=int(bar.volume),
        )

        self._persist_bar(bar_data)

        for callback in self._bar_callbacks:
            try:
                result = callback(bar_data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception(f"Error in bar callback for {bar_data.symbol}")

    def _persist_bar(self, bar_data: BarData) -> None:
        """Write a bar to SQLite."""
        session = get_session(self.config.db_path)
        try:
            db_bar = Bar(
                symbol=bar_data.symbol,
                timestamp=bar_data.timestamp,
                open=bar_data.open,
                high=bar_data.high,
                low=bar_data.low,
                close=bar_data.close,
                volume=bar_data.volume,
            )
            session.merge(db_bar)
            session.commit()
        except Exception:
            session.rollback()
            logger.exception(f"Failed to persist bar for {bar_data.symbol}")
        finally:
            session.close()

    def fetch_historical_bars(
        self,
        symbols: list[str],
        days_back: int = 30,
    ) -> dict[str, list[BarData]]:
        """Fetch historical 1-minute bars for strategy warm-up.

        Returns a dict mapping symbol -> list of BarData, sorted by timestamp.
        Also persists all bars to SQLite.
        """
        end = datetime.now()
        start = end - timedelta(days=days_back)

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
        )

        bars_response = self.historical_client.get_stock_bars(request)
        result: dict[str, list[BarData]] = {}
        session = get_session(self.config.db_path)

        try:
            for symbol, bars in bars_response.data.items():
                result[symbol] = []
                for bar in bars:
                    bar_data = BarData(
                        symbol=symbol,
                        timestamp=bar.timestamp,
                        open=float(bar.open),
                        high=float(bar.high),
                        low=float(bar.low),
                        close=float(bar.close),
                        volume=int(bar.volume),
                    )
                    result[symbol].append(bar_data)

                    db_bar = Bar(
                        symbol=symbol,
                        timestamp=bar.timestamp,
                        open=bar_data.open,
                        high=bar_data.high,
                        low=bar_data.low,
                        close=bar_data.close,
                        volume=bar_data.volume,
                    )
                    session.merge(db_bar)

            session.commit()
            logger.info(
                f"Fetched {sum(len(v) for v in result.values())} historical bars "
                f"for {len(result)} symbols"
            )
        except Exception:
            session.rollback()
            logger.exception("Failed to persist historical bars")
        finally:
            session.close()

        return result

    async def start_streaming(self, symbols: list[str]) -> None:
        """Start streaming 1-minute bars for the given symbols."""
        self._running = True
        self.stream.subscribe_bars(self._handle_bar, *symbols)
        logger.info(f"Starting bar stream for {len(symbols)} symbols")
        try:
            await self.stream._run_forever()
        except asyncio.CancelledError:
            logger.info("Bar stream cancelled")
        except Exception:
            logger.exception("Bar stream error")

    async def stop_streaming(self) -> None:
        """Stop the WebSocket stream."""
        self._running = False
        if self._stream:
            try:
                await self._stream.close()
            except Exception:
                logger.exception("Error closing stream")
            self._stream = None
        logger.info("Bar stream stopped")
