"""Alpaca market data feed: WebSocket streaming and historical bar fetching.

Provides real-time 1-minute bars via WebSocket, trade update streaming for
instant fill notifications, and historical data for strategy warm-up periods.
All data is persisted to SQLite as it arrives.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Callable, Optional

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.stream import TradingStream

from sqlalchemy.exc import IntegrityError

from src.core.config import Config
from src.core.database import Bar, get_session
from src.core.strategy import BarData

logger = logging.getLogger(__name__)


class AlpacaDataFeed:
    """Manages Alpaca WebSocket streaming and historical data fetching."""

    def __init__(self, config: Config):
        self.config = config
        self._stream: Optional[StockDataStream] = None
        self._trading_stream: Optional[TradingStream] = None
        self._historical_client: Optional[StockHistoricalDataClient] = None
        self._bar_callbacks: list[Callable] = []
        self._bar_batch_callbacks: list[Callable] = []  # receive list[BarData]
        self._quote_callbacks: list[Callable] = []
        self._trade_update_callbacks: list[Callable] = []
        self._subscribed_quotes: set[str] = set()
        self._running = False
        self._last_bar_time: Optional[datetime] = None  # for heartbeat monitoring
        self._connection_state: str = "disconnected"  # disconnected, connected, reconnecting
        self._bars_received: int = 0  # diagnostic counter
        # Bar batching: accumulate bars and flush as a group to reduce event loop work
        self._bar_batch: list[BarData] = []
        self._batch_flush_handle: Optional[asyncio.TimerHandle] = None
        self._batch_window_sec: float = 3.0  # seconds to accumulate before flushing

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
                feed=DataFeed.SIP,
            )
        return self._stream

    @property
    def trading_stream(self) -> TradingStream:
        if self._trading_stream is None:
            self._trading_stream = TradingStream(
                api_key=self.config.alpaca.api_key,
                secret_key=self.config.alpaca.secret_key,
                paper=True,
            )
        return self._trading_stream

    def clear_callbacks(self) -> None:
        """Remove all registered callbacks. Call before re-registering for a new session."""
        self._bar_callbacks.clear()
        self._bar_batch_callbacks.clear()
        self._quote_callbacks.clear()
        self._trade_update_callbacks.clear()
        self._bar_batch.clear()
        if self._batch_flush_handle:
            self._batch_flush_handle.cancel()
            self._batch_flush_handle = None

    def on_bar(self, callback: Callable) -> None:
        """Register a callback to receive individual bars (called per bar)."""
        self._bar_callbacks.append(callback)

    def on_bar_batch(self, callback: Callable) -> None:
        """Register a callback to receive batched bars (called with list[BarData]).

        Bars are accumulated for up to `_batch_window_sec` seconds, then flushed
        as a single list. This dramatically reduces event loop work during trading.
        """
        self._bar_batch_callbacks.append(callback)

    async def _handle_bar(self, bar) -> None:
        """Process an incoming bar from Alpaca WebSocket.

        Bars are persisted immediately but callbacks are batched: individual
        bar callbacks fire right away while batch callbacks accumulate bars
        and flush after a short window (default 3s).  This keeps the event
        loop responsive for HTTP/WebSocket during trading.
        """
        self._last_bar_time = datetime.utcnow()
        self._connection_state = "connected"
        self._bars_received += 1

        bar_data = BarData(
            symbol=bar.symbol,
            timestamp=bar.timestamp,
            open=float(bar.open),
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
            volume=int(bar.volume),
        )

        # Verbose logging for first 5 bars, then every 10th
        if self._bars_received <= 5 or self._bars_received % 10 == 0:
            logger.info(
                f"BAR #{self._bars_received}: {bar_data.symbol} "
                f"close={bar_data.close:.2f} vol={bar_data.volume} "
                f"@ {bar_data.timestamp}"
            )

        # Persist bar in a thread to avoid blocking the event loop
        asyncio.get_event_loop().run_in_executor(None, self._persist_bar, bar_data)

        # Individual bar callbacks (legacy, called immediately)
        for callback in self._bar_callbacks:
            try:
                result = callback(bar_data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception(f"Error in bar callback for {bar_data.symbol}")

        # Batch callbacks: accumulate and flush after window
        if self._bar_batch_callbacks:
            self._bar_batch.append(bar_data)
            if self._batch_flush_handle is None:
                loop = asyncio.get_event_loop()
                self._batch_flush_handle = loop.call_later(
                    self._batch_window_sec,
                    lambda: asyncio.ensure_future(self._flush_bar_batch()),
                )

    async def _flush_bar_batch(self) -> None:
        """Flush accumulated bars to batch callbacks as a single list."""
        self._batch_flush_handle = None
        if not self._bar_batch:
            return

        batch = list(self._bar_batch)
        self._bar_batch.clear()

        logger.info(f"Flushing batch of {len(batch)} bars ({', '.join(b.symbol for b in batch)})")

        for callback in self._bar_batch_callbacks:
            try:
                result = callback(batch)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("Error in bar batch callback")

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
        except IntegrityError:
            session.rollback()
            logger.debug(f"Duplicate bar skipped: {bar_data.symbol} @ {bar_data.timestamp}")
        except Exception:
            session.rollback()
            logger.exception(f"Failed to persist bar for {bar_data.symbol}")
        finally:
            session.close()

    def fetch_historical_bars(
        self,
        symbols: list[str],
        days_back: int = 30,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict[str, list[BarData]]:
        """Fetch historical 1-minute bars.

        Returns a dict mapping symbol -> list of BarData, sorted by timestamp.
        Also persists all bars to SQLite.
        """
        end = end_date or datetime.now()
        start = start_date or (end - timedelta(days=days_back))

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            feed=DataFeed.SIP,
        )

        bars_response = self.historical_client.get_stock_bars(request)
        result: dict[str, list[BarData]] = {}
        session = get_session(self.config.db_path)

        try:
            skipped = 0
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
                    try:
                        nested = session.begin_nested()
                        session.add(db_bar)
                        nested.commit()
                    except IntegrityError:
                        nested.rollback()
                        skipped += 1

            session.commit()
            total = sum(len(v) for v in result.values())
            msg = f"Fetched {total} historical bars for {len(result)} symbols"
            if skipped:
                msg += f" ({skipped} duplicate bars skipped)"
            logger.info(msg)
        except Exception:
            session.rollback()
            logger.exception("Failed to persist historical bars")
        finally:
            session.close()

        return result

    def fetch_premarket_bars(
        self,
        symbols: list[str],
        since: Optional[datetime] = None,
    ) -> dict[str, list[BarData]]:
        """Fetch today's pre-market bars (4:00 AM – 9:30 AM ET) via SIP feed.

        Pre-market data requires SIP feed. Raises on failure so the caller
        can handle graceful degradation for free accounts.

        Args:
            since: Only return bars newer than this timestamp (for dedup).
        """
        from zoneinfo import ZoneInfo
        _ET = ZoneInfo("America/New_York")

        now = datetime.now(_ET)
        today_premarket_start = now.replace(hour=4, minute=0, second=0, microsecond=0)
        today_market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)

        start = since if since else today_premarket_start
        end = min(now, today_market_open)

        if start >= end:
            return {}

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            feed=DataFeed.SIP,
        )

        bars_response = self.historical_client.get_stock_bars(request)
        result: dict[str, list[BarData]] = {}

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
            result[symbol].sort(key=lambda b: b.timestamp)

        total = sum(len(v) for v in result.values())
        logger.info(
            f"Fetched {total} pre-market bars for {len(result)} symbols "
            f"(since {start.strftime('%H:%M ET')})"
        )
        return result

    def on_trade_update(self, callback: Callable) -> None:
        """Register a callback for trade update events (fills, cancels, rejects).

        Callback signature: callback(event: str, order_data: dict)
        where event is 'fill', 'partial_fill', 'canceled', 'rejected', etc.
        and order_data contains the Alpaca order fields.
        """
        self._trade_update_callbacks.append(callback)

    async def _handle_trade_update(self, data) -> None:
        """Process an incoming trade update from Alpaca TradingStream."""
        try:
            event = data.event
            order = data.order

            order_data = {
                "id": str(order.id) if hasattr(order, "id") else str(getattr(order, "id", "")),
                "symbol": getattr(order, "symbol", ""),
                "side": str(getattr(order, "side", "")),
                "qty": str(getattr(order, "qty", "0")),
                "filled_qty": str(getattr(order, "filled_qty", "0")),
                "filled_avg_price": str(getattr(order, "filled_avg_price", "0")) if getattr(order, "filled_avg_price", None) else None,
                "status": str(getattr(order, "status", "")),
                "order_class": str(getattr(order, "order_class", "")),
                "type": str(getattr(order, "type", "")),
            }

            logger.info(
                f"Trade update: {event} | {order_data['symbol']} "
                f"{order_data['side']} {order_data['filled_qty']}/{order_data['qty']} "
                f"@ {order_data['filled_avg_price'] or 'pending'}"
            )

            for callback in self._trade_update_callbacks:
                try:
                    result = callback(event, order_data)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    logger.exception("Error in trade update callback")

        except Exception:
            logger.exception("Error processing trade update")

    def on_quote(self, callback: Callable) -> None:
        """Register a callback to receive quote data.

        Callback signature: callback(symbol: str, bid: float, ask: float, timestamp)
        """
        self._quote_callbacks.append(callback)

    async def _handle_quote(self, quote) -> None:
        """Process an incoming quote from Alpaca WebSocket."""
        symbol = quote.symbol
        bid = float(quote.bid_price) if quote.bid_price else 0.0
        ask = float(quote.ask_price) if quote.ask_price else 0.0
        timestamp = quote.timestamp

        # Skip quotes with zero bid or ask (pre-market gaps, etc.)
        if bid <= 0 or ask <= 0:
            return

        for callback in self._quote_callbacks:
            try:
                result = callback(symbol, bid, ask, timestamp)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception(f"Error in quote callback for {symbol}")

    def subscribe_quotes(self, symbols: list[str]) -> None:
        """Subscribe to real-time quotes for the given symbols.

        Safe to call while streaming -- Alpaca SDK handles incremental subscriptions.
        Skips symbols already subscribed.
        """
        new_symbols = [s for s in symbols if s not in self._subscribed_quotes]
        if not new_symbols:
            return
        self.stream.subscribe_quotes(self._handle_quote, *new_symbols)
        self._subscribed_quotes.update(new_symbols)
        logger.info(f"Subscribed to quotes for {new_symbols} (total: {len(self._subscribed_quotes)})")

    def unsubscribe_quotes(self, symbols: list[str]) -> None:
        """Unsubscribe from real-time quotes for the given symbols."""
        to_remove = [s for s in symbols if s in self._subscribed_quotes]
        if not to_remove:
            return
        try:
            self.stream.unsubscribe_quotes(*to_remove)
        except Exception:
            logger.exception(f"Error unsubscribing quotes for {to_remove}")
        self._subscribed_quotes -= set(to_remove)
        logger.info(f"Unsubscribed quotes for {to_remove} (remaining: {len(self._subscribed_quotes)})")

    async def _run_data_stream(self, symbols: list[str]) -> None:
        """Run the market data WebSocket with auto-reconnect on failure."""
        max_retries = 10
        backoff = 1.0
        subscribed = False

        for attempt in range(max_retries):
            if not self._running:
                break
            try:
                if not subscribed:
                    self.stream.subscribe_bars(self._handle_bar, *symbols)
                    subscribed = True
                logger.info(
                    f"Starting data stream for {len(symbols)} symbols"
                    + (f" (reconnect #{attempt})" if attempt > 0 else "")
                )
                self._connection_state = "connected"
                backoff = 1.0  # reset on successful connection
                await self.stream._run_forever()
            except asyncio.CancelledError:
                logger.info("Data stream cancelled")
                return
            except Exception:
                self._connection_state = "reconnecting"
                logger.exception(
                    f"Data stream error (attempt {attempt + 1}/{max_retries}). "
                    f"Reconnecting in {backoff:.0f}s..."
                )
                # Reset stream object for fresh connection
                try:
                    await self.stream.close()
                except Exception:
                    pass
                self._stream = None
                subscribed = False

                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)  # exponential backoff, cap at 60s

        if self._running:
            self._connection_state = "failed"
            logger.error(f"Data stream failed after {max_retries} attempts")

    async def _run_trading_stream(self) -> None:
        """Run the trading WebSocket with auto-reconnect on failure."""
        max_retries = 10
        backoff = 1.0
        subscribed = False

        for attempt in range(max_retries):
            if not self._running:
                break
            try:
                if not subscribed:
                    self.trading_stream.subscribe_trade_updates(self._handle_trade_update)
                    subscribed = True
                logger.info(
                    "Starting trading stream for fill notifications"
                    + (f" (reconnect #{attempt})" if attempt > 0 else "")
                )
                await self.trading_stream._run_forever()
            except asyncio.CancelledError:
                logger.info("Trading stream cancelled")
                return
            except Exception:
                logger.exception(
                    f"Trading stream error (attempt {attempt + 1}/{max_retries}). "
                    f"Reconnecting in {backoff:.0f}s..."
                )
                try:
                    await self.trading_stream.close()
                except Exception:
                    pass
                self._trading_stream = None
                subscribed = False

                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

        if self._running:
            logger.error(f"Trading stream failed after {max_retries} attempts")

    async def _heartbeat_monitor(self) -> None:
        """Monitor data stream health. Warn if no bars for 2 min, force reconnect at 5 min."""
        started_at = datetime.utcnow()
        while self._running:
            await asyncio.sleep(30)
            if not self._running:
                break

            if self._last_bar_time:
                ref_time = self._last_bar_time
            else:
                # No bars ever received — measure from stream start
                ref_time = started_at

            elapsed = (datetime.utcnow() - ref_time).total_seconds()

            if elapsed > 300:
                self._connection_state = "dead"
                logger.error(
                    f"HEARTBEAT: No bar in {elapsed:.0f}s — connection presumed dead. "
                    f"Forcing data stream reconnect."
                )
                if self._stream:
                    try:
                        await self._stream.close()
                    except Exception:
                        pass
                self._last_bar_time = datetime.utcnow()
                started_at = datetime.utcnow()
            elif elapsed > 120:
                self._connection_state = "stale"
                logger.warning(
                    f"HEARTBEAT: No bar received in {elapsed:.0f}s. "
                    f"Connection may be degraded."
                )
            elif elapsed > 60:
                logger.info(f"Heartbeat: last bar {elapsed:.0f}s ago")

    async def start_streaming(self, symbols: list[str]) -> None:
        """Start streaming bars + trade updates as background tasks.

        Tasks are stored in self._tasks so stop_streaming() can cancel them.
        This method blocks until all tasks complete or are cancelled.
        """
        self._running = True
        self._connection_state = "connected"
        self._bars_received = 0
        self._tasks: list[asyncio.Task] = []

        logger.info(f"start_streaming: creating tasks for {symbols}")
        self._tasks.append(asyncio.create_task(self._run_data_stream(symbols)))
        self._tasks.append(asyncio.create_task(self._heartbeat_monitor()))

        if self._trade_update_callbacks:
            logger.info("Starting combined data + trading streams")
            self._tasks.append(asyncio.create_task(self._run_trading_stream()))
        else:
            logger.info("Starting data stream only (no trade callbacks)")

        logger.info(f"start_streaming: awaiting {len(self._tasks)} tasks via gather")
        try:
            results = await asyncio.gather(*self._tasks, return_exceptions=True)
            # Log any task exceptions for diagnostics
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"start_streaming: task {i} failed: {result!r}")
        except asyncio.CancelledError:
            logger.info("start_streaming: gather cancelled")
        logger.info(f"start_streaming: finished. Total bars received: {self._bars_received}")

    async def stop_streaming(self) -> None:
        """Stop all WebSocket streams by cancelling tasks and closing connections."""
        self._running = False
        self._connection_state = "disconnected"
        self._subscribed_quotes.clear()

        # Flush any pending bar batch before shutdown
        if self._batch_flush_handle:
            self._batch_flush_handle.cancel()
            self._batch_flush_handle = None
        if self._bar_batch:
            await self._flush_bar_batch()

        # Cancel all streaming tasks
        for task in getattr(self, '_tasks', []):
            if not task.done():
                task.cancel()

        # Wait briefly for tasks to finish
        if hasattr(self, '_tasks') and self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()

        if self._stream:
            try:
                await self._stream.close()
            except Exception:
                logger.exception("Error closing data stream")
            self._stream = None

        if self._trading_stream:
            try:
                await self._trading_stream.close()
            except Exception:
                logger.exception("Error closing trading stream")
            self._trading_stream = None

        logger.info("All streams stopped")
