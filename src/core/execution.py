"""Central execution handler routing all model orders to Alpaca.

No strategy ever calls Alpaca directly. All orders flow through this handler,
which enforces risk limits, applies transaction costs, and persists fills.
"""

import asyncio
import logging
import threading
import time
from datetime import datetime
from typing import Optional

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide as AlpacaOrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest

from src.core.config import Config
from src.core.database import (
    Order,
    OrderSide,
    OrderStatus,
    Position,
    TradingModel,
    get_session,
)
from src.core.strategy import TradeSignal

logger = logging.getLogger(__name__)

# Alpaca trading API: 200 req/min, 10 req/sec burst.
# Market data: 10k/min with Algo Trader Plus subscription.
MAX_REQUESTS_PER_SECOND = 10  # burst limit
MIN_REQUEST_INTERVAL = 1.0 / MAX_REQUESTS_PER_SECOND  # 100ms between calls


class RiskLimitExceeded(Exception):
    """Raised when an order would violate risk limits."""
    pass


class ExecutionHandler:
    """Routes orders to Alpaca, enforces risk limits, tracks fills."""

    def __init__(self, config: Config, simulate: bool = False):
        self.config = config
        self.simulate = simulate  # True = fill at bar close, no Alpaca calls
        self._client: Optional[TradingClient] = None
        self._daily_pnl: dict[int, float] = {}  # model_id -> daily pnl
        self._session_number: int = 1
        self._last_api_call: float = 0.0
        self._api_lock = threading.Lock()
        self._last_prices: dict[str, float] = {}  # symbol -> last close
        # Wash trade cooldown: (model_id, symbol) -> last trade timestamp
        self._last_trade_time: dict[tuple[int, str], float] = {}
        # Pending orders: alpaca_order_id -> {model_id, signal, session_date, db_order_id}
        self._pending_orders: dict[str, dict] = {}
        # Retry stats
        self.retry_stats: dict[str, int] = {"buy_retries": 0, "sell_retries": 0, "buy_retry_success": 0, "sell_retry_success": 0}

    @property
    def client(self) -> TradingClient:
        if self._client is None:
            self._client = TradingClient(
                api_key=self.config.alpaca.api_key,
                secret_key=self.config.alpaca.secret_key,
                paper=True,
            )
        return self._client

    def handle_trade_update(self, event: str, order_data: dict) -> Optional[dict]:
        """Handle a trade update from TradingStream.

        Called by the feed layer when Alpaca pushes a fill/cancel/reject.
        Updates DB position + model capital for fills. Cleans up pending orders.

        Returns a dict with fill/reject info for arena-level bookkeeping,
        or None if no action needed. This replaces the old callback mechanism
        to avoid blocking the event loop with sync work.
        """
        alpaca_id = order_data.get("id", "")
        pending = self._pending_orders.get(alpaca_id)

        if event in ("fill", "filled"):
            filled_price = float(order_data["filled_avg_price"]) if order_data.get("filled_avg_price") else None
            filled_qty = float(order_data.get("filled_qty", 0))

            if pending and filled_price and filled_qty:
                model_id = pending["model_id"]
                signal = pending["signal"]
                session_date = pending["session_date"]

                db = get_session(self.config.db_path)
                try:
                    # Update the pending order to FILLED
                    db_order = db.query(Order).get(pending["db_order_id"])
                    if db_order:
                        tx_cost = filled_price * filled_qty * self.config.arena.transaction_cost_pct
                        db_order.status = OrderStatus.FILLED
                        db_order.fill_price = filled_price
                        db_order.fill_quantity = filled_qty
                        db_order.transaction_cost = tx_cost
                        db_order.filled_at = datetime.utcnow()

                        self._update_position(db, model_id, signal, filled_price, filled_qty, tx_cost, db_order=db_order)
                        self._record_trade_time(model_id, signal.symbol)
                        db.commit()

                        logger.info(
                            f"TradingStream fill: model {model_id} "
                            f"{signal.side} {filled_qty} {signal.symbol} @ {filled_price:.2f}"
                        )
                except Exception:
                    db.rollback()
                    logger.exception(f"Error processing TradingStream fill for {alpaca_id}")
                finally:
                    db.close()

                self._pending_orders.pop(alpaca_id, None)
                return {
                    "type": "fill", "model_id": model_id,
                    "symbol": signal.symbol, "side": signal.side,
                    "price": filled_price, "qty": filled_qty,
                }

            elif not pending and filled_price:
                # Fill for an order we don't have pending (bracket leg, etc.)
                logger.info(
                    f"TradingStream fill (untracked): {order_data.get('symbol')} "
                    f"{order_data.get('side')} {order_data.get('filled_qty')} "
                    f"@ {filled_price:.2f} (order_class={order_data.get('order_class')})"
                )
                return {
                    "type": "fill", "model_id": None,
                    "symbol": order_data.get("symbol", ""),
                    "side": order_data.get("side", ""),
                    "price": filled_price,
                    "qty": float(order_data.get("filled_qty", 0)),
                }

        elif event in ("canceled", "cancelled", "rejected", "expired"):
            if pending:
                db = get_session(self.config.db_path)
                try:
                    db_order = db.query(Order).get(pending["db_order_id"])
                    if db_order:
                        db_order.status = OrderStatus.REJECTED
                        db_order.rejected_reason = f"Alpaca {event}"
                        db.commit()
                except Exception:
                    db.rollback()
                    logger.exception(f"Error processing TradingStream {event} for {alpaca_id}")
                finally:
                    db.close()

                logger.warning(
                    f"TradingStream {event}: model {pending['model_id']} "
                    f"{pending['signal'].symbol}"
                )
                self._pending_orders.pop(alpaca_id, None)
                return {
                    "type": "reject", "model_id": pending["model_id"],
                    "symbol": pending["signal"].symbol, "event": event,
                }

        return None

    def _throttle(self) -> None:
        """Rate-limit Alpaca API calls to stay under burst limits (sync)."""
        with self._api_lock:
            now = time.monotonic()
            elapsed = now - self._last_api_call
            if elapsed < MIN_REQUEST_INTERVAL:
                time.sleep(MIN_REQUEST_INTERVAL - elapsed)
            self._last_api_call = time.monotonic()

    async def _async_throttle(self) -> None:
        """Rate-limit Alpaca API calls without blocking the event loop."""
        now = time.monotonic()
        elapsed = now - self._last_api_call
        if elapsed < MIN_REQUEST_INTERVAL:
            await asyncio.sleep(MIN_REQUEST_INTERVAL - elapsed)
        self._last_api_call = time.monotonic()

    def reset_daily_limits(self) -> None:
        """Reset daily P&L tracking at session start."""
        self._daily_pnl.clear()

    def check_risk_limits(
        self,
        model_id: int,
        signal: TradeSignal,
        current_capital: float,
    ) -> None:
        """Check if an order would violate risk limits. Raises RiskLimitExceeded.

        Only enforces daily loss limit - strategies are free to size positions
        however they want to maximize profit.
        """
        # Check max daily loss
        daily_pnl = self._daily_pnl.get(model_id, 0.0)
        max_loss = current_capital * self.config.arena.max_daily_loss_pct
        if daily_pnl < -max_loss:
            raise RiskLimitExceeded(
                f"Model {model_id} exceeded max daily loss: "
                f"{daily_pnl:.2f} < -{max_loss:.2f}"
            )

        # Check that buy orders don't exceed available capital
        if signal.side == "buy":
            estimated_cost = signal.quantity * (signal.limit_price or 0)
            if estimated_cost == 0:
                estimated_cost = signal.quantity * 100  # placeholder
            if estimated_cost > current_capital * 1.5:  # allow some margin
                raise RiskLimitExceeded(
                    f"Model {model_id} order cost {estimated_cost:.2f} "
                    f"exceeds available capital {current_capital:.2f}"
                )

    def _check_wash_trade(self, model_id: int, symbol: str) -> bool:
        """Return True if this trade would violate wash trade cooldown."""
        if self.simulate:
            return False
        key = (model_id, symbol)
        last = self._last_trade_time.get(key, 0.0)
        elapsed = time.monotonic() - last
        cooldown = self.config.arena.wash_trade_cooldown_sec
        if elapsed < cooldown:
            logger.debug(
                f"Wash trade cooldown: model {model_id} {symbol} "
                f"({elapsed:.1f}s < {cooldown:.0f}s)"
            )
            return True
        return False

    def _record_trade_time(self, model_id: int, symbol: str) -> None:
        """Record the time of a trade for wash trade cooldown."""
        self._last_trade_time[(model_id, symbol)] = time.monotonic()

    def _submit_with_retry(self, request, is_sell: bool = False):
        """Submit an order to Alpaca with retry on retryable errors.

        Sell orders: 3 retries with 0.5s/1.0s/2.0s backoff (closing positions is critical).
        Buy orders: 2 retries with 0.25s/0.5s backoff (missed opportunity recovery).

        Only retries on retryable errors (429 rate limit, network timeouts).
        Permanent rejections (insufficient funds, invalid symbol) fail immediately.
        """
        if is_sell:
            max_retries = 3
            backoff_delays = [0.5, 1.0, 2.0]
        else:
            max_retries = 2
            backoff_delays = [0.25, 0.5]

        side_label = "sell" if is_sell else "buy"

        for attempt in range(max_retries + 1):
            try:
                self._throttle()
                result = self.client.submit_order(request)
                if attempt > 0:
                    # Retry succeeded
                    stat_key = f"{side_label}_retry_success"
                    self.retry_stats[stat_key] = self.retry_stats.get(stat_key, 0) + 1
                return result
            except APIError as e:
                status_code = getattr(e, "status_code", None)
                is_retryable = status_code in (429, 500, 502, 503, 504)

                if is_retryable and attempt < max_retries:
                    delay = backoff_delays[attempt]
                    stat_key = f"{side_label}_retries"
                    self.retry_stats[stat_key] = self.retry_stats.get(stat_key, 0) + 1
                    logger.warning(
                        f"Retryable error ({status_code}) on {side_label} "
                        f"attempt {attempt + 1}/{max_retries + 1}, "
                        f"retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    continue
                if is_retryable:
                    logger.error(
                        f"Retryable error ({status_code}) on {side_label} "
                        f"after {max_retries + 1} attempts, giving up: {e}"
                    )
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                if attempt < max_retries:
                    delay = backoff_delays[attempt]
                    stat_key = f"{side_label}_retries"
                    self.retry_stats[stat_key] = self.retry_stats.get(stat_key, 0) + 1
                    logger.warning(
                        f"Network error on {side_label} attempt "
                        f"{attempt + 1}/{max_retries + 1}, retrying in {delay}s: {e}"
                    )
                    time.sleep(delay)
                    continue
                raise

    def submit_order(
        self,
        model_id: int,
        signal: TradeSignal,
        current_capital: float,
        session_date: str,
    ) -> Optional[Order]:
        """Submit an order to Alpaca after risk checks."""
        # Wash trade cooldown check (before DB session to avoid unnecessary work)
        if self._check_wash_trade(model_id, signal.symbol):
            return None

        db = get_session(self.config.db_path)
        try:
            # Risk check
            try:
                self.check_risk_limits(model_id, signal, current_capital)
            except RiskLimitExceeded as e:
                order = Order(
                    model_id=model_id,
                    session_date=session_date,
                    session_number=self._session_number,
                    symbol=signal.symbol,
                    side=OrderSide.BUY if signal.side == "buy" else OrderSide.SELL,
                    quantity=signal.quantity,
                    order_type=signal.order_type,
                    limit_price=signal.limit_price,
                    status=OrderStatus.REJECTED,
                    rejected_reason=str(e),
                    submitted_at=datetime.utcnow(),
                )
                db.add(order)
                db.commit()
                logger.warning(f"Order rejected for model {model_id}: {e}")
                return order

            # Fill the order (simulated or via Alpaca)
            fill_price = self._last_prices.get(signal.symbol, signal.limit_price or 100.0)
            fill_qty = signal.quantity
            alpaca_order_id = None

            if self.simulate and fill_price:
                # Apply simulated slippage: buys fill higher, sells fill lower
                slip = fill_price * self.config.arena.slippage_bps / 10_000
                if signal.side == "buy":
                    fill_price += slip
                else:
                    fill_price -= slip

            tx_cost = fill_price * fill_qty * self.config.arena.transaction_cost_pct if fill_price else 0.0

            if not self.simulate:
                # Alpaca doesn't allow fractional short sells. Before submitting
                # a sell, verify we hold a position. Skip if not.
                if signal.side == "sell":
                    pos = (
                        db.query(Position)
                        .filter(Position.model_id == model_id, Position.symbol == signal.symbol)
                        .first()
                    )
                    if not pos or pos.quantity <= 0:
                        logger.debug(
                            f"Skipping sell for model {model_id} {signal.symbol}: no position held"
                        )
                        return None

                # Live mode: submit to Alpaca, let TradingStream handle fill
                alpaca_side = (
                    AlpacaOrderSide.BUY if signal.side == "buy" else AlpacaOrderSide.SELL
                )
                try:
                    request = self._build_order_request(signal, alpaca_side)

                    alpaca_order = self._submit_with_retry(
                        request, is_sell=(signal.side == "sell")
                    )
                    alpaca_order_id = str(alpaca_order.id)

                    # Check if already filled in the response (common for market orders)
                    if alpaca_order.filled_avg_price:
                        fill_price = float(alpaca_order.filled_avg_price)
                        fill_qty = float(alpaca_order.filled_qty or signal.quantity)
                    else:
                        # Don't poll — TradingStream will push the fill.
                        # Record as PENDING; TradingStream callback updates to FILLED.
                        fill_price = None
                        fill_qty = signal.quantity

                except Exception as e:
                    order = Order(
                        model_id=model_id, session_date=session_date,
                        session_number=self._session_number, symbol=signal.symbol,
                        side=OrderSide.BUY if signal.side == "buy" else OrderSide.SELL,
                        quantity=signal.quantity, order_type=signal.order_type,
                        limit_price=signal.limit_price, status=OrderStatus.REJECTED,
                        rejected_reason=str(e), submitted_at=datetime.utcnow(),
                    )
                    db.add(order)
                    db.commit()
                    logger.warning(f"Alpaca order rejected for model {model_id}: {e}")
                    return order

            if fill_price:
                tx_cost = fill_price * fill_qty * self.config.arena.transaction_cost_pct

            status = OrderStatus.FILLED if fill_price else OrderStatus.PENDING
            order = Order(
                model_id=model_id, session_date=session_date,
                session_number=self._session_number, symbol=signal.symbol,
                side=OrderSide.BUY if signal.side == "buy" else OrderSide.SELL,
                quantity=signal.quantity, order_type=signal.order_type,
                limit_price=signal.limit_price, status=status,
                fill_price=fill_price, fill_quantity=fill_qty,
                transaction_cost=tx_cost if fill_price else 0.0,
                alpaca_order_id=alpaca_order_id,
                submitted_at=datetime.utcnow(),
                filled_at=datetime.utcnow() if fill_price else None,
            )
            db.add(order)
            db.flush()  # get order.id for pending tracking

            if fill_price and fill_qty:
                self._update_position(db, model_id, signal, fill_price, fill_qty, tx_cost, db_order=order)
                self._record_trade_time(model_id, signal.symbol)
            elif alpaca_order_id and not fill_price:
                # Track as pending — TradingStream will resolve it
                self._pending_orders[alpaca_order_id] = {
                    "model_id": model_id,
                    "signal": signal,
                    "session_date": session_date,
                    "db_order_id": order.id,
                }

            db.commit()
            if fill_price:
                logger.info(
                    f"{'[SIM] ' if self.simulate else ''}Order filled for model {model_id}: "
                    f"{signal.side} {fill_qty} {signal.symbol} @ {fill_price:.2f}"
                )
            elif alpaca_order_id:
                logger.info(
                    f"Order submitted (pending fill via stream) for model {model_id}: "
                    f"{signal.side} {signal.quantity} {signal.symbol}"
                )

        finally:
            db.close()

    async def async_submit_order(
        self,
        model_id: int,
        signal: TradeSignal,
        current_capital: float,
        session_date: str,
    ) -> Optional[Order]:
        """Async version of submit_order — doesn't block the event loop.

        Uses asyncio.to_thread for the actual Alpaca API call so the event
        loop remains free for quote processing and other models.
        """
        return await asyncio.to_thread(
            self.submit_order, model_id, signal, current_capital, session_date
        )

    def _build_order_request(self, signal: TradeSignal, alpaca_side):
        """Build an Alpaca order request from a TradeSignal.

        For buy signals with stop_loss_pct/take_profit_pct, creates a bracket
        order. Otherwise creates a simple market/limit order.
        """
        # NOTE: Bracket orders disabled — our on_quote() already manages
        # stop-loss/take-profit exits. Bracket legs on Alpaca conflict with
        # manual sells (wash trade rejections + insufficient qty holds).

        # Simple order (market or limit)
        # Alpaca only supports fractional shares on market orders.
        # For limit orders, round up to whole shares.
        if signal.order_type == "limit" and signal.limit_price:
            whole_qty = max(1, int(signal.quantity + 0.999))  # ceil
            return LimitOrderRequest(
                symbol=signal.symbol, qty=whole_qty,
                side=alpaca_side, time_in_force=TimeInForce.DAY,
                limit_price=signal.limit_price,
            )
        return MarketOrderRequest(
            symbol=signal.symbol, qty=signal.quantity,
            side=alpaca_side, time_in_force=TimeInForce.DAY,
        )

    def cancel_bracket_legs(self, alpaca_order_id: str) -> None:
        """Cancel outstanding bracket legs (stop-loss/take-profit) after manual exit.

        When on_quote() triggers a sell, any bracket legs from the original
        entry order must be cancelled to avoid double-exits.
        """
        if self.simulate or not alpaca_order_id:
            return
        try:
            self._throttle()
            self.client.cancel_order_by_id(alpaca_order_id)
            logger.info(f"Cancelled bracket legs for order {alpaca_order_id}")
        except Exception as e:
            # Order may already be filled/cancelled — that's fine
            logger.debug(f"Could not cancel bracket order {alpaca_order_id}: {e}")

    def _cancel_open_orders_for_symbol(self, symbol: str) -> None:
        """Cancel all open orders for a symbol before submitting a sell.

        Clears bracket legs (stop-loss/take-profit) that would otherwise cause
        wash trade rejections or hold shares unavailable.
        """
        if self.simulate:
            return
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            self._throttle()
            open_orders = self.client.get_orders(
                GetOrdersRequest(
                    status=QueryOrderStatus.OPEN,
                    symbols=[symbol],
                )
            )
            for order in open_orders:
                try:
                    self._throttle()
                    self.client.cancel_order_by_id(order.id)
                    logger.info(f"Cancelled open order {order.id} ({order.side} {symbol}) before sell")
                except Exception:
                    pass  # Already filled/cancelled
            if open_orders:
                # Brief pause to let cancellations propagate
                time.sleep(0.2)
        except Exception as e:
            logger.warning(f"Failed to cancel open orders for {symbol}: {e}")

    def _update_position(
        self,
        db,
        model_id: int,
        signal: TradeSignal,
        fill_price: float,
        fill_qty: float,
        tx_cost: float,
        db_order: Optional[Order] = None,
    ) -> None:
        """Update the model's position after a fill."""
        position = (
            db.query(Position)
            .filter(Position.model_id == model_id, Position.symbol == signal.symbol)
            .first()
        )

        if position is None:
            position = Position(
                model_id=model_id,
                symbol=signal.symbol,
                quantity=0,
                avg_entry_price=0.0,
                current_price=fill_price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
            )
            db.add(position)

        if signal.side == "buy":
            # Update average entry price
            total_cost = position.avg_entry_price * position.quantity + fill_price * fill_qty
            position.quantity += fill_qty
            if position.quantity > 0:
                position.avg_entry_price = total_cost / position.quantity

            # Deduct buy cost from capital
            buy_cost = fill_price * fill_qty + tx_cost
            model = db.query(TradingModel).get(model_id)
            if model:
                model.current_capital -= buy_cost
        else:
            # Selling: return proceeds to capital
            if position.quantity > 0:
                pnl = (fill_price - position.avg_entry_price) * fill_qty - tx_cost
                position.realized_pnl += pnl
                self._daily_pnl[model_id] = self._daily_pnl.get(model_id, 0.0) + pnl
                # Record per-trade P&L on the order for win rate calculation
                if db_order is not None:
                    db_order.realized_pnl = pnl

                # Credit sale proceeds (cost basis + P&L)
                proceeds = fill_price * fill_qty - tx_cost
                model = db.query(TradingModel).get(model_id)
                if model:
                    model.current_capital += proceeds

            position.quantity -= fill_qty
            if position.quantity <= 0:
                position.quantity = 0
                position.avg_entry_price = 0.0

        position.current_price = fill_price
        position.updated_at = datetime.utcnow()

    def liquidate_all(self, session_date: str) -> int:
        """Force-close all open positions via our order flow."""
        db = get_session(self.config.db_path)
        closed = 0
        try:
            positions = db.query(Position).filter(Position.quantity > 0).all()
            for pos in positions:
                signal = TradeSignal(
                    symbol=pos.symbol, side="sell", quantity=pos.quantity
                )
                model = db.query(TradingModel).get(pos.model_id)
                capital = model.current_capital if model else 0.0
                db.close()
                self.submit_order(
                    model_id=pos.model_id,
                    signal=signal,
                    current_capital=capital,
                    session_date=session_date,
                )
                closed += 1
                db = get_session(self.config.db_path)
            logger.info(f"Liquidated {closed} open positions (EOD safety net)")
        finally:
            db.close()
        return closed

    def liquidate_all_alpaca(self) -> int:
        """Hard safety net: close ALL positions directly on Alpaca.

        Bypasses our order flow entirely. Use when our DB positions
        don't match Alpaca (shared account problem).
        """
        if self.simulate:
            return 0
        try:
            self._throttle()
            positions = self.client.get_all_positions()
            if not positions:
                logger.info("Alpaca: no positions to liquidate")
                return 0
            logger.info(f"Alpaca: closing {len(positions)} positions directly")
            self._throttle()
            self.client.close_all_positions(cancel_orders=True)
            logger.info("Alpaca: all positions closed")
            return len(positions)
        except Exception:
            logger.exception("Failed to liquidate via Alpaca API")
            return 0

    def update_positions_price(self, model_id: int, symbol: str, price: float) -> None:
        """Update current price and unrealized P&L for a position."""
        db = get_session(self.config.db_path)
        try:
            position = (
                db.query(Position)
                .filter(Position.model_id == model_id, Position.symbol == symbol)
                .first()
            )
            if position and position.quantity > 0:
                position.current_price = price
                position.unrealized_pnl = (price - position.avg_entry_price) * position.quantity
                position.updated_at = datetime.utcnow()
                db.commit()
        finally:
            db.close()
