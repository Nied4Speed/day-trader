"""Central execution handler routing all model orders to Alpaca.

No strategy ever calls Alpaca directly. All orders flow through this handler,
which enforces risk limits, applies transaction costs, and persists fills.
"""

import logging
from datetime import datetime
from typing import Optional

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


class RiskLimitExceeded(Exception):
    """Raised when an order would violate risk limits."""
    pass


class ExecutionHandler:
    """Routes orders to Alpaca, enforces risk limits, tracks fills."""

    def __init__(self, config: Config):
        self.config = config
        self._client: Optional[TradingClient] = None
        self._daily_pnl: dict[int, float] = {}  # model_id -> daily pnl

    @property
    def client(self) -> TradingClient:
        if self._client is None:
            self._client = TradingClient(
                api_key=self.config.alpaca.api_key,
                secret_key=self.config.alpaca.secret_key,
                paper=True,
            )
        return self._client

    def reset_daily_limits(self) -> None:
        """Reset daily P&L tracking at session start."""
        self._daily_pnl.clear()

    def check_risk_limits(
        self,
        model_id: int,
        signal: TradeSignal,
        current_capital: float,
    ) -> None:
        """Check if an order would violate risk limits. Raises RiskLimitExceeded."""
        db = get_session(self.config.db_path)
        try:
            # Check max daily loss
            daily_pnl = self._daily_pnl.get(model_id, 0.0)
            max_loss = current_capital * self.config.arena.max_daily_loss_pct
            if daily_pnl < -max_loss:
                raise RiskLimitExceeded(
                    f"Model {model_id} exceeded max daily loss: "
                    f"{daily_pnl:.2f} < -{max_loss:.2f}"
                )

            # Check max position size
            estimated_cost = signal.quantity * (signal.limit_price or 0)
            # Use a rough estimate if no limit price
            if estimated_cost == 0:
                estimated_cost = signal.quantity * 100  # placeholder
            max_position_value = current_capital * self.config.arena.max_position_pct
            if signal.side == "buy" and estimated_cost > max_position_value:
                raise RiskLimitExceeded(
                    f"Model {model_id} position size {estimated_cost:.2f} "
                    f"exceeds max {max_position_value:.2f}"
                )

            # Check max open positions
            open_positions = (
                db.query(Position)
                .filter(Position.model_id == model_id, Position.quantity != 0)
                .count()
            )
            if signal.side == "buy" and open_positions >= self.config.arena.max_open_positions:
                raise RiskLimitExceeded(
                    f"Model {model_id} has {open_positions} open positions "
                    f"(max {self.config.arena.max_open_positions})"
                )
        finally:
            db.close()

    def submit_order(
        self,
        model_id: int,
        signal: TradeSignal,
        current_capital: float,
        session_date: str,
    ) -> Optional[Order]:
        """Submit an order to Alpaca after risk checks.

        Returns the Order record on success, or None if rejected.
        """
        db = get_session(self.config.db_path)
        try:
            # Risk check
            try:
                self.check_risk_limits(model_id, signal, current_capital)
            except RiskLimitExceeded as e:
                order = Order(
                    model_id=model_id,
                    session_date=session_date,
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

            # Submit to Alpaca
            alpaca_side = (
                AlpacaOrderSide.BUY if signal.side == "buy" else AlpacaOrderSide.SELL
            )

            try:
                if signal.order_type == "limit" and signal.limit_price:
                    request = LimitOrderRequest(
                        symbol=signal.symbol,
                        qty=signal.quantity,
                        side=alpaca_side,
                        time_in_force=TimeInForce.DAY,
                        limit_price=signal.limit_price,
                    )
                else:
                    request = MarketOrderRequest(
                        symbol=signal.symbol,
                        qty=signal.quantity,
                        side=alpaca_side,
                        time_in_force=TimeInForce.DAY,
                    )

                alpaca_order = self.client.submit_order(request)

                # For paper trading, market orders fill immediately
                fill_price = None
                fill_qty = None
                status = OrderStatus.PENDING

                if alpaca_order.filled_avg_price:
                    fill_price = float(alpaca_order.filled_avg_price)
                    fill_qty = int(alpaca_order.filled_qty or signal.quantity)
                    status = OrderStatus.FILLED

                # Apply transaction cost
                tx_cost = 0.0
                if fill_price:
                    tx_cost = fill_price * fill_qty * self.config.arena.transaction_cost_pct

                order = Order(
                    model_id=model_id,
                    session_date=session_date,
                    symbol=signal.symbol,
                    side=OrderSide.BUY if signal.side == "buy" else OrderSide.SELL,
                    quantity=signal.quantity,
                    order_type=signal.order_type,
                    limit_price=signal.limit_price,
                    status=status,
                    fill_price=fill_price,
                    fill_quantity=fill_qty,
                    transaction_cost=tx_cost,
                    alpaca_order_id=str(alpaca_order.id),
                    submitted_at=datetime.utcnow(),
                    filled_at=datetime.utcnow() if fill_price else None,
                )
                db.add(order)

                # Update position if filled
                if fill_price and fill_qty:
                    self._update_position(db, model_id, signal, fill_price, fill_qty, tx_cost)

                db.commit()
                logger.info(
                    f"Order filled for model {model_id}: "
                    f"{signal.side} {fill_qty} {signal.symbol} @ {fill_price}"
                )
                return order

            except Exception as e:
                order = Order(
                    model_id=model_id,
                    session_date=session_date,
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
                logger.error(f"Alpaca order failed for model {model_id}: {e}")
                return order

        finally:
            db.close()

    def _update_position(
        self,
        db,
        model_id: int,
        signal: TradeSignal,
        fill_price: float,
        fill_qty: int,
        tx_cost: float,
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
        else:
            # Selling: realize P&L
            if position.quantity > 0:
                pnl = (fill_price - position.avg_entry_price) * fill_qty - tx_cost
                position.realized_pnl += pnl
                self._daily_pnl[model_id] = self._daily_pnl.get(model_id, 0.0) + pnl

                # Update model capital
                model = db.query(TradingModel).get(model_id)
                if model:
                    model.current_capital += pnl

            position.quantity -= fill_qty
            if position.quantity <= 0:
                position.quantity = 0
                position.avg_entry_price = 0.0

        position.current_price = fill_price
        position.updated_at = datetime.utcnow()

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
