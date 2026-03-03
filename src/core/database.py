"""SQLite database schema and session management.

All market data, trades, positions, model metadata, and generational lineage
persist here. SQLite serves as the integration bus between the trading engine,
evolution engine, and dashboard.
"""

from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


class OrderSide(PyEnum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(PyEnum):
    PENDING = "pending"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class ModelStatus(PyEnum):
    ACTIVE = "active"
    ELIMINATED = "eliminated"
    RETIRED = "retired"


class Bar(Base):
    """1-minute OHLCV bar data from Alpaca."""

    __tablename__ = "bars"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)

    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", name="uq_bar_symbol_ts"),
    )


class TradingModel(Base):
    """A trading strategy instance competing in the arena."""

    __tablename__ = "models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    strategy_type = Column(String(50), nullable=False)
    parameters = Column(JSON, nullable=False, default=dict)
    generation = Column(Integer, nullable=False, default=1)
    parent_ids = Column(JSON, nullable=True)
    genetic_operation = Column(String(50), nullable=True)
    status = Column(Enum(ModelStatus), nullable=False, default=ModelStatus.ACTIVE)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    eliminated_at = Column(DateTime, nullable=True)
    initial_capital = Column(Float, nullable=False, default=100_000.0)
    current_capital = Column(Float, nullable=False, default=100_000.0)

    orders = relationship("Order", back_populates="model")
    positions = relationship("Position", back_populates="model")
    performance_snapshots = relationship("PerformanceSnapshot", back_populates="model")


class Order(Base):
    """An order placed by a model, routed through the execution handler."""

    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False, index=True)
    session_date = Column(String(10), nullable=False, index=True)
    symbol = Column(String(10), nullable=False)
    side = Column(Enum(OrderSide), nullable=False)
    quantity = Column(Integer, nullable=False)
    order_type = Column(String(20), nullable=False, default="market")
    limit_price = Column(Float, nullable=True)
    status = Column(Enum(OrderStatus), nullable=False, default=OrderStatus.PENDING)
    fill_price = Column(Float, nullable=True)
    fill_quantity = Column(Integer, nullable=True)
    transaction_cost = Column(Float, nullable=False, default=0.0)
    alpaca_order_id = Column(String(100), nullable=True)
    submitted_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    filled_at = Column(DateTime, nullable=True)
    rejected_reason = Column(Text, nullable=True)

    model = relationship("TradingModel", back_populates="orders")


class Position(Base):
    """Current position held by a model."""

    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False, index=True)
    symbol = Column(String(10), nullable=False)
    quantity = Column(Integer, nullable=False, default=0)
    avg_entry_price = Column(Float, nullable=False, default=0.0)
    current_price = Column(Float, nullable=False, default=0.0)
    unrealized_pnl = Column(Float, nullable=False, default=0.0)
    realized_pnl = Column(Float, nullable=False, default=0.0)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    model = relationship("TradingModel", back_populates="positions")

    __table_args__ = (
        UniqueConstraint("model_id", "symbol", name="uq_position_model_symbol"),
    )


class PerformanceSnapshot(Base):
    """Point-in-time performance metrics for a model."""

    __tablename__ = "performance_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False, index=True)
    session_date = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    equity = Column(Float, nullable=False)
    total_pnl = Column(Float, nullable=False, default=0.0)
    return_pct = Column(Float, nullable=False, default=0.0)
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=False, default=0.0)
    win_rate = Column(Float, nullable=True)
    total_trades = Column(Integer, nullable=False, default=0)
    winning_trades = Column(Integer, nullable=False, default=0)

    model = relationship("TradingModel", back_populates="performance_snapshots")


class GenerationRecord(Base):
    """Record of a generation's composition and outcomes."""

    __tablename__ = "generations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    generation_number = Column(Integer, nullable=False, unique=True)
    session_date = Column(String(10), nullable=False)
    model_ids = Column(JSON, nullable=False)
    eliminated_ids = Column(JSON, nullable=False, default=list)
    survivor_ids = Column(JSON, nullable=False, default=list)
    offspring_ids = Column(JSON, nullable=False, default=list)
    best_fitness = Column(Float, nullable=True)
    avg_fitness = Column(Float, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class SessionRecord(Base):
    """Record of a trading session."""

    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_date = Column(String(10), nullable=False, unique=True)
    generation = Column(Integer, nullable=False)
    started_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)
    total_bars = Column(Integer, nullable=False, default=0)
    total_trades = Column(Integer, nullable=False, default=0)
    summary = Column(JSON, nullable=True)


# Database setup

_engine = None
_SessionLocal = None


def get_engine(db_path: str = "day_trader.db"):
    global _engine
    if _engine is None:
        _engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            connect_args={"check_same_thread": False},
        )
    return _engine


def init_db(db_path: str = "day_trader.db"):
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return engine


def get_session(db_path: str = "day_trader.db") -> Session:
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine(db_path)
        _SessionLocal = sessionmaker(bind=engine)
    return _SessionLocal()


def reset_engine():
    """Reset the engine and session factory (useful for testing)."""
    global _engine, _SessionLocal
    if _engine:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
