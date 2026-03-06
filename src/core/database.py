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
    mutation_memory = Column(JSON, nullable=True, default=None)

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
    quantity = Column(Float, nullable=False)
    order_type = Column(String(20), nullable=False, default="market")
    limit_price = Column(Float, nullable=True)
    status = Column(Enum(OrderStatus), nullable=False, default=OrderStatus.PENDING)
    fill_price = Column(Float, nullable=True)
    fill_quantity = Column(Float, nullable=True)
    session_number = Column(Integer, nullable=False, default=1)  # 1 or 2
    transaction_cost = Column(Float, nullable=False, default=0.0)
    realized_pnl = Column(Float, nullable=True)  # P&L for sell orders
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
    quantity = Column(Float, nullable=False, default=0.0)
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
    session_number = Column(Integer, nullable=False, default=1)  # 1 or 2
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


class ModelSummary(Base):
    """Per-model reflection after a session or self-improvement phase."""

    __tablename__ = "model_summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False, index=True)
    session_date = Column(String(10), nullable=False, index=True)
    session_number = Column(Integer, nullable=True)  # 1, 2, or None for improvement
    summary_type = Column(String(20), nullable=False)  # "post_session" or "post_improvement"
    return_pct = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    total_trades = Column(Integer, nullable=True)
    win_rate = Column(Float, nullable=True)
    fitness = Column(Float, nullable=True)
    rank = Column(Integer, nullable=True)
    param_changes = Column(JSON, nullable=True)  # {param: {old: x, new: y}} for improvement
    reflection = Column(Text, nullable=False)  # the narrative text
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    model = relationship("TradingModel")


class NewsArticle(Base):
    """A news article fetched from Alpaca with sentiment score."""

    __tablename__ = "news_articles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alpaca_news_id = Column(String(100), nullable=False, unique=True, index=True)
    headline = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    source = Column(String(100), nullable=True)
    symbols = Column(JSON, nullable=False, default=list)
    sentiment_score = Column(Float, nullable=False, default=0.0)
    published_at = Column(DateTime, nullable=True)
    fetched_at = Column(DateTime, nullable=False, default=datetime.utcnow)


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


class DailyLedger(Base):
    """One row per model per day for tracking day-over-day performance."""

    __tablename__ = "daily_ledger"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False, index=True)
    session_date = Column(String(10), nullable=False, index=True)
    start_capital = Column(Float, nullable=False)
    end_capital = Column(Float, nullable=False)
    daily_return_pct = Column(Float, nullable=False, default=0.0)
    cumulative_return_pct = Column(Float, nullable=False, default=0.0)
    total_trades = Column(Integer, nullable=False, default=0)
    generation = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    model = relationship("TradingModel")

    __table_args__ = (
        UniqueConstraint("model_id", "session_date", name="uq_ledger_model_date"),
    )


class CfaReview(Base):
    """Daily CFA-style review of trading performance."""

    __tablename__ = "cfa_reviews"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_date = Column(String(10), nullable=False, unique=True, index=True)
    review_json = Column(JSON, nullable=True)
    raw_response = Column(Text, nullable=True)
    model_used = Column(String(100), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class SessionRecord(Base):
    """Record of a trading session (two per day: session 1 and session 2)."""

    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_date = Column(String(10), nullable=False, index=True)
    session_number = Column(Integer, nullable=False, default=1)  # 1 or 2
    generation = Column(Integer, nullable=False)
    started_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)
    total_bars = Column(Integer, nullable=False, default=0)
    total_trades = Column(Integer, nullable=False, default=0)
    summary = Column(JSON, nullable=True)

    __table_args__ = (
        UniqueConstraint("session_date", "session_number", name="uq_session_date_num"),
    )


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
    # Enable WAL mode for concurrent reads during writes (critical for dashboard)
    from sqlalchemy import text, event
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
        conn.commit()
    # Set WAL on every new connection too
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()
    # Add columns that may not exist on older databases
    with engine.connect() as conn:
        for col in ("mutation_memory",):
            try:
                conn.execute(text(f"ALTER TABLE models ADD COLUMN {col} JSON"))
                conn.commit()
            except Exception:
                conn.rollback()  # column already exists
        # Add realized_pnl to orders table (per-trade P&L tracking)
        try:
            conn.execute(text("ALTER TABLE orders ADD COLUMN realized_pnl FLOAT"))
            conn.commit()
        except Exception:
            conn.rollback()  # column already exists
    return engine


def get_session(db_path: str = "day_trader.db") -> Session:
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine(db_path)
        _SessionLocal = sessionmaker(bind=engine)
    return _SessionLocal()


# Separate engine/session for dashboard reads with short busy_timeout
_dashboard_engine = None
_DashboardSessionLocal = None


def get_dashboard_session(db_path: str = "day_trader.db") -> Session:
    """Get a DB session with short busy_timeout for dashboard reads.

    Won't block the event loop if the arena holds a write lock —
    raises OperationalError after 500ms instead of waiting 5s.
    """
    global _dashboard_engine, _DashboardSessionLocal
    if _dashboard_engine is None:
        _dashboard_engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            connect_args={"check_same_thread": False},
        )
        from sqlalchemy import event

        @event.listens_for(_dashboard_engine, "connect")
        def _set_fast_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=500")
            cursor.execute("PRAGMA query_only=ON")
            cursor.close()

        _DashboardSessionLocal = sessionmaker(bind=_dashboard_engine)
    return _DashboardSessionLocal()


def reset_engine():
    """Reset the engine and session factory (useful for testing)."""
    global _engine, _SessionLocal
    if _engine:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
