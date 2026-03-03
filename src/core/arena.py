"""Arena orchestrator: manages the trading session lifecycle.

Handles market-hours gating, bar fan-out to all models, session start/stop,
and coordinates between the data feed, execution handler, and performance
tracker. This is the central nervous system of the trading arena.
"""

import asyncio
import logging
from datetime import datetime, time
from typing import Optional
from zoneinfo import ZoneInfo

from src.core.config import Config
from src.core.database import (
    ModelStatus,
    SessionRecord,
    TradingModel,
    get_session,
    init_db,
)
from src.core.execution import ExecutionHandler
from src.core.performance import PerformanceTracker
from src.core.strategy import BarData, Strategy
from src.data.feed import AlpacaDataFeed
from src.strategies.registry import create_strategy

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)


class Arena:
    """Manages the full trading session lifecycle."""

    def __init__(self, config: Config):
        self.config = config
        self.feed = AlpacaDataFeed(config)
        self.execution = ExecutionHandler(config)
        self.tracker = PerformanceTracker(config.db_path)
        self._models: dict[int, Strategy] = {}  # model_id -> Strategy instance
        self._model_records: dict[int, TradingModel] = {}
        self._session_date: str = ""
        self._running = False
        self._bar_count = 0

    def _is_market_hours(self) -> bool:
        """Check if we're within US market hours."""
        now = datetime.now(ET).time()
        return MARKET_OPEN <= now <= MARKET_CLOSE

    def _load_or_create_models(self) -> list[TradingModel]:
        """Load active models from DB, or create initial pool if none exist."""
        db = get_session(self.config.db_path)
        try:
            active_models = (
                db.query(TradingModel)
                .filter(TradingModel.status == ModelStatus.ACTIVE)
                .all()
            )

            if active_models:
                return active_models

            # First run: create initial model pool
            logger.info("No active models found. Creating initial pool.")
            from src.strategies.registry import get_strategy_types

            strategy_types = get_strategy_types()
            models = []

            # Create ~2 models per strategy type to fill the pool
            model_count = self.config.arena.model_count
            type_idx = 0
            for i in range(model_count):
                st = strategy_types[type_idx % len(strategy_types)]
                type_idx += 1

                model = TradingModel(
                    name=f"{st}_gen1_{i+1}",
                    strategy_type=st,
                    parameters={},
                    generation=1,
                    initial_capital=self.config.arena.initial_capital,
                    current_capital=self.config.arena.initial_capital,
                )
                db.add(model)
                models.append(model)

            db.commit()

            # Refresh to get IDs
            for m in models:
                db.refresh(m)

            # Set default parameters
            for m in models:
                strategy = create_strategy(m.strategy_type, m.name)
                m.parameters = strategy.get_params()
            db.commit()

            logger.info(f"Created {len(models)} initial models")
            return models

        finally:
            db.close()

    def _instantiate_strategies(self, models: list[TradingModel]) -> None:
        """Create live Strategy instances from DB model records."""
        self._models.clear()
        self._model_records.clear()

        for model in models:
            strategy = create_strategy(
                model.strategy_type,
                model.name,
                params=model.parameters,
            )
            self._models[model.id] = strategy
            self._model_records[model.id] = model

        logger.info(f"Instantiated {len(self._models)} strategy instances")

    def _fan_out_bar(self, bar: BarData) -> None:
        """Distribute a completed bar to all active models."""
        self._bar_count += 1
        session_date = self._session_date

        for model_id, strategy in self._models.items():
            try:
                signal = strategy.on_bar(bar)
                if signal:
                    model_record = self._model_records.get(model_id)
                    if model_record:
                        self.execution.submit_order(
                            model_id=model_id,
                            signal=signal,
                            current_capital=model_record.current_capital,
                            session_date=session_date,
                        )
            except Exception:
                logger.exception(f"Error processing bar for model {model_id}")

            # Update position prices
            self.execution.update_positions_price(model_id, bar.symbol, bar.close)

        # Update performance metrics every 10 bars
        if self._bar_count % 10 == 0:
            self.tracker.update_all()

    async def start_session(self) -> None:
        """Start a trading session."""
        init_db(self.config.db_path)

        self._session_date = datetime.now(ET).strftime("%Y-%m-%d")
        self._bar_count = 0
        self._running = True

        logger.info(f"Starting session for {self._session_date}")

        # Load models
        models = self._load_or_create_models()
        self._instantiate_strategies(models)

        # Initialize tracker
        self.tracker.initialize_models(models)

        # Reset daily limits
        self.execution.reset_daily_limits()

        # Record session
        db = get_session(self.config.db_path)
        try:
            session_record = SessionRecord(
                session_date=self._session_date,
                generation=models[0].generation if models else 1,
                started_at=datetime.utcnow(),
            )
            db.add(session_record)
            db.commit()
        finally:
            db.close()

        # Register bar callback
        self.feed.on_bar(self._fan_out_bar)

        # Start streaming
        logger.info(f"Streaming bars for {len(self.config.arena.symbols)} symbols")
        await self.feed.start_streaming(self.config.arena.symbols)

    async def end_session(self) -> dict:
        """End the trading session and generate summary."""
        self._running = False
        await self.feed.stop_streaming()

        # Final performance update
        self.tracker.update_all()
        self.tracker.save_snapshots(self._session_date)

        # Generate summary
        summary = self.tracker.generate_session_summary()

        # Update session record
        db = get_session(self.config.db_path)
        try:
            session_record = (
                db.query(SessionRecord)
                .filter(SessionRecord.session_date == self._session_date)
                .first()
            )
            if session_record:
                session_record.ended_at = datetime.utcnow()
                session_record.total_bars = self._bar_count
                session_record.total_trades = sum(
                    m.total_trades
                    for m in self.tracker.get_leaderboard()
                )
                session_record.summary = summary
                db.commit()
        finally:
            db.close()

        logger.info(f"Session ended. {self._bar_count} bars processed.")
        return summary

    async def run(self) -> None:
        """Run the arena: wait for market open, trade, close at market end.

        For development/testing, this can be run outside market hours
        and will start immediately.
        """
        init_db(self.config.db_path)

        logger.info("Arena starting...")

        if not self._is_market_hours():
            logger.warning(
                "Outside market hours. Starting in development mode "
                "(will stream available data)."
            )

        try:
            await self.start_session()
        except KeyboardInterrupt:
            logger.info("Arena interrupted by user")
        finally:
            summary = await self.end_session()
            self._print_summary(summary)

    def _print_summary(self, summary: dict) -> None:
        """Print session summary to console."""
        print("\n" + "=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)
        print(f"Models: {summary['model_count']}")
        print(f"Best: {summary.get('best_model', 'N/A')}")
        print(f"Worst: {summary.get('worst_model', 'N/A')}")
        print()
        print(f"{'Rank':<6}{'Model':<25}{'Return %':<12}{'Sharpe':<10}{'Trades':<8}{'Fitness':<10}")
        print("-" * 71)
        for r in summary.get("rankings", []):
            print(
                f"{r['rank']:<6}"
                f"{r['name']:<25}"
                f"{r['return_pct']:>8.4f}%   "
                f"{r['sharpe']:>8.4f}  "
                f"{r['trades']:<8}"
                f"{r['fitness']:<10.4f}"
            )
        print("=" * 60)
