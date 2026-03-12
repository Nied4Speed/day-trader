"""Arena orchestrator: manages the trading session lifecycle.

Daily flow:
  Pre-market warmup -> Session 1 -> Reflect -> Self-improve -> Session 2 ->
  Reflect -> Self-improve -> Save ledger

Self-improvement uses mutation memory to bias toward historically successful
parameter changes. Evolution (culling/spawning) runs weekly, not daily.
"""

import asyncio
import copy
import json
import logging
import os
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from src.core.cfa_review import apply_cfa_strategy, extract_parameter_recommendations, run_cfa_review
from src.core.config import Config
from src.core.database import (
    DailyLedger,
    ModelStatus,
    ModelSummary,
    SessionRecord,
    TradingModel,
    get_session,
    init_db,
)
from src.core.execution import ExecutionHandler
from src.core.llm_rules import generate_watch_rules
from src.core.mutation_memory import MutationMemory
from src.core.watch_rules import INDICATOR_CATALOG
from src.core.performance import PerformanceTracker
from src.core.position_manager import ModelSignal, PositionManager
from src.core.regime import RegimeDetector, MacroRegimeOverlay
from src.core.strategy import BarData, DailyContext, Strategy, TradeSignal, WatchSignal
from src.data.feed import AlpacaDataFeed, QuoteAggregator
from src.data.news_feed import AlpacaNewsFeed
from src.data.screener import SymbolScreener
from src.strategies.registry import create_strategy

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)


class Arena:
    """Manages the full trading day: two sessions with a break between."""

    def __init__(self, config: Config, simulate: bool = False):
        self.config = config
        self.simulate = simulate
        self.feed = AlpacaDataFeed(config)
        self.news_feed = AlpacaNewsFeed(config)
        self.execution = ExecutionHandler(config, simulate=simulate)
        self.tracker = PerformanceTracker(config.db_path)
        self._models: dict[int, Strategy] = {}  # model_id -> Strategy instance
        self._model_records: dict[int, TradingModel] = {}
        self._session_date: str = ""
        self._session_number: int = 1
        self._running = False
        self._bar_count = 0
        self._session_start_time: Optional[datetime] = None
        self._current_session_minutes: Optional[int] = None
        self._news_sentiment: dict[str, float] = {}  # symbol -> latest sentiment
        self._quote_subscribed: set[str] = set()  # symbols with active quote subs
        # Position Manager: coordinates signals, eliminates wash trades
        self.position_manager = PositionManager()
        # Regime Detector: detects trending/ranging/volatility per symbol
        self.regime_detector = RegimeDetector()
        self.macro_regime = MacroRegimeOverlay()
        # Adaptation tracking
        self._adapt_bar_count: int = 0
        self._adapt_interval: int = 15  # adapt every 15 bars
        self._recent_signals: dict[int, list] = {}  # model_id -> recent signals
        self._recent_fills: dict[int, list] = {}  # model_id -> recent fills
        # Watch list state: models watching symbols for quote-level entry
        self._watch_list: dict[int, dict[str, dict]] = {}  # model_id -> {symbol -> entry}
        self._watch_subscribed: set[str] = set()  # symbols subscribed for watch quotes
        self._last_watch_dispatch: dict[tuple, float] = {}  # (model_id, symbol) -> last dispatch time
        # Incident notes for CFA review — operational issues that affect data interpretation
        self._incident_notes: list[str] = []
        # Per-model symbol locks: (model_id, symbol) with pending orders
        self._model_symbol_locks: set[tuple[int, str]] = set()
        # Track stop-loss/take-profit fires to avoid re-triggering: set of (model_id, symbol)
        self._stop_loss_fired: set[tuple[int, str]] = set()
        # High-water mark per position for trailing stops: (model_id, symbol) -> highest price seen
        self._high_water_mark: dict[tuple[int, str], float] = {}
        # Underwater bar counter for patience stops: (model_id, symbol) -> consecutive bars below entry
        self._bars_underwater: dict[tuple[int, str], int] = {}
        # Stagnation bar counter for profit stagnation exits: (model_id, symbol) -> consecutive flat bars
        self._stagnation_bars: dict[tuple[int, str], int] = {}
        # Breakeven stop: once a position goes green by 0.3%, lock floor at entry.
        # If price drops back to entry, sell. "Never let green go red."
        self._breakeven_activated: set[tuple[int, str]] = set()
        self._watch_stats: dict[str, int] = {"created": 0, "expired": 0, "converted": 0}
        # Daily context per symbol: daily_open, running_high, running_low, VWAP accumulators, prev_close
        self._daily_context: dict[str, dict] = {}  # symbol -> {open, high, low, vwap_pv, vwap_vol, prev_close}
        # Quote aggregator for synthetic bars (created per session)
        self._quote_aggregator: Optional[QuoteAggregator] = None
        # Health check state
        self._last_health_check: dict = {}
        self._health_check_interval_minutes: int = 30

    def add_incident_note(self, note: str) -> None:
        """Add an operational incident note that will be included in the CFA review."""
        self._incident_notes.append(note)
        logger.info(f"Incident note added: {note}")

    def _set_status(self, phase: str, detail: str = "", bar: int | None = None, total_bars: int | None = None) -> None:
        """Write current arena phase to a status file for the dashboard."""
        status = {
            "phase": phase,
            "detail": detail,
            "session_number": self._session_number,
            "bar": bar or self._bar_count,
            "total_bars": total_bars,
            "timestamp": datetime.now(ET).isoformat(),
            "connection_state": getattr(self.feed, '_connection_state', 'unknown'),
        }
        try:
            os.makedirs("logs", exist_ok=True)
            with open("logs/arena_status.json", "w") as f:
                json.dump(status, f)
        except Exception:
            pass

    def _is_market_hours(self) -> bool:
        """Check if we're within US market hours."""
        now = datetime.now(ET).time()
        return MARKET_OPEN <= now <= MARKET_CLOSE

    async def _wait_until_et(self, hour: int, minute: int) -> None:
        """Sleep until the specified ET time. Returns immediately if already past."""
        now = datetime.now(ET)
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

        if now >= target:
            logger.info(
                f"Already past {hour:02d}:{minute:02d} ET "
                f"(now {now.strftime('%H:%M ET')}). Continuing."
            )
            return

        wait_seconds = (target - now).total_seconds()
        logger.info(
            f"Waiting until {hour:02d}:{minute:02d} ET "
            f"({wait_seconds / 60:.1f} minutes from now)..."
        )

        while True:
            now = datetime.now(ET)
            remaining = (target - now).total_seconds()
            if remaining <= 0:
                break
            await asyncio.sleep(min(30.0, remaining))

        logger.info(f"Reached {hour:02d}:{minute:02d} ET. Proceeding.")

    def _session_elapsed_minutes(self) -> float:
        """How many minutes have elapsed since current session started."""
        if self._session_start_time is None:
            return 0.0
        elapsed = datetime.now(ET) - self._session_start_time
        return elapsed.total_seconds() / 60.0

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
                for m in active_models:
                    _ = m.id, m.name, m.strategy_type, m.parameters, m.generation
                    _ = m.initial_capital, m.current_capital, m.status
                    db.expunge(m)
            else:
                # First run: create initial model pool
                logger.info("No active models found. Creating initial pool.")
                from src.strategies.registry import get_strategy_types

                strategy_types = [
                    st for st in get_strategy_types() if st != "collab"
                ]
                active_models = []

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
                    active_models.append(model)

                db.commit()

                for m in active_models:
                    db.refresh(m)

                for m in active_models:
                    strategy = create_strategy(m.strategy_type, m.name)
                    m.parameters = strategy.get_params()
                db.commit()

                logger.info(f"Created {len(active_models)} initial models")
                for m in active_models:
                    db.refresh(m)
                    _ = m.id, m.name, m.strategy_type, m.parameters, m.generation
                    _ = m.initial_capital, m.current_capital, m.status
                    db.expunge(m)

            # COLLAB auto-creation disabled — needs redesign as between-session
            # hybrid synthesis rather than real-time voting ensemble.
            # See TODO #2 in MEMORY.md.

            return active_models

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
            strategy.current_capital = model.current_capital
            strategy.initial_capital = model.initial_capital
            # Apply stop_loss/take_profit from DB params (strategies may not
            # read these in set_params, but they're tracked at the base class).
            params = model.parameters or {}
            if "stop_loss_pct" in params:
                strategy.stop_loss_pct = max(0.5, min(10.0, float(params["stop_loss_pct"])))
            if "take_profit_pct" in params:
                strategy.take_profit_pct = max(0.5, min(15.0, float(params["take_profit_pct"])))
            # Load ratcheting trailing stop tiers
            if "trailing_stop_tiers" in params and isinstance(params["trailing_stop_tiers"], list):
                try:
                    tiers = [(float(t[0]), float(t[1])) for t in params["trailing_stop_tiers"]]
                    tiers.sort(key=lambda t: t[0])
                    strategy.trailing_stop_tiers = tiers
                except (TypeError, IndexError, ValueError):
                    logger.warning(f"Invalid trailing_stop_tiers for {model.name}, ignoring")
            elif params.get("trailing_stop_pct") and params.get("trailing_stop_activation_pct"):
                # Migrate old fixed trailing stop to single-tier
                act = float(params["trailing_stop_activation_pct"])
                trail = float(params["trailing_stop_pct"])
                strategy.trailing_stop_tiers = [(act, trail)]
                logger.info(f"Migrated {model.name} trailing stop to single tier: [({act}, {trail})]")
            # Load patience stop tiers
            if "patience_stop_tiers" in params and isinstance(params["patience_stop_tiers"], list):
                try:
                    tiers = [(int(t[0]), float(t[1])) for t in params["patience_stop_tiers"]]
                    tiers.sort(key=lambda t: t[0])
                    strategy.patience_stop_tiers = tiers
                except (TypeError, IndexError, ValueError):
                    logger.warning(f"Invalid patience_stop_tiers for {model.name}, ignoring")
            self._models[model.id] = strategy
            self._model_records[model.id] = model

        # Register per-model capital with execution handler for risk checks
        self.execution._model_initial_capital = {
            mid: s.initial_capital for mid, s in self._models.items()
        }

        logger.info(f"Instantiated {len(self._models)} strategy instances")

    def _minutes_until_market_close(self) -> float:
        """Minutes remaining until 4:00 PM ET."""
        now = datetime.now(ET)
        close_dt = now.replace(hour=16, minute=0, second=0, microsecond=0)
        remaining = (close_dt - now).total_seconds() / 60.0
        return max(0.0, remaining)

    def _compute_session_minutes(self, session_number: int) -> int:
        """Compute how long a session should run.

        For S2, cap duration so we finish before market close with buffer.
        """
        configured = (
            self.config.arena.session_1_minutes
            if session_number == 1
            else self.config.arena.session_2_minutes
        )

        if not self.simulate:
            buffer = self.config.arena.market_close_buffer_minutes
            available = self._minutes_until_market_close() - buffer
            if available < configured:
                capped = max(10, int(available))  # at least 10 min
                logger.info(
                    f"S{session_number} capped to {capped} min "
                    f"(market closes in {available + buffer:.0f} min, "
                    f"{buffer} min buffer)"
                )
                return capped

        return configured

    async def _fan_out_bar(self, bar: BarData) -> None:
        """Distribute a completed bar to all active models.

        Flow:
        1. Update regime detection for this symbol
        2. Collect signals from all models (no direct order submission)
        3. Pass signals through PositionManager for conflict resolution
        4. Submit only resolved net orders
        5. Run intra-session adaptation every 15 bars
        """
        self._bar_count += 1
        self._adapt_bar_count += 1
        if self._bar_count % 5 == 0:
            self._set_status("session", f"Session {self._session_number} trading", bar=self._bar_count)
        session_date = self._session_date

        # Set time remaining so strategies know when to liquidate.
        # In live mode, wind-down is anchored to market close minus buffer
        # (= 3:58 PM ET with 2-min buffer) so liquidation ALWAYS fires at
        # a fixed time regardless of when the session started.
        if self._session_start_time:
            session_mins = self._current_session_minutes or self.config.arena.session_1_minutes
            session_remaining = max(
                0.0,
                session_mins - self._session_elapsed_minutes(),
            )
            if not self.simulate:
                buffer = self.config.arena.market_close_buffer_minutes
                market_remaining = max(
                    0.0,
                    self._minutes_until_market_close() - buffer,
                )
                bar.minutes_remaining = min(session_remaining, market_remaining)
            else:
                bar.minutes_remaining = session_remaining
        else:
            bar.minutes_remaining = None

        # Fetch news sentiment periodically
        if not self.simulate and self._bar_count % self.config.arena.news_fetch_interval == 0:
            try:
                self._news_sentiment = await asyncio.to_thread(
                    self.news_feed.fetch_news,
                    self.config.arena.symbols,
                    self.config.arena.news_lookback_minutes,
                )
            except Exception:
                logger.exception("Failed to fetch news sentiment")

        bar.news_sentiment = self._news_sentiment.get(bar.symbol)

        # Update regime detection for this symbol
        regime_state = self.regime_detector.update(bar)
        bar.regime = regime_state

        # Collect signals from all models (no direct submission)
        model_signals: list[ModelSignal] = []
        liquidation_signals: list[tuple[int, TradeSignal, float]] = []

        def _process_models():
            # Single batch DB read for all 20 models (replaces 20+ individual sessions)
            positioned_symbols = self._refresh_all_models_batch()
            self._cached_positioned_symbols = positioned_symbols

            for model_id, strategy in self._models.items():
                try:
                    signal = strategy.on_bar(bar)
                    if signal and signal.quantity > 0:
                        if signal.symbol == "__SKIP__":
                            continue
                        # Liquidation signals bypass the position manager
                        if (
                            bar.minutes_remaining is not None
                            and bar.minutes_remaining <= Strategy.LIQUIDATION_WINDOW
                            and signal.side == "sell"
                        ):
                            liquidation_signals.append(
                                (model_id, signal, strategy.current_capital)
                            )
                        else:
                            model_signals.append(
                                ModelSignal(
                                    model_id=model_id,
                                    signal=signal,
                                    capital=strategy.current_capital,
                                )
                            )
                            # Track for adaptation
                            self._recent_signals.setdefault(model_id, []).append(signal)
                except Exception:
                    logger.exception(f"Error processing bar for model {model_id}")

            # Batch-update position prices (1 DB session instead of 20)
            self._batch_update_position_prices(bar.symbol, bar.close)

            # Update portfolio state using already-fetched data (no extra DB call)
            self._update_position_manager_state_cached(positioned_symbols)

        await asyncio.to_thread(_process_models)

        # Independent model trading: submit each model's order directly
        if model_signals:
            logger.info(
                f"Bar {self._bar_count}: {len(model_signals)} signals -> "
                f"independent orders"
            )

        if self.simulate:
            for ms in model_signals:
                strategy = self._models[ms.model_id]
                self.execution.submit_order(
                    model_id=ms.model_id,
                    signal=ms.signal,
                    current_capital=strategy.current_capital,
                    session_date=session_date,
                    strategy_qty=strategy._positions.get(ms.signal.symbol, 0),
                )
                self._refresh_model_state(ms.model_id, strategy)
        else:
            # Live mode: submit all model orders + liquidation signals
            async_tasks = []
            task_model_ids: list[int] = []
            for ms in model_signals:
                lock_key = (ms.model_id, ms.signal.symbol)
                if lock_key in self._model_symbol_locks:
                    continue
                self._model_symbol_locks.add(lock_key)
                async_tasks.append(
                    self.execution.async_submit_order(
                        model_id=ms.model_id,
                        signal=ms.signal,
                        current_capital=self._models[ms.model_id].current_capital,
                        session_date=session_date,
                        strategy_qty=self._models[ms.model_id]._positions.get(ms.signal.symbol, 0),
                    )
                )
                task_model_ids.append(ms.model_id)

            for mid, sig, cap in liquidation_signals:
                async_tasks.append(
                    self.execution.async_submit_order(
                        model_id=mid, signal=sig,
                        current_capital=cap, session_date=session_date,
                        strategy_qty=self._models[mid]._positions.get(sig.symbol, 0) if mid in self._models else 0,
                    )
                )
                task_model_ids.append(mid)

            if async_tasks:
                await asyncio.gather(*async_tasks, return_exceptions=True)
                for ms in model_signals:
                    self._model_symbol_locks.discard((ms.model_id, ms.signal.symbol))
                all_model_ids = set(task_model_ids)
                await asyncio.to_thread(
                    lambda: [self._refresh_model_state(mid, self._models[mid]) for mid in all_model_ids if mid in self._models]
                )

        # Intra-session adaptation every 15 bars
        if self._adapt_bar_count >= self._adapt_interval:
            self._adapt_bar_count = 0
            await asyncio.to_thread(self._run_adaptation)

        # Collect watch signals from strategies (live mode only)
        if not self.simulate and not self._is_in_liquidation_window():
            for model_id, strategy in self._models.items():
                try:
                    watches = strategy.get_watch_signals(bar)
                    for w in watches:
                        self._register_watch(model_id, w)
                except Exception:
                    logger.exception(f"Error getting watch signals from model {model_id}")

        # Tick watch TTLs and expire stale watches
        self._tick_watches()

        # Update quote subscriptions based on position changes from this bar
        self._update_quote_subscriptions()

        # Update performance metrics and save snapshots periodically
        interval = self.config.arena.snapshot_interval
        if self._bar_count % interval == 0:
            last_prices = dict(self.execution._last_prices)
            session_date = self._session_date
            session_number = self._session_number

            def _snapshot():
                self.tracker.set_last_prices(last_prices)
                self.tracker.update_all()
                self.tracker.save_snapshots(session_date, session_number)

            await asyncio.to_thread(_snapshot)

    # ------------------------------------------------------------------
    # Batched bar processing (live mode) — reduces 10 thread calls to 1
    # ------------------------------------------------------------------

    async def _fan_out_bars(self, bars: list[BarData]) -> None:
        """Process a batch of bars (typically all bars from one minute).

        Same logic as _fan_out_bar but amortises the expensive parts:
        - One _refresh_all_models_batch() DB read instead of N
        - One asyncio.to_thread() call instead of N
        - One position_manager.resolve() call
        - One adaptation check
        - One snapshot save

        The event loop is free during the 3s accumulation window AND during
        the single thread call, so HTTP/WebSocket stay responsive.
        """
        if not bars:
            return

        is_synthetic = bars[0].synthetic
        n_bars = len(bars)
        session_date = self._session_date

        # Only advance counters / status for real bars
        if not is_synthetic:
            self._bar_count += n_bars
            self._adapt_bar_count += n_bars
            self._set_status("session", f"Session {self._session_number} trading", bar=self._bar_count)

        # Set time remaining on each bar (needed for check_liquidation even on synthetic).
        # In live mode, anchor to market close minus buffer (3:58 PM ET).
        if self._session_start_time:
            session_mins = self._current_session_minutes or self.config.arena.session_1_minutes
            elapsed = self._session_elapsed_minutes()
            session_remaining = max(0.0, session_mins - elapsed)
            if not self.simulate:
                buffer = self.config.arena.market_close_buffer_minutes
                market_remaining = max(
                    0.0,
                    self._minutes_until_market_close() - buffer,
                )
                remaining = min(session_remaining, market_remaining)
            else:
                remaining = session_remaining
            for bar in bars:
                bar.minutes_remaining = remaining
        else:
            for bar in bars:
                bar.minutes_remaining = None

        # News and regime only on real bars
        if not is_synthetic:
            # Fetch news sentiment once per batch (not per bar)
            logger.info(f"[BATCH-DBG] start batch {n_bars} bars, bar_count={self._bar_count}")
            if self._bar_count % self.config.arena.news_fetch_interval < n_bars:
                try:
                    logger.info("[BATCH-DBG] fetching news...")
                    self._news_sentiment = await asyncio.to_thread(
                        self.news_feed.fetch_news,
                        self.config.arena.symbols,
                        self.config.arena.news_lookback_minutes,
                    )
                    logger.info("[BATCH-DBG] news done")
                except Exception:
                    logger.exception("Failed to fetch news sentiment")
        else:
            logger.info(f"[SYNTH] {n_bars} synthetic bars")

        # Set news sentiment and regime on each bar
        for bar in bars:
            bar.news_sentiment = self._news_sentiment.get(bar.symbol)
            if is_synthetic:
                # Carry forward last known regime — don't update ADX/ATR with mid-price data
                bar.regime = self.regime_detector.get_regime(bar.symbol)
            else:
                bar.regime = self.regime_detector.update(bar)

        # Update running daily context and attach to each bar
        self._update_and_attach_daily_context(bars)

        # Collect signals from all bars × all models in one thread call
        model_signals: list[ModelSignal] = []
        liquidation_signals: list[tuple[int, TradeSignal, float]] = []
        # Track capital pre-deducted during signal collection so we can restore it
        # after resolve() — only resolved orders should actually deduct capital.
        pre_deducted: dict[int, float] = {}  # model_id -> total amount pre-deducted

        def _process_all():
            # One batch DB read for all models
            positioned_symbols = self._refresh_all_models_batch()
            self._cached_positioned_symbols = positioned_symbols

            # --- Macro regime overlay: update with SPY bars BEFORE signals ---
            for bar in bars:
                if bar.symbol == "SPY":
                    self.macro_regime.update(bar)
            self.macro_regime.tick_transition()
            macro = self.macro_regime.multipliers

            # --- Emergency override: portfolio daily P&L < -1% ---
            total_initial = len(self._models) * self.config.arena.initial_capital
            total_daily_pnl = sum(self.execution._daily_pnl.values())
            if total_initial > 0:
                daily_pnl_pct = total_daily_pnl / total_initial * 100.0
                self.macro_regime.set_emergency_bear(daily_pnl_pct < -1.0)
                if self.macro_regime._emergency_bear:
                    macro = self.macro_regime.multipliers  # re-fetch after override

            for bar in bars:
                for model_id, strategy in self._models.items():
                    try:
                        signal = strategy.on_bar(bar)
                        if signal and signal.quantity > 0:
                            if signal.symbol == "__SKIP__":
                                continue
                            # --- Macro buy gate: block new buys in STRONG_BEAR ---
                            if signal.side == "buy" and not macro.buy_gate_open:
                                continue
                            if (
                                bar.minutes_remaining is not None
                                and bar.minutes_remaining <= Strategy.LIQUIDATION_WINDOW
                                and signal.side == "sell"
                            ):
                                liquidation_signals.append(
                                    (model_id, signal, strategy.current_capital)
                                )
                            else:
                                # Scale buy quantity by macro allocation multiplier
                                if signal.side == "buy" and macro.allocation < 1.0:
                                    signal.quantity = round(signal.quantity * macro.allocation, 4)
                                    if signal.quantity < 0.01:
                                        continue  # too small after scaling
                                model_signals.append(
                                    ModelSignal(
                                        model_id=model_id,
                                        signal=signal,
                                        capital=strategy.current_capital,
                                    )
                                )
                                self._recent_signals.setdefault(model_id, []).append(signal)
                                # Temporarily pre-deduct capital so next signal sees
                                # reduced balance (prevents over-commitment within a
                                # single batch).  We restore ALL pre-deducted amounts
                                # after resolve() and only permanently deduct for
                                # orders that actually get submitted.
                                if signal.side == "buy":
                                    est_price = self.execution._last_prices.get(signal.symbol, bar.close)
                                    deduct = signal.quantity * est_price
                                    strategy.current_capital -= deduct
                                    pre_deducted[model_id] = pre_deducted.get(model_id, 0.0) + deduct
                    except Exception:
                        logger.exception(f"Error processing bar for model {model_id}")

            # Batch-update position prices for each unique symbol
            seen_symbols: set[str] = set()
            for bar in bars:
                if bar.symbol not in seen_symbols:
                    seen_symbols.add(bar.symbol)
                    self._batch_update_position_prices(bar.symbol, bar.close)

            # Check stop-loss / take-profit for all models holding positions
            # Uses latest bar close prices as the check price.
            latest_prices: dict[str, float] = {}
            for bar in bars:
                latest_prices[bar.symbol] = bar.close
            for model_id, strategy in self._models.items():
                for symbol, qty in list(strategy._positions.items()):
                    if qty < 0.001 or symbol not in latest_prices:
                        continue
                    # Skip if we already fired a stop/TP for this position
                    sl_key = (model_id, symbol)
                    if sl_key in self._stop_loss_fired:
                        continue
                    price = latest_prices[symbol]
                    entry_price = strategy._entry_prices.get(symbol)
                    if not entry_price or entry_price <= 0:
                        continue
                    pct_change = (price - entry_price) / entry_price * 100.0
                    # Leverage-scaled exits: tighten thresholds proportionally
                    # to position notional vs model budget. A 2x leveraged
                    # position gets stops that are half as wide in % terms,
                    # keeping dollar-risk constant.
                    position_notional = qty * entry_price
                    leverage = max(1.0, position_notional / strategy.initial_capital)
                    triggered = None
                    # Apply macro regime multiplier to stop loss
                    base_stop = strategy.stop_loss_pct if strategy.stop_loss_pct else None
                    eff_stop = (base_stop * macro.stop_loss) / leverage if base_stop else None
                    eff_tp = strategy.take_profit_pct / leverage if strategy.take_profit_pct else None
                    if eff_stop and pct_change <= -eff_stop:
                        triggered = "stop_loss"
                    elif eff_tp and pct_change >= eff_tp:
                        # Instead of hard sell, activate trailing stop tiers
                        if not strategy.trailing_stop_tiers:
                            strategy.trailing_stop_tiers = [(3.0, 1.5), (6.0, 1.0), (10.0, 0.75)]
                            logger.info(
                                f"[TP->TRAIL] model {model_id} {symbol}: "
                                f"gain={pct_change:+.2f}% hit TP, activating trailing stop"
                            )
                        # Don't trigger sell — trailing stop check below manages exit

                    # Ratcheting trailing stop: runs independently of stop_loss/TP
                    # so it fires even after TP activates trailing tiers.
                    if not triggered and strategy.trailing_stop_tiers:
                        hwm_key = (model_id, symbol)
                        hwm = self._high_water_mark.get(hwm_key, entry_price)
                        if price > hwm:
                            self._high_water_mark[hwm_key] = price
                            hwm = price
                        gain_from_entry = (hwm - entry_price) / entry_price * 100.0
                        # Scale gain threshold DOWN by leverage so trailing
                        # activates sooner on leveraged positions
                        active_trail = strategy.get_active_trail_pct(gain_from_entry * leverage)
                        if active_trail is not None:
                            drop_from_peak = (hwm - price) / hwm * 100.0
                            # Apply macro regime multiplier to trail distance
                            adjusted_trail = active_trail * macro.trailing_trail
                            # Scale trail distance DOWN by leverage
                            if drop_from_peak >= adjusted_trail / leverage:
                                triggered = "trailing_stop"

                    # Breakeven stop: "never let green go red."
                    # Thresholds adjusted by macro regime.
                    BREAKEVEN_ACTIVATE_PCT = macro.breakeven_activate_pct
                    BREAKEVEN_FLOOR_PCT = macro.breakeven_floor_pct
                    if not triggered:
                        be_key = (model_id, symbol)
                        if be_key in self._breakeven_activated:
                            if pct_change <= BREAKEVEN_FLOOR_PCT:
                                triggered = "breakeven_stop"
                        elif pct_change >= BREAKEVEN_ACTIVATE_PCT:
                            self._breakeven_activated.add(be_key)
                            logger.info(
                                f"[BREAKEVEN] model {model_id} {symbol}: "
                                f"activated at {pct_change:+.2f}% — floor locked at +{BREAKEVEN_FLOOR_PCT}%"
                            )

                    # Profit stagnation exit: position is profitable but going nowhere.
                    # Activate when +0.5% to <TP and price is flat for 15+ bars.
                    if not triggered and pct_change >= 0.5 and (not eff_tp or pct_change < eff_tp):
                        stag_key = (model_id, symbol)
                        # Check if price is flat: (max-min)/entry < 0.2% over last 10 bars
                        sym_history = strategy.get_history(symbol, lookback=10)
                        if len(sym_history) >= 5:
                            recent_highs = [b.high for b in sym_history]
                            recent_lows = [b.low for b in sym_history]
                            price_range = (max(recent_highs) - min(recent_lows)) / entry_price
                            recent_volumes = [b.volume for b in sym_history]
                            avg_vol_all = sum(b.volume for b in strategy.get_history(symbol, lookback=50)) / max(1, len(strategy.get_history(symbol, lookback=50)))
                            vol_ratio = (sum(recent_volumes) / len(recent_volumes)) / avg_vol_all if avg_vol_all > 0 else 1.0
                            # Tighten threshold after 25 bars stagnant
                            bars_stag = self._stagnation_bars.get(stag_key, 0)
                            flat_threshold = 0.001 if bars_stag >= 25 else 0.002
                            is_flat = price_range < flat_threshold and vol_ratio < 0.8
                            if is_flat:
                                self._stagnation_bars[stag_key] = bars_stag + 1
                                if self._stagnation_bars[stag_key] >= 15:
                                    triggered = "profit_stagnation"
                            else:
                                # Reset if price makes new move or drops below +0.5%
                                self._stagnation_bars.pop(stag_key, None)
                        # Also reset if hwm changed significantly (new high)
                        hwm_key = (model_id, symbol)
                        hwm = self._high_water_mark.get(hwm_key, entry_price)
                        if price > hwm:
                            self._stagnation_bars.pop(stag_key, None)
                    elif pct_change < 0.5:
                        # Below +0.5% — not in stagnation zone
                        self._stagnation_bars.pop((model_id, symbol), None)

                    # Patience stop: track consecutive bars underwater, tighten
                    # exit as patience runs out. Resets when position goes green.
                    # Macro regime scales patience: lower multiplier = less patience = exits sooner.
                    if not triggered and strategy.patience_stop_tiers:
                        uw_key = (model_id, symbol)
                        if pct_change < 0:
                            self._bars_underwater[uw_key] = self._bars_underwater.get(uw_key, 0) + 1
                            bars_uw = self._bars_underwater[uw_key]
                            # Scale bars_underwater UP when macro.patience_bars < 1 so
                            # patience runs out faster in bearish regimes.
                            adjusted_bars_uw = int(bars_uw / macro.patience_bars) if macro.patience_bars > 0 else bars_uw
                            patience_exit = strategy.get_patience_exit_pct(adjusted_bars_uw)
                            # Scale patience exit threshold by leverage (less negative = tighter)
                            if patience_exit is not None and pct_change <= patience_exit / leverage:
                                triggered = "patience_stop"
                        else:
                            # Position is green — reset counter
                            self._bars_underwater.pop(uw_key, None)
                    if triggered:
                        signal = TradeSignal(symbol=symbol, side="sell", quantity=qty, reason=triggered)
                        liquidation_signals.append(
                            (model_id, signal, strategy.current_capital)
                        )
                        self._stop_loss_fired.add(sl_key)
                        # Build log extras before cleanup
                        extra = ""
                        if triggered == "trailing_stop":
                            hwm_val = self._high_water_mark.get((model_id, symbol), price)
                            trail_val = strategy.get_active_trail_pct(
                                (hwm_val - entry_price) / entry_price * 100.0
                            )
                            extra = f" HWM={hwm_val:.2f} trail={trail_val}%"
                        elif triggered == "patience_stop":
                            bars_uw = self._bars_underwater.get((model_id, symbol), 0)
                            patience_exit = strategy.get_patience_exit_pct(bars_uw)
                            extra = f" bars_underwater={bars_uw} patience_exit={patience_exit}%"
                        elif triggered == "profit_stagnation":
                            bars_stag = self._stagnation_bars.get((model_id, symbol), 0)
                            extra = f" stagnation_bars={bars_stag}"
                        # Clean up tracking state on exit
                        self._high_water_mark.pop((model_id, symbol), None)
                        self._bars_underwater.pop((model_id, symbol), None)
                        self._stagnation_bars.pop((model_id, symbol), None)
                        self._breakeven_activated.discard((model_id, symbol))
                        logger.info(
                            f"[{triggered.upper()}] model {model_id} {symbol}: "
                            f"entry={entry_price:.2f} now={price:.2f} ({pct_change:+.2f}%)"
                            f"{extra} -> selling {qty}"
                        )

            # NOTE: position manager state update moved to AFTER capital
            # restoration so it sees true capital, not pre-deducted amounts.

        logger.info("[BATCH-DBG] starting _process_all thread...")
        await asyncio.to_thread(_process_all)
        logger.info(f"[BATCH-DBG] _process_all done: {len(model_signals)} signals")

        # Restore ALL temporarily pre-deducted capital.  Only resolved orders
        # that actually get submitted will have capital permanently deducted below.
        for model_id, amount in pre_deducted.items():
            self._models[model_id].current_capital += amount

        # Update position manager portfolio state for drawdown circuit breaker
        self._update_position_manager_state_cached(
            self._cached_positioned_symbols
        )

        # Independent model trading: submit each model's order directly
        # Group signals by symbol to detect and log disagreements
        by_symbol: dict[str, list[ModelSignal]] = defaultdict(list)
        for ms in model_signals:
            by_symbol[ms.signal.symbol].append(ms)

        # Build ordered task list: sells first per symbol, then buys
        async_tasks = []
        task_model_ids: list[int] = []  # track model_id per task for refresh

        for symbol, sym_signals in by_symbol.items():
            buys = [ms for ms in sym_signals if ms.signal.side == "buy"]
            sells = [ms for ms in sym_signals if ms.signal.side == "sell"]

            # Log disagreements
            if buys and sells:
                logger.info(
                    f"[SIGNAL_CONFLICT] {symbol}: {len(buys)} buys vs {len(sells)} sells"
                )

            # Submit sells first (exits are priority), then buys
            for ms in sells + buys:
                lock_key = (ms.model_id, ms.signal.symbol)
                if lock_key in self._model_symbol_locks:
                    continue
                self._model_symbol_locks.add(lock_key)
                # Pre-deduct capital for buys so concurrent orders see reduced balance
                if ms.signal.side == "buy":
                    est_price = self.execution._last_prices.get(ms.signal.symbol, 100)
                    self._models[ms.model_id].current_capital -= ms.signal.quantity * est_price
                async_tasks.append(
                    self.execution.async_submit_order(
                        model_id=ms.model_id,
                        signal=ms.signal,
                        current_capital=self._models[ms.model_id].current_capital,
                        session_date=session_date,
                        strategy_qty=self._models[ms.model_id]._positions.get(ms.signal.symbol, 0),
                    )
                )
                task_model_ids.append(ms.model_id)

        if model_signals:
            logger.info(
                f"Batch ({n_bars} bars): {len(model_signals)} signals -> "
                f"{len(async_tasks)} independent orders"
            )

        # Liquidation signals (stop-loss, trailing stop, patience stop)
        n_regular = len(async_tasks)
        for mid, sig, cap in liquidation_signals:
            async_tasks.append(
                self.execution.async_submit_order(
                    model_id=mid, signal=sig,
                    current_capital=cap, session_date=session_date,
                    strategy_qty=self._models[mid]._positions.get(sig.symbol, 0) if mid in self._models else 0,
                )
            )
            task_model_ids.append(mid)

        if async_tasks:
            logger.info(f"[BATCH-DBG] submitting {len(async_tasks)} orders via gather...")
            results = await asyncio.gather(*async_tasks, return_exceptions=True)
            for i, r in enumerate(results):
                if isinstance(r, Exception):
                    logger.error(f"[BATCH-DBG] order task {i} failed: {r!r}")
            # Clear _stop_loss_fired for liquidation signals that failed
            for j, (mid, sig, _cap) in enumerate(liquidation_signals):
                result_idx = n_regular + j
                if result_idx < len(results):
                    r = results[result_idx]
                    failed = isinstance(r, Exception) or r is None or (
                        hasattr(r, 'status') and getattr(r.status, 'value', None) == 'rejected'
                    )
                    if failed:
                        sl_key = (mid, sig.symbol)
                        self._stop_loss_fired.discard(sl_key)
                        logger.warning(
                            f"[SL/TP RETRY] Cleared stop_loss_fired for model {mid} "
                            f"{sig.symbol} — will re-trigger next bar"
                        )
            # Unlock per-model symbol locks
            logger.info(f"[BATCH-DBG] gather done, unlocking model-symbol locks...")
            for ms in model_signals:
                self._model_symbol_locks.discard((ms.model_id, ms.signal.symbol))
            all_model_ids = set(task_model_ids)
            logger.info(f"[BATCH-DBG] refreshing {len(all_model_ids)} model states...")
            await asyncio.to_thread(
                lambda: [self._refresh_model_state(mid, self._models[mid]) for mid in all_model_ids if mid in self._models]
            )
            logger.info("[BATCH-DBG] refresh done")
        else:
            logger.info("[BATCH-DBG] no orders to submit")

        # Intra-session adaptation (real bars only)
        if not is_synthetic and self._adapt_bar_count >= self._adapt_interval:
            self._adapt_bar_count = 0
            logger.info("[BATCH-DBG] running adaptation...")
            await asyncio.to_thread(self._run_adaptation)
            logger.info("[BATCH-DBG] adaptation done")

        # Watch signals from all bars (real bars only)
        if not is_synthetic and not self._is_in_liquidation_window():
            for bar in bars:
                for model_id, strategy in self._models.items():
                    try:
                        watches = strategy.get_watch_signals(bar)
                        for w in watches:
                            self._register_watch(model_id, w)
                    except Exception:
                        logger.exception(f"Error getting watch signals from model {model_id}")

        if not is_synthetic:
            self._tick_watches()
        # NOTE: _update_quote_subscriptions() intentionally skipped in batch mode.
        # subscribe_quotes() on the Alpaca StockDataStream deadlocks the event loop
        # when called while _run_forever() is processing messages. Stop-loss/take-profit
        # logic in on_bar() handles exits without real-time quotes.

        # Snapshot once per batch (real bars only)
        if not is_synthetic:
            interval = self.config.arena.snapshot_interval
            if self._bar_count % interval < n_bars:
                last_prices = dict(self.execution._last_prices)
                session_date = self._session_date
                session_number = self._session_number

                def _snapshot():
                    self.tracker.set_last_prices(last_prices)
                    self.tracker.update_all()
                    self.tracker.save_snapshots(session_date, session_number)
                    self._save_daily_ledger()  # keep history page live

                logger.info("[BATCH-DBG] saving snapshot...")
                await asyncio.to_thread(_snapshot)
                logger.info("[BATCH-DBG] snapshot done")

        logger.info(f"[BATCH-DBG] batch complete, bar_count={self._bar_count}" + (" [synthetic]" if is_synthetic else ""))

    def _update_position_manager_state(self) -> None:
        """Sync portfolio state into the PositionManager for risk checks."""
        from src.core.database import Position
        db = get_session(self.config.db_path)
        try:
            total_capital = sum(
                m.current_capital for m in self._model_records.values()
            )
            positions = db.query(Position).filter(Position.quantity > 0).all()
            symbol_exposure: dict[str, float] = {}
            for p in positions:
                price = self.execution._last_prices.get(p.symbol, p.current_price)
                symbol_exposure[p.symbol] = (
                    symbol_exposure.get(p.symbol, 0.0) + p.quantity * price
                )
            total_value = total_capital + sum(symbol_exposure.values())
            self.position_manager.update_portfolio_state(total_value, symbol_exposure)
        finally:
            db.close()

    def _update_position_manager_state_cached(self, positioned_symbols: set[str]) -> None:
        """Update portfolio state using in-memory data (no DB call).

        Uses strategy._positions from the batch refresh instead of querying DB again.
        """
        total_capital = sum(
            s.current_capital for s in self._models.values()
        )
        symbol_exposure: dict[str, float] = {}
        for strategy in self._models.values():
            for symbol, qty in strategy._positions.items():
                if qty > 0:
                    price = self.execution._last_prices.get(symbol, 0.0)
                    symbol_exposure[symbol] = (
                        symbol_exposure.get(symbol, 0.0) + qty * price
                    )
        total_value = total_capital + sum(symbol_exposure.values())
        self.position_manager.update_portfolio_state(total_value, symbol_exposure)

    def _run_adaptation(self) -> None:
        """Run intra-session adaptation for all models."""
        for model_id, strategy in self._models.items():
            signals = self._recent_signals.get(model_id, [])
            fills = self._recent_fills.get(model_id, [])

            # Compute realized P&L from recent fills
            realized_pnl = sum(f.get("pnl", 0.0) for f in fills)

            try:
                strategy.adapt(signals, fills, realized_pnl)
            except Exception:
                logger.exception(f"Adaptation failed for model {model_id}")

        # Clear recent tracking for next window
        self._recent_signals.clear()
        self._recent_fills.clear()
        logger.debug(f"Adaptation cycle complete (bar {self._bar_count})")

    def _refresh_model_state(self, model_id: int, strategy) -> None:
        """Refresh strategy capital, positions, and entry prices from DB."""
        db = get_session(self.config.db_path)
        try:
            from src.core.database import Position
            model = db.query(TradingModel).get(model_id)
            if model:
                self._model_records[model_id] = model
                strategy.current_capital = model.current_capital
                strategy.initial_capital = model.initial_capital
            positions = db.query(Position).filter(
                Position.model_id == model_id, Position.quantity > 0
            ).all()
            strategy._positions = {p.symbol: p.quantity for p in positions if p.quantity >= 0.001}
            strategy._entry_prices = {p.symbol: p.avg_entry_price for p in positions if p.quantity >= 0.001 and p.avg_entry_price > 0}
        finally:
            db.close()

    def _batch_update_position_prices(self, symbol: str, price: float) -> None:
        """Update current price + unrealized P&L for all positions of a symbol in one DB session."""
        from src.core.database import Position
        db = get_session(self.config.db_path)
        try:
            positions = db.query(Position).filter(
                Position.symbol == symbol, Position.quantity > 0
            ).all()
            for p in positions:
                p.current_price = price
                p.unrealized_pnl = (price - p.avg_entry_price) * p.quantity
                p.updated_at = datetime.utcnow()
            if positions:
                db.commit()
        finally:
            db.close()
        # Also update execution's last prices cache
        self.execution._last_prices[symbol] = price

    def _refresh_all_models_batch(self) -> set[str]:
        """Refresh ALL model states in a single DB session. Returns positioned symbols.

        Replaces 20+ individual _refresh_model_state calls with 2 queries total.
        Also returns the set of symbols with open positions (saves another query).
        """
        from src.core.database import Position
        db = get_session(self.config.db_path)
        try:
            # Query all active models at once
            model_ids = list(self._models.keys())
            models = db.query(TradingModel).filter(
                TradingModel.id.in_(model_ids)
            ).all()
            for model in models:
                if model.id in self._models:
                    self._model_records[model.id] = model
                    self._models[model.id].current_capital = model.current_capital

            # Query all open positions at once
            positions = db.query(Position).filter(
                Position.model_id.in_(model_ids),
                Position.quantity > 0,
            ).all()

            # Group by model_id
            pos_by_model: dict[int, list] = {mid: [] for mid in model_ids}
            positioned_symbols: set[str] = set()
            for p in positions:
                if p.model_id in pos_by_model:
                    pos_by_model[p.model_id].append(p)
                positioned_symbols.add(p.symbol)

            # Update all strategies
            for model_id, strategy in self._models.items():
                model_positions = pos_by_model.get(model_id, [])
                strategy._positions = {p.symbol: p.quantity for p in model_positions if p.quantity >= 0.001}
                strategy._entry_prices = {
                    p.symbol: p.avg_entry_price
                    for p in model_positions if p.quantity >= 0.001 and p.avg_entry_price > 0
                }

            return positioned_symbols
        finally:
            db.close()

    def _get_positioned_symbols(self) -> set[str]:
        """Return set of symbols that have open positions across any model."""
        # Use cached value if available (set during batch refresh)
        if hasattr(self, '_cached_positioned_symbols'):
            return self._cached_positioned_symbols
        from src.core.database import Position
        db = get_session(self.config.db_path)
        try:
            rows = (
                db.query(Position.symbol)
                .filter(Position.quantity > 0)
                .distinct()
                .all()
            )
            return {r[0] for r in rows}
        finally:
            db.close()

    def _update_quote_subscriptions(self) -> None:
        """Subscribe/unsubscribe quotes based on positions AND watches.

        Called after order fills to keep quote feed aligned with positions.
        Only active in live mode (not simulate).
        """
        if self.simulate:
            return

        positioned = self._get_positioned_symbols()
        watched = self._get_watched_symbols()
        needed = positioned | watched
        to_subscribe = needed - self._quote_subscribed
        to_unsubscribe = self._quote_subscribed - needed

        if to_subscribe:
            self.feed.subscribe_quotes(list(to_subscribe))
            self._quote_subscribed |= to_subscribe

        if to_unsubscribe:
            self.feed.unsubscribe_quotes(list(to_unsubscribe))
            self._quote_subscribed -= to_unsubscribe

    # --- Watch list management ---

    def _get_watched_symbols(self) -> set[str]:
        """Return set of all symbols being watched by any model."""
        symbols = set()
        for watches in self._watch_list.values():
            symbols.update(watches.keys())
        return symbols

    def _register_watch(self, model_id: int, watch: WatchSignal) -> None:
        """Add or renew a watch for a model+symbol.

        Skips if model already holds a position in the symbol.
        Enforces max_watches_per_model (renewals don't count toward cap).
        """
        strategy = self._models.get(model_id)
        if not strategy:
            return

        # Don't watch symbols we already hold
        if strategy._positions.get(watch.symbol, 0) > 0:
            return

        if model_id not in self._watch_list:
            self._watch_list[model_id] = {}

        model_watches = self._watch_list[model_id]
        is_renewal = watch.symbol in model_watches

        # Enforce cap (renewals don't count)
        max_watches = self.config.arena.max_watches_per_model
        if not is_renewal and len(model_watches) >= max_watches:
            return

        model_watches[watch.symbol] = {
            "reason": watch.reason,
            "ttl": watch.ttl_bars,
            "context": watch.context,
        }

        if not is_renewal:
            self._watch_stats["created"] += 1
            logger.debug(
                f"Watch created: model {model_id} watching {watch.symbol} "
                f"({watch.reason}, ttl={watch.ttl_bars})"
            )

    def _tick_watches(self) -> None:
        """Decrement TTLs and remove expired watches. Called once per bar."""
        expired_keys: list[tuple[int, str]] = []

        for model_id, watches in self._watch_list.items():
            for symbol, entry in watches.items():
                entry["ttl"] -= 1
                if entry["ttl"] <= 0:
                    expired_keys.append((model_id, symbol))

        for model_id, symbol in expired_keys:
            del self._watch_list[model_id][symbol]
            self._watch_stats["expired"] += 1
            logger.debug(f"Watch expired: model {model_id} {symbol}")

        # Clean up empty model entries
        self._watch_list = {
            mid: w for mid, w in self._watch_list.items() if w
        }

    def _is_in_liquidation_window(self) -> bool:
        """Check if we're in the session's no-buy window (no new entries)."""
        if self._session_start_time and self._current_session_minutes:
            remaining = self._current_session_minutes - self._session_elapsed_minutes()
            if remaining <= Strategy.NO_BUY_WINDOW:
                return True
        if not self.simulate:
            if self._minutes_until_market_close() <= Strategy.NO_BUY_WINDOW:
                return True
        return False

    def _remove_watch(self, model_id: int, symbol: str) -> None:
        """Remove a specific watch (after conversion to position)."""
        if model_id in self._watch_list:
            self._watch_list[model_id].pop(symbol, None)
            if not self._watch_list[model_id]:
                del self._watch_list[model_id]

    def _clear_watches(self) -> None:
        """Clear all watches and stats (session teardown)."""
        if self._watch_stats["created"] > 0:
            logger.info(
                f"Watch stats: {self._watch_stats['created']} created, "
                f"{self._watch_stats['converted']} converted, "
                f"{self._watch_stats['expired']} expired"
            )
        self._watch_list.clear()
        self._watch_subscribed.clear()
        self._last_watch_dispatch.clear()
        self._watch_stats = {"created": 0, "expired": 0, "converted": 0}

    async def _on_quote(self, symbol: str, bid: float, ask: float, timestamp) -> None:
        """Handle an incoming quote for both exit monitoring and watch entries.

        1. Position exits: check all models holding this symbol for stop/take-profit
        2. Watch entries: check models watching this symbol for entry signals
        """
        import time as _time

        mid = (bid + ask) / 2.0
        self.execution._last_prices[symbol] = mid
        session_date = self._session_date

        # --- Position exits (existing logic) ---
        for model_id, strategy in self._models.items():
            qty = strategy._positions.get(symbol, 0)
            if qty <= 0:
                continue

            try:
                signal = strategy.on_quote(symbol, bid, ask, timestamp)
                if signal and signal.side == "sell" and signal.quantity > 0:
                    self._refresh_model_state(model_id, strategy)

                    # Find and cancel any bracket legs for this position
                    for alpaca_id, pending in list(self.execution._pending_orders.items()):
                        if pending["model_id"] == model_id and pending["signal"].symbol == symbol:
                            self.execution.cancel_bracket_legs(alpaca_id)

                    await self.execution.async_submit_order(
                        model_id=model_id,
                        signal=signal,
                        current_capital=strategy.current_capital,
                        session_date=session_date,
                        strategy_qty=qty,
                    )
                    self._refresh_model_state(model_id, strategy)

                    entry_price = strategy._entry_prices.get(symbol, 0)
                    pct = ((mid - entry_price) / entry_price * 100) if entry_price else 0
                    logger.info(
                        f"Quote exit: model {model_id} sold {signal.quantity} "
                        f"{symbol} @ mid {mid:.2f} (entry={entry_price:.2f}, {pct:+.2f}%)"
                    )
            except Exception:
                logger.exception(
                    f"Error processing quote for model {model_id} {symbol}"
                )

        # --- Watch entries ---
        if not self._is_in_liquidation_window():
            throttle_ms = self.config.arena.watch_quote_throttle_ms
            now_ms = _time.monotonic() * 1000

            for model_id, watches in list(self._watch_list.items()):
                if symbol not in watches:
                    continue

                # Throttle: skip if dispatched too recently
                dispatch_key = (model_id, symbol)
                last = self._last_watch_dispatch.get(dispatch_key, 0)
                if (now_ms - last) < throttle_ms:
                    continue
                self._last_watch_dispatch[dispatch_key] = now_ms

                strategy = self._models.get(model_id)
                if not strategy:
                    continue

                # Skip if model acquired a position since the watch was set
                if strategy._positions.get(symbol, 0) > 0:
                    self._remove_watch(model_id, symbol)
                    continue

                try:
                    entry = watches[symbol]
                    signal = strategy.on_watch_quote(
                        symbol, bid, ask, timestamp, entry["context"]
                    )
                    if signal and signal.side == "buy" and signal.quantity > 0:
                        self._refresh_model_state(model_id, strategy)
                        await self.execution.async_submit_order(
                            model_id=model_id,
                            signal=signal,
                            current_capital=strategy.current_capital,
                            session_date=session_date,
                        )
                        self._refresh_model_state(model_id, strategy)
                        self._remove_watch(model_id, symbol)
                        self._watch_stats["converted"] += 1
                        logger.info(
                            f"Watch converted: model {model_id} bought {signal.quantity} "
                            f"{symbol} @ mid {mid:.2f} (reason={entry['reason']})"
                        )
                except Exception:
                    logger.exception(
                        f"Error processing watch quote for model {model_id} {symbol}"
                    )

        # Update position prices for all models holding this symbol
        for model_id, strategy in self._models.items():
            if strategy._positions.get(symbol, 0) > 0:
                self.execution.update_positions_price(model_id, symbol, mid)

        # Check if we should unsubscribe (all positions/watches closed)
        self._update_quote_subscriptions()

    async def _on_trade_update(self, event: str, order_data: dict) -> None:
        """Handle trade update from TradingStream — non-blocking.

        Runs execution.handle_trade_update (sync DB work) in a thread to avoid
        blocking the event loop, then does lightweight arena bookkeeping.
        """
        # DB-heavy work runs off the event loop
        result = await asyncio.to_thread(
            self.execution.handle_trade_update, event, order_data
        )

        if not result:
            return

        if result["type"] == "fill":
            model_id = result["model_id"]
            symbol = result["symbol"]
            side = result["side"]
            price = result["price"]
            qty = result["qty"]

            if model_id and model_id in self._models:
                # Refresh model state from DB (in thread to stay non-blocking)
                await asyncio.to_thread(
                    self._refresh_model_state, model_id, self._models[model_id]
                )

                # Clear stop-loss fired flag on buy (new position can trigger fresh)
                # or on sell (position gone, flag no longer needed)
                self._stop_loss_fired.discard((model_id, symbol))
                # Reset tracking state on sell (position closed) or buy (new entry)
                if side == "sell":
                    self._high_water_mark.pop((model_id, symbol), None)
                    self._bars_underwater.pop((model_id, symbol), None)
                    self._stagnation_bars.pop((model_id, symbol), None)
                    self._breakeven_activated.discard((model_id, symbol))
                elif side == "buy":
                    self._high_water_mark[(model_id, symbol)] = price
                    self._bars_underwater.pop((model_id, symbol), None)
                    self._stagnation_bars.pop((model_id, symbol), None)

                # Track fill for intra-session adaptation
                entry_price = self._models[model_id]._entry_prices.get(symbol, price)
                pnl = (price - entry_price) * qty if side == "sell" else 0.0
                fill_record = {
                    "symbol": symbol, "side": side, "price": price,
                    "qty": qty, "pnl": pnl,
                }
                self._recent_fills.setdefault(model_id, []).append(fill_record)

            # Unlock per-model symbol lock
            self._model_symbol_locks.discard((model_id, symbol))

        elif result["type"] == "reject":
            model_id = result["model_id"]
            if model_id in self._models:
                await asyncio.to_thread(
                    self._refresh_model_state, model_id, self._models[model_id]
                )

    def _end_session_liquidate_sync(self) -> None:
        """Force-close remaining positions (sync version for backtest/offline)."""
        import time as _time
        logger.info("EOD liquidation: closing remaining positions")
        closed = self.execution.liquidate_all(self._session_date)
        if closed and not self.simulate:
            _time.sleep(2)
        alpaca_closed = self.execution.liquidate_all_alpaca(self._session_date)
        if alpaca_closed:
            logger.info(f"Alpaca safety net closed {alpaca_closed} extra positions")

        # Verify-and-retry loop (same logic as async version)
        if not self.simulate:
            for attempt in range(1, 5):
                _time.sleep(2)
                try:
                    positions = self.execution.client.get_all_positions()
                    real = [p for p in positions if abs(float(p.qty)) >= 0.001]
                    if not real:
                        logger.info("EOD liquidation verified: Alpaca is flat")
                        break
                    logger.warning(
                        f"EOD verify {attempt}/4: {len(real)} positions still open"
                    )
                    self.execution.client.close_all_positions(cancel_orders=True)
                    _time.sleep(2)
                    for p in self.execution.client.get_all_positions():
                        if abs(float(p.qty)) >= 0.001:
                            try:
                                self.execution.client.close_position(p.symbol)
                            except Exception:
                                pass
                except Exception:
                    logger.warning(f"EOD verify attempt {attempt} failed", exc_info=True)

        if closed or alpaca_closed:
            self.tracker.update_all()

    async def _end_session_liquidate(self) -> None:
        """Force-close remaining positions without blocking the event loop.

        Uses a verify-and-retry loop to ensure Alpaca is fully flat.
        Leftover positions from failed EOD liquidation is the #1 bug source.
        """
        logger.info("EOD liquidation: closing remaining positions")

        # Phase 1: close via our tracked order flow (gets realized_pnl)
        closed = await asyncio.to_thread(
            self.execution.liquidate_all, self._session_date
        )
        if closed and not self.simulate:
            await asyncio.sleep(2)

        # Phase 2: Alpaca safety net for anything our DB missed
        alpaca_closed = await asyncio.to_thread(
            self.execution.liquidate_all_alpaca, self._session_date
        )
        if alpaca_closed:
            logger.info(f"Alpaca safety net closed {alpaca_closed} extra positions")

        # Phase 3: verify-and-retry loop — keep going until Alpaca is flat
        if not self.simulate:
            MAX_VERIFY = 4
            for attempt in range(1, MAX_VERIFY + 1):
                await asyncio.sleep(2)
                try:
                    positions = await asyncio.to_thread(
                        self.execution.client.get_all_positions
                    )
                    real = [p for p in positions if abs(float(p.qty)) >= 0.001]
                    if not real:
                        logger.info(
                            f"EOD liquidation verified: Alpaca is flat"
                            f"{f' after {attempt} verification(s)' if attempt > 1 else ''}"
                        )
                        break

                    logger.warning(
                        f"EOD verify {attempt}/{MAX_VERIFY}: {len(real)} positions "
                        f"still open ({', '.join(f'{p.symbol}={float(p.qty):.4f}' for p in real[:8])})"
                    )

                    # Brute force: close_all_positions then individual closes
                    try:
                        await asyncio.to_thread(
                            self.execution.client.close_all_positions,
                            cancel_orders=True,
                        )
                    except Exception:
                        pass
                    await asyncio.sleep(2)

                    # Individual close for stubborn positions
                    remaining = await asyncio.to_thread(
                        self.execution.client.get_all_positions
                    )
                    for p in remaining:
                        if abs(float(p.qty)) < 0.001:
                            continue
                        try:
                            await asyncio.to_thread(
                                self.execution.client.close_position, p.symbol
                            )
                        except Exception:
                            logger.debug(f"EOD: failed to close {p.symbol}", exc_info=True)
                except Exception:
                    logger.warning(f"EOD verify attempt {attempt} failed", exc_info=True)
            else:
                # All attempts exhausted
                try:
                    final = await asyncio.to_thread(
                        self.execution.client.get_all_positions
                    )
                    final_real = [p for p in final if abs(float(p.qty)) >= 0.001]
                    if final_real:
                        symbols = [f"{p.symbol}={float(p.qty):.4f}" for p in final_real]
                        logger.error(
                            f"EOD LIQUIDATION INCOMPLETE: {len(final_real)} positions "
                            f"survived: {symbols}. Startup cleanup will retry."
                        )
                except Exception:
                    pass

            # Phase 4: Extended-hours limit sells for anything still open
            # Market orders submitted during Phase 1-3 may sit ACCEPTED after
            # 4:00 PM close, leaving positions exposed overnight.  Cancel those
            # stuck orders and resubmit as extended-hours limit sells.
            try:
                stragglers = await asyncio.to_thread(
                    self.execution.client.get_all_positions
                )
                straggler_real = [
                    p for p in stragglers if abs(float(p.qty)) >= 0.001
                ]
                if straggler_real:
                    logger.warning(
                        f"Phase 4 (extended-hours): {len(straggler_real)} "
                        f"positions still open after Phase 3"
                    )
                    # Cancel all open orders (clears stuck market sells)
                    try:
                        await asyncio.to_thread(
                            self.execution.client.cancel_orders
                        )
                        logger.info("Phase 4: cancelled all open orders")
                        await asyncio.sleep(1)
                    except Exception as e:
                        logger.warning(f"Phase 4: cancel_orders failed: {e}")

                    ext_submitted = 0
                    ext_skipped = 0
                    for p in straggler_real:
                        qty = abs(float(p.qty))
                        price = float(p.current_price)
                        if qty >= 1.0:
                            order_id = await asyncio.to_thread(
                                self.execution.submit_extended_hours_sell,
                                p.symbol,
                                qty,
                                price,
                            )
                            if order_id:
                                ext_submitted += 1
                        else:
                            logger.info(
                                f"Phase 4: {p.symbol} qty={qty:.4f} is "
                                f"fractional — will close at next market open"
                            )
                            ext_skipped += 1
                    logger.info(
                        f"Phase 4 (extended-hours) complete: "
                        f"{ext_submitted} limit sells submitted, "
                        f"{ext_skipped} fractional skipped"
                    )
            except Exception:
                logger.warning(
                    "Phase 4 (extended-hours) failed", exc_info=True
                )

        if closed or alpaca_closed:
            self.tracker.update_all()

    async def _get_alpaca_pnl(self) -> float | None:
        """Query Alpaca account for real daily P&L (equity - last_equity).

        Returns None if simulating or on error.
        """
        if self.simulate:
            return None
        try:
            acct = await asyncio.to_thread(self.execution.client.get_account)
            equity = float(acct.equity)
            last_equity = float(acct.last_equity)
            pnl = round(equity - last_equity, 2)
            logger.info(
                f"Alpaca account: equity=${equity:.2f}, "
                f"last_equity=${last_equity:.2f}, day_pnl=${pnl:+.2f}"
            )
            return pnl
        except Exception as e:
            logger.warning(f"Failed to query Alpaca account P&L: {e}")
            return None

    def _run_screener(self) -> list[str]:
        """Run the symbol screener to expand the trading universe.

        Queries Alpaca for top most-active stocks by volume, applies a price
        floor, and extends ``self.config.arena.symbols`` with the results.
        Also seeds daily context from snapshot data for new symbols.
        Safe to call multiple times (deduplicates).  Returns the list of
        newly added symbols (empty on failure or no new finds).
        """
        if not self.config.arena.screener_enabled:
            return []

        try:
            screener = SymbolScreener(self.config)
            new_symbols, snapshots = screener.screen_with_snapshots()
            if new_symbols:
                existing = set(self.config.arena.symbols)
                to_add = [s for s in new_symbols if s not in existing]
                if to_add:
                    self.config.arena.symbols.extend(to_add)
                    logger.info(
                        f"Screener added {len(to_add)} symbols: {to_add} "
                        f"(total: {len(self.config.arena.symbols)})"
                    )
                    # Seed daily context from snapshots for newly added symbols
                    if snapshots:
                        self._seed_daily_context_from_snapshots(snapshots)
                    return to_add
            else:
                logger.info("Screener returned no new symbols")
        except Exception:
            logger.exception("Screener failed (non-fatal, continuing with existing symbols)")
        return []

    def _warmup_symbols(self, symbols: list[str]) -> None:
        """Feed recent historical bars for specific symbols through all strategies.

        Used for mid-session warmup when the screener discovers new symbols.
        """
        warmup_count = self.config.arena.warmup_bars
        try:
            now = datetime.now(ET)
            days_back = max(3, warmup_count // 200 + 2)
            historical = self.feed.fetch_historical_bars(
                symbols, days_back=days_back, end_date=now,
            )
            total_fed = 0
            for symbol in symbols:
                bars = historical.get(symbol, [])
                bars.sort(key=lambda b: b.timestamp)
                for bar in bars[-warmup_count:]:
                    bar.minutes_remaining = None
                    for strategy in self._models.values():
                        try:
                            strategy.record_bar(bar)
                            strategy.on_bar(bar)
                        except Exception:
                            pass
                    total_fed += 1
            logger.info(f"Mid-session warmup: fed {total_fed} bars for {symbols}")
        except Exception:
            logger.exception(f"Mid-session warmup failed for {symbols} (non-fatal)")

    async def _periodic_screener(self) -> None:
        """Periodically re-run the screener during a session to discover new symbols.

        When new symbols are found, subscribes to their bar/quote streams and
        runs a quick historical warmup so strategies have indicator data.
        """
        interval = self.config.arena.screener_interval_minutes
        if not self.config.arena.screener_enabled or interval <= 0:
            return

        while self._running:
            await asyncio.sleep(interval * 60)
            if not self._running:
                break

            logger.info("Periodic screener: checking for new symbols...")
            new_symbols = await asyncio.to_thread(self._run_screener)
            if not new_symbols:
                continue

            # Subscribe to bar + quote streams for new symbols
            # Run in thread — Alpaca SDK subscribe can block if stream is degraded
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self.feed.subscribe_bars, new_symbols),
                    timeout=10.0,
                )
                if self._quote_aggregator:
                    await asyncio.wait_for(
                        asyncio.to_thread(self.feed.subscribe_quotes, new_symbols),
                        timeout=10.0,
                    )
            except (asyncio.TimeoutError, Exception) as e:
                logger.error(f"Periodic screener: subscribe failed: {e!r}")
                continue

            # Warm up strategies with historical data for new symbols
            await asyncio.to_thread(self._warmup_symbols, new_symbols)

    async def _health_check(self) -> dict:
        """Run health checks on data flow, subscriptions, sell path, and model state.

        Returns a results dict and stores it in self._last_health_check.
        Purely observational — never blocks or kills the session.
        """
        import time as _time

        results: dict = {
            "timestamp": datetime.now(ET).isoformat(),
            "checks": {},
            "warnings": [],
            "errors": [],
        }

        # --- 1a. Data flow verification ---
        data_flow = {"ok": True, "last_bar_time": None, "bar_symbols": []}
        if hasattr(self.feed, "_last_bar_time") and self.feed._last_bar_time:
            from datetime import timezone
            last_bar = self.feed._last_bar_time
            data_flow["last_bar_time"] = last_bar.isoformat()
            age_sec = (datetime.now(timezone.utc) - last_bar).total_seconds()
            if age_sec > 60:
                data_flow["ok"] = False
                msg = f"No bar received in {age_sec:.0f}s (last: {last_bar.isoformat()})"
                results["warnings"].append(msg)
                logger.warning(f"[HEALTH] Data flow: {msg}")
            else:
                logger.info(f"[HEALTH] Data flow: OK (last bar {age_sec:.0f}s ago)")
        elif self._bar_count > 0:
            logger.info("[HEALTH] Data flow: OK (bars received, no timestamp tracking)")
        else:
            data_flow["ok"] = False
            msg = "No bars received yet"
            results["warnings"].append(msg)
            logger.warning(f"[HEALTH] Data flow: {msg}")

        # Check core symbols in subscribed bars
        core = {"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "SPY"}
        subscribed = getattr(self.feed, "_subscribed_bars", set())
        data_flow["subscribed_count"] = len(subscribed)
        missing_core = core - subscribed
        if missing_core:
            msg = f"Core symbols not subscribed: {sorted(missing_core)}"
            results["warnings"].append(msg)
            logger.warning(f"[HEALTH] {msg}")
        data_flow["missing_core"] = sorted(missing_core) if missing_core else []
        results["checks"]["data_flow"] = data_flow

        # --- 1b. Subscription integrity ---
        sub_check = {"ok": True, "gaps": [], "auto_fixed": []}
        positioned = await asyncio.to_thread(self._get_positioned_symbols)
        sub_check["positioned_count"] = len(positioned)
        gaps = positioned - subscribed
        if gaps:
            sub_check["ok"] = False
            sub_check["gaps"] = sorted(gaps)
            msg = f"Position symbols not subscribed for bars: {sorted(gaps)}"
            results["warnings"].append(msg)
            logger.warning(f"[HEALTH] Subscription gap: {msg}")
            # Auto-fix: subscribe to missing symbols
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self.feed.subscribe_bars, sorted(gaps)),
                    timeout=10.0,
                )
                sub_check["auto_fixed"] = sorted(gaps)
                logger.info(f"[HEALTH] Auto-subscribed to {sorted(gaps)}")
            except Exception as e:
                logger.error(f"[HEALTH] Failed to auto-subscribe: {e!r}")
        else:
            logger.info(f"[HEALTH] Subscriptions: OK ({len(positioned)} positioned, {len(subscribed)} subscribed)")
        results["checks"]["subscriptions"] = sub_check

        # --- 1c. Sell path smoke test ---
        sell_path = {"ok": True, "blocked_at": None}
        dummy_signal = TradeSignal(symbol="__HEALTH_CHECK__", side="sell", quantity=0)
        try:
            # Walk through the sell guards in submit_order manually
            blocked = None
            # Guard 1: min quantity (qty=0 < 0.001 — this SHOULD block)
            # We test with qty=1 to get past min qty
            dummy_signal_test = TradeSignal(symbol="__HEALTH_CHECK__", side="sell", quantity=1.0)

            # Guard 2: circuit breaker — only blocks buys, should pass sells
            # (we don't need a real model_id, just check the logic)
            if dummy_signal_test.side == "buy" and 0 in self.execution._daily_loss_breached:
                blocked = "circuit_breaker"

            # Guard 3: wash trade cooldown — only blocks buys
            if not blocked and dummy_signal_test.side == "buy":
                blocked = "wash_trade_cooldown"

            # Guard 4: risk limits — check_risk_limits only blocks buys
            # (sell path doesn't raise RiskLimitExceeded)

            if blocked:
                sell_path["ok"] = False
                sell_path["blocked_at"] = blocked
                msg = f"Sell path BLOCKED at {blocked}"
                results["errors"].append(msg)
                logger.error(f"[HEALTH] {msg}")
            else:
                logger.info("[HEALTH] Sell path: OK (all guards pass sells)")
        except Exception as e:
            sell_path["ok"] = False
            sell_path["blocked_at"] = f"exception: {e!r}"
            logger.error(f"[HEALTH] Sell path check exception: {e!r}")
        results["checks"]["sell_path"] = sell_path

        # --- 1d. Model state sanity ---
        model_check = {"ok": True, "warnings": [], "zero_trade_models": []}
        for model_id, strategy in self._models.items():
            record = self._model_records.get(model_id)
            name = record.name if record else f"model_{model_id}"

            # Negative capital with positions
            if strategy.current_capital < 0 and any(q > 0 for q in strategy._positions.values()):
                msg = f"{name}: negative capital ({strategy.current_capital:.2f}) with open positions"
                model_check["warnings"].append(msg)
                results["errors"].append(msg)
                logger.error(f"[HEALTH] {msg}")

            # Check in-memory positions vs DB
            db_positions = {}
            try:
                from src.core.database import Position
                db = get_session(self.config.db_path)
                try:
                    rows = db.query(Position).filter(
                        Position.model_id == model_id, Position.quantity > 0.001
                    ).all()
                    db_positions = {r.symbol: r.quantity for r in rows}
                finally:
                    db.close()
            except Exception:
                pass

            mem_positions = {s: q for s, q in strategy._positions.items() if q > 0.001}
            if set(mem_positions.keys()) != set(db_positions.keys()):
                mem_only = set(mem_positions.keys()) - set(db_positions.keys())
                db_only = set(db_positions.keys()) - set(mem_positions.keys())
                msg = f"{name}: position mismatch (mem_only={sorted(mem_only)}, db_only={sorted(db_only)})"
                model_check["warnings"].append(msg)
                results["warnings"].append(msg)
                logger.warning(f"[HEALTH] {msg}")

        # Zero-trade models (info level)
        if self._bar_count > 50:
            for model_id, strategy in self._models.items():
                record = self._model_records.get(model_id)
                name = record.name if record else f"model_{model_id}"
                metrics = self.tracker._metrics.get(model_id)
                trades = metrics.total_trades if metrics else 0
                if trades == 0:
                    model_check["zero_trade_models"].append(name)
            if model_check["zero_trade_models"]:
                logger.info(
                    f"[HEALTH] Models with 0 trades after {self._bar_count} bars: "
                    f"{model_check['zero_trade_models']}"
                )

        if model_check["warnings"]:
            model_check["ok"] = False
        results["checks"]["model_state"] = model_check

        # Overall status
        all_ok = all(c.get("ok", True) for c in results["checks"].values())
        results["overall"] = "OK" if all_ok else "DEGRADED"
        logger.info(f"[HEALTH] Overall: {results['overall']} ({len(results['warnings'])} warnings, {len(results['errors'])} errors)")

        self._last_health_check = results
        return results

    async def _periodic_health_check(self) -> None:
        """Run health checks periodically during a session."""
        interval = self._health_check_interval_minutes
        if interval <= 0:
            return

        while self._running:
            await asyncio.sleep(interval * 60)
            if not self._running:
                break
            try:
                await self._health_check()
            except Exception:
                logger.exception("[HEALTH] Periodic health check failed (non-fatal)")

    # ── Daily context tracking ───────────────────────────────────────────

    def _seed_daily_context_from_warmup(self) -> None:
        """Derive daily context from warmup bars already in strategy history.

        Scans any strategy's _bar_history for today's bars (>= 9:30 AM ET)
        to find daily open, running high/low, cumulative VWAP, and yesterday's
        close per symbol.
        """
        try:
            self._seed_daily_context_from_warmup_inner()
        except Exception:
            logger.exception("Daily context warmup seeding failed (non-fatal)")

    def _seed_daily_context_from_warmup_inner(self) -> None:
        today = datetime.now(ET).date()
        market_open_dt = datetime.combine(today, MARKET_OPEN, tzinfo=ET)

        # Grab bar history from the first strategy that has one
        sample_strategy = next(iter(self._models.values()), None)
        if not sample_strategy or not hasattr(sample_strategy, '_bar_history'):
            logger.info("Daily context: no bar history available after warmup")
            return

        # _bar_history is dict[str, list[BarData]] — already keyed by symbol
        bars_by_symbol = sample_strategy._bar_history

        def _to_datetime(ts) -> datetime:
            """Convert pd.Timestamp or datetime to tz-aware datetime."""
            if hasattr(ts, 'to_pydatetime'):
                dt = ts.to_pydatetime()
            else:
                dt = ts
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=ET)
            return dt

        seeded = 0
        for symbol, bars in bars_by_symbol.items():
            if not bars:
                continue
            bars = list(bars)  # copy so sort doesn't mutate strategy state
            bars.sort(key=lambda b: b.timestamp)

            # Find yesterday's close (last bar before today's open)
            prev_close = None
            for bar in reversed(bars):
                bar_dt = _to_datetime(bar.timestamp)
                if bar_dt < market_open_dt:
                    prev_close = bar.close
                    break

            # Filter to today's market-hours bars only
            today_bars = []
            for bar in bars:
                bar_dt = _to_datetime(bar.timestamp)
                if bar_dt >= market_open_dt:
                    today_bars.append(bar)

            if not today_bars:
                # No today bars yet (pre-market) — still store prev_close if available
                if prev_close is not None:
                    self._daily_context[symbol] = {
                        "open": None,
                        "high": None,
                        "low": None,
                        "vwap_pv": 0.0,
                        "vwap_vol": 0,
                        "prev_close": prev_close,
                    }
                continue

            first = today_bars[0]
            high = max(b.high for b in today_bars)
            low = min(b.low for b in today_bars)
            vwap_pv = sum(
                ((b.high + b.low + b.close) / 3.0) * b.volume for b in today_bars
            )
            vwap_vol = sum(b.volume for b in today_bars)

            self._daily_context[symbol] = {
                "open": first.open,
                "high": high,
                "low": low,
                "vwap_pv": vwap_pv,
                "vwap_vol": vwap_vol,
                "prev_close": prev_close,
            }
            seeded += 1

        logger.info(f"Daily context seeded for {seeded} symbols from warmup bars")

    def _seed_daily_context_from_snapshots(self, snapshots: dict) -> None:
        """Seed daily context from Alpaca snapshot objects (screener additions).

        Each snapshot has ``daily_bar`` (open/high/low/close/volume/vwap) and
        ``previous_daily_bar`` (for prev_close).
        """
        seeded = 0
        for symbol, snap in snapshots.items():
            if symbol in self._daily_context:
                continue  # don't overwrite existing context

            daily = getattr(snap, 'daily_bar', None)
            if not daily:
                continue

            prev_daily = getattr(snap, 'previous_daily_bar', None)
            prev_close = float(prev_daily.close) if prev_daily and hasattr(prev_daily, 'close') else None
            vwap = float(daily.vwap) if hasattr(daily, 'vwap') and daily.vwap else 0.0
            volume = int(daily.volume) if hasattr(daily, 'volume') and daily.volume else 0

            self._daily_context[symbol] = {
                "open": float(daily.open),
                "high": float(daily.high),
                "low": float(daily.low),
                "vwap_pv": vwap * volume,  # reconstruct cumulative price*volume
                "vwap_vol": volume,
                "prev_close": prev_close,
            }
            seeded += 1

        if seeded:
            logger.info(f"Daily context seeded for {seeded} symbols from snapshots")

    def _update_and_attach_daily_context(self, bars: list[BarData]) -> None:
        """Update running high/low/VWAP from new bars and attach DailyContext to each."""
        for bar in bars:
            ctx = self._daily_context.get(bar.symbol)
            if ctx is None:
                # First time seeing this symbol — use this bar as the open
                ctx = {
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "vwap_pv": ((bar.high + bar.low + bar.close) / 3.0) * bar.volume,
                    "vwap_vol": bar.volume,
                    "prev_close": None,
                }
                self._daily_context[bar.symbol] = ctx
            elif ctx.get("open") is None:
                # Had prev_close from warmup but no today bars yet
                ctx["open"] = bar.open
                ctx["high"] = bar.high
                ctx["low"] = bar.low
                ctx["vwap_pv"] = ((bar.high + bar.low + bar.close) / 3.0) * bar.volume
                ctx["vwap_vol"] = bar.volume
            else:
                # Update running values
                ctx["high"] = max(ctx["high"], bar.high)
                ctx["low"] = min(ctx["low"], bar.low)
                typical = (bar.high + bar.low + bar.close) / 3.0
                ctx["vwap_pv"] += typical * bar.volume
                ctx["vwap_vol"] += bar.volume

            # Build DailyContext dataclass
            daily_open = ctx["open"]
            if daily_open is None:
                continue  # shouldn't happen here, but guard

            daily_range = ctx["high"] - ctx["low"]
            vwap = ctx["vwap_pv"] / ctx["vwap_vol"] if ctx["vwap_vol"] > 0 else bar.close

            change_from_open = (bar.close - daily_open) / daily_open * 100.0 if daily_open else 0.0
            prev_close = ctx.get("prev_close")
            change_from_prev = (
                (bar.close - prev_close) / prev_close * 100.0
                if prev_close
                else None
            )
            range_pos = (
                (bar.close - ctx["low"]) / daily_range
                if daily_range > 0
                else 0.5
            )

            bar.daily_context = DailyContext(
                daily_open=daily_open,
                running_high=ctx["high"],
                running_low=ctx["low"],
                daily_vwap=round(vwap, 4),
                prev_close=prev_close,
                change_from_open_pct=round(change_from_open, 4),
                change_from_prev_close_pct=round(change_from_prev, 4) if change_from_prev is not None else None,
                daily_range_position=round(range_pos, 4),
            )

        # Attach SPY's daily context to every bar as a market trend proxy
        spy_ctx = self._daily_context.get("SPY")
        if spy_ctx and spy_ctx.get("open") is not None:
            # Find the latest SPY bar in this batch for current close price,
            # otherwise use last known high as approximation
            spy_close = None
            for bar in reversed(bars):
                if bar.symbol == "SPY":
                    spy_close = bar.close
                    break
            if spy_close is None:
                spy_close = spy_ctx["high"]  # best available approximation

            spy_open = spy_ctx["open"]
            spy_range = spy_ctx["high"] - spy_ctx["low"]
            spy_vwap = spy_ctx["vwap_pv"] / spy_ctx["vwap_vol"] if spy_ctx["vwap_vol"] > 0 else spy_close
            spy_prev = spy_ctx.get("prev_close")

            spy_dc = DailyContext(
                daily_open=spy_open,
                running_high=spy_ctx["high"],
                running_low=spy_ctx["low"],
                daily_vwap=round(spy_vwap, 4),
                prev_close=spy_prev,
                change_from_open_pct=round((spy_close - spy_open) / spy_open * 100.0, 4) if spy_open else 0.0,
                change_from_prev_close_pct=round((spy_close - spy_prev) / spy_prev * 100.0, 4) if spy_prev else None,
                daily_range_position=round((spy_close - spy_ctx["low"]) / spy_range, 4) if spy_range > 0 else 0.5,
            )
            for bar in bars:
                bar.spy_daily_context = spy_dc

    def _warmup_strategies(self) -> None:
        """Feed recent historical bars through strategies without placing orders.

        Primes indicators (moving averages, RSI, etc.) so strategies don't
        need 15-50 bars of live data before generating meaningful signals.
        """
        warmup_count = self.config.arena.warmup_bars
        logger.info(f"Warming up strategies with {warmup_count} recent bars...")

        try:
            from datetime import timedelta as td
            now = datetime.now(ET)
            # Fetch bars from the last few trading days to get enough data
            days_back = max(3, warmup_count // 200 + 2)
            historical = self.feed.fetch_historical_bars(
                self.config.arena.symbols,
                days_back=days_back,
                end_date=now,
            )

            # Sort and take the last N bars per symbol
            total_fed = 0
            for symbol in self.config.arena.symbols:
                bars = historical.get(symbol, [])
                bars.sort(key=lambda b: b.timestamp)
                warmup_slice = bars[-warmup_count:]

                for bar in warmup_slice:
                    bar.minutes_remaining = None  # don't trigger liquidation
                    for strategy in self._models.values():
                        try:
                            strategy.record_bar(bar)
                            # Call on_bar silently so indicators update, ignore signals
                            strategy.on_bar(bar)
                        except Exception:
                            pass
                    total_fed += 1

            logger.info(f"Warmup complete: fed {total_fed} bars across {len(self.config.arena.symbols)} symbols")
        except Exception:
            logger.exception("Warmup failed (non-fatal, continuing)")

    async def _premarket_warmup(self) -> None:
        """Run pre-market warmup: historical bars + live pre-market polling.

        1. Wait until configured pre-market start time (default 9:00 AM ET)
        2. Run historical warmup (primes indicators with recent days' bars)
        3. Poll Alpaca SIP feed for today's pre-market bars until 9:30 AM ET
        4. If SIP is unavailable (free account), fall back to historical only
        """
        cfg = self.config.arena
        pm_hour = cfg.premarket_start_hour_et
        pm_minute = cfg.premarket_start_minute_et
        poll_interval = cfg.premarket_poll_interval_sec

        # Step 1: Wait until pre-market start time
        await self._wait_until_et(pm_hour, pm_minute)

        # Step 2: Screener (expand symbol universe before warmup)
        self._set_status("screener", "Running pre-market symbol screener")
        self._run_screener()

        # Step 3: Historical warmup (existing behavior)
        self._set_status("warmup", f"Historical warmup ({cfg.warmup_bars} bars)")
        self._warmup_strategies()
        self._seed_daily_context_from_warmup()

        # Step 3: Try fetching today's pre-market bars via SIP
        sip_available = True
        latest_timestamps: dict[str, datetime] = {}

        self._set_status("warmup", "Fetching pre-market bars (SIP)")
        try:
            premarket_bars = self.feed.fetch_premarket_bars(
                self.config.arena.symbols
            )
            bars_fed = self._feed_premarket_bars(premarket_bars, latest_timestamps)
            logger.info(f"Pre-market warmup: fed {bars_fed} pre-market bars to strategies")
        except Exception as e:
            sip_available = False
            logger.warning(
                f"Pre-market data unavailable (SIP feed required): {e}. "
                f"Falling back to historical warmup only. "
                f"Strategies are still primed with {cfg.warmup_bars} historical bars."
            )

        if not sip_available:
            self._set_status("waiting", "Waiting for market open (9:30 AM ET)")
            logger.info("Waiting for market open (9:30 AM ET)...")
            await self._wait_until_et(9, 30)
            return

        # Step 4: Poll for new pre-market bars until 9:30 AM ET
        logger.info(
            f"Pre-market polling active. Fetching new bars every "
            f"{poll_interval}s until 9:30 AM ET..."
        )
        poll_count = 0
        self._set_status("premarket", f"Polling pre-market bars every {poll_interval}s")

        while True:
            now = datetime.now(ET)
            if now.time() >= MARKET_OPEN:
                break

            seconds_to_open = (
                now.replace(hour=9, minute=30, second=0, microsecond=0) - now
            ).total_seconds()
            if seconds_to_open <= 0:
                break

            await asyncio.sleep(min(poll_interval, seconds_to_open))

            now = datetime.now(ET)
            if now.time() >= MARKET_OPEN:
                break

            poll_count += 1
            try:
                since = min(latest_timestamps.values()) if latest_timestamps else None
                new_bars = self.feed.fetch_premarket_bars(
                    self.config.arena.symbols,
                    since=since,
                )
                bars_fed = self._feed_premarket_bars(new_bars, latest_timestamps)
                if bars_fed > 0:
                    logger.info(
                        f"Pre-market poll #{poll_count}: fed {bars_fed} new bars "
                        f"({now.strftime('%H:%M:%S ET')})"
                    )
            except Exception:
                logger.exception(f"Pre-market poll #{poll_count} failed (non-fatal)")

        logger.info(
            f"Pre-market warmup complete. {poll_count} polls executed. "
            f"Market is open — starting Session 1."
        )

    def _feed_premarket_bars(
        self,
        bars_by_symbol: dict[str, list["BarData"]],
        latest_timestamps: dict[str, datetime],
    ) -> int:
        """Feed pre-market bars through strategies (indicators only, no orders).

        Deduplicates via latest_timestamps (updated in-place).
        Returns number of bars fed after dedup.
        """
        total_fed = 0

        for symbol in self.config.arena.symbols:
            bars = bars_by_symbol.get(symbol, [])
            if not bars:
                continue

            last_seen = latest_timestamps.get(symbol)
            if last_seen:
                bars = [b for b in bars if b.timestamp > last_seen]

            if not bars:
                continue

            for bar in bars:
                bar.minutes_remaining = None
                for strategy in self._models.values():
                    try:
                        strategy.record_bar(bar)
                        strategy.on_bar(bar)
                    except Exception:
                        pass
                total_fed += 1

            latest_timestamps[symbol] = bars[-1].timestamp

        return total_fed

    def _save_daily_ledger(self) -> None:
        """Record end-of-day capital for each model in the daily ledger."""
        db = get_session(self.config.db_path)
        try:
            from sqlalchemy import func

            from src.core.database import Order, OrderStatus, OrderSide

            for model_id, model_record in self._model_records.items():
                db_model = db.query(TradingModel).get(model_id)
                if not db_model:
                    continue

                start_capital = db_model.initial_capital

                # Use realized PnL from filled sell orders (source of truth)
                realized_pnl = (
                    db.query(func.sum(Order.realized_pnl))
                    .filter(
                        Order.model_id == model_id,
                        Order.session_date == self._session_date,
                        Order.status == OrderStatus.FILLED,
                        Order.side == OrderSide.SELL,
                    )
                    .scalar() or 0.0
                )
                end_capital = start_capital + realized_pnl
                daily_return = (
                    realized_pnl / start_capital * 100
                    if start_capital > 0 else 0.0
                )

                # Count trades for today
                trade_count = (
                    db.query(func.count(Order.id))
                    .filter(
                        Order.model_id == model_id,
                        Order.session_date == self._session_date,
                        Order.status == OrderStatus.FILLED,
                    )
                    .scalar() or 0
                )

                # Compute cumulative return from previous ledger entries
                prev_entry = (
                    db.query(DailyLedger)
                    .filter(
                        DailyLedger.model_id == model_id,
                        DailyLedger.session_date < self._session_date,
                    )
                    .order_by(DailyLedger.session_date.desc())
                    .first()
                )
                prev_cumulative = prev_entry.cumulative_return_pct if prev_entry else 0.0
                cumulative = prev_cumulative + daily_return

                # Upsert
                existing = (
                    db.query(DailyLedger)
                    .filter(
                        DailyLedger.model_id == model_id,
                        DailyLedger.session_date == self._session_date,
                    )
                    .first()
                )
                if existing:
                    existing.end_capital = end_capital
                    existing.daily_return_pct = daily_return
                    existing.cumulative_return_pct = cumulative
                    existing.total_trades = trade_count
                else:
                    db.add(DailyLedger(
                        model_id=model_id,
                        session_date=self._session_date,
                        start_capital=start_capital,
                        end_capital=end_capital,
                        daily_return_pct=round(daily_return, 4),
                        cumulative_return_pct=round(cumulative, 4),
                        total_trades=trade_count,
                        generation=model_record.generation,
                    ))

            db.commit()
            logger.info("Daily ledger saved")
        except Exception:
            db.rollback()
            logger.exception("Failed to save daily ledger")
        finally:
            db.close()

    async def _run_cfa_review(self) -> None:
        """Run CFA-style post-day review (non-fatal)."""
        if not self.config.arena.cfa_review_enabled:
            logger.info("CFA review disabled, skipping")
            return

        self._set_status("reviewing", "Running CFA review")
        try:
            incident_notes = "\n".join(f"- {n}" for n in self._incident_notes) if self._incident_notes else ""
            review = await asyncio.to_thread(
                run_cfa_review,
                self.config.db_path,
                self._session_date,
                self.config.arena.cfa_review_model,
                self.config.arena.cfa_review_timeout_sec,
                self.config.arena.cfa_review_lookback_days,
                incident_notes,
            )
            if review:
                self._print_cfa_review(review)
                # Apply CFA-generated strategy if present
                strategy_code = review.get("generated_strategy")
                if strategy_code:
                    self._set_status("reviewing", "Applying CFA-generated strategy")
                    success = await asyncio.to_thread(
                        apply_cfa_strategy,
                        strategy_code,
                        self.config.db_path,
                        self.config.arena.initial_capital,
                    )
                    if success:
                        logger.info("CFA-generated strategy applied — will be active next session")
                    else:
                        logger.warning("CFA-generated strategy failed validation — skipped")
            else:
                logger.warning("CFA Review returned no result")
        except Exception:
            logger.exception("CFA Review failed (non-fatal)")

    async def _run_cfa_review_with_recommendations(self) -> dict[str, dict]:
        """Run CFA review and extract parameter recommendations.

        Wraps _run_cfa_review() logic but returns the extracted recommendations
        dict for use by _self_improve(). Also handles strategy generation (existing
        behavior). Returns {} on failure (graceful fallback to random mutations).
        """
        if not self.config.arena.cfa_review_enabled:
            logger.info("CFA review disabled, skipping (no recommendations)")
            return {}

        self._set_status("reviewing", "Running CFA review")
        try:
            incident_notes = "\n".join(f"- {n}" for n in self._incident_notes) if self._incident_notes else ""
            review = await asyncio.to_thread(
                run_cfa_review,
                self.config.db_path,
                self._session_date,
                self.config.arena.cfa_review_model,
                self.config.arena.cfa_review_timeout_sec,
                self.config.arena.cfa_review_lookback_days,
                incident_notes,
            )
            if not review:
                logger.warning("CFA Review returned no result — no recommendations")
                return {}

            self._print_cfa_review(review)

            # Apply CFA-generated strategy if present
            strategy_code = review.get("generated_strategy")
            if strategy_code:
                self._set_status("reviewing", "Applying CFA-generated strategy")
                success = await asyncio.to_thread(
                    apply_cfa_strategy,
                    strategy_code,
                    self.config.db_path,
                    self.config.arena.initial_capital,
                )
                if success:
                    logger.info("CFA-generated strategy applied — will be active next session")
                else:
                    logger.warning("CFA-generated strategy failed validation — skipped")

            # Extract parameter recommendations
            valid_names = {m.model_name for m in self.tracker.get_leaderboard()}
            recs = extract_parameter_recommendations(review, valid_model_names=valid_names)
            return recs

        except Exception:
            logger.exception("CFA Review failed (non-fatal) — no recommendations")
            return {}

    def _print_cfa_review(self, review: dict) -> None:
        """Print CFA review highlights to the console."""
        grade = review.get("portfolio_grade", "?")
        justification = review.get("portfolio_grade_justification", "")
        summary = review.get("executive_summary", "")
        red_flags = review.get("red_flags", [])
        action_items = review.get("action_items", [])
        next_day = review.get("next_day_recommendations", "")
        md_path = f"logs/cfa_review_{self._session_date}.md"

        divider = "=" * 60
        print(f"\n{divider}")
        print(f"  CFA DAILY REVIEW — {self._session_date}")
        print(f"  Grade: {grade} — {justification}")
        print(divider)
        print(f"\n{summary}\n")

        if red_flags:
            print("RED FLAGS:")
            for flag in red_flags:
                print(f"  !! {flag}")
            print()

        if action_items:
            print("ACTION ITEMS:")
            for item in action_items:
                if isinstance(item, dict):
                    prio = item.get("priority", "medium").upper()
                    print(f"  [{prio}] {item.get('action', '')}")
                    print(f"         {item.get('rationale', '')}")
                else:
                    print(f"  - {item}")
            print()

        if next_day:
            print("TOMORROW:")
            print(f"  {next_day}\n")

        print(f"Full report: {md_path}")
        print(f"{divider}\n")

        logger.info(f"CFA Review complete — Grade: {grade}")

    def _evaluate_pending_mutations(self) -> None:
        """Evaluate whether pending mutations improved performance.

        Called before each _self_improve(). Compares current session return
        to the pre-mutation return stored in mutation_memory._pending.
        """
        if not self.config.arena.mutation_memory_enabled:
            return

        leaderboard = self.tracker.get_leaderboard()
        if not leaderboard:
            return

        db = get_session(self.config.db_path)
        try:
            evaluated = 0
            for metrics in leaderboard:
                db_model = db.query(TradingModel).get(metrics.model_id)
                if not db_model or not db_model.mutation_memory:
                    continue

                memory = db_model.mutation_memory
                if "_pending" not in memory:
                    continue

                memory = MutationMemory.evaluate_pending(
                    memory, metrics.return_pct,
                    decay=self.config.arena.mutation_memory_decay,
                )
                db_model.mutation_memory = memory
                evaluated += 1

            if evaluated > 0:
                db.commit()
                logger.info(f"Evaluated pending mutations for {evaluated} models")
        finally:
            db.close()

    def _self_improve(self, cfa_recommendations: dict[str, dict] | None = None) -> None:
        """Between-session self-improvement.

        Each strategy tweaks its parameters based on session performance.
        CFA recommendations are applied first (if available), then random
        mutations fill in params CFA didn't address.

        Args:
            cfa_recommendations: Optional dict from extract_parameter_recommendations().
                Maps model_name -> {param: {"value": num, "confidence": str, "rationale": str}}.
        """
        logger.info("=== BREAK: Self-improvement phase ===")
        if cfa_recommendations:
            logger.info(f"CFA recommendations available for {len(cfa_recommendations)} models")
        leaderboard = self.tracker.get_leaderboard()
        use_memory = self.config.arena.mutation_memory_enabled
        dampening = self.config.arena.mutation_bias_dampening

        # Wire COLLAB voter eligibility: top 5 non-collab strategy types vote
        top_types = [m.strategy_type for m in leaderboard if m.strategy_type != "collab"][:5]
        collab = next((s for s in self._models.values() if s.strategy_type == "collab"), None)
        if collab and top_types:
            collab.set_eligible_voters(top_types)
            logger.info(f"COLLAB voters updated: {top_types}")

        db = get_session(self.config.db_path)
        try:
            for metrics in leaderboard:
                model_id = metrics.model_id
                strategy = self._models.get(model_id)
                model_record = self._model_records.get(model_id)
                if not strategy or not model_record:
                    continue

                params = strategy.get_params()
                new_params = copy.deepcopy(params)

                # Inject base class risk params so CFA can tune them
                if "stop_loss_pct" not in new_params:
                    new_params["stop_loss_pct"] = getattr(strategy, "stop_loss_pct", 2.0)
                if "take_profit_pct" not in new_params:
                    new_params["take_profit_pct"] = getattr(strategy, "take_profit_pct", 3.0)
                # Inject trailing stop tiers for CFA visibility
                if strategy.trailing_stop_tiers:
                    new_params["trailing_stop_tiers"] = [list(t) for t in strategy.trailing_stop_tiers]
                # Inject patience stop tiers for CFA visibility
                if strategy.patience_stop_tiers:
                    new_params["patience_stop_tiers"] = [list(t) for t in strategy.patience_stop_tiers]

                # Strategies that lost money get stronger mutations
                if metrics.return_pct < 0:
                    mutation_strength = 0.10
                elif metrics.return_pct < 1.0:
                    mutation_strength = 0.05
                else:
                    mutation_strength = 0.02

                # Load mutation memory and compute biases
                db_model = db.query(TradingModel).get(model_id)
                memory = db_model.mutation_memory if (db_model and use_memory) else None
                biases = MutationMemory.get_biases(memory) if memory else {}
                mutations_applied: dict[str, str] = {}

                # Get CFA recs for this model (by name)
                model_cfa_recs = (cfa_recommendations or {}).get(metrics.model_name, {})
                cfa_applied_count = 0

                # Handle trailing_stop_tiers CFA rec before the numeric loop
                tiers_rec = model_cfa_recs.pop("trailing_stop_tiers", None)
                if tiers_rec and isinstance(tiers_rec.get("value"), list):
                    try:
                        raw_tiers = tiers_rec["value"]
                        validated = []
                        for pair in raw_tiers:
                            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                                gain = max(0.1, min(20.0, float(pair[0])))
                                trail = max(0.1, min(10.0, float(pair[1])))
                                validated.append((gain, trail))
                        if validated:
                            validated.sort(key=lambda t: t[0])
                            new_params["trailing_stop_tiers"] = [list(t) for t in validated]
                            strategy.trailing_stop_tiers = validated
                            cfa_applied_count += 1
                            logger.info(
                                f"  CFA-guided [{tiers_rec.get('confidence', '?')}] "
                                f"{metrics.model_name}.trailing_stop_tiers: {validated} "
                                f"(rationale={tiers_rec.get('rationale', '')[:80]})"
                            )
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Invalid CFA trailing_stop_tiers for {metrics.model_name}: {e}")

                # Handle patience_stop_tiers CFA rec before the numeric loop
                patience_rec = model_cfa_recs.pop("patience_stop_tiers", None)
                if patience_rec and isinstance(patience_rec.get("value"), list):
                    try:
                        raw_tiers = patience_rec["value"]
                        validated = []
                        for pair in raw_tiers:
                            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                                bars = max(1, min(120, int(pair[0])))
                                exit_pct = max(-10.0, min(-0.05, float(pair[1])))
                                validated.append((bars, exit_pct))
                        if validated:
                            validated.sort(key=lambda t: t[0])
                            new_params["patience_stop_tiers"] = [list(t) for t in validated]
                            strategy.patience_stop_tiers = validated
                            cfa_applied_count += 1
                            logger.info(
                                f"  CFA-guided [{patience_rec.get('confidence', '?')}] "
                                f"{metrics.model_name}.patience_stop_tiers: {validated} "
                                f"(rationale={patience_rec.get('rationale', '')[:80]})"
                            )
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Invalid CFA patience_stop_tiers for {metrics.model_name}: {e}")

                for key, value in new_params.items():
                    if not isinstance(value, (int, float)):
                        continue

                    cfa_rec = model_cfa_recs.get(key)
                    if cfa_rec:
                        target = cfa_rec["value"]
                        confidence = cfa_rec["confidence"]

                        if confidence == "high":
                            # Direct override
                            new_value = target
                            mutations_applied[key] = "cfa_high"
                        elif confidence == "medium":
                            # Blend 70% toward target
                            new_value = value + 0.7 * (target - value)
                            mutations_applied[key] = "cfa_medium"
                        else:  # low
                            # Bias random mutation toward CFA's target
                            direction_bias = 1.0 if target > value else -1.0
                            perturbation = abs(MutationMemory.apply_bias(
                                mutation_strength, 0.0, dampening
                            )) * direction_bias
                            new_value = value * (1 + perturbation)
                            mutations_applied[key] = "cfa_low"

                        if isinstance(value, int):
                            new_value = max(1, int(round(new_value)))
                        else:
                            new_value = round(new_value, 6)
                        new_params[key] = new_value
                        cfa_applied_count += 1
                        logger.info(
                            f"  CFA-guided [{confidence}] {metrics.model_name}.{key}: "
                            f"{value} -> {new_value} (target={target}, rationale={cfa_rec.get('rationale', '')[:80]})"
                        )
                    elif random.random() < 0.4:
                        # No CFA rec: existing random mutation
                        bias = biases.get(key, 0.0)
                        perturbation = MutationMemory.apply_bias(
                            mutation_strength, bias, dampening
                        )
                        new_value = value * (1 + perturbation)
                        if isinstance(value, int):
                            new_value = max(1, int(round(new_value)))
                        else:
                            new_value = round(new_value, 6)
                        new_params[key] = new_value

                        # Record direction for pending evaluation
                        if new_value > value:
                            mutations_applied[key] = "up"
                        elif new_value < value:
                            mutations_applied[key] = "down"

                strategy.set_params(new_params)

                # Apply stop_loss_pct / take_profit_pct back to base class with clamping
                if "stop_loss_pct" in new_params:
                    strategy.stop_loss_pct = max(0.5, min(10.0, float(new_params["stop_loss_pct"])))
                if "take_profit_pct" in new_params:
                    strategy.take_profit_pct = max(0.5, min(15.0, float(new_params["take_profit_pct"])))
                # Apply trailing_stop_tiers back to base class (already validated above)
                if "trailing_stop_tiers" in new_params and isinstance(new_params["trailing_stop_tiers"], list):
                    try:
                        tiers = [(float(t[0]), float(t[1])) for t in new_params["trailing_stop_tiers"]]
                        tiers.sort(key=lambda t: t[0])
                        strategy.trailing_stop_tiers = tiers
                    except (TypeError, IndexError, ValueError):
                        pass
                # Apply patience_stop_tiers back to base class (already validated above)
                if "patience_stop_tiers" in new_params and isinstance(new_params["patience_stop_tiers"], list):
                    try:
                        tiers = [(int(t[0]), float(t[1])) for t in new_params["patience_stop_tiers"]]
                        tiers.sort(key=lambda t: t[0])
                        strategy.patience_stop_tiers = tiers
                    except (TypeError, IndexError, ValueError):
                        pass

                # Generate LLM watch rules (skip COLLAB)
                if (
                    self.config.arena.llm_watch_rules_enabled
                    and strategy.strategy_type != "collab"
                ):
                    available = INDICATOR_CATALOG.get(strategy.strategy_type, [])
                    if available:
                        # Gather reflection text for context
                        reflection = ""
                        try:
                            reflection_db = get_session(self.config.db_path)
                            latest_summary = (
                                reflection_db.query(ModelSummary)
                                .filter(
                                    ModelSummary.model_id == model_id,
                                    ModelSummary.summary_type == "post_session",
                                )
                                .order_by(ModelSummary.id.desc())
                                .first()
                            )
                            if latest_summary:
                                reflection = latest_summary.reflection or ""
                            reflection_db.close()
                        except Exception:
                            pass

                        perf = {
                            "return_pct": metrics.return_pct,
                            "sharpe": metrics.sharpe_ratio,
                            "trades": metrics.total_trades,
                            "win_rate": metrics.win_rate,
                            "fitness": metrics.fitness,
                        }
                        new_rules = generate_watch_rules(
                            strategy_type=strategy.strategy_type,
                            model_name=metrics.model_name,
                            current_params=strategy.get_params(),
                            current_rules=strategy._watch_rules,
                            reflection=reflection,
                            available_indicators=available,
                            performance=perf,
                            model_id=self.config.arena.llm_watch_rules_model,
                            timeout_sec=self.config.arena.llm_watch_rules_timeout_sec,
                            max_rules=self.config.arena.llm_watch_rules_max,
                        )
                        if new_rules is not None:
                            strategy._watch_rules = new_rules

                # Record pending mutations and persist
                if db_model:
                    if use_memory and mutations_applied:
                        memory = MutationMemory.record_pending(
                            memory, metrics.return_pct, mutations_applied
                        )
                        db_model.mutation_memory = memory
                    # Persist params including _watch_rules
                    saved_params = strategy.get_params()
                    saved_params["_watch_rules"] = strategy._watch_rules
                    db_model.parameters = saved_params

                biased_count = sum(1 for b in biases.values() if b != 0.0)
                rules_count = len(strategy._watch_rules)
                cfa_tag = f", cfa={cfa_applied_count}" if cfa_applied_count else ""
                logger.info(
                    f"  {metrics.model_name}: return={metrics.return_pct:+.2f}%, "
                    f"mutation={mutation_strength*100:.0f}%, "
                    f"biased={biased_count}/{len(biases)} params, "
                    f"watch_rules={rules_count}{cfa_tag}"
                )

            db.commit()
        finally:
            db.close()

        # Reset strategy internal state for fresh Session 2
        for strategy in self._models.values():
            strategy.reset()

        logger.info("=== Self-improvement complete. ===")

    async def _run_session(self, session_number: int) -> dict:
        """Run a single trading session.

        Duration is dynamically capped to finish before market close.
        Returns session summary dict.
        """
        self._session_number = session_number
        self.execution._session_number = session_number
        self._bar_count = 0
        self._adapt_bar_count = 0
        self._session_start_time = datetime.now(ET)
        session_minutes = self._compute_session_minutes(session_number)
        self._current_session_minutes = session_minutes

        logger.info(
            f"Session {session_number} starting at {self._session_start_time.strftime('%H:%M ET')} "
            f"({session_minutes} min)"
        )

        # Initialize tracker for this session
        models = list(self._model_records.values())
        self.tracker.initialize_models(models)
        self.tracker.set_session_number(session_number)
        self.tracker.set_session_date(self._session_date)

        # Snapshot params BEFORE any adapt() calls for adaptation drift logging
        self._session_start_params: dict[int, dict] = {
            mid: {**strategy.get_params()} for mid, strategy in self._models.items()
        }

        # Reconcile DB positions with Alpaca before trading
        if not self.simulate:
            recon = await asyncio.to_thread(
                self.execution.reconcile_positions_with_alpaca
            )
            if recon.get("zeroed") or recon.get("adjusted"):
                # Refresh model state after reconciliation fixed positions/capital
                for model_id, strategy in self._models.items():
                    self._refresh_model_state(model_id, strategy)
                models = list(self._model_records.values())

        # Reset position manager and regime detector for this session
        total_capital = sum(m.current_capital for m in models)
        self.position_manager.start_session(total_capital)
        self.regime_detector.reset()
        self.macro_regime.reset()
        self._recent_signals.clear()
        self._recent_fills.clear()

        # Reset daily limits for session 1 only
        if session_number == 1:
            self.execution.reset_daily_limits()

        # Ensure all symbols with open positions are in the trading universe.
        # The screener may rotate symbols, but we must keep receiving price data
        # for anything we hold — otherwise stop-losses/take-profits can't fire.
        positioned = self._get_positioned_symbols()
        existing = set(self.config.arena.symbols)
        missing = positioned - existing
        if missing:
            self.config.arena.symbols.extend(sorted(missing))
            logger.info(
                f"Added {len(missing)} position symbols to universe: "
                f"{sorted(missing)} (total: {len(self.config.arena.symbols)})"
            )

        # Record session in DB (replace stale record from a failed prior run)
        db = get_session(self.config.db_path)
        try:
            stale = (
                db.query(SessionRecord)
                .filter(
                    SessionRecord.session_date == self._session_date,
                    SessionRecord.session_number == session_number,
                )
                .first()
            )
            if stale:
                db.delete(stale)
                db.flush()

            session_record = SessionRecord(
                session_date=self._session_date,
                session_number=session_number,
                generation=models[0].generation if models else 1,
                started_at=datetime.utcnow(),
            )
            db.add(session_record)
            db.commit()
        finally:
            db.close()

        # Clear stale callbacks from prior sessions, then register fresh ones
        self.feed.clear_callbacks()
        if self.simulate:
            self.feed.on_bar(self._fan_out_bar)
        else:
            # Live mode: use batched processing to keep event loop responsive
            self.feed.on_bar_batch(self._fan_out_bars)
            # NOTE: old on_quote handler disabled — it did sync DB + order submission
            # on the event loop, causing deadlocks. The new QuoteAggregator.on_quote
            # is purely synchronous list.append(), so it's safe to register.
            self.feed.on_trade_update(self._on_trade_update)

            # Synthetic bars from quote aggregation
            synth_interval = self.config.arena.synthetic_bar_interval_sec
            if synth_interval > 0:
                self._quote_aggregator = QuoteAggregator(interval_sec=synth_interval)
                self._quote_aggregator.set_callback(self._fan_out_bars)
                self.feed.on_quote(self._quote_aggregator.on_quote)
                self.feed.subscribe_quotes(self.config.arena.symbols)
                logger.info(
                    f"Synthetic bars enabled: {synth_interval}s interval, "
                    f"{len(self.config.arena.symbols)} symbols"
                )

        total_expected = session_minutes  # rough: ~1 bar per minute
        self._set_status("session", f"Session {session_number} streaming", bar=0, total_bars=total_expected)
        logger.info(f"Streaming bars for {len(self.config.arena.symbols)} symbols")

        # Timer task stops streaming after session duration
        async def _session_timer():
            await asyncio.sleep(session_minutes * 60)
            logger.info(f"Session {session_number} time limit reached ({session_minutes} min)")
            await self.feed.stop_streaming()

        # Run initial health check before trading starts
        try:
            await self._health_check()
        except Exception:
            logger.exception("[HEALTH] Initial health check failed (non-fatal)")

        timer_task = asyncio.create_task(_session_timer())
        screener_task = asyncio.create_task(self._periodic_screener())
        health_task = asyncio.create_task(self._periodic_health_check())
        try:
            # Start aggregator timer before streaming so it's ready for quotes
            if self._quote_aggregator:
                self._quote_aggregator.start()
            # Directly await start_streaming (not via create_task) so the
            # Alpaca SDK's _run_forever runs in the top-level gather context.
            await self.feed.start_streaming(self.config.arena.symbols)
        except asyncio.CancelledError:
            logger.info(f"Session {session_number} cancelled")
        finally:
            # Stop aggregator before streams to prevent flush after stream close
            if self._quote_aggregator:
                self._quote_aggregator.stop()
                self._quote_aggregator = None
            # Let timer_task finish naturally (it called stop_streaming) to avoid
            # cancelling SDK close mid-flight, which leaves streams half-closed and
            # causes the second stop_streaming call below to hang (freeze bug #3).
            for task in (timer_task, screener_task, health_task):
                if not task.done():
                    try:
                        await asyncio.wait_for(asyncio.shield(task), timeout=10.0)
                    except asyncio.TimeoutError:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                    except asyncio.CancelledError:
                        pass
            # Ensure streams are stopped even on unexpected exit (no-op if timer
            # already stopped them, since stop_streaming clears refs before closing)
            await self.feed.stop_streaming()
        self._quote_subscribed.clear()
        self._clear_watches()

        # EOD liquidation: force-close remaining positions
        await self._end_session_liquidate()

        # Let Alpaca settle fills from close_all_positions() before reconciling
        if not self.simulate:
            await asyncio.sleep(3)
            recon = await asyncio.to_thread(
                self.execution.reconcile_fills_with_alpaca, self._session_date
            )
            logger.info(f"Post-session fill reconciliation: {recon}")

        # Query Alpaca account for real P&L (equity vs last_equity)
        alpaca_pnl = await self._get_alpaca_pnl()

        # Final performance update
        self.tracker.update_all()
        self.tracker.save_snapshots(self._session_date, session_number)

        summary = self.tracker.generate_session_summary()
        summary["session_number"] = session_number

        # Update session record
        db = get_session(self.config.db_path)
        try:
            session_record = (
                db.query(SessionRecord)
                .filter(
                    SessionRecord.session_date == self._session_date,
                    SessionRecord.session_number == session_number,
                )
                .first()
            )
            if session_record:
                session_record.ended_at = datetime.utcnow()
                session_record.total_bars = self._bar_count
                session_record.total_trades = sum(
                    m.total_trades
                    for m in self.tracker.get_leaderboard()
                )
                session_record.alpaca_pnl = alpaca_pnl
                session_record.summary = summary
                db.commit()
                if alpaca_pnl is not None:
                    logger.info(f"Session {session_number} Alpaca real P&L: ${alpaca_pnl:+.2f}")
        finally:
            db.close()

        logger.info(f"Session {session_number} ended. {self._bar_count} bars processed.")
        return summary

    def _reset_all_capital(self) -> None:
        """Reset all active models to their initial capital, clear positions and stale orders.

        Each model's current_capital is reset to its initial_capital (set by CFA
        allocation). Models without a per-model allocation fall back to the
        config default. For same-day resume (preserving capital + positions),
        use run_custom(resume=True) which skips this method entirely.
        """
        from src.core.database import Order, OrderStatus, Position
        db = get_session(self.config.db_path)
        try:
            models = db.query(TradingModel).filter(
                TradingModel.status == ModelStatus.ACTIVE
            ).all()
            default_capital = self.config.arena.initial_capital
            for m in models:
                # Respect per-model allocation set by CFA review.
                # Only fall back to config default if initial_capital is unset.
                if m.initial_capital and m.initial_capital > 0:
                    m.current_capital = m.initial_capital
                else:
                    m.current_capital = default_capital
                    m.initial_capital = default_capital
            # DELETE all position rows (not just zero qty) to prevent stale
            # rows from being overwritten with Alpaca's account-wide qty during
            # reconciliation, which causes phantom position duplication.
            deleted_count = db.query(Position).delete()
            if deleted_count:
                logger.info(f"Deleted {deleted_count} position rows (fresh start)")
            total = sum(m.initial_capital for m in models)
            logger.info(
                f"Reset {len(models)} models — total pool ${total:,.0f} "
                f"(range ${min(m.initial_capital for m in models):,.0f}–"
                f"${max(m.initial_capital for m in models):,.0f})"
            )

            # Always resolve stale PENDING orders from prior sessions
            stale = db.query(Order).filter(Order.status == OrderStatus.PENDING).update(
                {Order.status: OrderStatus.REJECTED, Order.rejected_reason: "stale (reset)"}
            )
            if stale:
                logger.info(f"Cleaned {stale} stale PENDING orders")
            db.commit()
        finally:
            db.close()

    def _prepare(self) -> None:
        """Common setup: init DB, set date, reset capital, load models."""
        init_db(self.config.db_path)
        self._session_date = datetime.now(ET).strftime("%Y-%m-%d")
        self._running = True
        self._daily_context.clear()  # fresh day context

        # Increase default thread pool to avoid exhaustion from DB + API calls
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=32))

        self._reset_all_capital()
        models = self._load_or_create_models()
        self._instantiate_strategies(models)

    def _clean_alpaca_state(self) -> None:
        """Cancel all open orders and close ALL positions on Alpaca before starting.

        Retries in a verify loop until Alpaca reports zero non-dust positions.
        This MUST succeed — leftover positions corrupt the next session.
        """
        import time as _time
        MAX_ATTEMPTS = 5

        try:
            # Always cancel pending orders first
            try:
                cancelled = self.execution.client.cancel_orders()
                if cancelled:
                    logger.info(f"Startup cleanup: cancelled {len(cancelled)} stale Alpaca orders")
                    _time.sleep(1)
            except Exception:
                logger.warning("Startup cleanup: failed to cancel orders", exc_info=True)

            for attempt in range(1, MAX_ATTEMPTS + 1):
                positions = self.execution.client.get_all_positions()
                real = [p for p in positions if abs(float(p.qty)) >= 0.001]
                if not real:
                    if attempt == 1 and not positions:
                        logger.info("Startup cleanup: Alpaca account is clean")
                    else:
                        logger.info(
                            f"Startup cleanup: all positions closed after "
                            f"{attempt} attempt(s)"
                        )
                    return

                logger.info(
                    f"Startup cleanup attempt {attempt}/{MAX_ATTEMPTS}: "
                    f"{len(real)} positions to close "
                    f"({', '.join(f'{p.symbol}={float(p.qty):.4f}' for p in real[:10])})"
                )

                # Try bulk close first
                try:
                    self.execution.client.close_all_positions(cancel_orders=True)
                    _time.sleep(3)
                except Exception:
                    logger.warning("Startup cleanup: close_all_positions failed", exc_info=True)

                # Individually close anything that survived
                remaining = self.execution.client.get_all_positions()
                for p in remaining:
                    qty = abs(float(p.qty))
                    if qty < 0.001:
                        continue
                    try:
                        self.execution.client.close_position(p.symbol)
                        logger.info(f"Startup cleanup: individually closed {p.symbol} qty={qty:.6f}")
                    except Exception:
                        logger.warning(
                            f"Startup cleanup: failed to close {p.symbol} qty={qty:.6f}",
                            exc_info=True,
                        )
                _time.sleep(2)

            # Final check after all attempts
            final = self.execution.client.get_all_positions()
            final_real = [p for p in final if abs(float(p.qty)) >= 0.001]
            if final_real:
                symbols = [f"{p.symbol}={float(p.qty):.4f}" for p in final_real]
                logger.error(
                    f"STARTUP CLEANUP FAILED: {len(final_real)} positions still open "
                    f"after {MAX_ATTEMPTS} attempts: {symbols}. "
                    f"These WILL interfere with trading."
                )
            else:
                logger.info(f"Startup cleanup: all positions closed after {MAX_ATTEMPTS} attempts")
        except Exception as e:
            logger.error(f"Startup cleanup CRITICAL error: {e}", exc_info=True)

    async def run(self) -> None:
        """Run the full trading day:
        Pre-market warmup -> Session 1 -> Break -> Session 2 -> Evolution

        If started before market open, runs pre-market warmup (historical +
        live pre-market bar polling) then waits for 9:30 AM ET to begin Session 1.
        """
        logger.info("Arena starting full day...")
        self._set_status("starting", "Initializing arena")

        self._prepare()

        # Pre-market warmup or wait for market open
        now_et = datetime.now(ET)
        if not self.simulate and now_et.time() < MARKET_OPEN:
            self._set_status("premarket", "Pre-market warmup in progress")
            await self._premarket_warmup()

        if not self._is_market_hours() and self.simulate:
            logger.warning(
                "Outside market hours. Running in simulation mode."
            )

        try:
            # === SESSION 1 ===
            self._set_status("session", "Session 1 starting")
            summary_1 = await self._run_session(1)
            self._print_summary(summary_1, "SESSION 1")
            self._set_status("reflecting", "Generating S1 reflections")
            self._generate_session_summaries(1)

            # === BREAK: Evaluate + Self-improvement ===
            if self.config.arena.self_improve_enabled:
                self._set_status("improving", "Evaluating mutations & self-improving")
                self._evaluate_pending_mutations()
                param_snapshots = {
                    mid: {**strategy.get_params(), "_watch_rules": list(strategy._watch_rules)}
                    for mid, strategy in self._models.items()
                }
                session_start_params_1 = dict(self._session_start_params)
                self._self_improve()
                self._generate_improvement_summaries(param_snapshots, session_start_params=session_start_params_1)
            else:
                logger.info("Self-improvement disabled (manual only)")

            # Brief pause
            break_seconds = self.config.arena.break_minutes * 60
            self._set_status("break", f"Break ({self.config.arena.break_minutes} min)")
            logger.info(f"Break: {self.config.arena.break_minutes} minutes")
            await asyncio.sleep(min(break_seconds, 5))

            # === SESSION 2 ===
            self._set_status("session", "Session 2 starting")
            summary_2 = await self._run_session(2)
            self._print_summary(summary_2, "SESSION 2")
            self._set_status("reflecting", "Generating S2 reflections")
            self._generate_session_summaries(2)

            # === DAY TOTAL ===
            self._set_status("complete", "Day complete")
            day_summary = self._compute_day_total(summary_1, summary_2)
            self._print_summary(day_summary, "DAY TOTAL")

            # Save daily ledger
            self._save_daily_ledger()

            # CFA review first, then CFA-guided self-improvement
            cfa_recs = await self._run_cfa_review_with_recommendations()

            if self.config.arena.self_improve_enabled:
                self._set_status("improving", "CFA-guided post-S2 self-improvement")
                self._evaluate_pending_mutations()
                param_snapshots_2 = {
                    mid: {**strategy.get_params(), "_watch_rules": list(strategy._watch_rules)}
                    for mid, strategy in self._models.items()
                }
                session_start_params_2 = dict(self._session_start_params)
                self._self_improve(cfa_recommendations=cfa_recs)
                self._generate_improvement_summaries(param_snapshots_2, session_number=2, session_start_params=session_start_params_2)

        except KeyboardInterrupt:
            logger.info("Arena interrupted by user")
        finally:
            self._running = False

    async def run_custom(
        self,
        num_sessions: int,
        session_minutes: int,
        resume: bool = False,
        skip_cfa: bool = False,
    ) -> None:
        """Run N sessions of configurable duration with reflection between each.

        Args:
            num_sessions: Number of sessions to run (1-20).
            session_minutes: Duration of each session in minutes (5-390).
            resume: If True, skip capital reset and position cleanup; restore
                    model state from DB so models continue with existing positions.
            skip_cfa: If True, skip the post-session CFA review.
        """
        self._skip_cfa = skip_cfa
        logger.info(
            f"Arena starting custom run: {num_sessions} sessions x "
            f"{session_minutes} min each"
            + (" (RESUME)" if resume else "")
        )
        self._set_status("starting", "Initializing arena (custom run)")

        if resume:
            # Resume mode: init DB, set date, load models WITHOUT resetting capital
            init_db(self.config.db_path)
            self._session_date = datetime.now(ET).strftime("%Y-%m-%d")
            self._running = True
            loop = asyncio.get_running_loop()
            loop.set_default_executor(ThreadPoolExecutor(max_workers=32))

            # Clean stale PENDING orders but preserve capital + positions
            from src.core.database import Order, OrderStatus
            db = get_session(self.config.db_path)
            try:
                stale = db.query(Order).filter(
                    Order.status == OrderStatus.PENDING
                ).update({
                    Order.status: OrderStatus.REJECTED,
                    Order.rejected_reason: "stale (resume)",
                })
                if stale:
                    logger.info(f"Resume: cleaned {stale} stale PENDING orders")
                db.commit()
            finally:
                db.close()

            models = self._load_or_create_models()
            self._instantiate_strategies(models)

            # Restore positions + capital from DB as-is (DB is authoritative
            # for same-day resume — reconciliation would incorrectly credit
            # capital back for positions sold by other models in the shared
            # Alpaca account)
            for model_id, strategy in self._models.items():
                self._refresh_model_state(model_id, strategy)
            # Log restored state
            total_positions = sum(
                len(s._positions) for s in self._models.values()
            )
            logger.info(
                f"Resume: restored {len(self._models)} models with "
                f"{total_positions} open positions from DB"
            )
        else:
            self._prepare()

            # Clean up stale Alpaca state before starting
            await asyncio.to_thread(self._clean_alpaca_state)

        # Pre-market warmup (historical + live SIP polling) then wait for 9:30
        now_et = datetime.now(ET)
        if now_et.time() < MARKET_OPEN:
            self._set_status("premarket", "Pre-market warmup in progress")
            await self._premarket_warmup()
        else:
            self._run_screener()
            self._warmup_strategies()
            self._seed_daily_context_from_warmup()

        # Override configured session durations
        original_s1 = self.config.arena.session_1_minutes
        original_s2 = self.config.arena.session_2_minutes
        self.config.arena.session_1_minutes = session_minutes
        self.config.arena.session_2_minutes = session_minutes

        summaries: list[dict] = []
        try:
            for i in range(num_sessions):
                session_number = i + 1
                self._set_status(
                    "session",
                    f"Session {session_number}/{num_sessions} starting",
                )

                summary = await self._run_session(session_number)
                summaries.append(summary)
                self._print_summary(summary, f"SESSION {session_number}")

                self._set_status(
                    "reflecting",
                    f"Generating S{session_number} reflections",
                )
                self._generate_session_summaries(session_number)
                self._save_daily_ledger()  # Save BEFORE self-improvement can crash

                is_last_session = session_number == num_sessions

                # Self-improve between sessions (skip last — CFA-guided improve runs after loop)
                if self.config.arena.self_improve_enabled and not is_last_session:
                    self._set_status(
                        "improving",
                        f"Self-improvement after S{session_number}",
                    )
                    self._evaluate_pending_mutations()
                    param_snapshots = {
                        mid: {
                            **strategy.get_params(),
                            "_watch_rules": list(strategy._watch_rules),
                        }
                        for mid, strategy in self._models.items()
                    }
                    session_start_params = dict(self._session_start_params)
                    self._self_improve()
                    self._generate_improvement_summaries(
                        param_snapshots, session_number=session_number,
                        session_start_params=session_start_params,
                    )
                elif not self.config.arena.self_improve_enabled:
                    logger.info(f"Self-improvement disabled after S{session_number}")

                # Brief break between sessions (skip after last)
                if not is_last_session:
                    break_seconds = self.config.arena.break_minutes * 60
                    self._set_status(
                        "break",
                        f"Break before S{session_number + 1}/{num_sessions}",
                    )
                    await asyncio.sleep(min(break_seconds, 5))

            # Day total: combine all session summaries
            self._set_status("complete", f"All {num_sessions} sessions complete")
            if len(summaries) >= 2:
                day_total = self._compute_day_total(summaries[0], summaries[-1])
                self._print_summary(day_total, "DAY TOTAL")
            elif len(summaries) == 1:
                self._print_summary(summaries[0], "DAY TOTAL")

            self._save_daily_ledger()

            # CFA review first, then CFA-guided self-improvement
            cfa_recs: dict[str, dict] = {}
            if self._skip_cfa:
                logger.info("CFA review skipped (skip_cfa=True)")
            else:
                cfa_recs = await self._run_cfa_review_with_recommendations()

            if self.config.arena.self_improve_enabled:
                self._set_status("improving", "CFA-guided self-improvement (post-day)")
                self._evaluate_pending_mutations()
                param_snapshots = {
                    mid: {
                        **strategy.get_params(),
                        "_watch_rules": list(strategy._watch_rules),
                    }
                    for mid, strategy in self._models.items()
                }
                session_start_params = dict(self._session_start_params)
                self._self_improve(cfa_recommendations=cfa_recs)
                self._generate_improvement_summaries(
                    param_snapshots, session_number=num_sessions,
                    session_start_params=session_start_params,
                )

        except asyncio.CancelledError:
            logger.info("Arena custom run cancelled — liquidating positions")
            await self._end_session_liquidate()
            self._set_status("complete", "Run cancelled")
            raise
        except KeyboardInterrupt:
            logger.info("Arena interrupted — liquidating positions")
            await self._end_session_liquidate()
        finally:
            self.config.arena.session_1_minutes = original_s1
            self.config.arena.session_2_minutes = original_s2
            self._running = False

    async def _connectivity_check(self, symbols: list[str], timeout: int = 30) -> bool:
        """Quick WebSocket connectivity test. Returns True if at least one bar received."""
        logger.info(f"Connectivity check: connecting to Alpaca for {symbols} (timeout={timeout}s)...")
        from alpaca.data.live import StockDataStream
        from alpaca.data.enums import DataFeed

        received = asyncio.Event()
        bar_info: dict = {}

        async def _on_bar(bar):
            bar_info["symbol"] = bar.symbol
            bar_info["close"] = float(bar.close)
            bar_info["timestamp"] = str(bar.timestamp)
            received.set()

        test_stream = StockDataStream(
            api_key=self.config.alpaca.api_key,
            secret_key=self.config.alpaca.secret_key,
            feed=DataFeed.SIP,
        )
        test_stream.subscribe_bars(_on_bar, *symbols)

        async def _run():
            await test_stream._run_forever()

        task = asyncio.create_task(_run())
        try:
            await asyncio.wait_for(received.wait(), timeout=timeout)
            logger.info(
                f"Connectivity OK: received bar for {bar_info.get('symbol')} "
                f"close={bar_info.get('close')} @ {bar_info.get('timestamp')}"
            )
            return True
        except asyncio.TimeoutError:
            logger.warning(
                f"Connectivity check: no bars received in {timeout}s. "
                f"This may be normal outside market hours or during low-volume periods."
            )
            return False
        finally:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            try:
                await test_stream.close()
            except Exception:
                pass

    async def run_pipeline_test(self) -> dict:
        """Run a 3-minute micro-session with 3 models on 2 symbols.

        Validates the full pipeline: connect -> receive bars -> signal ->
        submit -> fill confirmation -> quote monitoring -> exit.
        Logs every API call, fill, quote, and timing for diagnostics.
        """
        test_duration_min = 3
        logger.info("=" * 60)
        logger.info(f"  PIPELINE TEST — {test_duration_min} minute micro-session")
        logger.info("=" * 60)

        self._set_status("pipeline-test", "Initializing pipeline test")

        # Use a subset of symbols and models
        original_symbols = self.config.arena.symbols
        self.config.arena.symbols = original_symbols[:2]  # 2 symbols only

        init_db(self.config.db_path)
        self._session_date = datetime.now(ET).strftime("%Y-%m-%d")
        self._running = True

        # Reset all models to initial capital
        self._reset_all_capital()

        # Step 0: Connectivity pre-check with a raw WebSocket
        logger.info("Step 0: Connectivity pre-check...")
        bars_available = await self._connectivity_check(self.config.arena.symbols, timeout=45)
        if not bars_available:
            logger.warning(
                "No bars during connectivity check. Continuing anyway — "
                "bars may arrive once the session starts."
            )

        # Load models but only use first 3
        logger.info("Step 1: Loading models...")
        models = self._load_or_create_models()
        models = models[:3]
        self._models.clear()
        self._model_records.clear()
        for model in models:
            strategy = create_strategy(
                model.strategy_type, model.name, params=model.parameters,
            )
            strategy.current_capital = model.current_capital
            strategy.initial_capital = model.initial_capital
            self._models[model.id] = strategy
            self._model_records[model.id] = model

        logger.info(f"Pipeline test: {len(self._models)} models, symbols={self.config.arena.symbols}")

        # Quick warmup
        logger.info("Step 2: Warmup...")
        self._warmup_strategies()
        self._seed_daily_context_from_warmup()

        # Run a micro-session
        logger.info(f"Step 3: Starting {test_duration_min}-min streaming session...")
        self._session_number = 1
        self.execution._session_number = 1
        self._bar_count = 0
        self._adapt_bar_count = 0
        self._session_start_time = datetime.now(ET)
        self._current_session_minutes = test_duration_min

        test_models = list(self._model_records.values())
        self.tracker.initialize_models(test_models)
        self.tracker.set_session_number(1)
        self.execution.reset_daily_limits()

        # Init position manager and regime detector
        total_capital = sum(m.current_capital for m in test_models)
        self.position_manager.start_session(total_capital)
        self.regime_detector.reset()
        self.macro_regime.reset()
        self._recent_signals.clear()
        self._recent_fills.clear()

        self.feed.clear_callbacks()
        if self.simulate:
            self.feed.on_bar(self._fan_out_bar)
        else:
            self.feed.on_bar_batch(self._fan_out_bars)
            self.feed.on_trade_update(self._on_trade_update)

            # Synthetic bars from quote aggregation (pipeline test)
            synth_interval = self.config.arena.synthetic_bar_interval_sec
            if synth_interval > 0:
                self._quote_aggregator = QuoteAggregator(interval_sec=synth_interval)
                self._quote_aggregator.set_callback(self._fan_out_bars)
                self.feed.on_quote(self._quote_aggregator.on_quote)
                self.feed.subscribe_quotes(self.config.arena.symbols)
                logger.info(
                    f"Synthetic bars enabled (pipeline test): {synth_interval}s interval, "
                    f"{len(self.config.arena.symbols)} symbols"
                )

        self._set_status("pipeline-test", f"Streaming ({test_duration_min} min)")
        logger.info(f"Pipeline test: streaming for {test_duration_min} minutes...")

        # Progress monitor logs bar count every 30s
        async def _progress_monitor():
            start = datetime.now(ET)
            while self._running:
                await asyncio.sleep(30)
                elapsed = (datetime.now(ET) - start).total_seconds()
                logger.info(
                    f"Pipeline progress: {self._bar_count} bars in {elapsed:.0f}s "
                    f"(feed: {self.feed._bars_received} raw, state: {self.feed._connection_state})"
                )
                if elapsed > 90 and self._bar_count == 0:
                    logger.warning(
                        "WARNING: No bars after 90s! Check: "
                        "1) Market/pre-market hours? "
                        "2) SIP feed access? "
                        "3) Symbol availability?"
                    )

        async def _pipeline_timer():
            await asyncio.sleep(test_duration_min * 60)
            logger.info(f"Pipeline test: {test_duration_min}-minute window complete")
            await self.feed.stop_streaming()

        timer_task = asyncio.create_task(_pipeline_timer())
        monitor_task = asyncio.create_task(_progress_monitor())
        try:
            if self._quote_aggregator:
                self._quote_aggregator.start()
            await self.feed.start_streaming(self.config.arena.symbols)
        except asyncio.CancelledError:
            logger.info("Pipeline test: cancelled")
        finally:
            if self._quote_aggregator:
                self._quote_aggregator.stop()
                self._quote_aggregator = None
            for t in [timer_task, monitor_task]:
                if not t.done():
                    t.cancel()
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass
            await self.feed.stop_streaming()
        self._quote_subscribed.clear()
        self._clear_watches()

        # Liquidate and summarize
        await self._end_session_liquidate()
        self.tracker.update_all()

        summary = self.tracker.generate_session_summary()
        summary["session_number"] = "pipeline_test"

        self._print_summary(summary, "PIPELINE TEST")

        # Print diagnostics
        logger.info("=" * 60)
        logger.info("  PIPELINE TEST RESULTS")
        logger.info(f"  Bars received (arena): {self._bar_count}")
        logger.info(f"  Bars received (feed):  {self.feed._bars_received}")
        logger.info(f"  Symbols: {self.config.arena.symbols}")
        logger.info(f"  Models tested: {len(self._models)}")
        total_trades = sum(
            m.total_trades for m in self.tracker.get_leaderboard()
        )
        logger.info(f"  Total trades: {total_trades}")
        if self._bar_count == 0:
            logger.error(
                "  FAIL: Zero bars received. The streaming pipeline is broken. "
                "Check logs above for connection errors."
            )
        else:
            logger.info("  PASS: Pipeline is functional.")
        logger.info("=" * 60)

        # Restore symbols
        self.config.arena.symbols = original_symbols
        self._running = False
        return summary

    async def run_session_only(self, session_number: int) -> dict:
        """Run a single session independently (for manual control).

        Handles setup, warmup, session, liquidation, summaries, and ledger.
        Returns session summary.
        """
        logger.info(f"Arena starting session {session_number} only...")

        if not self._is_market_hours():
            logger.warning("Outside market hours.")

        self._prepare()
        self._run_screener()
        self._warmup_strategies()
        self._seed_daily_context_from_warmup()

        try:
            summary = await self._run_session(session_number)
            self._print_summary(summary, f"SESSION {session_number}")
            self._generate_session_summaries(session_number)

            if session_number == 2:
                self._save_daily_ledger()

            return summary
        except KeyboardInterrupt:
            logger.info("Arena interrupted by user")
            return {}
        finally:
            self._running = False

    def _compute_day_total(self, s1: dict, s2: dict) -> dict:
        """Combine Session 1 and Session 2 results into a day total."""
        # Merge rankings by model_id, summing profits
        model_totals: dict[int, dict] = {}

        for r in s1.get("rankings", []):
            mid = r["model_id"]
            model_totals[mid] = {
                "model_id": mid,
                "name": r["name"],
                "strategy_type": r["strategy_type"],
                "s1_return": r["return_pct"],
                "s2_return": 0.0,
                "return_pct": r["return_pct"],
                "sharpe": r["sharpe"],
                "trades": r["trades"],
                "fitness": r["fitness"],
            }

        for r in s2.get("rankings", []):
            mid = r["model_id"]
            if mid in model_totals:
                model_totals[mid]["s2_return"] = r["return_pct"]
                model_totals[mid]["return_pct"] = (
                    model_totals[mid]["s1_return"] + r["return_pct"]
                )
                model_totals[mid]["trades"] += r["trades"]
                model_totals[mid]["sharpe"] = r["sharpe"]  # use latest
                model_totals[mid]["fitness"] = r["fitness"]
            else:
                model_totals[mid] = {
                    "model_id": mid,
                    "name": r["name"],
                    "strategy_type": r["strategy_type"],
                    "s1_return": 0.0,
                    "s2_return": r["return_pct"],
                    "return_pct": r["return_pct"],
                    "sharpe": r["sharpe"],
                    "trades": r["trades"],
                    "fitness": r["fitness"],
                }

        # Sort by total return (profit is king)
        rankings = sorted(model_totals.values(), key=lambda x: x["return_pct"], reverse=True)
        for i, r in enumerate(rankings):
            r["rank"] = i + 1

        return {
            "model_count": len(rankings),
            "rankings": rankings,
            "best_model": rankings[0]["name"] if rankings else None,
            "worst_model": rankings[-1]["name"] if rankings else None,
            "session_number": "day_total",
        }

    def _get_prior_session_summaries(self, prior_session: int) -> dict[int, dict]:
        """Fetch post-session and post-improvement summaries for a prior session today.

        Args:
            prior_session: The session number to fetch data for.

        Returns {model_id: {prev_return, prev_rank, prev_session, mutations: {param: {old, new, pct}}}}.
        """
        result: dict[int, dict] = {}
        db = get_session(self.config.db_path)
        try:
            # Prior session post-session summaries
            prev_rows = (
                db.query(ModelSummary)
                .filter(
                    ModelSummary.session_date == self._session_date,
                    ModelSummary.session_number == prior_session,
                    ModelSummary.summary_type == "post_session",
                )
                .all()
            )
            for row in prev_rows:
                result[row.model_id] = {
                    "prev_return": row.return_pct,
                    "prev_rank": row.rank,
                    "prev_session": prior_session,
                    "mutations": {},
                }

            # Prior session post-improvement summaries
            imp_rows = (
                db.query(ModelSummary)
                .filter(
                    ModelSummary.session_date == self._session_date,
                    ModelSummary.session_number == prior_session,
                    ModelSummary.summary_type == "post_improvement",
                )
                .all()
            )
            for row in imp_rows:
                if row.model_id in result and row.param_changes:
                    result[row.model_id]["mutations"] = row.param_changes
        finally:
            db.close()
        return result

    def _generate_session_summaries(self, session_number: int) -> None:
        """Generate per-model reflections after a session.

        For sessions after the first, includes prior session context:
        previous performance, mutations applied, and whether those mutations helped.
        """
        leaderboard = self.tracker.get_leaderboard()
        if not leaderboard:
            return

        total_models = len(leaderboard)
        summaries: list[tuple[str, dict]] = []

        # For sessions after the first, load prior session data for context
        prior_data: dict[int, dict] = {}
        if session_number > 1:
            prior_data = self._get_prior_session_summaries(session_number - 1)

        for metrics in leaderboard:
            model_id = metrics.model_id
            name = metrics.model_name
            stype = metrics.strategy_type

            # Determine performance tier
            if metrics.rank <= max(1, total_models // 4):
                tier = "top"
            elif metrics.rank <= total_models // 2:
                tier = "mid-upper"
            elif metrics.rank <= total_models * 3 // 4:
                tier = "mid-lower"
            else:
                tier = "bottom"

            # Build reflection
            lines = [f"[{name}] ({stype}) — Session {session_number} Reflection"]

            lines.append(f"  Rank: {metrics.rank}/{total_models} | Return: {metrics.return_pct:+.3f}%")
            lines.append(f"  Sharpe: {metrics.sharpe_ratio:.3f} | Drawdown: {metrics.max_drawdown:.4f} | Trades: {metrics.total_trades} | Win rate: {metrics.win_rate:.1%}")
            if metrics.fitness:
                lines.append(f"  Fitness: {metrics.fitness.composite:.4f} (profit={metrics.fitness.profit_component:.3f}, sharpe={metrics.fitness.sharpe_component:.3f}, dd={metrics.fitness.drawdown_component:.3f}, freq={metrics.fitness.frequency_component:.3f})")

            # Prior session context
            if session_number > 1 and model_id in prior_data:
                prev = prior_data[model_id]
                prev_ret = prev.get("prev_return", 0.0) or 0.0
                prev_rank = prev.get("prev_rank")
                prev_sess = prev.get("prev_session", session_number - 1)
                mutations = prev.get("mutations", {})

                lines.append(f"  Session {prev_sess}: Return {prev_ret:+.3f}%, Rank {prev_rank}/{total_models}")

                if mutations:
                    mut_parts = []
                    for param, delta in sorted(mutations.items()):
                        direction = "up" if delta.get("pct", 0) > 0 else "down"
                        mut_parts.append(f"{param} {delta.get('old')}→{delta.get('new')} ({direction})")
                    lines.append(f"  Mutations applied: {'; '.join(mut_parts[:5])}")
                    if len(mut_parts) > 5:
                        lines.append(f"    ...and {len(mut_parts) - 5} more")
                else:
                    lines.append("  Mutations applied: None")

                lines.append(f"  Session {session_number}: Return {metrics.return_pct:+.3f}%, Rank {metrics.rank}/{total_models}")

                # Did mutations help?
                delta_return = metrics.return_pct - prev_ret
                rank_change = (prev_rank or metrics.rank) - metrics.rank
                if delta_return > 0.1:
                    lines.append(f"  Assessment: Mutations appear to have helped ({delta_return:+.3f}% improvement, rank {'improved by ' + str(rank_change) if rank_change > 0 else 'unchanged'}).")
                elif delta_return < -0.1:
                    lines.append(f"  Assessment: Mutations may have hurt ({delta_return:+.3f}% decline, rank {'dropped by ' + str(-rank_change) if rank_change < 0 else 'unchanged'}).")
                else:
                    lines.append(f"  Assessment: Performance roughly stable across sessions ({delta_return:+.3f}% change).")

            # What happened
            if metrics.total_trades == 0:
                lines.append("  What happened: Did not trade this session. Signals may not have triggered, or market conditions didn't match entry criteria.")
            elif metrics.return_pct > 0:
                lines.append(f"  What happened: Profitable session with {metrics.total_trades} trades. Captured gains in favorable conditions.")
            else:
                lines.append(f"  What happened: Unprofitable session with {metrics.total_trades} trades. Entries were poorly timed or exits were too late.")

            # Why this performance
            reasons = []
            if metrics.total_trades == 0:
                reasons.append("no trades placed — parameters may be too conservative or lookback too long for session length")
            else:
                if metrics.win_rate >= 0.6:
                    reasons.append(f"strong win rate ({metrics.win_rate:.0%}) shows good entry timing")
                elif metrics.win_rate <= 0.35 and metrics.total_trades >= 2:
                    reasons.append(f"low win rate ({metrics.win_rate:.0%}) suggests entry criteria need refinement")

                if metrics.sharpe_ratio > 1.0:
                    reasons.append("high risk-adjusted returns — consistent profitability")
                elif metrics.sharpe_ratio < -0.5:
                    reasons.append("negative Sharpe — returns don't justify the volatility taken on")

                if metrics.max_drawdown > 0.05:
                    reasons.append(f"significant drawdown ({metrics.max_drawdown:.1%}) indicates poor loss management")
                elif metrics.max_drawdown < 0.01 and metrics.total_trades > 0:
                    reasons.append("tight drawdown control — disciplined risk management")

            if reasons:
                lines.append(f"  Why: {'; '.join(reasons)}.")

            # Outlook
            if tier == "top":
                lines.append("  Outlook: Performing well. Fine-tuning should preserve edge while reducing risk.")
            elif tier in ("mid-upper", "mid-lower"):
                lines.append("  Outlook: Middle of the pack. Moderate parameter adjustments could help find an edge.")
            else:
                lines.append("  Outlook: Underperforming. Needs meaningful parameter shifts to become competitive.")

            # Stakes — models know the rules but NOT their current rank
            lines.append("  Stakes: Top 5 weekly performers participate in COLLAB voting. Bottom 2 at week's end are eliminated and replaced.")

            # Available capabilities
            lines.append("  Available tools: You can add stop_loss_pct and take_profit_pct parameters to enable automatic risk management. Buy signals with these set become bracket orders — Alpaca enforces stop-loss and take-profit server-side, even between bars. You can also override on_quote() for custom real-time exit logic.")

            reflection = "\n".join(lines)
            summaries.append((reflection, {
                "model_id": model_id,
                "session_date": self._session_date,
                "session_number": session_number,
                "summary_type": "post_session",
                "return_pct": metrics.return_pct,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown": metrics.max_drawdown,
                "total_trades": metrics.total_trades,
                "win_rate": metrics.win_rate,
                "fitness": metrics.fitness.composite if metrics.fitness else None,
                "rank": metrics.rank,
                "reflection": reflection,
            }))

        self._save_and_print_summaries(summaries, f"SESSION {session_number} MODEL REFLECTIONS")

    def _generate_improvement_summaries(
        self,
        param_snapshots: dict[int, dict],
        session_number: int = 1,
        session_start_params: dict[int, dict] | None = None,
    ) -> None:
        """Generate per-model reflections after self-improvement.

        Args:
            param_snapshots: {model_id: old_params} captured post-adapt, pre-mutate.
            session_number: which session this improvement follows (1 or 2).
            session_start_params: {model_id: params} captured at session start, before adapt().
        """
        leaderboard = self.tracker.get_leaderboard()
        if not leaderboard:
            return

        summaries: list[tuple[str, dict]] = []

        for metrics in leaderboard:
            model_id = metrics.model_id
            name = metrics.model_name
            strategy = self._models.get(model_id)
            if not strategy:
                continue

            new_params = strategy.get_params()
            old_params = param_snapshots.get(model_id, {})

            # Compute mutation changes (post-adapt -> post-mutate)
            changes = {}
            for key in set(list(old_params.keys()) + list(new_params.keys())):
                old_val = old_params.get(key)
                new_val = new_params.get(key)
                if old_val != new_val and isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                    pct_change = ((new_val - old_val) / abs(old_val) * 100) if old_val != 0 else 0
                    changes[key] = {"old": old_val, "new": new_val, "pct": round(pct_change, 1)}

            # Compute adaptation drift (session-start -> post-adapt)
            adapt_changes = {}
            start_params = (session_start_params or {}).get(model_id, {})
            if start_params:
                for key in set(list(start_params.keys()) + list(old_params.keys())):
                    start_val = start_params.get(key)
                    adapted_val = old_params.get(key)
                    if (
                        start_val != adapted_val
                        and isinstance(start_val, (int, float))
                        and isinstance(adapted_val, (int, float))
                    ):
                        pct = ((adapted_val - start_val) / abs(start_val) * 100) if start_val != 0 else 0
                        adapt_changes[key] = {"old": start_val, "new": adapted_val, "pct": round(pct, 1)}

            # Compute net change (session-start -> post-mutate)
            net_changes = {}
            if start_params:
                for key in set(list(start_params.keys()) + list(new_params.keys())):
                    start_val = start_params.get(key)
                    final_val = new_params.get(key)
                    if (
                        start_val != final_val
                        and isinstance(start_val, (int, float))
                        and isinstance(final_val, (int, float))
                    ):
                        pct = ((final_val - start_val) / abs(start_val) * 100) if start_val != 0 else 0
                        net_changes[key] = {"old": start_val, "new": final_val, "pct": round(pct, 1)}

            # Determine mutation strength applied
            if metrics.return_pct < 0:
                mutation_label = "aggressive (10%)"
            elif metrics.return_pct < 1.0:
                mutation_label = "moderate (5%)"
            else:
                mutation_label = "fine-tune (2%)"

            adapt_cycles = self._adapt_bar_count // self._adapt_interval if self._adapt_interval else 0
            lines = [f"[{name}] — Post-S{session_number} Improvement"]
            lines.append(f"  S{session_number} return: {metrics.return_pct:+.3f}% | Mutation strength: {mutation_label}")

            # Adaptation drift section
            if adapt_changes:
                lines.append(f"  Adaptation drift ({adapt_cycles} adapt cycles):")
                for param, delta in sorted(adapt_changes.items()):
                    direction = "up" if delta["pct"] > 0 else "down"
                    lines.append(f"    {param}: {delta['old']} -> {delta['new']} ({direction} {abs(delta['pct']):.1f}%)")
            elif start_params:
                lines.append(f"  Adaptation drift ({adapt_cycles} adapt cycles): no params changed")

            # Mutation section
            if not changes:
                lines.append("  Mutation: No parameters were mutated this round.")
            else:
                # Load mutation memory for bias info
                db = get_session(self.config.db_path)
                try:
                    db_model = db.query(TradingModel).get(model_id)
                    memory = db_model.mutation_memory if db_model else None
                finally:
                    db.close()

                lines.append(f"  Mutation ({len(changes)} params):")
                for param, delta in sorted(changes.items()):
                    direction = "up" if delta["pct"] > 0 else "down"
                    obs = MutationMemory.get_observation_count(memory, param) if memory else 0
                    bias = MutationMemory.compute_bias(memory.get(param, {})) if (memory and param in memory) else 0.0
                    if obs >= 2:
                        bias_label = f" [bias: {bias:+.2f} from {obs} obs]"
                    else:
                        bias_label = " [no history]"
                    lines.append(f"    {param}: {delta['old']} -> {delta['new']} ({direction} {abs(delta['pct']):.1f}%){bias_label}")

            # Net change section
            if net_changes:
                lines.append(f"  Net change (start -> final, {len(net_changes)} params):")
                for param, delta in sorted(net_changes.items()):
                    direction = "up" if delta["pct"] > 0 else "down"
                    lines.append(f"    {param}: {delta['old']} -> {delta['new']} ({direction} {abs(delta['pct']):.1f}%)")

            # Rationale
            if not changes and not adapt_changes:
                lines.append(f"  Rationale: No changes. Will trade S{session_number + 1} with same config.")
            elif metrics.return_pct < 0:
                lines.append(f"  Rationale: Lost money in S{session_number}, so applying larger mutations to escape a losing parameter region.")
            elif metrics.return_pct < 1.0:
                lines.append(f"  Rationale: Modest S{session_number} performance. Moderate tweaks to nudge toward better entries/exits without losing what works.")
            else:
                lines.append(f"  Rationale: Strong S{session_number}. Minimal fine-tuning to preserve the winning edge while exploring nearby improvements.")

            # Watch rules summary
            watch_rules = strategy._watch_rules if strategy else []
            old_rules = old_params.get("_watch_rules", [])
            if watch_rules:
                lines.append(f"  Watch rules ({len(watch_rules)}):")
                for wr in watch_rules:
                    reason = wr.get("reason", "unnamed rule")
                    ww = wr.get("watch_when", {})
                    lines.append(f"    - {reason} ({ww.get('indicator', '?')} {ww.get('op', '?')} {ww.get('value', '?')})")
                if not old_rules:
                    lines.append("    [new — first LLM-generated rules]")
                elif old_rules != watch_rules:
                    lines.append("    [updated by LLM]")
            elif old_rules:
                lines.append("  Watch rules: cleared (LLM returned no valid rules)")

            reflection = "\n".join(lines)
            summaries.append((reflection, {
                "model_id": model_id,
                "session_date": self._session_date,
                "session_number": session_number,
                "summary_type": "post_improvement",
                "return_pct": metrics.return_pct,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown": metrics.max_drawdown,
                "total_trades": metrics.total_trades,
                "win_rate": metrics.win_rate,
                "fitness": metrics.fitness.composite if metrics.fitness else None,
                "rank": metrics.rank,
                "param_changes": changes if changes else None,
                "reflection": reflection,
            }))

        self._save_and_print_summaries(summaries, "POST-IMPROVEMENT MODEL REFLECTIONS")

    def _save_and_print_summaries(
        self, summaries: list[tuple[str, dict]], label: str
    ) -> None:
        """Print summaries to console, save to DB, and append to log file."""
        # Print to console
        print(f"\n{'~' * 70}")
        print(f"  {label}")
        print(f"{'~' * 70}")
        for reflection, _ in summaries:
            print(reflection)
            print()
        print(f"{'~' * 70}")

        # Save to DB
        db = get_session(self.config.db_path)
        try:
            for _, kwargs in summaries:
                db.add(ModelSummary(**kwargs))
            db.commit()
        except Exception:
            db.rollback()
            logger.exception("Failed to save model summaries to DB")
        finally:
            db.close()

        # Append to log file
        log_dir = os.path.join("logs", "summaries")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{self._session_date}.log")
        try:
            with open(log_file, "a") as f:
                f.write(f"\n{'=' * 70}\n")
                f.write(f"  {label} — {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}\n")
                f.write(f"{'=' * 70}\n\n")
                for reflection, _ in summaries:
                    f.write(reflection + "\n\n")
            logger.info(f"Summaries written to {log_file}")
        except Exception:
            logger.exception(f"Failed to write summaries to {log_file}")

    def _print_summary(self, summary: dict, label: str = "SESSION") -> None:
        """Print session summary to console."""
        print(f"\n{'=' * 70}")
        print(f"  {label} SUMMARY")
        print(f"{'=' * 70}")
        print(f"  Models: {summary['model_count']}")
        print(f"  Best:   {summary.get('best_model', 'N/A')}")
        print(f"  Worst:  {summary.get('worst_model', 'N/A')}")
        print()

        is_day = summary.get("session_number") == "day_total"

        if is_day:
            print(
                f"  {'Rank':<5}{'Model':<25}{'S1 Ret%':<10}{'S2 Ret%':<10}"
                f"{'Day Ret%':<10}{'Trades':<8}{'Fitness':<8}"
            )
            print(f"  {'-' * 74}")
            for r in summary.get("rankings", []):
                print(
                    f"  {r['rank']:<5}"
                    f"{r['name']:<25}"
                    f"{r.get('s1_return', 0):>+7.3f}%  "
                    f"{r.get('s2_return', 0):>+7.3f}%  "
                    f"{r['return_pct']:>+7.3f}%  "
                    f"{r['trades']:<8}"
                    f"{r['fitness']:<8.4f}"
                )
        else:
            print(
                f"  {'Rank':<5}{'Model':<25}{'Return %':<12}"
                f"{'Sharpe':<10}{'Trades':<8}{'Fitness':<8}"
            )
            print(f"  {'-' * 66}")
            for r in summary.get("rankings", []):
                print(
                    f"  {r['rank']:<5}"
                    f"{r['name']:<25}"
                    f"{r['return_pct']:>+8.4f}%   "
                    f"{r['sharpe']:>8.4f}  "
                    f"{r['trades']:<8}"
                    f"{r['fitness']:<8.4f}"
                )

        print(f"{'=' * 70}")

    def _replay_bars(self, bars: list["BarData"], session_number: int) -> dict:
        """Replay a list of bars for backtest. Returns summary."""
        self._session_number = session_number
        self.execution._session_number = session_number
        self._bar_count = 0

        models = list(self._model_records.values())
        self.tracker.initialize_models(models)
        self.tracker.set_session_number(session_number)
        self.tracker.set_session_date(self._session_date)

        if session_number == 1:
            self.execution.reset_daily_limits()

        # Record session (replace stale record from a failed prior run)
        db = get_session(self.config.db_path)
        try:
            stale = (
                db.query(SessionRecord)
                .filter(
                    SessionRecord.session_date == self._session_date,
                    SessionRecord.session_number == session_number,
                )
                .first()
            )
            if stale:
                db.delete(stale)
                db.flush()

            db.add(SessionRecord(
                session_date=self._session_date,
                session_number=session_number,
                generation=models[0].generation if models else 1,
                started_at=datetime.utcnow(),
            ))
            db.commit()
        finally:
            db.close()

        # Replay bars — _fan_out_bar is async but in simulate mode all order
        # submission is synchronous (no API calls), so we drive coroutines here
        snapshot_interval = max(1, len(bars) // 20)  # ~20 snapshots per session
        total_bars = len(bars)
        session_mins = self.config.arena.session_1_minutes
        loop = asyncio.new_event_loop()
        for i, bar in enumerate(bars):
            self.execution._last_prices[bar.symbol] = bar.close
            # Simulate time remaining based on position in bar list
            bar.minutes_remaining = max(0.0, session_mins * (1.0 - (i + 1) / total_bars))
            loop.run_until_complete(self._fan_out_bar(bar))

            if (i + 1) % snapshot_interval == 0:
                self.tracker.set_last_prices(dict(self.execution._last_prices))
                self.tracker.update_all()
                self.tracker.save_snapshots(self._session_date, session_number)

        loop.close()

        # Force-close anything still open (sync — backtest has no running event loop)
        self._end_session_liquidate_sync()

        # Final update
        self.tracker.set_last_prices(dict(self.execution._last_prices))
        self.tracker.update_all()
        self.tracker.save_snapshots(self._session_date, session_number)
        summary = self.tracker.generate_session_summary()
        summary["session_number"] = session_number

        # Update session record
        db = get_session(self.config.db_path)
        try:
            rec = (
                db.query(SessionRecord)
                .filter(SessionRecord.session_date == self._session_date,
                        SessionRecord.session_number == session_number)
                .first()
            )
            if rec:
                rec.ended_at = datetime.utcnow()
                rec.total_bars = self._bar_count
                rec.total_trades = sum(m.total_trades for m in self.tracker.get_leaderboard())
                rec.summary = summary
                db.commit()
        finally:
            db.close()

        logger.info(f"[BACKTEST] Session {session_number}: {self._bar_count} bars replayed")
        return summary

    def run_backtest(self, date: str) -> None:
        """Run a full backtest day on historical data.

        Args:
            date: Date string YYYY-MM-DD to backtest.
        """
        from datetime import datetime as dt

        init_db(self.config.db_path)
        self._session_date = date
        self._running = True

        logger.info(f"[BACKTEST] Fetching historical bars for {date}...")
        target = dt.strptime(date, "%Y-%m-%d")
        all_bars = self.feed.fetch_historical_bars(
            self.config.arena.symbols, days_back=1,
            start_date=target, end_date=target + timedelta(days=1),
        )

        # Flatten and sort by timestamp
        flat_bars: list[BarData] = []
        for symbol_bars in all_bars.values():
            flat_bars.extend(symbol_bars)
        flat_bars.sort(key=lambda b: b.timestamp)

        if not flat_bars:
            logger.error(f"No bars found for {date}")
            return

        logger.info(f"[BACKTEST] {len(flat_bars)} bars across {len(all_bars)} symbols")

        # Split into two halves for session 1 and session 2
        midpoint = len(flat_bars) // 2
        s1_bars = flat_bars[:midpoint]
        s2_bars = flat_bars[midpoint:]

        # Load models
        models = self._load_or_create_models()
        self._instantiate_strategies(models)

        # Session 1
        summary_1 = self._replay_bars(s1_bars, 1)
        self._print_summary(summary_1, f"SESSION 1 [{date}]")
        self._generate_session_summaries(1)

        # Self-improvement break
        if self.config.arena.self_improve_enabled:
            param_snapshots = {
                mid: {**strategy.get_params(), "_watch_rules": list(strategy._watch_rules)}
                for mid, strategy in self._models.items()
            }
            self._self_improve()
            self._generate_improvement_summaries(param_snapshots)

        # Session 2
        summary_2 = self._replay_bars(s2_bars, 2)
        self._print_summary(summary_2, f"SESSION 2 [{date}]")
        self._generate_session_summaries(2)

        # Post-S2 self-improvement
        if self.config.arena.self_improve_enabled:
            param_snapshots_2 = {
                mid: {**strategy.get_params(), "_watch_rules": list(strategy._watch_rules)}
                for mid, strategy in self._models.items()
            }
            self._self_improve()
            self._generate_improvement_summaries(param_snapshots_2, session_number=2)

        # Note: backtest doesn't call _run_session() so _session_start_params
        # is not set — adaptation drift will be skipped in summaries.

        # Day total
        day_summary = self._compute_day_total(summary_1, summary_2)
        self._print_summary(day_summary, f"DAY TOTAL [{date}]")
        self._running = False
