"""Tests for the batch DB refresh and self-improve config flag."""

import os
import tempfile
from unittest.mock import MagicMock

import pytest

from src.core.arena import Arena
from src.core.config import Config
from src.core.database import (
    ModelStatus,
    Position,
    TradingModel,
    get_session,
    init_db,
    reset_engine,
)
from src.strategies.rsi_reversion import RSIReversionStrategy


@pytest.fixture
def arena_with_db():
    """Create an Arena with a temp DB and 3 test models."""
    # Reset global engine state so we connect to the temp DB
    reset_engine()

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    config = Config.load()
    config.db_path = db_path
    config.arena.model_count = 3

    init_db(db_path)

    # Create test models in DB
    db = get_session(db_path)
    models = []
    for i in range(3):
        m = TradingModel(
            name=f"test_model_{i}",
            strategy_type="rsi_reversion",
            parameters={},
            generation=1,
            initial_capital=1000.0,
            current_capital=1000.0 + i * 50,  # vary capital
        )
        db.add(m)
        models.append(m)
    db.commit()
    for m in models:
        db.refresh(m)

    # Add some positions
    db.add(Position(
        model_id=models[0].id,
        symbol="AAPL",
        quantity=5,
        avg_entry_price=150.0,
        current_price=152.0,
    ))
    db.add(Position(
        model_id=models[1].id,
        symbol="MSFT",
        quantity=3,
        avg_entry_price=400.0,
        current_price=405.0,
    ))
    db.add(Position(
        model_id=models[2].id,
        symbol="AAPL",
        quantity=2,
        avg_entry_price=149.0,
        current_price=152.0,
    ))
    # Zero-qty position should be ignored
    db.add(Position(
        model_id=models[0].id,
        symbol="TSLA",
        quantity=0,
        avg_entry_price=200.0,
        current_price=210.0,
    ))
    db.commit()

    # Eagerly load all attributes before detaching
    for m in models:
        db.refresh(m)
        _ = m.id, m.name, m.strategy_type, m.parameters, m.generation
        _ = m.initial_capital, m.current_capital, m.status
        db.expunge(m)
    db.close()

    arena = Arena(config, simulate=True)

    # Instantiate strategies manually
    for m in models:
        strategy = RSIReversionStrategy(m.name)
        strategy.current_capital = m.current_capital
        arena._models[m.id] = strategy
        arena._model_records[m.id] = m

    yield arena, models, db_path

    reset_engine()
    os.unlink(db_path)


class TestBatchRefresh:
    def test_refreshes_all_capitals(self, arena_with_db):
        arena, models, _ = arena_with_db
        positioned = arena._refresh_all_models_batch()

        # Verify capitals are correct
        assert arena._models[models[0].id].current_capital == 1000.0
        assert arena._models[models[1].id].current_capital == 1050.0
        assert arena._models[models[2].id].current_capital == 1100.0

    def test_refreshes_all_positions(self, arena_with_db):
        arena, models, _ = arena_with_db
        positioned = arena._refresh_all_models_batch()

        # Model 0 has AAPL
        assert arena._models[models[0].id]._positions == {"AAPL": 5}
        assert arena._models[models[0].id]._entry_prices == {"AAPL": 150.0}

        # Model 1 has MSFT
        assert arena._models[models[1].id]._positions == {"MSFT": 3}

        # Model 2 has AAPL
        assert arena._models[models[2].id]._positions == {"AAPL": 2}

    def test_returns_positioned_symbols(self, arena_with_db):
        arena, _, _ = arena_with_db
        positioned = arena._refresh_all_models_batch()
        assert positioned == {"AAPL", "MSFT"}

    def test_zero_qty_positions_excluded(self, arena_with_db):
        arena, models, _ = arena_with_db
        positioned = arena._refresh_all_models_batch()
        # TSLA has qty=0, should not appear
        assert "TSLA" not in positioned
        assert "TSLA" not in arena._models[models[0].id]._positions

    def test_cached_positioned_symbols(self, arena_with_db):
        arena, _, _ = arena_with_db
        positioned = arena._refresh_all_models_batch()
        arena._cached_positioned_symbols = positioned
        # _get_positioned_symbols should use cache
        assert arena._get_positioned_symbols() == {"AAPL", "MSFT"}


class TestPositionManagerStateCached:
    def test_uses_in_memory_positions(self, arena_with_db):
        arena, models, _ = arena_with_db
        # Batch refresh to populate strategies
        positioned = arena._refresh_all_models_batch()

        # Set some last prices
        arena.execution._last_prices = {"AAPL": 155.0, "MSFT": 410.0}

        arena._update_position_manager_state_cached(positioned)

        # Verify position manager was updated
        # AAPL exposure: model0 5*155 + model2 2*155 = 775+310 = 1085
        # MSFT exposure: model1 3*410 = 1230
        state = arena.position_manager
        assert state._total_portfolio_value > 0
        assert state._symbol_exposure.get("AAPL", 0) == 5 * 155.0 + 2 * 155.0
        assert state._symbol_exposure.get("MSFT", 0) == 3 * 410.0


class TestEndOfDayLiquidation:
    def test_liquidate_closes_all_positions(self, arena_with_db):
        arena, models, db_path = arena_with_db
        arena._session_date = "2026-03-07"

        # Verify positions exist before liquidation
        db = get_session(db_path)
        open_before = db.query(Position).filter(Position.quantity > 0).count()
        db.close()
        assert open_before == 3  # AAPL(5), MSFT(3), AAPL(2)

        # Run EOD liquidation
        arena._end_session_liquidate()

        # All positions should now be zero
        db = get_session(db_path)
        open_after = db.query(Position).filter(Position.quantity > 0).count()
        db.close()
        assert open_after == 0


class TestSelfImproveConfig:
    def test_enabled_by_default(self):
        config = Config.load()
        assert config.arena.self_improve_enabled is True

    def test_can_enable(self):
        config = Config.load()
        config.arena.self_improve_enabled = True
        assert config.arena.self_improve_enabled is True
