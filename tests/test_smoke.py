"""Smoke tests for the arena API flow (start/stop/state)."""

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, patch

import pytest

from src.core.config import Config
from src.core.database import (
    ModelStatus,
    TradingModel,
    get_session,
    init_db,
    reset_engine,
)


@pytest.fixture
def temp_db():
    """Create a temp DB with 3 models."""
    reset_engine()

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    init_db(db_path)
    db = get_session(db_path)
    for i in range(3):
        db.add(TradingModel(
            name=f"smoke_model_{i}",
            strategy_type="rsi_reversion",
            parameters={},
            generation=1,
            initial_capital=1000.0,
            current_capital=1000.0,
        ))
    db.commit()
    db.close()

    yield db_path

    reset_engine()
    os.unlink(db_path)


@pytest.fixture
def config_with_db(temp_db):
    """Config pointing to the temp DB."""
    config = Config.load()
    config.db_path = temp_db
    config.arena.model_count = 3
    return config


class TestArenaRunState:
    """Test the arena run state helper used by /api/arena/state."""

    def test_idle_state(self):
        from src.dashboard.api.server import get_arena_run_state
        import src.dashboard.api.server as srv
        # Clear global state
        srv._arena_task = None
        srv._arena_config = None
        state = get_arena_run_state()
        assert state["state"] == "idle"
        assert state["config"] is None
        assert state["error"] is None

    def test_running_state(self):
        from src.dashboard.api.server import get_arena_run_state
        import src.dashboard.api.server as srv

        loop = asyncio.new_event_loop()
        # Create a task that blocks
        future = loop.create_future()
        task = loop.create_task(self._wait(future))
        srv._arena_task = task
        srv._arena_config = {"num_sessions": 1, "session_minutes": 5}

        state = get_arena_run_state()
        assert state["state"] == "running"
        assert state["config"]["num_sessions"] == 1

        # Cleanup
        future.set_result(None)
        loop.run_until_complete(task)
        loop.close()
        srv._arena_task = None
        srv._arena_config = None

    def test_finished_state(self):
        from src.dashboard.api.server import get_arena_run_state
        import src.dashboard.api.server as srv

        loop = asyncio.new_event_loop()
        task = loop.create_task(self._noop())
        loop.run_until_complete(task)
        srv._arena_task = task
        srv._arena_config = {"num_sessions": 1, "session_minutes": 5}

        state = get_arena_run_state()
        assert state["state"] == "finished"
        assert state["error"] is None

        loop.close()
        srv._arena_task = None
        srv._arena_config = None

    @staticmethod
    async def _wait(future):
        await future

    @staticmethod
    async def _noop():
        pass


class TestDashboardDuringTrading:
    """Test that /api/dashboard returns cached data when arena is running."""

    @pytest.mark.asyncio
    async def test_returns_cache_when_arena_running(self):
        import src.dashboard.api.server as srv

        # Set up fake "running" state
        future = asyncio.get_event_loop().create_future()
        task = asyncio.create_task(self._wait(future))
        srv._arena_task = task
        srv._arena_config = {"num_sessions": 1, "session_minutes": 5}

        # Seed a cache
        srv._dashboard_cache = {
            "models": [{"id": 1, "name": "test"}],
            "trades": [],
            "generations": [],
            "sessions": [],
        }

        from src.dashboard.api.server import dashboard
        result = await dashboard(session_date=None)

        assert "arena_run_state" in result
        assert result["arena_run_state"]["state"] == "running"
        # Should return cached models, not query DB
        assert result["models"] == [{"id": 1, "name": "test"}]

        # Cleanup
        future.set_result(None)
        await task
        srv._arena_task = None
        srv._arena_config = None
        srv._dashboard_cache = None

    @staticmethod
    async def _wait(future):
        await future


class TestDoubleStart:
    """Test that starting the arena twice returns 409."""

    @pytest.mark.asyncio
    async def test_double_start_returns_409(self, config_with_db):
        import src.dashboard.api.server as srv
        from src.dashboard.api.server import arena_start, ArenaStartRequest
        from fastapi import HTTPException

        # Simulate a running task
        future = asyncio.get_event_loop().create_future()
        task = asyncio.create_task(self._wait(future))
        srv._arena_task = task

        with pytest.raises(HTTPException) as exc_info:
            await arena_start(ArenaStartRequest(num_sessions=1, session_minutes=5))
        assert exc_info.value.status_code == 409

        # Cleanup
        future.set_result(None)
        await task
        srv._arena_task = None

    @staticmethod
    async def _wait(future):
        await future


class TestArenaStateEndpoint:
    """Test /api/arena/state response shape."""

    @pytest.mark.asyncio
    async def test_state_shape(self):
        import src.dashboard.api.server as srv
        from src.dashboard.api.server import arena_state

        srv._arena_task = None
        srv._arena_config = None

        result = await arena_state()
        assert "state" in result
        assert "config" in result
        assert "error" in result
        assert result["state"] == "idle"
