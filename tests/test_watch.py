"""Tests for the watch signal flow: creation, TTL expiry, quote entry, removal."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.strategy import BarData, Strategy, TradeSignal, WatchSignal
from src.core.watch_rules import (
    build_watch_context,
    evaluate_entry_condition,
    evaluate_watch_condition,
    validate_rule,
)
from src.strategies.rsi_reversion import RSIReversionStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bar(symbol: str = "AAPL", close: float = 150.0, volume: int = 1000, **kw) -> BarData:
    return BarData(
        symbol=symbol,
        timestamp=kw.get("timestamp", datetime.now(timezone.utc)),
        open=kw.get("open", close - 0.5),
        high=kw.get("high", close + 0.5),
        low=kw.get("low", close - 1.0),
        close=close,
        volume=volume,
    )


def rsi_watch_rule(
    oversold: float = 35.0,
    entry_price_op: str = "lt",
    ttl: int = 5,
) -> dict:
    """A realistic RSI-based watch rule: watch when RSI near oversold, enter on price dip."""
    return {
        "watch_when": {"indicator": "rsi", "op": "lt", "value": oversold},
        "entry_when": {"price_op": entry_price_op, "context_key": "entry_level"},
        "context_values": {"entry_level": "close"},
        "ttl_bars": ttl,
        "reason": f"RSI below {oversold}",
    }


def prime_rsi_strategy(strategy: RSIReversionStrategy, symbol: str, n_bars: int = 20, base_close: float = 150.0):
    """Feed enough bars to prime RSI indicator."""
    for i in range(n_bars):
        bar = make_bar(symbol=symbol, close=base_close + (i % 3 - 1) * 0.5)
        bar.minutes_remaining = None
        strategy.record_bar(bar)
        strategy.on_bar(bar)


# ---------------------------------------------------------------------------
# watch_rules DSL tests
# ---------------------------------------------------------------------------

class TestWatchRuleDSL:
    def test_evaluate_watch_condition_lt(self):
        rule = rsi_watch_rule(oversold=35.0)
        assert evaluate_watch_condition(rule, {"rsi": 30.0}) is True
        assert evaluate_watch_condition(rule, {"rsi": 40.0}) is False

    def test_evaluate_watch_condition_gt(self):
        rule = {"watch_when": {"indicator": "rsi", "op": "gt", "value": 70.0}}
        assert evaluate_watch_condition(rule, {"rsi": 75.0}) is True
        assert evaluate_watch_condition(rule, {"rsi": 65.0}) is False

    def test_evaluate_watch_condition_between(self):
        rule = {"watch_when": {"indicator": "rsi", "op": "between", "value": [25, 35]}}
        assert evaluate_watch_condition(rule, {"rsi": 30.0}) is True
        assert evaluate_watch_condition(rule, {"rsi": 20.0}) is False
        assert evaluate_watch_condition(rule, {"rsi": 40.0}) is False

    def test_evaluate_watch_condition_missing_indicator(self):
        rule = rsi_watch_rule()
        assert evaluate_watch_condition(rule, {"close": 100.0}) is False

    def test_build_watch_context(self):
        rule = rsi_watch_rule()
        indicators = {"close": 150.0, "rsi": 30.0}
        ctx = build_watch_context(rule, indicators)
        assert ctx["entry_level"] == 150.0

    def test_evaluate_entry_condition(self):
        rule = rsi_watch_rule(entry_price_op="lt")
        context = {"entry_level": 150.0}
        # Price drops below frozen close -> entry
        assert evaluate_entry_condition(rule, 149.0, context) is True
        # Price above -> no entry
        assert evaluate_entry_condition(rule, 151.0, context) is False

    def test_validate_rule_valid(self):
        rule = rsi_watch_rule()
        ok, err = validate_rule(rule, ["close", "rsi", "has_position"])
        assert ok, err

    def test_validate_rule_bad_indicator(self):
        rule = rsi_watch_rule()
        ok, err = validate_rule(rule, ["close"])  # rsi not available
        assert not ok
        assert "unknown indicator" in err

    def test_validate_rule_bad_op(self):
        rule = rsi_watch_rule()
        rule["watch_when"]["op"] = "eq"
        ok, err = validate_rule(rule, ["close", "rsi"])
        assert not ok
        assert "invalid op" in err

    def test_validate_rule_missing_context_key(self):
        rule = rsi_watch_rule()
        rule["entry_when"]["context_key"] = "nonexistent"
        ok, err = validate_rule(rule, ["close", "rsi"])
        assert not ok
        assert "not found in context_values" in err

    def test_validate_rule_ttl_out_of_range(self):
        rule = rsi_watch_rule(ttl=50)
        ok, err = validate_rule(rule, ["close", "rsi"])
        assert not ok
        assert "ttl_bars" in err


# ---------------------------------------------------------------------------
# Strategy-level watch tests
# ---------------------------------------------------------------------------

class TestStrategyWatch:
    def test_get_watch_signals_triggers(self):
        """Strategy with a watch rule returns WatchSignal when condition met."""
        strategy = RSIReversionStrategy("test_rsi")
        strategy._watch_rules = [rsi_watch_rule(oversold=55.0)]  # easy to trigger
        prime_rsi_strategy(strategy, "AAPL")

        # Feed a bar — RSI should be somewhere around neutral (40-60 range)
        bar = make_bar(symbol="AAPL", close=149.0)
        bar.minutes_remaining = 30.0
        strategy.record_bar(bar)
        strategy.on_bar(bar)

        signals = strategy.get_watch_signals(bar)
        assert len(signals) >= 1
        assert signals[0].symbol == "AAPL"
        assert signals[0].ttl_bars == 5

    def test_get_watch_signals_no_rules(self):
        """Strategy with no watch rules returns empty list."""
        strategy = RSIReversionStrategy("test_rsi")
        prime_rsi_strategy(strategy, "AAPL")
        bar = make_bar(symbol="AAPL")
        bar.minutes_remaining = 30.0
        strategy.record_bar(bar)
        strategy.on_bar(bar)
        assert strategy.get_watch_signals(bar) == []

    def test_get_watch_signals_skips_with_position(self):
        """Don't generate watch signals for symbols we already hold."""
        strategy = RSIReversionStrategy("test_rsi")
        strategy._watch_rules = [rsi_watch_rule(oversold=55.0)]
        strategy._positions = {"AAPL": 10}
        prime_rsi_strategy(strategy, "AAPL")

        bar = make_bar(symbol="AAPL")
        bar.minutes_remaining = 30.0
        strategy.record_bar(bar)
        strategy.on_bar(bar)
        assert strategy.get_watch_signals(bar) == []

    def test_on_watch_quote_triggers_entry(self):
        """Quote below frozen entry level triggers a buy signal."""
        strategy = RSIReversionStrategy("test_rsi")
        strategy.current_capital = 1000.0

        rule = rsi_watch_rule(entry_price_op="lt")
        context = {"entry_level": 150.0, "_rule": rule}

        signal = strategy.on_watch_quote("AAPL", 148.5, 149.0, datetime.now(), context)
        assert signal is not None
        assert signal.side == "buy"
        assert signal.quantity > 0

    def test_on_watch_quote_no_entry_above(self):
        """Quote above frozen entry level does NOT trigger."""
        strategy = RSIReversionStrategy("test_rsi")
        strategy.current_capital = 1000.0

        rule = rsi_watch_rule(entry_price_op="lt")
        context = {"entry_level": 150.0, "_rule": rule}

        signal = strategy.on_watch_quote("AAPL", 151.0, 151.5, datetime.now(), context)
        assert signal is None

    def test_on_watch_quote_no_rule_in_context(self):
        """Missing _rule in context returns None."""
        strategy = RSIReversionStrategy("test_rsi")
        signal = strategy.on_watch_quote("AAPL", 148.0, 149.0, datetime.now(), {})
        assert signal is None


# ---------------------------------------------------------------------------
# Arena-level watch management tests
# ---------------------------------------------------------------------------

class TestArenaWatchManagement:
    """Test the Arena's watch list bookkeeping (register, tick, expire, remove)."""

    def _make_arena(self):
        """Create a minimal Arena-like object with just the watch methods."""
        from src.core.arena import Arena
        from src.core.config import Config

        config = Config.load()
        config.arena.max_watches_per_model = 3
        arena = Arena(config, simulate=True)
        return arena

    def test_register_watch(self):
        arena = self._make_arena()
        strategy = RSIReversionStrategy("test")
        strategy.current_capital = 1000.0
        arena._models[1] = strategy
        arena._model_records[1] = MagicMock(id=1)

        watch = WatchSignal(symbol="AAPL", reason="test", ttl_bars=5, context={"x": 1})
        arena._register_watch(1, watch)

        assert 1 in arena._watch_list
        assert "AAPL" in arena._watch_list[1]
        assert arena._watch_list[1]["AAPL"]["ttl"] == 5
        assert arena._watch_stats["created"] == 1

    def test_register_watch_skips_if_position_held(self):
        arena = self._make_arena()
        strategy = RSIReversionStrategy("test")
        strategy._positions = {"AAPL": 10}
        arena._models[1] = strategy

        watch = WatchSignal(symbol="AAPL", reason="test", ttl_bars=5, context={})
        arena._register_watch(1, watch)

        assert 1 not in arena._watch_list

    def test_register_watch_enforces_max_cap(self):
        arena = self._make_arena()
        strategy = RSIReversionStrategy("test")
        strategy.current_capital = 1000.0
        arena._models[1] = strategy
        arena._model_records[1] = MagicMock(id=1)

        for sym in ["AAPL", "MSFT", "GOOG"]:
            arena._register_watch(1, WatchSignal(symbol=sym, reason="t", ttl_bars=5, context={}))
        assert len(arena._watch_list[1]) == 3

        # 4th should be rejected (max=3)
        arena._register_watch(1, WatchSignal(symbol="TSLA", reason="t", ttl_bars=5, context={}))
        assert "TSLA" not in arena._watch_list[1]
        assert arena._watch_stats["created"] == 3

    def test_register_watch_allows_renewal(self):
        arena = self._make_arena()
        strategy = RSIReversionStrategy("test")
        strategy.current_capital = 1000.0
        arena._models[1] = strategy
        arena._model_records[1] = MagicMock(id=1)

        for sym in ["AAPL", "MSFT", "GOOG"]:
            arena._register_watch(1, WatchSignal(symbol=sym, reason="t", ttl_bars=5, context={}))

        # Renewing AAPL should work even though at cap
        arena._register_watch(1, WatchSignal(symbol="AAPL", reason="renewed", ttl_bars=3, context={}))
        assert arena._watch_list[1]["AAPL"]["ttl"] == 3
        assert arena._watch_list[1]["AAPL"]["reason"] == "renewed"

    def test_tick_watches_decrements_ttl(self):
        arena = self._make_arena()
        strategy = RSIReversionStrategy("test")
        strategy.current_capital = 1000.0
        arena._models[1] = strategy
        arena._model_records[1] = MagicMock(id=1)

        arena._register_watch(1, WatchSignal(symbol="AAPL", reason="t", ttl_bars=3, context={}))
        assert arena._watch_list[1]["AAPL"]["ttl"] == 3

        arena._tick_watches()
        assert arena._watch_list[1]["AAPL"]["ttl"] == 2

        arena._tick_watches()
        assert arena._watch_list[1]["AAPL"]["ttl"] == 1

    def test_tick_watches_expires_at_zero(self):
        arena = self._make_arena()
        strategy = RSIReversionStrategy("test")
        strategy.current_capital = 1000.0
        arena._models[1] = strategy
        arena._model_records[1] = MagicMock(id=1)

        arena._register_watch(1, WatchSignal(symbol="AAPL", reason="t", ttl_bars=1, context={}))

        arena._tick_watches()
        assert 1 not in arena._watch_list  # model entry cleaned up
        assert arena._watch_stats["expired"] == 1

    def test_remove_watch_on_conversion(self):
        arena = self._make_arena()
        strategy = RSIReversionStrategy("test")
        strategy.current_capital = 1000.0
        arena._models[1] = strategy
        arena._model_records[1] = MagicMock(id=1)

        arena._register_watch(1, WatchSignal(symbol="AAPL", reason="t", ttl_bars=5, context={}))
        arena._register_watch(1, WatchSignal(symbol="MSFT", reason="t", ttl_bars=5, context={}))

        arena._remove_watch(1, "AAPL")
        assert "AAPL" not in arena._watch_list[1]
        assert "MSFT" in arena._watch_list[1]

    def test_remove_watch_cleans_empty_model(self):
        arena = self._make_arena()
        strategy = RSIReversionStrategy("test")
        strategy.current_capital = 1000.0
        arena._models[1] = strategy
        arena._model_records[1] = MagicMock(id=1)

        arena._register_watch(1, WatchSignal(symbol="AAPL", reason="t", ttl_bars=5, context={}))
        arena._remove_watch(1, "AAPL")
        assert 1 not in arena._watch_list

    def test_clear_watches(self):
        arena = self._make_arena()
        strategy = RSIReversionStrategy("test")
        strategy.current_capital = 1000.0
        arena._models[1] = strategy
        arena._model_records[1] = MagicMock(id=1)

        arena._register_watch(1, WatchSignal(symbol="AAPL", reason="t", ttl_bars=5, context={}))
        arena._clear_watches()
        assert arena._watch_list == {}
        assert arena._watch_stats == {"created": 0, "expired": 0, "converted": 0}

    def test_get_watched_symbols(self):
        arena = self._make_arena()
        s1 = RSIReversionStrategy("s1")
        s1.current_capital = 1000.0
        s2 = RSIReversionStrategy("s2")
        s2.current_capital = 1000.0
        arena._models[1] = s1
        arena._models[2] = s2
        arena._model_records[1] = MagicMock(id=1)
        arena._model_records[2] = MagicMock(id=2)

        arena._register_watch(1, WatchSignal(symbol="AAPL", reason="t", ttl_bars=5, context={}))
        arena._register_watch(2, WatchSignal(symbol="MSFT", reason="t", ttl_bars=5, context={}))
        arena._register_watch(2, WatchSignal(symbol="AAPL", reason="t", ttl_bars=5, context={}))

        watched = arena._get_watched_symbols()
        assert watched == {"AAPL", "MSFT"}


# ---------------------------------------------------------------------------
# End-to-end watch flow: bar triggers watch -> quote triggers entry
# ---------------------------------------------------------------------------

class TestWatchEndToEnd:
    def test_bar_creates_watch_quote_converts(self):
        """Full flow: bar triggers watch_when -> WatchSignal created ->
        quote triggers entry_when -> TradeSignal returned."""
        strategy = RSIReversionStrategy("e2e_test")
        strategy.current_capital = 1000.0

        # Rule: watch when RSI < 55 (easy trigger), enter when price < frozen close
        strategy._watch_rules = [rsi_watch_rule(oversold=55.0, entry_price_op="lt", ttl=5)]

        # Prime indicators
        prime_rsi_strategy(strategy, "AAPL", n_bars=20, base_close=150.0)

        # Produce a bar that should trigger the watch
        bar = make_bar(symbol="AAPL", close=149.5)
        bar.minutes_remaining = 30.0
        strategy.record_bar(bar)
        strategy.on_bar(bar)

        signals = strategy.get_watch_signals(bar)
        assert len(signals) >= 1, "Expected watch signal from RSI rule"

        watch = signals[0]
        assert watch.symbol == "AAPL"
        assert "entry_level" in watch.context

        # Now simulate a quote that dips below the frozen close
        entry_signal = strategy.on_watch_quote(
            "AAPL", 148.0, 148.5, datetime.now(), watch.context
        )
        assert entry_signal is not None
        assert entry_signal.side == "buy"
        assert entry_signal.quantity > 0

    def test_watch_expires_without_entry(self):
        """Watch that never gets a matching quote expires after TTL bars."""
        from src.core.arena import Arena
        from src.core.config import Config

        config = Config.load()
        arena = Arena(config, simulate=True)
        strategy = RSIReversionStrategy("expire_test")
        strategy.current_capital = 1000.0
        arena._models[1] = strategy
        arena._model_records[1] = MagicMock(id=1)

        # Register with TTL=3
        arena._register_watch(1, WatchSignal(
            symbol="AAPL", reason="test", ttl_bars=3,
            context={"entry_level": 150.0, "_rule": rsi_watch_rule()},
        ))
        assert "AAPL" in arena._watch_list[1]

        # Tick 3 times (simulating 3 bars with no matching quote)
        arena._tick_watches()
        assert "AAPL" in arena._watch_list.get(1, {})
        arena._tick_watches()
        assert "AAPL" in arena._watch_list.get(1, {})
        arena._tick_watches()
        # Should be expired now
        assert 1 not in arena._watch_list
        assert arena._watch_stats["expired"] == 1

    def test_watch_removed_when_position_acquired(self):
        """If model buys the symbol (from bar signal), watch is dropped on next quote check."""
        from src.core.arena import Arena
        from src.core.config import Config

        config = Config.load()
        arena = Arena(config, simulate=True)
        strategy = RSIReversionStrategy("pos_test")
        strategy.current_capital = 1000.0
        arena._models[1] = strategy
        arena._model_records[1] = MagicMock(id=1)

        arena._register_watch(1, WatchSignal(
            symbol="AAPL", reason="test", ttl_bars=5,
            context={"entry_level": 150.0, "_rule": rsi_watch_rule()},
        ))

        # Simulate model acquiring a position
        strategy._positions["AAPL"] = 5

        # _on_quote checks for this — simulate the position check from _on_quote
        # The arena's _on_quote does: if strategy._positions.get(symbol, 0) > 0: _remove_watch
        if strategy._positions.get("AAPL", 0) > 0:
            arena._remove_watch(1, "AAPL")

        assert 1 not in arena._watch_list
