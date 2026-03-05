"""Day Trader Arena - Main entry point.

Usage:
    python main.py run                  # Run live (pre-market warmup + session 1 + break + session 2)
    python main.py session1             # Run session 1 only (with warmup)
    python main.py session2             # Run session 2 only (auto-caps to market close)
    python main.py backtest YYYY-MM-DD  # Backtest on historical data
    python main.py dashboard            # Start the dashboard server
    python main.py evolve               # Run evolution (after session 2)
    python main.py improve              # Self-improve all models (no culling)
    python main.py status               # Show current model pool
    python main.py reset                # Reset all models to initial capital
    python main.py pipeline-test        # 5-min micro-session to validate pipeline
"""

import asyncio
import logging
import os
import sys

from src.core.arena import Arena
from src.core.config import Config
from src.core.database import (
    GenerationRecord,
    ModelStatus,
    TradingModel,
    get_session,
    init_db,
)
from src.core.performance import PerformanceTracker
from src.evolution.engine import EvolutionEngine

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/arena.log"),
    ],
)
logger = logging.getLogger(__name__)


def cmd_run():
    """Run the full trading day: Pre-market warmup -> S1 -> Reflect -> Improve -> S2 -> Reflect -> Improve."""
    config = Config.load()
    arena = Arena(config)
    asyncio.run(arena.run())


def cmd_session1():
    """Run session 1 only (with pre-market warmup)."""
    config = Config.load()
    arena = Arena(config)
    asyncio.run(arena.run_session_only(1))


def cmd_session2():
    """Run session 2 only. Auto-caps duration to finish before market close."""
    config = Config.load()
    arena = Arena(config)
    asyncio.run(arena.run_session_only(2))


def cmd_reset():
    """Reset all models to initial capital and clear positions."""
    config = Config.load()
    init_db(config.db_path)
    db = get_session(config.db_path)
    try:
        from src.core.database import Position
        models = db.query(TradingModel).filter(
            TradingModel.status == ModelStatus.ACTIVE
        ).all()
        if not models:
            print("No active models.")
            return

        for m in models:
            m.current_capital = config.arena.initial_capital
            m.initial_capital = config.arena.initial_capital

        # Clear all positions
        db.query(Position).update({
            Position.quantity: 0,
            Position.avg_entry_price: 0.0,
            Position.current_price: 0.0,
            Position.unrealized_pnl: 0.0,
            Position.realized_pnl: 0.0,
        })
        db.commit()

        print(f"\nReset {len(models)} models to ${config.arena.initial_capital:,.2f}")
        for m in models:
            print(f"  {m.name}: ${m.current_capital:,.2f}")
        print()
    finally:
        db.close()


def cmd_dashboard():
    """Start the FastAPI dashboard server."""
    import uvicorn
    from src.dashboard.api.server import app

    logger.info("Starting dashboard at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)


def cmd_evolve():
    """Run weekly evolution (cull bottom 2 + spawn offspring).

    Should be run at end of week after the last trading day completes.
    Also updates COLLAB eligible voters to top 5 weekly performers.
    """
    config = Config.load()
    init_db(config.db_path)

    db = get_session(config.db_path)
    try:
        models = (
            db.query(TradingModel)
            .filter(TradingModel.status == ModelStatus.ACTIVE)
            .all()
        )
        if not models:
            print("No active models. Run the arena first.")
            return
    finally:
        db.close()

    tracker = PerformanceTracker(config.db_path)
    tracker.initialize_models(models)
    tracker.update_all()

    engine = EvolutionEngine(config)
    from datetime import datetime
    session_date = datetime.now().strftime("%Y-%m-%d")
    result = engine.run_evolution(session_date, tracker)

    print(f"\nGeneration {result['generation']}:")
    print(f"  Eliminated: {result['eliminated']} ({', '.join(result['eliminated_names'])})")
    print(f"  Survivors: {result['survivors']}")
    print(f"  Offspring: {result['offspring']} ({', '.join(result['offspring_names'])})")
    print(f"  Next pool: {result['next_pool_size']} models")


def cmd_status():
    """Show current arena status."""
    config = Config.load()
    init_db(config.db_path)

    db = get_session(config.db_path)
    try:
        active = db.query(TradingModel).filter(TradingModel.status == ModelStatus.ACTIVE).all()
        eliminated = db.query(TradingModel).filter(TradingModel.status == ModelStatus.ELIMINATED).count()
        latest_gen = db.query(GenerationRecord).order_by(GenerationRecord.generation_number.desc()).first()

        print("\n" + "=" * 60)
        print("  DAY TRADER ARENA STATUS")
        print("=" * 60)
        print(f"  Active models: {len(active)}")
        print(f"  Eliminated: {eliminated}")
        print(f"  Current generation: {latest_gen.generation_number if latest_gen else 1}")
        print()

        if active:
            print(f"  {'Model':<30}{'Type':<18}{'Gen':<6}{'Capital':<12}")
            print(f"  {'-' * 66}")
            for m in active:
                print(f"  {m.name:<30}{m.strategy_type:<18}{m.generation:<6}${m.current_capital:>10,.2f}")

        print("=" * 60)
    finally:
        db.close()


def cmd_improve():
    """Self-improve all active models based on current performance. No culling."""
    import copy
    import random

    config = Config.load()
    init_db(config.db_path)

    db = get_session(config.db_path)
    try:
        models = (
            db.query(TradingModel)
            .filter(TradingModel.status == ModelStatus.ACTIVE)
            .all()
        )
        if not models:
            print("No active models. Run the arena first.")
            return
        for m in models:
            _ = m.id, m.name, m.strategy_type, m.parameters, m.generation
            _ = m.initial_capital, m.current_capital, m.status
            db.expunge(m)
    finally:
        db.close()

    tracker = PerformanceTracker(config.db_path)
    tracker.initialize_models(models)
    tracker.update_all()
    leaderboard = tracker.get_leaderboard()

    from src.strategies.registry import create_strategy

    print("\n" + "=" * 60)
    print("  SELF-IMPROVEMENT (no culling)")
    print("=" * 60)

    db = get_session(config.db_path)
    try:
        for metrics in leaderboard:
            model_record = next((m for m in models if m.id == metrics.model_id), None)
            if not model_record:
                continue

            strategy = create_strategy(
                model_record.strategy_type,
                model_record.name,
                params=model_record.parameters,
            )
            params = strategy.get_params()
            new_params = copy.deepcopy(params)

            if metrics.return_pct < 0:
                mutation_strength = 0.10
            elif metrics.return_pct < 1.0:
                mutation_strength = 0.05
            else:
                mutation_strength = 0.02

            for key, value in new_params.items():
                if isinstance(value, (int, float)) and random.random() < 0.4:
                    perturbation = random.uniform(-mutation_strength, mutation_strength)
                    new_value = value * (1 + perturbation)
                    if isinstance(value, int):
                        new_value = max(1, int(round(new_value)))
                    else:
                        new_value = round(new_value, 6)
                    new_params[key] = new_value

            strategy.set_params(new_params)

            db_model = db.query(TradingModel).get(metrics.model_id)
            if db_model:
                db_model.parameters = strategy.get_params()

            print(
                f"  {metrics.model_name:<30} "
                f"return={metrics.return_pct:+.3f}%  "
                f"mutation={mutation_strength*100:.0f}%"
            )

        db.commit()
    finally:
        db.close()

    print("=" * 60)
    print("  Improvement complete. Parameters updated in DB.")
    print()


def cmd_backtest():
    """Run a backtest on historical data for a given date.

    Usage: python main.py backtest YYYY-MM-DD
    """
    if len(sys.argv) < 3:
        print("Usage: python main.py backtest YYYY-MM-DD")
        print("Example: python main.py backtest 2025-02-28")
        sys.exit(1)

    date = sys.argv[2]
    config = Config.load()
    arena = Arena(config, simulate=True)
    arena.run_backtest(date)


def cmd_pipeline_test():
    """Run a 5-minute micro-session with 3 models on 2 symbols.

    Validates the full pipeline: connect -> bars -> signal -> submit ->
    fill confirmation -> quote monitoring -> exit.
    """
    config = Config.load()
    arena = Arena(config)
    asyncio.run(arena.run_pipeline_test())


COMMANDS = {
    "run": cmd_run,
    "session1": cmd_session1,
    "session2": cmd_session2,
    "dashboard": cmd_dashboard,
    "evolve": cmd_evolve,
    "improve": cmd_improve,
    "status": cmd_status,
    "backtest": cmd_backtest,
    "reset": cmd_reset,
    "pipeline-test": cmd_pipeline_test,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    COMMANDS[sys.argv[1]]()
