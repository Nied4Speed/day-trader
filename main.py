"""Day Trader Arena - Main entry point.

Usage:
    python main.py run       # Run the arena (stream + trade during market hours)
    python main.py dashboard # Start the dashboard server only
    python main.py evolve    # Run evolution on the latest session's data
    python main.py status    # Show current model pool and generation info
"""

import asyncio
import logging
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_run():
    """Run the full arena session."""
    config = Config.load()
    arena = Arena(config)
    asyncio.run(arena.run())


def cmd_dashboard():
    """Start the FastAPI dashboard server."""
    import uvicorn
    from src.dashboard.api.server import app

    logger.info("Starting dashboard at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)


def cmd_evolve():
    """Run evolution on the latest session."""
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

        print("\n" + "=" * 50)
        print("DAY TRADER ARENA STATUS")
        print("=" * 50)
        print(f"Active models: {len(active)}")
        print(f"Eliminated: {eliminated}")
        print(f"Current generation: {latest_gen.generation_number if latest_gen else 1}")
        print()

        if active:
            print(f"{'Model':<30}{'Type':<18}{'Gen':<6}{'Capital':<12}")
            print("-" * 66)
            for m in active:
                print(f"{m.name:<30}{m.strategy_type:<18}{m.generation:<6}${m.current_capital:>10,.2f}")

        print("=" * 50)
    finally:
        db.close()


COMMANDS = {
    "run": cmd_run,
    "dashboard": cmd_dashboard,
    "evolve": cmd_evolve,
    "status": cmd_status,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    COMMANDS[sys.argv[1]]()
