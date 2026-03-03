# Architecture Research

**Domain:** Evolutionary Algorithmic Trading Arena (local, multi-model, paper trading)
**Researched:** 2026-03-02
**Confidence:** MEDIUM-HIGH (event-driven trading architecture well-established; evolutionary tournament mechanics verified via analogous open-source project)

## Standard Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                          WEB DASHBOARD                               │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  Model Cards  │  │  Trade Feed  │  │  Generation History      │  │
│  │  (live P&L,   │  │  (recent     │  │  (lineage, fitness       │  │
│  │   ranking)    │  │   orders)    │  │   over time)             │  │
│  └───────────────┘  └──────────────┘  └──────────────────────────┘  │
│               FastAPI + WebSocket (read-only from DB)                │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ reads
┌───────────────────────────────▼──────────────────────────────────────┐
│                         PERSISTENCE LAYER                            │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │               SQLite (single-file, local)                      │  │
│  │   models | trades | positions | performance | generations      │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────┬─────────────────────────────────────────────────────┬─────────┘
       │ writes                                              │ reads
┌──────▼──────────────────────────────┐   ┌─────────────────▼─────────┐
│         TRADING ENGINE              │   │    EVOLUTION ENGINE        │
│  ┌──────────────────────────────┐   │   │  ┌─────────────────────┐  │
│  │       Arena Orchestrator     │   │   │  │  Fitness Ranker     │  │
│  │  (market hours loop,         │   │   │  │  (Sharpe, total     │  │
│  │   session lifecycle)         │   │   │  │   return, drawdown) │  │
│  └──────────┬───────────────────┘   │   │  └──────────┬──────────┘  │
│             │ manages               │   │             │             │
│  ┌──────────▼───────────────────┐   │   │  ┌──────────▼──────────┐  │
│  │     Model Pool (5-10)        │   │   │  │  Tournament Select  │  │
│  │  ┌────────┐  ┌────────┐      │   │   │  │  (bottom K elim.)   │  │
│  │  │ Model  │  │ Model  │ ...  │   │   │  └──────────┬──────────┘  │
│  │  │ (TA)   │  │ (ML)   │      │   │   │             │             │
│  │  └────────┘  └────────┘      │   │   │  ┌──────────▼──────────┐  │
│  └──────────────────────────────┘   │   │  │  Crossover / Merge  │  │
│  ┌──────────────────────────────┐   │   │  │  (param blend,      │  │
│  │      Execution Handler       │   │   │  │   strategy mix)     │  │
│  │  (order routing → Alpaca)    │   │   │  └──────────┬──────────┘  │
│  └──────────────────────────────┘   │   │             │             │
└──────────────────────────────────────   │  ┌──────────▼──────────┐  │
                                          │  │  Spawn New Gen      │  │
┌─────────────────────────────────────┐   │  │  (mutate + seed     │  │
│           DATA LAYER                │   │  │   fresh models)     │  │
│  ┌─────────────────────────────┐    │   │  └─────────────────────┘  │
│  │    Alpaca Market Data       │    │   └───────────────────────────┘
│  │    (WebSocket stream +      │    │
│  │     REST historical)        │    │
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │    Data Normalizer          │    │
│  │    (OHLCV bars, indicators) │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| Arena Orchestrator | Market-hours loop, session start/stop, triggers evolution at session end | Python main loop, schedule library or cron |
| Model Pool | Holds 5-10 live model instances, routes market data to each | List of Strategy objects with shared interface |
| Strategy (Model) | Consumes market data bars, emits buy/sell signals, manages its own position logic | Python class implementing `on_bar(bar) -> Optional[Order]` |
| Execution Handler | Translates model orders into Alpaca API calls, handles fills, tracks positions per model | alpaca-py REST calls, async order submission |
| Data Layer | Streams real-time bars from Alpaca WebSocket, buffers for indicator calculation | alpaca-py DataStream, pandas/numpy for indicator math |
| Fitness Ranker | Computes end-of-session metrics for each model (Sharpe ratio, total return, max drawdown, win rate) | Pure Python calculation from trade records in SQLite |
| Tournament Select | Ranks models by fitness, marks bottom performers for elimination, selects survivors for mating | Sorted list + configurable elimination threshold (e.g., bottom 2) |
| Crossover / Merge | Combines parameter sets or logic of two winning models into hybrid offspring | Single-point parameter crossover, weighted blending |
| Spawn New Gen | Creates next generation: surviving models + merged offspring + optional random mutations | Factory function that produces new Strategy instances |
| SQLite Persistence | Stores all trades, positions, model metadata, generation lineage, performance history | SQLite via sqlite3 or SQLAlchemy Core (no ORM needed) |
| FastAPI Backend | Serves dashboard, pushes live updates via WebSocket, reads from SQLite | FastAPI + asyncio background task for DB polling |
| Web Dashboard | Displays live model rankings, trade feeds, generation timeline | React or plain HTML/JS with WebSocket client |

## Recommended Project Structure

```
day-trader/
├── engine/                     # Core trading engine (runs during market hours)
│   ├── arena.py                # Main orchestrator: market-hours loop, session lifecycle
│   ├── models/                 # All strategy implementations
│   │   ├── base.py             # Abstract Strategy interface every model must implement
│   │   ├── technical.py        # Technical analysis strategies (MA, RSI, MACD)
│   │   ├── ml_model.py         # ML-based strategies (sklearn, simple RF or LR)
│   │   └── hybrid.py           # Merged/offspring strategies
│   ├── execution.py            # Alpaca order routing, fill tracking, per-model positions
│   └── data_feed.py            # Alpaca WebSocket stream + REST historical bars
│
├── evolution/                  # Evolutionary mechanics (runs at session end)
│   ├── fitness.py              # Fitness functions (Sharpe, return, drawdown)
│   ├── tournament.py           # Selection: rank models, pick survivors, mark eliminated
│   ├── crossover.py            # Parameter blending and strategy logic merging
│   └── spawn.py                # Factory: create next generation from survivors + offspring
│
├── persistence/                # Database layer
│   ├── schema.sql              # SQLite table definitions
│   ├── db.py                   # Connection pool, query helpers
│   └── models.py               # Data classes (Trade, ModelRecord, Generation, etc.)
│
├── dashboard/                  # Web dashboard
│   ├── server.py               # FastAPI app with REST + WebSocket endpoints
│   ├── broadcaster.py          # WebSocket connection manager, pushes updates to clients
│   └── static/                 # Frontend HTML/JS (or React build output)
│       ├── index.html
│       └── app.js
│
├── config.py                   # Environment variables, Alpaca keys, system parameters
├── main.py                     # Entry point: start trading engine + dashboard together
└── tests/
    ├── test_fitness.py
    ├── test_crossover.py
    └── test_execution.py
```

### Structure Rationale

- **engine/ vs evolution/:** Hard separation between market-hours real-time code and end-of-session batch processing. The engine never touches evolution logic; evolution only reads from DB and spawns the next pool. This keeps latency-sensitive code isolated.
- **persistence/:** Single SQLite database is the handoff point between engine and evolution and dashboard. No shared in-memory state between subsystems — everything flows through the DB. This makes the system restartable without losing state.
- **dashboard/:** Runs as a background thread or subprocess alongside the engine. It is a pure reader — it never writes to the DB. Decoupled from trading so a slow browser client never stalls order execution.
- **models/base.py:** A strict interface forces all strategies (technical, ML, hybrid) to be interchangeable. The arena only knows about the base class, never concrete types. This is the prerequisite for the merge system to work.

## Architectural Patterns

### Pattern 1: Strategy as Interchangeable Object (Strategy Pattern)

**What:** Each model is an instance of a class that implements a fixed interface: `on_bar(bar)`, `get_positions()`, `reset()`, `get_metadata()`. The arena calls the same methods on all models without knowing their internals.

**When to use:** Always. This is the foundation of the whole arena. Without a uniform interface, merging and swapping models becomes impossible.

**Trade-offs:** Forces discipline in how models are written; adds a layer of abstraction over raw scripts. Worth it. Hybrid offspring can only exist if both parents speak the same interface.

**Example:**
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class Bar:
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    timestamp: str

@dataclass
class Order:
    symbol: str
    side: str          # "buy" | "sell"
    qty: int
    order_type: str    # "market" | "limit"
    limit_price: Optional[float] = None

class Strategy(ABC):
    def __init__(self, model_id: str, params: dict):
        self.model_id = model_id
        self.params = params

    @abstractmethod
    def on_bar(self, bar: Bar) -> Optional[Order]:
        """Called for every new price bar. Return an Order or None."""
        ...

    @abstractmethod
    def get_metadata(self) -> dict:
        """Return serializable description of this model's type and params."""
        ...

    def reset(self):
        """Clear internal state at session start."""
        pass
```

### Pattern 2: Event-Driven Session Loop

**What:** The arena runs a tight loop during market hours. Each new bar from the Alpaca WebSocket fires an event. The arena fans that event out to all active models, collects orders, routes them to the execution handler, and logs results. No polling.

**When to use:** Whenever models need synchronized access to the same price bars. Also prevents race conditions — bars are processed sequentially, one at a time.

**Trade-offs:** Sequential fan-out means if you have 10 models and each takes 50ms to process a bar, you have 500ms of latency per bar. For 1-minute bars this is fine. For sub-second bars it becomes a bottleneck. The project explicitly rules out sub-second HFT, so sequential is correct.

**Example:**
```python
async def run_session(models: list[Strategy], data_feed, execution, db):
    async for bar in data_feed.stream():
        for model in models:
            order = model.on_bar(bar)
            if order:
                fill = await execution.submit(model.model_id, order)
                db.record_trade(fill)
        db.record_bar(bar)
```

### Pattern 3: Database as Integration Bus

**What:** The trading engine, evolution engine, and dashboard never call each other directly. They communicate exclusively through SQLite. The engine writes trades and positions. Evolution reads those records at session end, writes the next generation. The dashboard reads continuously and pushes to clients.

**When to use:** Any time you have two subsystems with different timing requirements (real-time trading vs. batch evolution vs. on-demand display). Database-as-bus makes each subsystem independently restartable.

**Trade-offs:** Slightly higher latency than in-memory queues. For this project the bottleneck is market data speed (1-minute bars), not inter-process messaging, so this is not a concern.

### Pattern 4: Fitness-First Merging

**What:** Crossover produces offspring by blending parameters from two parent models ranked by fitness. The fitter parent contributes more of its gene pool. Simple numeric parameters are blended arithmetically; categorical choices (which indicator to use) are inherited from the fitter parent.

**When to use:** At session end, when the evolution engine generates the next generation.

**Trade-offs:** Arithmetic blending works well for continuous parameters (moving average windows, thresholds). It does not naturally handle strategy type crossover (e.g., merging a momentum model with an ML model). Strategy-type crossover requires explicit rules about what can be mixed.

**Example:**
```python
def crossover(parent_a: dict, parent_b: dict, fitness_a: float, fitness_b: float) -> dict:
    """Blend params weighted by relative fitness. Parent A is assumed fitter."""
    total = fitness_a + fitness_b
    w_a = fitness_a / total
    w_b = fitness_b / total

    child_params = {}
    for key in parent_a["params"]:
        val_a = parent_a["params"][key]
        val_b = parent_b["params"][key]
        if isinstance(val_a, (int, float)):
            child_params[key] = w_a * val_a + w_b * val_b
        else:
            # Categorical: inherit from fitter parent
            child_params[key] = val_a
    return {"strategy_type": parent_a["strategy_type"], "params": child_params}
```

## Data Flow

### Market Hours Flow (Real-Time)

```
Alpaca WebSocket (bar stream)
    |
    v
data_feed.py (normalize bar, compute indicators)
    |
    v
Arena Orchestrator (fan out bar to all active models)
    |
    +---> Model 1.on_bar(bar) --> Optional[Order]
    +---> Model 2.on_bar(bar) --> Optional[Order]
    +---> ... (up to 10 models)
    |
    v
execution.py (submit orders to Alpaca paper API)
    |
    v
Alpaca fills Order --> FillEvent returned
    |
    v
persistence/db.py (record trade, update position)
    |
    v
broadcaster.py (detect new DB write, push to WebSocket clients)
    |
    v
Dashboard (display updated trade + P&L)
```

### End-of-Session Evolutionary Flow (Batch)

```
Market close (4:00 PM ET)
    |
    v
Arena Orchestrator (close all open positions, signal session end)
    |
    v
evolution/fitness.py
    (read all trades from DB for this session)
    (compute Sharpe ratio, total return, max drawdown per model)
    |
    v
evolution/tournament.py
    (rank models 1 to N by fitness score)
    (select bottom K for elimination, top M as survivors)
    (write elimination + survivor records to DB)
    |
    v
evolution/crossover.py
    (pair survivors by fitness rank)
    (produce hybrid offspring params via weighted blend)
    |
    v
evolution/spawn.py
    (instantiate survivors as-is for next generation)
    (instantiate hybrid offspring as new Strategy objects)
    (optionally mutate 1-2 params of each new model)
    (write new generation manifest to DB)
    |
    v
Arena Orchestrator
    (replace model pool with new generation)
    (ready for next market session)
```

### Dashboard Data Flow (Continuous Read)

```
FastAPI server (startup)
    |
    v
broadcaster.py (background asyncio task: poll DB every 1s)
    |
    v
WebSocket push to all connected browser clients
    |
    v
Browser JS (update model cards, trade feed, rankings table)
```

## Scaling Considerations

This is a single-operator local system. Scaling to multiple users or cloud deployment is out of scope for v1.

| Concern | Current Scale (local, 5-10 models) | If Scaling Later |
|---------|-------------------------------------|-----------------|
| Model count | 5-10 runs fine sequentially per bar | Beyond 50 models, consider async fan-out per bar |
| Data throughput | 1-min bars, ~390 bars/session, trivial | Sub-second bars require async model evaluation |
| DB writes | Low volume, SQLite handles easily | Replace SQLite with PostgreSQL if dashboard needs concurrent writes at scale |
| Dashboard clients | 1 operator browser, no concurrency concern | Add Redis pub/sub if supporting multiple concurrent viewers |

### Scaling Priorities (if ever needed)

1. **First bottleneck:** Model fan-out per bar becomes slow if models do heavy ML inference. Fix: run model evaluation in a thread pool, collect futures before moving to next bar.
2. **Second bottleneck:** SQLite write locking under high trade volume. Fix: batch writes per bar instead of per trade, or switch to PostgreSQL.

## Anti-Patterns

### Anti-Pattern 1: Shared Mutable State Between Models

**What people do:** Store a shared portfolio object that all models write to (positions, cash balance).
**Why it's wrong:** Models stomp on each other's positions. A sell from Model A erases a buy from Model B tracking the same ticker. Fitness attribution becomes impossible — you can't tell which model made which profit.
**Do this instead:** Each model owns its own virtual portfolio. Give each model its own position tracker and virtual capital allocation at session start. The execution handler routes orders per-model and tracks fills separately.

### Anti-Pattern 2: Merging by Raw Code Combination

**What people do:** Try to literally combine Python source code from two strategy files to create offspring.
**Why it's wrong:** This is fragile, usually produces broken code, and is nearly impossible to do generically. Code combination requires parsing and understanding semantics.
**Do this instead:** Represent strategies as parameter dictionaries plus a strategy type identifier. Crossover operates on parameters only. The strategy type (technical/ML/hybrid) is a categorical gene inherited from the fitter parent. This is how established GA trading systems like GeneTrader work.

### Anti-Pattern 3: Running Evolution During Market Hours

**What people do:** Continuously evolve and replace models mid-session as performance varies.
**Why it's wrong:** Creates attribution chaos — fitness changes mid-session make it impossible to compare models on equal footing. A model that had bad early hours might have recovered. Also introduces bugs where a newly spawned model has no warm-up period for indicators.
**Do this instead:** Lock the model pool for the entire trading session. Run the evolution cycle only after market close. Give each model a full session of equal opportunity before evaluation.

### Anti-Pattern 4: Overfitting the Fitness Function

**What people do:** Use raw total return as the only fitness metric.
**Why it's wrong:** A model that gets lucky with one big trade will dominate despite being inconsistent. High-return models with catastrophic drawdowns survive and destroy the portfolio in subsequent sessions.
**Do this instead:** Use Sharpe ratio as the primary fitness metric (return divided by volatility). Add a max drawdown penalty. Optionally weight win rate. This rewards consistent, risk-adjusted performance, which is what you actually want to survive and propagate.

### Anti-Pattern 5: Dashboard in the Hot Path

**What people do:** Have the trading engine call the dashboard directly on every trade to update the UI.
**Why it's wrong:** A slow browser client or a dashboard crash stalls trade execution.
**Do this instead:** Dashboard reads from the DB asynchronously on its own polling interval. The engine never knows the dashboard exists. This is the Polymarket Bot Arena pattern and it works cleanly.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| Alpaca Market Data | WebSocket streaming via alpaca-py DataStream | Subscribe to 1-minute bars for watchlist symbols; free real-time data via IEX on paper accounts |
| Alpaca Paper Trading | REST API via alpaca-py TradingClient | Separate API key from live; base URL `https://paper-api.alpaca.markets`; PDT rules apply if account < $25k |
| Alpaca Historical Data | REST via alpaca-py StockHistoricalDataClient | Use for model warm-up at session start (fetch last N bars to seed indicators) |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| Engine to Dashboard | SQLite (one-way: engine writes, dashboard reads) | Never call dashboard code from engine; decouple entirely |
| Engine to Evolution | SQLite (one-way: engine writes trades, evolution reads at session end) | Evolution is a batch job triggered after market close; no real-time coupling |
| Evolution to Engine | Python object handoff (evolution spawns Strategy instances, hands them to arena for next session) | Only coupling point between the two subsystems; use a simple registry or list |
| Model to Execution | Direct Python method call within the engine process | Models call `execution.submit(order)`, not Alpaca directly; execution is the only component that knows about Alpaca |
| Data Feed to Models | Arena orchestrator mediates; data feed never calls models directly | Keeps fan-out logic centralized and controllable |

## Suggested Build Order

Build dependencies determine what to build first. Each layer depends on the one below it.

```
1. persistence/schema.sql + persistence/db.py
   (Everything else writes to or reads from the DB. Build this first.)

2. engine/models/base.py
   (The Strategy interface is the contract everything else depends on.)

3. engine/data_feed.py
   (Can be tested standalone against Alpaca with fake model callbacks.)

4. engine/execution.py
   (Can be tested standalone by submitting test orders to paper account.)

5. engine/models/technical.py + engine/models/ml_model.py
   (First real strategies, buildable once base.py and data_feed are done.)

6. engine/arena.py
   (Wire data_feed + models + execution into the session loop.)

7. evolution/fitness.py + evolution/tournament.py
   (Pure calculation, no dependencies on Alpaca. Easy to unit test.)

8. evolution/crossover.py + evolution/spawn.py
   (Depend on fitness and tournament. Test with mock model records.)

9. dashboard/server.py + dashboard/broadcaster.py
   (Reads from DB; can be built and tested against recorded session data.)

10. dashboard/static/ (frontend)
    (Final layer; all data is already available via the API.)
```

## Sources

- Polymarket Bot Arena (open-source trading bot arena with tournament elimination, FastAPI dashboard): https://github.com/ThinkEnigmatic/polymarket-bot-arena
- Alpaca Paper Trading documentation (paper API constraints, PDT rules, IEX data): https://docs.alpaca.markets/docs/paper-trading
- alpaca-py SDK: https://alpaca.markets/sdks/python/
- Event-driven backtesting with Python (QuantStart series on component architecture): https://www.quantstart.com/articles/Event-Driven-Backtesting-with-Python-Part-I/
- DEAP: Distributed Evolutionary Algorithms in Python: https://github.com/DEAP/deap
- GeneTrader: Genetic Algorithm Optimization for Trading Strategies: https://github.com/imsatoshi/GeneTrader
- Genetic Algorithms for Trading in Python (IBKR Quant): https://www.interactivebrokers.com/campus/ibkr-quant-news/genetic-algorithms-for-trading-in-python/
- Agent-Based Genetic Algorithm for Crypto Trading (arxiv, 2025): https://arxiv.org/html/2510.07943v1
- FastAPI + WebSockets real-time dashboard (TestDriven.io): https://testdriven.io/blog/fastapi-postgres-websockets/
- NautilusTrader (reference for event-driven trading engine architecture): https://github.com/nautechsystems/nautilus_trader
- Algorithmic Trading Architecture deep dive (DEV.to, BlackRock/Tower Research case studies): https://dev.to/nashetking/algorithmic-trading-architecture-and-quants-a-deep-dive-with-case-studies-on-blackrock-and-tower-research-55ao

---
*Architecture research for: Evolutionary Algorithmic Trading Arena*
*Researched: 2026-03-02*
