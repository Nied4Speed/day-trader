# Project Research Summary

**Project:** Day Trader — Evolutionary Algorithmic Trading Arena
**Domain:** Local multi-model trading system with genetic selection and paper trading via Alpaca
**Researched:** 2026-03-02
**Confidence:** MEDIUM-HIGH

## Executive Summary

This is a novel category of software: a self-improving algorithmic trading arena where multiple trading strategies compete in real time under paper trading conditions, and genetic operators (elimination, crossover, mutation) evolve the strategy pool across sessions. The closest reference points are open-source tournament arenas like Polymarket Bot Arena and academic GA trading systems like GeneTrader, but the combination of live market data, real-time paper execution, and a persistent generational loop in a local single-operator deployment is genuinely uncommon. The recommended approach follows a strict layered build order: persistence first, then strategy interface, then data and execution, then the evolutionary mechanics, then the dashboard. Nothing can be built out of this order because each layer is a hard dependency for the next.

The core technical recommendation is Python 3.12 with alpaca-py for brokerage connectivity, DEAP for genetic operators, FastAPI for the dashboard backend, and React for the frontend. SQLite is the single source of truth and the integration bus between three independent subsystems (trading engine, evolution engine, dashboard). The architecture's most important constraint is that the dashboard must never be in the hot path of order execution. Equally critical: all strategies must implement a common abstract interface before any crossover or merging is attempted, because the merge mechanism operates on parameter dictionaries, not source code.

The primary risks are not technical but methodological. Degenerate fitness functions that reward buy-and-hold behavior, look-ahead bias in strategy signal logic, and premature population convergence from overly aggressive elimination will all produce an arena that appears functional but produces meaningless results. These three issues, combined with Alpaca paper trading's known data reliability quirks (streaming delays on non-funded accounts, no slippage simulation), are the failure modes that require attention from the first line of code, not from the last.

## Key Findings

### Recommended Stack

The stack splits cleanly into two runtimes that never share process space. The Python trading engine handles all computation: alpaca-py drives Alpaca connectivity, pandas and pandas-ta handle market data and indicator computation, DEAP manages genetic operators, scikit-learn supports ML-based strategy types, APScheduler handles market-hours scheduling, and SQLite via SQLAlchemy persists everything. The React dashboard runs in the browser and connects via WebSocket only.

Key version decisions: Python 3.12 (avoid 3.13+ until the ecosystem stabilizes), pandas-ta over TA-Lib (no C compiler dependency on macOS), DEAP over PyGAD (custom individual representations and multi-objective fitness are both required), multiprocessing over threading (GIL prevents true CPU parallelism for signal computation), and APScheduler 3.x over Celery (zero infrastructure dependencies for a local single-machine system).

**Core technologies:**
- Python 3.12: Runtime for all trading, data, and evolutionary logic — best ecosystem for this domain, stable library support
- alpaca-py 0.43.2: Paper trading execution and market data streaming — official SDK, WebSocket streaming, REST historical data
- DEAP 1.4.x: Genetic algorithm operators — supports multi-objective fitness and custom individual representations required for parameter-dict crossover
- FastAPI + React 18 + Vite 6: Dashboard stack — async-native WebSocket server with fast-iteration frontend
- SQLite + SQLAlchemy 2.x: Local persistence and integration bus — zero-ops, sufficient for 5-10 models at 1-minute bar resolution
- pandas-ta 0.3.14b: Technical indicators — pure Python, 150+ indicators, no build dependencies
- APScheduler 3.x: Market-hours scheduling — asyncio-compatible, in-process, no broker required

### Expected Features

The research found a clear dependency chain that determines what must be built first. Alpaca API integration is the root dependency for everything else. Performance tracking cannot happen without trade execution, and the evolutionary loop cannot happen without performance tracking. This creates a strict build order that the roadmap must respect.

**Must have (table stakes):**
- Alpaca API connection (paper trading keys, market data WebSocket) — the entire system rests on this
- Per-model trade execution and fill tracking — required to compute meaningful performance
- Real-time market data streaming via WebSocket — polling is too slow and burns rate limits
- Per-model performance metrics: Sharpe ratio, P&L, max drawdown, win rate — minimum viable fitness scoring
- Model rankings updated in real time — core competitive mechanic without which the arena has no visible output
- Basic risk limits per model (max position size, max daily loss) — prevents runaway models from poisoning session data
- Session-end elimination, crossover/merge, mutation, and next-generation spawn — these four together form the complete evolutionary loop
- Web dashboard: equity curves, trade log, rankings, generation number — CLI alone is insufficient to observe the competition
- Generational log with parentage records — required to validate that generations actually improve over time

**Should have (competitive):**
- Walk-forward out-of-sample validation — differentiates rigorous research from reckless optimization
- Configurable fitness function (Sharpe, Sortino, Calmar, raw return) — needed once first generations surface which metric captures strategy quality
- Strategy type diversity enforcement — prevents population collapse to a single strategy family
- Generational lineage visualization (DAG tree view in dashboard) — enables post-mortems on which ancestors produced winners
- Mutation and crossover parameter logging — required for debugging why offspring fail

**Defer (v2+):**
- Live trading graduation pipeline — meaningful only after multiple generations of stable paper performance
- Multi-asset expansion (ETFs, crypto) — US equities must be proven first
- Advanced ML model types (RL agents, transformers) — exotic types add training complexity before the loop is validated
- Cloud deployment — local machine is sufficient for the experimental phase

### Architecture Approach

The architecture uses three independent subsystems communicating exclusively through SQLite: a trading engine (real-time, market-hours only), an evolution engine (batch job at session end), and a dashboard (continuous read-only). Each subsystem can crash and restart independently without corrupting the others. The trading engine fans each incoming bar to all model instances sequentially via an event-driven loop. The evolution engine reads session records at 4:00 PM ET, computes fitness, runs selection, generates offspring, and writes the next generation manifest. The dashboard polls SQLite every second and pushes updates to connected browser clients via WebSocket.

**Major components:**
1. Arena Orchestrator — market-hours session loop, triggers evolution at session end, owns the model pool
2. Strategy (Model) — implements `on_bar(bar) -> Optional[Order]`, owns its own position ledger and capital allocation
3. Execution Handler — sole component that calls Alpaca REST API; all models route orders through it
4. Data Feed — single Alpaca WebSocket connection, fans data to all model queues
5. Fitness Ranker + Tournament Selector — batch computation of Sharpe/drawdown/frequency metrics, ranks and eliminates
6. Crossover + Spawn — parameter-dict blending weighted by fitness, factory for next-generation Strategy instances
7. SQLite Persistence Layer — integration bus between all subsystems, holds trades, positions, model metadata, generational lineage
8. FastAPI + WebSocket Broadcaster — read-only dashboard backend, polls DB and pushes to browser clients

### Critical Pitfalls

1. **Degenerate fitness function converges to buy-and-hold** — Use a multi-objective fitness function from session one: expected value per trade (not total return), minimum trade frequency floor, Sortino ratio, and a max drawdown penalty. A single-metric fitness function will always be gamed by the optimizer. If survivor models start showing very low trade counts after 2-3 generations, the fitness function is already broken.

2. **Look-ahead bias in strategy signal logic** — Build the event-driven session loop before writing any strategy. The loop structurally prevents this by passing only completed bars to models. Never use current in-progress bar price for entry decisions. Validate by running strategies against random-walk data — any significant alpha on random data means look-ahead bias is present.

3. **Premature convergence kills population diversity** — Eliminate at most 30-40% of models per generation. Evaluate on rolling multi-session performance, not single-session results. Include mutation-only offspring in every generation to continuously introduce novel variants. If all survivors share the same indicator type within 3 generations, the elimination rate is too aggressive.

4. **Alpaca paper trading gives false confidence** — Paper mode assumes perfect fills at last trade price with zero slippage. The free IEX data feed can lag by 20 minutes to 2 hours on non-funded accounts. Apply simulated transaction costs (0.05% per side minimum) to all paper results and validate WebSocket timestamps against system clock before trusting any signal timing.

5. **Shared state corruption across concurrent models** — Each model must own its own position ledger and capital allocation. A single Alpaca WebSocket connection fans data to per-model queues via the arena coordinator. A central order router serializes order submission. No model ever calls Alpaca directly or reads another model's state.

## Implications for Roadmap

The feature dependency chain and pitfall-to-phase mapping from the research directly determine the build order. Every shortcut from this sequence produces hidden bugs that become expensive to fix after the evolutionary loop is running.

### Phase 1: Foundation and Infrastructure

**Rationale:** SQLite schema, the abstract Strategy interface, and the Alpaca data feed are the dependencies for everything else. The fitness function must also be designed here, before any strategy is written, to avoid discovering degenerate convergence after the loop has already produced corrupted generations.
**Delivers:** A running Alpaca WebSocket connection streaming bars into a normalized data structure, an abstract Strategy base class that all models will implement, a SQLite schema covering trades/positions/models/generations, and a validated multi-objective fitness function design.
**Addresses:** Alpaca API connection, data streaming, persistence schema, fitness function definition.
**Avoids:** Look-ahead bias (correct event loop built first), degenerate fitness (fitness function locked before any strategy is written), shared state corruption (per-model position ledger designed here).

### Phase 2: Strategy Engine and Execution

**Rationale:** With the base interface and data feed in place, concrete strategy implementations and the execution handler can be built and tested in isolation against the paper account before the arena orchestrator is added.
**Delivers:** 4-6 starting strategies with genuine diversity (moving average crossover, RSI mean reversion, momentum, ML-based), a central execution handler routing orders to Alpaca paper API, and per-model position and fill tracking.
**Addresses:** Per-model trade execution, real-time market data routing to models, basic risk limits.
**Avoids:** Look-ahead bias (strategies use completed-bar-only event interface), shared state corruption (execution handler serializes all order submission).

### Phase 3: Arena Orchestrator and Session Loop

**Rationale:** With strategies and execution validated independently, the orchestrator wires them into a session loop. This is the first time the full real-time trading path runs end-to-end.
**Delivers:** Market-hours gating (9:30-4:00 ET via APScheduler), session start/stop lifecycle, bar fan-out to all model instances, real-time performance metric computation, model rankings updated per bar.
**Addresses:** Model rankings/leaderboard, graceful market-hours gating, session summary after close.
**Avoids:** Alpaca paper trading pitfalls (slippage simulation layer added here before the arena's first session).

### Phase 4: Evolutionary Loop

**Rationale:** The evolutionary mechanics are a batch job that runs after market close on session data already in SQLite. They have no real-time dependencies, which makes them independently testable with recorded session data.
**Delivers:** Session-end elimination (bottom N models removed), crossover/merge producing hybrid offspring, mutation with bounded parameter perturbation, automatic generation spawn for next session, generational log with parentage records.
**Addresses:** Competitive elimination, strategy merging, next-generation spawn, generational history.
**Avoids:** Premature convergence (elimination rate capped at 30-40%, rolling multi-session evaluation required before elimination, mutation-only offspring included in every generation).

### Phase 5: Dashboard

**Rationale:** The dashboard reads from a database that is already fully populated by Phase 4. It has no dependencies on real-time trading logic and can be built against recorded data.
**Delivers:** React dashboard with live equity curves per model, real-time trade feed, model rankings table, generation number and history, model health indicators (last trade timestamp, flagging frozen models).
**Addresses:** Web dashboard requirement for equity curves/trade log/rankings, generation display.
**Avoids:** Dashboard in the hot path (FastAPI reads from SQLite asynchronously; engine never calls dashboard code directly).

### Phase 6: Hardening and Validation Mechanics

**Rationale:** Once multiple generations have run, the gaps that cannot be designed in advance become visible: which fitness metric actually captures strategy quality, whether the population is converging too fast, whether slippage assumptions were correct.
**Delivers:** Configurable fitness function (swap between Sharpe, Sortino, Calmar), strategy type diversity enforcement, walk-forward out-of-sample validation layer, mutation and crossover parameter logging, generational lineage visualization in dashboard.
**Addresses:** Walk-forward validation, configurable fitness, diversity enforcement, detailed logging.
**Avoids:** All "looks done but isn't" checklist items from PITFALLS.md.

### Phase Ordering Rationale

- SQLite schema before strategies: the DB is the integration bus; building it first means nothing ever needs to be retrofitted to match
- Fitness function before strategy code: once strategies are running and generating results, changing the fitness function invalidates all comparisons between generations; it must be locked early
- Execution handler before arena orchestrator: isolating and testing the Alpaca paper API integration before adding the complexity of 5-10 concurrent models eliminates a major source of hard-to-reproduce bugs
- Evolutionary loop before dashboard: the loop should be validated with a simple CLI observer before adding the complexity of real-time WebSocket push
- Dashboard last: it has no dependencies that aren't already solved; building it last means it can be built against real data instead of mocked data

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 1 (Fitness function design):** The multi-objective fitness function for a combined Sharpe/frequency/drawdown scoring requires careful parameterization. Research the specific weighting and minimum frequency floor values before implementation.
- **Phase 4 (Crossover/merge mechanics):** The parameter-dict crossover approach (weighted blend by relative fitness) works for continuous parameters but requires explicit design decisions for categorical parameters (which indicator, which strategy type). Research GeneTrader's crossover implementation for concrete patterns.
- **Phase 4 (Multi-session rolling evaluation):** Alpaca paper account does not preserve cross-session position state automatically. Research how to snapshot and restore virtual portfolio state across sessions in SQLite before implementing rolling evaluation windows.

Phases with standard patterns (skip research-phase):
- **Phase 2 (Strategy implementations):** Moving average crossover, RSI mean reversion, and momentum strategies are thoroughly documented in the quantitative trading literature. MACD/RSI implementations via pandas-ta are straightforward.
- **Phase 3 (APScheduler market-hours gating):** Standard cron/interval scheduling with APScheduler is well-documented; ET timezone handling with `pytz` or `zoneinfo` is established pattern.
- **Phase 5 (Dashboard):** FastAPI WebSocket push, React with Recharts and lightweight-charts, and react-use-websocket all have thorough documentation and examples.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Core technologies (alpaca-py, FastAPI, SQLite, pandas) verified via official sources and PyPI; DEAP and pandas-ta verified at MEDIUM due to WebSearch sourcing |
| Features | MEDIUM-HIGH | Table stakes features verified against Alpaca official docs; evolutionary feature requirements derived from analogous systems (GeneTrader, Polymarket Bot Arena, academic research) |
| Architecture | MEDIUM-HIGH | Event-driven trading architecture pattern is well-established (QuantStart series, NautilusTrader reference); tournament elimination pattern verified via Polymarket Bot Arena open-source code |
| Pitfalls | MEDIUM-HIGH | Core algo trading pitfalls (look-ahead bias, fitness degeneration, survivorship bias) verified across multiple practitioner and academic sources; Alpaca-specific issues sourced from community forums (lower confidence) |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- **Alpaca free tier data reliability:** Community forum reports of 20-minute to 2-hour streaming delays on non-funded paper accounts are from 2022-2023. Validate that current account conditions reproduce or resolve this before building timing-sensitive signal logic that depends on stream freshness.
- **Multi-session portfolio state continuity:** The research does not address how Alpaca paper accounts handle overnight position carryover or whether per-model virtual capital allocations need to be reconstructed from SQLite at each session start. Validate this during Phase 1 infrastructure work.
- **Crossover for mixed strategy types:** Arithmetic parameter blending works when both parents share the same strategy type. The research identifies this gap but defers the solution ("requires explicit rules about what can be mixed"). This needs a concrete design decision before Phase 4.
- **ML model staleness and retraining cadence:** The "looks done but isn't" checklist in PITFALLS.md flags that ML-based strategies need a defined retraining schedule. The research does not specify when in the generational cycle ML models should retrain. Design this explicitly during Phase 2 or Phase 4 planning.

## Sources

### Primary (HIGH confidence)
- [alpaca-py on PyPI and GitHub](https://pypi.org/project/alpaca-py/) — version confirmation, Python compatibility, WebSocket streaming protocol
- [Alpaca Paper Trading Docs](https://docs.alpaca.markets/docs/paper-trading) — paper API constraints, PDT rules, IEX data limitations
- [scikit-learn on PyPI](https://pypi.org/project/scikit-learn/) — version 1.6.x Python 3.12 compatibility
- [lightweight-charts GitHub](https://github.com/tradingview/lightweight-charts) — 45kb canvas financial charts, React integration
- [Scientific Python SPEC 0](https://scientific-python.org/specs/spec-0000/) — version support windows for numpy/pandas/scikit-learn
- [Agent-Based Genetic Algorithm for Crypto Trading (arxiv 2025)](https://arxiv.org/html/2510.07943v1) — peer-reviewed GA trading architecture

### Secondary (MEDIUM confidence)
- [Polymarket Bot Arena (open source)](https://github.com/ThinkEnigmatic/polymarket-bot-arena) — tournament elimination pattern, FastAPI dashboard architecture
- [GeneTrader (open source)](https://github.com/imsatoshi/GeneTrader) — parameter-dict crossover, genetic operator patterns for trading
- [DEAP GitHub](https://github.com/DEAP/deap) — multi-objective GA, NSGA-II, custom toolbox
- [QuantStart Event-Driven Architecture series](https://www.quantstart.com/articles/Event-Driven-Backtesting-with-Python-Part-I/) — component isolation patterns
- [Genetic Algorithms for Trading (IBKR Quant)](https://www.interactivebrokers.com/campus/ibkr-quant-news/genetic-algorithms-for-trading-in-python/) — fitness function design patterns
- [Survivorship Bias in Backtesting (QuantRocket)](https://www.quantrocket.com/blog/survivorship-bias/) — universe selection methodology

### Tertiary (LOW confidence)
- [Alpaca Community Forum: streaming delays](https://forum.alpaca.markets/t/significant-delay-in-market-streaming-data-live-paper/2449) — Alpaca paper data reliability (older threads, needs current validation)
- [Alpaca Community Forum: slippage](https://forum.alpaca.markets/t/slippage-paper-trading-vs-real-trading/2801) — paper vs. live fill discrepancy (community sourced)
- [Selection Pressure and Genetic Convergence (Springer)](https://link.springer.com/article/10.1007/s40747-019-0102-7) — elimination rate thresholds (academic, not trading-specific)

---
*Research completed: 2026-03-02*
*Ready for roadmap: yes*
