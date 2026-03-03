# Roadmap: Day Trader

## Overview

Build the evolutionary trading arena in strict dependency order: persistence and data infrastructure first, then concrete strategies and execution, then the arena orchestrator that wires them into a live session loop, then the evolutionary mechanics that make generations improve, then the dashboard that makes the competition observable. Each phase delivers a coherent, independently testable capability. Nothing is built out of this order because each layer is a hard dependency for the next.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Foundation** - SQLite schema, abstract strategy interface, Alpaca data feed, and fitness function design
- [ ] **Phase 2: Strategy Engine and Execution** - 4-6 concrete strategies, central execution handler, per-model risk limits
- [ ] **Phase 3: Arena Orchestrator** - Session lifecycle, bar fan-out to all models, real-time performance tracking and rankings
- [ ] **Phase 4: Evolutionary Loop** - Session-end elimination, crossover/merge, mutation, generational spawn and logging
- [ ] **Phase 5: Dashboard** - React frontend with live equity curves, trade feed, rankings, and generation history

## Phase Details

### Phase 1: Foundation
**Goal**: The core infrastructure is in place so that any strategy can be written against a stable interface, all data is persisted from the first bar, and the fitness function is locked before any strategy produces results that could be gamed by changing it later.
**Depends on**: Nothing (first phase)
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, STRT-01, STRT-05
**Success Criteria** (what must be TRUE):
  1. The system connects to Alpaca with paper trading credentials and streams 1-minute bars via WebSocket without error
  2. Historical bar data can be fetched for any symbol to warm up strategy state before live trading begins
  3. All market data, model metadata, trades, and positions are written to SQLite from the moment the first bar arrives
  4. A concrete strategy can be written by implementing a single abstract interface (`on_bar(bar) -> Optional[Order]`) with no access to in-progress bar data
  5. Simulated transaction costs (minimum 0.05% per side) are applied to every paper trade recorded in the database
**Plans**: TBD

Plans:
- [ ] 01-01: Project scaffolding, SQLite schema, and SQLAlchemy models
- [ ] 01-02: Alpaca connection, WebSocket data feed, and historical bar fetching
- [ ] 01-03: Abstract strategy interface, multi-objective fitness function design

### Phase 2: Strategy Engine and Execution
**Goal**: Multiple concrete strategies exist and can execute paper trades through a single centralized handler, with per-model risk limits enforced before any order reaches Alpaca.
**Depends on**: Phase 1
**Requirements**: STRT-02, STRT-03, STRT-04, EXEC-01, EXEC-02, EXEC-03, EXEC-04
**Success Criteria** (what must be TRUE):
  1. At least 4 distinct strategies are running and generating orders against the paper account (moving average crossover, RSI mean reversion, momentum, ML-based)
  2. Each strategy owns its own isolated position ledger and capital allocation that no other strategy can read or modify
  3. All order submission flows through the central execution handler; no strategy calls Alpaca directly
  4. Order placements, fills, and rejections are tracked per model in SQLite
  5. A model that exceeds its max daily loss, max position size, or max open positions is blocked from placing further orders for that session
**Plans**: TBD

Plans:
- [ ] 02-01: Central execution handler with Alpaca paper API routing and fill tracking
- [ ] 02-02: Per-model risk limits and position ledger isolation
- [ ] 02-03: Concrete strategy implementations (MA crossover, RSI mean reversion, momentum, ML-based)

### Phase 3: Arena Orchestrator
**Goal**: All models compete simultaneously during market hours in a managed session loop, with real-time performance metrics and a ranked leaderboard updating on every bar.
**Depends on**: Phase 2
**Requirements**: PERF-01, PERF-02, PERF-03, PERF-04, ARNA-01, ARNA-02, ARNA-03, ARNA-04
**Success Criteria** (what must be TRUE):
  1. The system automatically starts trading at 9:30 AM ET and stops at 4:00 PM ET without manual intervention
  2. A single Alpaca WebSocket connection fans each incoming bar to all 5-10 model queues simultaneously
  3. Per-model P&L, return percentage, Sharpe ratio, max drawdown, and win rate are computed and updated in real time
  4. A ranked leaderboard of all active models is queryable after every bar
  5. A session summary is generated at market close with final rankings, which models were flagged for elimination, and which merges were triggered
**Plans**: TBD

Plans:
- [ ] 03-01: Arena orchestrator, market-hours gating (APScheduler), and session lifecycle management
- [ ] 03-02: Bar fan-out to per-model queues and concurrent model execution
- [ ] 03-03: Real-time performance metric computation, equity curve tracking, and ranked leaderboard

### Phase 4: Evolutionary Loop
**Goal**: After each trading session, the arena automatically eliminates the worst models, merges survivors into hybrid offspring, mutates parameters, and spawns the next generation, with a full generational log recording parentage and genetic operations.
**Depends on**: Phase 3
**Requirements**: EVOL-01, EVOL-02, EVOL-03, EVOL-04, EVOL-05, EVOL-06
**Success Criteria** (what must be TRUE):
  1. At the end of each session, the bottom 20-30% of models are eliminated based on a multi-objective fitness score (not a single metric)
  2. Surviving models merge their best parameter subsets via crossover to produce hybrid offspring, with new models filling the pool to the configured size
  3. Each offspring's parameters are mutated with bounded perturbation (+/- 10-20% of parameter range) so no two generations are identical
  4. The system populates the model pool for the next session automatically without manual intervention
  5. The generational log records each model's parentage, fitness scores, and which genetic operations were applied, so the lineage of any model can be traced back to generation 1
**Plans**: TBD

Plans:
- [ ] 04-01: Multi-objective fitness scoring and tournament selection/elimination
- [ ] 04-02: Parameter-dict crossover, mutation with bounded perturbation, and offspring generation
- [ ] 04-03: Generational log, parentage tracking, and next-session pool initialization

### Phase 5: Dashboard
**Goal**: The competition is observable through a web browser with live equity curves, a real-time trade feed, model rankings, and generational history, all pushed via WebSocket without polling.
**Depends on**: Phase 4
**Requirements**: DASH-01, DASH-02, DASH-03, DASH-04, DASH-05
**Success Criteria** (what must be TRUE):
  1. A web browser shows live equity curves for all active models during a trading session, updating as trades are executed
  2. A real-time trade feed shows every order placed, attributed to the model that placed it
  3. The model rankings table updates in the browser as each bar is processed, without requiring a page refresh
  4. The current generation number and a summary of past generations are visible in the dashboard
  5. All dashboard updates arrive via WebSocket push from the server; the browser never polls for data
**Plans**: TBD

Plans:
- [ ] 05-01: FastAPI backend with SQLite read layer and WebSocket broadcaster
- [ ] 05-02: React frontend with equity curves (lightweight-charts), trade feed, and rankings table
- [ ] 05-03: Generation history view and WebSocket integration wiring

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 0/3 | Not started | - |
| 2. Strategy Engine and Execution | 0/3 | Not started | - |
| 3. Arena Orchestrator | 0/3 | Not started | - |
| 4. Evolutionary Loop | 0/3 | Not started | - |
| 5. Dashboard | 0/3 | Not started | - |
