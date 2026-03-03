# Requirements: Day Trader

**Defined:** 2026-03-02
**Core Value:** The competitive elimination loop must work: models compete, losers die, winners merge, and the next generation is measurably better than the last.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Data Infrastructure

- [ ] **DATA-01**: System connects to Alpaca API with paper trading credentials
- [ ] **DATA-02**: System streams real-time market data via Alpaca WebSocket (bars at 1-minute resolution)
- [ ] **DATA-03**: System fetches historical bar data for strategy warm-up periods
- [ ] **DATA-04**: All market data, trades, positions, and model metadata persist in SQLite
- [ ] **DATA-05**: System applies simulated transaction costs (minimum 0.05% per side) to all paper trades

### Strategy Engine

- [ ] **STRT-01**: All strategies implement a common abstract interface (`on_bar(bar) -> Optional[Order]`)
- [ ] **STRT-02**: System ships with 4-6 diverse starting strategies (moving average crossover, RSI mean reversion, momentum, ML-based, and at least one additional)
- [ ] **STRT-03**: Each strategy owns its own isolated position ledger and capital allocation
- [ ] **STRT-04**: Strategies represent their tunable parameters as parameter dictionaries for crossover compatibility
- [ ] **STRT-05**: Strategies only receive completed bars (no access to current in-progress bar data)

### Execution

- [ ] **EXEC-01**: A central execution handler routes all model orders to Alpaca paper trading API
- [ ] **EXEC-02**: System tracks order placement, fills, and rejections per model
- [ ] **EXEC-03**: Each model has configurable risk limits (max position size, max daily loss, max open positions)
- [ ] **EXEC-04**: No model calls Alpaca directly; all orders go through the central execution handler

### Performance Tracking

- [ ] **PERF-01**: System computes per-model metrics in real time: P&L, return %, Sharpe ratio, max drawdown, win rate
- [ ] **PERF-02**: System maintains a live ranked leaderboard of all active models
- [ ] **PERF-03**: System generates a session summary after market close (final rankings, elimination decisions, merges triggered)
- [ ] **PERF-04**: System tracks per-model equity curves over time

### Arena Orchestrator

- [ ] **ARNA-01**: System gates trading activity to US market hours (9:30 AM - 4:00 PM ET)
- [ ] **ARNA-02**: System manages session lifecycle: startup, bar fan-out to all models, shutdown
- [ ] **ARNA-03**: System runs 5-10 models concurrently during each trading session
- [ ] **ARNA-04**: A single Alpaca WebSocket connection fans data to per-model queues via the orchestrator

### Evolution

- [ ] **EVOL-01**: System eliminates the bottom-performing models at end of each session (configurable elimination rate, default 20-30%)
- [ ] **EVOL-02**: Surviving models merge their best parameter subsets via crossover to produce hybrid offspring
- [ ] **EVOL-03**: Offspring parameters are mutated with bounded perturbation (+/- 10-20% of parameter range)
- [ ] **EVOL-04**: New models are spawned automatically to fill the pool for the next session
- [ ] **EVOL-05**: System uses multi-objective fitness function (not single metric) for elimination decisions
- [ ] **EVOL-06**: System logs generational history: which models existed, their scores, parentage, and genetic operations applied

### Dashboard

- [ ] **DASH-01**: Web dashboard displays live equity curves per model during trading sessions
- [ ] **DASH-02**: Web dashboard shows real-time trade feed with model attribution
- [ ] **DASH-03**: Web dashboard displays model rankings table updated in real time
- [ ] **DASH-04**: Web dashboard shows current generation number and generational history
- [ ] **DASH-05**: Dashboard updates via WebSocket push (not polling from browser)

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Validation

- **VALD-01**: Walk-forward out-of-sample validation before model graduation to live trading
- **VALD-02**: Configurable fitness function (swap between Sharpe, Sortino, Calmar, raw return)
- **VALD-03**: Strategy type diversity enforcement to prevent population collapse

### Visualization

- **VIZN-01**: Generational lineage visualization (DAG/tree view in dashboard)
- **VIZN-02**: Session replay: step through a past session's trades and decisions
- **VIZN-03**: Detailed mutation and crossover parameter logging in dashboard

### Production

- **PROD-01**: Live trading graduation pipeline with circuit breakers
- **PROD-02**: Cloud deployment for always-on execution
- **PROD-03**: Multi-asset expansion (ETFs, crypto via Alpaca)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Live/real money trading | Not until a model proves consistent profitability across many paper generations |
| High-frequency / sub-second trading | Local machine + Alpaca paper trading cannot simulate queue position; meaningless results |
| Multi-user / team features | Single-operator research tool; adds complexity with no benefit |
| Mobile app | Web dashboard is sufficient |
| Options / forex / crypto | US stocks only until the evolutionary loop is validated on equities |
| AutoML / hyperparameter tuning frameworks | The evolutionary loop IS the optimization; grid search is redundant |
| Social / copy trading features | Different product entirely; regulatory and liability concerns |
| Real-time push notifications | Dashboard is visible during market hours; terminal logging is sufficient |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 - Foundation | Pending |
| DATA-02 | Phase 1 - Foundation | Pending |
| DATA-03 | Phase 1 - Foundation | Pending |
| DATA-04 | Phase 1 - Foundation | Pending |
| DATA-05 | Phase 1 - Foundation | Pending |
| STRT-01 | Phase 1 - Foundation | Pending |
| STRT-05 | Phase 1 - Foundation | Pending |
| STRT-02 | Phase 2 - Strategy Engine and Execution | Pending |
| STRT-03 | Phase 2 - Strategy Engine and Execution | Pending |
| STRT-04 | Phase 2 - Strategy Engine and Execution | Pending |
| EXEC-01 | Phase 2 - Strategy Engine and Execution | Pending |
| EXEC-02 | Phase 2 - Strategy Engine and Execution | Pending |
| EXEC-03 | Phase 2 - Strategy Engine and Execution | Pending |
| EXEC-04 | Phase 2 - Strategy Engine and Execution | Pending |
| PERF-01 | Phase 3 - Arena Orchestrator | Pending |
| PERF-02 | Phase 3 - Arena Orchestrator | Pending |
| PERF-03 | Phase 3 - Arena Orchestrator | Pending |
| PERF-04 | Phase 3 - Arena Orchestrator | Pending |
| ARNA-01 | Phase 3 - Arena Orchestrator | Pending |
| ARNA-02 | Phase 3 - Arena Orchestrator | Pending |
| ARNA-03 | Phase 3 - Arena Orchestrator | Pending |
| ARNA-04 | Phase 3 - Arena Orchestrator | Pending |
| EVOL-01 | Phase 4 - Evolutionary Loop | Pending |
| EVOL-02 | Phase 4 - Evolutionary Loop | Pending |
| EVOL-03 | Phase 4 - Evolutionary Loop | Pending |
| EVOL-04 | Phase 4 - Evolutionary Loop | Pending |
| EVOL-05 | Phase 4 - Evolutionary Loop | Pending |
| EVOL-06 | Phase 4 - Evolutionary Loop | Pending |
| DASH-01 | Phase 5 - Dashboard | Pending |
| DASH-02 | Phase 5 - Dashboard | Pending |
| DASH-03 | Phase 5 - Dashboard | Pending |
| DASH-04 | Phase 5 - Dashboard | Pending |
| DASH-05 | Phase 5 - Dashboard | Pending |

**Coverage:**
- v1 requirements: 31 total
- Mapped to phases: 31
- Unmapped: 0

---
*Requirements defined: 2026-03-02*
*Last updated: 2026-03-02 after roadmap creation*
