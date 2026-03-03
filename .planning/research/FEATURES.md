# Feature Research

**Domain:** Evolutionary algorithmic trading arena (multi-model competition with genetic selection)
**Researched:** 2026-03-02
**Confidence:** MEDIUM (core trading platform features HIGH, evolutionary/arena-specific features MEDIUM based on limited direct comparisons)

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features the operator assumes exist. Missing these makes the system feel broken or untrustworthy.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Paper trade execution via Alpaca | The entire system depends on placing and tracking orders; without this nothing works | HIGH | Alpaca supports market, limit, stop, stop-limit, bracket, trailing stop in paper mode. Rate limit: 200 req/min |
| Real-time market data streaming | Models need price data to make decisions; polling is too slow and burns rate limits | MEDIUM | Alpaca provides free IEX feed (8-10% market volume) or paid comprehensive feed. Use WebSocket for streaming |
| Per-model performance tracking | Can't eliminate or rank models without knowing who's winning | MEDIUM | Minimum: P&L, return %, trade count. Expected: Sharpe ratio, max drawdown, win rate |
| Live equity curve per model | Standard way to see if a model is improving or deteriorating over a session | MEDIUM | A chart of portfolio value over time. Operators expect this from any trading dashboard |
| Trade log / order history | Debugging requires knowing what each model actually did and when | LOW | Timestamp, ticker, direction, quantity, fill price, model ID |
| Model rankings / leaderboard | Core competitive mechanic; without a ranked list the arena has no visible winner | LOW | Sorted by primary metric (Sharpe or return); updates continuously during market hours |
| Session summary after market close | The operator needs to know what happened today: who survived, who got cut | MEDIUM | Summary of final rankings, elimination decisions, what merges were triggered |
| Graceful market hours gating | System should not attempt to trade or generate signals outside 9:30-4:00 ET | LOW | Schedule-aware start/stop; handle early close days |
| Basic risk limits per model | Prevent a runaway model from blowing up its paper account in one session | MEDIUM | Max position size, max loss per session, max open positions. Without this a broken model causes noise |

### Differentiators (Competitive Advantage)

Features that make this system distinct. The core value is the evolutionary loop — everything else is supporting cast.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Competitive elimination loop | The defining mechanic. Worst N models per generation are removed, forcing selection pressure | HIGH | Requires: evaluation window, elimination threshold, configurable N. Must be deterministic so you can debug "why was this model cut?" |
| Strategy crossover / merging | Survivors exchange their best parameter subsets to produce hybrid offspring | HIGH | Key design decision: what gets merged? Parameter weights, indicator combinations, position sizing rules. Start simple: take top K params from each parent |
| Automatic spawn of next generation | After elimination and merging, new models fill the pool for the next session without manual intervention | HIGH | Triggers at end of session. Spawns variants via mutation (parameter perturbation) + crossover offspring |
| Mutation with bounded perturbation | Prevents the population from converging on a local optimum | MEDIUM | Mutate parameters by +/- 10-20% of their range. Track mutation rate as a logged variable |
| Generational history and lineage tree | Shows which models descended from which, and what changed across generations | MEDIUM | This is the "replay value" of the arena. Lets the operator trace why a particular model is good. Store as a DAG: parent IDs, merge recipe, generation number |
| Per-strategy type diversity enforcement | Ensures the arena doesn't converge on one strategy family too quickly | MEDIUM | Maintain at least one representative of each strategy type (technical, ML, hybrid). Eliminate the weakest within a type, not globally, if needed |
| Walk-forward out-of-sample validation before graduation | Before a model graduates to live trading, it must prove performance on unseen time periods | HIGH | Essential for catching overfit models. Use rolling windows. This is what separates rigorous from reckless |
| Configurable fitness function | The metric used to rank and eliminate models. Different operators care about different tradeoffs | MEDIUM | Default: Sharpe ratio. Allow switching to Calmar (return/max drawdown), Sortino, or raw return. Store what metric was used per generation |
| Mutation / crossover parameter logging | Records what genetic operation produced each model, enabling post-mortems on why offspring failed | LOW | Parent IDs, operation type, parameters changed, generation number. Low complexity but high value for debugging |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Live money trading in v1 | "What's the point if it's fake?" | A single broken model can cause real financial loss. The evolutionary loop needs many generations of paper trading to prove stability before real capital is warranted | Graduate to live only after consistent multi-generation paper performance. Keep live trading as a milestone gate, not a launch feature |
| Sub-second / high-frequency execution | Maximize trade opportunities | Requires colocation, co-tenancy with exchange, and ultra-low-latency infrastructure that a local machine cannot provide. Also: Alpaca paper trading doesn't simulate queue position, so HFT backtests are meaningless there | Target minute-bar or 5-minute-bar resolution. More than enough signal for a meaningful evolutionary experiment |
| Optimization by exhaustive grid search | "Try all parameter combinations" | Grid search over even 5 parameters with 10 values each is 100,000 backtests. Computationally infeasible locally, and the evolutionary loop IS the optimization | Let the generational loop drive optimization. Use mutation for exploration, crossover for exploitation |
| Real-time alerts / push notifications | "Notify me when a model is eliminated" | Adds infrastructure complexity (email, SMS, webhooks) for a single-operator local system | The dashboard is visible during market hours. Add a simple terminal bell or log-line notification at most |
| Multi-user / team collaboration | "Let others watch" | Session state, access control, and conflict resolution all multiply complexity without adding to the core experiment | Share the dashboard URL on a local network if needed. Full multi-user is v3+ |
| Options / crypto / forex support | "More instruments = more data" | US stocks via Alpaca is already a full-time scope. Multi-asset adds data normalization, different settlement rules, different risk profiles | Prove the loop works on equities first. Asset expansion is a natural v2 feature |
| Automated ML hyperparameter tuning (AutoML) | "Have the system tune its own neural nets" | AutoML frameworks like Optuna/Ray Tune add significant complexity and training time that breaks the session cadence | Use fixed but mutatable architecture. Genetic mutation of layer sizes and learning rates is sufficient without a full AutoML framework |
| Social / copy trading features | "Let the best model trade for others" | Completely different product. Regulatory, liability, and infrastructure concerns multiply immediately | Out of scope. The arena is a research tool, not a social platform |

---

## Feature Dependencies

```
[Alpaca API Integration]
    └──required by──> [Paper Trade Execution]
    └──required by──> [Real-Time Market Data Streaming]
                          └──required by──> [Per-Model Signal Generation]
                                                └──required by──> [Trade Execution per Model]
                                                                      └──required by──> [Performance Tracking]
                                                                                            └──required by──> [Model Rankings / Leaderboard]
                                                                                                                  └──required by──> [Elimination Decision]
                                                                                                                                        └──required by──> [Merging / Crossover]
                                                                                                                                                              └──required by──> [Generation Spawn]

[Performance Tracking]
    └──required by──> [Equity Curve Chart]
    └──required by──> [Session Summary]
    └──required by──> [Walk-Forward Validation] (v1.x)

[Generational History / Lineage]
    └──enhanced by──> [Mutation Parameter Logging]
    └──enhanced by──> [Fitness Function Logging]

[Basic Risk Limits]
    └──prevents──> [Runaway model poisoning session data]
    └──required before──> [Walk-Forward Validation] (garbage-in without this)

[Strategy Type Diversity Enforcement]
    └──conflicts with──> [Pure global elimination] (need type-aware elimination to preserve diversity)
```

### Dependency Notes

- **Alpaca API Integration** is the foundational dependency for the entire system. Until it works reliably, nothing else can be tested.
- **Performance Tracking requires Trade Execution**: You need real fills to compute real P&L. Simulated orders that never touch Alpaca produce meaningless metrics.
- **Elimination requires Performance Tracking**: The generational loop cannot function without a reliable fitness score per model. This creates a strict build order.
- **Walk-Forward Validation requires multiple sessions of data**: Cannot be implemented meaningfully until the system has run enough generations to have a history. This is a v1.x feature, not v1.
- **Strategy Type Diversity Enforcement conflicts with pure global elimination**: If you eliminate the worst N models globally, you may accidentally eliminate all representatives of one strategy type, collapsing diversity within one generation. Type-aware elimination prevents premature convergence.

---

## MVP Definition

### Launch With (v1)

Minimum to validate the core value proposition: models compete, losers are cut, winners breed, the loop runs.

- [ ] Alpaca API connection (paper trading keys, market data WebSocket) -- the whole system rests on this
- [ ] 5-10 starting models with distinct strategy implementations (at least: moving average crossover, RSI mean reversion, momentum, and one ML-based) -- diversity is required for meaningful selection
- [ ] Per-model order placement and fill tracking -- required to compute performance
- [ ] Per-model performance metrics: P&L, return %, Sharpe ratio, max drawdown, win rate -- minimum viable fitness scoring
- [ ] Model rankings updated in real time during session -- core mechanic visibility
- [ ] Basic risk limits: max single-position size, max daily loss per model -- prevents a broken model from polluting session data
- [ ] Session-end elimination: bottom N models removed, survivors logged -- required for the evolutionary loop
- [ ] Crossover / merge: surviving models exchange parameter subsets to produce offspring -- required for "better than parents"
- [ ] Mutation: offspring parameters perturbed within bounds -- required to escape local optima
- [ ] New generation spawned automatically for next session -- closes the loop
- [ ] Web dashboard: equity curves, trade log, rankings, current generation number -- required to observe competition; CLI alone is insufficient for this use case
- [ ] Generational log: record which models existed, their scores, and their parentage per generation -- required to validate that generations improve over time

### Add After Validation (v1.x)

Add once the core loop is running and producing multiple generations of results.

- [ ] Walk-forward out-of-sample validation layer -- trigger: when you want to consider graduating a model to live trading
- [ ] Configurable fitness function -- trigger: after first few generations, you'll learn which metric best captures "good" for your use case
- [ ] Strategy type diversity enforcement -- trigger: if you observe population collapsing to one strategy family within a few generations
- [ ] Generational lineage visualization (tree/DAG view in dashboard) -- trigger: when you want to debug which ancestors contributed to a winning model
- [ ] Session replay: step through a past session's trades and decisions -- trigger: when debugging unexpected model behavior
- [ ] Mutation / crossover parameter logging detail -- trigger: when post-mortems on failed offspring become needed

### Future Consideration (v2+)

Defer until the evolutionary loop is validated and a model is approaching graduation to live trading.

- [ ] Live trading graduation pipeline with circuit breakers -- defer: meaningful only after paper trading proves a model stable across many generations
- [ ] Multi-asset expansion (ETFs, then crypto on Alpaca) -- defer: US equity proof-of-concept must come first
- [ ] Advanced ML model types (RL agents, transformers) -- defer: start simple; exotic model types add training complexity before the loop is even validated
- [ ] Cloud deployment / always-on execution -- defer: local machine is sufficient for v1; cloud adds cost and ops complexity
- [ ] API / webhook for external strategy injection -- defer: single-operator system has no external consumers yet

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Alpaca API integration | HIGH | MEDIUM | P1 |
| Per-model trade execution | HIGH | MEDIUM | P1 |
| Real-time market data streaming | HIGH | MEDIUM | P1 |
| Performance metrics (Sharpe, drawdown, P&L) | HIGH | MEDIUM | P1 |
| Model rankings / leaderboard | HIGH | LOW | P1 |
| Basic risk limits | HIGH | LOW | P1 |
| Session-end elimination | HIGH | MEDIUM | P1 |
| Crossover / merge | HIGH | HIGH | P1 |
| Mutation | HIGH | MEDIUM | P1 |
| Next-generation spawn | HIGH | MEDIUM | P1 |
| Web dashboard (equity curves, trade log, rankings) | HIGH | HIGH | P1 |
| Generational log / lineage storage | HIGH | LOW | P1 |
| Walk-forward validation | HIGH | HIGH | P2 |
| Configurable fitness function | MEDIUM | LOW | P2 |
| Strategy type diversity enforcement | MEDIUM | MEDIUM | P2 |
| Generational lineage visualization (tree view) | MEDIUM | MEDIUM | P2 |
| Session replay | MEDIUM | HIGH | P2 |
| Mutation parameter logging detail | LOW | LOW | P2 |
| Live trading graduation pipeline | HIGH | HIGH | P3 |
| Multi-asset expansion | MEDIUM | HIGH | P3 |
| Advanced ML model types (RL, transformers) | MEDIUM | HIGH | P3 |
| Cloud deployment | LOW | HIGH | P3 |

**Priority key:**
- P1: Must have for launch (v1)
- P2: Should have, add when core loop validates
- P3: Future consideration (v2+)

---

## Competitor Feature Analysis

| Feature | AI Trading Arena (nof1.ai Alpha Arena) | QuantConnect LEAN | Day Trader (this project) |
|---------|----------------------------------------|-------------------|---------------------------|
| Multi-model competition | Yes, 4 AI models competing in real time | No, single strategy backtesting | Yes, 5-10 models |
| Elimination mechanic | Yes, survival-of-fittest switches live execution | No | Yes, elimination at session end |
| Crossover / merging | No, models are fixed AI instances | No | Yes, core differentiator |
| Mutation / offspring spawn | No | No | Yes, core differentiator |
| Generational loop | No | No | Yes, core differentiator |
| Live equity curves | Yes, per-model | Yes, backtesting only | Yes, during live session |
| Leaderboard | Yes, real-time ranking | No | Yes |
| Walk-forward validation | Implicit (best model takes live capital) | Yes, full backtesting framework | Planned (v1.x) |
| Paper trading | Yes (virtual layer before live) | Yes | Yes (Alpaca) |
| Live trading graduation | Yes, automatic best-model promotion | Yes, via broker integration | Planned (v2+) |
| Generational history / lineage | No | No | Yes, key differentiator |
| Open source | Yes | Yes (LEAN engine) | Yes (personal project) |

---

## Sources

- [Alpaca Paper Trading Docs](https://docs.alpaca.markets/docs/paper-trading) -- HIGH confidence, official documentation
- [AI Trading Arena: Real-time Multi-Model Competition](https://dev.to/quant001/ai-trading-arena-real-time-multi-model-competition-for-optimal-execution-1n6m) -- MEDIUM confidence, implementation reference
- [Advanced Trading Metrics: Sharpe, Sortino, Calmar](https://algostrategyanalyzer.com/en/blog/advanced-trading-metrics/) -- HIGH confidence, standard quant metrics
- [Using Genetic Algorithms to Build Stock Trading Strategies](https://towardsdatascience.com/using-genetic-algorithms-to-build-stock-trading-strategies-d227951d3df0/) -- MEDIUM confidence
- [Optimizing trading strategies using genetic algorithms (2025)](https://learning-gate.com/index.php/2576-8484/article/view/10407) -- MEDIUM confidence, peer-reviewed
- [Backtesting Done Wrong: Common Mistakes](https://www.quantifiedstrategies.com/backtesting-done-wrong-common-mistakes-that-sabotage-trading-performance/) -- MEDIUM confidence, practitioner source
- [Agent-Based Genetic Algorithm for Crypto Trading (2025)](https://arxiv.org/html/2510.07943v1) -- HIGH confidence, peer-reviewed research
- [Performance Metrics to Evaluate Algorithmic Trading Strategies 2025](https://www.utradealgos.com/blog/5-key-metrics-to-evaluate-the-performance-of-your-trading-algorithms) -- MEDIUM confidence
- [Best Paper Trading Platform 2025 (ETNA)](https://www.etnasoft.com/best-paper-trading-platform-for-u-s-broker-dealers-why-advanced-simulation-sets-the-2025-standard/) -- MEDIUM confidence

---

*Feature research for: evolutionary algorithmic trading arena*
*Researched: 2026-03-02*
