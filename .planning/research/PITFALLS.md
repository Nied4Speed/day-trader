# Pitfalls Research

**Domain:** Evolutionary algorithmic trading arena (multi-model competition with genetic elimination/merging)
**Researched:** 2026-03-02
**Confidence:** MEDIUM-HIGH — Core algo trading pitfalls are well-documented; evolutionary-specific and Alpaca-specific details partially from community forums

---

## Critical Pitfalls

### Pitfall 1: Fitness Function Evaluates the Wrong Thing

**What goes wrong:**
Using raw return (or even Sharpe ratio alone) as the fitness metric causes the evolutionary loop to converge on a degenerate "buy and hold" strategy. The genetic optimizer finds that the lazy solution — buy at the start of the period, hold until the end, execute zero trades — beats most active strategies on a return basis after transaction costs. This produces "survivors" that have nothing to teach each other, no repeatable edge, and will fail the moment they face a different market regime.

**Why it happens:**
Developers reach for total PnL or Sharpe ratio because they are familiar metrics. They don't account for trade frequency as a required signal of active strategy health. A strategy with 1 trade and +8% beats a strategy with 200 trades and +6% on raw return, so the optimizer kills the active strategy — even though the 200-trade strategy is the one with repeatable signal.

**How to avoid:**
Compose a multi-objective fitness function from the start. Minimum viable fitness function:
- Expected value per trade (not total return)
- Maximum drawdown penalty
- Minimum trade frequency floor (e.g., disqualify models with fewer than N trades per session)
- Sortino ratio preferred over Sharpe (penalizes downside volatility specifically)

Never use a single-objective fitness function. Require at least trade frequency + risk-adjusted return.

**Warning signs:**
- Survivor models converge on very low trade counts after 2-3 generations
- All survivors hold the same position (e.g., all long SPY equivalent)
- Fitness scores are climbing but all models look identical in dashboard
- Hybrid offspring immediately revert to buy-and-hold behavior

**Phase to address:** Foundation phase — fitness function must be defined before the first elimination loop runs. Changing it mid-evolution invalidates comparisons between generations.

---

### Pitfall 2: Look-Ahead Bias Contaminates Strategy Signals

**What goes wrong:**
A model receives market data that was not available at the time the trade decision should have been made. Common examples: using the closing price of bar N to decide whether to buy at bar N (instead of bar N+1), using a technical indicator that repaints (e.g., some implementations of ZigZag), or fetching fundamentals data that was disclosed after market close but applying it to intraday signals. The model shows spectacular backtest performance but produces random or losing results in live paper trading.

**Why it happens:**
When writing strategy logic, the bar indexing is easy to get wrong by one step. `data[-1]` is the last completed bar. `data[0]` is the current (in-progress) bar. Accessing `data[0]` for a signal and then also placing an order on `data[0]` means you're trading on a price you haven't seen yet. This is especially dangerous in vectorized backtests run on pandas DataFrames, where `.shift()` is required but easy to omit.

**How to avoid:**
- Write all strategies in event-driven style (process each incoming bar, not a full price array)
- Never use current in-progress bar price for entry decisions — only completed bars
- Run a "sanity test" against a random-walk dataset: if a strategy shows significant alpha on pure random data, look-ahead bias is present
- Require a separate hold-out paper trading period for every strategy before it can enter the arena

**Warning signs:**
- Backtest Sharpe ratio above 3.0 (extremely rare in legitimate strategies)
- Strategy logic touches `close[0]` or equivalent current-bar price on the same tick as order placement
- Performance collapses immediately when the strategy runs in paper mode

**Phase to address:** Strategy engine phase — build the event loop correctly before writing any strategy logic. A correct event loop prevents this structurally.

---

### Pitfall 3: Alpaca Paper Trading Results Don't Transfer to Live

**What goes wrong:**
Alpaca paper trading executes market orders at the last trade price with no slippage simulation and assumes 100% fill rate. Live trading fills at bid/ask spread, experiences partial fills, and is subject to queue position. Developers build and evolve strategies in paper mode, validate what looks like consistent profitability, graduate to live, and discover they lose money on every trade due to transaction friction they never modeled.

Additionally, Alpaca's paper environment has documented data reliability issues: users have reported streaming data delays of 20 minutes to 2 hours on non-funded accounts. Strategy timing signals built on assumption of real-time data will be corrupted by these delays.

**Why it happens:**
Paper trading simulators are designed to be convenient, not realistic. Alpaca's paper mode assumes perfect fills as a deliberate design choice. Developers trust the paper results without understanding what "paper" actually means for order execution simulation.

**How to avoid:**
- Add a simulated transaction cost layer on top of paper results: assume 0.05% slippage per side plus any commission
- Never use market orders in the evolutionary arena — require all strategies to place limit orders to force realistic fill modeling
- Monitor order fill latency even in paper mode; treat delays above 1 second as a warning
- Validate WebSocket stream timestamps against system clock — flag if delta exceeds 5 seconds
- Before any strategy graduates to live, add a 30-day paper trading period with manually logged slippage costs

**Warning signs:**
- Strategy PnL looks much better on market orders than limit orders in paper
- Order timestamps show consistent gap between signal time and fill time
- Paper results show 100% fill rate on every order
- WebSocket `bar.t` timestamp is more than 10 seconds behind `datetime.utcnow()`

**Phase to address:** Data infrastructure and execution engine phases — must be addressed before the competitive arena launches.

---

### Pitfall 4: Premature Convergence Kills Population Diversity

**What goes wrong:**
Aggressive selection pressure (eliminating too many models per generation, or using a fitness function with too much spread) causes all surviving strategies to converge to the same local optimum within 3-5 generations. The "merging" step then produces offspring that are slight variations of the same strategy. The arena stops exploring the strategy space. You have an evolutionary system that degenerates into a hill-climber with artificial biodiversity theater.

**Why it happens:**
The natural impulse is to kill "bad" strategies quickly and keep "good" ones. Eliminating 70-80% of models per generation feels efficient. But genetic diversity is the engine that drives improvement. If you eliminate too aggressively, you remove the only carriers of novel signal combinations that haven't had time to prove themselves yet.

A related failure: defining "loser" as any model with negative PnL in a single session. A legitimately good strategy can have losing sessions. Eliminating based on short windows rewards strategies that look good quickly (often mean-reversion strategies in low-volatility markets) and punishes strategies that need time to play out (trend-following strategies during range-bound markets).

**How to avoid:**
- Eliminate at most 30-40% of models per generation
- Use rolling multi-session PnL for elimination decisions, not single-session results
- Maintain a "genetic diversity metric" (e.g., parameter overlap between surviving strategies) — if it drops below a threshold, inject new random strategies
- Include mutation-only offspring in every generation (not just merges of survivors), to continuously introduce novel variants
- Tournament selection with small tournament size rather than rank selection

**Warning signs:**
- All survivor strategies share the same primary indicator type within 3 generations
- Strategy parameter variance drops to near-zero in population
- Adding new strategies to the arena has no effect — they get eliminated immediately
- Fitness scores plateau after 5-7 generations

**Phase to address:** Evolutionary loop phase — the elimination rate and selection mechanism must be designed before the first generational loop.

---

### Pitfall 5: Shared State Corruption Across Concurrent Models

**What goes wrong:**
When 5-10 models run simultaneously, they may share global objects: data feeds, position trackers, order managers, or connection pools. Without careful isolation, Model A's state mutation can corrupt Model B's decision context. Common failure modes: Model A marks a position as closed, Model B's portfolio calculation reads a stale position and makes a trade it shouldn't, the system now has phantom positions. Or: two models both trigger a buy signal simultaneously and each submits a full-size order, resulting in double exposure.

**Why it happens:**
Single-strategy systems don't need to think about concurrency. When you scale to 10 concurrent models, all the implicit "there's only one of me" assumptions break simultaneously. Python's GIL prevents true thread parallelism for CPU-bound work, but async operations (API calls, websocket events) can still interleave in unexpected ways.

**How to avoid:**
- Give each model its own position ledger — no shared position tracking state
- Each model maintains its own paper account allocation (divide capital N ways at start)
- Use a message queue (e.g., asyncio.Queue) between the data feed and each strategy, never shared mutable objects
- Implement a central order router that serializes order submission — no model submits orders directly
- All inter-model communication goes through the arena coordinator, not direct references

**Warning signs:**
- Occasional "ghost" positions that appear and can't be explained by any single model's log
- Order quantities don't match any model's intended size
- Concurrency-related errors in logs (race conditions surfacing as KeyErrors, IndexErrors, or None values)
- Position reconciliation between arena ledger and Alpaca account shows discrepancies

**Phase to address:** Arena architecture phase — must be solved before running more than one model simultaneously.

---

### Pitfall 6: Survivorship Bias in Stock Universe Selection

**What goes wrong:**
The system runs strategies on current S&P 500 constituents or another "alive today" stock list. All the stocks in this universe survived to 2026. Strategies optimized on this universe learn to buy things that look like companies that survived and grew. When these strategies run in real time, they will encounter stocks that are declining, being delisted, or going bankrupt. The strategies have no learned behavior for these cases, and they are not included in the evolutionary training signal. Expected result: strategies appear profitable in backtesting / early paper trading but perform poorly in out-of-sample periods when delisted names are present.

**Why it happens:**
Using a current ticker list is the path of least resistance. Alpaca makes it easy to query current tradeable symbols. Historical constituent lists require separate data sources (CRSP, Compustat, or CSIData).

**How to avoid:**
- For the evolutionary arena's universe, fix the stock universe at session start and hold it constant — do not dynamically add newly listed stocks
- Avoid backtesting on a dynamic universe — use a fixed list and acknowledge it will have mild survivorship bias
- Explicitly include at least a few struggling/high-volatility stocks in the universe to stress-test strategies
- Document the limitation: "This system has survivorship bias in universe construction; results are likely to be optimistic"

**Warning signs:**
- Backtested strategies all favor the same large-cap growth names
- Strategy win rate drops significantly in forward paper trading vs. historical simulation
- No strategy ever holds a position in a declining stock for any meaningful duration

**Phase to address:** Data infrastructure and universe selection phase — define the stock universe before any model touches market data.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Use market orders in paper trading | Simple execution, always fills | Paper results don't translate to live; slippage invisible | Never — use limit orders from the start |
| Single session evaluation for elimination | Fast generational loop | Eliminates good strategies on bad luck days; premature convergence | Never — require multi-session rolling window |
| Global shared data feed object for all models | Simple code | State corruption bugs, subtle race conditions | Only in single-model prototype phase |
| Use current S&P 500 ticker list as universe | Easy to fetch | Survivorship bias inflates all results | Acceptable if documented as known limitation |
| Use total PnL as sole fitness metric | Easy to understand | Converges to buy-and-hold degenerate strategies | Never for the elimination decision |
| Skip walk-forward validation | Ships faster | Overfitted strategies look like champions | Never — at minimum run 1-week hold-out window |

---

## Integration Gotchas

Common mistakes when connecting to external services.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Alpaca WebSocket | Using paper credentials for the market data WebSocket | Paper account credentials have separate WebSocket endpoint — verify you're using `wss://stream.data.alpaca.markets` not the broker stream endpoint |
| Alpaca rate limits | Submitting 10+ REST calls/second when models all react to same bar | Implement exponential backoff + jitter; centralize order submission; batch market data fetches for all symbols in one call |
| Alpaca free tier data | Assuming real-time data — free tier delivers IEX only (one exchange), not SIP consolidated | Explicitly accept that IEX prices may differ from NBBO by 0.1-0.5%; factor this into fill assumptions |
| Alpaca paper fills | Trusting paper fill prices as indicative of live fills | Paper uses last trade price; live uses bid/ask spread. Adjust paper PnL by -0.1% per trade minimum for live estimation |
| WebSocket reconnect | Not implementing reconnect logic — treating disconnect as terminal | Network hiccups will drop WebSocket connections during a 6.5-hour trading session; always implement exponential backoff reconnect with state resync |
| Multiple models, one WS connection | Opening one WebSocket per model | Alpaca limits to one WebSocket connection per account; fan out data from a single connection to all model queues internally |

---

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Fetching OHLCV bars for each model separately | Slow session starts, rate limit 429s | Fetch once, distribute to all model queues via fan-out | With 5+ models, 10+ symbols each |
| Computing indicators inside each model independently on each bar | CPU spikes at bar close, lag in order submission | Precompute indicators once per bar per symbol, share read-only | With 10 models and 20+ symbols |
| Storing full trade history in memory as Python dicts | Slows down after a full trading day of bars | Use SQLite or similar for trade persistence; keep only rolling window in memory | After 2-3 hours of market data at 1-min bars |
| Logging every tick/quote to console | I/O blocks event loop, signals arrive late | Use async logging, structured log files, not stdout | With 10+ symbols streaming quotes |
| Running ML model inference synchronously on each bar | Event loop stalls waiting for model output | Offload ML inference to thread pool with asyncio.run_in_executor | With any moderately complex ML model |

---

## Security Mistakes

Domain-specific security issues beyond general web security.

| Mistake | Risk | Prevention |
|---------|------|------------|
| Hardcoding Alpaca API key and secret in source code | Credentials committed to git, accessible to anyone with repo access | Load from environment variables or a secrets file excluded from git |
| No safeguard against runaway model submitting unlimited orders | One buggy model could submit hundreds of orders, consuming paper capital | Implement per-model order rate limiter (max N orders per minute, max open positions) |
| Dashboard exposes model parameters and signals publicly | Competitors (in a future live scenario) could front-run known signals | Dashboard is local-only; if network-exposed, add authentication even for paper |
| No paper account balance floor | A model in a death spiral short-sells until paper account goes to zero or negative | Arena coordinator checks total paper balance before allowing any model to submit orders |

---

## UX Pitfalls

Common user experience mistakes in this domain.

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Dashboard updates only on generation boundary | Can't see if models are trading or frozen during a session | Show real-time trade feed and position updates within the session, not just between generations |
| Only displaying current generation | Impossible to see if system is improving over time | Show generational history chart — fitness score trend across all generations |
| Showing raw PnL without market context | Can't tell if a model's 2% gain was skill or luck on a 3% up day | Display model PnL vs. SPY benchmark on same chart |
| No visual indicator of model "health" | A frozen/crashed model looks like a cautious model | Show last trade timestamp per model; flag models with no activity for >30 min |
| Genealogy hidden | Can't see which survivors merged to produce a given model | Show parent lineage in model detail view — "Model 7 is offspring of Models 2 and 4" |

---

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **Evolutionary loop:** Often missing multi-session rolling evaluation window — verify that elimination uses at least 3 sessions of performance, not 1
- [ ] **Strategy merging:** Often missing diversity check — verify that merged offspring differ meaningfully from both parents (parameter similarity score)
- [ ] **Paper trading execution:** Often missing slippage simulation layer — verify that reported PnL subtracts estimated transaction costs
- [ ] **Data feed:** Often missing heartbeat monitoring — verify system detects and recovers from a dropped WebSocket within 30 seconds
- [ ] **Position tracking:** Often missing cross-model reconciliation — verify that sum of all model positions matches Alpaca account positions on each bar
- [ ] **Fitness function:** Often missing minimum trade frequency threshold — verify that a buy-and-hold strategy would score below active strategies
- [ ] **Market regime context:** Often missing benchmark comparison — verify that model PnL is shown relative to SPY return for the same period
- [ ] **ML-based strategies:** Often missing retraining cadence — verify that ML models have a defined retrain schedule and staleness detection

---

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Fitness function produces degenerate buy-and-hold survivors | HIGH — restart evolution | Redesign fitness function, reset population to generation 0, restart generational loop |
| Look-ahead bias discovered after multiple generations | HIGH — all results invalid | Fix the event loop, re-run backtests from scratch; do not reuse "winner" strategies from corrupted runs |
| WebSocket stream drops mid-session | LOW | Implement reconnect logic; resync positions from REST API on reconnect; mark any trades during gap as "unverified" |
| Population diversity collapses | MEDIUM | Inject fresh random strategies to restore diversity; temporarily reduce elimination rate; add mutation-only generation |
| Shared state corruption causes phantom positions | MEDIUM — session restart | Stop all models, reconcile position state against Alpaca REST account API, restart with clean state |
| Alpaca rate limit 429 during session | LOW | Implement exponential backoff; reduce polling frequency; batch requests |
| Model PnL looks great but collapses when simulating transaction costs | MEDIUM — strategy redesign | Apply 0.1% per-side cost to all historical results; discard strategies that don't survive this filter |

---

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Degenerate fitness function | Data infrastructure + strategy engine (early) | Run buy-and-hold against fitness function — it must score below active strategies |
| Look-ahead bias | Strategy engine phase | Backtest against random-walk data — should show no alpha |
| Paper-to-live performance gap | Execution engine phase | Add slippage simulation layer before first arena session |
| Premature convergence | Evolutionary loop phase | Measure parameter variance across population each generation |
| Shared state corruption | Arena architecture phase | Run 2 models simultaneously and reconcile positions against Alpaca on each bar |
| Survivorship bias | Universe selection (early) | Audit universe list — include at least 5 stocks with negative 12-month performance |
| Alpaca WebSocket unreliability | Data infrastructure phase | Implement reconnect and run a 3-hour stress test with intentional connection drops |
| Fitness evaluated on single session | Evolutionary loop phase | Verify elimination code uses rolling N-session window |

---

## Sources

- [Common Algorithmic Trading Mistakes and Automated Solutions — NURP](https://nurp.com/wisdom/common-algorithmic-trading-errors-and-solutions/)
- [Key Risks in Automated Trading — DarkBot](https://darkbot.io/blog/key-risks-in-automated-trading-what-traders-miss)
- [Top Algo Trading Mistakes — EFX Algo (Dec 2025)](https://efxalgo.com/2025/12/18/top-algo-trading-mistakes-and-how-to-avoid-them/)
- [Backtesting Traps: Common Errors to Avoid — LuxAlgo](https://www.luxalgo.com/blog/backtesting-traps-common-errors-to-avoid/)
- [Common Pitfalls in Backtesting — Medium / Funny AI & Quant](https://medium.com/funny-ai-quant/ai-algorithmic-trading-common-pitfalls-in-backtesting-a-comprehensive-guide-for-algorithmic-ce97e1b1f7f7)
- [Stop Faking Your Results: Backtesting Pitfalls — Medium (Jan 2026)](https://medium.com/algorithmic-and-quantitative-trading/stop-faking-your-results-the-most-common-backtesting-pitfalls-to-avoid-f8dd94d1ca8e)
- [Backtesting Bias — Robot Wealth](https://robotwealth.com/backtesting-bias-feels-good-until-you-blow-up/)
- [Evolving Trading Strategies — Fitness Functions (Fabian Kostadinov)](https://fabian-kostadinov.github.io/2014/12/22/evolving-trading-strategies-with-genetic-programming-fitness-functions/)
- [Failure of Genetic-Programming Induced Trading Strategies — Springer](https://link.springer.com/chapter/10.1007/978-3-540-72821-4_11)
- [Survivorship Bias in Backtesting — QuantRocket](https://www.quantrocket.com/blog/survivorship-bias/)
- [Survivorship Bias in Backtesting (Jan 2026) — adventuresofgreg.blog](http://adventuresofgreg.com/blog/2026/01/14/survivorship-bias-backtesting-avoiding-traps/)
- [Slippage: Paper Trading vs Real Trading — Alpaca Community Forum](https://forum.alpaca.markets/t/slippage-paper-trading-vs-real-trading/2801)
- [Significant Delay in Market Streaming Data — Alpaca Community Forum](https://forum.alpaca.markets/t/significant-delay-in-market-streaming-data-live-paper/2449)
- [429 Rate Limit Exceeded — Alpaca Community Forum](https://forum.alpaca.markets/t/429-rate-limit-exceeded-when-creating-orders/14120)
- [WebSocket Streaming Unserviceable in Paper Trading — Alpaca Forum](https://forum.alpaca.markets/t/websocket-streaming-unserviceable-in-paper-trading/4101)
- [Selection Pressure and Genetic Convergence — Springer](https://link.springer.com/article/10.1007/s40747-019-0102-7)
- [Concurrency, State Management in Trading Bots — Medium (Jan 2026)](https://medium.com/@halljames9963/concurrency-state-management-and-fault-tolerance-in-stock-trading-bots-da774736c58c)
- [Model Drift in FinTech — FinTech Weekly](https://www.fintechweekly.com/magazine/articles/ai-model-drift-management-fintech-applications)
- [Autoregressive Drift Detection in Trading — QuantInsti](https://blog.quantinsti.com/autoregressive-drift-detection-method/)

---
*Pitfalls research for: Day Trader — evolutionary algorithmic trading arena*
*Researched: 2026-03-02*
