# Day Trader

## What This Is

An evolutionary trading arena where multiple AI/algorithmic models compete against each other on US stock markets using paper money. Models that underperform get eliminated, survivors merge their best elements into hybrid offspring, and new generations spawn automatically. The system runs locally during market hours, with a web dashboard to observe the competition in real time. The long-term goal is to identify a consistently profitable model and graduate it to live trading with real money.

## Core Value

The competitive elimination loop must work: models compete, losers die, winners merge, and the next generation is measurably better than the last.

## Requirements

### Validated

(None yet - ship to validate)

### Active

- [ ] Multiple trading models (5-10) competing simultaneously on paper trades via Alpaca
- [ ] Strategy diversity: technical analysis, ML-based, copy trading, and hybrid approaches
- [ ] Real-time and historical market data from Alpaca API
- [ ] Performance tracking and ranking across models
- [ ] Elimination of worst-performing models after each trading session
- [ ] Surviving models merge their winning elements into new hybrid models
- [ ] Models learn and improve: parameter tuning, ML retraining, and spawning mutated variants
- [ ] Automatic generational loop: compete, eliminate, merge, spawn, repeat
- [ ] Web dashboard showing live model performance, trades, rankings, and generational history
- [ ] Paper trading only (no real money in v1)
- [ ] Runs locally on developer machine during market hours

### Out of Scope

- Live/real money trading - not until a model proves consistent profitability on paper
- Mobile app - web dashboard is sufficient
- Multi-asset (crypto, forex, options) - US stocks only for now
- High-frequency trading (sub-second) - not targeting ultra-low latency
- Multi-user support - single operator system

## Context

- Alpaca provides both market data and paper trading through a single API, which simplifies the data and execution layer
- US stock market hours are 9:30 AM - 4:00 PM ET, so the system operates within that window
- The evolutionary/genetic algorithm approach to strategy selection is well-established in quantitative finance research
- Starting with 5-10 models keeps the system debuggable while still providing meaningful competitive pressure
- Python has the strongest ecosystem for trading (pandas, numpy) and ML (scikit-learn, pytorch), but the dashboard will likely benefit from a web framework

## Constraints

- **Data source**: Alpaca API (free tier for paper trading, real-time data)
- **Execution**: Local machine only for v1
- **Market hours**: System operates during US stock market hours (9:30 AM - 4:00 PM ET)
- **Capital**: Paper money only, no real financial risk in v1

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Alpaca for data + paper trading | Single API for both market data and trade execution, free tier available | - Pending |
| Start with 5-10 models | Small enough to debug, large enough for meaningful competition | - Pending |
| Merge-based collaboration | Survivors combine their best elements rather than voting or sharing signals | - Pending |
| Web dashboard for monitoring | Visual feedback on model competition, more engaging than CLI logs | - Pending |
| Local execution | Simpler deployment, no cloud costs while experimenting | - Pending |

---
*Last updated: 2026-03-02 after initialization*
