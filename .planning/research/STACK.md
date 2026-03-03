# Stack Research

**Domain:** Evolutionary algorithmic trading arena (local, Python backend, React dashboard)
**Researched:** 2026-03-02
**Confidence:** MEDIUM-HIGH (core Python trading stack HIGH; evolutionary algo layer MEDIUM; dashboard MEDIUM)

---

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | 3.12 | Runtime for all trading, data, and evolutionary logic | Best ecosystem for trading + ML; 3.12 is stable with full scikit-learn/numpy/pandas support through 2026; avoid 3.13+ until library ecosystem catches up |
| alpaca-py | 0.43.2 | Alpaca paper trading execution + market data streaming | Official Alpaca SDK; object-oriented, pydantic-validated, supports paper endpoint, WebSocket streaming, REST historical data; 0.43.2 released Nov 2025 |
| FastAPI | >=0.115 | Backend API + WebSocket server for dashboard | Async-native, ideal for pushing live model state to browser via WebSocket; acts as the bridge between Python trading engine and React UI |
| React + Vite | React 18, Vite 6 | Web dashboard UI | Fastest dev loop for a single-operator local dashboard; React's component model handles live-updating tables and charts cleanly |
| SQLite (via SQLAlchemy) | SQLite 3 / SQLAlchemy 2.x | Local persistence for trades, model state, generational history | Zero-ops for a local machine; SQLAlchemy 2.x provides async support; sufficient for 5-10 models at non-HFT speeds; no server to manage |

### Data and Signal Layer

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pandas | 2.2.x | Time-series manipulation, OHLCV data frames | Core data structure for all market data; every model ingests and outputs pandas DataFrames |
| numpy | 2.x | Numerical computation, vector math | Underlies pandas and scikit-learn; used directly in genetic operator math (crossover, mutation on parameter arrays) |
| pandas-ta | 0.3.14b | Technical indicators (RSI, MACD, Bollinger Bands, 150+) | Pure Python, integrates directly as a pandas extension (.ta accessor); covers all standard TA signals without a C compiler; use over TA-Lib to avoid native build pain on macOS |
| scikit-learn | 1.6.x | ML-based model strategies (random forest, regression, classification) | Standard for tabular financial ML; works with pandas DataFrames; use for ML-strategy models in the arena |

### Evolutionary / Genetic Algorithm Layer

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| DEAP | 1.4.x | Genetic algorithms: selection, crossover, mutation, population management | Most mature and flexible GA library in Python; supports multi-objective optimization (NSGA-II) which matters when balancing Sharpe ratio vs. drawdown vs. win rate; use over PyGAD because the arena's merge/crossover operations need custom representation |
| multiprocessing (stdlib) | built-in | Run 5-10 trading models as isolated processes | Each model is CPU-bound + I/O-bound; process isolation prevents one model's crash from killing others; Python's GIL makes threading insufficient for true parallelism here |

### Scheduling and Orchestration

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| APScheduler | 3.x | Market hours scheduling (9:30-4:00 ET), generational loop timing | Cron triggers for market open/close; interval triggers for intra-session model evaluation; asyncio-compatible for FastAPI integration |

### Dashboard Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| lightweight-charts | 4.x | Candlestick charts for individual model trade visualization | TradingView's open-source charting library; 45kb, canvas-based, handles real-time data efficiently; best-in-class for financial charts in React |
| Recharts | 2.x | Rankings, equity curves, P&L over time | SVG-based, JSX-native React integration; good for line/bar/area charts that update via WebSocket; simpler to customize than lightweight-charts for non-OHLCV data |
| react-use-websocket | 4.x | WebSocket client management in React | Handles reconnection, message batching, and lifecycle cleanup automatically; removes WebSocket boilerplate |
| Tailwind CSS | 3.x | Dashboard styling | Zero-config styling for a single-operator tool; fast iteration without writing CSS |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| uv | Python package and virtualenv manager | Faster than pip/poetry; single tool for project deps; install with `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| pytest + pytest-asyncio | Backend testing | Required for async FastAPI route and WebSocket handler tests |
| ruff | Python linting and formatting | Replaces flake8 + black + isort; single tool, extremely fast |
| Vitest | Frontend unit testing | Vite-native test runner; works without config for React components |

---

## Installation

```bash
# Python environment (using uv)
uv venv --python 3.12
source .venv/bin/activate

# Core trading engine
uv pip install alpaca-py==0.43.2 fastapi uvicorn[standard]

# Data and signal layer
uv pip install pandas numpy pandas-ta scikit-learn

# Evolutionary layer
uv pip install deap

# Scheduling
uv pip install apscheduler

# Persistence
uv pip install sqlalchemy aiosqlite

# Dev tools
uv pip install --dev pytest pytest-asyncio ruff

# Frontend (in /dashboard)
npm create vite@latest dashboard -- --template react-ts
cd dashboard
npm install lightweight-charts recharts react-use-websocket tailwindcss
npm install -D vitest @testing-library/react
```

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| GA library | DEAP | PyGAD | PyGAD is simpler but less flexible; the arena needs custom individual representations (model param dicts) and multi-objective fitness functions that DEAP handles natively via its toolbox system |
| Technical indicators | pandas-ta | TA-Lib | TA-Lib requires compiling C code on macOS (frequent pain point); pandas-ta is pure Python with equivalent indicator coverage; performance difference is irrelevant at 1-minute bar frequencies |
| Database | SQLite | TimescaleDB | TimescaleDB requires running a PostgreSQL server; SQLite is zero-ops and sufficient for 5-10 models at non-HFT frequencies; switch to TimescaleDB if data volume exceeds ~10GB or you need sub-second queries |
| Dashboard backend | FastAPI | Flask | FastAPI has native async and WebSocket support; Flask requires extensions and thread workarounds for WebSocket; FastAPI is the correct choice for real-time push |
| Concurrency | multiprocessing | asyncio-only | Trading models are CPU-bound (signal computation) + I/O-bound (Alpaca API); asyncio alone can't parallelize CPU work due to GIL; multiprocessing gives true parallelism with process isolation |
| Frontend charts | lightweight-charts | Plotly.js | Plotly is heavy and designed for data science notebooks, not real-time financial dashboards; lightweight-charts renders 10x faster at high update rates |
| Frontend charts (non-OHLCV) | Recharts | Chart.js | Recharts is React-idiomatic (JSX, declarative); Chart.js requires imperative canvas manipulation; Recharts fits the React component model better |
| Scheduling | APScheduler | Celery | Celery requires a Redis/RabbitMQ broker; overkill for a single-machine local system; APScheduler runs in-process with zero infrastructure |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Zipline / Zipline-reloaded | Deprecated or poorly maintained; built for Quantopian which shut down; complex installation with conda dependencies | backtesting.py for any historical validation; the arena's forward-testing loop doesn't need Zipline |
| Backtrader | Last meaningful commit was 2021; not maintained; Python 3.12 compatibility is uncertain | pandas-ta + custom signal functions for the arena's real-time signal needs |
| alpaca-trade-api (the old SDK) | Officially deprecated by Alpaca in favor of alpaca-py; no new features, bugs not fixed | alpaca-py 0.43.2 |
| threading for model parallelism | Python GIL prevents true CPU parallelism; models computing signals will contend; one slow model blocks others | multiprocessing with a Queue or Pipe for inter-model communication |
| Streamlit for dashboard | Streamlit rerenders the full page on state change; not suitable for live real-time updates at 1-second intervals; poor WebSocket semantics | FastAPI + React with WebSocket push |
| InfluxDB / ClickHouse | Cloud or server-based time-series DBs; massive overkill for a local machine with 10 models; setup complexity destroys iteration speed | SQLite with an indexed timestamp column handles thousands of rows/second locally |
| PyGAD for evolutionary logic | Easier to start but has limited crossover/mutation customization; the merge operation (combining two model strategies) requires custom operators that PyGAD makes awkward | DEAP with a custom toolbox |
| Node.js backend | Python owns the trading and ML ecosystem; a Node backend would require serializing everything and lose access to pandas/numpy/scikit-learn natively | FastAPI serves both the REST API and the WebSocket in Python |

---

## Stack Patterns by Variant

**If models need sub-second signal computation:**
- Replace pandas-ta with TA-Lib (accept the C build dependency for the performance)
- Move from multiprocessing to asyncio + C-extension for signal math
- This project explicitly excludes HFT, so this variant is not needed

**If model count grows beyond 20:**
- Add Redis as an inter-process message bus instead of multiprocessing Queues
- Migrate SQLite to PostgreSQL + TimescaleDB for concurrent write performance
- This is a v2+ concern

**If you want to add ML model retraining during generational cycles:**
- Add PyTorch (lightweight, modern) for deep learning strategies
- scikit-learn handles tree-based and regression models; add PyTorch only when neural net strategies are needed
- Keep scikit-learn as the default; add PyTorch only on explicit need

---

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| alpaca-py 0.43.2 | Python 3.8-3.14 | Confirmed on PyPI; use 3.12 for best ecosystem compatibility |
| scikit-learn 1.6.x | Python 3.9-3.13, NumPy 2.x | Full Python 3.12 support confirmed; released Dec 2025 |
| pandas 2.2.x | NumPy 2.x, Python 3.9+ | Supported through 2027 per Scientific Python SPEC 0 |
| pandas-ta 0.3.14b | pandas 1.3+, Python 3.6+ | Works with pandas 2.x; verify on install; pure Python so no C conflicts |
| DEAP 1.4.x | Python 3.x, NumPy | Actively maintained; check GitHub for latest release before pinning |
| FastAPI >=0.115 | Python 3.8+, Pydantic v2 | Pydantic v2 is a breaking change from v1; alpaca-py also uses pydantic; confirm shared pydantic version |
| APScheduler 3.x | Python 3.x, asyncio | v4.x is in alpha as of 2025; stay on 3.x for stability |

---

## Architecture Implication

The stack splits cleanly into two runtimes:

1. **Python trading engine** (one FastAPI process + N model subprocesses): alpaca-py, pandas, pandas-ta, DEAP, scikit-learn, SQLite, APScheduler
2. **React dashboard** (browser): lightweight-charts, Recharts, react-use-websocket, Tailwind

Communication is WebSocket from FastAPI to browser only. The trading engine is the single source of truth. The dashboard is read-only (no user control over trades in v1).

---

## Sources

- [alpaca-py on PyPI](https://pypi.org/project/alpaca-py/) — confirmed version 0.43.2, Python 3.8-3.14 support (HIGH confidence)
- [alpaca-py GitHub](https://github.com/alpacahq/alpaca-py) — official SDK, paper trading endpoint, WebSocket streaming (HIGH confidence)
- [Alpaca WebSocket Streaming Docs](https://docs.alpaca.markets/docs/websocket-streaming) — streaming market data protocol (HIGH confidence)
- [scikit-learn on PyPI](https://pypi.org/project/scikit-learn/) — version 1.8.0 released Dec 2025, Python 3.12 support confirmed (HIGH confidence)
- [DEAP GitHub](https://github.com/DEAP/deap) — multi-objective GA, custom operators, actively maintained (MEDIUM confidence, last verified via WebSearch)
- [pandas-ta on PyPI](https://pypi.org/project/pandas-ta/) — 130+ indicators, pure Python pandas extension (MEDIUM confidence)
- [lightweight-charts GitHub](https://github.com/tradingview/lightweight-charts) — 45kb canvas-based financial charts, React integration (HIGH confidence)
- [FastAPI WebSocket guide 2025](https://dev-faizan.medium.com/building-real-time-applications-with-fastapi-websockets-a-complete-guide-2025-40f29d327733) — real-time dashboard patterns (MEDIUM confidence)
- [The Ultimate Python Quantitative Trading Ecosystem 2025](https://medium.com/@mahmoud.abdou2002/the-ultimate-python-quantitative-trading-ecosystem-2025-guide-074c480bce2e) — ecosystem overview (MEDIUM confidence, WebSearch-sourced)
- [Comparing TA-Lib to pandas-ta](https://www.slingacademy.com/article/comparing-ta-lib-to-pandas-ta-which-one-to-choose/) — indicators comparison (MEDIUM confidence)
- [Scientific Python SPEC 0](https://scientific-python.org/specs/spec-0000/) — version support windows for numpy/pandas/scikit-learn (HIGH confidence)

---

*Stack research for: Evolutionary algorithmic trading arena*
*Researched: 2026-03-02*
