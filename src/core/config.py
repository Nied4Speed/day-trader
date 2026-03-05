"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class AlpacaConfig:
    api_key: str = ""
    secret_key: str = ""
    base_url: str = "https://paper-api.alpaca.markets"
    data_url: str = "https://data.alpaca.markets"

    def __post_init__(self):
        self.api_key = self.api_key or os.getenv("ALPACA_API_KEY", "")
        self.secret_key = self.secret_key or os.getenv("ALPACA_SECRET_KEY", "")
        self.base_url = os.getenv("ALPACA_BASE_URL", self.base_url)


@dataclass
class ArenaConfig:
    model_count: int = 19  # 19 competitive + 1 COLLAB = 20 total
    initial_capital: float = 1_000.0
    transaction_cost_pct: float = 0.0  # Alpaca is commission-free
    slippage_bps: float = 5.0  # simulated slippage in basis points (5 bps = 0.05%)
    max_daily_loss_pct: float = 0.50  # 50% daily loss before cutoff (generous)
    elimination_rate: float = 0.25  # bottom 25% eliminated
    mutation_range: float = 0.15  # +/- 15% parameter perturbation
    session_1_minutes: int = 180  # 3 hours
    session_2_minutes: int = 180  # 3 hours
    break_minutes: int = 15  # break between sessions for self-improvement
    market_close_buffer_minutes: int = 2  # session timer ends this many min before close
    warmup_bars: int = 50  # historical bars to prime indicators
    premarket_warmup_enabled: bool = True  # poll for pre-market bars before Session 1
    premarket_start_hour_et: int = 9  # begin pre-market warmup at this hour (ET)
    premarket_start_minute_et: int = 0  # begin pre-market warmup at this minute (ET)
    premarket_poll_interval_sec: int = 30  # seconds between pre-market bar polls
    snapshot_interval: int = 5  # save snapshots every N bars during live
    wash_trade_cooldown_sec: float = 60.0  # min seconds between same-symbol trades per model
    news_fetch_interval: int = 5  # fetch news every N bars
    news_lookback_minutes: int = 30  # how far back to look for news
    mutation_memory_enabled: bool = True  # track mutation history for smart self-improvement
    mutation_bias_dampening: float = 0.6  # how much to trust historical bias (0=ignore, 1=full)
    mutation_memory_decay: float = 0.95  # exponential decay per evaluation cycle
    max_watches_per_model: int = 3  # max symbols a single model can watch simultaneously
    watch_quote_throttle_ms: int = 100  # min ms between watch quote dispatches per model+symbol
    llm_watch_rules_enabled: bool = True  # generate watch rules via LLM during self-improvement
    llm_watch_rules_model: str = "claude-haiku-4-5-20251001"  # Anthropic model for rule generation
    llm_watch_rules_max: int = 3  # max rules per model
    llm_watch_rules_timeout_sec: int = 10  # API timeout per call
    self_improve_enabled: bool = True  # auto-mutate params between sessions
    weekly_elimination_count: int = 2  # fixed number of models culled at weekly evolution
    symbols: list[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "META", "TSLA", "JPM", "V", "SPY",
    ])


@dataclass
class Config:
    alpaca: AlpacaConfig = field(default_factory=AlpacaConfig)
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    db_path: str = ""

    @classmethod
    def load(cls) -> "Config":
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        default_db = os.path.join(project_root, "day_trader.db")
        return cls(
            alpaca=AlpacaConfig(),
            arena=ArenaConfig(),
            db_path=os.getenv("DB_PATH", default_db),
        )
