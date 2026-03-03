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
    model_count: int = 10
    initial_capital: float = 1_000.0
    transaction_cost_pct: float = 0.0005  # 0.05% per side
    max_position_pct: float = 0.20  # max 20% of capital in one position
    max_daily_loss_pct: float = 0.10  # max 10% daily loss per model
    max_open_positions: int = 3
    elimination_rate: float = 0.25  # bottom 25% eliminated
    mutation_range: float = 0.15  # +/- 15% parameter perturbation
    symbols: list[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "META", "TSLA", "JPM", "V", "SPY",
    ])


@dataclass
class Config:
    alpaca: AlpacaConfig = field(default_factory=AlpacaConfig)
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    db_path: str = "day_trader.db"

    @classmethod
    def load(cls) -> "Config":
        return cls(
            alpaca=AlpacaConfig(),
            arena=ArenaConfig(),
            db_path=os.getenv("DB_PATH", "day_trader.db"),
        )
