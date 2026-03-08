"""ML-based Price Predictor Strategy.

Uses a simple linear regression on recent price features to predict
short-term direction. Buys when predicted return exceeds threshold,
sells when it falls below. The model retrains periodically on
accumulated bar data.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor

from src.core.strategy import BarData, Strategy, TradeSignal

logger = logging.getLogger(__name__)


class MLPredictorStrategy(Strategy):
    strategy_type = "ml_predictor"

    def __init__(self, name: str, params: Optional[dict] = None):
        self.feature_lookback: int = 10
        self.train_window: int = 100
        self.retrain_interval: int = 15
        self.buy_threshold: float = 0.001  # 0.1% predicted return
        self.sell_threshold: float = -0.0005
        self.allocation_pct: float = 0.30
        self.learning_rate: float = 0.001
        self._model: Optional[SGDRegressor] = None
        self._bars_since_train: int = 0
        super().__init__(name, params)

    def _build_features(self, closes: list[float]) -> np.ndarray:
        """Build feature vector from recent closes."""
        arr = np.array(closes)
        features = []

        for horizon in [1, 3, 5, self.feature_lookback]:
            if len(arr) > horizon:
                features.append((arr[-1] - arr[-1 - horizon]) / arr[-1 - horizon])
            else:
                features.append(0.0)

        if len(arr) >= self.feature_lookback:
            returns = np.diff(arr[-self.feature_lookback:]) / arr[-self.feature_lookback:-1]
            features.append(np.std(returns) if len(returns) > 0 else 0.0)
        else:
            features.append(0.0)

        if len(arr) >= self.feature_lookback:
            x = np.arange(self.feature_lookback)
            y = arr[-self.feature_lookback:]
            slope = np.polyfit(x, y, 1)[0]
            features.append(slope / arr[-1])
        else:
            features.append(0.0)

        return np.array(features).reshape(1, -1)

    def _train(self, symbol: str) -> None:
        """Train the model on accumulated bar data."""
        history = self.get_history(symbol)
        if len(history) < self.train_window:
            return

        closes = [b.close for b in history]
        X_list = []
        y_list = []

        for i in range(self.feature_lookback, len(closes) - 1):
            window = closes[max(0, i - self.train_window):i + 1]
            features = self._build_features(window)
            next_return = (closes[i + 1] - closes[i]) / closes[i]
            X_list.append(features[0])
            y_list.append(next_return)

        if len(X_list) < 20:
            return

        X = np.array(X_list)
        y = np.array(y_list)

        self._model = SGDRegressor(
            loss="squared_error",
            learning_rate="adaptive",
            eta0=self.learning_rate,
            max_iter=1000,
            tol=1e-4,
        )
        self._model.fit(X, y)
        self._bars_since_train = 0

    def on_bar(self, bar: BarData) -> Optional[TradeSignal]:
        self.record_bar(bar)
        liq = self.check_liquidation(bar)
        if liq:
            return liq if liq.quantity > 0 else None
        history = self.get_history(bar.symbol)

        if len(history) < self.train_window:
            return None

        self._bars_since_train += 1

        if self._model is None or self._bars_since_train >= self.retrain_interval:
            self._train(bar.symbol)

        if self._model is None:
            return None

        closes = [b.close for b in history]
        features = self._build_features(closes)

        try:
            predicted_return = self._model.predict(features)[0]
        except Exception:
            return None

        # Cache indicators for watch rule evaluation
        self._indicators[bar.symbol] = {
            "close": bar.close,
            "predicted_return": float(predicted_return),
        }

        if predicted_return > self.buy_threshold:
            qty = self.compute_quantity(bar.close, self.allocation_pct)
            if qty > 0:
                return TradeSignal(symbol=bar.symbol, side="buy", quantity=qty)

        if predicted_return < self.sell_threshold:
            qty = self.compute_quantity(bar.close, self.allocation_pct)
            if qty > 0:
                return TradeSignal(symbol=bar.symbol, side="sell", quantity=qty)

        return None

    def get_params(self) -> dict:
        return {
            "feature_lookback": self.feature_lookback,
            "train_window": self.train_window,
            "retrain_interval": self.retrain_interval,
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
            "allocation_pct": self.allocation_pct,
            "learning_rate": self.learning_rate,
        }

    def set_params(self, params: dict) -> None:
        self.feature_lookback = max(3, int(params.get("feature_lookback", self.feature_lookback)))
        self.train_window = max(50, int(params.get("train_window", self.train_window)))
        self.retrain_interval = max(10, int(params.get("retrain_interval", self.retrain_interval)))
        self.buy_threshold = float(params.get("buy_threshold", self.buy_threshold))
        self.sell_threshold = float(params.get("sell_threshold", self.sell_threshold))
        self.allocation_pct = max(0.05, min(1.0, float(params.get("allocation_pct", self.allocation_pct))))
        self.learning_rate = max(0.0001, float(params.get("learning_rate", self.learning_rate)))

    def adapt(self, recent_signals: list, recent_fills: list, realized_pnl: float) -> None:
        old_buy, old_sell = self.buy_threshold, self.sell_threshold
        if realized_pnl < 0:
            self.buy_threshold += 0.0005
            self.sell_threshold -= 0.0002
        else:
            self.buy_threshold -= 0.0002
        self.buy_threshold = max(0.0001, min(0.01, self.buy_threshold))
        self.sell_threshold = max(-0.005, min(-0.0001, self.sell_threshold))
        logger.info(
            f"{self.name} adapt: pnl={realized_pnl:.2f} "
            f"buy_thresh {old_buy:.4f}->{self.buy_threshold:.4f} "
            f"sell_thresh {old_sell:.4f}->{self.sell_threshold:.4f}"
        )

    def reset(self) -> None:
        super().reset()
        self._model = None
        self._bars_since_train = 0
