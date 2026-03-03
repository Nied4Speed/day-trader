"""ML-based Price Predictor Strategy.

Uses a simple linear regression on recent price features to predict
short-term direction. Buys when predicted return exceeds threshold,
sells when it falls below. The model retrains periodically on
accumulated bar data.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor

from src.core.strategy import BarData, Strategy, TradeSignal


class MLPredictorStrategy(Strategy):
    strategy_type = "ml_predictor"

    def __init__(self, name: str, params: Optional[dict] = None):
        self.feature_lookback: int = 10
        self.train_window: int = 100
        self.retrain_interval: int = 50
        self.buy_threshold: float = 0.001  # 0.1% predicted return
        self.sell_threshold: float = -0.0005
        self.position_size: int = 10
        self.learning_rate: float = 0.001
        self._model: Optional[SGDRegressor] = None
        self._bars_since_train: int = 0
        super().__init__(name, params)

    def _build_features(self, closes: list[float]) -> np.ndarray:
        """Build feature vector from recent closes.

        Features: returns at multiple horizons, volatility, trend strength.
        """
        arr = np.array(closes)
        features = []

        # Returns at different horizons
        for horizon in [1, 3, 5, self.feature_lookback]:
            if len(arr) > horizon:
                features.append((arr[-1] - arr[-1 - horizon]) / arr[-1 - horizon])
            else:
                features.append(0.0)

        # Rolling volatility
        if len(arr) >= self.feature_lookback:
            returns = np.diff(arr[-self.feature_lookback:]) / arr[-self.feature_lookback:-1]
            features.append(np.std(returns) if len(returns) > 0 else 0.0)
        else:
            features.append(0.0)

        # Trend strength (linear regression slope normalized by price)
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

        # Build training data: predict next-bar return from features
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
        history = self.get_history(bar.symbol)

        if len(history) < self.train_window:
            return None

        self._bars_since_train += 1

        # Retrain periodically
        if self._model is None or self._bars_since_train >= self.retrain_interval:
            self._train(bar.symbol)

        if self._model is None:
            return None

        # Predict next-bar return
        closes = [b.close for b in history]
        features = self._build_features(closes)

        try:
            predicted_return = self._model.predict(features)[0]
        except Exception:
            return None

        if predicted_return > self.buy_threshold:
            return TradeSignal(
                symbol=bar.symbol,
                side="buy",
                quantity=self.position_size,
            )

        if predicted_return < self.sell_threshold:
            return TradeSignal(
                symbol=bar.symbol,
                side="sell",
                quantity=self.position_size,
            )

        return None

    def get_params(self) -> dict:
        return {
            "feature_lookback": self.feature_lookback,
            "train_window": self.train_window,
            "retrain_interval": self.retrain_interval,
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
            "position_size": self.position_size,
            "learning_rate": self.learning_rate,
        }

    def set_params(self, params: dict) -> None:
        self.feature_lookback = max(3, int(params.get("feature_lookback", self.feature_lookback)))
        self.train_window = max(50, int(params.get("train_window", self.train_window)))
        self.retrain_interval = max(10, int(params.get("retrain_interval", self.retrain_interval)))
        self.buy_threshold = float(params.get("buy_threshold", self.buy_threshold))
        self.sell_threshold = float(params.get("sell_threshold", self.sell_threshold))
        self.position_size = max(1, int(params.get("position_size", self.position_size)))
        self.learning_rate = max(0.0001, float(params.get("learning_rate", self.learning_rate)))

    def reset(self) -> None:
        super().reset()
        self._model = None
        self._bars_since_train = 0
