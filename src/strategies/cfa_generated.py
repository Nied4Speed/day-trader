from src.core.strategy import Strategy, TradeSignal, BarData
import pandas as pd
import numpy as np
import logging
import math

class CfaGeneratedStrategy(Strategy):
    """Aggressive quick-start strategy designed to trade with minimal data requirements.
    
    Key improvements from previous version:
    1. Reduced lookback requirements (5-10 bars vs 20-50)
    2. More aggressive entry signals to ensure trading activity
    3. Time-of-day awareness to avoid illiquid periods
    4. Simplified logic for faster decision making
    """
    
    strategy_type = "cfa_generated"
    
    def __init__(self, name, params=None):
        # Quick indicators with minimal lookback
        self.fast_ma = 5
        self.slow_ma = 10
        self.rsi_period = 6  # Reduced from 14
        self.rsi_oversold = 35.0  # Raised from 28
        self.rsi_overbought = 65.0  # Lowered from 72
        
        # Volume spike detection
        self.volume_lookback = 5  # Only need 5 bars
        self.volume_spike = 1.5  # 50% above average
        
        # Momentum detection
        self.momentum_bars = 3  # Very short-term momentum
        self.momentum_threshold = 0.002  # 0.2% move
        
        # Risk and position parameters
        self.allocation_pct = 0.25
        self.max_positions = 4  # Increased from 3
        self.min_bars_held = 2  # Reduced from 3
        self.stop_loss_pct = 2.0  # 2% stop loss (percentage points, consistent with base class)
        self.take_profit_pct = 3.0  # 3% take profit (percentage points, consistent with base class)
        
        # Time-based filters
        self.min_minutes_after_open = 5  # Wait 5 minutes after open
        self.min_minutes_before_close = 10  # Stop trading 10 min before close
        
        super().__init__(name, params)
        self.entry_prices = {}  # Track entry prices for profit targets
        self.bars_held = {}  # Track holding period
    
    def _is_valid_trading_time(self, bar):
        """Check if current time is appropriate for trading."""
        # Simple check based on minutes remaining
        # Assuming ~390 minute session, avoid first 5 and last 10 minutes
        if bar.minutes_remaining < self.min_minutes_before_close:
            return False
        if bar.minutes_remaining > 385:  # Within first 5 minutes
            return False
        return True
    
    def _compute_rsi(self, closes):
        """Fast RSI calculation."""
        if len(closes) < self.rsi_period + 1:
            return 50.0
        
        deltas = closes.diff().dropna()
        gains = deltas.where(deltas > 0, 0.0)
        losses = -deltas.where(deltas < 0, 0.0)
        
        avg_gain = gains.rolling(self.rsi_period).mean().iloc[-1]
        avg_loss = losses.rolling(self.rsi_period).mean().iloc[-1]
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))
    
    def _detect_volume_spike(self, volumes):
        """Detect if current volume is elevated."""
        if len(volumes) < self.volume_lookback:
            return False
        
        avg_volume = volumes[-self.volume_lookback:-1].mean()
        current_volume = volumes.iloc[-1]
        
        return current_volume > (avg_volume * self.volume_spike)
    
    def _calculate_momentum(self, closes):
        """Calculate short-term price momentum."""
        if len(closes) < self.momentum_bars:
            return 0.0
        
        old_price = closes.iloc[-self.momentum_bars]
        new_price = closes.iloc[-1]
        
        return (new_price - old_price) / old_price
    
    def on_bar(self, bar: BarData):
        self.record_bar(bar)
        
        # Check for liquidation
        liq = self.check_liquidation(bar)
        if liq:
            # Clean up tracking
            if bar.symbol in self.entry_prices:
                del self.entry_prices[bar.symbol]
            if bar.symbol in self.bars_held:
                del self.bars_held[bar.symbol]
            return liq if liq.quantity > 0 else None
        
        # Skip if not valid trading time
        if not self._is_valid_trading_time(bar):
            return None
        
        # Update holding period
        if bar.symbol in self._positions and self._positions[bar.symbol] > 0:
            self.bars_held[bar.symbol] = self.bars_held.get(bar.symbol, 0) + 1
        
        # Get minimal data needed
        history = self.get_history(bar.symbol, lookback=self.slow_ma)
        if len(history) < self.slow_ma:
            return None
        
        closes = pd.Series([b.close for b in history])
        volumes = pd.Series([b.volume for b in history])
        
        # Calculate indicators
        fast_ma_val = closes[-self.fast_ma:].mean()
        slow_ma_val = closes.mean()
        rsi = self._compute_rsi(closes)
        volume_spike = self._detect_volume_spike(volumes)
        momentum = self._calculate_momentum(closes)
        
        current_position = self._positions.get(bar.symbol, 0)
        current_price = bar.close
        
        # EXIT LOGIC
        if current_position > 0:
            entry_price = self.entry_prices.get(bar.symbol, current_price)
            pnl_pct = (current_price - entry_price) / entry_price
            bars = self.bars_held.get(bar.symbol, 0)
            
            # Exit conditions
            should_exit = False
            
            # 1. Stop loss (pnl_pct is a fraction, stop_loss_pct is percentage points)
            if pnl_pct * 100.0 <= -self.stop_loss_pct:
                should_exit = True
                logging.info(f"Stop loss triggered for {bar.symbol} at {pnl_pct:.2%}")

            # 2. Take profit
            elif pnl_pct * 100.0 >= self.take_profit_pct:
                should_exit = True
                logging.info(f"Take profit triggered for {bar.symbol} at {pnl_pct:.2%}")
            
            # 3. Technical exit signals
            elif bars >= self.min_bars_held:
                # Exit on MA crossover down
                if fast_ma_val < slow_ma_val:
                    should_exit = True
                # Exit on RSI overbought
                elif rsi > self.rsi_overbought:
                    should_exit = True
                # Exit on momentum reversal
                elif momentum < -self.momentum_threshold:
                    should_exit = True
            
            if should_exit:
                # Clean up tracking
                if bar.symbol in self.entry_prices:
                    del self.entry_prices[bar.symbol]
                if bar.symbol in self.bars_held:
                    del self.bars_held[bar.symbol]
                return TradeSignal(symbol=bar.symbol, side="sell", quantity=current_position)
        
        # ENTRY LOGIC
        elif current_position == 0 and len(self._positions) < self.max_positions:
            should_enter = False
            
            # 1. MA crossover with momentum
            if fast_ma_val > slow_ma_val and momentum > self.momentum_threshold:
                should_enter = True
            
            # 2. Oversold bounce with volume
            elif rsi < self.rsi_oversold and volume_spike:
                should_enter = True
            
            # 3. Strong momentum with volume
            elif momentum > self.momentum_threshold * 2 and volume_spike:
                should_enter = True
            
            # 4. Simple RSI + MA alignment
            elif rsi < 40 and fast_ma_val > slow_ma_val:
                should_enter = True
            
            if should_enter:
                qty = self.compute_quantity(current_price, self.allocation_pct)
                if qty > 0:
                    self.entry_prices[bar.symbol] = current_price
                    self.bars_held[bar.symbol] = 0
                    logging.info(f"Entering {bar.symbol} at {current_price:.2f}, RSI={rsi:.1f}, momentum={momentum:.3%}")
                    return TradeSignal(symbol=bar.symbol, side="buy", quantity=qty)
        
        return None
    
    def get_params(self):
        return {
            "fast_ma": self.fast_ma,
            "slow_ma": self.slow_ma,
            "rsi_period": self.rsi_period,
            "rsi_oversold": self.rsi_oversold,
            "rsi_overbought": self.rsi_overbought,
            "volume_lookback": self.volume_lookback,
            "volume_spike": self.volume_spike,
            "momentum_bars": self.momentum_bars,
            "momentum_threshold": self.momentum_threshold,
            "allocation_pct": self.allocation_pct,
            "max_positions": self.max_positions,
            "min_bars_held": self.min_bars_held,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "min_minutes_after_open": self.min_minutes_after_open,
            "min_minutes_before_close": self.min_minutes_before_close
        }
    
    def set_params(self, params):
        # MA parameters
        self.fast_ma = max(2, min(10, int(params.get("fast_ma", self.fast_ma))))
        self.slow_ma = max(5, min(20, int(params.get("slow_ma", self.slow_ma))))
        
        # RSI parameters
        self.rsi_period = max(3, min(14, int(params.get("rsi_period", self.rsi_period))))
        self.rsi_oversold = max(20.0, min(45.0, float(params.get("rsi_oversold", self.rsi_oversold))))
        self.rsi_overbought = max(55.0, min(80.0, float(params.get("rsi_overbought", self.rsi_overbought))))
        
        # Volume parameters
        self.volume_lookback = max(3, min(10, int(params.get("volume_lookback", self.volume_lookback))))
        self.volume_spike = max(1.1, min(3.0, float(params.get("volume_spike", self.volume_spike))))
        
        # Momentum parameters
        self.momentum_bars = max(2, min(5, int(params.get("momentum_bars", self.momentum_bars))))
        self.momentum_threshold = max(0.001, min(0.01, float(params.get("momentum_threshold", self.momentum_threshold))))
        
        # Position parameters
        self.allocation_pct = max(0.1, min(0.5, float(params.get("allocation_pct", self.allocation_pct))))
        self.max_positions = max(1, min(6, int(params.get("max_positions", self.max_positions))))
        self.min_bars_held = max(1, min(5, int(params.get("min_bars_held", self.min_bars_held))))
        
        # Risk parameters (percentage points: 2.0 = 2%)
        self.stop_loss_pct = max(0.5, min(5.0, float(params.get("stop_loss_pct", self.stop_loss_pct))))
        self.take_profit_pct = max(1.0, min(10.0, float(params.get("take_profit_pct", self.take_profit_pct))))
        
        # Time parameters
        self.min_minutes_after_open = max(0, min(30, int(params.get("min_minutes_after_open", self.min_minutes_after_open))))
        self.min_minutes_before_close = max(5, min(30, int(params.get("min_minutes_before_close", self.min_minutes_before_close))))
    
    def adapt(self, recent_signals, recent_fills, realized_pnl):
        """Dynamically adjust parameters based on performance."""
        if not recent_fills:
            # If no recent trades, make parameters more aggressive
            self.rsi_oversold = min(45.0, self.rsi_oversold + 2.0)
            self.rsi_overbought = max(55.0, self.rsi_overbought - 2.0)
            self.momentum_threshold = max(0.001, self.momentum_threshold - 0.0005)
            logging.info("No recent trades - making parameters more aggressive")
            return
        
        # Calculate performance metrics
        wins = [f for f in recent_fills if f.get('realized_pnl', 0) > 0]
        losses = [f for f in recent_fills if f.get('realized_pnl', 0) < 0]
        win_rate = len(wins) / len(recent_fills) if recent_fills else 0
        
        # Adjust based on win rate
        if win_rate < 0.3:
            # Poor performance - widen stops, be more selective
            self.stop_loss_pct = min(5.0, self.stop_loss_pct + 0.5)
            self.momentum_threshold = min(0.01, self.momentum_threshold + 0.0005)
            self.rsi_oversold = max(20.0, self.rsi_oversold - 2.0)
        elif win_rate > 0.6:
            # Good performance - can be slightly more aggressive
            self.stop_loss_pct = max(1.0, self.stop_loss_pct - 0.2)
            self.take_profit_pct = min(5.0, self.take_profit_pct + 0.2)
            self.max_positions = min(6, self.max_positions + 1)
        
        # Adjust based on realized PnL
        if realized_pnl < -50:  # Lost more than $50
            self.allocation_pct = max(0.1, self.allocation_pct - 0.05)
            logging.info(f"Reduced allocation to {self.allocation_pct:.1%} after losses")
        elif realized_pnl > 100:  # Made more than $100
            self.allocation_pct = min(0.4, self.allocation_pct + 0.05)
            logging.info(f"Increased allocation to {self.allocation_pct:.1%} after gains")