from src.core.strategy import Strategy, TradeSignal, BarData
import pandas as pd
import numpy as np
import logging
import math

class CfaGeneratedStrategy(Strategy):
    """Multi-regime adaptive strategy using volume profile and momentum divergence.
    
    Key improvements from previous version:
    1. Volume profile analysis to identify institutional activity
    2. Regime detection to adapt to trending vs ranging markets  
    3. Momentum divergence detection for high-probability reversals
    4. Strict risk management with dynamic position sizing
    5. Post-stop-loss cooldown to prevent re-entry bleeding
    """
    
    strategy_type = "cfa_generated"
    
    def __init__(self, name, params=None):
        # Volume profile parameters
        self.volume_lookback = 20  # Bars for volume analysis
        self.volume_threshold = 1.5  # Institutional activity threshold
        self.vwap_deviation = 0.002  # Distance from VWAP for mean reversion
        
        # Momentum parameters
        self.fast_momentum = 5  # Fast momentum period
        self.slow_momentum = 15  # Slow momentum period  
        self.divergence_threshold = 0.003  # Momentum divergence threshold
        
        # Regime detection
        self.atr_period = 10  # ATR for volatility regime
        self.trend_period = 20  # Period for trend strength
        self.regime_threshold = 0.6  # Trend strength threshold
        
        # Risk management
        self.base_allocation = 0.20  # Base position size
        self.max_positions = 3  # Maximum concurrent positions
        self.stop_loss_pct = 1.5  # Tighter stop loss
        self.take_profit_pct = 2.5  # Reasonable profit target
        self.stop_cooldown_bars = 10  # Bars to wait after stop loss
        
        # Entry filters
        self.min_price = 5.0  # Avoid penny stocks
        self.min_volume = 1000000  # Minimum daily volume
        self.min_bars = 30  # Minimum history required
        
        super().__init__(name, params)
        self.entry_prices = {}  # Track entry prices
        self.stop_loss_triggered = {}  # Track stop losses by symbol
        self.regime_state = {}  # Track regime by symbol
    
    def _calculate_vwap(self, bars):
        """Calculate volume-weighted average price."""
        if not bars:
            return None
        
        total_volume = sum(b.volume for b in bars)
        if total_volume == 0:
            return bars[-1].close
        
        vwap = sum(b.close * b.volume for b in bars) / total_volume
        return vwap
    
    def _calculate_atr(self, bars):
        """Calculate Average True Range for volatility."""
        if len(bars) < 2:
            return 0.0
        
        trs = []
        for i in range(1, len(bars)):
            high_low = bars[i].high - bars[i].low
            high_close = abs(bars[i].high - bars[i-1].close)
            low_close = abs(bars[i].low - bars[i-1].close)
            trs.append(max(high_low, high_close, low_close))
        
        if len(trs) >= self.atr_period:
            return np.mean(trs[-self.atr_period:])
        elif trs:
            return np.mean(trs)
        return 0.0
    
    def _detect_regime(self, bars):
        """Detect market regime: trending or ranging."""
        if len(bars) < self.trend_period:
            return 'unknown'
        
        closes = pd.Series([b.close for b in bars[-self.trend_period:]])
        
        # Linear regression slope
        x = np.arange(len(closes))
        slope = np.polyfit(x, closes.values, 1)[0]
        
        # Normalize by price level
        avg_price = closes.mean()
        normalized_slope = slope / avg_price
        
        # R-squared for trend strength
        y_pred = np.polyval(np.polyfit(x, closes.values, 1), x)
        ss_res = np.sum((closes.values - y_pred) ** 2)
        ss_tot = np.sum((closes.values - closes.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Strong trend if high R-squared and significant slope
        if r_squared > self.regime_threshold:
            if normalized_slope > 0.001:
                return 'uptrend'
            elif normalized_slope < -0.001:
                return 'downtrend'
        
        return 'ranging'
    
    def _calculate_momentum_divergence(self, bars):
        """Detect momentum divergence for reversal signals."""
        if len(bars) < self.slow_momentum:
            return 0.0
        
        closes = pd.Series([b.close for b in bars])
        
        # Calculate momentum
        fast_mom = (closes.iloc[-1] - closes.iloc[-self.fast_momentum]) / closes.iloc[-self.fast_momentum]
        slow_mom = (closes.iloc[-1] - closes.iloc[-self.slow_momentum]) / closes.iloc[-self.slow_momentum]
        
        # Divergence: fast momentum weakening relative to slow
        divergence = fast_mom - slow_mom
        return divergence
    
    def _is_institutional_volume(self, current_volume, avg_volume):
        """Check if current volume indicates institutional activity."""
        if avg_volume == 0:
            return False
        return current_volume > (avg_volume * self.volume_threshold)
    
    def _check_cooldown(self, symbol, bar):
        """Check if symbol is in post-stop-loss cooldown."""
        if symbol not in self.stop_loss_triggered:
            return False
        
        bars_since_stop = self.stop_loss_triggered[symbol]
        if bars_since_stop < self.stop_cooldown_bars:
            self.stop_loss_triggered[symbol] += 1
            return True
        else:
            # Cooldown expired
            del self.stop_loss_triggered[symbol]
            return False
    
    def on_bar(self, bar: BarData):
        self.record_bar(bar)
        
        # Check for liquidation
        liq = self.check_liquidation(bar)
        if liq:
            if bar.symbol in self.entry_prices:
                del self.entry_prices[bar.symbol]
            return liq if liq.quantity > 0 else None
        
        # Skip if in cooldown
        if self._check_cooldown(bar.symbol, bar):
            return None
        
        # Skip if price/volume filters not met
        if bar.close < self.min_price or bar.volume < self.min_volume:
            return None
        
        # Get history
        history = self.get_history(bar.symbol, lookback=max(self.volume_lookback, self.trend_period))
        if len(history) < self.min_bars:
            return None
        
        # Calculate indicators
        vwap = self._calculate_vwap(history[-self.volume_lookback:])
        atr = self._calculate_atr(history)
        regime = self._detect_regime(history)
        divergence = self._calculate_momentum_divergence(history)
        
        # Volume analysis
        recent_volumes = [b.volume for b in history[-self.volume_lookback:-1]]
        avg_volume = np.mean(recent_volumes) if recent_volumes else bar.volume
        institutional_volume = self._is_institutional_volume(bar.volume, avg_volume)
        
        # Current position
        current_position = self._positions.get(bar.symbol, 0)
        current_price = bar.close
        
        # Store regime
        self.regime_state[bar.symbol] = regime
        
        # EXIT LOGIC
        if current_position > 0:
            entry_price = self.entry_prices.get(bar.symbol, current_price)
            pnl_pct = (current_price - entry_price) / entry_price * 100.0
            
            should_exit = False
            exit_reason = ""
            
            # 1. Stop loss
            if pnl_pct <= -self.stop_loss_pct:
                should_exit = True
                exit_reason = "stop_loss"
                self.stop_loss_triggered[bar.symbol] = 0  # Start cooldown
            
            # 2. Take profit
            elif pnl_pct >= self.take_profit_pct:
                should_exit = True
                exit_reason = "take_profit"
            
            # 3. Regime change exit
            elif regime == 'downtrend' and entry_price < current_price:
                should_exit = True
                exit_reason = "regime_change"
            
            # 4. Momentum divergence exit
            elif divergence < -self.divergence_threshold and pnl_pct > 0:
                should_exit = True
                exit_reason = "divergence"
            
            # 5. VWAP reversion exit in ranging market
            elif regime == 'ranging' and vwap and current_price > vwap * (1 + self.vwap_deviation):
                should_exit = True
                exit_reason = "vwap_reversion"
            
            if should_exit:
                if bar.symbol in self.entry_prices:
                    del self.entry_prices[bar.symbol]
                logging.info(f"Exit {bar.symbol} - {exit_reason}: {pnl_pct:.2f}% PnL")
                return TradeSignal(symbol=bar.symbol, side="sell", quantity=current_position)
        
        # ENTRY LOGIC
        elif current_position == 0 and len(self._positions) < self.max_positions:
            should_enter = False
            entry_reason = ""
            
            # Dynamic position sizing based on volatility
            if atr > 0:
                volatility_scalar = min(1.5, max(0.5, 1.0 / (atr / current_price)))
                position_size = self.base_allocation * volatility_scalar
            else:
                position_size = self.base_allocation
            
            # 1. Trend following with volume confirmation
            if regime == 'uptrend' and institutional_volume:
                should_enter = True
                entry_reason = "trend_volume"
            
            # 2. Mean reversion in ranging market
            elif regime == 'ranging' and vwap:
                if current_price < vwap * (1 - self.vwap_deviation) and divergence > 0:
                    should_enter = True
                    entry_reason = "vwap_reversion_long"
            
            # 3. Momentum divergence reversal
            elif abs(divergence) > self.divergence_threshold:
                if divergence > 0 and institutional_volume:
                    should_enter = True
                    entry_reason = "bullish_divergence"
            
            # 4. Breakout with volume
            elif len(history) > 20:
                twenty_bar_high = max(b.high for b in history[-20:])
                if current_price > twenty_bar_high * 0.995 and institutional_volume:
                    should_enter = True
                    entry_reason = "breakout"
            
            if should_enter:
                qty = self.compute_quantity(current_price, position_size)
                if qty > 0:
                    self.entry_prices[bar.symbol] = current_price
                    logging.info(f"Enter {bar.symbol} - {entry_reason}: regime={regime}, vol_spike={institutional_volume}")
                    return TradeSignal(symbol=bar.symbol, side="buy", quantity=qty)
        
        return None
    
    def get_params(self):
        return {
            "volume_lookback": self.volume_lookback,
            "volume_threshold": self.volume_threshold,
            "vwap_deviation": self.vwap_deviation,
            "fast_momentum": self.fast_momentum,
            "slow_momentum": self.slow_momentum,
            "divergence_threshold": self.divergence_threshold,
            "atr_period": self.atr_period,
            "trend_period": self.trend_period,
            "regime_threshold": self.regime_threshold,
            "base_allocation": self.base_allocation,
            "max_positions": self.max_positions,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "stop_cooldown_bars": self.stop_cooldown_bars,
            "min_price": self.min_price,
            "min_volume": self.min_volume,
            "min_bars": self.min_bars
        }
    
    def set_params(self, params):
        # Volume parameters
        self.volume_lookback = max(10, min(50, int(params.get("volume_lookback", self.volume_lookback))))
        self.volume_threshold = max(1.1, min(3.0, float(params.get("volume_threshold", self.volume_threshold))))
        self.vwap_deviation = max(0.001, min(0.01, float(params.get("vwap_deviation", self.vwap_deviation))))
        
        # Momentum parameters
        self.fast_momentum = max(3, min(10, int(params.get("fast_momentum", self.fast_momentum))))
        self.slow_momentum = max(10, min(30, int(params.get("slow_momentum", self.slow_momentum))))
        self.divergence_threshold = max(0.001, min(0.01, float(params.get("divergence_threshold", self.divergence_threshold))))
        
        # Regime parameters
        self.atr_period = max(5, min(20, int(params.get("atr_period", self.atr_period))))
        self.trend_period = max(10, min(50, int(params.get("trend_period", self.trend_period))))
        self.regime_threshold = max(0.3, min(0.9, float(params.get("regime_threshold", self.regime_threshold))))
        
        # Risk parameters
        self.base_allocation = max(0.05, min(0.3, float(params.get("base_allocation", self.base_allocation))))
        self.max_positions = max(1, min(5, int(params.get("max_positions", self.max_positions))))
        self.stop_loss_pct = max(0.5, min(3.0, float(params.get("stop_loss_pct", self.stop_loss_pct))))
        self.take_profit_pct = max(1.0, min(5.0, float(params.get("take_profit_pct", self.take_profit_pct))))
        self.stop_cooldown_bars = max(5, min(30, int(params.get("stop_cooldown_bars", self.stop_cooldown_bars))))
        
        # Filter parameters
        self.min_price = max(1.0, min(20.0, float(params.get("min_price", self.min_price))))
        self.min_volume = max(100000, min(10000000, int(params.get("min_volume", self.min_volume))))
        self.min_bars = max(20, min(100, int(params.get("min_bars", self.min_bars))))
    
    def adapt(self, recent_signals, recent_fills, realized_pnl):
        """Adapt parameters based on recent performance."""
        if not recent_fills:
            # No trades - loosen filters
            self.volume_threshold = max(1.1, self.volume_threshold - 0.1)
            self.min_volume = max(100000, self.min_volume - 100000)
            logging.info("Loosening filters due to lack of trades")
            return
        
        # Analyze regime performance
        regime_performance = {}
        for fill in recent_fills:
            symbol = fill.get('symbol')
            pnl = fill.get('realized_pnl', 0)
            regime = self.regime_state.get(symbol, 'unknown')
            
            if regime not in regime_performance:
                regime_performance[regime] = []
            regime_performance[regime].append(pnl)
        
        # Adjust based on regime success
        for regime, pnls in regime_performance.items():
            avg_pnl = np.mean(pnls)
            if regime == 'trending' and avg_pnl < 0:
                # Trend following not working - tighten criteria
                self.regime_threshold = min(0.9, self.regime_threshold + 0.05)
            elif regime == 'ranging' and avg_pnl > 0:
                # Mean reversion working - widen VWAP bands
                self.vwap_deviation = min(0.01, self.vwap_deviation + 0.0005)
        
        # Overall performance adjustments
        if realized_pnl < -20:
            # Tighten risk
            self.stop_loss_pct = max(0.5, self.stop_loss_pct - 0.25)
            self.base_allocation = max(0.05, self.base_allocation - 0.02)
            logging.info(f"Tightened risk: stop={self.stop_loss_pct}%, allocation={self.base_allocation:.1%}")
        elif realized_pnl > 50:
            # Loosen profit targets to let winners run
            self.take_profit_pct = min(5.0, self.take_profit_pct + 0.25)
            logging.info(f"Increased profit target to {self.take_profit_pct}%")