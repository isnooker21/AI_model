"""
Trading Environment Module for XAUUSD Reinforcement Learning System
(Candlestick-Only Price Action Approach)

This module implements a Gymnasium environment using raw candlestick sequences
without any lagging indicators. The AI learns from price action patterns.

Author: AI Trading System
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import gymnasium as gym
from gymnasium import spaces
from collections import deque


class ForexTradingEnv(gym.Env):
    """
    Gymnasium environment for XAUUSD trading using candlestick sequences.
    
    State Space:
        - Sequence of last 50 candles
        - For each candle: Body Size, Upper Wick, Lower Wick, Price Change
        - Time of Day features
        - Portfolio state (Balance, Equity, Floating P/L, Positions, Avg Entry, Drawdown %)
    
    Action Space:
        - 0: Hold
        - 1: Initial Buy
        - 2: Initial Sell
        - 3: Recovery Buy (Scale-in)
        - 4: Recovery Sell (Scale-in)
        - 5: Close All Positions
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(
        self,
        data: pd.DataFrame,
        sequence_length: int = 50,
        initial_balance: float = 10000.0,
        lot_size: float = 0.01,
        max_positions: int = 5,
        max_drawdown_pct: float = 0.20,  # 20% max drawdown
        recovery_threshold_pct: float = 0.05,  # 5% drawdown triggers recovery
        commission_per_lot: float = 7.0,  # $7 per lot
        spread_pips: float = 2.0,  # 2 pips spread
        time_penalty: float = -0.001,  # Small penalty per step
        reward_scale: float = 1.0,
        normalize_features: bool = True
    ):
        """
        Initialize the trading environment.
        
        Args:
            data: DataFrame with OHLC and candlestick features
            sequence_length: Number of candles in the sequence (default: 50)
            initial_balance: Starting account balance
            lot_size: Size of each position (in lots)
            max_positions: Maximum number of concurrent positions
            max_drawdown_pct: Maximum allowed drawdown percentage
            recovery_threshold_pct: Drawdown threshold to trigger recovery
            commission_per_lot: Commission per lot traded
            spread_pips: Spread in pips
            time_penalty: Penalty per time step
            reward_scale: Scaling factor for rewards
            normalize_features: Whether to normalize feature values
        """
        super().__init__()
        
        self.data = data.copy()
        self.sequence_length = sequence_length
        self.initial_balance = initial_balance
        self.lot_size = lot_size
        self.max_positions = max_positions
        self.max_drawdown_pct = max_drawdown_pct
        self.recovery_threshold_pct = recovery_threshold_pct
        self.commission_per_lot = commission_per_lot
        self.spread_pips = spread_pips
        self.time_penalty = time_penalty
        self.reward_scale = reward_scale
        self.normalize_features = normalize_features
        
        # Trading parameters
        self.pip_value = 0.01  # For XAUUSD, 1 pip = $0.01 per lot
        self.point_value = 0.01
        
        # Required candlestick feature columns
        self.candle_features = [
            'body_size', 'upper_wick', 'lower_wick', 
            'price_change', 'price_change_pct', 'candle_direction',
            'body_to_range'
        ]
        
        # Time features
        self.time_features = [
            'hour_sin', 'hour_cos', 'day_of_week',
            'session_asia', 'session_europe', 'session_us'
        ]
        
        # Ensure all required columns exist
        required_cols = ['open', 'high', 'low', 'close'] + self.candle_features + self.time_features
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Normalize features if requested
        if self.normalize_features:
            self._normalize_data()
        
        # Calculate feature statistics for normalization
        self.feature_stats = self._calculate_feature_stats()
        
        # State space dimensions
        # Sequence: 50 candles × 7 features = 350
        # Time features: 6 (current candle)
        # Portfolio state: 6
        # Total: 350 + 6 + 6 = 362 dimensions (flattened)
        # OR we can use a 2D shape: (sequence_length, features_per_candle + time_features + portfolio_state)
        
        # For LSTM/Transformer, we'll use a 2D observation space
        # Shape: (sequence_length, features_per_candle)
        n_candle_features = len(self.candle_features)
        n_time_features = len(self.time_features)
        n_portfolio = 6  # Balance, Equity, Floating P/L, Num Positions, Avg Entry, Drawdown %
        
        # Observation: (sequence_length, n_candle_features + n_time_features + n_portfolio)
        # But portfolio state is same for all candles, so we'll add it separately
        # Actually, let's flatten: sequence of candles + current time + portfolio
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(sequence_length * n_candle_features + n_time_features + n_portfolio,),
            dtype=np.float32
        )
        
        # Action space: 6 discrete actions
        self.action_space = spaces.Discrete(6)
        
        # Reset environment
        self.reset()
    
    def _normalize_data(self) -> None:
        """Normalize candlestick features using rolling statistics."""
        # Normalize OHLC using rolling mean and std
        for col in ['open', 'high', 'low', 'close']:
            rolling_mean = self.data[col].rolling(window=100, min_periods=1).mean()
            rolling_std = self.data[col].rolling(window=100, min_periods=1).std()
            self.data[f'{col}_normalized'] = (self.data[col] - rolling_mean) / (rolling_std + 1e-8)
        
        # Normalize candlestick features
        for col in self.candle_features:
            if col in self.data.columns:
                rolling_mean = self.data[col].rolling(window=100, min_periods=1).mean()
                rolling_std = self.data[col].rolling(window=100, min_periods=1).std()
                self.data[f'{col}_normalized'] = (self.data[col] - rolling_mean) / (rolling_std + 1e-8)
    
    def _calculate_feature_stats(self) -> Dict:
        """Calculate statistics for feature normalization."""
        stats = {}
        for col in self.candle_features + self.time_features:
            if col in self.data.columns:
                stats[col] = {
                    'mean': self.data[col].mean(),
                    'std': self.data[col].std(),
                    'min': self.data[col].min(),
                    'max': self.data[col].max()
                }
        return stats
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Initial observation and info dictionary
        """
        super().reset(seed=seed)
        
        # Reset trading state
        self.current_step = self.sequence_length  # Start after we have enough history
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.realized_pnl = 0.0
        self.peak_equity = self.initial_balance
        
        # Position tracking
        self.positions: List[Dict] = []  # List of open positions
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_commission = 0.0
        
        # History for analysis
        self.equity_history = [self.initial_balance]
        self.balance_history = [self.initial_balance]
        self.action_history = []
        self.reward_history = []
        
        # Price prediction tracking (for reward calculation)
        self.last_prediction = None
        self.last_prediction_step = None
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (state) - sequence of last 50 candles.
        
        Returns:
            Observation array: [Sequence of 50 candles, Time features, Portfolio State]
        """
        if self.current_step >= len(self.data):
            # Return last valid observation
            self.current_step = len(self.data) - 1
        
        # Get sequence of last N candles
        start_idx = max(0, self.current_step - self.sequence_length + 1)
        end_idx = self.current_step + 1
        
        sequence_data = self.data.iloc[start_idx:end_idx]
        
        # If we don't have enough history, pad with first candle
        if len(sequence_data) < self.sequence_length:
            padding = self.sequence_length - len(sequence_data)
            first_candle = sequence_data.iloc[0:1]
            padding_data = pd.concat([first_candle] * padding, ignore_index=True)
            sequence_data = pd.concat([padding_data, sequence_data], ignore_index=True)
        
        # Extract candlestick features for each candle in sequence
        candle_sequence = []
        for idx in range(self.sequence_length):
            row = sequence_data.iloc[idx]
            
            # Get candlestick features (normalized if available)
            candle_features = []
            for col in self.candle_features:
                if self.normalize_features and f'{col}_normalized' in row:
                    candle_features.append(row[f'{col}_normalized'])
                else:
                    # Normalize on the fly
                    val = row[col]
                    if col in ['price_change_pct', 'body_to_range']:
                        # Already normalized
                        candle_features.append(val)
                    else:
                        # Normalize by price
                        if col in ['body_size', 'upper_wick', 'lower_wick', 'price_change']:
                            norm_val = val / (row['close'] + 1e-8)
                            candle_features.append(norm_val)
                        else:
                            candle_features.append(val)
            
            candle_sequence.extend(candle_features)
        
        # Get current time features
        current_row = self.data.iloc[self.current_step]
        time_features = []
        for col in self.time_features:
            time_features.append(current_row[col])
        
        # Portfolio state
        floating_pnl = self._calculate_floating_pnl()
        current_equity = self.balance + floating_pnl
        self.equity = current_equity
        
        # Update peak equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Calculate drawdown percentage
        drawdown_pct = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        
        # Average entry price (weighted by lot size)
        if len(self.positions) > 0:
            total_lots = sum(pos['lot_size'] for pos in self.positions)
            avg_entry = sum(pos['entry_price'] * pos['lot_size'] for pos in self.positions) / total_lots
        else:
            avg_entry = current_row['close']  # Use current price if no positions
        
        portfolio_state = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            current_equity / self.initial_balance,  # Normalized equity
            floating_pnl / self.initial_balance,  # Normalized floating P/L
            len(self.positions) / self.max_positions,  # Normalized position count
            (avg_entry - current_row['close']) / current_row['close'],  # Normalized entry price difference
            drawdown_pct  # Drawdown percentage
        ], dtype=np.float32)
        
        # Combine all observations
        observation = np.concatenate([
            np.array(candle_sequence, dtype=np.float32),
            np.array(time_features, dtype=np.float32),
            portfolio_state
        ])
        
        return observation
    
    def _calculate_floating_pnl(self) -> float:
        """
        Calculate current floating P/L for all open positions.
        
        Returns:
            Total floating P/L
        """
        if len(self.positions) == 0:
            return 0.0
        
        current_price = self.data.iloc[self.current_step]['close']
        floating_pnl = 0.0
        
        for position in self.positions:
            entry_price = position['entry_price']
            lot_size = position['lot_size']
            position_type = position['type']
            
            if position_type == 'buy':
                pnl = (current_price - entry_price) * lot_size * 100  # $100 per lot per $1 move
            else:  # sell
                pnl = (entry_price - current_price) * lot_size * 100
            
            floating_pnl += pnl
        
        return floating_pnl
    
    def _calculate_avg_entry_price(self, position_type: str) -> float:
        """
        Calculate weighted average entry price for positions of given type.
        
        Args:
            position_type: 'buy' or 'sell'
            
        Returns:
            Weighted average entry price
        """
        matching_positions = [pos for pos in self.positions if pos['type'] == position_type]
        
        if len(matching_positions) == 0:
            return 0.0
        
        total_lots = sum(pos['lot_size'] for pos in matching_positions)
        weighted_price = sum(pos['entry_price'] * pos['lot_size'] for pos in matching_positions)
        
        return weighted_price / total_lots if total_lots > 0 else 0.0
    
    def _can_recovery_trade(self, action_type: str) -> bool:
        """
        Check if recovery trade is allowed based on drawdown and position state.
        
        Args:
            action_type: 'buy' or 'sell'
            
        Returns:
            True if recovery trade is allowed
        """
        if len(self.positions) == 0:
            return False
        
        # Check if we have positions of the same type
        matching_positions = [pos for pos in self.positions if pos['type'] == action_type]
        if len(matching_positions) == 0:
            return False
        
        # Check drawdown threshold
        current_equity = self.balance + self._calculate_floating_pnl()
        drawdown_pct = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        
        if drawdown_pct < self.recovery_threshold_pct:
            return False  # Not enough drawdown to trigger recovery
        
        # Check max positions
        if len(self.positions) >= self.max_positions:
            return False
        
        # Check max drawdown
        if drawdown_pct >= self.max_drawdown_pct:
            return False  # Too much drawdown, don't add more risk
        
        return True
    
    def _detect_market_regime(self) -> str:
        """
        Detect market regime (Trend/Sideways) from candlestick sequence.
        This is a helper function - the AI will learn this pattern.
        
        Returns:
            'trend' or 'sideways'
        """
        if self.current_step < self.sequence_length:
            return 'sideways'
        
        # Get recent candles
        recent_data = self.data.iloc[max(0, self.current_step - 20):self.current_step + 1]
        
        # Simple trend detection: check if price is consistently moving in one direction
        price_changes = recent_data['close'].diff().dropna()
        positive_changes = (price_changes > 0).sum()
        negative_changes = (price_changes < 0).sum()
        
        # If 70%+ moves in one direction, consider it a trend
        if positive_changes / len(price_changes) > 0.7:
            return 'trend'
        elif negative_changes / len(price_changes) > 0.7:
            return 'trend'
        else:
            return 'sideways'
    
    def _execute_trade(self, action: int) -> Dict:
        """
        Execute a trading action.
        
        Args:
            action: Action to execute (0-5)
            
        Returns:
            Dictionary with trade execution details
        """
        current_price = self.data.iloc[self.current_step]['close']
        trade_result = {
            'executed': False,
            'action': action,
            'pnl': 0.0,
            'commission': 0.0,
            'message': ''
        }
        
        if action == 0:  # Hold
            trade_result['message'] = 'Hold'
            return trade_result
        
        elif action == 1:  # Initial Buy
            if len(self.positions) >= self.max_positions:
                trade_result['message'] = 'Max positions reached'
                return trade_result
            
            commission = self.commission_per_lot * self.lot_size
            if self.balance < commission:
                trade_result['message'] = 'Insufficient balance for commission'
                return trade_result
            
            self.balance -= commission
            self.total_commission += commission
            
            self.positions.append({
                'type': 'buy',
                'entry_price': current_price,
                'lot_size': self.lot_size,
                'entry_step': self.current_step
            })
            
            trade_result['executed'] = True
            trade_result['commission'] = commission
            trade_result['message'] = f'Buy opened at {current_price:.2f}'
            self.total_trades += 1
        
        elif action == 2:  # Initial Sell
            if len(self.positions) >= self.max_positions:
                trade_result['message'] = 'Max positions reached'
                return trade_result
            
            commission = self.commission_per_lot * self.lot_size
            if self.balance < commission:
                trade_result['message'] = 'Insufficient balance for commission'
                return trade_result
            
            self.balance -= commission
            self.total_commission += commission
            
            self.positions.append({
                'type': 'sell',
                'entry_price': current_price,
                'lot_size': self.lot_size,
                'entry_step': self.current_step
            })
            
            trade_result['executed'] = True
            trade_result['commission'] = commission
            trade_result['message'] = f'Sell opened at {current_price:.2f}'
            self.total_trades += 1
        
        elif action == 3:  # Recovery Buy
            if not self._can_recovery_trade('buy'):
                trade_result['message'] = 'Recovery buy not allowed'
                return trade_result
            
            avg_entry = self._calculate_avg_entry_price('buy')
            total_lots = sum(pos['lot_size'] for pos in self.positions if pos['type'] == 'buy')
            new_lots = total_lots + self.lot_size
            new_avg_entry = (avg_entry * total_lots + current_price * self.lot_size) / new_lots
            
            commission = self.commission_per_lot * self.lot_size
            if self.balance < commission:
                trade_result['message'] = 'Insufficient balance for commission'
                return trade_result
            
            self.balance -= commission
            self.total_commission += commission
            
            self.positions.append({
                'type': 'buy',
                'entry_price': current_price,
                'lot_size': self.lot_size,
                'entry_step': self.current_step
            })
            
            trade_result['executed'] = True
            trade_result['commission'] = commission
            trade_result['message'] = f'Recovery buy at {current_price:.2f}, new avg: {new_avg_entry:.2f}'
        
        elif action == 4:  # Recovery Sell
            if not self._can_recovery_trade('sell'):
                trade_result['message'] = 'Recovery sell not allowed'
                return trade_result
            
            avg_entry = self._calculate_avg_entry_price('sell')
            total_lots = sum(pos['lot_size'] for pos in self.positions if pos['type'] == 'sell')
            new_lots = total_lots + self.lot_size
            new_avg_entry = (avg_entry * total_lots + current_price * self.lot_size) / new_lots
            
            commission = self.commission_per_lot * self.lot_size
            if self.balance < commission:
                trade_result['message'] = 'Insufficient balance for commission'
                return trade_result
            
            self.balance -= commission
            self.total_commission += commission
            
            self.positions.append({
                'type': 'sell',
                'entry_price': current_price,
                'lot_size': self.lot_size,
                'entry_step': self.current_step
            })
            
            trade_result['executed'] = True
            trade_result['commission'] = commission
            trade_result['message'] = f'Recovery sell at {current_price:.2f}, new avg: {new_avg_entry:.2f}'
        
        elif action == 5:  # Close All
            if len(self.positions) == 0:
                trade_result['message'] = 'No positions to close'
                return trade_result
            
            total_pnl = 0.0
            positions_to_close = self.positions.copy()
            
            for position in positions_to_close:
                entry_price = position['entry_price']
                lot_size = position['lot_size']
                position_type = position['type']
                
                if position_type == 'buy':
                    pnl = (current_price - entry_price) * lot_size * 100
                else:  # sell
                    pnl = (entry_price - current_price) * lot_size * 100
                
                total_pnl += pnl
            
            close_commission = self.commission_per_lot * sum(pos['lot_size'] for pos in positions_to_close)
            net_pnl = total_pnl - close_commission
            
            self.balance += net_pnl
            self.realized_pnl += net_pnl
            self.total_commission += close_commission
            
            for position in positions_to_close:
                if net_pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
            
            self.positions = []
            
            trade_result['executed'] = True
            trade_result['pnl'] = net_pnl
            trade_result['commission'] = close_commission
            trade_result['message'] = f'Closed all positions, P/L: {net_pnl:.2f}'
        
        return trade_result
    
    def _calculate_reward(self, trade_result: Dict) -> float:
        """
        Calculate reward based on trading action and portfolio state.
        Rewards correct price direction prediction and penalizes drawdowns.
        
        Args:
            trade_result: Result from trade execution
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Time penalty for every step
        reward += self.time_penalty
        
        # Reward for realized profit
        if trade_result.get('pnl', 0) > 0:
            reward += trade_result['pnl'] / self.initial_balance * 10.0
        
        # Penalty for realized loss
        elif trade_result.get('pnl', 0) < 0:
            reward += trade_result['pnl'] / self.initial_balance * 10.0
        
        # Reward for correct price direction prediction (if we opened a position)
        if trade_result['executed'] and trade_result['action'] in [1, 2]:
            # Check if price moved in predicted direction
            if self.current_step < len(self.data) - 1:
                current_price = self.data.iloc[self.current_step]['close']
                next_price = self.data.iloc[self.current_step + 1]['close']
                price_change = next_price - current_price
                
                if trade_result['action'] == 1:  # Buy - expect price to go up
                    if price_change > 0:
                        reward += 0.1  # Small reward for correct prediction
                    else:
                        reward -= 0.05  # Small penalty for wrong prediction
                elif trade_result['action'] == 2:  # Sell - expect price to go down
                    if price_change < 0:
                        reward += 0.1
                    else:
                        reward -= 0.05
        
        # Penalty for high drawdown
        current_equity = self.balance + self._calculate_floating_pnl()
        drawdown_pct = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        
        if drawdown_pct > 0.10:  # 10% drawdown
            reward -= drawdown_pct * 5.0
        
        # Penalty for excessive positions
        if len(self.positions) >= self.max_positions:
            reward -= 0.1
        
        # Small reward for closing positions
        if trade_result['action'] == 5 and trade_result['executed']:
            reward += 0.05
        
        return reward * self.reward_scale
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0-5)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.current_step >= len(self.data) - 1:
            terminated = True
            truncated = False
        else:
            self.current_step += 1
            terminated = False
            truncated = False
        
        # Execute trade
        trade_result = self._execute_trade(action)
        
        # Calculate reward
        reward = self._calculate_reward(trade_result)
        
        # Check termination conditions
        current_equity = self.balance + self._calculate_floating_pnl()
        
        if current_equity < self.initial_balance * 0.1:
            terminated = True
            reward -= 10.0
        
        drawdown_pct = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        if drawdown_pct >= self.max_drawdown_pct:
            terminated = True
            reward -= 5.0
        
        # Get new observation
        observation = self._get_observation()
        
        # Update history
        self.equity_history.append(current_equity)
        self.balance_history.append(self.balance)
        self.action_history.append(action)
        self.reward_history.append(reward)
        
        # Prepare info
        info = self._get_info()
        info.update(trade_result)
        info['market_regime'] = self._detect_market_regime()
        
        return observation, reward, terminated, truncated, info
    
    def _get_info(self) -> Dict:
        """Get current environment information."""
        current_equity = self.balance + self._calculate_floating_pnl()
        drawdown_pct = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        
        return {
            'step': self.current_step,
            'balance': self.balance,
            'equity': current_equity,
            'realized_pnl': self.realized_pnl,
            'floating_pnl': self._calculate_floating_pnl(),
            'num_positions': len(self.positions),
            'drawdown_pct': drawdown_pct,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_commission': self.total_commission,
            'win_rate': self.winning_trades / max(self.total_trades, 1)
        }
    
    def render(self, mode: str = "human") -> None:
        """Render the environment state."""
        if mode == "human":
            info = self._get_info()
            print(f"Step: {info['step']}, Equity: ${info['equity']:.2f}, "
                  f"Positions: {info['num_positions']}, Drawdown: {info['drawdown_pct']*100:.2f}%")
    
    def close(self) -> None:
        """Clean up environment resources."""
        pass
