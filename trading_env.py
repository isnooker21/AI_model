"""
Trading Environment Module for XAUUSD Reinforcement Learning System

This module implements a Gymnasium environment for forex trading with
recovery logic, dynamic exits, and comprehensive state/action spaces.

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
    Gymnasium environment for XAUUSD trading with recovery and dynamic exit management.
    
    State Space:
        - OHLC data (normalized)
        - Technical indicators (RSI, ATR, Bollinger Bands)
        - Session indicators (Asia/Europe/US)
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
            data: DataFrame with OHLC and technical indicators
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
        
        # Feature columns (excluding OHLC which we'll handle separately)
        self.feature_columns = [
            'rsi', 'atr', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'session_asia', 'session_europe', 'session_us'
        ]
        
        # Ensure all required columns exist
        required_cols = ['open', 'high', 'low', 'close'] + self.feature_columns
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Normalize features if requested
        if self.normalize_features:
            self._normalize_data()
        
        # Calculate feature statistics for normalization
        self.feature_stats = self._calculate_feature_stats()
        
        # State space dimensions
        # OHLC (4) + Features (9) + Portfolio State (6) = 19 dimensions
        n_ohlc = 4
        n_features = len(self.feature_columns)
        n_portfolio = 6  # Balance, Equity, Floating P/L, Num Positions, Avg Entry, Drawdown %
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_ohlc + n_features + n_portfolio,),
            dtype=np.float32
        )
        
        # Action space: 6 discrete actions
        self.action_space = spaces.Discrete(6)
        
        # Reset environment
        self.reset()
    
    def _normalize_data(self) -> None:
        """Normalize OHLC and feature data using rolling statistics."""
        # Normalize OHLC using rolling mean and std
        for col in ['open', 'high', 'low', 'close']:
            rolling_mean = self.data[col].rolling(window=100, min_periods=1).mean()
            rolling_std = self.data[col].rolling(window=100, min_periods=1).std()
            self.data[f'{col}_normalized'] = (self.data[col] - rolling_mean) / (rolling_std + 1e-8)
        
        # Normalize RSI (0-100 scale, normalize to -1 to 1)
        if 'rsi' in self.data.columns:
            self.data['rsi_normalized'] = (self.data['rsi'] - 50) / 50
        
        # Normalize ATR relative to price
        if 'atr' in self.data.columns and 'close' in self.data.columns:
            self.data['atr_normalized'] = self.data['atr'] / (self.data['close'] + 1e-8)
        
        # Normalize Bollinger Bands
        for col in ['bb_upper', 'bb_middle', 'bb_lower', 'bb_width']:
            if col in self.data.columns:
                rolling_mean = self.data[col].rolling(window=100, min_periods=1).mean()
                rolling_std = self.data[col].rolling(window=100, min_periods=1).std()
                self.data[f'{col}_normalized'] = (self.data[col] - rolling_mean) / (rolling_std + 1e-8)
    
    def _calculate_feature_stats(self) -> Dict:
        """Calculate statistics for feature normalization."""
        stats = {}
        for col in self.feature_columns:
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
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.realized_pnl = 0.0
        self.peak_equity = self.initial_balance
        
        # Position tracking
        self.positions: List[Dict] = []  # List of open positions
        # Each position: {'type': 'buy'/'sell', 'entry_price': float, 'lot_size': float, 'entry_step': int}
        
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
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (state).
        
        Returns:
            Observation array: [OHLC, Features, Portfolio State]
        """
        if self.current_step >= len(self.data):
            # Return last valid observation
            self.current_step = len(self.data) - 1
        
        row = self.data.iloc[self.current_step]
        
        # OHLC data (normalized if available, otherwise raw)
        if self.normalize_features:
            ohlc = np.array([
                row.get('open_normalized', row['open']),
                row.get('high_normalized', row['high']),
                row.get('low_normalized', row['low']),
                row.get('close_normalized', row['close'])
            ], dtype=np.float32)
        else:
            ohlc = np.array([
                row['open'],
                row['high'],
                row['low'],
                row['close']
            ], dtype=np.float32)
        
        # Feature data
        features = []
        for col in self.feature_columns:
            if self.normalize_features and f'{col}_normalized' in row:
                features.append(row[f'{col}_normalized'])
            else:
                # Normalize on the fly
                val = row[col]
                if col == 'rsi':
                    features.append((val - 50) / 50)  # Normalize RSI
                elif col == 'atr':
                    features.append(val / (row['close'] + 1e-8))  # Normalize ATR
                else:
                    features.append(val)
        
        features = np.array(features, dtype=np.float32)
        
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
            avg_entry = row['close']  # Use current price if no positions
        
        portfolio_state = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            current_equity / self.initial_balance,  # Normalized equity
            floating_pnl / self.initial_balance,  # Normalized floating P/L
            len(self.positions) / self.max_positions,  # Normalized position count
            (avg_entry - row['close']) / row['close'],  # Normalized entry price difference
            drawdown_pct  # Drawdown percentage
        ], dtype=np.float32)
        
        # Combine all observations
        observation = np.concatenate([ohlc, features, portfolio_state])
        
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
            
            # Open new buy position
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
            
            # Open new sell position
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
            
            # Calculate new average entry price
            avg_entry = self._calculate_avg_entry_price('buy')
            total_lots = sum(pos['lot_size'] for pos in self.positions if pos['type'] == 'buy')
            new_lots = total_lots + self.lot_size
            new_avg_entry = (avg_entry * total_lots + current_price * self.lot_size) / new_lots
            
            # Open recovery position
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
            
            # Calculate new average entry price
            avg_entry = self._calculate_avg_entry_price('sell')
            total_lots = sum(pos['lot_size'] for pos in self.positions if pos['type'] == 'sell')
            new_lots = total_lots + self.lot_size
            new_avg_entry = (avg_entry * total_lots + current_price * self.lot_size) / new_lots
            
            # Open recovery position
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
            
            # Close all positions and realize P/L
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
            
            # Deduct commission for closing
            close_commission = self.commission_per_lot * sum(pos['lot_size'] for pos in positions_to_close)
            net_pnl = total_pnl - close_commission
            
            self.balance += net_pnl
            self.realized_pnl += net_pnl
            self.total_commission += close_commission
            
            # Update trade statistics
            for position in positions_to_close:
                if net_pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
            
            # Clear positions
            self.positions = []
            
            trade_result['executed'] = True
            trade_result['pnl'] = net_pnl
            trade_result['commission'] = close_commission
            trade_result['message'] = f'Closed all positions, P/L: {net_pnl:.2f}'
        
        return trade_result
    
    def _calculate_reward(self, trade_result: Dict) -> float:
        """
        Calculate reward based on trading action and portfolio state.
        
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
            reward += trade_result['pnl'] / self.initial_balance * 10.0  # Scale reward
        
        # Penalty for realized loss
        elif trade_result.get('pnl', 0) < 0:
            reward += trade_result['pnl'] / self.initial_balance * 10.0  # Negative reward
        
        # Penalty for high drawdown
        current_equity = self.balance + self._calculate_floating_pnl()
        drawdown_pct = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        
        if drawdown_pct > 0.10:  # 10% drawdown
            reward -= drawdown_pct * 5.0  # Penalty increases with drawdown
        
        # Penalty for excessive positions (encourage position management)
        if len(self.positions) >= self.max_positions:
            reward -= 0.1
        
        # Small reward for closing positions (encourage active management)
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
            # End of data
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
        
        # Terminate if balance is too low (account blown)
        if current_equity < self.initial_balance * 0.1:  # Less than 10% of initial
            terminated = True
            reward -= 10.0  # Large penalty for account blow-up
        
        # Terminate if max drawdown exceeded
        drawdown_pct = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        if drawdown_pct >= self.max_drawdown_pct:
            terminated = True
            reward -= 5.0  # Penalty for exceeding max drawdown
        
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

