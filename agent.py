"""
Agent Module for XAUUSD Trading System (Candlestick-Only)

This module implements a PPO agent with LSTM/Transformer architecture
for learning patterns from candlestick sequences.

Author: AI Trading System
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, Callable
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from trading_env import ForexTradingEnv

# Check if tensorboard is available (optional)
try:
    import tensorboard
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class LSTMFeaturesExtractor(BaseFeaturesExtractor):
    """
    LSTM-based feature extractor for candlestick sequences.
    Processes sequences of candles to extract temporal patterns.
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        lstm_hidden_size: int = 128,
        num_lstm_layers: int = 2,
        sequence_length: int = 50,
        candle_features: int = 7
    ):
        """
        Initialize LSTM feature extractor.
        
        Args:
            observation_space: Observation space from environment
            features_dim: Dimension of output features
            lstm_hidden_size: Hidden size of LSTM
            num_lstm_layers: Number of LSTM layers
            sequence_length: Length of candle sequence
            candle_features: Number of features per candle
        """
        super().__init__(observation_space, features_dim)
        
        self.sequence_length = sequence_length
        self.candle_features = candle_features
        
        # Calculate dimensions
        # Observation: (sequence_length * candle_features + time_features + portfolio_state)
        # We need to separate sequence from other features
        total_dim = observation_space.shape[0]
        other_features_dim = total_dim - (sequence_length * candle_features)
        
        # LSTM for processing candle sequence
        self.lstm = nn.LSTM(
            input_size=candle_features,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.1 if num_lstm_layers > 1 else 0
        )
        
        # Process other features (time + portfolio)
        self.other_features_net = nn.Sequential(
            nn.Linear(other_features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Combine LSTM output with other features
        combined_dim = lstm_hidden_size + 32
        self.combined_net = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim)
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM feature extractor.
        
        Args:
            observations: Batch of observations [batch_size, total_features]
            
        Returns:
            Extracted features [batch_size, features_dim]
        """
        batch_size = observations.shape[0]
        
        # Separate sequence from other features
        sequence_dim = self.sequence_length * self.candle_features
        candle_sequence = observations[:, :sequence_dim]
        other_features = observations[:, sequence_dim:]
        
        # Reshape sequence: [batch, sequence_length, candle_features]
        candle_sequence = candle_sequence.view(batch_size, self.sequence_length, self.candle_features)
        
        # Process sequence through LSTM
        lstm_out, (hidden, cell) = self.lstm(candle_sequence)
        
        # Use the last hidden state (or mean pooling)
        # Option 1: Use last hidden state
        lstm_features = lstm_out[:, -1, :]  # [batch, hidden_size]
        
        # Option 2: Mean pooling (alternative)
        # lstm_features = lstm_out.mean(dim=1)  # [batch, hidden_size]
        
        # Process other features
        other_features_processed = self.other_features_net(other_features)
        
        # Combine features
        combined = torch.cat([lstm_features, other_features_processed], dim=1)
        
        # Final processing
        output = self.combined_net(combined)
        
        return output


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """
    Transformer-based feature extractor for candlestick sequences.
    Uses self-attention to find patterns in candle sequences.
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        sequence_length: int = 50,
        candle_features: int = 7
    ):
        """
        Initialize Transformer feature extractor.
        
        Args:
            observation_space: Observation space from environment
            features_dim: Dimension of output features
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            sequence_length: Length of candle sequence
            candle_features: Number of features per candle
        """
        super().__init__(observation_space, features_dim)
        
        self.sequence_length = sequence_length
        self.candle_features = candle_features
        
        total_dim = observation_space.shape[0]
        other_features_dim = total_dim - (sequence_length * candle_features)
        
        # Input projection
        self.input_projection = nn.Linear(candle_features, d_model)
        
        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, sequence_length, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Process other features
        self.other_features_net = nn.Sequential(
            nn.Linear(other_features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Combine and output
        combined_dim = d_model + 32
        self.combined_net = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim)
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Transformer feature extractor.
        
        Args:
            observations: Batch of observations [batch_size, total_features]
            
        Returns:
            Extracted features [batch_size, features_dim]
        """
        batch_size = observations.shape[0]
        
        # Separate sequence from other features
        sequence_dim = self.sequence_length * self.candle_features
        candle_sequence = observations[:, :sequence_dim]
        other_features = observations[:, sequence_dim:]
        
        # Reshape sequence: [batch, sequence_length, candle_features]
        candle_sequence = candle_sequence.view(batch_size, self.sequence_length, self.candle_features)
        
        # Project to model dimension
        x = self.input_projection(candle_sequence)
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Process through transformer
        transformer_out = self.transformer(x)
        
        # Use mean pooling or last token
        transformer_features = transformer_out.mean(dim=1)  # [batch, d_model]
        # Alternative: transformer_features = transformer_out[:, -1, :]
        
        # Process other features
        other_features_processed = self.other_features_net(other_features)
        
        # Combine
        combined = torch.cat([transformer_features, other_features_processed], dim=1)
        
        # Final processing
        output = self.combined_net(combined)
        
        return output


# Custom policies are now defined via policy_kwargs in PPO initialization
# No need to register them separately


class SaveOnBestRewardCallback(BaseCallback):
    """
    Callback to save the model when it achieves the best mean reward.
    """
    
    def __init__(
        self,
        check_freq: int,
        log_dir: str,
        verbose: int = 1,
        best_mean_reward: float = -np.inf
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.best_mean_reward = best_mean_reward
        self.save_path = os.path.join(log_dir, 'best_model')
        
        os.makedirs(log_dir, exist_ok=True)
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if len(self.model.episode_reward_buffer) > 0:
                mean_reward = np.mean(self.model.episode_reward_buffer)
                
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    
                    if self.verbose > 0:
                        print(f"\nNew best mean reward: {mean_reward:.2f}")
                        print(f"Saving model to {self.save_path}")
                    
                    self.model.save(self.save_path)
        
        return True


class TradingAgent:
    """
    PPO-based trading agent with LSTM/Transformer architecture for candlestick sequences.
    """
    
    def __init__(
        self,
        env: ForexTradingEnv,
        model_path: Optional[str] = None,
        architecture: str = "lstm",  # "lstm" or "transformer"
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        tensorboard_log: Optional[str] = None,
        verbose: int = 1
    ):
        """
        Initialize the trading agent.
        
        Args:
            env: Trading environment
            model_path: Path to load existing model (optional)
            architecture: "lstm" or "transformer"
            learning_rate: Learning rate for PPO
            n_steps: Number of steps per update
            batch_size: Batch size for training
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: PPO clip range
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Maximum gradient norm for clipping
            tensorboard_log: Directory for tensorboard logs
            verbose: Verbosity level
        """
        self.env = env
        self.architecture = architecture
        
        # Wrap environment in vectorized environment
        self.vec_env = DummyVecEnv([lambda: Monitor(env)])
        
        # Create or load model
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.model = PPO.load(model_path, env=self.vec_env, verbose=verbose)
        else:
            print(f"Creating new PPO model with {architecture.upper()} architecture")
            
            # Select policy based on architecture
            # Use ActorCriticPolicy directly with custom features extractor
            policy_name = ActorCriticPolicy
            
            if architecture.lower() == "lstm":
                policy_kwargs = {
                    "features_extractor_class": LSTMFeaturesExtractor,
                    "features_extractor_kwargs": {
                        "sequence_length": env.sequence_length,
                        "candle_features": 7,
                        "features_dim": 256,
                        "lstm_hidden_size": 128,
                        "num_lstm_layers": 2
                    }
                }
            elif architecture.lower() == "transformer":
                policy_kwargs = {
                    "features_extractor_class": TransformerFeaturesExtractor,
                    "features_extractor_kwargs": {
                        "sequence_length": env.sequence_length,
                        "candle_features": 7,
                        "features_dim": 256,
                        "d_model": 128,
                        "nhead": 8,
                        "num_layers": 2
                    }
                }
            else:
                raise ValueError(f"Unknown architecture: {architecture}. Use 'lstm' or 'transformer'")
            
            # Handle tensorboard_log - set to None if tensorboard not available
            if tensorboard_log and not TENSORBOARD_AVAILABLE:
                print("Warning: TensorBoard not installed. Logging disabled.")
                print("To enable: pip install tensorboard")
                tensorboard_log = None
            
            self.model = PPO(
                policy_name,
                self.vec_env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                policy_kwargs=policy_kwargs,
                tensorboard_log=tensorboard_log,
                verbose=verbose,
                device='cpu'  # Use 'cuda' if GPU available
            )
    
    def train(
        self,
        total_timesteps: int,
        log_dir: str = "logs",
        save_freq: int = 10000,
        eval_env: Optional[ForexTradingEnv] = None,
        eval_freq: int = 50000
    ) -> None:
        """
        Train the agent.
        
        Args:
            total_timesteps: Total number of training timesteps
            log_dir: Directory for logs and model saves
            save_freq: Frequency of model saves (in steps)
            eval_env: Evaluation environment (optional)
            eval_freq: Frequency of evaluation (in steps)
        """
        os.makedirs(log_dir, exist_ok=True)
        
        callbacks = []
        
        save_callback = SaveOnBestRewardCallback(
            check_freq=save_freq,
            log_dir=log_dir,
            verbose=1
        )
        callbacks.append(save_callback)
        
        if eval_env is not None:
            eval_vec_env = DummyVecEnv([lambda: Monitor(eval_env)])
            eval_callback = EvalCallback(
                eval_vec_env,
                best_model_save_path=os.path.join(log_dir, 'best_eval_model'),
                log_path=os.path.join(log_dir, 'eval_logs'),
                eval_freq=eval_freq,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)
        
        print(f"Starting training for {total_timesteps} timesteps...")
        print(f"Architecture: {self.architecture.upper()}")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        final_model_path = os.path.join(log_dir, 'final_model')
        self.model.save(final_model_path)
        print(f"Training completed. Final model saved to {final_model_path}")
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[int, np.ndarray]:
        """
        Predict action for given observation.
        
        Args:
            observation: Current state observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Action and action probabilities
        """
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return action, _states
    
    def evaluate(
        self,
        eval_env: ForexTradingEnv,
        n_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate the agent on a test environment.
        
        Args:
            eval_env: Evaluation environment
            n_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic policy
            render: Whether to render episodes
            
        Returns:
            Dictionary with evaluation metrics
        """
        eval_vec_env = DummyVecEnv([lambda: eval_env])
        
        mean_reward, std_reward = evaluate_policy(
            self.model,
            eval_vec_env,
            n_eval_episodes=n_episodes,
            deterministic=deterministic,
            render=render
        )
        
        results = {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'episodes': n_episodes
        }
        
        total_equity = []
        total_trades = []
        win_rates = []
        final_balances = []
        
        for _ in range(n_episodes):
            obs, _ = eval_env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = eval_env.step(int(action))
                done = terminated or truncated
                episode_reward += reward
            
            final_info = eval_env._get_info()
            total_equity.append(final_info['equity'])
            total_trades.append(final_info['total_trades'])
            win_rates.append(final_info['win_rate'])
            final_balances.append(final_info['balance'])
        
        results.update({
            'mean_final_equity': np.mean(total_equity),
            'std_final_equity': np.std(total_equity),
            'mean_total_trades': np.mean(total_trades),
            'mean_win_rate': np.mean(win_rates),
            'mean_final_balance': np.mean(final_balances),
            'total_return_pct': (np.mean(total_equity) - eval_env.initial_balance) / eval_env.initial_balance * 100
        })
        
        return results
    
    def save(self, path: str) -> None:
        """Save the model to disk."""
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load the model from disk."""
        self.model = PPO.load(path, env=self.vec_env)
        print(f"Model loaded from {path}")


def split_data_for_training(
    data: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into training, validation, and test sets.
    
    Args:
        data: Full dataset
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = data.iloc[:train_end].copy()
    val_data = data.iloc[train_end:val_end].copy()
    test_data = data.iloc[val_end:].copy()
    
    print(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    return train_data, val_data, test_data


if __name__ == "__main__":
    from data_provider import load_data_from_csv
    
    print("Loading data...")
    data = load_data_from_csv("data/xauusd_m15.csv")
    
    train_data, val_data, test_data = split_data_for_training(data)
    
    print("Creating training environment...")
    train_env = ForexTradingEnv(
        data=train_data,
        sequence_length=50,
        initial_balance=10000.0,
        lot_size=0.01,
        max_positions=5
    )
    
    print("Creating validation environment...")
    val_env = ForexTradingEnv(
        data=val_data,
        sequence_length=50,
        initial_balance=10000.0,
        lot_size=0.01,
        max_positions=5
    )
    
    print("Creating agent with LSTM architecture...")
    agent = TradingAgent(
        env=train_env,
        architecture="lstm",  # or "transformer"
        tensorboard_log="logs/tensorboard"
    )
    
    print("Starting training...")
    agent.train(
        total_timesteps=100000,
        log_dir="logs",
        save_freq=10000,
        eval_env=val_env,
        eval_freq=50000
    )
    
    print("Evaluating on test set...")
    test_env = ForexTradingEnv(
        data=test_data,
        sequence_length=50,
        initial_balance=10000.0,
        lot_size=0.01,
        max_positions=5
    )
    
    results = agent.evaluate(test_env, n_episodes=10)
    print("\nEvaluation Results:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")
