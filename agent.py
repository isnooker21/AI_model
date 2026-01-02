"""
Agent Module for XAUUSD Trading System

This module implements a PPO (Proximal Policy Optimization) agent using
Stable Baselines3 for reinforcement learning.

Author: AI Trading System
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, Callable
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
from trading_env import ForexTradingEnv


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
        """
        Initialize the callback.
        
        Args:
            check_freq: Frequency of checks (in steps)
            log_dir: Directory to save models
            verbose: Verbosity level
            best_mean_reward: Initial best mean reward
        """
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.best_mean_reward = best_mean_reward
        self.save_path = os.path.join(log_dir, 'best_model')
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
    
    def _on_step(self) -> bool:
        """
        Called at each step of training.
        
        Returns:
            True to continue training, False to stop
        """
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
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
    PPO-based trading agent for XAUUSD.
    """
    
    def __init__(
        self,
        env: ForexTradingEnv,
        model_path: Optional[str] = None,
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
        
        # Wrap environment in vectorized environment
        self.vec_env = DummyVecEnv([lambda: Monitor(env)])
        
        # Create or load model
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.model = PPO.load(model_path, env=self.vec_env, verbose=verbose)
        else:
            print("Creating new PPO model")
            self.model = PPO(
                "MlpPolicy",
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
        
        # Create callbacks
        callbacks = []
        
        # Save on best reward callback
        save_callback = SaveOnBestRewardCallback(
            check_freq=save_freq,
            log_dir=log_dir,
            verbose=1
        )
        callbacks.append(save_callback)
        
        # Evaluation callback (if eval_env provided)
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
        
        # Train the model
        print(f"Starting training for {total_timesteps} timesteps...")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        # Save final model
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
        # Wrap environment
        eval_vec_env = DummyVecEnv([lambda: eval_env])
        
        # Evaluate policy
        mean_reward, std_reward = evaluate_policy(
            self.model,
            eval_vec_env,
            n_eval_episodes=n_episodes,
            deterministic=deterministic,
            render=render
        )
        
        # Run additional evaluation to collect detailed metrics
        results = {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'episodes': n_episodes
        }
        
        # Collect detailed statistics
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
            
            # Collect final statistics
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
            'total_return_pct': (np.mean(final_equity) - eval_env.initial_balance) / eval_env.initial_balance * 100
        })
        
        return results
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
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
        test_ratio: Remaining data for testing (calculated automatically)
        
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
    # Example usage
    from data_provider import load_data_from_csv
    
    # Load data
    print("Loading data...")
    data = load_data_from_csv("data/xauusd_m15.csv")
    
    # Split data
    train_data, val_data, test_data = split_data_for_training(data)
    
    # Create environments
    print("Creating training environment...")
    train_env = ForexTradingEnv(
        data=train_data,
        initial_balance=10000.0,
        lot_size=0.01,
        max_positions=5
    )
    
    print("Creating validation environment...")
    val_env = ForexTradingEnv(
        data=val_data,
        initial_balance=10000.0,
        lot_size=0.01,
        max_positions=5
    )
    
    # Create agent
    print("Creating agent...")
    agent = TradingAgent(
        env=train_env,
        tensorboard_log="logs/tensorboard"
    )
    
    # Train agent
    print("Starting training...")
    agent.train(
        total_timesteps=100000,
        log_dir="logs",
        save_freq=10000,
        eval_env=val_env,
        eval_freq=50000
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_env = ForexTradingEnv(
        data=test_data,
        initial_balance=10000.0,
        lot_size=0.01,
        max_positions=5
    )
    
    results = agent.evaluate(test_env, n_episodes=10)
    print("\nEvaluation Results:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

