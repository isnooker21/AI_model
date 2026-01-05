"""
Test Model Script for XAUUSD Trading System

This script tests a trained model by running it through a full simulation
using the same data and environment settings as training.

Author: AI Trading System
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Import required modules
from data_provider import load_data_from_csv, fetch_historical_data
from trading_env import ForexTradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Default paths
DEFAULT_MODEL_PATHS = [
    "./logs/best_eval_model.zip",
    "./logs/best_model.zip",
    "./logs/best_eval_model/best_model.zip"
]


def find_model(model_path: str = None) -> str:
    """
    Find the model file to load.
    
    Args:
        model_path: Optional explicit path to model
        
    Returns:
        Path to model file
        
    Raises:
        FileNotFoundError: If no model file is found
    """
    if model_path:
        if os.path.exists(model_path):
            return model_path
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Try default paths
    for path in DEFAULT_MODEL_PATHS:
        if os.path.exists(path):
            print(f"Found model: {path}")
            return path
    
    raise FileNotFoundError(
        f"No model file found. Tried:\n" + "\n".join(f"  - {p}" for p in DEFAULT_MODEL_PATHS)
    )


def test_model(
    model_path: str = None,
    data_file: str = "data/xauusdc_m5.csv",
    sequence_length: int = 50,
    initial_balance: float = 10000.0,
    lot_size: float = 0.1,
    max_positions: int = 100,
    architecture: str = "lstm",
    deterministic: bool = True
) -> dict:
    """
    Test a trained model by running it through a simulation.
    
    Args:
        model_path: Path to model file (auto-detect if None)
        data_file: Path to data file
        sequence_length: Number of candles in sequence
        initial_balance: Initial account balance
        lot_size: Position size in lots
        max_positions: Maximum positions (not enforced, for compatibility)
        architecture: Model architecture ('lstm' or 'transformer')
        deterministic: Whether to use deterministic predictions
        
    Returns:
        Dictionary with test results
    """
    print("="*60)
    print("MODEL TESTING")
    print("="*60)
    
    # Find and load model
    model_file = find_model(model_path)
    print(f"\nLoading model from: {model_file}")
    
    # Load data
    print(f"\nLoading data from: {data_file}")
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"Data file not found: {data_file}\n"
            "Please run data fetching first: python main.py --mode fetch"
        )
    
    data = load_data_from_csv(data_file)
    print(f"Loaded {len(data)} bars of data")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    
    # Create environment with same settings as training
    print("\nCreating test environment...")
    test_env = ForexTradingEnv(
        data=data,
        sequence_length=sequence_length,
        initial_balance=initial_balance,
        lot_size=lot_size,
        max_positions=max_positions,
        max_drawdown_pct=0.95,
        recovery_threshold_pct=0.0,
        min_grid_distance=0.0
    )
    
    # Wrap environment for SB3 (same as training)
    vec_env = DummyVecEnv([lambda: Monitor(test_env)])
    
    # Load model
    print(f"\nLoading PPO model...")
    try:
        model = PPO.load(model_file, env=vec_env, verbose=1)
        print("Model loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    
    # Reset environment
    obs = vec_env.reset()
    
    # Run simulation
    print(f"\nRunning simulation (deterministic={deterministic})...")
    print("-" * 60)
    
    total_reward = 0.0
    step_count = 0
    done = [False]
    
    while not done[0]:
        # Predict action (deterministic mode)
        action, _ = model.predict(obs, deterministic=deterministic)
        
        # Execute action
        obs, reward, done, info = vec_env.step(action)
        
        total_reward += float(reward[0])
        step_count += 1
        
        # Print progress every 1000 steps
        if step_count % 1000 == 0:
            # Get info from environment directly
            env_info = test_env._get_info()
            current_equity = env_info.get('equity', initial_balance)
            print(f"Step {step_count:,}: Equity=${current_equity:,.2f}, Reward={total_reward:.2f}")
    
    print("-" * 60)
    
    # Get final state from environment
    final_info = test_env._get_info()
    
    # Extract results
    results = {
        'initial_balance': initial_balance,
        'final_balance': final_info.get('balance', initial_balance),
        'final_equity': final_info.get('equity', initial_balance),
        'total_pnl': final_info.get('balance', initial_balance) - initial_balance,
        'realized_pnl': final_info.get('realized_pnl', 0.0),
        'floating_pnl': final_info.get('floating_pnl', 0.0),
        'total_trades': final_info.get('total_trades', 0),
        'winning_trades': final_info.get('winning_trades', 0),
        'losing_trades': final_info.get('losing_trades', 0),
        'total_reward': total_reward,
        'total_steps': step_count,
        'total_commission': final_info.get('total_commission', 0.0),
        'drawdown_pct': final_info.get('drawdown_pct', 0.0)
    }
    
    return results


def print_summary(results: dict) -> None:
    """Print test results summary."""
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    initial = results['initial_balance']
    final = results['final_balance']
    pnl = results['total_pnl']
    pnl_pct = (pnl / initial * 100) if initial > 0 else 0.0
    
    print(f"\nInitial Balance:      ${initial:,.2f}")
    print(f"Final Balance:        ${final:,.2f}")
    print(f"Total P&L:            ${pnl:,.2f} ({pnl_pct:+.2f}%)")
    print(f"  Realized P&L:       ${results['realized_pnl']:,.2f}")
    print(f"  Floating P&L:       ${results['floating_pnl']:,.2f}")
    
    print(f"\nTotal Trades:         {results['total_trades']}")
    print(f"  Winning Trades:     {results['winning_trades']}")
    print(f"  Losing Trades:      {results['losing_trades']}")
    
    if results['total_trades'] > 0:
        win_rate = (results['winning_trades'] / results['total_trades']) * 100
        print(f"  Win Rate:           {win_rate:.2f}%")
    
    print(f"\nTotal Reward:         {results['total_reward']:.2f}")
    print(f"Total Steps:          {results['total_steps']:,}")
    print(f"Total Commission:     ${results['total_commission']:,.2f}")
    print(f"Max Drawdown:         {results['drawdown_pct']*100:.2f}%")
    
    print("\n" + "="*60)


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test trained XAUUSD trading model"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model file (auto-detect if not specified)"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/xauusdc_m5.csv",
        help="Path to data file"
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=50,
        help="Number of candles in sequence"
    )
    parser.add_argument(
        "--initial_balance",
        type=float,
        default=10000.0,
        help="Initial account balance"
    )
    parser.add_argument(
        "--lot_size",
        type=float,
        default=0.1,
        help="Position size in lots"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic predictions (default: True)"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic predictions (overrides --deterministic)"
    )
    
    args = parser.parse_args()
    
    # Determine deterministic mode
    deterministic = not args.stochastic if args.stochastic else args.deterministic
    
    try:
        # Run test
        results = test_model(
            model_path=args.model_path,
            data_file=args.data_file,
            sequence_length=args.sequence_length,
            initial_balance=args.initial_balance,
            lot_size=args.lot_size,
            deterministic=deterministic
        )
        
        # Print summary
        print_summary(results)
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

