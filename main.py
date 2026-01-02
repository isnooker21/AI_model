"""
Main Orchestration Script for XAUUSD Trading System (Candlestick-Only)

This script coordinates data fetching, model training, evaluation, and ONNX export.
Uses raw candlestick data (Price Action) without lagging indicators.

Author: AI Trading System
"""

import os
import argparse
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from data_provider import (
    fetch_historical_data,
    save_data_to_csv,
    load_data_from_csv
)
from trading_env import ForexTradingEnv
from agent import TradingAgent, split_data_for_training

# Check if TensorBoard is available (optional)
try:
    import tensorboard
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# Import ONNX functions (optional - only needed for export)
try:
    from export_onnx import export_ppo_to_onnx, verify_onnx_model
    ONNX_AVAILABLE = True
except ImportError as e:
    ONNX_AVAILABLE = False
    print(f"Warning: ONNX export not available: {e}")
    print("Export functionality will be disabled. Install with: pip install onnx onnxruntime")
    
    # Create dummy functions to prevent errors
    def export_ppo_to_onnx(*args, **kwargs):
        raise RuntimeError("ONNX export not available. Please install: pip install onnx onnxruntime")
    
    def verify_onnx_model(*args, **kwargs):
        return False


def fetch_and_prepare_data(
    symbol: str = "XAUUSDc",
    timeframe: int = 5,  # M5
    lookback_days: int = 730,  # 2 years
    data_file: str = "data/xauusdc_m5.csv",
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Fetch and prepare data for training.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe in minutes
        lookback_days: Number of days to fetch
        data_file: Path to save/load data
        force_refresh: Whether to force data refresh
        
    Returns:
        Prepared DataFrame
    """
    # Check if data file exists and force_refresh is False
    if os.path.exists(data_file) and not force_refresh:
        print(f"Loading existing data from {data_file}...")
        return load_data_from_csv(data_file)
    
    # Fetch new data
    print(f"Fetching {lookback_days} days of {symbol} M{timeframe} data...")
    try:
        df = fetch_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            lookback_days=lookback_days
        )
        
        # Save to CSV
        save_data_to_csv(df, data_file)
        
        return df
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        if os.path.exists(data_file):
            print(f"Falling back to existing data file: {data_file}")
            return load_data_from_csv(data_file)
        else:
            raise


def train_model(
    data: pd.DataFrame,
    train_timesteps: int = 100000,
    initial_balance: float = 10000.0,
    lot_size: float = 0.1,  # Increased for more visible impact
    max_positions: int = 5,
    sequence_length: int = 50,
    architecture: str = "lstm",  # "lstm" or "transformer"
    log_dir: str = "logs",
    model_dir: str = "models",
    save_freq: int = 10000,
    eval_freq: int = 50000
) -> TradingAgent:
    """
    Train the trading agent.
    
    Args:
        data: Training data
        train_timesteps: Number of training timesteps
        initial_balance: Initial account balance
        lot_size: Position size in lots
        max_positions: Maximum concurrent positions
        log_dir: Directory for logs
        model_dir: Directory for saved models
        save_freq: Frequency of model saves
        eval_freq: Frequency of evaluation
        
    Returns:
        Trained agent
    """
    print("\n" + "="*60)
    print("TRAINING PHASE")
    print("="*60)
    
    # Split data
    train_data, val_data, test_data = split_data_for_training(
        data,
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    # Create environments
    print("\nCreating training environment...")
    train_env = ForexTradingEnv(
        data=train_data,
        sequence_length=sequence_length,
        initial_balance=initial_balance,
        lot_size=lot_size,
        max_positions=max_positions,
        max_drawdown_pct=0.20,
        recovery_threshold_pct=0.005,  # 0.5% aggressive recovery
        min_grid_distance=2.0  # 2.0 USD minimum grid distance
    )
    
    print("Creating validation environment...")
    val_env = ForexTradingEnv(
        data=val_data,
        sequence_length=sequence_length,
        initial_balance=initial_balance,
        lot_size=lot_size,
        max_positions=max_positions,
        max_drawdown_pct=0.20,
        recovery_threshold_pct=0.005,  # 0.5% aggressive recovery
        min_grid_distance=2.0  # 2.0 USD minimum grid distance
    )
    
    # Create agent
    print(f"\nInitializing PPO agent with {architecture.upper()} architecture...")
    
    # Set tensorboard_log only if available
    tensorboard_log_path = None
    if TENSORBOARD_AVAILABLE:
        tensorboard_log_path = os.path.join(log_dir, "tensorboard")
    else:
        print("Note: TensorBoard not available. Training logs will not be saved.")
        print("To enable: pip install tensorboard")
    
    agent = TradingAgent(
        env=train_env,
        architecture=architecture,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log=tensorboard_log_path,
        verbose=1
    )
    
    # Train agent
    print(f"\nStarting training for {train_timesteps} timesteps...")
    agent.train(
        total_timesteps=train_timesteps,
        log_dir=log_dir,
        save_freq=save_freq,
        eval_env=val_env,
        eval_freq=eval_freq
    )
    
    # Save final model
    os.makedirs(model_dir, exist_ok=True)
    final_model_path = os.path.join(model_dir, "final_model.zip")
    agent.save(final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    
    return agent, test_data


def evaluate_model(
    agent: TradingAgent,
    test_data: pd.DataFrame,
    sequence_length: int = 50,
    initial_balance: float = 10000.0,
    lot_size: float = 0.1,  # Increased for more visible impact
    max_positions: int = 5,
    n_episodes: int = 10
) -> dict:
    """
    Evaluate the trained model on test data.
    
    Args:
        agent: Trained agent
        test_data: Test dataset
        initial_balance: Initial account balance
        lot_size: Position size in lots
        max_positions: Maximum concurrent positions
        n_episodes: Number of evaluation episodes
        
    Returns:
        Evaluation results dictionary
    """
    print("\n" + "="*60)
    print("EVALUATION PHASE")
    print("="*60)
    
    # Create test environment
    print("\nCreating test environment...")
    test_env = ForexTradingEnv(
        data=test_data,
        sequence_length=sequence_length,
        initial_balance=initial_balance,
        lot_size=lot_size,
        max_positions=max_positions,
        max_drawdown_pct=0.20,
        recovery_threshold_pct=0.005,  # 0.5% aggressive recovery
        min_grid_distance=2.0  # 2.0 USD minimum grid distance
    )
    
    # Evaluate
    print(f"\nEvaluating on {n_episodes} episodes...")
    results = agent.evaluate(
        eval_env=test_env,
        n_episodes=n_episodes,
        deterministic=True
    )
    
    # Print results
    print("\n" + "-"*60)
    print("EVALUATION RESULTS")
    print("-"*60)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:10.4f}")
        else:
            print(f"{key:30s}: {value}")
    print("-"*60)
    
    return results


def export_to_onnx(
    model_path: str,
    output_path: str = "models/trading_model.onnx",
    input_shape: tuple = (362,),  # 50 candles * 7 features + 6 time + 6 portfolio
    create_mql5_example: bool = True
) -> str:
    """
    Export trained model to ONNX format.
    
    Args:
        model_path: Path to saved SB3 model
        output_path: Path to save ONNX model
        input_shape: Input shape for the model
        create_mql5_example: Whether to create MQL5 integration example
        
    Returns:
        Path to exported ONNX model
    """
    print("\n" + "="*60)
    print("ONNX EXPORT PHASE")
    print("="*60)
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Export model
    print(f"\nExporting model to ONNX...")
    print(f"  Input model: {model_path}")
    print(f"  Output model: {output_path}")
    print(f"  Input shape: {input_shape}")
    
    export_ppo_to_onnx(
        model_path=model_path,
        output_path=output_path,
        input_shape=input_shape,
        input_name="observations",
        output_name="actions",
        verbose=True
    )
    
    # Create MQL5 example if requested
    if create_mql5_example:
        print("\nCreating MQL5 integration example...")
        from export_onnx import create_mql5_integration_example
        create_mql5_integration_example(
            onnx_model_path=output_path,
            output_file="mql5_integration_example.mq5"
        )
    
    return output_path


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="XAUUSD Autonomous Trading System - Training and Export"
    )
    
    # Data arguments
    parser.add_argument(
        "--mode",
        type=str,
        choices=["fetch", "train", "eval", "export", "full"],
        default="full",
        help="Execution mode"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="XAUUSDc",
        help="Trading symbol (default: XAUUSDc for Cent account)"
    )
    parser.add_argument(
        "--timeframe",
        type=int,
        default=5,
        help="Timeframe in minutes (default: 5 for M5)"
    )
    parser.add_argument(
        "--lookback_days",
        type=int,
        default=730,
        help="Number of days of historical data (default: 730 = 2 years)"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/xauusdc_m5.csv",
        help="Path to data file"
    )
    parser.add_argument(
        "--force_refresh",
        action="store_true",
        help="Force data refresh even if file exists"
    )
    
    # Training arguments
    parser.add_argument(
        "--train_timesteps",
        type=int,
        default=100000,
        help="Number of training timesteps"
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
        help="Position size in lots (default: 0.1 for more visible impact)"
    )
    parser.add_argument(
        "--max_positions",
        type=int,
        default=5,
        help="Maximum concurrent positions"
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=50,
        help="Number of candles in sequence (default: 50)"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["lstm", "transformer"],
        default="lstm",
        help="Neural network architecture: 'lstm' or 'transformer'"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/final_model.zip",
        help="Path to saved model (for eval/export)"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory for logs"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Directory for saved models"
    )
    
    # Export arguments
    parser.add_argument(
        "--onnx_output",
        type=str,
        default="models/trading_model.onnx",
        help="Path to save ONNX model"
    )
    parser.add_argument(
        "--no_mql5_example",
        action="store_true",
        help="Don't create MQL5 integration example"
    )
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.data_file) if os.path.dirname(args.data_file) else '.', exist_ok=True)
    
    print("="*60)
    print("XAUUSD AUTONOMOUS TRADING SYSTEM (CANDLESTICK-ONLY)")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Symbol: {args.symbol}")
    print(f"Architecture: {args.architecture.upper()}")
    print(f"Sequence Length: {args.sequence_length} candles")
    print(f"Initial Balance: ${args.initial_balance:,.2f}")
    print(f"Lot Size: {args.lot_size}")
    print(f"Max Positions: {args.max_positions}")
    print("="*60)
    
    # Execute based on mode
    if args.mode in ["fetch", "full"]:
        data = fetch_and_prepare_data(
            symbol=args.symbol,
            lookback_days=args.lookback_days,
            data_file=args.data_file,
            force_refresh=args.force_refresh
        )
        print(f"\nData loaded: {len(data)} bars")
    
    if args.mode in ["train", "full"]:
        if args.mode == "train":
            data = load_data_from_csv(args.data_file)
        
        agent, test_data = train_model(
            data=data,
            train_timesteps=args.train_timesteps,
            initial_balance=args.initial_balance,
            lot_size=args.lot_size,
            max_positions=args.max_positions,
            sequence_length=args.sequence_length,
            architecture=args.architecture,
            log_dir=args.log_dir,
            model_dir=args.model_dir
        )
    
    if args.mode in ["eval", "full"]:
        if args.mode == "eval":
            # Load data and model
            data = load_data_from_csv(args.data_file)
            _, _, test_data = split_data_for_training(data)
            
            # Create test environment
            test_env = ForexTradingEnv(
                data=test_data,
                sequence_length=args.sequence_length if hasattr(args, 'sequence_length') else 50,
                initial_balance=args.initial_balance,
                lot_size=args.lot_size,
                max_positions=args.max_positions
            )
            
            # Load agent
            agent = TradingAgent(env=test_env, model_path=args.model_path)
        else:
            # Use agent and test_data from training
            pass
        
        results = evaluate_model(
            agent=agent,
            test_data=test_data,
            sequence_length=args.sequence_length,
            initial_balance=args.initial_balance,
            lot_size=args.lot_size,
            max_positions=args.max_positions
        )
    
    if args.mode in ["export", "full"]:
        if not ONNX_AVAILABLE:
            print("\n" + "="*60)
            print("WARNING: ONNX export is not available")
            print("="*60)
            print("To enable ONNX export, please install:")
            print("  pip install onnx onnxruntime")
            print("\nIf you encounter DLL errors, try:")
            print("  pip uninstall onnx onnxruntime")
            print("  pip install onnxruntime")
            print("="*60)
        else:
            if args.mode == "export":
                model_path = args.model_path
            else:
                model_path = os.path.join(args.model_dir, "final_model.zip")
            
            try:
                onnx_path = export_to_onnx(
                    model_path=model_path,
                    output_path=args.onnx_output,
                    input_shape=(362,),  # 50 candles * 7 features + 6 time + 6 portfolio
                    create_mql5_example=not args.no_mql5_example
                )
                
                print(f"\n✓ ONNX model ready for MT5 integration: {onnx_path}")
            except Exception as e:
                print(f"\n✗ ONNX export failed: {e}")
                print("You can still use the model for training and evaluation.")
    
    print("\n" + "="*60)
    print("PROCESS COMPLETED SUCCESSFULLY")
    print("="*60)


if __name__ == "__main__":
    main()

