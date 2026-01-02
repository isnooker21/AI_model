# XAUUSD Autonomous AI Trading System

A Reinforcement Learning-based trading system for XAUUSD (Gold) using Proximal Policy Optimization (PPO) with recovery logic, dynamic exits, and MetaTrader5 integration.

## Features

- **Market Regime Prediction**: Identifies trend and sideways market conditions
- **Recovery Logic**: Implements scaling-in based on drawdown thresholds
- **Dynamic Exits**: Trailing stops and early exit on reversal signals
- **Cross-Platform**: Works on Mac (development) and Windows VPS (deployment)
- **MT5 Integration**: ONNX export for MetaTrader5 integration via MQL5

## Project Structure

```
AI_model/
├── data_provider.py      # Data fetching and feature engineering
├── trading_env.py        # Gymnasium trading environment
├── agent.py              # PPO agent implementation
├── export_onnx.py        # ONNX model export for MT5
├── main.py               # Main orchestration script
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## Installation

### Prerequisites

- Python 3.11 or higher
- MetaTrader5 (for Windows VPS deployment)

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd AI_model
```

2. Create a virtual environment:
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Mac/Linux
# or
venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: If `ta-lib` installation fails, you can:
- Install TA-Lib binary: `pip install TA-Lib-binary`
- Or download from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

## Usage

### Quick Start

Run the full pipeline (fetch data, train, evaluate, export):

```bash
python main.py --mode full
```

### Individual Modes

#### 1. Fetch Data Only

```bash
python main.py --mode fetch --lookback_days 365
```

#### 2. Train Model

```bash
python main.py --mode train --train_timesteps 200000
```

#### 3. Evaluate Model

```bash
python main.py --mode eval --model_path models/final_model.zip
```

#### 4. Export to ONNX

```bash
python main.py --mode export --model_path models/final_model.zip --onnx_output models/trading_model.onnx
```

### Advanced Options

```bash
python main.py \
    --mode full \
    --symbol XAUUSD \
    --lookback_days 365 \
    --train_timesteps 200000 \
    --initial_balance 10000.0 \
    --lot_size 0.01 \
    --max_positions 5 \
    --log_dir logs \
    --model_dir models
```

## Configuration

### Trading Environment Parameters

- `initial_balance`: Starting account balance (default: 10000.0)
- `lot_size`: Position size in lots (default: 0.01)
- `max_positions`: Maximum concurrent positions (default: 5)
- `max_drawdown_pct`: Maximum allowed drawdown (default: 0.20 = 20%)
- `recovery_threshold_pct`: Drawdown threshold to trigger recovery (default: 0.05 = 5%)

### Agent Parameters

- `learning_rate`: PPO learning rate (default: 3e-4)
- `n_steps`: Steps per update (default: 2048)
- `batch_size`: Training batch size (default: 64)
- `gamma`: Discount factor (default: 0.99)

## State Space

The environment uses a 19-dimensional state space:

1. **OHLC** (4): Open, High, Low, Close prices
2. **Technical Indicators** (6): RSI, ATR, Bollinger Bands (Upper, Middle, Lower, Width)
3. **Session Indicators** (3): Asia, Europe, US session flags
4. **Portfolio State** (6): Balance, Equity, Floating P/L, Position count, Average entry price, Drawdown %

## Action Space

6 discrete actions:

- **0**: Hold (no action)
- **1**: Initial Buy (open new buy position)
- **2**: Initial Sell (open new sell position)
- **3**: Recovery Buy (scale-in on existing buy positions)
- **4**: Recovery Sell (scale-in on existing sell positions)
- **5**: Close All (close all open positions)

## Recovery Logic

The system implements intelligent recovery (scaling-in) that:

- Triggers when drawdown exceeds `recovery_threshold_pct` (default: 5%)
- Calculates new weighted average entry price
- Prevents excessive drawdown by respecting `max_drawdown_pct`
- Limits maximum concurrent positions

## Reward Function

The reward function considers:

- **Realized Profit/Loss**: Rewards profitable trades, penalizes losses
- **Drawdown Penalty**: Penalizes high drawdowns
- **Time Penalty**: Small penalty per step to encourage active management
- **Position Management**: Rewards closing positions, penalizes excessive positions

## MetaTrader5 Integration

### ONNX Export

The trained model is exported to ONNX format compatible with MQL5's `OnnxRun` function:

```bash
python export_onnx.py \
    --model_path models/final_model.zip \
    --output_path models/trading_model.onnx \
    --create_mql5_example
```

This generates:
- `models/trading_model.onnx`: ONNX model file
- `mql5_integration_example.mq5`: Example MQL5 code

### MQL5 Integration

1. Copy the ONNX model to your MT5 `MQL5/Files/` directory
2. Use the generated `mql5_integration_example.mq5` as a starting point
3. Implement technical indicator calculations in MQL5
4. Implement portfolio state tracking
5. Deploy as an Expert Advisor (EA)

## Development on Mac

The system includes a **mock MetaTrader5 implementation** for Mac development:

- Automatically detects if MT5 is unavailable
- Generates realistic synthetic XAUUSD data
- Allows full development and testing without MT5

## Deployment to Windows VPS

1. Install Python 3.11+ on Windows VPS
2. Install MetaTrader5
3. Clone repository from GitHub
4. Install dependencies: `pip install -r requirements.txt`
5. Run training/evaluation with real MT5 data
6. Export ONNX model
7. Deploy to MT5

## File Structure

### Data Files

- `data/xauusd_m15.csv`: Historical OHLCV data with indicators

### Model Files

- `models/final_model.zip`: Trained SB3 PPO model
- `models/best_model.zip`: Best model during training
- `models/trading_model.onnx`: ONNX model for MT5

### Logs

- `logs/`: Training logs and TensorBoard data
- `logs/tensorboard/`: TensorBoard event files

## Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir logs/tensorboard
```

## Evaluation Metrics

The evaluation reports:

- Mean reward and standard deviation
- Final equity and balance
- Total trades executed
- Win rate
- Total return percentage

## Troubleshooting

### MT5 Connection Issues

- Ensure MT5 is installed and running
- Check account credentials
- Verify symbol name (XAUUSD)
- On Mac, the mock provider will be used automatically

### ONNX Export Issues

- Ensure PyTorch and ONNX are properly installed
- Check model file path is correct
- Verify input shape matches state space (19 dimensions)

### Training Issues

- Reduce `train_timesteps` for faster testing
- Adjust `batch_size` based on available memory
- Monitor TensorBoard for training progress

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions are welcome! Please ensure:

- Code follows PEP 8 style guidelines
- All functions are documented
- Tests are included for new features

## Support

For issues or questions, please open an issue on GitHub.

