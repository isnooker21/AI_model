"""
Data Provider Module for XAUUSD Trading System (Candlestick-Only)

This module handles data fetching from MetaTrader5 and provides
candlestick-only data without any lagging indicators.

Author: AI Trading System
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple
import warnings

# Import MetaTrader5 - required for data fetching
try:
    import MetaTrader5 as mt5
except ImportError:
    raise ImportError(
        "MetaTrader5 library is required but not installed.\n"
        "Please install it with: pip install MetaTrader5\n"
        "Note: MetaTrader5 is only available on Windows.\n"
        "For development on Mac/Linux, you need to run this on Windows VPS."
    )


def calculate_candlestick_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate candlestick features: Body Size, Upper Wick, Lower Wick, Price Change.
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        DataFrame with additional candlestick features
    """
    df = df.copy()
    
    # Body size (absolute value)
    df['body_size'] = abs(df['close'] - df['open'])
    
    # Upper wick (high - max(open, close))
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    
    # Lower wick (min(open, close) - low)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    
    # Price change relative to previous candle (close - previous close)
    df['price_change'] = df['close'].diff()
    
    # Normalize price change by previous close (percentage change)
    df['price_change_pct'] = df['price_change'] / df['close'].shift(1)
    df['price_change_pct'] = df['price_change_pct'].fillna(0)
    
    # Candle direction (1 for bullish, -1 for bearish, 0 for doji)
    df['candle_direction'] = np.where(
        df['close'] > df['open'], 1,
        np.where(df['close'] < df['open'], -1, 0)
    )
    
    # Body to range ratio (how much of the range is body)
    df['range'] = df['high'] - df['low']
    df['body_to_range'] = df['body_size'] / (df['range'] + 1e-8)  # Avoid division by zero
    
    return df


def get_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features for market session context.
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        DataFrame with time features
    """
    df = df.copy()
    
    # Hour of day (0-23)
    df['hour'] = df.index.hour
    
    # Minute of hour (0-59)
    df['minute'] = df.index.minute
    
    # Day of week (0=Monday, 6=Sunday)
    df['day_of_week'] = df.index.dayofweek
    
    # Market session (normalized to 0-1 for each session)
    # Asia: 0-8 UTC, Europe: 8-16 UTC, US: 16-24 UTC
    df['session_asia'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(float)
    df['session_europe'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(float)
    df['session_us'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(float)
    
    # Normalize hour to 0-1 (circular encoding)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    return df


def fetch_historical_data(
    symbol: str = "XAUUSDc",
    timeframe: int = 5,  # M5
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    lookback_days: int = 730  # 2 years
) -> pd.DataFrame:
    """
    Fetch historical data from MetaTrader5.
    Returns only raw candlestick data with calculated features.
    
    Args:
        symbol: Trading symbol (default: XAUUSDc for Cent account)
        timeframe: Timeframe in minutes (default: 5 for M5)
        date_from: Start date (optional)
        date_to: End date (optional)
        lookback_days: Number of days to look back if date_from not provided (default: 730 = 2 years)
        
    Returns:
        DataFrame with OHLCV data and candlestick features
    """
    # Initialize MT5 connection
    if not mt5.initialize():
        raise ConnectionError(
            "Failed to initialize MetaTrader5.\n"
            "Please ensure:\n"
            "1. MetaTrader5 is installed and running\n"
            "2. You are logged into your trading account\n"
            "3. The symbol is available in Market Watch"
        )
    
    try:
        # Set default dates if not provided
        if date_to is None:
            date_to = datetime.now()
        if date_from is None:
            date_from = date_to - timedelta(days=lookback_days)
        
        # Convert timeframe to MT5 constant
        if timeframe == 5:
            mt5_timeframe = mt5.TIMEFRAME_M5
        elif timeframe == 15:
            mt5_timeframe = mt5.TIMEFRAME_M15
        elif timeframe == 60:
            mt5_timeframe = mt5.TIMEFRAME_H1
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        # Try to fetch data from MT5
        # Try primary symbol first
        rates = mt5.copy_rates_range(symbol, mt5_timeframe, date_from, date_to)
        
        # If failed, try alternative symbols (common variations)
        if rates is None or len(rates) == 0:
            alternative_symbols = ['XAUUSD', 'GOLD', 'XAUUSD.c', 'XAUUSD#']
            for alt_symbol in alternative_symbols:
                if alt_symbol != symbol:
                    print(f"Trying alternative symbol: {alt_symbol}")
                    rates = mt5.copy_rates_range(alt_symbol, mt5_timeframe, date_from, date_to)
                    if rates is not None and len(rates) > 0:
                        print(f"Successfully fetched data using symbol: {alt_symbol}")
                        symbol = alt_symbol  # Update symbol for consistency
                        break
        
        # If still no data, raise error with helpful message
        if rates is None or len(rates) == 0:
            error_msg = f"No data retrieved for {symbol}\n\n"
            error_msg += "Possible solutions:\n"
            error_msg += "1. Check if the symbol name is correct in your MT5 terminal\n"
            error_msg += "2. Common symbol names: XAUUSD, XAUUSDc, GOLD, XAUUSD.c\n"
            error_msg += "3. Make sure the symbol is available in Market Watch\n"
            error_msg += "4. Try using --symbol argument with the correct symbol name\n"
            error_msg += "5. Ensure you have historical data for the requested date range"
            raise ValueError(error_msg)
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Rename columns for consistency
        df.columns = ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
        
        # Calculate candlestick features
        print("Calculating candlestick features...")
        df = calculate_candlestick_features(df)
        
        # Add time features
        print("Adding time-based features...")
        df = get_time_features(df)
        
        # Remove rows with NaN values (from feature calculations)
        df = df.dropna()
        
        print(f"Successfully fetched {len(df)} bars of candlestick data")
        print(f"Features: {df.columns.tolist()}")
        
        return df
        
    except Exception as e:
        error_msg = str(e)
        raise RuntimeError(f"Error fetching data: {error_msg}")
    
    finally:
        mt5.shutdown()


def save_data_to_csv(df: pd.DataFrame, filepath: str = "data/xauusdc_m5.csv") -> None:
    """
    Save DataFrame to CSV file.
    
    Args:
        df: DataFrame to save
        filepath: Path to save the CSV file
    """
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    df.to_csv(filepath)
    print(f"Data saved to {filepath}")


def load_data_from_csv(filepath: str = "data/xauusdc_m5.csv") -> pd.DataFrame:
    """
    Load DataFrame from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        Loaded DataFrame with datetime index
    """
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"Data loaded from {filepath}: {len(df)} bars")
    return df


if __name__ == "__main__":
    # Test data fetching
    print("Testing candlestick-only data provider...")
    try:
        df = fetch_historical_data(symbol="XAUUSDc", timeframe=5, lookback_days=730)
        print(f"\nData shape: {df.shape}")
        print(f"\nColumns: {df.columns.tolist()}")
        print(f"\nFirst few rows:\n{df.head()}")
        print(f"\nLast few rows:\n{df.tail()}")
        
        # Save to CSV
        save_data_to_csv(df, "data/xauusdc_m5.csv")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
