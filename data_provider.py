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
    lookback_days: int = 730,  # 2 years (target, will fetch as much as available)
    batch_days: int = 90  # Fetch in batches of 90 days
) -> pd.DataFrame:
    """
    Fetch historical data from MetaTrader5 incrementally.
    Fetches data in batches going backwards from latest date.
    Returns only raw candlestick data with calculated features.
    
    Args:
        symbol: Trading symbol (default: XAUUSDc for Cent account)
        timeframe: Timeframe in minutes (default: 5 for M5)
        date_from: Start date (optional)
        date_to: End date (optional)
        lookback_days: Target number of days to look back (default: 730 = 2 years)
        batch_days: Number of days per batch (default: 90 days)
        
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
        
        # Find working symbol first (try primary and alternatives)
        working_symbol = None
        test_end = date_to
        test_start = test_end - timedelta(days=7)  # Test with 7 days
        
        # Try primary symbol first
        test_rates = mt5.copy_rates_range(symbol, mt5_timeframe, test_start, test_end)
        if test_rates is not None and len(test_rates) > 0:
            working_symbol = symbol
            print(f"Symbol {symbol} is available")
        else:
            # Try alternative symbols
            alternative_symbols = ['XAUUSD', 'GOLD', 'XAUUSD.c', 'XAUUSD#']
            for alt_symbol in alternative_symbols:
                if alt_symbol != symbol:
                    print(f"Trying alternative symbol: {alt_symbol}")
                    test_rates = mt5.copy_rates_range(alt_symbol, mt5_timeframe, test_start, test_end)
                    if test_rates is not None and len(test_rates) > 0:
                        working_symbol = alt_symbol
                        print(f"Successfully found working symbol: {alt_symbol}")
                        break
        
        if working_symbol is None:
            error_msg = f"No data available for {symbol} or alternatives\n\n"
            error_msg += "Possible solutions:\n"
            error_msg += "1. Check if the symbol name is correct in your MT5 terminal\n"
            error_msg += "2. Common symbol names: XAUUSD, XAUUSDc, GOLD, XAUUSD.c\n"
            error_msg += "3. Make sure the symbol is available in Market Watch\n"
            error_msg += "4. Try using --symbol argument with the correct symbol name"
            raise ValueError(error_msg)
        
        # Fetch data incrementally in batches (going backwards from latest)
        print(f"\nFetching data incrementally for {working_symbol}...")
        print(f"Target: {lookback_days} days, Fetching in batches of {batch_days} days")
        
        all_rates = []
        current_end = date_to
        total_fetched_days = 0
        batch_num = 1
        
        while current_end > date_from:
            # Calculate batch start date
            batch_start = current_end - timedelta(days=batch_days)
            if batch_start < date_from:
                batch_start = date_from
            
            # Fetch batch
            print(f"Batch {batch_num}: Fetching from {batch_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}...", end=' ')
            batch_rates = mt5.copy_rates_range(working_symbol, mt5_timeframe, batch_start, current_end)
            
            if batch_rates is not None and len(batch_rates) > 0:
                all_rates.append(batch_rates)
                batch_days_fetched = (current_end - batch_start).days
                total_fetched_days += batch_days_fetched
                print(f"✓ Got {len(batch_rates)} bars ({batch_days_fetched} days)")
                
                # Move to next batch (go backwards)
                current_end = batch_start
                batch_num += 1
            else:
                # No more data available, stop
                print(f"✗ No more data available")
                break
        
        # Combine all batches
        if len(all_rates) == 0:
            raise ValueError(f"No data retrieved for {working_symbol}")
        
        print(f"\nCombining {len(all_rates)} batches...")
        rates = np.concatenate(all_rates)
        
        # Remove duplicates (in case of overlap) and sort by time
        df_temp = pd.DataFrame(rates)
        df_temp['time'] = pd.to_datetime(df_temp['time'], unit='s')
        df_temp = df_temp.drop_duplicates(subset=['time'])
        df_temp = df_temp.sort_values('time')
        
        print(f"Total: {len(df_temp)} bars covering {total_fetched_days} days")
        print(f"Date range: {df_temp['time'].min()} to {df_temp['time'].max()}")
        
        # Convert to DataFrame (already processed above)
        df = df_temp.copy()
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
