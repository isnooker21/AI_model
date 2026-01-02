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

# Try to import MetaTrader5, fallback to mock if not available
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("Warning: MetaTrader5 not available. Using mock data provider for development.")


class MockMT5:
    """
    Mock MetaTrader5 implementation for Mac development.
    Generates realistic synthetic XAUUSD M15 data.
    """
    
    @staticmethod
    def initialize(path: Optional[str] = None) -> bool:
        """Mock initialization - always returns True."""
        return True
    
    @staticmethod
    def shutdown() -> None:
        """Mock shutdown - no-op."""
        pass
    
    @staticmethod
    def copy_rates_range(
        symbol: str,
        timeframe: int,
        date_from: datetime,
        date_to: datetime
    ) -> Optional[np.ndarray]:
        """
        Generate synthetic OHLCV data for XAUUSD.
        
        Args:
            symbol: Trading symbol (e.g., 'XAUUSD')
            timeframe: MT5 timeframe constant
            date_from: Start date
            date_to: End date
            
        Returns:
            numpy array with OHLCV data
        """
        # Calculate number of 15-minute bars
        time_diff = date_to - date_from
        total_minutes = int(time_diff.total_seconds() / 60)
        num_bars = total_minutes // 15
        
        if num_bars <= 0:
            return None
        
        # Generate realistic gold price data (starting around 2000)
        np.random.seed(42)  # For reproducibility
        base_price = 2000.0
        prices = []
        current_price = base_price
        
        for i in range(num_bars):
            # Random walk with slight upward bias
            change = np.random.normal(0, 0.5)  # ~$0.50 volatility per 15min
            current_price += change
            
            # Generate OHLC
            high = current_price + abs(np.random.normal(0, 0.3))
            low = current_price - abs(np.random.normal(0, 0.3))
            open_price = current_price + np.random.normal(0, 0.2)
            close_price = current_price
            
            # Ensure OHLC consistency
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)
            
            prices.append([
                int(date_from.timestamp()) + i * 15 * 60,  # time
                open_price,   # open
                high,        # high
                low,         # low
                close_price, # close
                1000,        # tick_volume (mock)
                0,           # spread
                0            # real_volume
            ])
        
        return np.array(prices, dtype=np.float64)
    
    @staticmethod
    def symbol_info(symbol: str):
        """Mock symbol info."""
        class SymbolInfo:
            point = 0.01
            digits = 2
            contract_size = 100.0
            trade_mode = 4  # TRADE_MODE_FULL
        
        return SymbolInfo()


# Use mock if MT5 is not available
if not MT5_AVAILABLE:
    mt5 = MockMT5()


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
    symbol: str = "XAUUSD",
    timeframe: int = 15,  # M15
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    lookback_days: int = 365
) -> pd.DataFrame:
    """
    Fetch historical data from MetaTrader5 or generate mock data.
    Returns only raw candlestick data with calculated features.
    
    Args:
        symbol: Trading symbol (default: XAUUSD)
        timeframe: Timeframe in minutes (default: 15 for M15)
        date_from: Start date (optional)
        date_to: End date (optional)
        lookback_days: Number of days to look back if date_from not provided
        
    Returns:
        DataFrame with OHLCV data and candlestick features
    """
    # Initialize MT5 connection
    if not mt5.initialize():
        if MT5_AVAILABLE:
            raise ConnectionError("Failed to initialize MetaTrader5")
        else:
            print("Using mock data provider for development.")
    
    try:
        # Set default dates if not provided
        if date_to is None:
            date_to = datetime.now()
        if date_from is None:
            date_from = date_to - timedelta(days=lookback_days)
        
        # Convert timeframe to MT5 constant
        if timeframe == 15:
            mt5_timeframe = mt5.TIMEFRAME_M15 if MT5_AVAILABLE else 15
        elif timeframe == 60:
            mt5_timeframe = mt5.TIMEFRAME_H1 if MT5_AVAILABLE else 60
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        # Fetch data
        rates = mt5.copy_rates_range(symbol, mt5_timeframe, date_from, date_to)
        
        if rates is None or len(rates) == 0:
            raise ValueError(f"No data retrieved for {symbol}")
        
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
        raise RuntimeError(f"Error fetching data: {str(e)}")
    
    finally:
        mt5.shutdown()


def save_data_to_csv(df: pd.DataFrame, filepath: str = "data/xauusd_m15.csv") -> None:
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


def load_data_from_csv(filepath: str = "data/xauusd_m15.csv") -> pd.DataFrame:
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
        df = fetch_historical_data(symbol="XAUUSD", lookback_days=30)
        print(f"\nData shape: {df.shape}")
        print(f"\nColumns: {df.columns.tolist()}")
        print(f"\nFirst few rows:\n{df.head()}")
        print(f"\nLast few rows:\n{df.tail()}")
        
        # Save to CSV
        save_data_to_csv(df, "data/xauusd_m15.csv")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
