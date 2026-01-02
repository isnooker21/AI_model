"""
Data Provider Module for XAUUSD Trading System

This module handles data fetching from MetaTrader5 and provides
a mock implementation for Mac development environments.

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

# Try to import ta-lib, fallback to manual calculations
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib not available. Using manual indicator calculations.")


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


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices: Price series (typically close prices)
        period: RSI period (default 14)
        
    Returns:
        RSI values as pandas Series
    """
    if TALIB_AVAILABLE:
        return pd.Series(talib.RSI(prices.values, timeperiod=period), index=prices.index)
    
    # Manual RSI calculation
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # Fill NaN with neutral value


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period (default 14)
        
    Returns:
        ATR values as pandas Series
    """
    if TALIB_AVAILABLE:
        return pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=period), 
                        index=high.index)
    
    # Manual ATR calculation
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr.bfill()


def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        prices: Price series (typically close prices)
        period: Moving average period (default 20)
        std_dev: Standard deviation multiplier (default 2.0)
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    if TALIB_AVAILABLE:
        upper, middle, lower = talib.BBANDS(
            prices.values,
            timeperiod=period,
            nbdevup=std_dev,
            nbdevdn=std_dev,
            matype=0
        )
        return (
            pd.Series(upper, index=prices.index),
            pd.Series(middle, index=prices.index),
            pd.Series(lower, index=prices.index)
        )
    
    # Manual Bollinger Bands calculation
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    return upper.bfill(), middle.bfill(), lower.bfill()


def get_session(hour: int) -> str:
    """
    Determine trading session based on hour (UTC).
    
    Args:
        hour: Hour in UTC (0-23)
        
    Returns:
        Session name: 'Asia', 'Europe', or 'US'
    """
    if 0 <= hour < 8:
        return 'Asia'
    elif 8 <= hour < 16:
        return 'Europe'
    else:
        return 'US'


def fetch_historical_data(
    symbol: str = "XAUUSD",
    timeframe: int = 15,  # M15
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    lookback_days: int = 365
) -> pd.DataFrame:
    """
    Fetch historical data from MetaTrader5 or generate mock data.
    
    Args:
        symbol: Trading symbol (default: XAUUSD)
        timeframe: Timeframe in minutes (default: 15 for M15)
        date_from: Start date (optional)
        date_to: End date (optional)
        lookback_days: Number of days to look back if date_from not provided
        
    Returns:
        DataFrame with OHLCV data and technical indicators
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
        
        # Calculate technical indicators
        print("Calculating technical indicators...")
        df['rsi'] = calculate_rsi(df['close'], period=14)
        df['atr'] = calculate_atr(df['high'], df['low'], df['close'], period=14)
        
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['close'], period=20, std_dev=2.0)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle  # Normalized BB width
        
        # Add session information
        df['session'] = df.index.hour.apply(get_session)
        df['session_asia'] = (df['session'] == 'Asia').astype(int)
        df['session_europe'] = (df['session'] == 'Europe').astype(int)
        df['session_us'] = (df['session'] == 'US').astype(int)
        
        # Drop session string column (keep only one-hot encoded)
        df.drop('session', axis=1, inplace=True)
        
        # Remove rows with NaN values (from indicator calculations)
        df = df.dropna()
        
        print(f"Successfully fetched {len(df)} bars of data")
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
    print("Testing data provider...")
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

