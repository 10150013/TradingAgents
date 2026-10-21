"""A symbol's full daily history, for backtests that need every bar Yahoo has."""

import pandas as pd
import yfinance as yf


def load_full_history(symbol: str) -> pd.DataFrame:
    """Daily OHLCV with a Date column, oldest first."""
    df = yf.download(symbol, period="max", interval="1d", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna().reset_index().rename(columns={"index": "Date"})
