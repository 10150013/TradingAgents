"""A symbol's full daily history, for backtests that need every bar Yahoo has."""

import pandas as pd
import yfinance as yf


def load_full_history(symbol: str) -> pd.DataFrame:
    """Daily OHLCV with a Date column, oldest first."""
    df = yf.download(symbol, period="max", interval="1d", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna().reset_index().rename(columns={"index": "Date"})


def load_intraday(symbol: str, interval: str = "5m", period: str = "60d") -> pd.DataFrame:
    """Intraday OHLCV with a Date column in exchange wall-clock time, oldest first.

    Yahoo serves 5m and 15m bars for about 60 days and 1h bars for about 2 years."""
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna().reset_index().rename(columns={"index": "Date", "Datetime": "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df
