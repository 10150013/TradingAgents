"""Backtest the supply and demand zone strategy on daily bars.

Run: python -m tradingagents.zone_backtest MNQ=F

Each day the zones are rebuilt from bars up to that close only. A resting limit
order sits at the nearest fresh zone's edge, with the stop beyond the zone and the
target at the next opposite zone; trades under MIN_RR are skipped. One trade at a
time. If a bar hits both stop and target, the stop counts first. Results are in R
(1R = the distance from entry to stop), so they do not depend on position size.
Daily bars only: there is no lower-time-frame reversal-candle confirmation.
"""

from __future__ import annotations

import sys

import pandas as pd
from stockstats import wrap

from tradingagents.dataflows.vendors.yahoo.supply_demand import MIN_RR, find_zones

WINDOW = 250      # bars of history each day's zones are built from
MAX_HOLD = 30     # a trade still open after this many bars exits at the close
ORDER_LIFE = 5    # a resting order is cancelled if it has not filled by then


def _plan(zone: dict, zones: list[dict]):
    """(side, entry, stop, target) for a zone, or None without an opposite zone."""
    pad = 0.1 * zone["atr"]
    if zone["type"] == "demand":
        entry, stop = zone["top"], zone["bottom"] - pad
        above = [z["bottom"] for z in zones if z["type"] == "supply" and z["bottom"] > entry]
        return ("long", entry, stop, min(above)) if above else None
    entry, stop = zone["bottom"], zone["top"] + pad
    below = [z["top"] for z in zones if z["type"] == "demand" and z["top"] < entry]
    return ("short", entry, stop, max(below)) if below else None


def _exit(side, stop, target, bar):
    """Exit price if this bar hits the stop or target (stop first), else None."""
    if side == "long":
        if bar.Low <= stop:
            return stop
        if bar.High >= target:
            return target
    else:
        if bar.High >= stop:
            return stop
        if bar.Low <= target:
            return target
    return None


def run(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    atr = wrap(df.copy())["atr"]
    trades, used, t = [], set(), WINDOW
    while t < len(df) - 1:
        lo = t + 1 - WINDOW
        zones = find_zones(df.iloc[lo:t + 1].reset_index(drop=True), atr.iloc[lo:t + 1].reset_index(drop=True))
        last = df.Close.iloc[t]
        order = None
        for z in zones:
            key = (z["type"], z["date"])
            plan = _plan(z, zones)
            if key in used or plan is None:
                continue
            side, entry, stop, target = plan
            if (side == "long") != (entry < last) or abs(entry - stop) <= 0:
                continue
            if abs(target - entry) / abs(entry - stop) < MIN_RR:
                continue
            order = (key, side, entry, stop, target)
            break
        if order is None:
            t += 1
            continue
        key, side, entry, stop, target = order
        fill = None
        for i in range(t + 1, min(t + 1 + ORDER_LIFE, len(df))):
            bar = df.iloc[i]
            touched = bar.Low <= entry if side == "long" else bar.High >= entry
            if touched:
                fill = i
                break
        used.add(key)
        if fill is None:
            t += 1
            continue
        px = entry
        if side == "long" and df.Open.iloc[fill] <= entry:
            px = df.Open.iloc[fill]
        if side == "short" and df.Open.iloc[fill] >= entry:
            px = df.Open.iloc[fill]
        risk = abs(px - stop)
        out, end = None, fill
        for i in range(fill, min(fill + MAX_HOLD, len(df))):
            hit = _exit(side, stop, target, df.iloc[i])
            if hit is not None:
                out, end = hit, i
                break
            end = i
        if out is None:
            out = df.Close.iloc[end]
        r = (out - px) / risk if side == "long" else (px - out) / risk
        trades.append({"entered": df.Date.iloc[fill], "exited": df.Date.iloc[end], "side": side,
                       "entry": px, "stop": stop, "target": target, "exit": out, "R": r})
        t = end + 1
    return pd.DataFrame(trades)


def summarize(trades: pd.DataFrame) -> str:
    if trades.empty:
        return "No trades."
    wins = trades.R > 0
    eq = trades.R.cumsum()
    lines = [
        f"Trades: {len(trades)}  ({trades.entered.min():%Y-%m-%d} to {trades.exited.max():%Y-%m-%d})",
        f"Win rate: {100 * wins.mean():.1f}%",
        f"Average R per trade: {trades.R.mean():+.2f}   Total: {trades.R.sum():+.1f}R",
        f"Average win: {trades.R[wins].mean():+.2f}R   Average loss: {trades.R[~wins].mean():+.2f}R",
        f"Max drawdown: {(eq.cummax() - eq).max():.1f}R",
    ]
    for side, g in trades.groupby("side"):
        lines.append(f"  {side}: {len(g)} trades, win rate {100 * (g.R > 0).mean():.1f}%, avg {g.R.mean():+.2f}R")
    return "\n".join(lines)


def load(symbol: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(symbol, period="max", interval="1d", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna().reset_index().rename(columns={"index": "Date"})


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "MNQ=F"
    data = load(symbol)
    print(f"{symbol}: {len(data)} daily bars, {data.Date.iloc[0]:%Y-%m-%d} to {data.Date.iloc[-1]:%Y-%m-%d}")
    result = run(data)
    print(summarize(result))
    result.to_csv("zone_backtest_trades.csv", index=False)
    print("Trade list saved to zone_backtest_trades.csv")
