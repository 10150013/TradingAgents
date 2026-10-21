"""Deterministic supply and demand zone detection on daily bars.

A zone is a base (a few narrow-range candles) that price left with an impulsive
move: a rally out of the base is a demand zone, a drop out of it is a supply
zone. A zone is fresh until a later close trades through it. No LLM involved, so
the analyst can quote zone prices without inventing them.
"""

from __future__ import annotations

import pandas as pd
from stockstats import wrap

from tradingagents.dataflows.vendors.yahoo.snapshot import _fmt, _verified_rows

IMPULSE_ATR = 1.5   # leg-out move needed from the base, in ATRs
BASE_ATR = 0.8      # a base candle's range must be under this many ATRs
LEG_BARS = 3        # bars allowed for the leg-out
MAX_BASE_BARS = 3
MIN_RR = 2.0


def find_zones(df: pd.DataFrame, atr: pd.Series) -> list[dict]:
    """Return fresh zones, nearest to the latest close first."""
    high, low, close = df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy()
    n, zones, i = len(df), [], 1
    while i < n - 1:
        a = atr.iloc[i]
        if pd.isna(a) or a <= 0:
            i += 1
            continue
        end = i
        while end + 1 < n and end - i + 1 < MAX_BASE_BARS and high[end + 1] - low[end + 1] < BASE_ATR * a:
            end += 1
        if high[i] - low[i] >= BASE_ATR * a:
            i += 1
            continue
        top, bottom = high[i:end + 1].max(), low[i:end + 1].min()
        leg = slice(end + 1, min(end + 1 + LEG_BARS, n))
        kind = None
        if leg.start < n:
            if close[leg].max() - top >= IMPULSE_ATR * a:
                kind = "demand"
            elif bottom - close[leg].min() >= IMPULSE_ATR * a:
                kind = "supply"
        if kind:
            after = close[end + 1:]
            broken = (after < bottom).any() if kind == "demand" else (after > top).any()
            if not broken:
                zones.append({"type": kind, "bottom": float(bottom), "top": float(top),
                              "date": df["Date"].iloc[i], "atr": float(a)})
            i = end + LEG_BARS + 1
        else:
            i = end + 1
    last = close[-1]
    return sorted(zones, key=lambda z: abs(last - (z["top"] + z["bottom"]) / 2))


def _plan(zone: dict, zones: list[dict], last: float) -> str:
    """Stop beyond the zone, target at the next opposite zone, risk:reward check."""
    pad = 0.1 * zone["atr"]
    if zone["type"] == "demand":
        entry, stop = zone["top"], zone["bottom"] - pad
        opposite = [z for z in zones if z["type"] == "supply" and z["bottom"] > entry]
        target = min(z["bottom"] for z in opposite) if opposite else None
        side = "long"
    else:
        entry, stop = zone["bottom"], zone["top"] + pad
        opposite = [z for z in zones if z["type"] == "demand" and z["top"] < entry]
        target = max(z["top"] for z in opposite) if opposite else None
        side = "short"
    risk = abs(entry - stop)
    if target is None or risk <= 0:
        return f"{side}: stop {_fmt(stop)}, no opposite zone to target"
    rr = abs(target - entry) / risk
    verdict = "meets" if rr >= MIN_RR else "fails"
    return f"{side}: entry {_fmt(entry)}, stop {_fmt(stop)}, target {_fmt(target)}, R:R {rr:.1f} ({verdict} 1:{MIN_RR:g})"


def build_supply_demand_report(symbol: str, as_of_date: str, look_back_days: int = 250) -> str:
    df = _verified_rows(symbol, as_of_date).tail(max(60, int(look_back_days))).reset_index(drop=True)
    atr = wrap(df.copy())["atr"]
    last = float(df["Close"].iloc[-1])
    zones = find_zones(df, atr)

    lines = [
        f"## Supply and demand zones for {symbol.upper()} (daily bars)",
        "",
        f"- Analysis date: {as_of_date}; last close {_fmt(last)} on {_fmt(df['Date'].iloc[-1])}",
        "- Zones are fresh (no later close through them), base candles plus an impulsive leg-out.",
        "- Daily data only: the 15m/1h reversal-candle confirmation is not available here.",
        "",
    ]
    if not zones:
        return "\n".join(lines + ["No fresh zones found in this window."])

    lines += ["| Type | Zone bottom | Zone top | Formed | Distance from close | Trade plan |",
              "|---|---:|---:|---|---:|---|"]
    for z in zones[:6]:
        mid = (z["top"] + z["bottom"]) / 2
        lines.append(f"| {z['type']} | {_fmt(z['bottom'])} | {_fmt(z['top'])} | {_fmt(z['date'])} "
                     f"| {100 * (mid - last) / last:+.1f}% | {_plan(z, zones, last)} |")
    return "\n".join(lines)
