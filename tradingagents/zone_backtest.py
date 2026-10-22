"""Backtest the supply and demand zone strategy on daily bars.

Run: python -m tradingagents.zone_backtest MNQ=F   (add --compare for the filters, --grid for the settings, --optimize to tune win rate, --improve for extra filters, --intraday for 5-minute bars (last 60 days); several symbols allowed)

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


# Tuned on 5-minute bars (see run_intraday); costs are round-trip slippage plus commission
# as a fraction of price, roughly four ticks on Micro Nasdaq.
INTRADAY = dict(pad=0.5, target_r=1.0, window=500, max_hold=60, order_life=12)
INTRADAY_ZONES = dict(impulse=2.0, base=1)
COST_FRAC = 0.00004


def _plan(zone: dict, zones: list[dict], pad: float = 0.1, target_r: float | None = None):
    """(side, entry, stop, target) for a zone.

    The target is the next opposite zone, or target_r times the risk when set.
    None when there is no opposite zone to aim at."""
    gap = pad * zone["atr"]
    if zone["type"] == "demand":
        entry, stop = zone["top"], zone["bottom"] - gap
        if target_r:
            return "long", entry, stop, entry + target_r * (entry - stop)
        above = [z["bottom"] for z in zones if z["type"] == "supply" and z["bottom"] > entry]
        return ("long", entry, stop, min(above)) if above else None
    entry, stop = zone["bottom"], zone["top"] + gap
    if target_r:
        return "short", entry, stop, entry - target_r * (stop - entry)
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


def run(df: pd.DataFrame, trend: bool = False, confirm: bool = False, pad: float = 0.1,
        target_r: float | None = None, min_rr: float | None = None, cache: dict | None = None,
        rsi: tuple[float, float] | None = None, max_age: int | None = None,
        max_height: float | None = None, window: int = WINDOW, max_hold: int = MAX_HOLD,
        order_life: int = ORDER_LIFE, cost: float = 0.0, min_risk: float = 0.0,
        trend_up: pd.Series | None = None, sides: tuple[str, ...] = ("long", "short")) -> pd.DataFrame:
    """trend: longs only above the 200 SMA, shorts only below it.
    confirm: enter at the close of a bar that touches the zone and rejects it.
    pad: stop distance beyond the zone, in ATRs.  target_r: fixed R target instead of the
    next opposite zone.  min_rr: smallest reward-to-risk taken (zone targets only).
    cache: reuse each day's zones between runs on the same data and zone settings.
    rsi: (long_max, short_min) - longs only when RSI(14) is at or below the first value, shorts
    only at or above the second.  max_age: ignore zones formed more than this many calendar days
    ago.  max_height: ignore zones taller than this many ATRs.
    window / max_hold / order_life: bars of zone history, bars a trade may stay open, bars a
    resting order lives (the defaults suit daily bars).  cost: round-trip slippage plus
    commission in price points, taken off every trade.  min_risk: skip trades whose stop is
    closer than this many points, where costs would eat the target.
    trend_up: one True/False per bar (the market is in an uptrend); when given, longs are taken
    only in an uptrend and shorts only in a downtrend.  sides: which directions may trade."""
    min_rr = MIN_RR if min_rr is None else min_rr
    cache = {} if cache is None else cache
    df = df.reset_index(drop=True)
    atr = wrap(df.copy())["atr"]
    sma = df.Close.rolling(200).mean()
    rsi_s = wrap(df.copy())["rsi"]
    trades, used, t = [], set(), window
    while t < len(df) - 1:
        lo = t + 1 - window
        if t not in cache:
            cache[t] = find_zones(df.iloc[lo:t + 1].reset_index(drop=True), atr.iloc[lo:t + 1].reset_index(drop=True))
        zones = cache[t]
        last = df.Close.iloc[t]
        order = None
        for z in zones:
            key = (z["type"], z["date"])
            plan = _plan(z, zones, pad, target_r)
            if key in used or plan is None:
                continue
            side, entry, stop, target = plan
            if (side == "long") != (entry < last) or abs(entry - stop) <= max(min_risk, 1e-9):
                continue
            if side not in sides:
                continue
            if trend_up is not None and (pd.isna(trend_up.iloc[t]) or (side == "long") != bool(trend_up.iloc[t])):
                continue
            if trend and (pd.isna(sma.iloc[t]) or (side == "long") != (last > sma.iloc[t])):
                continue
            if not target_r and abs(target - entry) / abs(entry - stop) < min_rr:
                continue
            if rsi and not pd.isna(rsi_s.iloc[t]) and (rsi_s.iloc[t] > rsi[0] if side == "long" else rsi_s.iloc[t] < rsi[1]):
                continue
            if max_age and (df.Date.iloc[t] - z["date"]).days > max_age:
                continue
            if max_height and (z["top"] - z["bottom"]) > max_height * z["atr"]:
                continue
            order = (key, side, entry, stop, target)
            break
        if order is None:
            t += 1
            continue
        key, side, entry, stop, target = order
        fill = None
        for i in range(t + 1, min(t + 1 + order_life, len(df))):
            bar = df.iloc[i]
            touched = bar.Low <= entry if side == "long" else bar.High >= entry
            if touched and confirm:
                rejected = bar.Close > entry and bar.Close > bar.Open if side == "long" \
                    else bar.Close < entry and bar.Close < bar.Open
                if not rejected:
                    continue
                risk_c = abs(bar.Close - stop)
                if risk_c <= 0 or (not target_r and abs(target - bar.Close) / risk_c < min_rr):
                    continue
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
        first = fill
        if confirm:
            px, first = df.Close.iloc[fill], fill + 1
            if first >= len(df):
                break
        risk = abs(px - stop)
        out, end = None, first
        for i in range(first, min(first + max_hold, len(df))):
            hit = _exit(side, stop, target, df.iloc[i])
            if hit is not None:
                out, end = hit, i
                break
            end = i
        if out is None:
            out = df.Close.iloc[end]
        gross = out - px if side == "long" else px - out
        r = (gross - cost) / risk
        trades.append({"entered": df.Date.iloc[fill], "exited": df.Date.iloc[end], "side": side,
                       "entry": px, "stop": stop, "target": target, "exit": out, "R": r})
        t = end + 1
    return pd.DataFrame(trades)


def daily_trend(df: pd.DataFrame, daily: pd.DataFrame, length: int = 20) -> pd.Series:
    """For each bar of `df`: True when the previous day closed above its `length`-day average."""
    day = daily.set_index("Date").Close
    up = (day > day.rolling(length).mean()).shift(1)
    return pd.Series(up.reindex(df.Date.dt.normalize(), method="ffill").values, index=df.index)


def run_intraday(df: pd.DataFrame, daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """The strategy with the 5-minute settings and costs taken off.

    With `daily` (daily bars of the same market) it only buys in a daily uptrend and only
    sells in a daily downtrend."""
    from tradingagents.dataflows.vendors.yahoo import supply_demand as sd

    saved = (sd.IMPULSE_ATR, sd.MAX_BASE_BARS)
    sd.IMPULSE_ATR, sd.MAX_BASE_BARS = INTRADAY_ZONES["impulse"], INTRADAY_ZONES["base"]
    try:
        trend = daily_trend(df, daily) if daily is not None else None
        return run(df, cost=COST_FRAC * df.Close.median(), trend_up=trend, **INTRADAY)
    finally:
        sd.IMPULSE_ATR, sd.MAX_BASE_BARS = saved


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


def compare(df: pd.DataFrame, split: str = "2016-01-01") -> str:
    """Each variant over all years, the years before `split` and the years after it."""
    rows = ["variant           period     trades  win%   avgR  totalR"]
    for name, kw in (("baseline", {}), ("trend", {"trend": True}), ("confirm", {"confirm": True}),
                     ("trend+confirm", {"trend": True, "confirm": True})):
        trades = run(df, **kw)
        for label, g in (("all", trades), ("before", trades[trades.entered < split]),
                         ("after", trades[trades.entered >= split])):
            if g.empty:
                rows.append(f"{name:<17} {label:<10} {0:>6}")
                continue
            rows.append(f"{name:<17} {label:<10} {len(g):>6} {100 * (g.R > 0).mean():>5.1f} "
                        f"{g.R.mean():>+6.2f} {g.R.sum():>+7.1f}")
    return "\n".join(rows)


def grid(df: pd.DataFrame, split: str = "2016-01-01") -> str:
    """Try zone settings; a setting only counts if it earns in both halves."""
    from tradingagents.dataflows.vendors.yahoo import supply_demand as sd

    global MIN_RR
    saved = (sd.IMPULSE_ATR, sd.MAX_BASE_BARS, sd.MIN_RR, MIN_RR)
    rows = []
    try:
        for impulse in (1.5, 2.0, 3.0):
            for base in (1, 3, 5):
                for rr in (2.0, 3.0):
                    sd.IMPULSE_ATR, sd.MAX_BASE_BARS, sd.MIN_RR, MIN_RR = impulse, base, rr, rr
                    t = run(df)
                    a, b = t[t.entered < split], t[t.entered >= split]
                    avg = lambda g: g.R.mean() if len(g) else float("nan")
                    rows.append((impulse, base, rr, len(t), len(a), avg(a), len(b), avg(b)))
    finally:
        sd.IMPULSE_ATR, sd.MAX_BASE_BARS, sd.MIN_RR, MIN_RR = saved
    out = ["impulse base  rr  trades | before: n  avgR | after: n  avgR | both positive"]
    for r in sorted(rows, key=lambda r: -(r[5] if r[5] == r[5] else -9)):
        ok = "YES" if r[5] > 0 and r[7] > 0 and r[3] >= 30 else ""
        out.append(f"{r[0]:>7} {r[1]:>4} {r[2]:>4} {r[3]:>7} | {r[4]:>9} {r[5]:>+5.2f} | {r[6]:>8} {r[7]:>+5.2f} | {ok}")
    return "\n".join(out)


def optimize(dfs: dict[str, pd.DataFrame], split: str = "2016-01-01", min_train: int = 30) -> str:
    """Pick the setting with the highest win rate that still earns, using the years before
    `split` on all symbols pooled, then report it on the years after (not used to choose)."""
    from tradingagents.dataflows.vendors.yahoo import supply_demand as sd

    saved = (sd.IMPULSE_ATR, sd.MAX_BASE_BARS)
    rows = []
    try:
        for impulse in (1.5, 2.0):
            for base in (3, 5):
                sd.IMPULSE_ATR, sd.MAX_BASE_BARS = impulse, base
                caches = {k: {} for k in dfs}
                for trend in (False, True):
                    for pad in (0.1, 0.5):
                        for target_r, min_rr in [(None, 1.5), (None, 2.0), (None, 3.0), (1.0, None), (1.5, None), (2.0, None)]:
                            parts = [run(d, trend=trend, pad=pad, target_r=target_r, min_rr=min_rr, cache=caches[k])
                                     for k, d in dfs.items()]
                            t = pd.concat([x for x in parts if not x.empty], ignore_index=True) if any(
                                not x.empty for x in parts) else pd.DataFrame(columns=["entered", "R"])
                            a, b = t[t.entered < split], t[t.entered >= split]
                            name = f"impulse={impulse} base={base} trend={'on' if trend else 'off'} pad={pad} " + \
                                (f"target={target_r}R" if target_r else f"target=zone rr>={min_rr}")
                            rows.append((name, a, b))
    finally:
        sd.IMPULSE_ATR, sd.MAX_BASE_BARS = saved

    def stat(g):
        return (len(g), 100 * (g.R > 0).mean(), g.R.mean()) if len(g) else (0, float("nan"), float("nan"))

    ok = [r for r in rows if len(r[1]) >= min_train and r[1].R.mean() > 0]
    ok.sort(key=lambda r: (-stat(r[1])[1], -stat(r[1])[2]))
    out = [f"Chosen on years before {split}, pooled over {', '.join(dfs)}; 'after' was not used to choose.",
           f"Settings tried: {len(rows)}; eligible (>= {min_train} early trades and avg R > 0): {len(ok)}", "",
           "setting                                                   | early n win%  avgR | later n win%  avgR"]
    for name, a, b in ok[:8]:
        (n1, w1, r1), (n2, w2, r2) = stat(a), stat(b)
        out.append(f"{name:<57} | {n1:>7} {w1:>4.0f} {r1:>+5.2f} | {n2:>7} {w2:>4.0f} {r2:>+5.2f}")
    if not ok:
        out.append("No setting earned in the early years with enough trades.")
    return "\n".join(out)


def improve(tune: dict[str, pd.DataFrame], check: dict[str, pd.DataFrame], split: str = "2016-01-01",
            min_train: int = 25) -> str:
    """Add indicator and zone-quality filters to the best base setting. Chosen on the years before
    `split` on the `tune` markets; shown on their later years and on the `check` markets, which are
    never used to choose."""
    from tradingagents.dataflows.vendors.yahoo import supply_demand as sd

    saved = (sd.IMPULSE_ATR, sd.MAX_BASE_BARS)
    sd.IMPULSE_ATR, sd.MAX_BASE_BARS = 1.5, 3
    caches = {k: {} for k in {**tune, **check}}

    def pooled(dfs, **kw):
        parts = [run(d, cache=caches[k], **kw) for k, d in dfs.items()]
        parts = [x for x in parts if not x.empty]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["entered", "R"])

    def stat(g):
        return (len(g), 100 * (g.R > 0).mean(), g.R.mean()) if len(g) else (0, float("nan"), float("nan"))

    rows = []
    try:
        for target_r in (1.0, 1.5):
            for rsi in (None, (45, 55), (40, 60)):
                for max_age in (None, 90, 180):
                    for max_height in (None, 1.0, 2.0):
                        kw = dict(target_r=target_r, rsi=rsi, max_age=max_age, max_height=max_height)
                        t = pooled(tune, **kw)
                        c = pooled(check, **kw)
                        name = f"target={target_r}R rsi={rsi or 'off'} age<={max_age or 'any'} height<={max_height or 'any'}"
                        rows.append((name, t[t.entered < split], t[t.entered >= split], c, kw == dict(
                            target_r=1.0, rsi=None, max_age=None, max_height=None)))
    finally:
        sd.IMPULSE_ATR, sd.MAX_BASE_BARS = saved

    ok = [r for r in rows if len(r[1]) >= min_train and r[1].R.mean() > 0]
    ok.sort(key=lambda r: (-stat(r[1])[1], -stat(r[1])[2]))
    head = (f"Tuned on {', '.join(tune)} before {split}; 'later' = same markets after; "
            f"'other markets' = {', '.join(check)}, never used to choose.")
    cols = "setting                                                     | early n win% avgR | later n win% avgR | other n win% avgR"
    fmt = lambda r: " | ".join(f"{n:>5} {w:>4.0f} {a:>+5.2f}" for n, w, a in (stat(r[1]), stat(r[2]), stat(r[3])))
    out = [head, f"Settings tried: {len(rows)}; eligible: {len(ok)}", "", cols]
    base = [r for r in rows if r[4]][0]
    out.append(f"{'CURRENT BEST (no extra filters)':<59} | {fmt(base)}")
    for r in ok[:8]:
        out.append(f"{r[0]:<59} | {fmt(r)}")
    return "\n".join(out)


def load(symbol: str, interval: str = "1d") -> pd.DataFrame:
    from tradingagents.dataflows.vendors.yahoo.history import load_full_history, load_intraday

    if interval == "1d":
        return load_full_history(symbol)
    return load_intraday(symbol, interval, "730d" if interval.endswith("h") else "60d")


if __name__ == "__main__":
    symbols = [a for a in sys.argv[1:] if not a.startswith("--")] or ["MNQ=F"]
    intraday = next((a.split("=")[1] if "=" in a else "5m" for a in sys.argv if a.startswith("--intraday")), None)
    if intraday:
        for symbol in symbols:
            data = load(symbol, intraday)
            print(f"\n{symbol} {intraday}: {len(data)} bars, {data.Date.iloc[0]:%Y-%m-%d} to {data.Date.iloc[-1]:%Y-%m-%d}")
            daily = load(symbol)
            print("-- with the daily trend filter (buy only in an uptrend, sell only in a downtrend)")
            print(summarize(run_intraday(data, daily)))
            print("-- without it")
            print(summarize(run_intraday(data)))
        sys.exit()
    if "--improve" in sys.argv:
        tune, check = ("NQ=F", "ES=F"), ("YM=F", "RTY=F")
        print(improve({k: load(k) for k in tune}, {k: load(k) for k in check}))
        sys.exit()
    if "--optimize" in sys.argv:
        print(optimize({sym: load(sym) for sym in symbols}))
        sys.exit()
    for symbol in symbols:
        data = load(symbol)
        print(f"\n{symbol}: {len(data)} daily bars, {data.Date.iloc[0]:%Y-%m-%d} to {data.Date.iloc[-1]:%Y-%m-%d}")
        if "--grid" in sys.argv:
            print(grid(data))
        elif "--compare" in sys.argv:
            print(compare(data))
        else:
            result = run(data)
            print(summarize(result))
            result.to_csv("zone_backtest_trades.csv", index=False)
            print("Trade list saved to zone_backtest_trades.csv")
