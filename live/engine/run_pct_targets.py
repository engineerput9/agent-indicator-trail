#!/usr/bin/env python3
"""Percent-target variants of the cached Yahoo 5m backtest.

Reuses indicator math and the opening-scan / range-filter entry path from
backtest.py. Does not download. Stops are the swing/ATR level from f_levels,
clamped to 0.8%–2.0% of the fill. Targets are 1.5% or 3% of the fill.
No 1R lock. Session flatten falls back to the last in-session bar when the
15:25 bar is absent.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import backtest as bt

ROOT = Path("/workspace/backtest")
DATA = ROOT / "data"
INITIAL = bt.INITIAL
RISK_PCT = bt.RISK_PCT
TZ = bt.TZ
MIN_STOP_PCT = 0.008  # 0.8% of entry
MAX_STOP_PCT = 0.020  # 2.0% of entry


def adjusted_stop(is_long: bool, entry: float, atr_i: float, swing_low: float, swing_high: float):
    """Swing/ATR stop from f_levels, then clamp distance to [0.8%, 2.0%] of entry.

    Returns (sl, dist) or None if a positive raw stop cannot be formed.
    After a successful clamp the distance is inside the 0.8–2.0 band, which is
    the new stopOk test (the old 0.4–0.8 reject is not applied).
    """
    if not np.isfinite(entry) or entry <= 0:
        return None
    sl_raw, _tp1, _tp2, _risk = bt.f_levels(is_long, float(entry), float(atr_i), float(swing_low), float(swing_high))
    if not np.isfinite(sl_raw):
        return None
    dist = (entry - sl_raw) if is_long else (sl_raw - entry)
    if not np.isfinite(dist) or dist <= 0:
        return None
    min_d = entry * MIN_STOP_PCT
    max_d = entry * MAX_STOP_PCT
    if dist < min_d:
        dist = min_d
    elif dist > max_d:
        dist = max_d
    stop_pct = dist / entry
    # new stopOk band (replaces 0.4–0.8). Clamp makes this true when raw risk > 0.
    if stop_pct < MIN_STOP_PCT - 1e-9 or stop_pct > MAX_STOP_PCT + 1e-9:
        return None
    sl = (entry - dist) if is_long else (entry + dist)
    return float(sl), float(dist)


def levels_from_fill(is_long: bool, fill: float, atr_i: float, swing_low: float, swing_high: float, target_pct: float):
    """Stop and percent target from the fill price. None if stopOk fails."""
    adj = adjusted_stop(is_long, fill, atr_i, swing_low, swing_high)
    if adj is None:
        return None
    sl, dist = adj
    if is_long:
        tp = fill * (1.0 + target_pct)
        price_ok = (fill > sl) and (fill < tp)
    else:
        tp = fill * (1.0 - target_pct)
        price_ok = (fill < sl) and (fill > tp)
    if not price_ok or not np.isfinite(tp):
        return None
    return float(sl), float(tp), float(dist)


def load_symbol(symbol: str) -> pd.DataFrame | None:
    return bt.load_cache(symbol)


def precompute(df: pd.DataFrame) -> dict:
    df = df.sort_index()
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    volume = df["volume"].to_numpy(dtype=np.float64)
    times = df.index
    n = len(df)
    hours = np.asarray(times.hour)
    minutes = np.asarray(times.minute)
    day_ids = np.asarray(times.strftime("%Y-%m-%d"))

    in_sess = ((hours > 9) | ((hours == 9) & (minutes >= 15))) & (
        (hours < 15) | ((hours == 15) & (minutes <= 30))
    )
    by_cut = (hours < 9) | ((hours == 9) & (minutes <= 20))
    legacy_flat = (hours == 15) & (minutes >= 25)

    ind = bt.compute_indicators(high, low, close, volume)
    atr = ind["atr"]
    smrng = ind["smrng"]
    swing_low = ind["swing_low"]
    swing_high = ind["swing_high"]
    bull_core = ind["bull_core"]
    bear_core = ind["bear_core"]

    long_flip = np.zeros(n, dtype=bool)
    short_flip = np.zeros(n, dtype=bool)
    scan_side = np.zeros(n, dtype=np.int8)

    cond_ini = 0
    upward = 0
    downward = 0
    rf_prev = np.nan
    scan_checked = False
    cur_scan = 0
    cum_pv = 0.0
    cum_v = 0.0
    vwap = np.nan

    for i in range(n):
        new_day = i == 0 or day_ids[i] != day_ids[i - 1]
        if new_day:
            scan_checked = False
            cur_scan = 0
            cum_pv = 0.0
            cum_v = 0.0
            vwap = np.nan

        vol_i = volume[i]
        if in_sess[i] and np.isfinite(vol_i) and vol_i > 0:
            typical = (high[i] + low[i] + close[i]) / 3.0
            cum_pv += typical * vol_i
            cum_v += vol_i
            vwap = cum_pv / cum_v if cum_v > 0 else np.nan

        c = close[i]
        sm = smrng[i]
        if not np.isfinite(c) or not np.isfinite(sm):
            rf_i = np.nan
        else:
            prev_rf_nz = rf_prev if np.isfinite(rf_prev) else 0.0
            if c > prev_rf_nz:
                rf_i = prev_rf_nz if (c - sm < prev_rf_nz) else (c - sm)
            else:
                rf_i = prev_rf_nz if (c + sm > prev_rf_nz) else (c + sm)

        rf_ok = np.isfinite(rf_i) and np.isfinite(rf_prev)
        if rf_ok:
            if rf_i > rf_prev:
                upward += 1
                downward = 0
            elif rf_i < rf_prev:
                downward += 1
                upward = 0

        long_cond = rf_ok and c > rf_i and upward > 0
        short_cond = rf_ok and c < rf_i and downward > 0
        if long_cond and cond_ini == -1:
            long_flip[i] = True
        elif short_cond and cond_ini == 1:
            short_flip[i] = True
        if long_cond:
            cond_ini = 1
        elif short_cond:
            cond_ini = -1
        rf_prev = rf_i

        if in_sess[i] and np.isfinite(vol_i) and vol_i > 0 and not scan_checked:
            scan_checked = True
            if bull_core[i] and np.isfinite(vwap) and c > vwap:
                cur_scan = 1
            elif bear_core[i] and np.isfinite(vwap) and c < vwap:
                cur_scan = -1
            else:
                cur_scan = 0
        scan_side[i] = cur_scan

    last_sess = np.full(n, -1, dtype=np.int32)
    day_has_legacy = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and day_ids[j + 1] == day_ids[i]:
            j += 1
        last = -1
        has = False
        for k in range(i, j + 1):
            if in_sess[k]:
                last = k
            if legacy_flat[k]:
                has = True
        for k in range(i, j + 1):
            last_sess[k] = last
            day_has_legacy[k] = has
        i = j + 1

    return {
        "high": high,
        "low": low,
        "close": close,
        "times": times,
        "n": n,
        "in_sess": in_sess,
        "by_cut": by_cut,
        "legacy_flat": legacy_flat,
        "day_ids": day_ids,
        "atr": atr,
        "swing_low": swing_low,
        "swing_high": swing_high,
        "long_flip": long_flip,
        "short_flip": short_flip,
        "scan_side": scan_side,
        "last_sess": last_sess,
        "day_has_legacy": day_has_legacy,
    }


def simulate(symbol: str, pre: dict, target_pct: float, use_fallback: bool) -> dict:
    high = pre["high"]
    low = pre["low"]
    close = pre["close"]
    times = pre["times"]
    n = pre["n"]
    in_sess = pre["in_sess"]
    by_cut = pre["by_cut"]
    legacy_flat = pre["legacy_flat"]
    day_ids = pre["day_ids"]
    atr = pre["atr"]
    swing_low = pre["swing_low"]
    swing_high = pre["swing_high"]
    long_flip = pre["long_flip"]
    short_flip = pre["short_flip"]
    scan_side = pre["scan_side"]
    last_sess = pre["last_sess"]
    day_has_legacy = pre["day_has_legacy"]

    cash = INITIAL
    pos = 0
    entry_price = 0.0
    entry_i = -1
    qty = 0.0
    risk_pts = 0.0
    equity_at_entry = INITIAL
    act_sl = np.nan
    act_tp = np.nan
    signal_i = -1

    took = False
    printed = False
    pend_side = 0

    trades = []
    open_at_end = 0

    def close_trade(i, exit_price, reason):
        nonlocal cash, pos, qty
        if pos == 1:
            pnl = (exit_price - entry_price) * qty
            r_mult = (exit_price - entry_price) / risk_pts if risk_pts > 0 else np.nan
            side = "long"
            price_chg = (exit_price - entry_price) / entry_price * 100.0
        else:
            pnl = (entry_price - exit_price) * qty
            r_mult = (entry_price - exit_price) / risk_pts if risk_pts > 0 else np.nan
            side = "short"
            price_chg = (entry_price - exit_price) / entry_price * 100.0
        cash += pnl
        trades.append(
            {
                "symbol": symbol,
                "side": side,
                "entry_time": times[entry_i].isoformat(),
                "exit_time": times[i].isoformat(),
                "entry": float(entry_price),
                "exit": float(exit_price),
                "sl": float(act_sl),
                "tp": float(act_tp),
                "price_chg_pct": float(price_chg),
                "r_multiple": float(r_mult),
                "pnl": float(pnl),
                "exit_reason": reason,
                "fallback_flatten": bool(reason == "session" and not legacy_flat[i]),
            }
        )
        pos = 0

    for i in range(n):
        new_day = i == 0 or day_ids[i] != day_ids[i - 1]
        if new_day:
            printed = False
            pend_side = 0
            took = False

        c = close[i]

        # Exits after the entry bar. Stop wins if stop and target share a bar.
        if pos != 0 and i > entry_i:
            if pos == 1:
                stop_hit = low[i] <= act_sl
                tp_hit = high[i] >= act_tp
            else:
                stop_hit = high[i] >= act_sl
                tp_hit = low[i] <= act_tp
            if stop_hit:
                close_trade(i, float(act_sl), "stop")
            elif tp_hit:
                close_trade(i, float(act_tp), "target")
            else:
                do_flat = bool(legacy_flat[i]) or (
                    use_fallback and (not day_has_legacy[i]) and last_sess[i] == i
                )
                if do_flat:
                    close_trade(i, float(c), "session")

        # First in-session range-filter flip. stopOk is the clamped 0.8–2% band
        # on this signal close (old 0.4–0.8 filter is not used).
        if in_sess[i] and (not printed) and (long_flip[i] or short_flip[i]):
            printed = True
            is_long = bool(long_flip[i])
            if by_cut[i] and levels_from_fill(
                is_long, float(c), float(atr[i]), float(swing_low[i]), float(swing_high[i]), target_pct
            ) is not None:
                pend_side = 1 if is_long else -1
                signal_i = i

        # Entry at this bar's close. Levels recomputed from the fill (this close),
        # not the signal close. priceOk: fill strictly between that stop and target.
        if (
            pos == 0
            and in_sess[i]
            and by_cut[i]
            and pend_side != 0
            and (not took)
            and np.isfinite(c)
        ):
            is_long = pend_side == 1
            lv = levels_from_fill(
                is_long, float(c), float(atr[i]), float(swing_low[i]), float(swing_high[i]), target_pct
            )
            scan_ok = int(scan_side[i]) == pend_side
            if scan_ok and lv is not None:
                sl, tp, dist = lv
                q = (cash * (RISK_PCT / 100.0)) / dist if dist > 0 else 0.0
                if q > 0 and np.isfinite(q):
                    pos = pend_side
                    entry_price = float(c)
                    entry_i = i
                    qty = float(q)
                    risk_pts = float(dist)
                    equity_at_entry = float(cash)
                    act_sl = float(sl)
                    act_tp = float(tp)
                    took = True
                    if signal_i < 0:
                        signal_i = i

    if pos != 0:
        open_at_end = 1
    return {"trades": trades, "open_at_end": open_at_end}


def summarize(trades: list[dict]) -> dict:
    if not trades:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "breakeven": 0,
            "win_rate": None,
            "profit_factor": None,
            "average_r": None,
            "average_price_chg_pct": None,
            "median_price_chg_pct": None,
            "total_net_pnl": 0.0,
            "symbols_traded": 0,
            "symbols_positive": 0,
            "exit_reason_counts": {},
            "long_vs_short": {
                "long": {"trades": 0, "net_pnl": 0.0},
                "short": {"trades": 0, "net_pnl": 0.0},
            },
        }
    pnls = np.array([t["pnl"] for t in trades], dtype=np.float64)
    rs = np.array([t["r_multiple"] for t in trades], dtype=np.float64)
    chg = np.array([t["price_chg_pct"] for t in trades], dtype=np.float64)
    wins = int(np.sum(pnls > 0))
    losses = int(np.sum(pnls < 0))
    be = int(np.sum(pnls == 0))
    gp = float(pnls[pnls > 0].sum()) if wins else 0.0
    gl = float(-pnls[pnls < 0].sum()) if losses else 0.0
    pf = (gp / gl) if gl > 0 else None
    by_sym = {}
    for t in trades:
        by_sym[t["symbol"]] = by_sym.get(t["symbol"], 0.0) + t["pnl"]
    exits = Counter(t["exit_reason"] for t in trades)
    long_pnl = float(sum(t["pnl"] for t in trades if t["side"] == "long"))
    short_pnl = float(sum(t["pnl"] for t in trades if t["side"] == "short"))
    long_n = int(sum(1 for t in trades if t["side"] == "long"))
    short_n = int(sum(1 for t in trades if t["side"] == "short"))
    return {
        "trades": int(len(trades)),
        "wins": wins,
        "losses": losses,
        "breakeven": be,
        "win_rate": wins / len(trades),
        "profit_factor": pf,
        "average_r": float(np.nanmean(rs)),
        "average_price_chg_pct": float(np.nanmean(chg)),
        "median_price_chg_pct": float(np.nanmedian(chg)),
        "total_net_pnl": float(pnls.sum()),
        "symbols_traded": int(len(by_sym)),
        "symbols_positive": int(sum(1 for v in by_sym.values() if v > 0)),
        "exit_reason_counts": dict(exits),
        "long_vs_short": {
            "long": {"trades": long_n, "net_pnl": long_pnl},
            "short": {"trades": short_n, "net_pnl": short_pnl},
        },
        "gross_profit": gp,
        "gross_loss": gl,
        "_by_sym": by_sym,
    }


def rank_symbols(by_sym: dict, k: int = 5):
    items = sorted(by_sym.items(), key=lambda kv: (-kv[1], kv[0]))
    top = [{"symbol": s, "net_pnl": float(v)} for s, v in items[:k]]
    bot_items = sorted(by_sym.items(), key=lambda kv: (kv[1], kv[0]))
    bottom = [{"symbol": s, "net_pnl": float(v)} for s, v in bot_items[:k]]
    return top, bottom


def diff_vs_legacy(official: list[dict], legacy: list[dict]) -> dict:
    def key(t):
        return (t["symbol"], t["side"], t["entry_time"])

    def ex(t):
        return (t["exit_time"], t["exit_reason"], round(float(t["exit"]), 6), round(float(t["pnl"]), 4))

    om = {key(t): ex(t) for t in official}
    lm = {key(t): ex(t) for t in legacy}
    same_entry_diff_exit = 0
    only_fix = 0
    only_legacy = 0
    for k in set(om) | set(lm):
        if k not in om:
            only_legacy += 1
        elif k not in lm:
            only_fix += 1
        elif om[k] != lm[k]:
            same_entry_diff_exit += 1
    direct = int(sum(1 for t in official if t.get("fallback_flatten")))
    return {
        "fallback_session_exits": direct,
        "same_entry_different_exit": same_entry_diff_exit,
        "entries_only_with_fallback_flatten": only_fix,
        "entries_only_without_fallback_flatten": only_legacy,
        "trade_records_changed": same_entry_diff_exit + only_fix + only_legacy,
        "official_trades": len(official),
        "legacy_trades": len(legacy),
    }


def symbol_list() -> list[str]:
    syms = ["^NSEI", "^NSEBANK"]
    ns = sorted(p.name[: -len(".csv")] for p in DATA.glob("*.NS.csv"))
    # cache names are the yahoo tickers for equities
    return syms + ns


def main() -> None:
    bt._self_check()
    symbols = symbol_list()
    missing = [s for s in symbols if not bt.cache_path(s).exists()]
    if missing:
        raise SystemExit(f"missing cache files: {missing}")

    variants = [
        ("1.5pct", 0.015),
        ("3pct", 0.030),
    ]
    official = {name: [] for name, _ in variants}
    legacy = {name: [] for name, _ in variants}
    opens = {name: [] for name, _ in variants}
    loaded = 0

    for si, sym in enumerate(symbols):
        df = load_symbol(sym)
        if df is None:
            print(f"SKIP unloadable {sym}", flush=True)
            continue
        loaded += 1
        pre = precompute(df)
        for name, tpct in variants:
            off = simulate(sym, pre, tpct, use_fallback=True)
            leg = simulate(sym, pre, tpct, use_fallback=False)
            official[name].extend(off["trades"])
            legacy[name].extend(leg["trades"])
            if off["open_at_end"]:
                opens[name].append(sym)
        if (si + 1) % 25 == 0:
            print(
                f"done {si+1}/{len(symbols)} trades_1.5={len(official['1.5pct'])} trades_3={len(official['3pct'])}",
                flush=True,
            )

    # validate stop band and targets on official trades
    for name, tpct in variants:
        for t in official[name]:
            dist = abs(t["entry"] - t["sl"])
            pct = dist / t["entry"]
            if pct < MIN_STOP_PCT - 1e-6 or pct > MAX_STOP_PCT + 1e-6:
                raise SystemExit(f"stop band fail {name} {t}")
            if t["side"] == "long":
                expect = t["entry"] * (1.0 + tpct)
            else:
                expect = t["entry"] * (1.0 - tpct)
            if abs(expect - t["tp"]) > 1e-6 * max(1.0, abs(t["entry"])):
                raise SystemExit(f"tp fail {name} {t} expect {expect}")

    rows = []
    summary = {
        "costs": {"commission": 0.0, "slippage": 0.0},
        "capital": {
            "initial_per_symbol": INITIAL,
            "risk": "1% of cash equity at entry (starts at 100000, compounds with closed pnl) / stop distance",
            "qty": "fractional",
            "accounts": "independent per symbol",
        },
        "rules": {
            "stop": "f_levels swing/ATR, widen to 0.8% of fill if tighter, cap at 2.0% of fill",
            "no_1r_lock": True,
            "old_0.4_0.8_filter": "dropped; stopOk is the clamped 0.8–2.0% band on the signal close, levels recomputed from the fill",
            "target": "percent of fill close",
            "intrabar": "skip entry bar; if stop and target on same bar, stop fills (adverse first)",
            "session": "flatten at first bar >= 15:25 if that bar exists; else flatten at last in-session bar (time <= 15:30)",
            "symbols_loaded": loaded,
            "symbols_requested": len(symbols),
        },
        "variants": {},
        "session_flatten_vs_not_flattening": {},
    }

    frames = []
    for name, tpct in variants:
        trades = official[name]
        st = summarize(trades)
        by_sym = st.pop("_by_sym")
        top, bottom = rank_symbols(by_sym)
        st["top5_by_net_pnl"] = top
        st["bottom5_by_net_pnl"] = bottom
        st["open_positions_at_end"] = opens[name]
        diff = diff_vs_legacy(trades, legacy[name])
        st["session_flatten_impact"] = diff
        summary["variants"][name] = st
        summary["session_flatten_vs_not_flattening"][name] = diff
        for t in trades:
            frames.append(
                {
                    "variant": name,
                    "symbol": t["symbol"],
                    "side": t["side"],
                    "entry_time": t["entry_time"],
                    "exit_time": t["exit_time"],
                    "entry": t["entry"],
                    "exit": t["exit"],
                    "sl": t["sl"],
                    "tp": t["tp"],
                    "price_chg_pct": t["price_chg_pct"],
                    "r_multiple": t["r_multiple"],
                    "pnl": t["pnl"],
                    "exit_reason": t["exit_reason"],
                }
            )

    cols = [
        "variant",
        "symbol",
        "side",
        "entry_time",
        "exit_time",
        "entry",
        "exit",
        "sl",
        "tp",
        "price_chg_pct",
        "r_multiple",
        "pnl",
        "exit_reason",
    ]
    tdf = pd.DataFrame(frames, columns=cols)
    if not tdf.empty:
        tdf = tdf.sort_values(["variant", "symbol", "entry_time", "exit_time"]).reset_index(drop=True)
    tdf.to_csv(ROOT / "trades_pct_targets.csv", index=False)

    class Enc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.floating,)):
                if not np.isfinite(o):
                    return None
                return float(o)
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, float):
                if not math.isfinite(o):
                    return None
                return o
            return super().default(o)

    with open(ROOT / "results_pct_targets.json", "w") as f:
        json.dump(summary, f, indent=2, cls=Enc)

    print(json.dumps(summary, indent=2, cls=Enc), flush=True)


if __name__ == "__main__":
    main()
