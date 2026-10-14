#!/usr/bin/env python3
"""Pine v6 range-filter + opening scan backtest on Yahoo Finance 5-minute bars.

Universe source (retrieved this run; official NSE fo_mktlots.csv and
nsearchives returned HTTP 403):
  https://techfanetechnologies.github.io/XTS_MasterInstruments/csv/NSEFO.csv
  HTTP Last-Modified: Fri, 02 Oct 2026 20:01:20 GMT
Equity underlyings = unique Name on FUTSTK and OPTSTK rows whose contract
expiration is on or after 2026-10-04, excluding *NSETEST symbols.
Yahoo ticker = SYMBOL.NS. Indices: ^NSEI, ^NSEBANK.

Costs: zero commission, zero slippage.
Each symbol is an independent account starting at 100000.
"""

from __future__ import annotations

import json
import math
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__import__('os').environ.get('BT_ROOT', '/tmp/bt_engine')); ROOT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

TZ = "Asia/Kolkata"
INITIAL = 100_000.0
RISK_PCT = 1.0
PER = 100
MULT = 3.0
SWING_LEN = 10
ATR_LEN = 14
ATR_MULT = 1.0
MIN_STOP = 0.4
MAX_STOP = 0.8
DI_LEN = 14
ADX_SMOOTH = 14

INDICES = ["^NSEI", "^NSEBANK"]
UNIVERSE_URL = "https://techfanetechnologies.github.io/XTS_MasterInstruments/csv/NSEFO.csv"
UNIVERSE_LAST_MODIFIED = "Fri, 02 Oct 2026 20:01:20 GMT"
ASOF = "2026-10-04"


def tv_smooth(src: np.ndarray, length: int, alpha: float) -> np.ndarray:
    """TradingView EMA/RMA: SMA seed of the first `length` finite window, then recursive.

    Matches pine_rma: na(prev) ? sma(src, length) : alpha*src + (1-alpha)*prev.
    A non-finite source value yields na and forces a reseed.
    """
    n = len(src)
    out = np.full(n, np.nan, dtype=np.float64)
    prev = np.nan
    dq: deque[float] = deque()
    run = 0.0
    for i in range(n):
        x = src[i]
        if np.isfinite(prev):
            if np.isfinite(x):
                prev = alpha * x + (1.0 - alpha) * prev
                out[i] = prev
            else:
                prev = np.nan
                dq.clear()
                run = 0.0
            continue
        if not np.isfinite(x):
            dq.clear()
            run = 0.0
            continue
        dq.append(float(x))
        run += float(x)
        if len(dq) > length:
            run -= dq.popleft()
        if len(dq) == length:
            prev = run / length
            out[i] = prev
    return out


def tv_ema(src: np.ndarray, length: int) -> np.ndarray:
    return tv_smooth(src, length, 2.0 / (length + 1.0))


def tv_rma(src: np.ndarray, length: int) -> np.ndarray:
    return tv_smooth(src, length, 1.0 / length)


def tv_sma(src: np.ndarray, length: int) -> np.ndarray:
    n = len(src)
    out = np.full(n, np.nan, dtype=np.float64)
    dq: deque[float] = deque()
    run = 0.0
    for i in range(n):
        x = src[i]
        if not np.isfinite(x):
            dq.clear()
            run = 0.0
            continue
        dq.append(float(x))
        run += float(x)
        if len(dq) > length:
            run -= dq.popleft()
        if len(dq) == length:
            out[i] = run / length
    return out


def fixnan(src: np.ndarray) -> np.ndarray:
    out = src.copy()
    last = np.nan
    for i in range(len(out)):
        if np.isfinite(out[i]):
            last = out[i]
        elif np.isfinite(last):
            out[i] = last
    return out


def _self_check() -> None:
    src = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    ema = tv_ema(src, 3)
    # alpha = 0.5; seed = mean(1,2,3) = 2 at index 2
    assert abs(ema[2] - 2.0) < 1e-12, ema
    assert abs(ema[3] - 3.0) < 1e-12, ema  # 0.5*4+0.5*2
    assert abs(ema[4] - 4.0) < 1e-12, ema
    assert math.isnan(ema[0]) and math.isnan(ema[1])
    src2 = np.array([np.nan, 1, 2, 3, 4, 5], dtype=np.float64)
    ema2 = tv_ema(src2, 3)
    assert math.isnan(ema2[2])
    assert abs(ema2[3] - 2.0) < 1e-12, ema2
    rma = tv_rma(src, 3)  # alpha = 1/3
    assert abs(rma[2] - 2.0) < 1e-12
    expect = (1.0 / 3.0) * 4 + (2.0 / 3.0) * 2
    assert abs(rma[3] - expect) < 1e-12, (rma[3], expect)


def f_levels(is_long: bool, entry: float, atr_i: float, swing_low: float, swing_high: float):
    floor = atr_i * ATR_MULT if np.isfinite(atr_i) else np.nan
    if is_long:
        dist = entry - swing_low if np.isfinite(swing_low) else np.nan
        if np.isfinite(floor) and np.isfinite(dist) and dist < floor:
            sl = entry - floor
        else:
            sl = swing_low
        risk = entry - sl if np.isfinite(sl) else np.nan
        tp1 = entry + risk if np.isfinite(risk) else np.nan
        tp2 = entry + risk * 1.5 if np.isfinite(risk) else np.nan
    else:
        dist = swing_high - entry if np.isfinite(swing_high) else np.nan
        if np.isfinite(floor) and np.isfinite(dist) and dist < floor:
            sl = entry + floor
        else:
            sl = swing_high
        risk = sl - entry if np.isfinite(sl) else np.nan
        tp1 = entry - risk if np.isfinite(risk) else np.nan
        tp2 = entry - risk * 1.5 if np.isfinite(risk) else np.nan
    return sl, tp1, tp2, risk


def compute_indicators(high, low, close, volume):
    n = len(close)
    prev_close = np.empty(n, dtype=np.float64)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]

    hl = high - low
    tr_true = np.maximum(hl, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    # ta.atr uses ta.tr(true): first bar falls back to high-low
    tr_atr = tr_true.copy()
    tr_atr[0] = hl[0]
    atr = tv_rma(tr_atr, ATR_LEN)

    # ta.dmi uses ta.tr (handle_na false) so first TR is na
    tr_dmi = tr_true.copy()
    tr_dmi[0] = np.nan

    up = np.empty(n, dtype=np.float64)
    down = np.empty(n, dtype=np.float64)
    up[0] = np.nan
    down[0] = np.nan
    up[1:] = high[1:] - high[:-1]
    down[1:] = low[:-1] - low[1:]
    plus_dm = np.where(np.isfinite(up), np.where((up > down) & (up > 0), up, 0.0), np.nan)
    minus_dm = np.where(np.isfinite(down), np.where((down > up) & (down > 0), down, 0.0), np.nan)

    rma_tr = tv_rma(tr_dmi, DI_LEN)
    rma_plus = tv_rma(plus_dm, DI_LEN)
    rma_minus = tv_rma(minus_dm, DI_LEN)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * rma_plus / rma_tr
        minus_di = 100.0 * rma_minus / rma_tr
    plus_di = np.where(np.isfinite(rma_tr) & (rma_tr != 0) & np.isfinite(rma_plus), plus_di, np.nan)
    minus_di = np.where(np.isfinite(rma_tr) & (rma_tr != 0) & np.isfinite(rma_minus), minus_di, np.nan)
    plus_di = fixnan(plus_di)
    minus_di = fixnan(minus_di)

    di_sum = plus_di + minus_di
    with np.errstate(divide="ignore", invalid="ignore"):
        dx_ratio = np.abs(plus_di - minus_di) / np.where(di_sum == 0, 1.0, di_sum)
    dx_ratio = np.where(np.isfinite(plus_di) & np.isfinite(minus_di), dx_ratio, np.nan)
    adx = 100.0 * tv_rma(dx_ratio, ADX_SMOOTH)

    ema12 = tv_ema(close, 12)
    ema26 = tv_ema(close, 26)
    macd = ema12 - ema26
    signal = tv_ema(macd, 9)
    hist = macd - signal
    hist_prev = np.empty(n, dtype=np.float64)
    hist_prev[0] = np.nan
    hist_prev[1:] = hist[:-1]

    ema9 = tv_ema(close, 9)
    ema21 = tv_ema(close, 21)
    ema55 = tv_ema(close, 55)

    delta = np.empty(n, dtype=np.float64)
    delta[0] = np.nan
    delta[1:] = close[1:] - close[:-1]
    gain = np.where(np.isfinite(delta), np.maximum(delta, 0.0), np.nan)
    loss = np.where(np.isfinite(delta), np.maximum(-delta, 0.0), np.nan)
    avg_gain = tv_rma(gain, 14)
    avg_loss = tv_rma(loss, 14)
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        u = avg_gain[i]
        d = avg_loss[i]
        if not np.isfinite(u) or not np.isfinite(d):
            continue
        if d == 0:
            rsi[i] = 100.0
        elif u == 0:
            rsi[i] = 0.0
        else:
            rs = u / d
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    vol_avg = tv_sma(volume, 20)

    change = np.empty(n, dtype=np.float64)
    change[0] = np.nan
    change[1:] = np.abs(close[1:] - close[:-1])
    ema_change = tv_ema(change, PER)
    smrng = tv_ema(ema_change, PER * 2 - 1) * MULT

    swing_low = pd.Series(low).rolling(SWING_LEN, min_periods=SWING_LEN).min().shift(1).to_numpy()
    swing_high = pd.Series(high).rolling(SWING_LEN, min_periods=SWING_LEN).max().shift(1).to_numpy()

    bull_core = (
        (ema9 > ema21)
        & (close > ema9)
        & (close > ema21)
        & (close > ema55)
        & (adx >= 20.0)
        & (plus_di > minus_di)
        & (rsi > 50.0)
        & (rsi < 75.0)
        & (hist > 0.0)
        & (hist > hist_prev)
        & (volume > 1.2 * vol_avg)
    )
    bear_core = (
        (ema9 < ema21)
        & (close < ema9)
        & (close < ema21)
        & (close < ema55)
        & (adx >= 20.0)
        & (minus_di > plus_di)
        & (rsi > 25.0)
        & (rsi < 50.0)
        & (hist < 0.0)
        & (hist < hist_prev)
        & (volume > 1.2 * vol_avg)
    )

    return {
        "atr": atr,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "adx": adx,
        "smrng": smrng,
        "swing_low": swing_low,
        "swing_high": swing_high,
        "bull_core": bull_core,
        "bear_core": bear_core,
    }


def run_symbol(symbol: str, df: pd.DataFrame) -> dict:
    df = df.sort_index()
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    volume = df["volume"].to_numpy(dtype=np.float64)
    times = df.index
    n = len(df)
    hours = times.hour.to_numpy()
    minutes = times.minute.to_numpy()
    # calendar date in Asia/Kolkata
    dates = times.tz_convert(TZ).date

    in_sess = ((hours > 9) | ((hours == 9) & (minutes >= 15))) & (
        (hours < 15) | ((hours == 15) & (minutes <= 30))
    )
    by_cut = (hours < 9) | ((hours == 9) & (minutes <= 20))
    flatten_bar = (hours == 15) & (minutes >= 25)

    ind = compute_indicators(high, low, close, volume)
    atr = ind["atr"]
    smrng = ind["smrng"]
    swing_low = ind["swing_low"]
    swing_high = ind["swing_high"]
    bull_core = ind["bull_core"]
    bear_core = ind["bear_core"]

    cash = INITIAL
    pos = 0
    entry_price = 0.0
    entry_i = -1
    qty = 0.0
    risk_pts = 0.0
    equity_at_entry = INITIAL
    act_sl = np.nan
    act_tp1 = np.nan
    act_tp2 = np.nan
    orig_sl = np.nan
    orig_tp1 = np.nan
    orig_tp2 = np.nan
    locked = False
    signal_i = -1

    took = False
    scan_checked = False
    scan_side = 0
    printed = False
    pend_side = 0
    pend_sl = np.nan
    pend_tp1 = np.nan
    pend_tp2 = np.nan

    cond_ini = 0
    upward = 0
    downward = 0
    rf_prev = np.nan

    cum_pv = 0.0
    cum_v = 0.0
    vwap = np.nan

    trades = []
    n_signals = 0
    n_pending = 0
    n_entries = 0
    open_at_end = 0

    def close_trade(i, exit_price, reason):
        nonlocal cash, pos, qty
        if pos == 1:
            pnl = (exit_price - entry_price) * qty
            r_mult = (exit_price - entry_price) / risk_pts if risk_pts > 0 else np.nan
            side = "long"
        else:
            pnl = (entry_price - exit_price) * qty
            r_mult = (entry_price - exit_price) / risk_pts if risk_pts > 0 else np.nan
            side = "short"
        pnl_pct = (pnl / equity_at_entry * 100.0) if equity_at_entry != 0 else np.nan
        cash += pnl
        trades.append(
            {
                "symbol": symbol,
                "side": side,
                "signal_time": times[signal_i].isoformat(),
                "entry_time": times[entry_i].isoformat(),
                "exit_time": times[i].isoformat(),
                "entry": entry_price,
                "sl": orig_sl,
                "tp1": orig_tp1,
                "tp2": orig_tp2,
                "exit": exit_price,
                "exit_reason": reason,
                "r_multiple": r_mult,
                "pnl": pnl,
                "pnl_pct_equity": pnl_pct,
            }
        )
        pos = 0

    for i in range(n):
        new_day = i == 0 or dates[i] != dates[i - 1]
        if new_day:
            scan_checked = False
            scan_side = 0
            printed = False
            pend_side = 0
            pend_sl = np.nan
            pend_tp1 = np.nan
            pend_tp2 = np.nan
            took = False
            cum_pv = 0.0
            cum_v = 0.0
            vwap = np.nan

        vol_i = volume[i]
        if in_sess[i] and np.isfinite(vol_i) and vol_i > 0:
            typical = (high[i] + low[i] + close[i]) / 3.0
            cum_pv += typical * vol_i
            cum_v += vol_i
            vwap = cum_pv / cum_v if cum_v > 0 else np.nan

        # Range filter + direction counters + condIni (all bars, for warmup)
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
        long_flip = long_cond and cond_ini == -1
        short_flip = short_cond and cond_ini == 1
        if long_cond:
            cond_ini = 1
        elif short_cond:
            cond_ini = -1
        rf_prev = rf_i

        # Exits on bars after entry, before any new entry on this bar
        if pos != 0 and i > entry_i:
            if pos == 1:
                stop_hit = low[i] <= act_sl
                tp2_hit = high[i] >= act_tp2
                tp1_hit = high[i] >= act_tp1
            else:
                stop_hit = high[i] >= act_sl
                tp2_hit = low[i] <= act_tp2
                tp1_hit = low[i] <= act_tp1
            if stop_hit:
                close_trade(i, float(act_sl), "stop")
            elif tp2_hit:
                close_trade(i, float(act_tp2), "tp2")
            else:
                if (not locked) and tp1_hit:
                    act_sl = act_tp1
                    locked = True
                if pos != 0 and flatten_bar[i]:
                    close_trade(i, float(close[i]), "session")

        # Opening scan: first in-session bar with volume only
        if in_sess[i] and np.isfinite(vol_i) and vol_i > 0 and not scan_checked:
            scan_checked = True
            if bull_core[i] and np.isfinite(vwap) and c > vwap:
                scan_side = 1
            elif bear_core[i] and np.isfinite(vwap) and c < vwap:
                scan_side = -1
            else:
                scan_side = 0

        # First in-session range-filter flip of the day
        if in_sess[i] and (not printed) and (long_flip or short_flip):
            printed = True
            n_signals += 1
            is_long = bool(long_flip)
            sl, tp1, tp2, risk = f_levels(
                is_long, float(c), float(atr[i]), float(swing_low[i]), float(swing_high[i])
            )
            stop_ok = False
            if np.isfinite(risk) and np.isfinite(c) and c != 0:
                stop_pct = risk / c * 100.0
                stop_ok = (stop_pct >= MIN_STOP) and (stop_pct <= MAX_STOP) and (risk > 0)
            if by_cut[i] and stop_ok:
                pend_side = 1 if is_long else -1
                pend_sl = sl
                pend_tp1 = tp1
                pend_tp2 = tp2
                signal_i = i
                n_pending += 1

        # Entry at this bar's close
        if (
            pos == 0
            and in_sess[i]
            and by_cut[i]
            and pend_side != 0
            and (not took)
            and np.isfinite(c)
        ):
            if pend_side == 1:
                price_ok = (c > pend_sl) and (c < pend_tp1)
            else:
                price_ok = (c < pend_sl) and (c > pend_tp1)
            scan_ok = scan_side == pend_side
            if scan_ok and price_ok:
                rp = abs(c - pend_sl)
                q = (cash * (RISK_PCT / 100.0)) / rp if rp > 0 else 0.0
                if q > 0:
                    pos = pend_side
                    entry_price = float(c)
                    entry_i = i
                    qty = float(q)
                    risk_pts = float(rp)
                    equity_at_entry = float(cash)
                    act_sl = float(pend_sl)
                    act_tp1 = float(pend_tp1)
                    act_tp2 = float(pend_tp2)
                    orig_sl = act_sl
                    orig_tp1 = act_tp1
                    orig_tp2 = act_tp2
                    locked = False
                    took = True
                    n_entries += 1
                    if signal_i < 0:
                        signal_i = i

    if pos != 0:
        open_at_end = 1

    vol_finite = volume[np.isfinite(volume)]
    meta = {
        "symbol": symbol,
        "bars": int(n),
        "first": times[0].isoformat() if n else None,
        "last": times[-1].isoformat() if n else None,
        "volume_sum": float(np.nansum(volume)) if n else 0.0,
        "volume_positive_bars": int(np.sum(np.isfinite(volume) & (volume > 0))) if n else 0,
        "signals": n_signals,
        "pending": n_pending,
        "entries": n_entries,
        "open_at_end": open_at_end,
        "trades_closed": len(trades),
    }
    return {"trades": trades, "meta": meta}


def cache_path(symbol: str) -> Path:
    safe = (
        symbol.replace("^", "IDX_")
        .replace("&", "AND")
        .replace("/", "_")
    )
    return DATA / f"{safe}.csv"


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "") for c in out.columns]
    need = ["open", "high", "low", "close"]
    if any(c not in out.columns for c in need):
        return None
    if "volume" not in out.columns:
        out["volume"] = np.nan
    out = out[["open", "high", "low", "close", "volume"]]
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    idx = out.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, utc=True)
        out.index = idx
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC").tz_convert(TZ)
    else:
        out.index = out.index.tz_convert(TZ)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["open", "high", "low", "close"])
    if out.empty:
        return None
    # reject all-nan / empty downloads that only have a handful of bad rows
    if len(out) < 50:
        return None
    return out


def save_cache(symbol: str, df: pd.DataFrame) -> None:
    path = cache_path(symbol)
    tmp = df.copy()
    tmp.index.name = "datetime"
    tmp.to_csv(path)


def load_cache(symbol: str) -> pd.DataFrame | None:
    path = cache_path(symbol)
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.set_index("datetime")
    return normalize_ohlcv(df)


def extract_ticker(raw: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(map(str, raw.columns.get_level_values(0)))
        if ticker in level0:
            sub = raw[ticker].copy()
        elif "Close" in level0 or "close" in {x.lower() for x in level0}:
            try:
                sub = raw.xs(ticker, axis=1, level=1).copy()
            except KeyError:
                return None
        else:
            return None
    else:
        sub = raw.copy()
    # drop rows that are entirely nan for this ticker
    sub = sub.dropna(how="all")
    return normalize_ohlcv(sub)


def download_symbols(symbols: list[str]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    import yfinance as yf

    loaded: dict[str, pd.DataFrame] = {}
    need: list[str] = []
    for s in symbols:
        cached = load_cache(s)
        if cached is not None:
            loaded[s] = cached
        else:
            need.append(s)
    print(f"cache hit {len(loaded)} need download {len(need)}", flush=True)
    failed: list[str] = []
    batch = 20
    for i in range(0, len(need), batch):
        chunk = need[i : i + batch]
        print(f"download {i+1}-{i+len(chunk)} / {len(need)}", flush=True)
        raw = None
        err = None
        for attempt in range(3):
            try:
                raw = yf.download(
                    chunk,
                    period="60d",
                    interval="5m",
                    auto_adjust=False,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                    timeout=60,
                )
                err = None
                break
            except Exception as e:
                err = e
                print(f"  batch error attempt {attempt+1}: {e}", flush=True)
                time.sleep(2 + attempt * 2)
        if err is not None or raw is None or (hasattr(raw, "empty") and raw.empty):
            # individual fallback
            for s in chunk:
                ok = False
                for attempt in range(2):
                    try:
                        one = yf.download(
                            s,
                            period="60d",
                            interval="5m",
                            auto_adjust=False,
                            group_by="ticker",
                            threads=False,
                            progress=False,
                            timeout=60,
                        )
                        df = extract_ticker(one, s)
                        if df is None and not isinstance(one.columns, pd.MultiIndex):
                            df = normalize_ohlcv(one)
                        if df is not None:
                            save_cache(s, df)
                            loaded[s] = df
                            ok = True
                            break
                    except Exception as e:
                        print(f"  {s} fail {e}", flush=True)
                        time.sleep(1)
                if not ok:
                    failed.append(s)
            continue
        for s in chunk:
            df = extract_ticker(raw, s)
            if df is None:
                failed.append(s)
            else:
                save_cache(s, df)
                loaded[s] = df
        time.sleep(0.5)
    # one individual retry for batch failures
    still = []
    for s in failed:
        try:
            one = yf.download(
                s,
                period="60d",
                interval="5m",
                auto_adjust=False,
                progress=False,
                threads=False,
                timeout=60,
            )
            df = extract_ticker(one, s)
            if df is None and one is not None and not one.empty and not isinstance(one.columns, pd.MultiIndex):
                df = normalize_ohlcv(one)
            if df is not None:
                save_cache(s, df)
                loaded[s] = df
                print(f"  retry ok {s} bars={len(df)}", flush=True)
            else:
                still.append(s)
                print(f"  retry empty {s}", flush=True)
        except Exception as e:
            still.append(s)
            print(f"  retry fail {s}: {e}", flush=True)
    return loaded, still


def load_fno_symbols() -> list[str]:
    path = DATA / "NSEFO.csv"
    if not path.exists() or path.stat().st_size < 1000:
        import urllib.request

        req = urllib.request.Request(UNIVERSE_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            path.write_bytes(resp.read())
    df = pd.read_csv(path)
    df["exp"] = pd.to_datetime(df["ContractExpiration"], errors="coerce")
    stk = df[df["Series"].isin(["OPTSTK", "FUTSTK"]) & (df["exp"] >= ASOF)].copy()
    stk = stk[~stk["Name"].astype(str).str.contains("NSETEST", na=False)]
    names = sorted(stk["Name"].dropna().astype(str).unique())
    # lot sizes from nearest future
    fut = stk[stk["Series"] == "FUTSTK"].sort_values("exp")
    nearest = fut.groupby("Name", as_index=False).first()
    nearest = nearest.sort_values("Name")
    out = nearest[["Name", "LotSize", "exp"]].rename(
        columns={"Name": "symbol", "LotSize": "lot_size", "exp": "nearest_fut_expiry"}
    )
    out.to_csv(DATA / "fno_universe.csv", index=False)
    # symbols present in either series
    return names


def stats_from_trades(trades: list[dict], start: float = INITIAL) -> dict:
    if not trades:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "breakeven": 0,
            "win_rate": None,
            "net_pnl": 0.0,
            "profit_factor": None,
            "profit_factor_undefined": True,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "average_r": None,
            "ending_equity": start,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
        }
    pnls = np.array([t["pnl"] for t in trades], dtype=np.float64)
    rs = np.array([t["r_multiple"] for t in trades], dtype=np.float64)
    wins = int(np.sum(pnls > 0))
    losses = int(np.sum(pnls < 0))
    be = int(np.sum(pnls == 0))
    gp = float(pnls[pnls > 0].sum()) if wins else 0.0
    gl = float(-pnls[pnls < 0].sum()) if losses else 0.0
    if gl > 0:
        pf = gp / gl
        pf_undef = False
    else:
        pf = None
        pf_undef = True
    eq = start
    peak = start
    max_dd = 0.0
    max_dd_pct = 0.0
    for p in pnls:
        eq += p
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd
        if peak > 0:
            ddp = dd / peak
            if ddp > max_dd_pct:
                max_dd_pct = ddp
    return {
        "trades": int(len(trades)),
        "wins": wins,
        "losses": losses,
        "breakeven": be,
        "win_rate": wins / len(trades),
        "net_pnl": float(pnls.sum()),
        "profit_factor": pf,
        "profit_factor_undefined": pf_undef,
        "max_drawdown": float(max_dd),
        "max_drawdown_pct": float(max_dd_pct * 100.0),
        "average_r": float(np.nanmean(rs)),
        "ending_equity": float(eq),
        "gross_profit": gp,
        "gross_loss": gl,
    }


def side_breakdown(trades: list[dict]) -> dict:
    out = {}
    for side in ("long", "short"):
        sub = [t for t in trades if t["side"] == side]
        st = stats_from_trades(sub)
        out[side] = {
            "trades": st["trades"],
            "win_rate": st["win_rate"],
            "net_pnl": st["net_pnl"],
            "average_r": st["average_r"],
            "profit_factor": st["profit_factor"],
        }
    return out


def exit_counts(trades: list[dict]) -> dict:
    counts = {"stop": 0, "tp2": 0, "session": 0}
    for t in trades:
        r = t["exit_reason"]
        counts[r] = counts.get(r, 0) + 1
    return counts


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


def main() -> None:
    _self_check()
    print("indicator self-check ok", flush=True)
    fno = load_fno_symbols()
    print(f"F&O equity underlyings: {len(fno)}", flush=True)
    yahoo_fno = [f"{s}.NS" for s in fno]
    requested = INDICES + yahoo_fno
    frames, skipped = download_symbols(requested)
    print(f"downloaded {len(frames)} skipped {len(skipped)}", flush=True)
    if skipped:
        print("skipped:", ", ".join(skipped), flush=True)

    all_trades = []
    per_symbol = {}
    metas = {}
    open_positions = []
    for i, sym in enumerate(requested):
        if sym not in frames:
            continue
        try:
            result = run_symbol(sym, frames[sym])
        except Exception:
            print(f"BACKTEST FAIL {sym}", flush=True)
            traceback.print_exc()
            skipped.append(sym)
            continue
        metas[sym] = result["meta"]
        trades = result["trades"]
        all_trades.extend(trades)
        per_symbol[sym] = stats_from_trades(trades)
        per_symbol[sym]["signals"] = result["meta"]["signals"]
        per_symbol[sym]["pending"] = result["meta"]["pending"]
        per_symbol[sym]["entries"] = result["meta"]["entries"]
        per_symbol[sym]["open_at_end"] = result["meta"]["open_at_end"]
        if result["meta"]["open_at_end"]:
            open_positions.append(sym)
        if (i + 1) % 25 == 0:
            print(f"backtested {i+1}/{len(requested)} closed_trades_so_far={len(all_trades)}", flush=True)

    # trades.csv in requested symbol order then time
    order = {s: k for k, s in enumerate(requested)}
    all_trades.sort(key=lambda t: (order.get(t["symbol"], 10**9), t["entry_time"], t["exit_time"]))
    cols = [
        "symbol",
        "side",
        "signal_time",
        "entry_time",
        "exit_time",
        "entry",
        "sl",
        "tp1",
        "tp2",
        "exit",
        "exit_reason",
        "r_multiple",
        "pnl",
        "pnl_pct_equity",
    ]
    tdf = pd.DataFrame(all_trades, columns=cols)
    tdf.to_csv(ROOT / "trades.csv", index=False)

    def window_block(symbols: list[str]) -> dict:
        present = [s for s in symbols if s in metas]
        if not present:
            return {"symbols": 0}
        firsts = [metas[s]["first"] for s in present]
        lasts = [metas[s]["last"] for s in present]
        bars = [metas[s]["bars"] for s in present]
        return {
            "symbols": len(present),
            "first_bar_min": min(firsts),
            "last_bar_max": max(lasts),
            "total_bars": int(sum(bars)),
            "min_bars": int(min(bars)),
            "max_bars": int(max(bars)),
            "median_bars": float(np.median(bars)),
            "per_symbol_bars": {s: {"bars": metas[s]["bars"], "first": metas[s]["first"], "last": metas[s]["last"],
                                    "volume_sum": metas[s]["volume_sum"],
                                    "volume_positive_bars": metas[s]["volume_positive_bars"]} for s in present},
        }

    index_stats = {s: per_symbol.get(s) for s in INDICES}
    fno_present = [s for s in yahoo_fno if s in per_symbol]
    fno_trades = [t for t in all_trades if t["symbol"] in set(fno_present)]
    fno_nets = np.array([per_symbol[s]["net_pnl"] for s in fno_present], dtype=np.float64) if fno_present else np.array([])
    traded_syms = [s for s in fno_present if per_symbol[s]["trades"] > 0]
    traded_nets = np.array([per_symbol[s]["net_pnl"] for s in traded_syms], dtype=np.float64) if traded_syms else np.array([])
    pooled = stats_from_trades(fno_trades)  # profit factor / win rate / avg R on pooled trades; net pnl is sum
    # max dd on pooled trades is NOT a single equity curve. Primary per-symbol dd is separate.
    # Sum of independent accounts:
    sum_net = float(fno_nets.sum()) if len(fno_nets) else 0.0

    # ranking all downloaded symbols
    ranked = sorted(per_symbol.items(), key=lambda kv: kv[1]["net_pnl"], reverse=True)
    def brief(sym, st):
        return {
            "symbol": sym,
            "net_pnl": st["net_pnl"],
            "trades": st["trades"],
            "win_rate": st["win_rate"],
            "profit_factor": st["profit_factor"],
            "max_drawdown": st["max_drawdown"],
            "max_drawdown_pct": st["max_drawdown_pct"],
            "average_r": st["average_r"],
            "ending_equity": st["ending_equity"],
        }

    top5 = [brief(s, st) for s, st in ranked[:5]]
    bottom5 = [brief(s, st) for s, st in ranked[-5:]]

    # equal-start sum across every downloaded symbol (indices + fno)
    all_present = [s for s in requested if s in per_symbol]
    total_net_all = float(sum(per_symbol[s]["net_pnl"] for s in all_present))
    n_all = len(all_present)

    summary = {
        "costs": {
            "commission": 0.0,
            "slippage": 0.0,
            "note": "Zero commission and zero slippage (TradingView strategy defaults; none set in the script).",
        },
        "capital": {
            "initial_per_symbol": INITIAL,
            "risk_pct": RISK_PCT,
            "accounts": "independent, not a shared portfolio equity curve",
            "qty": "fractional shares",
        },
        "universe": {
            "source": UNIVERSE_URL,
            "source_last_modified_http": UNIVERSE_LAST_MODIFIED,
            "asof_filter": "FUTSTK/OPTSTK contract expiration >= 2026-10-04, exclude NSETEST",
            "official_nse_fetch": "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv and https://archives.nseindia.com/content/fo/fo_mktlots.csv returned HTTP 403",
            "fno_equity_underlyings": len(fno),
            "yahoo_mapping": "SYMBOL.NS",
            "indices": INDICES,
        },
        "symbols": {
            "requested": len(requested),
            "requested_indices": INDICES,
            "requested_fno": len(yahoo_fno),
            "downloaded": len(per_symbol),
            "downloaded_indices": [s for s in INDICES if s in per_symbol],
            "downloaded_fno": len(fno_present),
            "skipped": skipped,
        },
        "data_window": {
            "indices": window_block(INDICES),
            "fno": window_block(yahoo_fno),
        },
        "indices": {
            s: None if s not in per_symbol else {
                **{k: per_symbol[s][k] for k in (
                    "trades", "win_rate", "net_pnl", "profit_factor", "profit_factor_undefined",
                    "max_drawdown", "max_drawdown_pct", "average_r", "ending_equity",
                    "wins", "losses", "gross_profit", "gross_loss", "signals", "pending", "entries", "open_at_end"
                )},
                "volume_sum": metas[s]["volume_sum"],
                "volume_positive_bars": metas[s]["volume_positive_bars"],
                "bars": metas[s]["bars"],
                "first": metas[s]["first"],
                "last": metas[s]["last"],
            }
            for s in INDICES
        },
        "fno_pooled": {
            "symbols_downloaded": len(fno_present),
            "symbols_traded": len(traded_syms),
            "symbols_with_zero_trades": len(fno_present) - len(traded_syms),
            "total_trades": pooled["trades"],
            "win_rate": pooled["win_rate"],
            "average_net_pnl_per_downloaded_symbol": (sum_net / len(fno_present)) if fno_present else None,
            "median_net_pnl_per_downloaded_symbol": float(np.median(fno_nets)) if len(fno_nets) else None,
            "average_net_pnl_per_traded_symbol": float(traded_nets.mean()) if len(traded_nets) else None,
            "median_net_pnl_per_traded_symbol": float(np.median(traded_nets)) if len(traded_nets) else None,
            "total_net_pnl_sum_of_independent_accounts": sum_net,
            "sum_of_ending_equities": (len(fno_present) * INITIAL + sum_net) if fno_present else None,
            "profit_factor_pooled_trades": pooled["profit_factor"],
            "profit_factor_undefined": pooled["profit_factor_undefined"],
            "average_r": pooled["average_r"],
            "symbols_positive_net_pnl": int(np.sum(fno_nets > 0)) if len(fno_nets) else 0,
            "symbols_negative_net_pnl": int(np.sum(fno_nets < 0)) if len(fno_nets) else 0,
            "symbols_flat_net_pnl": int(np.sum(fno_nets == 0)) if len(fno_nets) else 0,
            "gross_profit": pooled["gross_profit"],
            "gross_loss": pooled["gross_loss"],
            "wins": pooled["wins"],
            "losses": pooled["losses"],
            "breakeven": pooled["breakeven"],
        },
        "equal_start_sum_all_downloaded_symbols": {
            "symbols": n_all,
            "initial_each": INITIAL,
            "sum_initial": n_all * INITIAL,
            "total_net_pnl": total_net_all,
            "sum_ending_equity": n_all * INITIAL + total_net_all,
        },
        "top5_by_net_pnl": top5,
        "bottom5_by_net_pnl": bottom5,
        "long_short": {
            "all_downloaded": side_breakdown(all_trades),
            "indices": side_breakdown([t for t in all_trades if t["symbol"] in set(INDICES)]),
            "fno": side_breakdown(fno_trades),
        },
        "exit_reason_counts": {
            "all": exit_counts(all_trades),
            "indices": exit_counts([t for t in all_trades if t["symbol"] in set(INDICES)]),
            "fno": exit_counts(fno_trades),
        },
        "open_positions_at_end": open_positions,
        "per_symbol": {s: per_symbol[s] for s in all_present},
        "caveats": [
            "Yahoo Finance 5-minute history is limited to about 60 days, not a full multi-year sample.",
            "This is an approximation of TradingView: EMA/RMA SMA seeding and the intrabar path (stop checked before 1.5R; 1R stop move applies the next bar; adverse stop wins if stop and target are both touched) follow the stated rules but are not a tick-for-tick TradingView fill engine.",
            "Index 5-minute volume from Yahoo is often zero, so the volume-based opening scan cannot pass on those symbols.",
            "Qty is fractional. Sizing uses cash equity at entry (flat). One independent 100000 account per symbol.",
        ],
    }

    with open(ROOT / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=Enc)

    # console digest
    print("==== DIGEST ====", flush=True)
    print("skipped", skipped, flush=True)
    print("total closed trades", len(all_trades), flush=True)
    print("open", open_positions, flush=True)
    for s in INDICES:
        if s in per_symbol:
            st = per_symbol[s]
            print(f"INDEX {s} trades={st['trades']} win={st['win_rate']} net={st['net_pnl']} pf={st['profit_factor']} dd={st['max_drawdown']} dd%={st['max_drawdown_pct']} avgR={st['average_r']} vol+={metas[s]['volume_positive_bars']}", flush=True)
    fp = summary["fno_pooled"]
    print("FNO", json.dumps({k: fp[k] for k in fp if k != "per_symbol"}, cls=Enc), flush=True)
    print("TOP", top5, flush=True)
    print("BOTTOM", bottom5, flush=True)
    print("SIDES", summary["long_short"], flush=True)
    print("EXITS", summary["exit_reason_counts"], flush=True)
    print("EQUAL", summary["equal_start_sum_all_downloaded_symbols"], flush=True)
    print("signals", sum(per_symbol[s]["signals"] for s in per_symbol), "pending", sum(per_symbol[s]["pending"] for s in per_symbol), "entries", sum(per_symbol[s]["entries"] for s in per_symbol), flush=True)


if __name__ == "__main__":
    main()
