#!/usr/bin/env python3
"""Standalone morning scanner for F&O equities on Yahoo Finance 5-minute bars."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.request
from collections import deque
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

TZ = "Asia/Kolkata"
UNIVERSE_URL = "https://techfanetechnologies.github.io/XTS_MasterInstruments/csv/NSEFO.csv"
PER = 100
MULT = 3.0
SESSION_START = (9, 15)
SESSION_END = (15, 30)
CUTOFF = (9, 20)
SCAN_BAR = (9, 20)    # opening scan always runs on the 09:20 bar (closes 09:25 IST)
READY_AT = (9, 25)    # nothing is scanned or sent before 09:25 IST
READY_SHARE = 0.8     # share of tickers that must already show a bar after 09:20
READY_RETRIES = 4     # re-downloads while the 09:20 bar is not final (about 3 min)
READY_SLEEP = 30
STOP_PCT = 0.005
LOCK_PCT = 0.015
TRAIL_PCT = 0.001  # Informational rule for trade management after lock.

logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(levelname)s: %(message)s")


def tv_smooth(src: np.ndarray, length: int, alpha: float) -> np.ndarray:
    """TradingView-style EMA/RMA: SMA seed, then recursive smoothing."""
    out = np.full(len(src), np.nan, dtype=np.float64)
    prev = np.nan
    window: deque[float] = deque()
    total = 0.0
    for i, value in enumerate(src):
        if np.isfinite(prev):
            if np.isfinite(value):
                prev = alpha * value + (1.0 - alpha) * prev
                out[i] = prev
            else:
                prev = np.nan
                window.clear()
                total = 0.0
            continue
        if not np.isfinite(value):
            window.clear()
            total = 0.0
            continue
        window.append(float(value))
        total += float(value)
        if len(window) > length:
            total -= window.popleft()
        if len(window) == length:
            prev = total / length
            out[i] = prev
    return out


def tv_ema(src: np.ndarray, length: int) -> np.ndarray:
    return tv_smooth(src, length, 2.0 / (length + 1.0))


def tv_rma(src: np.ndarray, length: int) -> np.ndarray:
    return tv_smooth(src, length, 1.0 / length)


def tv_sma(src: np.ndarray, length: int) -> np.ndarray:
    out = np.full(len(src), np.nan, dtype=np.float64)
    window: deque[float] = deque()
    total = 0.0
    for i, value in enumerate(src):
        if not np.isfinite(value):
            window.clear()
            total = 0.0
            continue
        window.append(float(value))
        total += float(value)
        if len(window) > length:
            total -= window.popleft()
        if len(window) == length:
            out[i] = total / length
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


def indicators(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> dict[str, np.ndarray]:
    """The indicator math mirrors the corresponding backtest implementation."""
    n = len(close)
    prev_close = np.empty(n, dtype=np.float64)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]

    hl = high - low
    tr_true = np.maximum(hl, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    tr_atr = tr_true.copy()
    tr_atr[0] = hl[0]
    atr = tv_rma(tr_atr, 14)

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

    rma_tr = tv_rma(tr_dmi, 14)
    rma_plus = tv_rma(plus_dm, 14)
    rma_minus = tv_rma(minus_dm, 14)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * rma_plus / rma_tr
        minus_di = 100.0 * rma_minus / rma_tr
    plus_di = np.where(np.isfinite(rma_tr) & (rma_tr != 0) & np.isfinite(rma_plus), plus_di, np.nan)
    minus_di = np.where(np.isfinite(rma_tr) & (rma_tr != 0) & np.isfinite(rma_minus), minus_di, np.nan)
    plus_di = fixnan(plus_di)
    minus_di = fixnan(minus_di)

    di_sum = plus_di + minus_di
    with np.errstate(divide="ignore", invalid="ignore"):
        dx = np.abs(plus_di - minus_di) / np.where(di_sum == 0, 1.0, di_sum)
    dx = np.where(np.isfinite(plus_di) & np.isfinite(minus_di), dx, np.nan)
    adx = 100.0 * tv_rma(dx, 14)

    ema12 = tv_ema(close, 12)
    ema26 = tv_ema(close, 26)
    macd = ema12 - ema26
    macd_signal = tv_ema(macd, 9)
    hist = macd - macd_signal
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
        if not np.isfinite(avg_gain[i]) or not np.isfinite(avg_loss[i]):
            continue
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        elif avg_gain[i] == 0:
            rsi[i] = 0.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    vol_avg = tv_sma(volume, 20)
    change = np.empty(n, dtype=np.float64)
    change[0] = np.nan
    change[1:] = np.abs(close[1:] - close[:-1])
    ema_change = tv_ema(change, PER)
    smrng = tv_ema(ema_change, PER * 2 - 1) * MULT

    bull = (
        (ema9 > ema21) & (close > ema9) & (close > ema21) & (close > ema55)
        & (adx >= 20.0) & (plus_di > minus_di) & (rsi > 50.0) & (rsi < 75.0)
        & (hist > 0.0) & (hist > hist_prev) & (volume > 1.2 * vol_avg)
    )
    bear = (
        (ema9 < ema21) & (close < ema9) & (close < ema21) & (close < ema55)
        & (adx >= 20.0) & (minus_di > plus_di) & (rsi > 25.0) & (rsi < 50.0)
        & (hist < 0.0) & (hist < hist_prev) & (volume > 1.2 * vol_avg)
    )
    return {"smrng": smrng, "bull": bull, "bear": bear}


def normalize(raw: pd.DataFrame | None) -> pd.DataFrame | None:
    if raw is None or raw.empty:
        return None
    out = raw.copy()
    out.columns = [str(c).strip().lower().replace(" ", "") for c in out.columns]
    required = ["open", "high", "low", "close"]
    if any(c not in out.columns for c in required):
        return None
    if "volume" not in out.columns:
        out["volume"] = np.nan
    out = out[["open", "high", "low", "close", "volume"]]
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
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
    return out if not out.empty else None


def extract_ticker(raw: pd.DataFrame | None, ticker: str) -> pd.DataFrame | None:
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        for level in range(raw.columns.nlevels):
            values = {str(x) for x in raw.columns.get_level_values(level)}
            if ticker in values:
                try:
                    return normalize(raw.xs(ticker, axis=1, level=level, drop_level=True))
                except KeyError:
                    return None
        return None
    return normalize(raw)


def load_universe(today) -> list[str]:
    request = urllib.request.Request(UNIVERSE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        content = response.read()
    df = pd.read_csv(pd.io.common.BytesIO(content))
    lookup = {str(c).lower().replace(" ", ""): c for c in df.columns}
    series_col = lookup.get("series")
    name_col = lookup.get("name")
    expiry_col = lookup.get("contractexpiration")
    if not series_col or not name_col or not expiry_col:
        raise ValueError("NSEFO.csv is missing Series, Name, or ContractExpiration")
    expiry = pd.to_datetime(df[expiry_col], errors="coerce")
    series = df[series_col].astype(str).str.upper()
    names = df[name_col].astype(str).str.strip()
    keep = series.isin(["FUTSTK", "OPTSTK"]) & (expiry >= pd.Timestamp(today))
    keep &= ~names.str.contains("NSETEST", case=False, na=False)
    return sorted(names[keep].dropna().unique().tolist())


def download_frames(tickers: list[str]) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    loaded: dict[str, pd.DataFrame] = {}
    batch_size = 20
    for start in range(0, len(tickers), batch_size):
        batch = tickers[start : start + batch_size]
        raw = None
        for attempt in range(3):
            try:
                raw = yf.download(
                    batch, period="60d", interval="5m", auto_adjust=False,
                    group_by="ticker", threads=True, progress=False, timeout=60,
                )
                break
            except Exception as exc:
                logging.warning("download batch failed (%d/3): %s", attempt + 1, exc)
                time.sleep(2 + attempt * 2)
        for ticker in batch:
            frame = extract_ticker(raw, ticker)
            if frame is not None and len(frame) >= 50:
                loaded[ticker] = frame
            else:
                # A single-ticker retry handles Yahoo's occasional partial batch response.
                try:
                    one = yf.download(
                        ticker, period="60d", interval="5m", auto_adjust=False,
                        progress=False, threads=False, timeout=60,
                    )
                    frame = extract_ticker(one, ticker)
                    if frame is not None and len(frame) >= 50:
                        loaded[ticker] = frame
                except Exception as exc:
                    logging.warning("download failed for %s: %s", ticker, exc)
        if start + batch_size < len(tickers):
            time.sleep(0.5)
    return loaded


def prepare(df: pd.DataFrame, today) -> pd.DataFrame | None:
    """09:26 rules: zero every 09:15 bar's volume (Yahoo later zeroes it anyway, so live and backtest
    agree), drop today's bars after 09:20 (never scan a later or partial bar), require today's 09:20 bar."""
    df = df.sort_index().copy()
    t = df.index
    is_0915 = (t.hour == 9) & (t.minute == 15)
    df.loc[is_0915, "volume"] = 0.0
    dates = np.asarray(t.date)
    late = (dates == today) & ((t.hour > SCAN_BAR[0]) | ((t.hour == SCAN_BAR[0]) & (t.minute > SCAN_BAR[1])))
    df = df[~late]
    t = df.index
    if not ((np.asarray(t.date) == today) & (t.hour == SCAN_BAR[0]) & (t.minute == SCAN_BAR[1])).any():
        return None
    return df


def bar_final(df: pd.DataFrame, today) -> bool:
    """The 09:20 bar is final once Yahoo has a later bar for today."""
    t = df.index
    after = (np.asarray(t.date) == today) & ((t.hour > SCAN_BAR[0]) | ((t.hour == SCAN_BAR[0]) & (t.minute > SCAN_BAR[1])))
    return bool(after.any())


def scan_symbol(symbol: str, df: pd.DataFrame, today) -> dict | None:
    df = prepare(df, today)
    if df is None:
        return None
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    volume = df["volume"].to_numpy(dtype=np.float64)
    times = df.index
    n = len(df)
    if n < 50:
        return None

    hour = np.asarray(times.hour)
    minute = np.asarray(times.minute)
    in_session = (((hour > SESSION_START[0]) | ((hour == SESSION_START[0]) & (minute >= SESSION_START[1])))
                   & ((hour < SESSION_END[0]) | ((hour == SESSION_END[0]) & (minute <= SESSION_END[1]))))
    by_cutoff = (hour < CUTOFF[0]) | ((hour == CUTOFF[0]) & (minute <= CUTOFF[1]))
    dates = np.asarray(times.date)
    ind = indicators(high, low, close, volume)

    smrng = ind["smrng"]
    bull = ind["bull"]
    bear = ind["bear"]
    cond_ini = 0
    upward = 0
    downward = 0
    rf_prev = np.nan
    scan_checked = False
    scan_side = 0
    pending_side = 0
    printed = False
    took = False
    signal_i = -1
    entry_i = -1
    cum_pv = 0.0
    cum_v = 0.0
    vwap = np.nan

    for i in range(n):
        new_day = i == 0 or dates[i] != dates[i - 1]
        if new_day:
            scan_checked = False
            scan_side = 0
            pending_side = 0
            printed = False
            took = False
            signal_i = -1
            cum_pv = 0.0
            cum_v = 0.0
            vwap = np.nan

        vol = volume[i]
        if in_session[i] and np.isfinite(vol) and vol > 0:
            typical = (high[i] + low[i] + close[i]) / 3.0
            cum_pv += typical * vol
            cum_v += vol
            vwap = cum_pv / cum_v

        c = close[i]
        sm = smrng[i]
        if not np.isfinite(c) or not np.isfinite(sm):
            rf_i = np.nan
        else:
            previous_nonzero = rf_prev if np.isfinite(rf_prev) else 0.0
            if c > previous_nonzero:
                rf_i = previous_nonzero if c - sm < previous_nonzero else c - sm
            else:
                rf_i = previous_nonzero if c + sm > previous_nonzero else c + sm

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

        # The opening scan is evaluated once per IST day, on the 09:20 bar only. The 09:15 bar's volume
        # is zeroed in prepare(); if the 09:20 bar has no volume the scan stays blank for the day.
        if (in_session[i] and not scan_checked and hour[i] == SCAN_BAR[0] and minute[i] == SCAN_BAR[1]):
            scan_checked = True
            if not (np.isfinite(vol) and vol > 0):
                scan_side = 0
            elif bull[i] and np.isfinite(vwap) and c > vwap:
                scan_side = 1
            elif bear[i] and np.isfinite(vwap) and c < vwap:
                scan_side = -1
            else:
                scan_side = 0

        if dates[i] != today or not in_session[i]:
            continue

        # First in-session flip of today; a flip after 09:20 is not eligible.
        if not printed and (long_flip or short_flip):
            printed = True
            if by_cutoff[i]:
                pending_side = 1 if long_flip else -1
                signal_i = i

        # Match the backtest's close-of-bar entry behavior, without a fixed TP.
        if pending_side and not took and by_cutoff[i] and scan_side == pending_side:
            took = True
            entry_i = i
            entry = float(c)
            side = "long" if pending_side == 1 else "short"
            sl = entry * (1.0 - STOP_PCT) if pending_side == 1 else entry * (1.0 + STOP_PCT)
            lock = entry * (1.0 + LOCK_PCT) if pending_side == 1 else entry * (1.0 - LOCK_PCT)
            return {
                "symbol": symbol,
                "side": side,
                "signal_time": times[signal_i].isoformat(),
                "entry_time": times[entry_i].isoformat(),
                "entry": entry,
                "sl": float(sl),
                "lock": float(lock),
            }
    return None


def ist_now() -> datetime:
    return datetime.now(ZoneInfo(TZ))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="replay a past IST date (YYYY-MM-DD); skips the time gate and readiness wait")
    args = ap.parse_args()
    now = ist_now()
    if args.date:
        today = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        today = now.date()
        if (now.hour, now.minute) < READY_AT:
            # Early or manual run: the 09:20 bar is not closed. Print nothing that notify.py would send.
            print(f"TOO_EARLY {now:%H:%M} IST (scan runs on the 09:20 bar after {READY_AT[0]:02d}:{READY_AT[1]:02d})")
            return
    try:
        names = load_universe(today)
        tickers = [f"{name}.NS" for name in names]
        frames = download_frames(tickers)
        for attempt in range(READY_RETRIES + 1):
            if args.date:
                break
            have = [f for f in frames.values() if (np.asarray(f.index.date) == today).any()]
            ready = sum(bar_final(f, today) for f in have)
            share = ready / len(have) if have else 0.0
            logging.warning("09:20 bar final for %d/%d tickers (%.0f%%)", ready, len(have), share * 100)
            if have and share >= READY_SHARE:
                break
            if attempt == READY_RETRIES:
                logging.error("09:20 bar still not final after %d retries", READY_RETRIES)
                print("DATA_NOT_READY")
                sys.exit(1)
            time.sleep(READY_SLEEP)
            frames = download_frames(tickers)
    except SystemExit:
        raise
    except Exception as exc:
        # Do not print NO_SIGNALS here: a failed setup must not become a 'No trades' message.
        logging.error("scanner setup failed: %s", exc)
        print("SCAN_ERROR")
        sys.exit(1)

    signals = []
    for ticker in tickers:
        frame = frames.get(ticker)
        if frame is None:
            continue
        if not args.date and not bar_final(frame, today):
            continue  # this name's 09:20 bar may still be forming
        try:
            signal = scan_symbol(ticker, frame, today)
            if signal is not None:
                signals.append(signal)
        except Exception as exc:
            logging.warning("scan failed for %s: %s", ticker, exc)

    if not signals:
        print("NO_SIGNALS")
        return
    for signal in signals:
        print(json.dumps(signal, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
