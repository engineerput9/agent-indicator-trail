#!/usr/bin/env python3
"""Live morning signals on FYERS data: 09:20 bar rule A (same engine as trades_fyers_0920A.csv).

Rule A: every 09:15 bar's volume zeroed; opening scan on today's 09:20 bar (blank if it has no volume);
the day's first Range Filter flip on the 09:15 or 09:20 bar, same side as the scan; entry = 09:20 bar close.
Engine: /workspace/backtest run_pct_targets.precompute -> run_ladder.collect_baseline (unchanged).
Data: cached Fyers 5m history (/workspace/fyers/data, from 1 May 2026) + bars fetched from Fyers at run time.

Usage:
  fyers_morning.py                 live: refuses before 09:25 IST, waits for the 09:20 bar, sends to Telegram
  fyers_morning.py --dry-run       same, but never sends, never writes the marker, never sets the GitHub variable
  fyers_morning.py --date D        replay day D from the cache only (implies --dry-run, no gate, no fetch)
Exit codes: 0 sent / dry-run ok / already sent, 2 too early, 3 data not ready, 4 login/setup error, 5 send failure, 6 past 09:27:30 send deadline, 7 another live run active.
Secrets: Fyers and Telegram credentials come from env only and are never printed or written.
"""
import argparse, fcntl, json, logging, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "engine")); sys.path.insert(0, os.environ.get("FYERS_LOGIN_DIR", "/workspace/fyers"))
import backtest as bt, run_pct_targets as pct, run_ladder as ladder

TZ = ZoneInfo("Asia/Kolkata")
DATA = os.environ.get("FYERS_CACHE_DIR", "/workspace/fyers/data/")   # empty/missing on GitHub: history fetched live
MARKER_DIR = os.environ.get("MARKER_DIR", "/workspace/backtest/eod/")
REPO = "engineerput9/agent-indicator-trail"
GH_VAR = "LAST_FYERS_SEND_DATE"
NOTIFY_DIR = os.environ.get("NOTIFY_DIR", "/workspace/agent-indicator-trail")
ON_GH = os.environ.get("GITHUB_ACTIONS") == "true"
HIST_DAYS = 45   # warm-up history when there is no cache (tested: 20-60 days give identical signals)   # reuse the repo's Telegram format + targets
READY_AT = (9, 25, 0)
READY_SHARE = 0.90
WAIT_MAX_S, WAIT_STEP_S = 120, 10
STOP_PCT, LOCK_PCT = 0.005, 0.015
# Entry filters (approved 12 Oct, live from 13 Oct; see /workspace/backtest/winners_study).
# Set either to None (or env FILTERS_OFF=1) to disable.
GAP_MAX = 0.003     # skip if signed opening gap in trade direction > +0.3%
BODY_MAX = 0.91     # skip if the 09:15 5m candle is in trade direction with body > 91% of its range
VWAP_MIN = 0.0015   # skip if signed VWAP distance (entry/VWAP-1, x -1 for sells) <= +0.15%
if os.environ.get("FILTERS_OFF") == "1": GAP_MAX = VWAP_MIN = BODY_MAX = None
from datetime import time as dtime
SEND_DEADLINE = dtime(*[int(x) for x in os.environ.get("SEND_DEADLINE", "09:27:30").split(":")])  # GitHub run sets its own
log = logging.getLogger("fyers_morning")


def universe():
    if os.path.exists(DATA + "_download_report.json"):
        rep = json.load(open(DATA + "_download_report.json"))
        return sorted(k for k, v in rep.items() if v["status"] == "ok" and k != "NIFTY50")
    return json.load(open(os.path.join(HERE, "universe.json")))


def fyers_symbol(s): return "NSE:" + s.replace("MANDM", "M&M").replace("GVTANDD", "GVT&D") + "-EQ"
def yahoo_symbol(s): return s.replace("MANDM", "M&M").replace("GVTANDD", "GVT&D") + ".NS"


def load_cache(s):
    if not os.path.exists(DATA + f"{s}.csv"):
        return candles_to_df([])
    df = pd.read_csv(DATA + f"{s}.csv")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("Asia/Kolkata")
    return df.set_index("datetime")


def candles_to_df(c):
    df = pd.DataFrame(c, columns=["ts", "open", "high", "low", "close", "volume"]).drop_duplicates("ts").sort_values("ts")
    df["datetime"] = pd.to_datetime(df.ts, unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
    return df.set_index("datetime")[["open", "high", "low", "close", "volume"]]


def is_bar(idx, d, hh, mm):
    return (np.asarray(idx.date) == d) & (idx.hour == hh) & (idx.minute == mm)


def scan_one(args):
    """Rule A on one symbol for day d. df must hold history up to (at least) d's 09:20 bar."""
    sym, df, d = args
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    t = df.index
    late = (np.asarray(t.date) > d) | ((np.asarray(t.date) == d) & ((t.hour > 9) | ((t.hour == 9) & (t.minute > 20))))
    df = bt.normalize_ohlcv(df[~late].copy())
    if not is_bar(df.index, d, 9, 20).any():
        return sym, None, "no 09:20 bar"
    df.loc[(df.index.hour == 9) & (df.index.minute == 15), "volume"] = 0.0
    pre = pct.precompute(df)
    tt = pre["times"]; hh = np.asarray(tt.hour); mm = np.asarray(tt.minute); vol = df.volume.to_numpy(); days = pre["day_ids"]
    for dd in np.unique(days):   # identical to run_fyers_books.py (book 0920A)
        idx = np.where(days == dd)[0]
        sb = [i for i in idx if pre["in_sess"][i] and np.isfinite(vol[i]) and vol[i] > 0]
        if not sb or (hh[sb[0]], mm[sb[0]]) != (9, 20):
            pre["scan_side"][idx] = 0
    for ent in ladder.collect_baseline(sym, pre)["entries"]:
        ei = ent["entry_i"]
        if tt[ei].date() != d:
            continue
        assert (hh[ei], mm[ei]) == (9, 20)
        e = float(ent["entry"]); lng = ent["side"] == "long"
        return sym, {"symbol": yahoo_symbol(sym), "fyers_symbol": fyers_symbol(sym), "side": ent["side"],
                     "signal_time": tt[ent["signal_i"]].isoformat(), "flip_bar": tt[ent["signal_i"]].strftime("%H:%M"), "entry_time": tt[ei].isoformat(), "entry": e,
                     "sl": e * (1 - STOP_PCT) if lng else e * (1 + STOP_PCT),
                     "lock": e * (1 + LOCK_PCT) if lng else e * (1 - LOCK_PCT), "source": "fyers"}, "ok"
    return sym, None, "ok"


def entry_features(df, d, side, entry):
    """Same definitions as winners_study/feat.py: session bars 09:15-15:25 (raw Fyers volume, 09:15 included);
    prior close = previous session's last bar close; gap = today 09:15 open / prior close - 1 (x -1 for sells);
    VWAP = cumulative typical-price x volume / volume from today's 09:15 through the 09:20 bar."""
    df = df.sort_index(); df = df[~df.index.duplicated(keep="last")]
    t = df.index; dates = np.asarray(t.date)
    sess = (t.hour * 60 + t.minute >= 555) & (t.hour * 60 + t.minute < 930)
    prev = df[sess & (dates < d)]; today = df[sess & (dates == d) & (t.hour * 60 + t.minute <= 560)]
    if prev.empty or today.empty: return None, None
    s = 1 if side == "long" else -1
    gap = s * (float(today.open.iloc[0]) / float(prev.close.iloc[-1]) - 1)
    tp = (today.high + today.low + today.close) / 3; vv = float(today.volume.sum())
    vwap = float((tp * today.volume).sum()) / vv if vv > 0 else None
    b = today.iloc[0]; rng = float(b.high - b.low)   # 09:15 5m bar
    body = s * float(b.close - b.open) / rng if rng > 0 else None
    return gap, (s * (entry / vwap - 1) if vwap else None), body   # signed: + = entry on the trade side of VWAP


def apply_filters(sigs, frames, d):
    keep, sk_gap, sk_vwap, sk_body = [], [], [], []
    for sg in sigs:
        sym = sg["fyers_symbol"][4:-3].replace("M&M", "MANDM").replace("GVT&D", "GVTANDD")
        gap, vd, body = entry_features(frames[sym], d, sg["side"], sg["entry"])
        sg = dict(sg, gap_signed=gap, vwap_dist=vd, b15_body=body)
        if GAP_MAX is not None and gap is not None and gap > GAP_MAX: sk_gap.append(sg)
        elif VWAP_MIN is not None and vd is not None and vd <= VWAP_MIN: sk_vwap.append(sg)
        elif BODY_MAX is not None and body is not None and body > BODY_MAX: sk_body.append(sg)
        else: keep.append(sg)
    for k, lst in (("gap", sk_gap), ("vwap", sk_vwap), ("body", sk_body)):
        for sg in lst: log.info("skipped_%s %s %s gap=%.4f vwap_dist=%s", k, sg["symbol"], sg["side"], sg["gap_signed"], None if sg["vwap_dist"] is None else round(sg["vwap_dist"], 5))
    return keep, sk_gap, sk_vwap, sk_body


def rvol(df, d, n=14):
    """(09:15 + 09:20 5m volume today) / mean of the same window over the prior n sessions (as in noise_tests)."""
    df = df.sort_index(); df = df[~df.index.duplicated(keep="last")]
    t = df.index; dates = np.asarray(t.date)
    win = df.volume[(t.hour == 9) & (t.minute <= 20) & (t.minute >= 15)]
    per = win.groupby(np.asarray(win.index.date)).sum()
    if d not in per.index: return None
    prev = per[per.index < d].tail(n)
    if len(prev) < 5 or prev.mean() <= 0: return None
    return float(per[d] / prev.mean())


def add_priority(sigs, frames, d):
    try:
        for sg in sigs:
            sym = sg["fyers_symbol"][4:-3].replace("M&M", "MANDM").replace("GVT&D", "GVTANDD")
            sg["rvol"] = rvol(frames[sym], d)
        if sigs and all(sg["rvol"] is not None for sg in sigs):
            sigs.sort(key=lambda sg: -sg["rvol"])
            for k, sg in enumerate(sigs, 1): sg["priority"] = k
        else:
            for sg in sigs: sg["rvol"] = None; sg["priority"] = None
    except Exception as exc:
        log.warning("RVOL failed, keeping original order: %s", type(exc).__name__)
        for sg in sigs: sg["rvol"] = None; sg["priority"] = None
    if len(sigs) >= 4 and sigs[0].get("priority"): sigs[0]["top3_note"] = True
    return sigs


def run_scan(frames, d):
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 2)) as ex:
        res = list(ex.map(scan_one, [(s, f, d) for s, f in frames.items()], chunksize=4))
    sigs = [r for _, r, _ in res if r]
    sigs.sort(key=lambda r: r["symbol"])
    return sigs, {s: st for s, _, st in res}


class Fetcher:
    """Fyers history with a simple rate limiter (Fyers allows ~10/s and ~200/min)."""
    def __init__(self, fy): self.fy, self.calls = fy, []
    def _throttle(self):
        now = time.time(); self.calls = [c for c in self.calls if now - c < 60]
        if len(self.calls) >= 180: time.sleep(60 - (now - self.calls[0]) + 0.1)
        self.calls.append(time.time()); time.sleep(0.12)
    def history(self, s, a, b):
        from fyers_login import scrub
        r = None
        for attempt in range(4):
            self._throttle()
            try:
                r = self.fy.history({"symbol": fyers_symbol(s), "resolution": "5", "date_format": "1",
                                     "range_from": a.isoformat(), "range_to": b.isoformat(), "cont_flag": "1"})
            except Exception as exc:
                r = {"s": "error", "message": type(exc).__name__}
            if r.get("s") == "ok": return candles_to_df(r.get("candles") or [])
            if r.get("s") == "no_data": return candles_to_df([])
            time.sleep(1 + attempt * 2)
        log.warning("history failed %s: %s", s, scrub(r)[:150])
        return None


def update_cache(sym, cache, fresh, d):
    """Append completed days (< d) to the on-disk cache so warm-up history keeps growing."""
    past = fresh[np.asarray(fresh.index.date) < d]
    if past.empty: return
    merged = pd.concat([cache, past]); merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    if len(merged) > len(cache):
        tmp = DATA + f".{sym}.tmp"
        merged.reset_index().to_csv(tmp, index=False); os.replace(tmp, DATA + f"{sym}.csv")


def gh_set_var(d):
    if ON_GH:   # GITHUB_TOKEN cannot write repo variables: the workflow commits state/last_send.json instead
        return True, "state file (workflow commits it)"
    try:
        r = subprocess.run(["gh", "variable", "set", GH_VAR, "--body", d.isoformat(), "-R", REPO], capture_output=True, text=True, timeout=60)
        return r.returncode == 0, (r.stderr or r.stdout).strip()[:200]
    except Exception as exc:
        return False, type(exc).__name__


def marker_path(d, mdir): return os.path.join(mdir, f"signals_{d.isoformat()}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--marker-dir", default=MARKER_DIR, help="testing only")
    ap.add_argument("--now", help="testing only: pretend IST wall clock HH:MM[:SS] for the gates")
    ap.add_argument("--fetch", action="store_true", help="with --date: log in and fetch day D from Fyers instead of the cache (test)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    replay = bool(a.date) and not a.fetch; dry = a.dry_run or bool(a.date)
    now = datetime.now(TZ)
    if a.now:
        h, m, *s = [int(x) for x in a.now.split(":")]; now = now.replace(hour=h, minute=m, second=(s or [0])[0])
    d = date.fromisoformat(a.date) if a.date else now.date()
    mpath = marker_path(d, a.marker_dir)

    if not a.date:
        if now.weekday() >= 5:
            print(f"WEEKEND {d}: nothing to do"); return 0
        if (now.hour, now.minute, now.second) < READY_AT:
            print(f"TOO_EARLY {now:%H:%M:%S} IST: the 09:20 bar closes at 09:25, nothing scanned or sent"); return 2
        if not dry and os.path.exists(mpath):
            print(f"ALREADY_SENT {mpath} exists: refusing to send twice"); return 0
    if not a.date:
        lockf = open(os.path.join(os.environ.get("RUNNER_TEMP", "/workspace/live"), ".fyers_morning.lock"), "w")
        try:
            fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("ANOTHER_RUN_ACTIVE: exiting"); return 7

    syms = universe()
    caches = {s: load_cache(s) for s in syms}
    if replay:
        frames = {s: c[np.asarray(c.index.date) <= d] for s, c in caches.items()}
    else:
        try:
            from fyers_login import get_fyers, LoginError
            fy = None
            for attempt in range(3):
                try:
                    fy = get_fyers(force=attempt > 0); break
                except Exception as exc:
                    log.error("login attempt %d failed: %s", attempt + 1, str(exc)[:200]); time.sleep(5)
            if fy is None:
                print("LOGIN_ERROR"); return 4
            fe = Fetcher(fy)
            known = [c.index.max().date() for c in caches.values() if len(c)]
            start = (min(known) + timedelta(days=1)) if known else d - timedelta(days=HIST_DAYS)
            start = min(start, d)
            # probe a few liquid names until the 09:20 candle is out (bar closes 09:25:00)
            probe = [s for s in ("RELIANCE", "HDFCBANK", "ICICIPRULI", "INFY", "SBIN", "TCS", "ITC", "AXISBANK", "LT", "BHARTIARTL") if s in caches]
            t0 = time.time()
            while True:
                got = [fe.history(s, d, d) for s in probe]
                okn = sum(1 for g in got if g is not None and is_bar(g.index, d, 9, 20).any())
                log.info("probe: 09:20 bar present for %d/%d (bars: %s)", okn, len(probe), [None if g is None else len(g) for g in got][:3])
                if okn >= READY_SHARE * len(probe): break
                if time.time() - t0 > WAIT_MAX_S:
                    print("DATA_NOT_READY (probe)"); return 3
                time.sleep(WAIT_STEP_S)
            fresh = {}
            for s in syms:
                fresh[s] = fe.history(s, start, d)
            t1 = time.time()
            while True:
                missing = [s for s in syms if fresh.get(s) is None or not is_bar(fresh[s].index, d, 9, 20).any()]
                share = 1 - len(missing) / len(syms)
                log.info("09:20 bar present for %d/%d symbols (%.1f%%)", len(syms) - len(missing), len(syms), 100 * share)
                if not missing or time.time() - t1 > WAIT_MAX_S: break
                time.sleep(WAIT_STEP_S)
                for s in missing:
                    g = fe.history(s, start, d)
                    if g is not None: fresh[s] = g
            if share < READY_SHARE:
                print(f"DATA_NOT_READY only {share:.0%} of symbols have the 09:20 bar"); return 3
            frames = {}
            for s in syms:
                f = fresh.get(s)
                if f is None or not is_bar(f.index, d, 9, 20).any(): continue
                c = caches[s]
                frames[s] = pd.concat([c[np.asarray(c.index.date) < start], f])
                if not dry and len(c):
                    try: update_cache(s, c, f, d)
                    except Exception as exc: log.warning("cache update failed %s: %s", s, exc)
        except Exception as exc:
            log.exception("setup failed"); print("SCAN_ERROR", type(exc).__name__); return 4

    sigs, status = run_scan(frames, d)
    errs = {s: v for s, v in status.items() if v not in ("ok", "no 09:20 bar")}
    log.info("scanned %d symbols, %d signals, %d errors", len(frames), len(sigs), len(errs))
    sigs, sk_gap, sk_vwap, sk_body = apply_filters(sigs, frames, d)
    log.info("filters GAP_MAX=%s VWAP_MIN=%s BODY_MAX=%s: kept %d, skipped_gap %d, skipped_vwap %d, skipped_body %d", GAP_MAX, VWAP_MIN, BODY_MAX, len(sigs), len(sk_gap), len(sk_vwap), len(sk_body))
    for sg in sk_gap: print("SKIPPED_GAP", sg["symbol"], sg["side"], f"gap={sg['gap_signed']:.4%}")
    for sg in sk_vwap: print("SKIPPED_VWAP", sg["symbol"], sg["side"], f"vwap_dist={sg['vwap_dist']:.4%}")
    for sg in sk_body: print("SKIPPED_BODY", sg["symbol"], sg["side"], f"b15_body={sg['b15_body']:.3f}")
    sigs = add_priority(sigs, frames, d)
    log.info("priority: %s", [(sg["symbol"], sg.get("priority"), None if sg.get("rvol") is None else round(sg["rvol"], 2)) for sg in sigs])
    sys.path.insert(0, NOTIFY_DIR); import notify
    messages = [notify.fmt(s) for s in sigs] or ["No trades at 09:20 IST"]
    for s in sigs: print(json.dumps(s, separators=(",", ":")))
    if not sigs: print("NO_SIGNALS")
    if dry:
        print(f"DRY_RUN: {len(messages)} message(s) not sent:"); print("\n---\n".join(messages)); return 0

    # Hand-off with the GitHub Yahoo fallback: it checks LAST_FYERS_SEND_DATE at/after 09:29 IST.
    # Never start sending so late that the variable might not be set by then.
    if datetime.now(TZ).time() > SEND_DEADLINE:
        print(f"TOO_LATE {datetime.now(TZ):%H:%M:%S} IST: past {SEND_DEADLINE}, leaving today to the Yahoo fallback"); return 6
    # --- live send: marker first (refuse twice even if we crash mid-send), then Telegram, then GitHub variable
    if os.path.exists(mpath):
        print(f"ALREADY_SENT {mpath}"); return 0
    rec = {"date": d.isoformat(), "source": "fyers", "rule": "09:20 bar rule A", "status": "sending",
           "started_at": datetime.now(TZ).isoformat(), "signals": sigs, "symbols_scanned": len(frames),
           "filters": {"GAP_MAX": GAP_MAX, "VWAP_MIN": VWAP_MIN, "BODY_MAX": BODY_MAX}, "skipped_gap": sk_gap, "skipped_vwap": sk_vwap, "skipped_body": sk_body}
    fd = os.open(mpath, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as fh: json.dump(rec, fh, indent=1)
    dests = notify.targets(); okc = fail = 0
    for token, chat in dests:
        for text in messages:
            if notify.send(token, chat, text): okc += 1
            else: fail += 1
    rec.update(status="sent" if okc and not fail else ("partial" if okc else "failed"), sent_ok=okc, sent_failed=fail,
               targets=len(dests), finished_at=datetime.now(TZ).isoformat())
    if okc:
        ok, msg = gh_set_var(d); rec["github_var_set"] = ok
        if not ok: log.error("could not set %s: %s", GH_VAR, msg)
    else:
        os.replace(mpath, mpath + ".failed")   # nothing went out: let a retry (or the Yahoo fallback) send
    if okc:
        json.dump(rec, open(mpath, "w"), indent=1)
    else:
        json.dump(rec, open(mpath + ".failed", "w"), indent=1)
    print(f"{rec['status'].upper()}: {okc} ok, {fail} failed, {len(messages)} message(s) x {len(dests)} chat(s)")
    return 0 if okc and not fail else 5


if __name__ == "__main__":
    sys.exit(main())
