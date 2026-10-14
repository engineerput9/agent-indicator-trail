#!/usr/bin/env python3
"""Replay the cached 1.5% carry entries with a stepped trailing ladder."""
from __future__ import annotations
import json, math, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import backtest as bt
import run_pct_targets as pct

ROOT = Path('/workspace/backtest')
INITIAL = bt.INITIAL
RISK_PCT = bt.RISK_PCT
TARGET_PCT = 0.015
LOCK_PCT = 0.015
RUNG_STEP = 0.001


def collect_baseline(symbol: str, pre: dict) -> dict:
    """Exact no-flatten carry path, also retaining every entry record."""
    high, low, close = pre['high'], pre['low'], pre['close']
    times, n = pre['times'], pre['n']
    in_sess, by_cut, day_ids = pre['in_sess'], pre['by_cut'], pre['day_ids']
    atr, swing_low, swing_high = pre['atr'], pre['swing_low'], pre['swing_high']
    long_flip, short_flip, scan_side = pre['long_flip'], pre['short_flip'], pre['scan_side']
    cash = INITIAL
    pos = 0; entry_price = 0.0; entry_i = -1; qty = 0.0; risk_pts = 0.0
    act_sl = np.nan; act_tp = np.nan; signal_i = -1
    took = False; printed = False; pend_side = 0
    entries = []; closed = []

    def close_trade(i: int, exit_price: float, reason: str):
        nonlocal cash, pos, qty
        if pos == 1:
            pnl = (exit_price-entry_price)*qty; signed=(exit_price-entry_price)/entry_price*100.0; side='long'
            r=(exit_price-entry_price)/risk_pts if risk_pts > 0 else np.nan
        else:
            pnl = (entry_price-exit_price)*qty; signed=(entry_price-exit_price)/entry_price*100.0; side='short'
            r=(entry_price-exit_price)/risk_pts if risk_pts > 0 else np.nan
        cash += pnl
        rec = {
            'symbol': symbol, 'side': side, 'entry_time': times[entry_i].isoformat(),
            'exit_time': times[i].isoformat(), 'entry': float(entry_price), 'exit': float(exit_price),
            'sl': float(act_sl), 'tp': float(act_tp), 'price_chg_pct': float(signed),
            'r_multiple': float(r), 'pnl': float(pnl), 'unrealized_pnl': None,
            'exit_reason': reason, 'entry_i': int(entry_i), 'exit_i': int(i),
            'qty': float(qty), 'risk_pts': float(risk_pts), 'initial_sl': float(entries[-1]['initial_sl']) if entries else float(act_sl),
        }
        closed.append(rec); pos=0; qty=0.0

    for i in range(n):
        new_day = i == 0 or day_ids[i] != day_ids[i-1]
        if new_day:
            printed=False; pend_side=0; took=False
        c=close[i]
        if pos != 0 and i > entry_i:
            if pos == 1:
                stop_hit = low[i] <= act_sl; target_hit = high[i] >= act_tp
            else:
                stop_hit = high[i] >= act_sl; target_hit = low[i] <= act_tp
            if stop_hit: close_trade(i, float(act_sl), 'stop')
            elif target_hit: close_trade(i, float(act_tp), 'target')
        if in_sess[i] and (not printed) and (long_flip[i] or short_flip[i]):
            printed=True; is_long=bool(long_flip[i])
            if by_cut[i] and pct.levels_from_fill(is_long, float(c), float(atr[i]), float(swing_low[i]), float(swing_high[i]), TARGET_PCT) is not None:
                pend_side=1 if is_long else -1; signal_i=i
        if pos == 0 and in_sess[i] and by_cut[i] and pend_side != 0 and (not took) and np.isfinite(c):
            is_long=pend_side == 1
            lv=pct.levels_from_fill(is_long, float(c), float(atr[i]), float(swing_low[i]), float(swing_high[i]), TARGET_PCT)
            if int(scan_side[i]) == pend_side and lv is not None:
                sl,tp,dist=lv; q=(cash*(RISK_PCT/100.0))/dist if dist > 0 else 0.0
                if q > 0 and np.isfinite(q):
                    pos=pend_side; entry_price=float(c); entry_i=i; qty=float(q); risk_pts=float(dist)
                    act_sl=float(sl); act_tp=float(tp); took=True
                    entry_rec={'symbol':symbol,'side':'long' if is_long else 'short','entry_time':times[i].isoformat(),
                               'entry':float(c),'initial_sl':float(sl),'target':float(tp),'risk_pts':float(dist),
                               'qty':float(q),'entry_i':int(i),'signal_i':int(signal_i if signal_i >= 0 else i)}
                    entries.append(entry_rec)
                    if signal_i < 0: signal_i=i
    opens=[]
    if pos != 0:
        last_i=n-1; last_close=float(close[last_i]); side='long' if pos==1 else 'short'
        unreal=(last_close-entry_price)*qty if pos==1 else (entry_price-last_close)*qty
        signed=(last_close-entry_price)/entry_price*100.0 if pos==1 else (entry_price-last_close)/entry_price*100.0
        opens.append({'symbol':symbol,'side':side,'entry_time':times[entry_i].isoformat(),'exit_time':times[last_i].isoformat(),
                      'entry':float(entry_price),'exit':last_close,'sl':float(act_sl),'tp':float(act_tp),
                      'price_chg_pct':float(signed),'r_multiple':float(unreal/(risk_pts*qty)) if risk_pts>0 and qty>0 else None,
                      'pnl':None,'unrealized_pnl':float(unreal),'exit_reason':'open','entry_i':int(entry_i),'exit_i':int(last_i),
                      'qty':float(qty),'risk_pts':float(risk_pts),'initial_sl':float(act_sl)})
    return {'entries':entries,'closed':closed,'opens':opens}


def rung_for_extreme(is_long: bool, entry: float, hi: float, lo: float) -> float:
    fav = (hi/entry - 1.0) if is_long else (1.0 - lo/entry)
    # Rungs are the 0.1%-spaced levels touched; floor avoids crediting a rung
    # that lies above the bar extreme. Tolerance handles decimal representation.
    if not np.isfinite(fav) or fav < LOCK_PCT - 1e-12:
        return 0.0
    k = int(math.floor(fav / RUNG_STEP + 1e-10))
    return round(k * RUNG_STEP * 100.0, 10)  # percent units: 1.5, 1.6, ...


def replay_entry(pre: dict, ent: dict, mode: str) -> dict:
    high, low, close, times = pre['high'], pre['low'], pre['close'], pre['times']
    is_long = ent['side'] == 'long'; entry=float(ent['entry']); entry_i=int(ent['entry_i'])
    qty=float(ent['qty']); initial_sl=float(ent['initial_sl']); target=float(ent['target'])
    active_sl=initial_sl; armed=False; active_rung=0.0; max_rung=0.0
    exit_i=None; exit_price=None; reason=None
    for i in range(entry_i+1, len(close)):
        # Best excursion is observable on every post-entry bar, including an
        # exit bar. Stop/target decisions precede any rung update on this bar.
        rung = rung_for_extreme(is_long, entry, float(high[i]), float(low[i]))
        if rung > max_rung: max_rung=rung
        if is_long:
            stop_hit = low[i] <= active_sl
            target_hit = (mode == 'fixed_1.5' and high[i] >= target)
        else:
            stop_hit = high[i] >= active_sl
            target_hit = (mode == 'fixed_1.5' and low[i] <= target)
        if stop_hit:
            exit_i=i; exit_price=float(active_sl); reason='stop'; break
        if target_hit:
            exit_i=i; exit_price=float(target); reason='target'; break
        # With no exit, arm/advance only for the next bar.
        if rung >= 1.5:
            if not armed:
                armed=True
                # The first bar can jump over several 0.1% rungs.  Its
                # furthest touched rung determines the stop that becomes
                # active on the next bar (1.5% is the special lock level).
                active_rung=max(1.5, rung)
                stop_pct=1.5 if active_rung <= 1.5 else active_rung-0.1
                stop_frac=stop_pct/100.0
                active_sl = entry*(1.0-stop_frac) if is_long else entry*(1.0+stop_frac)
            elif rung > active_rung + 0.05:
                active_rung=rung
                trail_pct=(active_rung-0.1)/100.0
                active_sl=entry*(1.0+trail_pct) if is_long else entry*(1.0-trail_pct)
    if exit_i is None:
        i=len(close)-1; exit_i=i; exit_price=float(close[i]); reason='open'
        unreal=(exit_price-entry)*qty if is_long else (entry-exit_price)*qty
        pnl=None
    else:
        pnl=(exit_price-entry)*qty if is_long else (entry-exit_price)*qty
        unreal=None
    signed=(exit_price-entry)/entry*100.0 if is_long else (entry-exit_price)/entry*100.0
    risk=float(ent['risk_pts'])
    r=((exit_price-entry)/risk if is_long else (entry-exit_price)/risk) if risk>0 else np.nan
    return {'symbol':ent['symbol'],'side':ent['side'],'entry_time':ent['entry_time'],'exit_time':times[exit_i].isoformat(),
            'entry':entry,'exit':float(exit_price),'sl':float(active_sl),'tp':target,'price_chg_pct':float(signed),
            'r_multiple':float(r),'pnl':None if pnl is None else float(pnl),
            'unrealized_pnl':None if unreal is None else float(unreal),'exit_reason':reason,
            'entry_i':entry_i,'exit_i':int(exit_i),'qty':qty,'risk_pts':risk,'initial_sl':initial_sl,
            'max_favorable_rung_pct':float(max_rung),'armed':bool(max_rung >= 1.5)}


def add_max_rung_baseline(pre: dict, rec: dict) -> dict:
    # Recompute the same best post-entry excursion for fixed-target records.
    high, low = pre['high'], pre['low']; entry=float(rec['entry']); is_long=rec['side']=='long'
    mx=0.0
    for i in range(int(rec['entry_i'])+1, int(rec['exit_i'])+1):
        mx=max(mx, rung_for_extreme(is_long, entry, float(high[i]), float(low[i])))
    rec=dict(rec); rec['max_favorable_rung_pct']=float(mx); rec['armed']=bool(mx >= 1.5)
    return rec


def summarize(records: list[dict]) -> dict:
    closed=[t for t in records if t['exit_reason'] != 'open']; opens=[t for t in records if t['exit_reason']=='open']
    pnls=np.array([t['pnl'] for t in closed], dtype=float)
    chg=np.array([t['price_chg_pct'] for t in closed], dtype=float)
    wins=int(np.sum(pnls>0)) if len(pnls) else 0; losses=int(np.sum(pnls<0)) if len(pnls) else 0
    gp=float(pnls[pnls>0].sum()) if wins else 0.0; gl=float(-pnls[pnls<0].sum()) if losses else 0.0
    holds=[(pd.Timestamp(t['exit_time'])-pd.Timestamp(t['entry_time'])).total_seconds()/60.0 for t in closed]
    dist=Counter()
    for t in records:
        r=t['max_favorable_rung_pct']
        key='<1.5' if r < 1.5 else f'{r:.1f}'
        dist[key]+=1
    def side(s): return float(sum(t['pnl'] for t in closed if t['side']==s))
    return {'closed':len(closed),'open':len(opens),'wins':wins,'losses':losses,
            'win_rate':(wins/len(closed)) if closed else None,'profit_factor':(gp/gl) if gl else None,
            'realized_pnl':float(pnls.sum()) if len(pnls) else 0.0,
            'unrealized_on_opens':float(sum(t['unrealized_pnl'] for t in opens)),
            'average_signed_price_chg_pct':float(np.mean(chg)) if len(chg) else None,
            'median_signed_price_chg_pct':float(np.median(chg)) if len(chg) else None,
            'median_hold_minutes':float(np.median(holds)) if holds else None,
            'max_favorable_rung_distribution':dict(sorted(dist.items(), key=lambda kv: (-1 if kv[0]=='<1.5' else float(kv[0])))),
            'long_vs_short_realized_pnl':{'long':side('long'),'short':side('short')},
            'exit_reason_counts':dict(Counter(t['exit_reason'] for t in closed)),
            'gross_profit':gp,'gross_loss':gl}


class Enc(json.JSONEncoder):
    def default(self,o):
        if isinstance(o,(np.floating,)):
            return float(o) if np.isfinite(o) else None
        if isinstance(o,(np.integer,)): return int(o)
        if isinstance(o,float): return o if math.isfinite(o) else None
        return super().default(o)


def main():
    bt._self_check(); symbols=pct.symbol_list()
    missing=[s for s in symbols if not bt.cache_path(s).exists()]
    if missing: raise SystemExit(f'missing cache files (no downloads allowed): {missing}')
    all_entries=[]; fixed=[]; ladder=[]; loaded=0
    for si,sym in enumerate(symbols):
        df=pct.load_symbol(sym)
        if df is None: continue
        loaded+=1; pre=pct.precompute(df); base=collect_baseline(sym,pre)
        all_entries.extend(base['entries'])
        for t in base['closed']+base['opens']:
            fixed.append(add_max_rung_baseline(pre,t))
        for ent in base['entries']:
            ladder.append(replay_entry(pre, ent, 'ladder'))
        if (si+1)%25==0: print(f'done {si+1}/{len(symbols)} entries={len(all_entries)}', flush=True)
    fixed.sort(key=lambda t:(t['symbol'],t['entry_time'],t['exit_time']))
    ladder.sort(key=lambda t:(t['symbol'],t['entry_time'],t['exit_time']))
    # Verify against the cached carry run: same 1.5% entry/exit population.
    cached=pd.read_csv(ROOT/'trades_carry.csv')
    c=cached[cached['variant']=='1.5pct']
    fixed_keys={(t['symbol'],t['side'],t['entry_time']) for t in fixed}
    cached_keys=set(zip(c['symbol'],c['side'],c['entry_time']))
    verification={'cached_closed_rows':int(len(c[c['exit_reason']!='open'])),'recomputed_closed':int(sum(t['exit_reason']!='open' for t in fixed)),
                  'cached_open_rows':int(len(c[c['exit_reason']=='open'])),'recomputed_open':int(sum(t['exit_reason']=='open' for t in fixed)),
                  'entry_key_match':bool(fixed_keys==cached_keys),'symbols_loaded':loaded,'symbols_requested':len(symbols)}
    if not verification['entry_key_match'] or verification['cached_closed_rows'] != verification['recomputed_closed']:
        raise SystemExit(f'carry verification failed: {verification}')
    stats_fixed=summarize(fixed); stats_ladder=summarize(ladder)
    summary={'data':{'source':'cached CSVs in /workspace/backtest/data','downloads':0,'symbols_requested':len(symbols),'symbols_loaded':loaded},
      'verification_against_existing_carry_1.5pct':verification,
      'rules':{'entries':'exact same 1.5% overnight carry entries; both sides; all signal times accepted by carry path',
               'stop':'same swing/ATR initial stop as carry, widened to 0.8% of fill if tighter, capped at 2.0%',
               'target_baseline':'fixed 1.5% of fill; no session flatten; adverse stop first; skip entry bar',
               'ladder':'arm at 1.5%; locked stop applies next bar; each 0.1% rung moves stop to rung minus 0.1%, next-bar activation; no final target',
               'risk':'same carry sizing: 1% of each independent symbol account cash equity at entry, starting at 100000; fractional quantity; zero costs',
               'max_rung':'post-entry bars through exit bar, observationally bucketed to the furthest 0.1% rung touched'},
      'variants':{'ladder':stats_ladder,'fixed_1.5':stats_fixed},
      'pnl_difference_ladder_minus_fixed_1.5':float(stats_ladder['realized_pnl']-stats_fixed['realized_pnl'])}
    rows=[]
    for variant, records in [('ladder',ladder),('fixed_1.5',fixed)]:
        for t in records:
            rows.append({'variant':variant,'symbol':t['symbol'],'side':t['side'],'entry_time':t['entry_time'],'exit_time':t['exit_time'],
              'entry':t['entry'],'exit':t['exit'],'initial_sl':t['initial_sl'],'effective_sl_at_exit':t['sl'],'target_1.5':t['tp'],
              'price_chg_pct':t['price_chg_pct'],'r_multiple':t['r_multiple'],'pnl':t['pnl'],'unrealized_pnl':t['unrealized_pnl'],
              'max_favorable_rung_pct':t['max_favorable_rung_pct'],'exit_reason':t['exit_reason']})
    cols=['variant','symbol','side','entry_time','exit_time','entry','exit','initial_sl','effective_sl_at_exit','target_1.5','price_chg_pct','r_multiple','pnl','unrealized_pnl','max_favorable_rung_pct','exit_reason']
    pd.DataFrame(rows,columns=cols).sort_values(['variant','symbol','entry_time','exit_time']).to_csv(ROOT/'trades_ladder.csv',index=False)
    with open(ROOT/'results_ladder.json','w') as f: json.dump(summary,f,indent=2,cls=Enc)
    print(json.dumps(summary,indent=2,cls=Enc))

if __name__=='__main__': main()
