# Morning F&O scanner

This public repository runs a standalone `scan.py` weekday morning scanner. It
uses the public NSEFO instrument CSV and selects unique equity underlyings from
`FUTSTK` and `OPTSTK` contracts whose expiry is still current, excluding
`NSETEST` names. Each name is downloaded from Yahoo Finance as `SYMBOL.NS`
with `yfinance`, 5-minute bars, and `period="60d"`. Indices are not requested.

The GitHub Actions job runs at **09:26 IST** (`56 3 * * 1-5` UTC), with
09:35/09:45 backups and a watchdog (09:30-11:30) that re-dispatches until one
run succeeds. It can also be started with `workflow_dispatch`. A run that
starts before 09:25 IST sends nothing (not even "No trades"); a second
successful run on the same day is skipped. If Yahoo has not finalised the
09:20 bar yet, `scan.py` re-downloads every 30 s for about 3 minutes and fails
(no message) rather than scanning a partial bar. `NOTIFY_DRY_RUN=1` prints the
messages instead of sending, and `python scan.py --date YYYY-MM-DD` replays a
past day. Set the repository Actions secrets
`TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`; the workflow does not print the
bot token. The repository is public, so secrets belong only in GitHub Actions
Secrets, never in source files.

## Locked morning rules

* EMA/RMA/SMA calculations use TradingView-style SMA seeding, matching the
  backtest indicator math. The range filter is `per=100`, `mult=3`.
* Only the first range-filter flip during today's IST session is considered,
  and it must be on the 09:15 or 09:20 bar, on the same side as the scan.
* The opening scan always runs on the **09:20 bar** (closes 09:25 IST). The
  09:15 bar's volume is set to 0 on every day (Yahoo zeroes it later anyway),
  so live runs and backtests use the same bar; if the 09:20 bar has no volume
  the scan is blank. A long
  scan requires EMA 9 > EMA 21, close above EMA 9/21/55, ADX >= 20, +DI > -DI,
  RSI in (50, 75), rising positive MACD histogram, volume > 1.2 times SMA20,
  and close above the session VWAP. The short scan is the mirrored rule with
  RSI in (25, 50) and close below VWAP.
* A signal is emitted only when the scan side matches the flip side. There is
  at most one signal per symbol per IST day.
* Entry is the 09:20 bar close (signal sent about 09:26). The stop is 0.5%
  from entry; the lock price is 1.5% in favor. Once +1.5% is reached the stop
  moves to entry +1.5% (long) / -1.5% (short), on the profit side, then trails
  in 0.1% steps. There is no fixed take-profit.

Each signal is one compact JSON line containing `symbol`, `side`,
`signal_time` (flip bar), `entry_time` (09:20 entry bar), `entry`, `sl`, and
`lock`. With no signals, the program prints
`NO_SIGNALS`.

This scanner is not connected to TradingView and TradingView is not involved
in its data download, indicator calculations, or notifications.
