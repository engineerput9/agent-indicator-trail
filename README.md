# Morning F&O scanner

This public repository runs a standalone `scan.py` weekday morning scanner. It
uses the public NSEFO instrument CSV and selects unique equity underlyings from
`FUTSTK` and `OPTSTK` contracts whose expiry is still current, excluding
`NSETEST` names. Each name is downloaded from Yahoo Finance as `SYMBOL.NS`
with `yfinance`, 5-minute bars, and `period="60d"`. Indices are not requested.

The GitHub Actions job runs at **09:25 IST** (`55 3 * * 1-5` UTC) and can also
be started with `workflow_dispatch`. Set the repository Actions secrets
`TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`; the workflow does not print the
bot token. The repository is public, so secrets belong only in GitHub Actions
Secrets, never in source files.

## Locked morning rules

* EMA/RMA/SMA calculations use TradingView-style SMA seeding, matching the
  backtest indicator math. The range filter is `per=100`, `mult=3`.
* Only the first range-filter flip during today's IST session is considered,
  and it must be on a bar no later than 09:20 IST.
* The opening scan is the first in-session bar with positive volume. A long
  scan requires EMA 9 > EMA 21, close above EMA 9/21/55, ADX >= 20, +DI > -DI,
  RSI in (50, 75), rising positive MACD histogram, volume > 1.2 times SMA20,
  and close above the session VWAP. The short scan is the mirrored rule with
  RSI in (25, 50) and close below VWAP.
* A signal is emitted only when the scan side matches the flip side. There is
  at most one signal per symbol per IST day.
* Entry is the eligible bar close. The stop is 0.5% from entry; the lock price
  is 1.5% in favor. After lock, the management note is a 0.1% trail. There is
  no fixed take-profit.

Each signal is one compact JSON line containing `symbol`, `side`,
`signal_time`, `entry`, `sl`, and `lock`. With no signals, the program prints
`NO_SIGNALS`.

This scanner is not connected to TradingView and TradingView is not involved
in its data download, indicator calculations, or notifications.
