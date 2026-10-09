"""Send scan-output.txt to every configured Telegram bot, in labeled form."""
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

# Chat ids are not secrets; keep them as fallbacks so a local backup never
# fails just because TELEGRAM_CHAT_ID is unset. Tokens stay in the env only.
DEFAULT_CHATS = {
    "": "803261082",
    "_2": "985694441",
}


def targets():
    out = []
    for suffix in ("", "_2"):
        token = os.environ.get(f"TELEGRAM_BOT_TOKEN{suffix}", "").strip()
        chat = os.environ.get(f"TELEGRAM_CHAT_ID{suffix}", "").strip() or DEFAULT_CHATS[suffix]
        if token and chat:
            out.append((token, chat))
    return out


def send(token, chat, text):
    payload = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
    req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage", data=payload)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return bool(json.loads(resp.read().decode()).get("ok"))
    except urllib.error.HTTPError as exc:
        print(f"send failed: {exc.code}", file=sys.stderr)
        return False


def fmt(sig):
    side = "Buy" if sig["side"] == "long" else "Sell"
    return (
        f"Stock: {sig['symbol'].replace('.NS', '')}\n"
        f"Side: {side}\n"
        f"Entry: {sig['entry']:.2f}\n"
        f"Target: {sig['lock']:.2f}\n"
        f"S/L: {sig['sl']:.2f}\n"
        f"Time: {sig['signal_time'][11:16]} IST"
    )


READY_AT = (9, 25)  # no message of any kind before 09:25 IST (the 09:20 bar closes at 09:25)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "scan-output.txt"
    dry_run = os.environ.get("NOTIFY_DRY_RUN", "").strip() == "1"
    lines = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
    signals = [json.loads(l) for l in lines if l.startswith("{")]
    now = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    if any(l.startswith("TOO_EARLY") for l in lines) or (now.hour, now.minute) < READY_AT:
        print(f"{now:%H:%M} IST: run is before 09:25 or scan reported TOO_EARLY, nothing sent")
        return
    if signals:
        messages = [fmt(s) for s in signals]
    elif "NO_SIGNALS" in lines:
        messages = ["No trades at 09:20 IST"]
    else:
        # Scanner error or unknown output: never turn it into a 'No trades' message.
        sys.exit("scan output has no signals and no NO_SIGNALS marker: nothing sent")
    if dry_run:
        print("NOTIFY_DRY_RUN=1, not sending:")
        print("\n---\n".join(messages))
        return
    dests = targets()
    if not dests:
        sys.exit("no Telegram targets configured")
    failures = 0
    for token, chat in dests:
        for text in messages:
            if not send(token, chat, text):
                failures += 1
    if failures:
        sys.exit(f"{failures} Telegram sends failed")
    print(f"sent {len(messages)} message(s) to {len(dests)} bot(s)")


if __name__ == "__main__":
    main()
