"""Headless TOTP login for Fyers API v3. Secrets read from env; never printed.
Usage: from fyers_login import get_fyers; fyers = get_fyers()
"""
import base64, datetime as dt, json, os, sys, time
from urllib.parse import urlparse, parse_qs
import pyotp, requests
from fyers_apiv3 import fyersModel

HERE = os.path.dirname(os.path.abspath(__file__))
TOKEN_FILE = os.path.join(HERE, ".token")
LOG_DIR = os.path.join(HERE, "logs"); os.makedirs(LOG_DIR, exist_ok=True)
REDIRECT = os.environ.get("FYERS_REDIRECT_URI", "https://127.0.0.1/")
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
UA = {"Accept": "application/json", "Content-Type": "application/json",
      "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}

def _env(k):
    v = os.environ.get(k, "").strip()
    if not v: raise SystemExit(f"missing env {k}")
    return v

def _secrets():
    return [os.environ.get(k, "").strip() for k in
            ("FYERS_APP_ID","FYERS_SECRET_ID","FYERS_CLIENT_ID","FYERS_PIN","FYERS_TOTP_KEY")]

def scrub(x):
    s = x if isinstance(x, str) else json.dumps(x, default=str)
    vals = [v for v in _secrets() if v]
    vals += [base64.b64encode(v.encode()).decode() for v in vals]
    a = os.environ.get("FYERS_APP_ID","").strip()
    if "-" in a: vals.append(a.rsplit("-",1)[0])
    for v in sorted(set(vals), key=len, reverse=True):
        if len(v) >= 3: s = s.replace(v, "***")
    # tokens / keys in responses
    import re
    s = re.sub(r'("?(access_token|refresh_token|request_key|auth_code|auth|Url|url|data)"?\s*[:=]\s*")[^"]{12,}"', r'\1<redacted>"', s)
    s = re.sub(r'eyJ[\w-]+\.[\w-]+\.[\w-]+', '<jwt>', s)
    return s

class LoginError(Exception):
    def __init__(self, step, resp): super().__init__(f"{step} failed: {scrub(resp)[:500]}"); self.step = step

def _fresh_totp(key):
    t = pyotp.TOTP(key.replace(" ", "").upper())
    rem = 30 - (time.time() % 30)
    if rem < 5: time.sleep(rem + 1)  # avoid rollover mid-request
    return t.now()

def _b64(s): return base64.b64encode(s.encode()).decode()

def _jget(r):
    try: return r.json()
    except Exception: return {"http": r.status_code, "text": r.text[:300]}

def login():
    app_id, secret = _env("FYERS_APP_ID"), _env("FYERS_SECRET_ID")
    fy_id, pin, totp_key = _env("FYERS_CLIENT_ID").upper(), _env("FYERS_PIN"), _env("FYERS_TOTP_KEY")
    s = requests.Session(); s.headers.update(UA)
    r1 = _jget(s.post("https://api-t2.fyers.in/vagator/v2/send_login_otp_v2", json={"fy_id": _b64(fy_id), "app_id": "2"}))
    if "request_key" not in r1: raise LoginError("1 send_login_otp (client id)", r1)
    r2 = None
    for attempt in range(2):
        r2 = _jget(s.post("https://api-t2.fyers.in/vagator/v2/verify_otp", json={"request_key": r1["request_key"], "otp": _fresh_totp(totp_key)}))
        if "request_key" in r2: break
        time.sleep(31 - time.time() % 30)
    if "request_key" not in r2: raise LoginError("2 verify_otp (TOTP key)", r2)
    r3 = _jget(s.post("https://api-t2.fyers.in/vagator/v2/verify_pin_v2", json={"request_key": r2["request_key"], "identity_type": "pin", "identifier": _b64(pin)}))
    tok = (r3.get("data") or {}).get("access_token")
    if not tok: raise LoginError("3 verify_pin (PIN)", r3)
    app_base, app_type = (app_id.rsplit("-", 1) + ["100"])[:2]
    payload = {"fyers_id": fy_id, "app_id": app_base, "redirect_uri": REDIRECT, "appType": app_type,
               "code_challenge": "", "state": "sample", "scope": "", "nonce": "", "response_type": "code", "create_cookie": True}
    resp4 = s.post("https://api-t1.fyers.in/api/v3/token", json=payload, headers={"Authorization": f"Bearer {tok}"})
    r4 = _jget(resp4)
    url = r4.get("Url") or r4.get("url") or (r4.get("data") or {}).get("url") if isinstance(r4, dict) else None
    auth_code = None
    if url: auth_code = (parse_qs(urlparse(url).query).get("auth_code") or [None])[0]
    if not auth_code and isinstance(r4.get("data"), dict): auth_code = r4["data"].get("auth")
    if not auth_code:
        sess = fyersModel.SessionModel(client_id=app_id, secret_key=secret, redirect_uri=REDIRECT, response_type="code", grant_type="authorization_code", state="sample")
        e = LoginError("4 token/auth_code (App ID / redirect URI / app consent)", {"http": resp4.status_code, **r4} if isinstance(r4, dict) else r4)
        e.consent_url = sess.generate_authcode()
        raise e
    sess = fyersModel.SessionModel(client_id=app_id, secret_key=secret, redirect_uri=REDIRECT, response_type="code", grant_type="authorization_code")
    sess.set_token(auth_code)
    r5 = sess.generate_token()
    if not isinstance(r5, dict) or not r5.get("access_token"): raise LoginError("5 generate_token (App ID / Secret ID)", r5)
    return r5["access_token"]

def get_token(force=False):
    today = dt.datetime.now(IST).date().isoformat()
    if not force and os.path.exists(TOKEN_FILE):
        try:
            d = json.load(open(TOKEN_FILE))
            if d.get("date") == today and d.get("access_token"): return d["access_token"]
        except Exception: pass
    tok = login()
    fd = os.open(TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f: json.dump({"date": today, "access_token": tok}, f)
    os.chmod(TOKEN_FILE, 0o600)
    return tok

def get_fyers(force=False):
    return fyersModel.FyersModel(client_id=_env("FYERS_APP_ID"), token=get_token(force), is_async=False, log_path=LOG_DIR)

if __name__ == "__main__":
    try:
        f = get_fyers(force="--force" in sys.argv)
        p = f.get_profile()
        print("profile:", p.get("s"), (p.get("data") or {}).get("name"))
    except LoginError as e:
        print("LOGIN FAILED at step", e)
        if getattr(e, "consent_url", None): print("consent URL (contains app id only):", e.consent_url)
        sys.exit(1)
