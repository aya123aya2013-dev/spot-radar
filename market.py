#!/usr/bin/env python3
"""Shared market helpers for the agent: Binance public candles and prices, a synthetic-data mode for code tests,
and the few indicators the agent uses.  No API key, no orders.

  SPOT_LAB_DATA   folder for state files (default ./lab_data)
  --synthetic     fake deterministic data (proves the code runs, never performance)
"""
import json, os, random, sys, time, urllib.request

CAPITAL = 5000.0
TFS = {"1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800}
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("SPOT_LAB_DATA") or os.path.join(HERE, "lab_data")
FEE, SLIP = 0.001, 0.0005          # per side: 0.1% exchange fee, 0.05% slippage
os.makedirs(OUT, exist_ok=True)
SYN = "--synthetic" in sys.argv
HOSTS = ["https://api.binance.com", "https://data-api.binance.vision"]


# ---------------------------------------------------------------- data
def http_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "spot-radar-agent"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def api(path):
    err = None
    for h in HOSTS:
        try:
            return http_json(h + path)
        except Exception as e:
            err = e
    raise RuntimeError(f"all data hosts failed: {err}")


def synthetic(sym, tf, n):
    rnd = random.Random(sum(map(ord, sym + tf)))
    step = TFS[tf] * 1000
    t = int(time.time() * 1000) // step * step - n * step
    p, drift, rows = 100.0, 0.0, []
    for i in range(n):
        if i % 250 == 0:
            drift = rnd.choice([-0.0015, 0.0, 0.0012, 0.0025])
        o = p
        p *= 1 + drift + rnd.gauss(0, 0.012)
        hi, lo = max(o, p) * (1 + abs(rnd.gauss(0, 0.004))), min(o, p) * (1 - abs(rnd.gauss(0, 0.004)))
        rows.append([t + i * step, o, hi, lo, p, 1000 + rnd.random() * 500])
    return rows


def klines(sym, tf, total=3000, closed_only=True):
    """[time_ms, open, high, low, close, volume]; the still-forming candle is dropped unless closed_only=False."""
    if SYN:
        rows = synthetic(sym, tf, total)
        return rows[:-1] if closed_only else rows
    rows, end = [], None
    while len(rows) < total:
        q = f"/api/v3/klines?symbol={sym}USDT&interval={tf}&limit=1000" + (f"&endTime={end}" if end else "")
        b = api(q)
        if not b:
            break
        rows = [[int(k[0]), *map(float, k[1:6])] for k in b] + rows
        end = int(b[0][0]) - 1
        if len(b) < 1000:
            break
    rows = rows[-total:]
    if closed_only and rows and rows[-1][0] + TFS[tf] * 1000 > time.time() * 1000:
        rows = rows[:-1]
    return rows


def last_price(sym, rows):
    if SYN:
        return rows[-1][4] * (1 + random.gauss(0, 0.002))
    return float(api(f"/api/v3/ticker/price?symbol={sym}USDT")["price"])


# ---------------------------------------------------------------- indicators
def ema(a, n):
    k, out, e = 2 / (n + 1), [], None
    for x in a:
        e = x if e is None else x * k + e * (1 - k)
        out.append(e)
    return out


def atr(rows, n=14):
    out, a = [], 0.0
    for i, r in enumerate(rows):
        tr = r[2] - r[3] if i == 0 else max(r[2] - r[3], abs(r[2] - rows[i - 1][4]), abs(r[3] - rows[i - 1][4]))
        a = (a * i + tr) / (i + 1) if i < n else (a * (n - 1) + tr) / n
        out.append(a)
    return out


def agent_coins(limit=10):
    """The coins worth measuring now: BTC and ETH, plus what the agent holds and watches."""
    out = ["BTC", "ETH"]
    try:
        st = json.load(open(os.path.join(OUT, "paper_G.json")))
        out += list((st.get("holdings") or st.get("pos") or {}).keys()) + list(st.get("watch") or [])
    except Exception:
        pass
    return list(dict.fromkeys(out))[:limit]
